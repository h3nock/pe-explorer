import os
import time
import random
import pickle
from dataclasses import asdict, is_dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import wandb
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size 

    return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def format_budget(budget: int) -> str:
    """Format budget as human-readable string (e.g., 20000000 → '20M')."""
    if budget < 1e9:
        return f"{budget // 1_000_000}M"
    return f"{budget / 1e9:.1f}B"


class Trainer: 
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        rank: int,
        local_rank: int,
        world_size: int,
        batch_size: int,
        max_seq_len: int,
        grad_accum_steps: int = 1,
        grad_clip: float = 1.0,
        warmup_steps: int = 2000,
        checkpoint_interval: int = 5000,
        checkpoint_dir: str = "checkpoints",
        run_metadata: dict | None = None,
    ):
        """Initialize trainer with core settings. Call configure_wsd() after loading checkpoint."""
        self.rank = rank 
        self.local_rank = local_rank 
        self.world_size = world_size 
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps 
        self.warmup_steps = warmup_steps
        self.base_lr = optimizer.defaults['lr'] 
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.tokens_per_step = batch_size * max_seq_len * grad_accum_steps * world_size
        self.run_metadata = run_metadata

        # training state (will be restored from checkpoint or defaults)
        self.step = 0
        self.micro_step = 0
        self.consumed_tokens = 0
        self.run_id = None  # single run ID for wandb
        self.samples_seen = 0 # Global samples seen (for deterministic resume)
        self.target_budget = None  # set from branch checkpoint for decay
        self.branch_tokens = None  # branch point for decay
        self.next_target_idx = 0
        self.best_val_loss = float('inf')  # track best validation loss

        # WSD state (set by configure_wsd after checkpoint load)
        self.wsd_stage = None
        self.max_steps = 0
        self.decay_steps = 0
        self.decay_start_step = None
        self.target_checkpoints = []
        self.resume_checkpoint_path = None  # path to checkpoint we resumed from
        self.wandb_buffer = []  # metrics buffer, flushed to disk at checkpoints

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
            if torch.cuda.is_bf16_supported(): 
                self.dtype = torch.bfloat16
                self.scaler = None 
            else:
                self.dtype = torch.float16
                self.scaler = GradScaler() 
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.dtype = torch.float32 
            self.scaler = None 
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32 
            self.scaler = None 

        model = model.to(self.device)

        if world_size > 1:
            self.model = DDP(model, device_ids=[local_rank])
        else:
            self.model = model 
        
        self.optimizer = optimizer
    
    def configure_wsd(self, stage: str, max_steps: int, decay_steps: int, target_checkpoints: list = None):
        """Configure WSD schedule. should be called after loading checkpoint if resuming."""
        self.wsd_stage = stage
        self.max_steps = max_steps
        self.decay_steps = decay_steps
        
        new_targets = sorted(target_checkpoints or [], key=lambda t: t[0])
        
        # Validate target_checkpoints hasn't changed for resumed full stage
        if stage == "full" and self.step > 0 and self.target_checkpoints:
            if new_targets != self.target_checkpoints:
                raise ValueError(
                    "target_checkpoints changed during resumed run! "
                    f"Checkpoint: {self.target_checkpoints}, New: {new_targets}"
                )
        
        self.target_checkpoints = new_targets
        
        # for decay_only, decay starts from current step
        if stage == "decay_only" and self.decay_start_step is None:
            self.decay_start_step = self.step
    
    def init_wandb(self, base_config: dict, group: str, target_budget: int, run_name: str | None = None):
        """Initialize wandb run with run ID persistence and custom metric axes.

        Args:
            base_config: Config dict for wandb
            group: Wandb group name (e.g., "tiny_rope")
            target_budget: Token budget this run targets
            run_name: Optional explicit run name (uses auto-generated if None)
        """
        if self.rank != 0:
            return

        # use provided run_name or generate one
        if run_name is None:
            budget_str = format_budget(target_budget)
            run_name = f"{base_config.get('model_size', 'unknown')}_{base_config.get('pe_type', 'unknown')}_{budget_str}_s{base_config.get('seed', 42)}"

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # W&B initialization: only resume if run_id came from checkpoint (explicit --resume)
        if self.run_id:
            print(f"W&B: Resuming run {self.run_id} from checkpoint")
            wandb.init(
                project="pe-explorer",
                resume_from=f"{self.run_id}?_step={self.consumed_tokens}"
            )
        else:
            print(f"W&B: Creating new run '{run_name}'")
            run = wandb.init(
                project="pe-explorer",
                group=group,
                name=run_name,
                config={**base_config, "target_budget": target_budget},
            )
            self.run_id = run.id

            # fresh decay: replay shared history up to branch point
            if self.wsd_stage == "decay_only" and self.branch_tokens and self.resume_checkpoint_path:
                history_dir = os.path.dirname(self.resume_checkpoint_path)
                history_path = os.path.join(history_dir, "wandb_history.pkl")
                if os.path.exists(history_path):
                    with open(history_path, "rb") as f:
                        while True:
                            try:
                                tokens, metrics = pickle.load(f)
                                if tokens <= self.branch_tokens:
                                    wandb.log({**metrics, "tokens": tokens}, step=tokens)
                            except EOFError:
                                break

        # define custom step axis: tokens as primary x-axis
        wandb.define_metric("tokens")
        wandb.define_metric("train/*", step_metric="tokens")
        wandb.define_metric("val/*", step_metric="tokens")
        wandb.define_metric("time/*", step_metric="tokens")
        wandb.define_metric("grad/*", step_metric="tokens")
        wandb.define_metric("wsd/*", step_metric="tokens")
        wandb.define_metric("gpu/*", step_metric="tokens")
        wandb.define_metric("param/*", step_metric="tokens")

        # log comprehensive config (merge base_config with additional metadata)
        full_config = {
            **base_config,
            "target_budget": target_budget,
            "run_name": run_name,
            "group": group,
            "world_size": self.world_size,
            "effective_batch_size": self.batch_size * self.world_size * self.grad_accum_steps,
            "tokens_per_step": self.tokens_per_step,
            "grad_accum_steps": self.grad_accum_steps,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
        }

        # add model-specific config (tie_embedding, dropout, etc.)
        model_obj = self.model.module if hasattr(self.model, 'module') else self.model
        model_config = getattr(model_obj, 'config', None)
        if model_config is not None:
            full_config["tie_embedding"] = getattr(model_config, 'tie_embedding', True)
            full_config["pe_params"] = getattr(model_config, 'pe_params', {})
            full_config["dropout"] = getattr(model_config, 'dropout', 0.0)
        # count parameters
        n_params = sum(p.numel() for p in model_obj.parameters())
        n_trainable = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
        full_config["n_params"] = n_params
        full_config["n_trainable_params"] = n_trainable

        if self.run_metadata:
            full_config["data_config"] = self.run_metadata.get("data_config", {})
            full_config["training_config"] = self.run_metadata.get("training_config", {})
            full_config["validation_config"] = self.run_metadata.get("validation_config", {})
            full_config["environment"] = self.run_metadata.get("environment", {})
            full_config["cli_args"] = self.run_metadata.get("cli_args", {})
        wandb.config.update(full_config, allow_val_change=True)

    def finish_wandb(self):
        """Finish wandb run."""
        if self.rank == 0:
            wandb.finish()

    @torch.no_grad()
    def compute_logit_stats(self, logits: torch.Tensor) -> dict:
        """Compute logit statistics for stability monitoring."""
        flat_logits = logits[:, ::8, :].reshape(-1, logits.size(-1))  # sample every 8th token

        probs = F.softmax(flat_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()

        return {
            "train/logits_mean": flat_logits.mean().item(),
            "train/logits_std": flat_logits.std().item(),
            "train/logits_min": flat_logits.min().item(),
            "train/logits_max": flat_logits.max().item(),
            "train/entropy": entropy.item(),
        }

    def compute_gradient_stats(self) -> dict:
        """Compute per-layer gradient norms and aggregate statistics."""
        model_obj = self.model.module if hasattr(self.model, 'module') else self.model
        stats = {}

        # per-layer gradient norms
        for i, block in enumerate(model_obj.blocks):
            block_grad_norm = sum(
                p.grad.data.norm(2).item() ** 2 for p in block.parameters() if p.grad is not None
            ) ** 0.5
            if block_grad_norm > 0:
                stats[f"grad/layer_{i}_norm"] = block_grad_norm

        # aggregate stats
        all_grads = [p.grad.data.view(-1) for p in model_obj.parameters() if p.grad is not None]
        if all_grads:
            flat_grads = torch.cat(all_grads)
            flat_params = torch.cat([p.data.view(-1) for p in model_obj.parameters() if p.grad is not None])

            stats["grad/mean"] = flat_grads.mean().item()
            stats["grad/std"] = flat_grads.std().item()
            stats["grad/max"] = flat_grads.abs().max().item()

            param_norm = flat_params.norm(2).item()
            if param_norm > 0:
                stats["grad/update_norm_ratio"] = flat_grads.norm(2).item() / param_norm

        return stats

    def estimate_mfu(self, tokens_per_sec: float) -> float:
        """Estimate Model FLOPs Utilization as percentage of peak GPU performance."""
        model_obj = self.model.module if hasattr(self.model, 'module') else self.model
        config = getattr(model_obj, 'config', None)
        if config is None:
            return 0.0

        N, d, V, d_ff = config.n_layers, config.d_model, config.vocab_size, config.d_ff

        # FLOPs per token (x3 for forward+backward)
        flops_per_token = (12 * N * d * d + 8 * N * d * d_ff + 2 * d * V) * 3
        achieved_flops = tokens_per_sec * flops_per_token

        # peak FLOPs by GPU type
        peak_flops = 100e12  # default
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(self.device).lower()
            peak_map = {
                'a100': 312e12, 'h100': 989e12, 'v100': 125e12,
                '4090': 165e12, '3090': 71e12, 'l40': 181e12
            }
            for name, flops in peak_map.items():
                if name in gpu:
                    peak_flops = flops
                    break

        return (achieved_flops / peak_flops) * 100

    def log_wandb(self, metrics: dict, consumed_tokens: int):
        """Log to wandb. For full stage, also buffers for later flush."""
        if self.rank != 0:
            return
        wandb.log({**metrics, "tokens": consumed_tokens}, step=consumed_tokens)
        # buffer for history (full stage only, flushed at checkpoints)
        if self.wsd_stage == "full":
            self.wandb_buffer.append((consumed_tokens, metrics))
    
    def flush_wandb_history(self):
        """Append buffered history to disk and clear buffer."""
        if self.rank != 0 or not self.wandb_buffer:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        history_path = os.path.join(self.checkpoint_dir, "wandb_history.pkl")
        with open(history_path, "ab") as f:
            for rec in self.wandb_buffer:
                pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        self.wandb_buffer.clear()

    def get_lr(self, step: int) -> float:
        """Calculate learning rate based on WSD schedule."""
        if self.wsd_stage == "full":
            # Warmup -> Constant -> Decay (all in one go)
            decay_start = self.max_steps - self.decay_steps
            
            if step < self.warmup_steps:
                return self.base_lr * (step / self.warmup_steps)
            elif step < decay_start:
                return self.base_lr
            else:
                # decay phase
                progress = (step - decay_start) / max(1, self.decay_steps)
                progress = min(1.0, max(0.0, progress))
                return self.base_lr * (1.0 - progress)

        elif self.wsd_stage == "decay_only":
            # linear decay from base_lr to 0 over decay_steps
            # we start decaying from self.decay_start_step
            if self.decay_start_step is None:
                return self.base_lr # fallback if not set yet
            
            info_step = step - self.decay_start_step
            progress = info_step / max(1, self.decay_steps)
            progress = min(1.0, max(0.0, progress))
            return self.base_lr * (1.0 - progress)
        else:
            return self.base_lr # fallback
    def save_checkpoint(self, path: str, dataloader=None, extra: dict | None = None):
        """Save training checkpoint."""
        if self.rank != 0:
            return

        checkpoint = {
            # Core training state
            'step': self.step,
            'micro_step': self.micro_step,
            'consumed_tokens': self.consumed_tokens,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            
            # training geometry (for validation on resume)
            'batch_size': self.batch_size,
            'max_seq_len': self.max_seq_len,
            'grad_accum_steps': self.grad_accum_steps,
            'world_size': self.world_size,
            
            # WSD config (for resume)
            'max_steps': self.max_steps,
            'wsd_stage': self.wsd_stage,
            'decay_steps': self.decay_steps,
            'decay_start_step': self.decay_start_step,
            'target_budget': self.target_budget,
            'target_checkpoints': self.target_checkpoints,
            'branch_tokens': self.branch_tokens,
            
            # progress tracking
            'next_target_idx': self.next_target_idx,
            'wandb_run_id': self.run_id,

            # resume state (deterministic offset)
            'samples_seen': self.samples_seen,
            'best_val_loss': self.best_val_loss,
        }
        
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        model_config = getattr(model_obj, "config", None)
        if model_config is not None:
            checkpoint["config"] = asdict(model_config) if is_dataclass(model_config) else model_config
        if self.run_metadata:
            checkpoint["run_metadata"] = self.run_metadata

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # save RNG states for exact reproducibility on resume
        rng_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states['cuda'] = torch.cuda.get_rng_state_all()
        checkpoint['rng_states'] = rng_states

        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, dataloader=None):
        """Load training checkpoint and restore state."""
        self.resume_checkpoint_path = path
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        model_config = getattr(model_obj, "config", None)
        ckpt_config = checkpoint.get("config")

        # config validation
        if model_config is not None and ckpt_config is not None:
            current_config = asdict(model_config) if is_dataclass(model_config) else model_config
            if current_config != ckpt_config:
                raise ValueError(
                    "Checkpoint config does not match current model config. "
                    "Recreate the model from the checkpoint config to resume safely."
                )

        # restore model and optimizer
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # restore training state
        self.step = checkpoint['step']
        self.micro_step = checkpoint.get('micro_step', self.step * self.grad_accum_steps)
        self.consumed_tokens = checkpoint.get('consumed_tokens', 0)
        
        # validate training geometry hasn't changed (would break token math)
        if 'batch_size' in checkpoint:
            ckpt_tokens_per_step = (checkpoint['batch_size'] * checkpoint['max_seq_len'] * 
                                    checkpoint['grad_accum_steps'] * checkpoint['world_size'])
            if ckpt_tokens_per_step != self.tokens_per_step:
                raise ValueError(
                    f"Training geometry changed! Checkpoint: {ckpt_tokens_per_step} tokens/step, "
                    f"Current: {self.tokens_per_step} tokens/step. "
                    "This would break LR schedule and token accounting."
                )
        
        # restore WSD state (this might be overwritten by configure_wsd for decay_only phase)
        self.max_steps = checkpoint.get('max_steps', 0)
        self.decay_steps = checkpoint.get('decay_steps', 0)
        self.decay_start_step = checkpoint.get('decay_start_step')
        self.target_budget = checkpoint.get('target_budget')
        self.target_checkpoints = checkpoint.get('target_checkpoints', [])
        self.branch_tokens = checkpoint.get('branch_tokens')
        
        # restore progress tracking
        self.next_target_idx = checkpoint.get('next_target_idx', 0)
        self.run_id = checkpoint.get('wandb_run_id')
        
        # for fresh decay, CLEAR run_id so we create a NEW wandb run (not resume full run)
        if checkpoint.get('checkpoint_type') == 'pre_decay':
            self.run_id = None
            
        self.samples_seen = checkpoint.get('samples_seen', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # restore RNG states for exact reproducibility
        if 'rng_states' in checkpoint:
            rng_states = checkpoint['rng_states']
            random.setstate(rng_states['python'])
            np.random.set_state(rng_states['numpy'])
            torch.set_rng_state(rng_states['torch'])
            if 'cuda' in rng_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_states['cuda'])
            if self.rank == 0:
                print("  RNG states restored for deterministic resume")

        return checkpoint
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor, compute_grad_stats: bool = False) -> tuple[float, float, float, bool, float, dict]:
        """Single training step with gradient accumulation.

        note: self.step counts optimizer updates (not iterations).

        Args:
            x: Input tensor
            y: Target tensor
            compute_grad_stats: If True, compute per-layer gradient stats before zeroing gradients

        Returns:
            loss: The loss value for this micro-batch (unscaled by grad_accum)
            grad_norm: Gradient norm after clipping (0 if no optimizer step)
            weight_norm: L2 norm of all parameters (0 if no optimizer step)
            did_step: True if optimizer step was taken
            lr_used: Learning rate actually used for this step (0 if no optimizer step)
            grad_stats: Per-layer gradient statistics (empty dict if not computed)
        """
        x, y = x.to(self.device), y.to(self.device)
        logit_stats = {}

        if self.device.type == "cuda":
            with autocast("cuda", dtype=self.dtype):
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                # compute logit stats for stability monitoring (before backward)
                if compute_grad_stats:
                    logit_stats = self.compute_logit_stats(logits)
        else:
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            if compute_grad_stats:
                logit_stats = self.compute_logit_stats(logits)

        loss = loss / self.grad_accum_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_norm = 0.0
        weight_norm = 0.0
        did_step = False
        lr_used = 0.0
        grad_stats = logit_stats  # start with logit stats, will add gradient stats later

        self.micro_step += 1

        # only update weights every grad_accum_steps
        if self.micro_step % self.grad_accum_steps == 0:
            # capture LR BEFORE incrementing step (this is the LR actually used)
            lr_used = self.get_lr(self.step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_used

            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
                self.optimizer.step()

            with torch.no_grad():
                weight_norm = sum(p.norm(2).item() ** 2 for p in self.model.parameters()) ** 0.5

            # compute per-layer gradient stats before zeroing (this is expensive, so only when requested)
            if compute_grad_stats:
                grad_stats.update(self.compute_gradient_stats())

            self.optimizer.zero_grad()
            self.step += 1  # step = optimizer update count
            did_step = True

        return loss.item() * self.grad_accum_steps, grad_norm, weight_norm, did_step, lr_used, grad_stats
    
    @torch.no_grad()
    def validate(self, dataloader) -> float:
        """Run validation and return average loss."""
        if self.world_size > 1:
            dist.barrier()

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            with autocast("cuda", dtype=self.dtype, enabled=self.device.type == "cuda"):
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            batch_tokens = x.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

        # aggregate across ranks
        if self.world_size > 1:
            stats = torch.tensor([total_loss, total_tokens], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = stats[0].item(), int(stats[1].item())

        self.model.train()
        return total_loss / total_tokens if total_tokens > 0 else 0.0

    def train(self, dataloader, val_dataloader=None, eval_interval: int = 500, log_interval: int = 100, max_tokens: int | None = None):
        """Train the model using self.max_steps set by configure_wsd().

        Args:
            log_interval: Log every N optimizer steps
            max_tokens: Optional hard stop on consumed tokens
        """
        self.model.train()
        self.optimizer.zero_grad()

        # accumulators (reset every log_interval steps)
        accum_loss = 0.0
        steps_since_log = 0
        tokens_since_log = 0
        start_time = time.time()
        step_loss = 0.0
        last_lr_used = 0.0 

        # timing accumulators (for time/* metrics)
        dt_data_accum = 0.0
        dt_step_accum = 0.0
        timing_count = 0

        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"Training started!")
            print(f"  Schedule: WSD ({self.wsd_stage})")
            print(f"  Device: {self.device}")
            print(f"  Tokens per step: {self.tokens_per_step:,}")
            print(f"  Max steps: {self.max_steps:,} ({self.max_steps * self.tokens_per_step / 1e9:.2f}B tokens)")
            print(f"  Current step: {self.step}")
            print(f"  Checkpoint interval: {self.checkpoint_interval}")
            print(f"  Checkpoint dir: {self.checkpoint_dir}")
            print(f"{'='*60}\n")

        t_data_start = time.time()
        last_grad_stats = {}  # store gradient stats for logging
        for x, y in dataloader:
            t_data_end = time.time()
            dt_data_accum += (t_data_end - t_data_start) * 1000  # ms

            # compute gradient stats only on steps that will be logged to avoid overhead
            should_compute_grad_stats = ((self.step + 1) % log_interval == 0)
            t_step_start = time.time()
            loss, grad_norm, weight_norm, did_step, lr_used, grad_stats = self.train_step(
                x, y, compute_grad_stats=should_compute_grad_stats
            )
            t_step_end = time.time()
            dt_step_accum += (t_step_end - t_step_start) * 1000  # ms
            timing_count += 1

            if grad_stats:
                last_grad_stats = grad_stats

            step_loss += loss  # accumulate across grad_accum iterations
            current_tokens = x.numel() * self.world_size
            tokens_since_log += current_tokens
            self.consumed_tokens += current_tokens

            # count samples (global batch size)
            # x.shape[0] is local batch size
            self.samples_seen += x.shape[0] * self.world_size

            if did_step:
                # this was an optimizer step
                # step_loss contains SUM of losses from micro-batches
                # we want AVERAGE loss for the effective batch
                accum_loss += (step_loss / self.grad_accum_steps)
                steps_since_log += 1
                step_loss = 0.0  # reset for next step
                last_lr_used = lr_used  # save for logging

                # log every log_interval optimizer steps
                if self.step % log_interval == 0 and self.step > 0:
                    elapsed = time.time() - start_time
                    avg_loss = accum_loss / steps_since_log
                    tokens_per_sec = tokens_since_log / elapsed

                    # DDP: reduce loss across all ranks for accurate global average
                    if self.world_size > 1:
                        # pack accum_loss and steps_since_log for reduction
                        loss_stats = torch.tensor([accum_loss, float(steps_since_log)], device=self.device)
                        dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                        global_loss_sum, global_steps = loss_stats[0].item(), loss_stats[1].item()
                        avg_loss = global_loss_sum / global_steps if global_steps > 0 else 0.0

                    if self.rank == 0:
                        # calculate per-GPU MFU
                        tokens_per_sec_per_gpu = tokens_per_sec / self.world_size
                        mfu = self.estimate_mfu(tokens_per_sec_per_gpu)

                        # core training metrics
                        metrics = {
                            "train/loss": avg_loss,
                            "train/ppl": np.exp(avg_loss) if avg_loss < 20 else float('inf'),
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/step": self.step,
                            "train/tokens": self.consumed_tokens,
                            "train/lr": last_lr_used,
                            "train/grad_norm": grad_norm,
                            "train/grad_clipped": 1.0 if grad_norm > self.grad_clip else 0.0,
                            "train/samples_seen": self.samples_seen,
                            "train/tokens_per_step": self.tokens_per_step,
                            "train/progress_pct": (self.step / self.max_steps) * 100 if self.max_steps > 0 else 0,
                            "train/mfu": mfu,
                            "param/weight_norm": weight_norm,
                        }

                        # WSD specific metrics
                        metrics["wsd/stage"] = 1 if self.wsd_stage == "full" else 2  # 1=full, 2=decay_only
                        if self.target_budget:
                            metrics["wsd/target_budget"] = self.target_budget
                        if self.branch_tokens:
                            metrics["wsd/branch_tokens"] = self.branch_tokens
                        if self.decay_start_step is not None:
                            metrics["wsd/decay_start_step"] = self.decay_start_step
                            if self.decay_steps > 0:
                                decay_progress = (self.step - self.decay_start_step) / self.decay_steps
                                metrics["wsd/decay_progress_pct"] = min(100.0, max(0.0, decay_progress * 100))

                        # GPU memory metrics
                        if torch.cuda.is_available():
                            metrics["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated(self.device) / 1e9
                            metrics["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved(self.device) / 1e9
                            metrics["gpu/memory_allocated_max_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9

                        # AMP scaler metrics
                        if self.scaler is not None:
                            metrics["train/loss_scale"] = self.scaler.get_scale()

                        # timing breakdown
                        if timing_count > 0:
                            metrics["time/data_ms"] = dt_data_accum / timing_count
                            metrics["time/step_ms"] = dt_step_accum / timing_count

                        # per-layer gradient stats
                        if last_grad_stats:
                            metrics.update(last_grad_stats)

                        self.log_wandb(metrics, self.consumed_tokens)

                    # reset accumulators
                    accum_loss = 0.0
                    steps_since_log = 0
                    tokens_since_log = 0
                    dt_data_accum = 0.0
                    dt_step_accum = 0.0
                    timing_count = 0
                    start_time = time.time()
                
                # validation
                if val_dataloader and self.step % eval_interval == 0 and self.step > 0:
                    val_loss = self.validate(val_dataloader)
                    val_ppl = np.exp(val_loss)

                    # track best model
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        os.makedirs(self.checkpoint_dir, exist_ok=True)
                        self.save_checkpoint(f"{self.checkpoint_dir}/best.pt", dataloader)

                    if self.rank == 0:
                        best_marker = " (best)" if is_best else ""
                        print(f"VALIDATION (step {self.step}): Loss={val_loss:.4f}, PPL={val_ppl:.2f}{best_marker}")
                        metrics = {
                            "val/loss": val_loss,
                            "val/ppl": val_ppl,
                            "val/best_loss": self.best_val_loss,
                        }
                        self.log_wandb(metrics, self.consumed_tokens)

                # checkpoint
                if self.step % self.checkpoint_interval == 0 and self.step > 0:
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    self.save_checkpoint(f"{self.checkpoint_dir}/step_{self.step}.pt", dataloader)
                    self.flush_wandb_history()
                
                # force save for targeted budgets (WSD branching points)
                # use while to handle multiple targets being crossed in one step
                while self.next_target_idx < len(self.target_checkpoints):
                    branch_tokens, target_budget = self.target_checkpoints[self.next_target_idx]
                    if self.consumed_tokens >= branch_tokens:
                        os.makedirs(self.checkpoint_dir, exist_ok=True)
                        budget_str = format_budget(target_budget)
                        fname = f"ready_for_decay_{budget_str}.pt"
                        extra = {"checkpoint_type": "pre_decay", "target_budget": target_budget, "branch_tokens": branch_tokens}
                        self.save_checkpoint(f"{self.checkpoint_dir}/{fname}", dataloader, extra=extra)
                        self.next_target_idx += 1
                        if self.rank == 0:
                            print(f"CHECKPOINT: Reached {branch_tokens:,} tokens (budget={budget_str}) -> Saved {fname}")
                    else:
                        break  # no more targets reached

                if max_tokens is not None and self.consumed_tokens >= max_tokens:
                    break

                if self.step >= self.max_steps:
                    break

            # reset data timer for next iteration
            t_data_start = time.time()

        # save final checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_checkpoint(f"{self.checkpoint_dir}/final.pt", dataloader)
        self.flush_wandb_history()
