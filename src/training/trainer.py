import os
import time
import random
import pickle
import torch
import wandb
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast 
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
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
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, rank: int, local_rank: int, world_size: int, batch_size: int, max_seq_len: int, grad_accum_steps: int = 1, grad_clip: float = 1.0, warmup_steps: int = 2000, checkpoint_interval: int = 5000, checkpoint_dir: str = "checkpoints"):
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

        # training state (will be restored from checkpoint or defaults)
        self.step = 0
        self.micro_step = 0
        self.step = 0
        self.micro_step = 0
        self.consumed_tokens = 0
        self.run_id = None  # single run ID for wandb
        self.samples_seen = 0 # Global samples seen (for deterministic resume)
        self.target_budget = None  # set from branch checkpoint for decay
        self.branch_tokens = None  # branch point for decay
        self.next_target_idx = 0

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
    
    def init_wandb(self, base_config: dict, group: str, target_budget: int):
        """Initialize wandb run. target_budget is the budget this run is named for."""
        if self.rank != 0:
            return
        
        budget_str = format_budget(target_budget)
        run_name = f"{base_config.get('model_size', 'unknown')}_{base_config.get('pe_type', 'unknown')}_{budget_str}_s{base_config.get('seed', 42)}"
        
        if self.run_id:
            # resuming existing run - rewind to checkpoint step to match local history
            wandb.init(
                project="pos-enc-bench",
                resume_from=f"{self.run_id}?_step={self.consumed_tokens}"
            )
        else:
            # new run
            run = wandb.init(
                project="pos-enc-bench",
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

    def finish_wandb(self):
        """Finish wandb run."""
        if self.rank == 0:
            wandb.finish()
    
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
        """Save training checkpoint with all state for exact resumption."""
        if self.rank != 0:
            return  # only rank 0 saves
        
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
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        


        if extra:
            checkpoint.update(extra)
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, dataloader=None):
        """Load training checkpoint and restore all state for exact resumption."""
        self.resume_checkpoint_path = path  # remember where we loaded from
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Restore model and optimizer
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
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        


        return checkpoint
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float, bool]:
        """Single training step with gradient accumulation.
        
        note: self.step counts optimizer updates (not iterations).
        """
        x, y = x.to(self.device), y.to(self.device)

        if self.device.type == "cuda":
            with autocast("cuda", dtype=self.dtype):
                logits = self.model(x) 
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))  
        else:
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))  

        loss = loss / self.grad_accum_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward() 

        grad_norm = 0.0
        weight_norm = 0.0
        did_step = False

        self.micro_step += 1
        
        # only update weights every grad_accum_steps
        if self.micro_step % self.grad_accum_steps == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.get_lr(self.step)

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

            self.optimizer.zero_grad()
            self.step += 1  # step = optimizer update count
            did_step = True
        
        return loss.item() * self.grad_accum_steps, grad_norm, weight_norm, did_step
    
    def train(self, dataloader, log_interval: int = 100, max_tokens: int | None = None):
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

        for x, y in dataloader:
            loss, grad_norm, weight_norm, did_step = self.train_step(x, y)
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
                
                # log every log_interval optimizer steps
                if self.step % log_interval == 0 and self.step > 0:
                    elapsed = time.time() - start_time
                    avg_loss = accum_loss / steps_since_log
                    tokens_per_sec = tokens_since_log / elapsed
                    
                    if self.rank == 0:
                        metrics = {
                            "train/loss": avg_loss,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/step": self.step,
                            "train/tokens": self.consumed_tokens, 
                            "train/lr": self.get_lr(self.step),
                            "train/grad_norm": grad_norm,
                            "param/weight_norm": weight_norm,
                        }
                        self.log_wandb(metrics, self.consumed_tokens)
                    
                    # reset accumulators
                    accum_loss = 0.0
                    steps_since_log = 0
                    tokens_since_log = 0
                    start_time = time.time()
                
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
        
        # save final checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_checkpoint(f"{self.checkpoint_dir}/final.pt", dataloader)
        self.flush_wandb_history()
