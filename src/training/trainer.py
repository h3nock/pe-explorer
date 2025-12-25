import os
import time
import math
import random
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


class Trainer: 
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, rank: int, local_rank: int, world_size: int, grad_accum_steps: int = 1, grad_clip: float = 1.0, warmup_steps: int = 2000, max_steps: int = 10000, checkpoint_interval: int = 5000, checkpoint_dir: str = "checkpoints", wsd_stage: str = "full", decay_steps: int = 0, target_checkpoints: list[int] = None, wandb_manager=None, run_ids: dict = None):
        self.rank = rank 
        self.local_rank = local_rank 
        self.world_size = world_size 
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps 
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps 
        self.base_lr = optimizer.defaults['lr'] 
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        # WSD scheduler parameters
        self.wsd_stage = wsd_stage
        self.decay_steps = decay_steps
        self.decay_start_step = None # set in train()

        if torch.cuda.is_available():
            # move model to GPU 
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
        self.step = 0        # optimizer update count
        self.micro_step = 0  # iteration count (for grad accumulation)
        self.consumed_tokens = 0 # total tokens processed
    
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
    def save_checkpoint(self, path: str, dataloader=None):
        """Save training checkpoint with optional dataloader state for exact resumption."""
        if self.rank != 0:
            return  # only rank 0 saves
        
        checkpoint = {
            'step': self.step,
            'micro_step': self.micro_step,
            'consumed_tokens': self.consumed_tokens,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # RNG states for reproducibility
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # save dataloader state if provided (StatefulDataLoader)
        if dataloader is not None and hasattr(dataloader, 'state_dict'):
            checkpoint['dataloader_state'] = dataloader.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, dataloader=None):
        """Load training checkpoint and restore dataloader state if available."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.micro_step = checkpoint.get('micro_step', self.step * self.grad_accum_steps)
        self.consumed_tokens = checkpoint.get('consumed_tokens', 0)
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # restore RNG states for reproducibility
        if 'rng_state' in checkpoint:
            rng = checkpoint['rng_state']
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            if rng['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng['cuda'])
        
        # restore dataloader state if provided
        if dataloader is not None and 'dataloader_state' in checkpoint:
            if hasattr(dataloader, 'load_state_dict'):
                dataloader.load_state_dict(checkpoint['dataloader_state'])
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
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
        
        return loss.item() * self.grad_accum_steps, grad_norm, weight_norm
    
    def train(self, dataloader, max_steps: int, log_interval: int = 100):
        """Train the model.
        
        Args:
            max_steps: Number of optimizer updates (not iterations)
            log_interval: Log every N optimizer steps
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # accumulators (reset every log_interval steps)
        accum_loss = 0.0
        steps_since_log = 0
        tokens_since_log = 0
        start_time = time.time()
        
        # per-step tracking (reset every optimizer step)
        step_loss = 0.0
        if self.wsd_stage == "decay_only" and self.decay_start_step is None:
            self.decay_start_step = self.step
            if self.rank == 0:
                print(f"Starting WSD decay phase from step {self.step} for {self.decay_steps} steps")

        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"Training started!")
            print(f"  Schedule: WSD ({self.wsd_stage})")
            
            tokens_per_step = self.grad_accum_steps * self.world_size * dataloader.batch_size * dataloader.dataset.seq_len
            print(f"  Device: {self.device}")
            print(f"  Tokens per step: {tokens_per_step:,}")
            print(f"  Max steps: {max_steps:,} ({max_steps * tokens_per_step / 1e9:.2f}B tokens)")
            print(f"  Grad accum steps: {self.grad_accum_steps}")
            print(f"  Checkpoint interval: {self.checkpoint_interval}")
            print(f"  Checkpoint dir: {self.checkpoint_dir}")
            print(f"{'='*60}\n")

        for x, y in dataloader:
            loss, grad_norm, weight_norm = self.train_step(x, y)
            step_loss += loss  # accumulate across grad_accum iterations
            current_tokens = x.numel() * self.world_size
            tokens_since_log += current_tokens
            self.consumed_tokens += current_tokens
            
            # check if optimizer step happened
            if grad_norm > 0:
                # this was an optimizer step
                # step_loss contains SUM of losses from micro-batches
                # we want AVERAGE loss for the effective batch
                accum_loss += (step_loss / self.grad_accum_steps)
                steps_since_log += 1
                last_grad_norm = grad_norm
                last_weight_norm = weight_norm
                step_loss = 0.0  # reset for next step
                
                # log every log_interval optimizer steps
                if self.step % log_interval == 0 and self.step > 0:
                    elapsed = time.time() - start_time
                    avg_loss = accum_loss / steps_since_log
                    tokens_per_sec = tokens_since_log / elapsed
                    
                    if self.rank == 0:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/step": self.step,
                            "train/tokens": self.consumed_tokens, 
                            "train/lr": self.get_lr(self.step),
                            "train/grad_norm": last_grad_norm,
                            "param/weight_norm": last_weight_norm,
                        })
                    
                    # reset accumulators
                    accum_loss = 0.0
                    steps_since_log = 0
                    tokens_since_log = 0
                    start_time = time.time()
                
                # checkpoint
                if self.step % self.checkpoint_interval == 0 and self.step > 0:
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    self.save_checkpoint(f"{self.checkpoint_dir}/step_{self.step}.pt", dataloader)
                
                if self.step >= max_steps:
                    break
        
        # save final checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_checkpoint(f"{self.checkpoint_dir}/final.pt", dataloader)