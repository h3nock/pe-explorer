import os
import time
import math
import torch
import wandb
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
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, rank: int, local_rank: int, world_size: int, grad_accum_steps: int = 1, grad_clip: float = 1.0, warmup_steps: int = 100, max_steps: int = 10000, checkpoint_interval: int = 5000, checkpoint_dir: str = "checkpoints"):
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
        self.step = 0
    
    def get_lr(self, step: int) -> float:
        """cosine LR with linear warmup"""
        if step < self.warmup_steps:
            return self.base_lr * (step / self.warmup_steps) 
        else:
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps) 
            return self.base_lr * 0.5 * ( 1 + math.cos(math.pi * progress))
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        if self.rank != 0:
            return  # only rank 0 saves
        
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """single training step with gradient accumulation."""
        x,y = x.to(self.device), y.to(self.device)

        if self.device.type == "cuda":
            with autocast("cuda", dtype=self.dtype):
                logits = self.model(x) 
                # compute loss 
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))  
        else:
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))  

        loss = loss / self.grad_accum_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward() 

        if (self.step + 1) % self.grad_accum_steps == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.get_lr(self.step)

            if self.scaler:
                self.scaler.unscale_(self.optimizer) 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step() 
            self.optimizer.zero_grad() 
        
        self.step += 1 
        return loss.item() * self.grad_accum_steps
    
    def train(self, dataloader, max_steps: int, log_interval: int = 100):
        self.model.train() # sets the mode to training 
        self.optimizer.zero_grad() 
        total_loss = 0
        start_time = time.time() 

        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"Training started!")
            print(f"  Device: {self.device}")
            print(f"  Max steps: {max_steps}")
            print(f"  Checkpoint interval: {self.checkpoint_interval}")
            print(f"  Checkpoint dir: {self.checkpoint_dir}")
            print(f"{'='*60}\n")

        for x,y in dataloader:
            loss = self.train_step(x, y)
            total_loss += loss 

            if self.step % log_interval == 0:
                avg_loss = total_loss / log_interval
                elapsed = time.time() - start_time 
                tokens_per_sec = (x.numel() * log_interval * self.world_size) / elapsed 

                if self.rank == 0:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/step": self.step,
                        "train/lr": self.get_lr(self.step), 
                    })
            
                total_loss = 0
                start_time = time.time()
            
            if self.step % self.checkpoint_interval == 0 and self.step > 0:
                os.makedirs(self.checkpoint_dir, exist_ok=True) 
                self.save_checkpoint(f"{self.checkpoint_dir}/step_{self.step}.pt")
            
            if self.step >= max_steps:
                break 
        
        # save final checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_checkpoint(f"{self.checkpoint_dir}/final.pt")