#!/usr/bin/env python3
"""Scale hyperparameters from tiny to larger models using scaling laws."""
import math

MODELS = {
    'tiny': 59e6,
    'small': 164e6,
    'medium': 470e6,
    'large': 1.2e9,
}

# best config from tiny sweep (update after sweep completes)
BASE_CONFIG = {
    'lr': 0.0006,
    'warmup_steps': 1000,
    'grad_clip': 1.0,
    'batch_size': 32,
}

def scale_hyperparams():
    N_base = MODELS['tiny']
    
    for name, N in MODELS.items():
        scale = math.sqrt(N / N_base)
        
        print(f"\n{name.upper()} ({N/1e6:.0f}M):")
        print(f"  lr: {BASE_CONFIG['lr'] / scale:.6f}")
        print(f"  warmup_steps: {int(BASE_CONFIG['warmup_steps'] * scale)}")
        print(f"  grad_clip: {BASE_CONFIG['grad_clip']}")
        print(f"  batch_size: {BASE_CONFIG['batch_size']}")

if __name__ == '__main__':
    scale_hyperparams()