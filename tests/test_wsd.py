import torch
import pytest
from src.training.trainer import Trainer
from src.model.transformer import Transformer, ModelConfig

class MockOptimizer:
    def __init__(self):
        self.defaults = {'lr': 1.0}


def test_wsd_full_schedule():
    """Test WSD 'full' phase: Warmup -> Stable -> Decay."""
    # 100 steps total
    # 10 steps warmup (10%)
    # 10 steps decay (10%) -> Decay starts at step 90
    max_steps = 100
    decay_steps = 10
    
    trainer = Trainer(
        model=torch.nn.Linear(1, 1),
        optimizer=MockOptimizer(),
        rank=0, local_rank=0, world_size=1,
        batch_size=4, max_seq_len=32,
        warmup_steps=10,
    )
    trainer.configure_wsd("full", max_steps=max_steps, decay_steps=decay_steps)
    
    # warmup
    assert trainer.get_lr(5) == 0.5
    assert trainer.get_lr(10) == 1.0
    
    # stable (Step 50)
    assert trainer.get_lr(50) == 1.0
    
    # start of Decay (Step 90)
    # decay_start = 100 - 10 = 90
    assert trainer.get_lr(89) == 1.0 # still stable
    
    # decay phase
    # Step 95: 5 steps into 10 step decay -> 50% LR
    assert trainer.get_lr(95) == 0.5 
    
    # End
    assert trainer.get_lr(100) == 0.0
    print("WSD Full schedule passed")

def test_wsd_decay_only_schedule():
    """Test WSD 'decay_only' phase: Linear decay from start."""
    trainer = Trainer(
        model=torch.nn.Linear(1, 1),
        optimizer=MockOptimizer(),
        rank=0, local_rank=0, world_size=1,
        batch_size=4, max_seq_len=32,
    )
    
    # Simulate resuming at step 100
    trainer.step = 100
    trainer.configure_wsd("decay_only", max_steps=110, decay_steps=10)
    
    # start of decay (decay_start_step set to 100 by configure_wsd)
    assert trainer.get_lr(100) == 1.0
    
    # midway
    assert trainer.get_lr(105) == 0.5
    
    # end
    assert trainer.get_lr(110) == 0.0
    print("WSD Decay Only schedule passed")

if __name__ == "__main__":
    test_wsd_full_schedule()
    test_wsd_decay_only_schedule()
