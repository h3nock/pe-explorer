"""Tests for training infrastructure."""
import torch
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import FineWebEduDataset, get_dataloader
from src.training.trainer import Trainer, setup_distributed, cleanup_distributed


def test_trainer_init():
    """Test Trainer initialization on CPU."""
    config = ModelConfig(
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
        vocab_size=1000,
        pe_type="none",
        dropout=0.0,
    )
    model = Transformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        rank=0,
        local_rank=0,
        world_size=1,
        grad_accum_steps=1,
        grad_clip=1.0,
        warmup_steps=10,
        max_steps=100,
    )
    
    assert trainer.device.type in ["cpu", "mps", "cuda"]
    assert trainer.step == 0
    print(f"Trainer initialized on {trainer.device}")


def test_trainer_step():
    """Test single training step."""
    config = ModelConfig(
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
        vocab_size=1000,
        pe_type="sinusoidal",
        dropout=0.0,
    )
    model = Transformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        rank=0,
        local_rank=0,
        world_size=1,
    )
    
    # create dummy batch
    x = torch.randint(0, 1000, (4, 32))  # batch=4, seq=32
    y = torch.randint(0, 1000, (4, 32))
    
    loss = trainer.train_step(x, y)
    
    assert isinstance(loss, float)
    assert loss > 0
    assert trainer.step == 1
    print(f"Train step loss: {loss:.4f}")


def test_lr_schedule():
    """Test LR warmup and cosine decay."""
    config = ModelConfig(d_model=64, n_heads=2, n_layers=2, d_ff=128, max_seq_len=32)
    model = Transformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        rank=0, local_rank=0, world_size=1,
        warmup_steps=10,
        max_steps=100,
    )
    
    # warmup: LR should increase
    lr_0 = trainer.get_lr(0)
    lr_5 = trainer.get_lr(5)
    lr_10 = trainer.get_lr(10)
    
    assert lr_0 == 0.0  # Step 0 = 0 LR
    assert lr_5 > lr_0  # Increasing during warmup
    assert lr_10 == 1e-3  # Full LR at end of warmup
    
    # cosine decay: LR should decrease
    lr_50 = trainer.get_lr(50)
    lr_100 = trainer.get_lr(100)
    
    assert lr_50 < lr_10  # Decaying
    assert lr_100 < lr_50  # Still decaying
    print(f"LR schedule: 0={lr_0:.6f}, 5={lr_5:.6f}, 10={lr_10:.6f}, 50={lr_50:.6f}, 100={lr_100:.6f}")


if __name__ == "__main__":
    test_trainer_init()
    test_trainer_step()
    test_lr_schedule()
    print("\nAll training tests passed!")
