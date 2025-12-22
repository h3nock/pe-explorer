#!/usr/bin/env python3
"""Training script for positional encoding benchmark."""
import argparse
import yaml
import wandb
from torch.optim.adamw import AdamW

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import get_dataloader
from src.training.trainer import Trainer, setup_distributed, cleanup_distributed

from dotenv import load_dotenv 
load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, default="xs", choices=["xs", "s", "m", "l", "xl"])
    parser.add_argument("--pe-type", type=str, default="sinusoidal", choices=["none", "sinusoidal"])
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--tokens", type=int, default=None, help="Override total tokens from config")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()

    # load configs 
    with open("configs/model_sizes.yaml") as f:
        all_configs= yaml.safe_load(f)
    
    # current model size specific configs 
    config = all_configs[args.model_size]
    # shared training settings 
    training_config = all_configs["training"]
    
    batch_size = args.batch_size or config["batch_size"]
    total_tokens = args.tokens or config["optimal_tokens"]
    max_seq_len = config["max_seq_len"]
    tokens_per_step = batch_size * max_seq_len * world_size 
    max_steps = total_tokens // tokens_per_step 

    model_config = ModelConfig(
        d_model=config["d_model"], 
        n_layers=config["n_layers"], 
        n_heads=config["n_heads"], 
        d_ff=config["d_ff"], 
        max_seq_len=max_seq_len, 
        vocab_size=training_config["vocab_size"],
        pe_type=args.pe_type,
        dropout=training_config["dropout"] 
    )

    model = Transformer(model_config)
    optimizer = AdamW(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=training_config["weight_decay"], 
        betas=(training_config["beta1"], training_config["beta2"])
    )

    dataloader = get_dataloader(
        seq_len=max_seq_len,  
        batch_size=batch_size 
    )

    # unique run name from experiment params
    run_name = f"{args.model_size}_{args.pe_type}_b{batch_size}_t{total_tokens//1_000_000}M"

    if rank == 0: 
        wandb.init(project="pos-enc-bench", 
        name=run_name, 
        config={**config, "model_size": args.model_size, "pe_type": args.pe_type, "batch_size": batch_size, "total_tokens": total_tokens}
        )
    
    warmup_steps = int(training_config["warmup_ratio"] * max_steps) 
    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        rank=rank, 
        local_rank=local_rank, 
        world_size=world_size, 
        grad_accum_steps=args.grad_accum_steps, 
        grad_clip=training_config["grad_clip"], 
        warmup_steps=warmup_steps, 
        max_steps=max_steps, 
        checkpoint_interval=all_configs["evaluation"]["checkpoint_interval"],
        checkpoint_dir=f"checkpoints/{run_name}",
    )

    trainer.train(dataloader, max_steps=max_steps)

    if rank == 0:
        wandb.finish()
    cleanup_distributed()

if __name__ == "__main__":
    main()
    