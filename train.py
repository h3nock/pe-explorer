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
    parser.add_argument("--pe-type", type=str, default="sinusoidal", choices=["none", "sinusoidal", "binary", "binary_norm", "decimal", "decimal_norm"])
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--tokens", type=int, default=None, help="Override total tokens from config")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Custom checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log-interval", type=int, default=None, help="Override log interval from config")
    
    # WSD scheduler args (CLI overrides config)
    parser.add_argument("--wsd-stage", type=str, default=None, choices=["decay_only", "full"], help="WSD stage (overrides config)")
    parser.add_argument("--decay-tokens", type=int, default=None, help="Number of tokens for WSD decay phase")
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
    
    effective_batch_size = batch_size * world_size * args.grad_accum_steps
    tokens_per_step = effective_batch_size * max_seq_len
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
        batch_size=batch_size,
        rank=rank,
        world_size=world_size
    )

    warmup_steps = training_config.get("warmup_steps", 2000)
    # schedule_type is implicit: "wsd"
    wsd_stage = args.wsd_stage or training_config.get("wsd_stage", "full")
    # calculate decay steps based on stage
    # full: auto 10% of max_token_budget
    # decay_only: requires --decay-tokens
    if wsd_stage == "full":
        decay_steps = int(0.1 * max_steps) # 10% of total budget
        if rank == 0:
            decay_tokens = decay_steps * tokens_per_step
            print(f"WSD Full: Decay phase = 10% ({decay_tokens:,} tokens / {decay_steps} steps)")
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
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        wsd_stage=wsd_stage,
        decay_steps=decay_steps,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume, dataloader)
        if rank == 0:
            print(f"Resumed from {args.resume} at step {trainer.step}")

    # WSD decay logic: reset max_steps relative to current step
    if wsd_stage == "decay_only":
        if args.decay_tokens:
            decay_steps = args.decay_tokens // tokens_per_step
        else:
            raise ValueError("For WSD decay_only stage, you must provide --decay-tokens")
        
        trainer.decay_steps = decay_steps
        trainer.max_steps = trainer.step + decay_steps
        if rank == 0:
            print(f"WSD decay only: Training for {decay_steps} steps (stopping at step {trainer.max_steps})")

    eval_config = all_configs.get("evaluation", {})
    log_interval = args.log_interval or eval_config.get("log_interval", 10)

    trainer.train(dataloader, max_steps=max_steps, log_interval=log_interval)

    if rank == 0:
        wandb.finish()
    cleanup_distributed()

if __name__ == "__main__":
    main()
    