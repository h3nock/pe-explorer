#!/usr/bin/env python3
"""Training script with WSD scheduler.

Supports two modes:
- full: Train from scratch through warmup -> stable -> decay phases, saving
        branch checkpoints at each target budget's 90% point for later decay runs.
- decay_only: Resume from a branch checkpoint and run only the decay phase.
"""
import argparse
import random
import yaml
import numpy as np
import torch
import torch.multiprocessing as mp

from torch.optim.adamw import AdamW

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import get_dataloader
from src.training.trainer import Trainer, setup_distributed, cleanup_distributed, format_budget

from dotenv import load_dotenv 
load_dotenv()

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    config, training_config, eval_config, data_config = load_configs(args.model_size)
    
    wsd_stage = args.wsd_stage or training_config.get("wsd_stage", "full")
    
    # training parameters
    batch_size = args.batch_size or config["batch_size"]
    max_token_budget = args.tokens or config["max_token_budget"]
    max_seq_len = config["max_seq_len"]
    seed = args.seed
    
    base_checkpoint_dir = args.checkpoint_dir or f"checkpoints/{args.model_size}_{args.pe_type}_s{seed}"
    warmup_steps = training_config.get("warmup_steps", 2000)
    log_interval = args.log_interval or eval_config.get("log_interval", 10)
    group_name = f"{args.model_size}_{args.pe_type}"
    
    # set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    model, optimizer = create_model_and_optimizer(config, training_config, args)
    dataloader = get_dataloader(
        seq_len=max_seq_len, 
        batch_size=batch_size, 
        rank=rank, 
        world_size=world_size,
        data_config=data_config,
        seed=seed,
    )
    
    effective_batch_size = batch_size * world_size * args.grad_accum_steps
    tokens_per_step = effective_batch_size * max_seq_len
    
    # base config for wandb
    base_config = {
        **config,
        "model_size": args.model_size,
        "pe_type": args.pe_type,
        "batch_size": batch_size,
        "seed": seed,
    }
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        grad_accum_steps=args.grad_accum_steps,
        grad_clip=training_config["grad_clip"],
        warmup_steps=warmup_steps,
        checkpoint_interval=max(max_token_budget // tokens_per_step // 20, 25),
        checkpoint_dir=base_checkpoint_dir,
    )
    
    checkpoint = None
    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume, dataloader)
        if rank == 0:
            print(f"Resumed from {args.resume} at step {trainer.step}")
    
    # WSD stage specific configuration
    if wsd_stage == "full":
        run_full_stage(trainer, dataloader, config, training_config, tokens_per_step, max_token_budget, base_config, group_name, rank, log_interval)
    elif wsd_stage == "decay_only":
        run_decay_stage(trainer, dataloader, checkpoint, args, tokens_per_step, base_config, group_name, rank, log_interval)
    
    cleanup_distributed()


def run_full_stage(trainer, dataloader, config, training_config, tokens_per_step, max_token_budget, base_config, group_name, rank, log_interval):
    """Multi-budget training with branch checkpoints."""
    target_budgets = config.get("target_budgets", [max_token_budget])
    target_budgets = sorted(set(int(b) for b in target_budgets))
    
    decay_ratio = training_config.get("decay_ratio", 0.1)
    max_steps = max_token_budget // tokens_per_step
    decay_steps = int(decay_ratio * max_token_budget) // tokens_per_step
    target_checkpoints = [(int(b * (1 - decay_ratio)), b) for b in target_budgets]
    
    # configure WSD (uses checkpoint's max_steps if resuming, else calculated)
    if trainer.max_steps == 0:
        trainer.configure_wsd("full", max_steps, decay_steps, target_checkpoints)
    else:
        trainer.configure_wsd("full", trainer.max_steps, trainer.decay_steps or decay_steps, target_checkpoints)
    
    trainer.init_wandb(base_config, group_name, max_token_budget)
    trainer.train(dataloader, log_interval=log_interval)
    trainer.finish_wandb()


def run_decay_stage(trainer, dataloader, checkpoint, args, tokens_per_step, base_config, group_name, rank, log_interval):
    """Decay-only training from branch checkpoint."""
    if not args.resume:
        raise ValueError("decay_only requires --resume")
    if not args.decay_tokens:
        raise ValueError("decay_only requires --decay-tokens")
    if not trainer.target_budget:
        raise ValueError(
            "decay_only requires a branch checkpoint (ready_for_decay_*.pt) with a target_budget set."
        )
    
    decay_steps = args.decay_tokens // tokens_per_step
    
    # differentiate between start fresh decay vs resume crashed decay
    is_fresh_decay = checkpoint and checkpoint.get("checkpoint_type") == "pre_decay"
    
    if is_fresh_decay:
        decay_start = trainer.step
    else:
        decay_start = trainer.decay_start_step or trainer.step
    
    max_steps = decay_start + decay_steps
    
    if trainer.target_budget and args.checkpoint_dir is None:
        trainer.checkpoint_dir = f"{trainer.checkpoint_dir}/decay_{format_budget(trainer.target_budget)}"
    
    trainer.configure_wsd("decay_only", max_steps, decay_steps)
    
    trainer.init_wandb(base_config, group_name, trainer.target_budget)
    if rank == 0:
        print(f"WSD decay: {decay_steps} steps (from {trainer.step} to {max_steps})")
    
    trainer.train(dataloader, log_interval=log_interval)
    trainer.finish_wandb()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, default="tiny", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--pe-type", type=str, default="sinusoidal", choices=["none", "sinusoidal", "binary", "binary_norm", "decimal", "decimal_norm"])
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--tokens", type=int, default=None, help="Override total tokens from config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Custom checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log-interval", type=int, default=None, help="Override log interval")
    parser.add_argument("--wsd-stage", type=str, default=None, choices=["decay_only", "full"], help="WSD stage")
    parser.add_argument("--decay-tokens", type=int, default=None, help="Decay phase tokens (required for decay_only)")
    return parser.parse_args()


def load_configs(model_size):
    with open("configs/config.yaml") as f:
        all_configs = yaml.safe_load(f)
    return all_configs[model_size], all_configs["training"], all_configs.get("evaluation", {}), all_configs.get("data", {})


def create_model_and_optimizer(config, training_config, args):
    model_config = ModelConfig(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
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
    return model, optimizer


if __name__ == "__main__":
    main()
