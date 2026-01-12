#!/usr/bin/env python3
"""Training script with WSD scheduler.

Supports two modes:
- full: Train from scratch through warmup -> stable -> decay phases, saving
        branch checkpoints at each target budget's 90% point for later decay runs.
- decay_only: Resume from a branch checkpoint and run only the decay phase.
"""
import argparse
import os
import random
import subprocess
from pathlib import Path

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

def capture_environment() -> dict:
    """Capture git commit and check for uncommitted changes.

    Returns:
        dict with 'git_commit' and 'git_dirty' keys
    """
    env_info = {'git_commit': 'unknown', 'git_dirty': False}

    try:
        # get current commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            env_info['git_commit'] = result.stdout.strip()

        # check for uncommitted changes
        result = subprocess.run(
            ['git', 'diff', 'HEAD', '--quiet'],
            capture_output=True, timeout=5
        )
        env_info['git_dirty'] = result.returncode != 0

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass  # not in a git repo or git not available

    return env_info


def save_requirements(checkpoint_dir: str, rank: int = 0) -> None:
    """Save pip freeze output to checkpoint directory"""
    if rank != 0:
        return

    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        req_path = os.path.join(checkpoint_dir, 'requirements.txt')

        result = subprocess.run(
            ['pip', 'freeze'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            with open(req_path, 'w') as f:
                f.write(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"Warning: could not save requirements.txt: {e}")


def configs_match(config1: dict, config2: dict) -> bool:
    """Check if two configs match on fields that should trigger version bump.

    Includes architecture (affects weights) and training behavior (dropout).
    """
    critical_keys = [
        'd_model', 'n_layers', 'n_heads', 'd_ff',
        'max_seq_len', 'vocab_size', 'pe_type',
        'pe_params', 'tie_embedding', 'dropout'
    ]
    for key in critical_keys:
        if config1.get(key) != config2.get(key):
            return False
    return True


def get_run_name(base_name: str, current_config: dict, checkpoint_base_dir: Path, rank: int = 0) -> str:
    """Get run name with auto-versioning if config differs from existing checkpoint.

    This only determines the directory name - actual resume requires --resume flag.
    """
    checkpoint_dir = checkpoint_base_dir / base_name

    # case 1: no existing checkpoint -> use base name
    if not checkpoint_dir.exists() or not any(checkpoint_dir.glob("*.pt")):
        return base_name

    # case 2: existing checkpoint -> check if configs match
    try:
        ckpt_path = next(checkpoint_dir.glob("*.pt"))
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        saved_config = checkpoint.get('config', {})

        if configs_match(current_config, saved_config):
            return base_name  # same config, use same directory
    except (StopIteration, Exception):
        pass  # can't load checkpoint, treat as config mismatch

    # case 3: config differs -> find next available version
    for version in range(1, 101):
        versioned_name = f"{base_name}_v{version}"
        versioned_dir = checkpoint_base_dir / versioned_name

        if not versioned_dir.exists():
            if rank == 0:
                print(f"Config changed, using new version: {versioned_name}")
            return versioned_name

        # check if this version's config matches
        try:
            ckpt_files = list(versioned_dir.glob("*.pt"))
            if ckpt_files:
                checkpoint = torch.load(ckpt_files[0], map_location='cpu', weights_only=False)
                if configs_match(current_config, checkpoint.get('config', {})):
                    return versioned_name
        except Exception:
            pass

    raise RuntimeError(f"Too many versions for {base_name} (max 100)")


if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    config, training_config, validation_config, data_config = load_configs(args.model_size)

    wsd_stage = args.wsd_stage or training_config.get("wsd_stage", "full")

    # training parameters
    batch_size = args.batch_size or config["batch_size"]
    max_token_budget = args.tokens or config["max_token_budget"]
    max_seq_len = config["max_seq_len"]
    seed = args.seed

    log_interval = args.log_interval or validation_config.get("log_interval", 10)
    eval_interval = args.eval_interval or validation_config.get("interval", 500)
    group_name = f"{args.model_size}_{args.pe_type}"

    # build config dict for version comparison (must match ModelConfig fields)
    version_config = {
        'd_model': config['d_model'],
        'n_layers': config['n_layers'],
        'n_heads': config['n_heads'],
        'd_ff': config['d_ff'],
        'max_seq_len': max_seq_len,
        'vocab_size': training_config['vocab_size'],
        'pe_type': args.pe_type,
        'pe_params': training_config.get('pe_params', {}),
        'tie_embedding': training_config.get('tie_embedding', True),
        'dropout': training_config.get('dropout', 0.0),
    }

    # determine run name (explicit, from checkpoint dir, or auto-generated)
    if args.run_name:
        run_name = args.run_name
    elif args.checkpoint_dir:
        run_name = Path(args.checkpoint_dir).name
    else:
        budget_str = format_budget(max_token_budget)
        base_name = f"{args.model_size}_{args.pe_type}_{budget_str}_s{seed}"
        run_name = get_run_name(base_name, version_config, Path("checkpoints"), rank)

    base_checkpoint_dir = args.checkpoint_dir or f"checkpoints/{run_name}"

    # set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, optimizer = create_model_and_optimizer(config, training_config, args)

    # training dataloader
    dataloader = get_dataloader(
        mode="train",
        data_config=data_config,
        seq_len=max_seq_len,
        batch_size=batch_size,
        token_budget=max_token_budget,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    # validation dataloader
    val_token_budget = validation_config.get("token_budget")
    val_dataloader = get_dataloader(
        mode="val",
        data_config=data_config,
        seq_len=max_seq_len,
        batch_size=batch_size,
        token_budget=val_token_budget,
        rank=rank,
        world_size=world_size,
        seed=seed,
        num_workers=2,
    )
    if rank == 0:
        print(f"Validation: {len(val_dataloader)} batches ({val_token_budget:,} tokens)")

    effective_batch_size = batch_size * world_size * args.grad_accum_steps
    tokens_per_step = effective_batch_size * max_seq_len
    
    # dynamic warmup: min(config value, 3% of total steps)
    max_train_steps = max_token_budget // tokens_per_step
    warmup_steps = min(training_config.get("warmup_steps", 2000), int(0.03 * max_train_steps))
    
    # base config for wandb
    base_config = {
        **config,
        "model_size": args.model_size,
        "pe_type": args.pe_type,
        "batch_size": batch_size,
        "seed": seed,
        "run_name": run_name,
        "group": group_name,
    }

    # capture environment for reproducibility
    env_info = capture_environment()
    if rank == 0:
        if env_info['git_dirty']:
            print(f"WARNING: Uncommitted changes detected! Commit hash: {env_info['git_commit']}")
        else:
            print(f"Git commit: {env_info['git_commit']}")

    run_metadata = {
        "model_size": args.model_size,
        "pe_type": args.pe_type,
        "seed": seed,
        "tokenizer_name": "gpt2",
        "size_config": config,
        "training_config": training_config,
        "validation_config": validation_config,
        "data_config": data_config,
        "cli_args": vars(args),
        "environment": env_info,
    }

    # save requirements.txt to checkpoint dir (happens only on fresh runs and rank 0)
    if not args.resume:
        save_requirements(base_checkpoint_dir, rank)

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
        run_metadata=run_metadata,
    )
    
    checkpoint = None
    samples_seen = 0
    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume, dataloader)
        samples_seen = checkpoint.get("samples_seen", 0)
        if rank == 0:
            print(f"Resumed from {args.resume} at step {trainer.step} ({samples_seen:,} samples seen)")

        # re-create dataloader with correct offset if needed
        if samples_seen > 0:
            dataloader = get_dataloader(
                mode="train",
                data_config=data_config,
                seq_len=max_seq_len,
                batch_size=batch_size,
                token_budget=max_token_budget,
                rank=rank,
                world_size=world_size,
                samples_seen=samples_seen,
                seed=seed,
            )
    
    # WSD stage specific configuration
    if wsd_stage == "full":
        run_full_stage(trainer, dataloader, val_dataloader, eval_interval, config, training_config, tokens_per_step, max_token_budget, base_config, group_name, rank, log_interval, run_name)
    elif wsd_stage == "decay_only":
        run_decay_stage(trainer, dataloader, val_dataloader, eval_interval, checkpoint, args, tokens_per_step, base_config, group_name, rank, log_interval, run_name)
    
    cleanup_distributed()


def run_full_stage(trainer, dataloader, val_dataloader, eval_interval, config, training_config, tokens_per_step, max_token_budget, base_config, group_name, rank, log_interval, run_name):
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

    trainer.init_wandb(base_config, group_name, max_token_budget, run_name=run_name)
    trainer.train(dataloader, val_dataloader=val_dataloader, log_interval=log_interval, eval_interval=eval_interval)
    trainer.finish_wandb()


def run_decay_stage(trainer, dataloader, val_dataloader, eval_interval, checkpoint, args, tokens_per_step, base_config, group_name, rank, log_interval, run_name):
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
        
        # validate consistency for mid-decay resume
        if trainer.decay_steps != decay_steps:
            raise ValueError(
                f"Resuming mid-decay run with different decay length!\n"
                f"Checkpoint decay_steps: {trainer.decay_steps}\n"
                f"Current --decay-tokens implies: {decay_steps}\n"
                "You must use the same --decay-tokens value as the run you are resuming."
            )
    
    max_steps = decay_start + decay_steps
    
    if trainer.target_budget and args.checkpoint_dir is None:
        trainer.checkpoint_dir = f"{trainer.checkpoint_dir}/decay_{format_budget(trainer.target_budget)}"
    
    trainer.configure_wsd("decay_only", max_steps, decay_steps)

    trainer.init_wandb(base_config, group_name, trainer.target_budget, run_name=run_name)
    if rank == 0:
        print(f"WSD decay: {decay_steps} steps (from {trainer.step} to {max_steps})")

    trainer.train(dataloader, val_dataloader=val_dataloader, log_interval=log_interval, eval_interval=eval_interval)
    trainer.finish_wandb()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, default="tiny", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--pe-type", type=str, default="rope", choices=["none", "sinusoidal", "sinonly", "binary", "binary_norm", "decimal", "decimal_norm", "rope"])
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--tokens", type=int, default=None, help="Override total tokens from config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Custom checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log-interval", type=int, default=None, help="Override log interval")
    parser.add_argument("--eval-interval", type=int, default=None, help="Override eval interval")
    parser.add_argument("--wsd-stage", type=str, default=None, choices=["decay_only", "full"], help="WSD stage")
    parser.add_argument("--decay-tokens", type=int, default=None, help="Decay phase tokens (required for decay_only)")
    parser.add_argument("--run-name", type=str, default=None, help="Override auto-generated run name (skips auto-versioning)")
    return parser.parse_args()


def load_configs(model_size):
    with open("configs/config.yaml") as f:
        all_configs = yaml.safe_load(f)
    return (
        all_configs[model_size],
        all_configs["training"],
        all_configs.get("validation", {}),
        all_configs.get("data", {}),
    )


def create_model_and_optimizer(config, training_config, args):
    model_config = ModelConfig(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        vocab_size=training_config["vocab_size"],
        pe_type=args.pe_type,
        pe_params=training_config.get("pe_params", {}),
        dropout=training_config.get("dropout", 0.0),
        tie_embedding=training_config.get("tie_embedding", True),
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
