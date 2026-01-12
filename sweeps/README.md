# Hyperparameter Sweeps

## Quick Start

```bash
# 0. Make the sweep wrapper executable (one-time setup)
chmod +x scripts/run_sweep.sh

# 1. Create sweep and save the id inside sweep.log
wandb sweep sweeps/tiny_hyperparams.yaml | tee sweep.log

# 2. Launch agents
# Single GPU per agent:
CUDA_VISIBLE_DEVICES=0 wandb agent <sweep-id> &
CUDA_VISIBLE_DEVICES=1 wandb agent <sweep-id> &

# Two GPUs per agent:
CUDA_VISIBLE_DEVICES=0,1 wandb agent <sweep-id> &
```

## Scaling Hyperparameters

After finding optimal config on tiny, scale to larger models:

```bash
python scripts/scale_hyperparams.py
```

See script for scaling formulas.
