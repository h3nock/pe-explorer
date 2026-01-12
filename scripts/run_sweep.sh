#!/bin/bash
# flexible sweep wrapper - auto-detects GPU configuration

# count available GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    [ $NUM_GPUS -eq 0 ] && NUM_GPUS=1
fi

# single GPU: direct execution
if [ $NUM_GPUS -eq 1 ]; then
    exec python train.py "$@"
fi

# multi-GPU: use DDP
exec torchrun --standalone --nproc-per-node=$NUM_GPUS train.py "$@"
