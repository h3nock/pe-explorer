# Positional Encoding Exploration

Empirical exploration of positional encoding methods in transformers. Training small language models with different PE variants (including unconventional ones like decimal/binary encodings) to validate whether they fail or succeed for the theoretical reasons commonly cited, not just that they do.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (FineWeb-Edu)
python -m src.data.prepare_fineweb download --max-shards 100
python -m src.data.prepare_fineweb tokenize

# 3. Train tiny model with RoPE
python train.py --model-size tiny --pe-type rope

# 4. Evaluate
python eval.py checkpoints/tiny_rope_2.4B_s42/latest.pt
```

## Positional Encodings Implemented

| Type | Description |
|------|-------------|
| `none` | No PE (baseline) |
| `sinusoidal` | Original sin/cos from "Attention Is All You Need" |
| `sinonly` | Sinusoidal without cosine |
| `binary` | Binary position representation |
| `binary_norm` | Normalized binary |
| `decimal` | Decimal digits |
| `decimal_norm` | Normalized decimal |
| `rope` | Rotary Position Embedding (RoFormer) |

## Model Sizes

| Name | Params | d_model | Layers | Heads | Token Budget |
|------|--------|---------|--------|-------|--------------|
| tiny | 59M | 512 | 6 | 8 | 2.4B |
| small | 164M | 768 | 12 | 12 | 6.6B |
| medium | 470M | 1024 | 24 | 16 | 18.8B |
| large | 1.2B | 2048 | 16 | 32 | 48.4B |

> Params assume `tie_embedding=true` (sharing token embeddings with LM head)

## Project Structure

```
pos-enc-bench/
├── train.py           # Main training script
├── eval.py            # Evaluation on algorithmic tasks
├── configs/
│   ├── config.yaml              # Model & training configs
│   └── data_generation.yaml     # Synthetic task configs
├── src/
│   ├── data/          # Data pipeline
│   ├── model/         # Transformer implementation
│   ├── training/      # Trainer with WSD scheduler
│   └── evaluation/    # Task evaluators
└── checkpoints/       # Saved models
```

## Training Features

- **WSD Scheduler:** Warmup → Stable → Decay with auto-branching for multi-budget experiments
- **DDP Support:** Multi-GPU training with `torchrun`
- **Auto-versioning:** Config-based checkpoint organization
- **WandB Logging:** Comprehensive metrics with token-based x-axis

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Sinusoidal PE
- [RoFormer: RoPE](https://arxiv.org/abs/2104.09864) - Rotary embeddings
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU activation
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - RMSNorm
- [QK-Normalization](https://arxiv.org/abs/2302.05442) - Query-key norm for stability
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Memory-efficient attention
- [Decoupled Weight Decay (AdamW)](https://arxiv.org/abs/1711.05101) - Optimizer
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556) - Compute-optimal training
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) - Training dataset
