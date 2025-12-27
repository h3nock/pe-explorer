# Positional Encoding Benchmark

A benchmark for comparing, experimenting with, and understanding different positional encoding methods in transformer architectures.

## Goal
1. Training small language models with different positional encodings
2. Extracting scaling laws to predict behavior at larger scale
3. Analyzing how different PE methods affect model capabilities

## Experimental Approach

### Phase 1: Small LM Training
Train a family of small language models (1M → 500M params) with each positional encoding variant on the same dataset, measuring:
- Training loss curves
- Validation perplexity
- Compute efficiency (FLOPs to reach target loss)

### Phase 2: Scaling Law Extraction
Fit scaling laws of the form `L(N, D) = E + A/N^α + B/D^β` for each PE method and extrapolate to predict large-scale behavior.

### Phase 3: In-Context Learning
Evaluate how PE choice affects emergent few-shot learning capabilities.

## Dataset

**[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)**


## Positional Encoding Variants

| Name | Type | Description |
|------|------|-------------|
| `none` | Baseline | No positional encoding (NoPE) |
| `learned` | Absolute | Trainable position embeddings |
| `sinusoidal` | Absolute | Original "Attention Is All You Need" sin/cos |
| `sinusoidal_sin_only` | Absolute | Sinusoidal without cosine components |
| `binary` | Absolute | Binary representation of position |
| `decimal_padded` | Absolute | Decimal digits padded to d_model |
| `integer` | Absolute | Raw integer position values |
| `rope` | Relative | Rotary Position Embedding |

## Project Structure

```
pos-enc-bench/
├── README.md
├── requirements.txt
├── configs/                    # training configurations
│   ├── config.yaml             # 1M, 5M, 25M, 100M, 500M configs
│   └── pe_variants.yaml        # PE-specific settings
├── src/
│   ├── encodings/              # PE implementations
│   │   ├── base.py             # abstract base class
│   │   ├── sinusoidal.py
│   │   ├── rope.py
│   │   └── ...
│   ├── model/                  # transformer LM
│   │   ├── transformer.py
│   │   ├── attention.py
│   │   └── config.py
│   ├── data/                   # data loading & tokenization
│   │   └── fineweb.py
│   └── training/               # training loop & logging
│       ├── trainer.py
│       └── scaling.py          # scaling law fitting
├── experiments/                # experiment scripts
├── analysis/                   # notebooks & visualization
└── results/                    # checkpoints, logs, plots
    ├── checkpoints/
    ├── logs/
    └── figures/
```

## Model Sizes (Chinchilla-Optimal)

| Name | Params | d_model | n_layers | n_heads | Training Budget (20N) |
|------|--------|---------|----------|---------|-----------------|
| tiny | ~70M | 512 | 6 | 8 | 1.4B |
| small | ~160M | 768 | 12 | 12 | 3.2B |
| medium | ~410M | 1024 | 24 | 16 | 8.2B |
| large | ~1B | 2048 | 16 | 32 | 20B |

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original sinusoidal PE
- [RoFormer: RoPE](https://arxiv.org/abs/2104.09864) - Rotary embeddings
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556) - Optimal compute allocation
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) - Dataset
