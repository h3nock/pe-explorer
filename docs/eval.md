# PE Evaluation Suite

This document serves as the reference manual for evaluating models trained in this repository.

## Usage

```bash
# evaluate algorithmic tasks (Tier 2)
python eval.py --checkpoint checkpoints/your_run/final.pt --tasks algorithmic

# evaluate Perplexity (Tier 1)
python eval.py --checkpoint checkpoints/your_run/final.pt --tasks ppl

# run everything
python eval.py --checkpoint checkpoints/your_run/final.pt --tasks all
```

---

## Tier 1: Language Modeling
General capability check using FineWeb validation data.

| Task | Dataset | Metric |
| :--- | :--- | :--- |
| **PPL** | FineWeb (Eval Split) | Perplexity |

---

## Tier 2: Algorithmic (Logic & Retrieval)
Synthetic tasks designed to test specific positional generalization capabilities.
The evaluator (`src/evaluation/eval_algorithmic.py`) generates fresh samples using the configuration in `configs/data_generation.yaml`.

### Task Specifications

| Task | Description | Train Range | Eval ID (In-Distrib) | Eval OOD (Extrapolation) |
| :--- | :--- | :--- | :--- | :--- |
| **Passkey** | Retrieve key at specific position | dist 0.25-1.0 | dist 0.25-1.0 | dist 1.5-2.0 |
| **Copy Dist** | Copy 5-item sequence from past | dist 0.25-1.0 | dist 0.25-1.0 | dist 1.5-2.0 |
| **Reverse** | Reverse sequence order | len 3-8 | len 3-8 | len 9-16 |
| **Sort** | Sort items ascending | len 3-6 | len 3-6 | len 7-10 |
| **Add** | No-carry addition | 2-5 digits | 2-5 digits | 6-8 digits |
| **Simple Copy** | Identity function | len 4-12 | len 4-12 | len 13-20 |

### Metrics
All algorithmic tasks use **Exact Match** accuracy.
- **ID Accuracy**: Measures if the model learned the training distribution.
- **OOD Accuracy**: Measures if the positional encoding generalizes to longer sequences/distances.
 - **Exact Match definition**: Token-level target-span match. The generated continuation must start with the exact target tokens; any extra tokens after the target are ignored.

### Defaults
- The number of evaluation samples per task/mode defaults to `generation.num_eval_per_task` from `configs/data_generation.yaml`.

---
