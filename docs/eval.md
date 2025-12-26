# Evaluation Strategy

## Tier 1: Language Modeling
| Task | Dataset | Metric |
| :--- | :--- | :--- |
| Short Context PPL | WikiText-103 | Perplexity |
| Long Context PPL | PG-19 (2k, 4k tokens) | Perplexity |

## Tier 2: Algorithmic (PE-Specific)
| Task | Metric | Train Range | Eval Range |
| :--- | :--- | :--- | :--- |
| Addition | Exact Match | 1-20 digits | 21-40 digits |
| Sorting | Exact Match | 1-10 items | 11-20 items |
| Reversal | Exact Match | 1-20 chars | 21-40 chars |
| Copy | Exact Match | 1-50 chars | 51-100 chars |

## Tier 3: Precision
| Task | Dataset | Metric |
| :--- | :--- | :--- |
| Passkey Retrieval | Synthetic (512-8k) | Accuracy |
| LAMBADA | HuggingFace | Accuracy |

## Scripts
- `src/evaluation/eval_ppl.py` - WikiText-103 / PG-19
- `src/evaluation/eval_algorithmic.py` - Algorithmic tasks
- `src/evaluation/eval_passkey.py` - Passkey retrieval
- `src/evaluation/eval_lambada.py` - LAMBADA
