"""
Perplexity Evaluator (Tier 1)

Evaluates language modeling perplexity on:
- WikiText-103 (short context)
- PG-19 (long context) at various context lengths

Usage:
    python -m src.evaluation.eval_ppl --checkpoint <path> --dataset wikitext103
    python -m src.evaluation.eval_ppl --checkpoint <path> --dataset pg19 --context 4096
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.evaluation.base import BaseEvaluator, load_tokenizer


EVAL_DATA_DIR = Path("data/eval")


class PPLEvaluator(BaseEvaluator):
    """Evaluator for perplexity on language modeling datasets."""
    
    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(checkpoint_path, config_path, device, dtype)
        self.tokenizer = load_tokenizer()
        
    def load_dataset(self, dataset: str, split: str = "test") -> list[str]:
        """Load dataset texts."""
        data_path = EVAL_DATA_DIR / dataset / f"{split}.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. "
                "Run: python -m src.data.prepare_eval_data --all"
            )
        
        texts = []
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                # WikiText uses 'text', PG-19 uses 'text' or 'short_book_title'
                text = item.get("text", "")
                if text.strip():
                    texts.append(text)
        
        return texts
    
    def compute_perplexity(
        self,
        texts: list[str],
        context_length: int,
        stride: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, float]:
        """
        Compute perplexity using sliding window.
        
        Args:
            texts: List of text strings
            context_length: Maximum context length for each forward pass
            stride: Overlap between windows (default: context_length // 2)
            max_tokens: Maximum total tokens to evaluate (for speed)
        """
        if stride is None:
            stride = context_length // 2
        
        total_loss = 0.0
        total_tokens = 0
        
        # Tokenize all texts into one long sequence
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            if max_tokens and len(all_tokens) >= max_tokens:
                all_tokens = all_tokens[:max_tokens]
                break
        
        all_tokens = torch.tensor(all_tokens, device=self.device)
        seq_len = len(all_tokens)
        
        print(f"Evaluating {seq_len:,} tokens with context={context_length}, stride={stride}")
        
        # Sliding window evaluation
        num_windows = max(1, (seq_len - context_length) // stride + 1)
        
        for i in tqdm(range(0, seq_len - 1, stride), desc="Computing PPL"):
            end = min(i + context_length, seq_len)
            
            # Get input and target
            input_ids = all_tokens[i:end-1].unsqueeze(0)  # (1, seq)
            targets = all_tokens[i+1:end].unsqueeze(0)     # (1, seq)
            
            if input_ids.size(1) == 0:
                continue
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(input_ids)  # (1, seq, vocab)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="sum",
            )
            
            # Only count tokens in the non-overlapping part (after first window)
            if i == 0:
                counted_tokens = input_ids.size(1)
            else:
                # Only count the stride portion (non-overlapping)
                counted_tokens = min(stride, input_ids.size(1))
                # Recompute loss for just the counted portion
                start_idx = max(0, input_ids.size(1) - stride)
                loss = F.cross_entropy(
                    logits[:, start_idx:].reshape(-1, logits.size(-1)),
                    targets[:, start_idx:].reshape(-1),
                    reduction="sum",
                )
            
            total_loss += loss.item()
            total_tokens += counted_tokens
            
            if end >= seq_len:
                break
        
        ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
        
        return {
            "perplexity": ppl,
            "total_loss": total_loss,
            "total_tokens": total_tokens,
            "context_length": context_length,
        }
    
    def evaluate(
        self,
        dataset: str = "wikitext103",
        context_lengths: list[int] | None = None,
        max_tokens: int = 1_000_000,
    ) -> dict[str, Any]:
        """Run perplexity evaluation."""
        if self.model is None:
            self.load_model()
        
        # Default context lengths based on training max_seq_len
        if context_lengths is None:
            train_len = self.config.max_seq_len if self.config else 2048
            context_lengths = [512, 1024, train_len, train_len * 2, train_len * 4]
            # Remove lengths larger than 8192 for practicality
            context_lengths = [c for c in context_lengths if c <= 8192]
        
        texts = self.load_dataset(dataset)
        print(f"Loaded {len(texts)} texts from {dataset}")

        train_len = self.config.max_seq_len if self.config else 2048

        results = {
            "model": str(self.checkpoint_path),
            "pe_type": self.config.pe_type if self.config else "unknown",
            "dataset": dataset,
            "train_max_seq_len": train_len,
        }

        for ctx_len in context_lengths:
            # label as interpolation or extrapolation
            regime = "interp" if ctx_len <= train_len else "extrap"
            print(f"\nEvaluating at context length {ctx_len} ({regime})...")
            ppl_result = self.compute_perplexity(
                texts,
                context_length=ctx_len,
                max_tokens=max_tokens,
            )
            results[f"ppl_{ctx_len}_{regime}"] = ppl_result["perplexity"]
            print(f"  PPL@{ctx_len}: {ppl_result['perplexity']:.2f}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="wikitext103", choices=["wikitext103", "pg19"])
    parser.add_argument("--context", type=int, nargs="+", help="Context lengths to test")
    parser.add_argument("--max-tokens", type=int, default=1_000_000, help="Max tokens to evaluate")
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()
    
    evaluator = PPLEvaluator(checkpoint_path=args.checkpoint)
    results = evaluator.evaluate(
        dataset=args.dataset,
        context_lengths=args.context,
        max_tokens=args.max_tokens,
    )
    evaluator.log_results(results, args.output)


if __name__ == "__main__":
    main()
