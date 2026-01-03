"""
Needle-in-a-Haystack (NIAH) Sweep Evaluator (Tier 3)

Systematically tests model's ability to retrieve information placed at
different positions across varying context lengths. Produces heatmap data
showing exactly where each PE type struggles.

This is more comprehensive than passkey retrieval - it sweeps across
position percentages (0% to 100%) and multiple context lengths.

Usage:
    python -m src.evaluation.eval_niah --checkpoint <path>
    python -m src.evaluation.eval_niah --checkpoint <path> --context 2048 4096
"""

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.evaluation.base import BaseEvaluator, load_tokenizer


# Filler text for padding context
FILLER_SENTENCE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


class NIAHEvaluator(BaseEvaluator):
    """Evaluator for needle-in-a-haystack position sweep."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(checkpoint_path, config_path, device, dtype)
        self.tokenizer = load_tokenizer()

    def generate_needle(self, length: int = 5) -> str:
        """Generate a random numeric needle (secret number)."""
        return "".join(str(random.randint(0, 9)) for _ in range(length))

    def generate_example(
        self,
        target_length: int,
        position_percent: float,
    ) -> tuple[str, str]:
        """
        Generate a NIAH example with needle at specified position.

        Args:
            target_length: Target total token length
            position_percent: Where to place needle (0.0 = start, 1.0 = end)

        Returns:
            (prompt, needle)
        """
        needle = self.generate_needle()

        # simple pattern matching template - avoids instruction-following confounds
        # at small scales models can't follow complex instructions anyway
        instruction = ""
        needle_line = f"KEY:{needle}\n"
        question = "KEY:"

        # tokenize fixed parts
        instruction_tokens = self.tokenizer.encode(instruction)
        needle_tokens = self.tokenizer.encode(needle_line)
        question_tokens = self.tokenizer.encode(question)
        fixed_token_count = len(instruction_tokens) + len(needle_tokens) + len(question_tokens)

        # filler budget in tokens
        filler_budget = max(0, target_length - fixed_token_count)

        if filler_budget == 0:
            # context too short for filler
            final_tokens = instruction_tokens + needle_tokens + question_tokens
            prompt = self.tokenizer.decode(final_tokens)
            return prompt, needle

        # generate enough filler tokens
        filler_tokens_per_sentence = len(self.tokenizer.encode(FILLER_SENTENCE))
        num_sentences = max(1, (filler_budget // filler_tokens_per_sentence) + 2)
        filler_text = FILLER_SENTENCE * num_sentences
        filler_tokens = self.tokenizer.encode(filler_text)

        # truncate to exact budget
        if len(filler_tokens) > filler_budget:
            filler_tokens = filler_tokens[:filler_budget]

        # split filler based on position_percent
        # 0.0 = needle at start, 1.0 = needle at end
        split_idx = int(len(filler_tokens) * position_percent)
        filler_before_tokens = filler_tokens[:split_idx]
        filler_after_tokens = filler_tokens[split_idx:]

        # build final sequence in token space to avoid round-trip issues
        final_tokens = (
            instruction_tokens
            + filler_before_tokens
            + needle_tokens
            + filler_after_tokens
            + question_tokens
        )

        # decode to text
        prompt = self.tokenizer.decode(final_tokens)

        return prompt, needle

    def evaluate_single(self, prompt: str, needle: str) -> bool:
        """Evaluate a single NIAH retrieval."""
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=self.device)

        # Generate (needle is typically 5 digits)
        output = self.model.generate(input_ids, max_new_tokens=10, temperature=0.0)

        generated_tokens = output[0, len(tokens):].tolist()
        generated_text = self.tokenizer.decode(generated_tokens).strip()

        # Strict match: generated text should start with the needle
        # (more precise than substring match which can inflate accuracy)
        return generated_text.startswith(needle)

    def evaluate(
        self,
        context_lengths: list[int] | None = None,
        position_percentages: list[float] | None = None,
        samples_per_cell: int = 20,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Run NIAH sweep evaluation.

        Args:
            context_lengths: List of context lengths to test
            position_percentages: List of position percentages (0.0 to 1.0)
            samples_per_cell: Number of samples per (context, position) cell
            seed: Random seed for reproducibility

        Returns:
            Dict with accuracy for each (context_length, position) combination
            and a 'heatmap' key with the full matrix
        """
        # set seed for reproducibility
        random.seed(seed)

        if self.model is None:
            self.load_model()

        train_len = self.config.max_seq_len if self.config else 2048

        # default context lengths including extrapolation
        if context_lengths is None:
            context_lengths = [512, 1024, train_len, train_len * 2, train_len * 4]
            context_lengths = [c for c in context_lengths if c <= 8192]

        # default position sweep
        if position_percentages is None:
            position_percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        results = {
            "model": str(self.checkpoint_path),
            "pe_type": self.config.pe_type if self.config else "unknown",
            "train_max_seq_len": train_len,
            "context_lengths": context_lengths,
            "position_percentages": position_percentages,
            "samples_per_cell": samples_per_cell,
        }

        # initialize heatmap matrix
        heatmap = {}

        for ctx_len in context_lengths:
            # label as interpolation or extrapolation
            regime = "interp" if ctx_len <= train_len else "extrap"
            print(f"\nEvaluating NIAH at context={ctx_len} ({regime})...")
            heatmap[ctx_len] = {}

            for pos_pct in tqdm(position_percentages, desc=f"ctx={ctx_len}"):
                correct = 0

                for _ in range(samples_per_cell):
                    prompt, needle = self.generate_example(ctx_len, pos_pct)
                    if self.evaluate_single(prompt, needle):
                        correct += 1

                accuracy = correct / samples_per_cell
                heatmap[ctx_len][pos_pct] = accuracy

                # store with regime label
                pos_label = f"{int(pos_pct * 100)}pct"
                results[f"niah_{ctx_len}_{pos_label}_{regime}"] = accuracy

            # row average (across positions for this context length)
            row_avg = sum(heatmap[ctx_len].values()) / len(position_percentages)
            results[f"niah_{ctx_len}_avg_{regime}"] = row_avg
            print(f"  Average: {row_avg:.2%}")

        # Store full heatmap for visualization
        results["niah_heatmap"] = heatmap

        # Overall average
        all_accs = [acc for row in heatmap.values() for acc in row.values()]
        results["niah_overall_avg"] = sum(all_accs) / len(all_accs) if all_accs else 0.0

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate needle-in-a-haystack retrieval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--context", type=int, nargs="+", help="Context lengths to test")
    parser.add_argument("--samples", type=int, default=20, help="Samples per cell")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    evaluator = NIAHEvaluator(checkpoint_path=args.checkpoint)
    results = evaluator.evaluate(
        context_lengths=args.context,
        samples_per_cell=args.samples,
    )
    evaluator.log_results(results, args.output)


if __name__ == "__main__":
    main()
