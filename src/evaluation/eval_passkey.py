"""
Passkey Retrieval Evaluator (Tier 3)

Tests model's ability to retrieve a passkey hidden in a long context.
Critical for evaluating positional encoding's ability to handle long sequences.

Usage:
    python -m src.evaluation.eval_passkey --checkpoint <path> --context 4096
    python -m src.evaluation.eval_passkey --checkpoint <path> --all-contexts
"""

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.evaluation.base import BaseEvaluator, load_tokenizer


# Filler text (simple repetitive text to pad context)
FILLER_SENTENCE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


class PasskeyEvaluator(BaseEvaluator):
    """Evaluator for passkey retrieval task."""
    
    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(checkpoint_path, config_path, device, dtype)
        self.tokenizer = load_tokenizer()
    
    def generate_passkey(self, length: int = 5) -> str:
        """Generate a random numeric passkey."""
        return "".join(str(random.randint(0, 9)) for _ in range(length))
    
    def generate_example(
        self,
        target_length: int,
        passkey_position: str = "middle",
    ) -> tuple[str, str]:
        """
        Generate a passkey retrieval example.

        Args:
            target_length: Target total token length
            passkey_position: 'beginning', 'middle', or 'end'

        Returns:
            (prompt, passkey)
        """
        passkey = self.generate_passkey()

        # simple pattern matching template - avoids instruction-following confounds
        # at small scales models can't follow complex instructions anyway
        instruction = ""
        passkey_line = f"KEY:{passkey}\n"
        question = "KEY:"

        # tokenize fixed parts
        instruction_tokens = self.tokenizer.encode(instruction)
        passkey_tokens = self.tokenizer.encode(passkey_line)
        question_tokens = self.tokenizer.encode(question)
        fixed_token_count = len(instruction_tokens) + len(passkey_tokens) + len(question_tokens)

        # filler budget in tokens
        filler_budget = max(0, target_length - fixed_token_count)

        # generate enough filler tokens
        filler_tokens_per_sentence = len(self.tokenizer.encode(FILLER_SENTENCE))
        num_sentences = max(1, (filler_budget // filler_tokens_per_sentence) + 2)
        filler_text = FILLER_SENTENCE * num_sentences
        filler_tokens = self.tokenizer.encode(filler_text)

        # truncate to exact budget
        if len(filler_tokens) > filler_budget:
            filler_tokens = filler_tokens[:filler_budget]

        # split filler based on position
        if passkey_position == "beginning":
            filler_before_tokens = []
            filler_after_tokens = filler_tokens
        elif passkey_position == "end":
            filler_before_tokens = filler_tokens
            filler_after_tokens = []
        else:  # middle
            mid = len(filler_tokens) // 2
            filler_before_tokens = filler_tokens[:mid]
            filler_after_tokens = filler_tokens[mid:]

        # build final sequence in token space to avoid round-trip issues
        final_tokens = (
            instruction_tokens
            + filler_before_tokens
            + passkey_tokens
            + filler_after_tokens
            + question_tokens
        )

        # Decode to text for the prompt
        prompt = self.tokenizer.decode(final_tokens)

        return prompt, passkey
    
    def evaluate_single(self, prompt: str, passkey: str) -> bool:
        """Evaluate a single passkey retrieval."""
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=self.device)
        
        # Generate (passkey is typically 5 digits)
        output = self.model.generate(input_ids, max_new_tokens=10, temperature=0.0)
        
        generated_tokens = output[0, len(tokens):].tolist()
        generated_text = self.tokenizer.decode(generated_tokens).strip()
        
        # Strict match: generated text should start with the passkey
        # (more precise than substring match which can inflate accuracy)
        return generated_text.startswith(passkey)
    
    def evaluate(
        self,
        context_lengths: list[int] | None = None,
        samples_per_context: int = 100,
        positions: list[str] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run passkey retrieval evaluation.

        Args:
            context_lengths: List of context lengths to test
            samples_per_context: Number of samples per context length
            positions: List of positions to test ('beginning', 'middle', 'end')
            seed: Random seed for reproducibility
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

        positions = positions or ["beginning", "middle", "end"]

        results = {
            "model": str(self.checkpoint_path),
            "pe_type": self.config.pe_type if self.config else "unknown",
            "train_max_seq_len": train_len,
        }

        for ctx_len in context_lengths:
            # label as interpolation or extrapolation
            regime = "interp" if ctx_len <= train_len else "extrap"
            print(f"\nEvaluating passkey retrieval at context={ctx_len} ({regime})...")

            for position in positions:
                correct = 0
                total = samples_per_context // len(positions)

                for _ in tqdm(range(total), desc=f"{position}", leave=False):
                    prompt, passkey = self.generate_example(ctx_len, position)
                    if self.evaluate_single(prompt, passkey):
                        correct += 1

                acc = correct / total if total > 0 else 0.0
                results[f"passkey_{ctx_len}_{position}_{regime}"] = acc
                print(f"  {position}: {correct}/{total} = {acc:.2%}")

            # average across positions
            pos_accs = [results[f"passkey_{ctx_len}_{p}_{regime}"] for p in positions]
            results[f"passkey_{ctx_len}_avg_{regime}"] = sum(pos_accs) / len(pos_accs)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate passkey retrieval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--context", type=int, nargs="+", help="Context lengths to test")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    
    evaluator = PasskeyEvaluator(checkpoint_path=args.checkpoint)
    results = evaluator.evaluate(
        context_lengths=args.context,
        samples_per_context=args.samples,
    )
    evaluator.log_results(results, args.output)


if __name__ == "__main__":
    main()
