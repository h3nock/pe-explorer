"""
Algorithmic Task Evaluator (Tier 2)

Evaluates on algorithmic tasks:
- Addition (multi-digit)
- Sorting
- Reversal
- Copy

Tests both:
- In-distribution (ID): lengths seen during training
- Out-of-distribution (OOD): lengths beyond training

Usage:
    python -m src.evaluation.eval_algorithmic --checkpoint <path> --task addition
    python -m src.evaluation.eval_algorithmic --checkpoint <path> --all
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from src.evaluation.base import BaseEvaluator, load_tokenizer


# Task configurations matching data_generation.yaml
TASK_CONFIG = {
    "addition": {"train_range": [1, 20], "eval_range": [21, 40]},
    "sorting": {"train_range": [1, 10], "eval_range": [11, 20]},
    "reversal": {"train_range": [1, 20], "eval_range": [21, 40]},
    "copy": {"train_range": [1, 50], "eval_range": [51, 100]},
}

# Default eval parquet location (matches prepare_algorithmic.py generate output).
DEFAULT_EVAL_PATH = Path.home() / ".cache" / "algorithmic" / "eval.parquet"

# Few-shot examples for in-context learning
# Must match format from prepare_algorithmic.py generators
FEW_SHOT_EXAMPLES = {
    "addition": [
        ("12+34=", "46"),
        ("99+1=", "100"),
        ("456+789=", "1245"),
    ],
    "sorting": [
        ("sort:[3,1,2]->", "[1,2,3]"),
        ("sort:[5,2,8,1]->", "[1,2,5,8]"),
        ("sort:[9,4,7]->", "[4,7,9]"),
    ],
    "reversal": [
        ("rev:abc->", "cba"),
        ("rev:hello->", "olleh"),
        ("rev:world->", "dlrow"),
    ],
    "copy": [
        ("copy:abc->", "abc"),
        ("copy:hello->", "hello"),
        ("copy:test->", "test"),
    ],
}


class AlgorithmicEvaluator(BaseEvaluator):
    """Evaluator for algorithmic reasoning tasks."""
    
    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        eval_data_path: str | Path = DEFAULT_EVAL_PATH,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(checkpoint_path, config_path, device, dtype)
        self.eval_data_path = Path(eval_data_path)
        self.tokenizer = load_tokenizer()
        
    def load_eval_data(self) -> pd.DataFrame:
        """Load evaluation data from parquet."""
        if not self.eval_data_path.exists():
            raise FileNotFoundError(
                f"Eval data not found at {self.eval_data_path}. "
                "Generate it with: python -m src.data.prepare_algorithmic generate "
                f"--output_dir {self.eval_data_path.parent}"
            )
        return pd.read_parquet(self.eval_data_path)
    
    def parse_example(self, text: str) -> tuple[str, str]:
        """Parse a text example into prompt and answer.

        Handles two formats from prepare_algorithmic.py:
        - addition: "12+34=46" -> ("12+34=", "46")
        - sorting/reversal/copy: "sort:[3,1,2]->[1,2,3]" -> ("sort:[3,1,2]->", "[1,2,3]")
        """
        # try -> separator first (sorting, reversal, copy)
        if "->" in text:
            parts = text.split("->", 1)
            prompt = parts[0] + "->"
            answer = parts[1] if len(parts) > 1 else ""
            return prompt, answer
        # fall back to = separator (addition)
        if "=" in text:
            parts = text.split("=", 1)
            prompt = parts[0] + "="
            answer = parts[1] if len(parts) > 1 else ""
            return prompt, answer
        return text, ""
    
    def build_few_shot_prompt(self, task: str, test_prompt: str, num_shots: int = 3) -> str:
        """Build few-shot prompt with examples."""
        examples = FEW_SHOT_EXAMPLES.get(task, [])[:num_shots]
        prompt_parts = []
        for ex_prompt, ex_answer in examples:
            prompt_parts.append(f"{ex_prompt}{ex_answer}")
        prompt_parts.append(test_prompt)
        return "\n".join(prompt_parts)
    
    def evaluate_single(
        self,
        prompt: str,
        answer: str,
        max_new_tokens: int = 256,
        return_generated: bool = False,
    ) -> bool | tuple[bool, str]:
        """Evaluate a single example."""
        # tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_tokens], device=self.device)

        # generate
        output_tokens = self.model.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            eos_token=self.tokenizer.eot_token,
        )

        # decode only the generated part
        generated_tokens = output_tokens[0, len(prompt_tokens):].tolist()
        generated_text = self.tokenizer.decode(generated_tokens)

        # extract answer - stop at EOT or newline (commas/spaces are valid in answers like [1,2,3])
        if "<|endoftext|>" in generated_text:
            generated_text = generated_text.split("<|endoftext|>", 1)[0]
        if "\n" in generated_text:
            generated_text = generated_text.split("\n", 1)[0]

        # exact match
        is_correct = generated_text.strip() == answer.strip()
        if return_generated:
            return is_correct, generated_text
        return is_correct
    
    def evaluate_task(
        self,
        task: str,
        examples: list[tuple[str, str]],
        use_few_shot: bool = False,
        verbose: bool = False,
        max_new_tokens: int = 256,
    ) -> dict[str, float]:
        """Evaluate a single task."""
        correct = 0
        total = len(examples)
        
        for prompt, answer in tqdm(examples, desc=f"{task}", leave=False):
            eval_prompt = self.build_few_shot_prompt(task, prompt) if use_few_shot else prompt

            if verbose:
                is_correct, generated_text = self.evaluate_single(
                    eval_prompt,
                    answer,
                    max_new_tokens,
                    return_generated=True,
                )
                detail = "\n".join(
                    [
                        "-" * 40,
                        f"task: {task}",
                        f"prompt: {eval_prompt!r}",
                        f"expected: {answer!r}",
                        f"generated: {generated_text!r}",
                        f"correct: {is_correct}",
                    ]
                )
                tqdm.write(detail)
            else:
                is_correct = self.evaluate_single(eval_prompt, answer, max_new_tokens)

            if is_correct:
                correct += 1
        
        return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}
    
    def classify_examples(
        self,
        df: pd.DataFrame,
        task: str,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Split examples into ID (in-distribution) and OOD (out-of-distribution)."""
        task_df = df[df["task"] == task]
        config = TASK_CONFIG[task]
        train_max = config["train_range"][1]
        has_split = "split" in task_df.columns
        
        id_examples = []
        ood_examples = []
        
        for _, row in task_df.iterrows():
            prompt, answer = self.parse_example(row["text"])
            if has_split:
                split = row.get("split")
                if isinstance(split, str):
                    split_lower = split.lower()
                    if split_lower.startswith("interp"):
                        id_examples.append((prompt, answer))
                        continue
                    if split_lower.startswith("extrap"):
                        ood_examples.append((prompt, answer))
                        continue

            length = row.get("length", len(prompt))
            if length <= train_max:
                id_examples.append((prompt, answer))
            else:
                ood_examples.append((prompt, answer))
        
        return id_examples, ood_examples
    
    def evaluate(
        self,
        tasks: list[str] | None = None,
        use_few_shot: bool = False,
        verbose: bool = False,
        max_examples_per_split: int = 500,
    ) -> dict[str, Any]:
        """Run evaluation on all or specified tasks."""
        if self.model is None:
            self.load_model()
        
        df = self.load_eval_data()
        tasks = tasks or list(TASK_CONFIG.keys())
        
        results = {
            "model": str(self.checkpoint_path),
            "pe_type": self.config.pe_type if self.config else "unknown",
            "mode": "few_shot" if use_few_shot else "zero_shot",
        }
        
        for task in tasks:
            print(f"\nEvaluating {task}...")
            id_examples, ood_examples = self.classify_examples(df, task)
            
            # limit examples if needed
            id_examples = id_examples[:max_examples_per_split]
            ood_examples = ood_examples[:max_examples_per_split]

            # evaluate ID
            if id_examples:
                id_result = self.evaluate_task(task, id_examples, use_few_shot, verbose=verbose)
                results[f"{task}_id_accuracy"] = id_result["accuracy"]
                results[f"{task}_id_correct"] = id_result["correct"]
                results[f"{task}_id_total"] = id_result["total"]
                print(f"  ID: {id_result['correct']}/{id_result['total']} = {id_result['accuracy']:.2%}")

            # evaluate OOD
            if ood_examples:
                ood_result = self.evaluate_task(task, ood_examples, use_few_shot, verbose=verbose)
                results[f"{task}_ood_accuracy"] = ood_result["accuracy"]
                results[f"{task}_ood_correct"] = ood_result["correct"]
                results[f"{task}_ood_total"] = ood_result["total"]
                print(f"  OOD: {ood_result['correct']}/{ood_result['total']} = {ood_result['accuracy']:.2%}")

        # compute aggregates
        id_accs = [v for k, v in results.items() if k.endswith("_id_accuracy")]
        ood_accs = [v for k, v in results.items() if k.endswith("_ood_accuracy")]
        
        if id_accs:
            results["avg_id_accuracy"] = sum(id_accs) / len(id_accs)
        if ood_accs:
            results["avg_ood_accuracy"] = sum(ood_accs) / len(ood_accs)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on algorithmic tasks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--task", type=str, choices=list(TASK_CONFIG.keys()), help="Specific task to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all tasks")
    parser.add_argument("--few-shot", action="store_true", help="Use few-shot prompting")
    parser.add_argument("--verbose", action="store_true", help="Print per-example outputs")
    parser.add_argument("--max-examples", type=int, default=500, help="Max examples per split")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--eval-data", type=str, default=str(DEFAULT_EVAL_PATH))
    args = parser.parse_args()
    
    if not args.all and not args.task:
        parser.print_help()
        print("\nError: Specify --all or --task <name>")
        return
    
    evaluator = AlgorithmicEvaluator(
        checkpoint_path=args.checkpoint,
        eval_data_path=args.eval_data,
    )
    
    tasks = [args.task] if args.task else None
    results = evaluator.evaluate(
        tasks=tasks,
        use_few_shot=args.few_shot,
        verbose=args.verbose,
        max_examples_per_split=args.max_examples,
    )
    
    evaluator.log_results(results, args.output)


if __name__ == "__main__":
    main()
