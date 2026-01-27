import torch
from tqdm import tqdm

from src.evaluation.base import BaseEvaluator
from src.data.prepare_algorithmic import AlgorithmicGenerator
from src.data.tokenization import BOS_TOKEN
from src.utils import set_seed


class AlgorithmicEvaluator(BaseEvaluator):
    """Evaluator for algorithmic/synthetic tasks."""

    def run(self, num_samples: int | None = None, seed: int | None = None) -> dict:
        """Run algorithmic evaluation on all configured tasks."""
        num_samples = self.eval_config.num_samples if num_samples is None else num_samples
        seed = self.eval_config.seed if seed is None else seed
        set_seed(seed)

        # init generator with config
        generator = AlgorithmicGenerator(self.eval_config)

        tasks = list(self.eval_config.tasks.keys())
        if not tasks:
            print("Warning: No tasks configured")
            return {}

        results = {}
        modes = ["id", "ood"]

        for mode in modes:
            print(f"\n--- Evaluating Mode: {mode.upper()} ---")
            mode_results = {}

            for task in tasks:
                task_result = self._evaluate_task(
                    task, mode, generator, num_samples
                )
                mode_results[task] = task_result

                if task_result.get("skipped"):
                    print(f"  {task}: SKIPPED ({task_result['skip_reason']})")
                else:
                    print(f"  {task}: {task_result['accuracy']:.2%}")

            results[mode] = mode_results
            self.log_results(f"algorithmic_{mode}", mode_results)

        return results

    def _evaluate_task(
        self,
        task: str,
        mode: str,
        generator: AlgorithmicGenerator,
        num_samples: int,
    ) -> dict:
        """Evaluate a single task.

        Returns:
            Dictionary with keys:
            - 'accuracy': float (0.0-1.0) if evaluated
            - 'total': int, number of samples evaluated
            - 'correct': int, number of correct samples
            - 'skipped': bool, True if task was skipped
            - 'skip_reason': str, reason for skipping (if skipped)
        """
        samples = []
        skip_reason = None

        for _ in range(num_samples):
            try:
                sample = generator.generate_one(task, mode=mode)
                samples.append(sample)
            except ValueError as e:
                # Expected for filler-dependent tasks without filler pool
                if "FillerPool required" in str(e) or "filler" in str(e).lower():
                    skip_reason = "no filler pool"
                    break
                raise

        if not samples:
            return {
                "skipped": True,
                "skip_reason": skip_reason or "no samples",
                "accuracy": 0.0,
                "total": 0,
                "correct": 0,
            }

        correct = 0
        total = 0
        failures_shown = 0

        for sample in tqdm(samples, desc=f"Eval {task}", leave=False):
            match, debug_info = self._evaluate_sample(sample)

            if match:
                correct += 1
            elif failures_shown < 3:
                print(f"\n[FAIL] {task}")
                print(f"  Prompt: {debug_info['prompt'][:100]}...")
                print(f"  Target: {debug_info['target']}")
                print(f"  Got:    {debug_info['generated']}")
                failures_shown += 1

            total += 1

        return {
            "skipped": False,
            "accuracy": correct / total if total > 0 else 0.0,
            "total": total,
            "correct": correct,
        }

    def _evaluate_sample(self, sample: dict) -> tuple[bool, dict]:
        """Evaluate a single sample using token-based matching."""
        prompt_text = sample["prompt"]
        target_text = sample["target"]

        # tokenize prompt
        input_ids = self.tokenizer.encode(prompt_text, prepend=BOS_TOKEN)
        input_tensor = torch.tensor(
            input_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        # tokenize target for comparison (convert to list for matching)
        target_ids = self.tokenizer.encode(target_text, prepend=None).tolist()

        # generate
        max_new = max(len(target_ids) + 10, 50)  # Enough for target + buffer

        with torch.no_grad():
            output_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_new,
                temperature=0.0,  # Greedy decoding
            )

        # extract generated tokens (after prompt)
        generated_ids = output_ids[0, input_tensor.shape[1]:].cpu().tolist()
        generated_text = self.tokenizer.enc.decode(generated_ids)

        # token-based matching: check if generated starts with target tokens
        match = self._check_token_match(generated_ids, target_ids)

        debug_info = {
            "prompt": prompt_text,
            "target": target_text,
            "generated": generated_text.strip(),
        }

        return match, debug_info

    def _check_token_match(
        self, generated_ids: list[int], target_ids: list[int]
    ) -> bool:
        """Check if generated output starts with exact target tokens.

        This is more robust than string matching because:
        - No issues with whitespace normalization
        - No ambiguity about token boundaries
        - Handles special tokens correctly
        """
        if len(generated_ids) < len(target_ids):
            return False

        # exact match on first N tokens
        return generated_ids[: len(target_ids)] == target_ids
