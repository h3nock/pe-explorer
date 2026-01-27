"""Perplexity evaluator using FineWeb validation split."""

import math
import torch
from tqdm import tqdm

from src.evaluation.base import BaseEvaluator
from src.data.dataset import MemmapDataset


class PPLEvaluator(BaseEvaluator):
    """Evaluator for perplexity on FineWeb validation data."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        output_dir: str = "eval_results",
        batch_size: int = 4,
    ):
        super().__init__(checkpoint_path, device, output_dir)
        self.batch_size = batch_size

    def run(self, num_samples: int | None = None) -> dict:
        num_samples = num_samples or 1000  # PPL default differs from algorithmic
        """Run PPL evaluation on FineWeb validation split.

        Args:
            num_samples: Maximum number of samples to evaluate.

        Returns:
            Dictionary with 'ppl' and 'tokens' keys, or empty dict on failure.

        Raises:
            FileNotFoundError: If validation dataset path doesn't exist.
        """
        print("\nEvaluating PPL on FineWeb validation...")

        ds = self._load_dataset()
        n_samples = min(len(ds), num_samples) if num_samples else len(ds)

        total_nll = 0.0
        total_tokens = 0
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

        pbar = tqdm(range(0, n_samples, self.batch_size), desc="PPL eval", leave=False)

        for i in pbar:
            batch_end = min(i + self.batch_size, n_samples)
            batch_x, batch_y = self._collate_batch(ds, range(i, batch_end))

            if batch_x is None:
                continue

            x = batch_x.to(self.device)
            y = batch_y.to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            total_nll += loss.item()
            total_tokens += y.numel()

            if total_tokens > 0:
                pbar.set_postfix({"ppl": f"{math.exp(total_nll / total_tokens):.2f}"})

        if total_tokens == 0:
            print("Warning: No tokens evaluated")
            return {}

        ppl = math.exp(total_nll / total_tokens)
        print(f"  PPL = {ppl:.4f} ({total_tokens:,} tokens)")

        results = {"ppl": ppl, "tokens": total_tokens}
        self.log_results("ppl_fineweb_val", results)

        return results

    def _load_dataset(self) -> MemmapDataset:
        """Load FineWeb validation dataset.

        Returns:
            MemmapDataset instance.

        Raises:
            FileNotFoundError: If validation data path doesn't exist.
        """
        path = self.eval_config.fineweb_val_path

        if not path.exists():
            raise FileNotFoundError(f"Validation data not found at {path}")

        return MemmapDataset(path, seq_len=self.config.max_seq_len)

    def _collate_batch(self, ds, indices) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Collate a batch of samples."""
        batch_x = []
        batch_y = []

        for idx in indices:
            x, y = ds[idx]
            batch_x.append(x)
            batch_y.append(y)

        if not batch_x:
            return None, None

        return torch.stack(batch_x), torch.stack(batch_y)
