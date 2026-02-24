#!/usr/bin/env python3
"""
PE-Explorer Evaluation Suite

Run evaluation on trained models:
    # Single checkpoint
    python eval.py --checkpoint checkpoints/model.pt --tasks algorithmic,ppl

    # Multiple checkpoints (batch mode)
    python eval.py --checkpoint ckpt1.pt ckpt2.pt ckpt3.pt --tasks all

    # All checkpoints under a directory
    python eval.py --checkpoint-dir checkpoints/ --tasks ppl
"""

import argparse
import sys
from pathlib import Path


ALL_TASKS = ["algorithmic", "ppl"]

# checkpoint filenames considered final (others like step_*.pt are skipped)
_FINAL_NAMES = {"final.pt", "best.pt", "latest.pt"}


def _get_evaluators() -> dict:
    """Lazy import evaluator classes to avoid top-level torch/wandb dependency."""
    from src.evaluation.eval_algorithmic import AlgorithmicEvaluator
    from src.evaluation.eval_ppl import PPLEvaluator

    return {
        "algorithmic": AlgorithmicEvaluator,
        "ppl": PPLEvaluator,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="PE-Explorer Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model
  python eval.py --checkpoint checkpoints/model.pt --tasks algorithmic

  # Multiple models (batch)
  python eval.py --checkpoint ckpt1.pt ckpt2.pt --tasks ppl

  # All final checkpoints in a directory
  python eval.py --checkpoint-dir checkpoints/ --tasks all

  # Combine explicit + directory
  python eval.py --checkpoint extra.pt --checkpoint-dir checkpoints/ --tasks ppl
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        default=[],
        help="Path(s) to model checkpoint(s)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Auto-discover final/best checkpoints under this directory",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help=f"Comma-separated tasks: {', '.join(ALL_TASKS)}, or 'all'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save results (default: eval_results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: cuda, mps, or cpu (default: cuda)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate. Defaults: 1000 (PPL), config (Algo).",
    )
    return parser.parse_args()


def _discover_checkpoints(base_dir: Path) -> list[Path]:
    """Find final/best checkpoints under *base_dir*, skip intermediates."""
    results: list[Path] = []
    for pt_file in sorted(base_dir.rglob("*.pt")):
        name = pt_file.name
        if name in _FINAL_NAMES:
            results.append(pt_file)
        elif not name.startswith("step_") and not name.startswith("ready_for_decay"):
            results.append(pt_file)
    return results


def _run_name_from_checkpoint(checkpoint_path: Path) -> str:
    """Derive a human-readable run name from the checkpoint.

    Tries to extract the run_name from the checkpoint metadata.
    Falls back to the parent directory name.
    """
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        meta = ckpt.get("run_metadata", {})
        effective = meta.get("effective_config", {})
        name = effective.get("run", {}).get("run_name")
        if name:
            return name
    except Exception:
        pass
    return checkpoint_path.parent.name


def _evaluate_one(checkpoint_path: Path, tasks: list[str], device: str,
                  output_dir: str, num_samples: int | None,
                  use_wandb: bool) -> dict:
    """Run all requested tasks on a single checkpoint. Returns results dict."""
    evaluators = _get_evaluators()
    run_name = _run_name_from_checkpoint(checkpoint_path)

    # per-model output directory to keep results organized
    model_output_dir = str(Path(output_dir) / run_name)

    if use_wandb:
        import wandb
        wandb.init(
            project="pe-explorer",
            name=f"eval_{run_name}",
            job_type="eval",
        )

    print(f"\nEvaluating: {checkpoint_path}")
    print(f"  Run name: {run_name}")
    print(f"  Tasks: {', '.join(tasks)}")
    print(f"  Device: {device}")
    print(f"  Output: {model_output_dir}")

    all_results: dict = {}

    for task_name in tasks:
        print(f"\n{'=' * 50}")
        print(f"  {task_name.upper()} EVALUATION")
        print(f"{'=' * 50}")

        evaluator_cls = evaluators[task_name]

        try:
            evaluator = evaluator_cls(
                checkpoint_path=str(checkpoint_path),
                device=device,
                output_dir=model_output_dir,
            )

            run_kwargs = {}
            if num_samples is not None:
                run_kwargs["num_samples"] = num_samples

            results = evaluator.run(**run_kwargs)
            all_results[task_name] = results
        except FileNotFoundError as e:
            print(f"  Skipping {task_name}: {e}")
            all_results[task_name] = {"error": str(e)}
        except Exception as e:
            print(f"  Error in {task_name}: {e}")
            all_results[task_name] = {"error": str(e)}

    if use_wandb:
        import wandb
        wandb.finish()

    return all_results


def _print_summary(all_results: dict[str, dict]):
    """Print a formatted summary of results for one model."""
    print(f"\n{'=' * 50}")
    print("  SUMMARY")
    print(f"{'=' * 50}")

    for task_name, results in all_results.items():
        print(f"\n{task_name}:")
        if isinstance(results, dict):
            if "error" in results:
                print(f"  ERROR: {results['error']}")
                continue
            for key, value in results.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, dict):
                            # nested: e.g. algorithmic -> id -> task -> {accuracy, ...}
                            acc = v.get("accuracy")
                            if acc is not None:
                                print(f"  {key}/{k}: {acc:.2%}")
                            elif v.get("skipped"):
                                print(f"  {key}/{k}: SKIPPED ({v.get('skip_reason', '')})")
                        elif isinstance(v, float):
                            print(f"  {key}/{k}: {v:.4f}")
                        else:
                            print(f"  {key}/{k}: {v}")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


def _print_batch_summary(batch_results: dict[str, dict]):
    """Print a side-by-side comparison for batch evaluation."""
    if len(batch_results) <= 1:
        return

    print(f"\n{'='*60}")
    print("  BATCH COMPARISON")
    print(f"{'='*60}")

    # PPL comparison
    ppl_data = {}
    for name, results in batch_results.items():
        ppl = results.get("ppl", {})
        if isinstance(ppl, dict) and "ppl" in ppl:
            ppl_data[name] = ppl["ppl"]

    if ppl_data:
        print("\nPerplexity (lower is better):")
        for name, ppl in sorted(ppl_data.items(), key=lambda x: x[1]):
            print(f"  {ppl:>8.2f}  {name}")

    # Algorithmic comparison (ID average)
    algo_data = {}
    for name, results in batch_results.items():
        algo = results.get("algorithmic", {})
        if isinstance(algo, dict) and "id" in algo:
            id_results = algo["id"]
            accs = [
                v["accuracy"]
                for v in id_results.values()
                if isinstance(v, dict) and not v.get("skipped") and "accuracy" in v
            ]
            if accs:
                algo_data[name] = sum(accs) / len(accs)

    if algo_data:
        print("\nAlgorithmic ID avg accuracy (higher is better):")
        for name, acc in sorted(algo_data.items(), key=lambda x: -x[1]):
            print(f"  {acc:>7.1%}  {name}")

    print()


def main():
    args = parse_args()

    # Collect all checkpoint paths
    checkpoint_paths: list[Path] = []

    for p in args.checkpoint:
        path = Path(p)
        if path.exists():
            checkpoint_paths.append(path)
        else:
            print(f"Error: Checkpoint not found: {path}")
            sys.exit(1)

    if args.checkpoint_dir:
        discovered = _discover_checkpoints(Path(args.checkpoint_dir))
        if not discovered:
            print(f"No checkpoints found under {args.checkpoint_dir}/")
        else:
            print(f"Discovered {len(discovered)} checkpoint(s) under {args.checkpoint_dir}/")
            checkpoint_paths.extend(discovered)

    if not checkpoint_paths:
        print("Error: No checkpoints specified.")
        print("Use --checkpoint <path> or --checkpoint-dir <dir>")
        sys.exit(1)

    # De-duplicate preserving order
    checkpoint_paths = list(dict.fromkeys(checkpoint_paths))

    # Parse tasks
    if args.tasks.lower() == "all":
        tasks = ALL_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]
        unknown = set(tasks) - set(ALL_TASKS)
        if unknown:
            print(f"Error: Unknown task(s): {unknown}")
            print(f"Available: {', '.join(ALL_TASKS)}")
            sys.exit(1)

    print(f"Checkpoints: {len(checkpoint_paths)}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")

    # Evaluate each checkpoint
    batch_results: dict[str, dict] = {}

    for i, cp in enumerate(checkpoint_paths, 1):
        if len(checkpoint_paths) > 1:
            print(f"\n{'#'*60}")
            print(f"  Checkpoint [{i}/{len(checkpoint_paths)}]")
            print(f"{'#'*60}")

        run_name = _run_name_from_checkpoint(cp)
        results = _evaluate_one(
            checkpoint_path=cp,
            tasks=tasks,
            device=args.device,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            use_wandb=args.wandb,
        )
        batch_results[run_name] = results
        _print_summary(results)

    # Batch comparison
    if len(batch_results) > 1:
        _print_batch_summary(batch_results)

    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
