#!/usr/bin/env python3
"""
PE-Explorer Evaluation Suite

Run evaluation on trained models:
    python eval.py --checkpoint checkpoints/model.pt --tasks algorithmic,ppl
    python eval.py --checkpoint checkpoints/model.pt --tasks all
"""

import argparse
import sys
from pathlib import Path

import wandb

from src.evaluation.eval_algorithmic import AlgorithmicEvaluator
from src.evaluation.eval_ppl import PPLEvaluator


# task registry: maps task names to evaluator classes
EVALUATORS = {
    "algorithmic": AlgorithmicEvaluator,
    "ppl": PPLEvaluator,
}

ALL_TASKS = list(EVALUATORS.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="PE-Explorer Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py --checkpoint checkpoints/model.pt --tasks algorithmic
  python eval.py --checkpoint checkpoints/model.pt --tasks ppl
  python eval.py --checkpoint checkpoints/model.pt --tasks all --wandb
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
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


def main():
    args = parse_args()

    # validate checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # parse tasks
    if args.tasks.lower() == "all":
        tasks = ALL_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]
        unknown = set(tasks) - set(EVALUATORS.keys())
        if unknown:
            print(f"Error: Unknown task(s): {unknown}")
            print(f"Available: {', '.join(EVALUATORS.keys())}")
            sys.exit(1)

    # initialize WandB if requested
    if args.wandb:
        run_name = f"eval_{checkpoint_path.stem}"
        wandb.init(project="pe-explorer", name=run_name, job_type="eval")

    print(f"Evaluating: {checkpoint_path}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print()

    # run evaluations
    all_results = {}

    for task_name in tasks:
        print(f"\n{'=' * 50}")
        print(f"  {task_name.upper()} EVALUATION")
        print(f"{'=' * 50}")

        evaluator_cls = EVALUATORS[task_name]
        evaluator = evaluator_cls(
            checkpoint_path=str(checkpoint_path),
            device=args.device,
            output_dir=args.output_dir,
        )

        run_kwargs = {}
        if args.num_samples is not None:
            run_kwargs["num_samples"] = args.num_samples

        results = evaluator.run(**run_kwargs)
        all_results[task_name] = results

    # summary
    print(f"\n{'=' * 50}")
    print("  SUMMARY")
    print(f"{'=' * 50}")
    for task_name, results in all_results.items():
        print(f"\n{task_name}:")
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, float):
                            print(f"  {key}/{k}: {v:.4f}")
                        else:
                            print(f"  {key}/{k}: {v}")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    if args.wandb:
        wandb.finish()

    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
