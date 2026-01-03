#!/usr/bin/env python3
"""
Evaluation CLI

Unified entry point for running all evaluations on a trained model checkpoint.

Usage:
    # Run all evaluations
    python eval.py --checkpoint checkpoints/tiny_sinusoidal/final.pt
    
    # Run specific tier
    python eval.py --checkpoint <path> --tier 2
    
    # Run specific task
    python eval.py --checkpoint <path> --task algorithmic
    python eval.py --checkpoint <path> --task ppl --dataset pg19 --context 4096
    python eval.py --checkpoint <path> --task passkey --context 4096
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.evaluation.eval_algorithmic import AlgorithmicEvaluator
from src.evaluation.eval_ppl import PPLEvaluator
from src.evaluation.eval_passkey import PasskeyEvaluator
from src.evaluation.eval_niah import NIAHEvaluator


TIER_TASKS = {
    1: ["ppl"],
    2: ["algorithmic"],
    3: ["passkey", "niah"],
}


def run_evaluation(args) -> dict:
    """Run specified evaluations and return combined results."""
    results = {
        "checkpoint": str(args.checkpoint),
        "timestamp": datetime.now().isoformat(),
    }
    
    tasks_to_run = []
    
    if args.tier:
        tasks_to_run = TIER_TASKS.get(args.tier, [])
    elif args.task:
        tasks_to_run = [args.task]
    else:
        # Run all
        for tier_tasks in TIER_TASKS.values():
            tasks_to_run.extend(tier_tasks)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION PIPELINE")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tasks: {tasks_to_run}")
    print(f"{'='*60}\n")
    
    # Run evaluations
    if "ppl" in tasks_to_run:
        print("\n" + "="*40)
        print("TIER 1: Perplexity Evaluation")
        print("="*40)
        evaluator = PPLEvaluator(checkpoint_path=args.checkpoint, config_path=args.config)
        ppl_results = evaluator.evaluate(
            dataset=args.dataset or "wikitext103",
            context_lengths=args.context,
            max_tokens=args.max_tokens or 1_000_000,
        )
        results.update({k: v for k, v in ppl_results.items() if k.startswith("ppl_")})
        for key in ("dataset", "train_max_seq_len"):
            if key in ppl_results:
                results[f"ppl_{key}"] = ppl_results[key]
    
    if "algorithmic" in tasks_to_run:
        print("\n" + "="*40)
        print("TIER 2: Algorithmic Evaluation")
        print("="*40)
        evaluator = AlgorithmicEvaluator(checkpoint_path=args.checkpoint, config_path=args.config)
        
        # Zero-shot
        alg_results = evaluator.evaluate(use_few_shot=False, verbose=args.verbose)
        results.update({f"zeroshot_{k}": v for k, v in alg_results.items() 
                       if "_accuracy" in k or "_correct" in k})
        
        # Few-shot
        if args.few_shot:
            alg_results_fs = evaluator.evaluate(use_few_shot=True, verbose=args.verbose)
            results.update({f"fewshot_{k}": v for k, v in alg_results_fs.items() 
                           if "_accuracy" in k or "_correct" in k})
    
    if "passkey" in tasks_to_run:
        print("\n" + "="*40)
        print("TIER 3: Passkey Retrieval")
        print("="*40)
        evaluator = PasskeyEvaluator(checkpoint_path=args.checkpoint, config_path=args.config)
        pk_results = evaluator.evaluate(
            context_lengths=args.context,
            samples_per_context=args.samples or 100,
        )
        results.update({k: v for k, v in pk_results.items() if k.startswith("passkey_")})
    
    if "niah" in tasks_to_run:
        print("\n" + "="*40)
        print("TIER 3: Needle-in-a-Haystack Sweep")
        print("="*40)
        evaluator = NIAHEvaluator(checkpoint_path=args.checkpoint, config_path=args.config)
        niah_results = evaluator.evaluate(
            context_lengths=args.context,
            samples_per_cell=args.samples or 20,
        )
        results.update({k: v for k, v in niah_results.items() if k.startswith("niah_")})
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation CLI for PE Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python eval.py --checkpoint ckpt.pt --tier 2        # Algorithmic only
    python eval.py --checkpoint ckpt.pt --task ppl     # PPL only
    python eval.py --checkpoint ckpt.pt                 # All evaluations
        """,
    )
    
    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config YAML if not embedded in checkpoint")
    
    # Task selection
    parser.add_argument("--tier", type=int, choices=[1, 2, 3],
                       help="Run all tasks in specified tier")
    parser.add_argument("--task", type=str,
                       choices=["ppl", "algorithmic", "passkey", "niah"],
                       help="Run specific task")
    
    # Task-specific options
    parser.add_argument("--dataset", type=str, choices=["wikitext103", "pg19"],
                       help="Dataset for PPL evaluation")
    parser.add_argument("--context", type=int, nargs="+",
                       help="Context lengths to test")
    parser.add_argument("--few-shot", action="store_true",
                       help="Include few-shot evaluation for algorithmic")
    parser.add_argument("--verbose", action="store_true",
                       help="Print per-example outputs for algorithmic eval")
    parser.add_argument("--max-tokens", type=int,
                       help="Max tokens for PPL evaluation")
    parser.add_argument("--samples", type=int,
                       help="Samples per context for passkey")
    
    # Output
    parser.add_argument("--output", type=str, 
                       help="Output JSON path (default: results/<checkpoint_name>.json)")
    
    args = parser.parse_args()
    
    # Run evaluations
    results = run_evaluation(args)
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for key, value in sorted(results.items()):
        if key in ["checkpoint", "timestamp"]:
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        ckpt_name = Path(args.checkpoint).stem
        output_path = Path("results") / f"eval_{ckpt_name}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
