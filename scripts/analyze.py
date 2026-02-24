#!/usr/bin/env python3
"""
PE-Explorer Model Analysis & Comparison

Batch evaluate and compare models across PE types and model sizes.
Produces comparison tables for PPL and algorithmic task performance.

Usage:
    # Discover all checkpoints
    python scripts/analyze.py discover

    # Evaluate and compare specific checkpoints
    python scripts/analyze.py compare checkpoint1.pt checkpoint2.pt

    # Evaluate all checkpoints found under a directory
    python scripts/analyze.py compare --checkpoint-dir checkpoints/

    # Compare from previously saved results (no GPU needed)
    python scripts/analyze.py compare --results-dir eval_results/

    # Control which tasks to run
    python scripts/analyze.py compare ckpt1.pt ckpt2.pt --tasks ppl
    python scripts/analyze.py compare ckpt1.pt ckpt2.pt --tasks algorithmic
    python scripts/analyze.py compare ckpt1.pt ckpt2.pt --tasks ppl,algorithmic

    # Save consolidated results for later analysis
    python scripts/analyze.py compare ckpt1.pt ckpt2.pt --save results.json
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

SIZE_BY_DMODEL = {512: "tiny", 768: "small", 1024: "medium", 2048: "large"}
MODEL_SIZES = ["tiny", "small", "medium", "large"]
PE_TYPES = [
    "none", "sinusoidal", "sinonly",
    "binary", "binary_norm",
    "decimal", "decimal_norm",
    "rope",
]


@dataclass
class ModelInfo:
    """Metadata extracted from a single checkpoint."""
    checkpoint_path: str
    pe_type: str
    model_size: str
    d_model: int = 0
    n_layers: int = 0
    n_heads: int = 0
    params: int = 0
    tokens_trained: int = 0
    run_name: str = ""
    step: int = 0


def extract_model_info(checkpoint_path: Path) -> ModelInfo:
    """Load checkpoint header and extract identity without loading weights."""
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    metadata = ckpt.get("run_metadata", {})

    pe_type = config.get("pe_type", "unknown")
    d_model = config.get("d_model", 0)
    n_layers = config.get("n_layers", 0)
    n_heads = config.get("n_heads", 0)

    model_size = SIZE_BY_DMODEL.get(d_model, f"{d_model}d")

    effective_config = metadata.get("effective_config", {})
    run_info = effective_config.get("run", {})
    run_name = run_info.get("run_name", checkpoint_path.parent.name)

    # override model_size from run metadata if available
    if run_info.get("model_size"):
        model_size = run_info["model_size"]

    return ModelInfo(
        checkpoint_path=str(checkpoint_path),
        pe_type=pe_type,
        model_size=model_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        tokens_trained=ckpt.get("consumed_tokens", 0),
        run_name=run_name,
        step=ckpt.get("step", 0),
    )


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(base_dir: Path) -> list[ModelInfo]:
    """Recursively find usable checkpoints under *base_dir*.

    Prefers ``final.pt`` and ``best.pt``; skips intermediate step
    checkpoints and pre-decay branch checkpoints.
    """
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return []

    results: list[ModelInfo] = []
    for pt_file in sorted(base_dir.rglob("*.pt")):
        name = pt_file.name
        # skip intermediate / branch checkpoints
        if name.startswith("step_") or name.startswith("ready_for_decay"):
            continue
        try:
            info = extract_model_info(pt_file)
            results.append(info)
        except Exception as e:
            print(f"  Warning: skipping {pt_file}: {e}")

    return results


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def run_single_eval(checkpoint_path: str, tasks: list[str], device: str,
                    output_dir: str, num_samples: int | None = None) -> dict:
    """Run evaluation tasks on one checkpoint and return results dict."""
    # deferred imports so the script stays importable without torch/data deps
    from src.evaluation.eval_algorithmic import AlgorithmicEvaluator
    from src.evaluation.eval_ppl import PPLEvaluator

    results: dict = {}

    for task in tasks:
        try:
            if task == "ppl":
                evaluator = PPLEvaluator(
                    checkpoint_path=checkpoint_path,
                    device=device,
                    output_dir=output_dir,
                )
                r = evaluator.run(num_samples=num_samples)
                results["ppl"] = r
            elif task == "algorithmic":
                evaluator = AlgorithmicEvaluator(
                    checkpoint_path=checkpoint_path,
                    device=device,
                    output_dir=output_dir,
                )
                r = evaluator.run(num_samples=num_samples)
                results["algorithmic"] = r
        except FileNotFoundError as e:
            print(f"  Skipping {task}: {e}")
            results[task] = {"error": str(e)}
        except Exception as e:
            print(f"  Error running {task}: {e}")
            results[task] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Result aggregation from saved JSON files
# ---------------------------------------------------------------------------

def load_results_from_dir(results_dir: Path) -> dict:
    """Load all ``*.json`` result files under *results_dir* and aggregate.

    This supports the directory structure created by ``eval.py`` where
    each model's results are saved in a subdirectory:

        eval_results/<run_name>/ppl_fineweb_val.json
        eval_results/<run_name>/algorithmic_id.json
        eval_results/<run_name>/algorithmic_ood.json
    """
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return {}

    aggregated: dict = {}

    for json_file in sorted(results_dir.rglob("*.json")):
        rel = json_file.relative_to(results_dir)
        parts = list(rel.parts)

        # skip meta files
        if json_file.name == "meta.json":
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Use the parent directory name as the model identifier
        if len(parts) >= 2:
            model_key = parts[0]
        else:
            model_key = json_file.stem

        if model_key not in aggregated:
            aggregated[model_key] = {}

        task_name = json_file.stem  # e.g. "ppl_fineweb_val", "algorithmic_id"
        aggregated[model_key][task_name] = data

    return aggregated


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _pad(text: str, width: int, align: str = "left") -> str:
    if align == "right":
        return text.rjust(width)
    if align == "center":
        return text.center(width)
    return text.ljust(width)


def format_table(headers: list[str], rows: list[list[str]],
                 title: str = "", col_align: list[str] | None = None) -> str:
    """Render a simple ASCII table.

    Args:
        headers: Column header strings.
        rows: List of rows, each a list of cell strings.
        title: Optional title printed above the table.
        col_align: Per-column alignment ("left", "right", "center").
    """
    if not headers:
        return ""

    n_cols = len(headers)
    col_align = col_align or ["left"] * n_cols

    # compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    def _row_str(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            w = widths[i] if i < len(widths) else len(cell)
            a = col_align[i] if i < len(col_align) else "left"
            parts.append(_pad(cell, w, a))
        return "  ".join(parts)

    lines: list[str] = []
    if title:
        lines.append("")
        lines.append(title)
        lines.append("=" * len(title))
    separator = "  ".join("-" * w for w in widths)
    lines.append(_row_str(headers))
    lines.append(separator)
    for row in rows:
        lines.append(_row_str(row))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison reports
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Holds all evaluation results for one model."""
    info: ModelInfo
    ppl: float | None = None
    ppl_tokens: int = 0
    algo_id: dict = field(default_factory=dict)    # task -> accuracy
    algo_ood: dict = field(default_factory=dict)   # task -> accuracy
    errors: list[str] = field(default_factory=list)


def build_model_results(models: list[ModelInfo], raw_results: dict[str, dict]) -> list[ModelResult]:
    """Merge ModelInfo list with raw evaluation dicts into ModelResult list."""
    out: list[ModelResult] = []

    for info in models:
        key = info.run_name or Path(info.checkpoint_path).parent.name
        raw = raw_results.get(key, {})
        mr = ModelResult(info=info)

        # PPL
        ppl_data = raw.get("ppl", {})
        if isinstance(ppl_data, dict) and "ppl" in ppl_data:
            mr.ppl = ppl_data["ppl"]
            mr.ppl_tokens = ppl_data.get("tokens", 0)
        elif isinstance(ppl_data, dict) and "error" in ppl_data:
            mr.errors.append(f"ppl: {ppl_data['error']}")

        # Algorithmic
        algo_data = raw.get("algorithmic", {})
        if isinstance(algo_data, dict) and "error" not in algo_data:
            for mode in ("id", "ood"):
                mode_data = algo_data.get(mode, {})
                for task, task_data in mode_data.items():
                    if isinstance(task_data, dict) and not task_data.get("skipped"):
                        acc = task_data.get("accuracy", 0.0)
                        if mode == "id":
                            mr.algo_id[task] = acc
                        else:
                            mr.algo_ood[task] = acc
        elif isinstance(algo_data, dict) and "error" in algo_data:
            mr.errors.append(f"algorithmic: {algo_data['error']}")

        out.append(mr)

    return out


def _fmt_ppl(val: float | None) -> str:
    if val is None:
        return "-"
    return f"{val:.2f}"


def _fmt_acc(val: float | None) -> str:
    if val is None:
        return "-"
    return f"{val:.1%}"


def _fmt_tokens(val: int) -> str:
    if val == 0:
        return "-"
    if val >= 1e9:
        return f"{val / 1e9:.1f}B"
    if val >= 1e6:
        return f"{val / 1e6:.0f}M"
    return f"{val:,}"


def generate_report(model_results: list[ModelResult]) -> str:
    """Generate a full comparison report as a string."""
    lines: list[str] = []

    if not model_results:
        return "No models to compare.\n"

    # ---- Overview table ----
    lines.append("")
    lines.append("PE-Explorer Model Comparison Report")
    lines.append("=" * 50)

    headers = ["Model", "PE Type", "Size", "Params", "Tokens", "Step"]
    rows = []
    for mr in sorted(model_results, key=lambda m: (m.info.model_size, m.info.pe_type)):
        rows.append([
            mr.info.run_name or Path(mr.info.checkpoint_path).name,
            mr.info.pe_type,
            mr.info.model_size,
            f"{mr.info.d_model}d/{mr.info.n_layers}L",
            _fmt_tokens(mr.info.tokens_trained),
            str(mr.info.step),
        ])
    lines.append(format_table(headers, rows, title="Models Evaluated"))

    # Group by model_size for cross-PE comparison
    by_size: dict[str, list[ModelResult]] = defaultdict(list)
    for mr in model_results:
        by_size[mr.info.model_size].append(mr)

    # ---- PPL comparison ----
    has_ppl = any(mr.ppl is not None for mr in model_results)
    if has_ppl:
        # Cross-size PPL table (rows = PE types, cols = model sizes)
        sizes_present = [s for s in MODEL_SIZES if s in by_size]
        pe_types_present = sorted({mr.info.pe_type for mr in model_results if mr.ppl is not None})

        if sizes_present and pe_types_present:
            headers = ["PE Type"] + [s.capitalize() for s in sizes_present]
            col_align = ["left"] + ["right"] * len(sizes_present)
            rows = []

            # collect values for ranking
            ppl_values: dict[str, dict[str, float]] = {}  # pe -> size -> ppl
            for mr in model_results:
                if mr.ppl is not None:
                    ppl_values.setdefault(mr.info.pe_type, {})[mr.info.model_size] = mr.ppl

            for pe in pe_types_present:
                row = [pe]
                for size in sizes_present:
                    val = ppl_values.get(pe, {}).get(size)
                    row.append(_fmt_ppl(val))
                rows.append(row)

            lines.append(format_table(headers, rows,
                                       title="Perplexity (lower is better)",
                                       col_align=col_align))

            # Best PE per size
            lines.append("Best PE by model size:")
            for size in sizes_present:
                best_pe = None
                best_ppl = float("inf")
                for pe, size_map in ppl_values.items():
                    if size in size_map and size_map[size] < best_ppl:
                        best_ppl = size_map[size]
                        best_pe = pe
                if best_pe:
                    lines.append(f"  {size:>8s}: {best_pe} (PPL={best_ppl:.2f})")
            lines.append("")

    # ---- Algorithmic task comparison ----
    algo_tasks_id = set()
    algo_tasks_ood = set()
    for mr in model_results:
        algo_tasks_id.update(mr.algo_id.keys())
        algo_tasks_ood.update(mr.algo_ood.keys())

    for mode, task_set, accessor in [
        ("In-Distribution (ID)", algo_tasks_id, "algo_id"),
        ("Out-of-Distribution (OOD)", algo_tasks_ood, "algo_ood"),
    ]:
        if not task_set:
            continue

        tasks = sorted(task_set)

        for size, mrs in sorted(by_size.items()):
            # only show sizes that have algorithmic results
            mrs_with_algo = [mr for mr in mrs if getattr(mr, accessor)]
            if not mrs_with_algo:
                continue

            short_tasks = []
            for t in tasks:
                short = t.replace("copy_distance", "copy_d").replace("simple_copy", "s_copy")
                short = short.replace("no_carry_add", "add")
                short_tasks.append(short)

            headers = ["PE Type"] + short_tasks + ["Avg"]
            col_align = ["left"] + ["right"] * (len(tasks) + 1)
            rows = []

            for mr in sorted(mrs_with_algo, key=lambda m: m.info.pe_type):
                task_accs = getattr(mr, accessor)
                row = [mr.info.pe_type]
                vals = []
                for t in tasks:
                    acc = task_accs.get(t)
                    row.append(_fmt_acc(acc))
                    if acc is not None:
                        vals.append(acc)
                avg = sum(vals) / len(vals) if vals else None
                row.append(_fmt_acc(avg))
                rows.append(row)

            title = f"Algorithmic Accuracy - {mode} - {size.capitalize()} Models"
            lines.append(format_table(headers, rows, title=title, col_align=col_align))

    # ---- Per-size ranking ----
    for size, mrs in sorted(by_size.items()):
        if len(mrs) < 2:
            continue

        lines.append(f"Rankings - {size.capitalize()}")
        lines.append("-" * 30)

        # PPL ranking
        ppl_ranked = [(mr.info.pe_type, mr.ppl) for mr in mrs if mr.ppl is not None]
        if ppl_ranked:
            ppl_ranked.sort(key=lambda x: x[1])
            lines.append("  PPL (best to worst):")
            for i, (pe, ppl) in enumerate(ppl_ranked, 1):
                delta = ""
                if i > 1:
                    delta = f"  (+{ppl - ppl_ranked[0][1]:.2f})"
                lines.append(f"    {i}. {pe}: {ppl:.2f}{delta}")

        # Algorithmic ID avg ranking
        id_ranked = []
        for mr in mrs:
            if mr.algo_id:
                avg = sum(mr.algo_id.values()) / len(mr.algo_id)
                id_ranked.append((mr.info.pe_type, avg))
        if id_ranked:
            id_ranked.sort(key=lambda x: -x[1])
            lines.append("  Algorithmic ID avg (best to worst):")
            for i, (pe, avg) in enumerate(id_ranked, 1):
                lines.append(f"    {i}. {pe}: {avg:.1%}")

        # Algorithmic OOD avg ranking
        ood_ranked = []
        for mr in mrs:
            if mr.algo_ood:
                avg = sum(mr.algo_ood.values()) / len(mr.algo_ood)
                ood_ranked.append((mr.info.pe_type, avg))
        if ood_ranked:
            ood_ranked.sort(key=lambda x: -x[1])
            lines.append("  Algorithmic OOD avg (best to worst):")
            for i, (pe, avg) in enumerate(ood_ranked, 1):
                lines.append(f"    {i}. {pe}: {avg:.1%}")

        lines.append("")

    # ---- Errors / warnings ----
    errors = [(mr.info.run_name, err) for mr in model_results for err in mr.errors]
    if errors:
        lines.append("Warnings / Errors")
        lines.append("-" * 30)
        for name, err in errors:
            lines.append(f"  {name}: {err}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_discover(args):
    """List all discoverable checkpoints."""
    base = Path(args.checkpoint_dir)
    models = discover_checkpoints(base)

    if not models:
        print(f"No checkpoints found under {base}/")
        print("Expected structure: checkpoints/<run_name>/final.pt")
        return

    print(f"\nFound {len(models)} checkpoint(s) under {base}/\n")

    headers = ["Run Name", "PE Type", "Size", "Arch", "Tokens", "Step", "Path"]
    rows = []
    for m in sorted(models, key=lambda x: (x.model_size, x.pe_type)):
        rows.append([
            m.run_name,
            m.pe_type,
            m.model_size,
            f"{m.d_model}d/{m.n_layers}L/{m.n_heads}H",
            _fmt_tokens(m.tokens_trained),
            str(m.step),
            str(Path(m.checkpoint_path).relative_to(base)),
        ])

    print(format_table(headers, rows))

    # Summary
    pe_counts: dict[str, int] = defaultdict(int)
    size_counts: dict[str, int] = defaultdict(int)
    for m in models:
        pe_counts[m.pe_type] += 1
        size_counts[m.model_size] += 1

    print("Coverage:")
    print(f"  PE types:    {', '.join(f'{pe}({n})' for pe, n in sorted(pe_counts.items()))}")
    print(f"  Model sizes: {', '.join(f'{s}({n})' for s, n in sorted(size_counts.items()))}")

    # Show which PE x size combinations exist
    print("\n  PE x Size matrix:")
    sizes_present = sorted(size_counts.keys(), key=lambda s: MODEL_SIZES.index(s) if s in MODEL_SIZES else 99)
    print(f"  {'PE Type':<16s}  " + "  ".join(f"{s:>8s}" for s in sizes_present))
    print(f"  {'-'*16}  " + "  ".join("-" * 8 for _ in sizes_present))
    for pe in sorted(pe_counts.keys()):
        cells = []
        for s in sizes_present:
            has = any(m.pe_type == pe and m.model_size == s for m in models)
            cells.append("Y" if has else "-")
        print(f"  {pe:<16s}  " + "  ".join(f"{c:>8s}" for c in cells))
    print()


def cmd_compare(args):
    """Evaluate (if needed) and compare multiple models."""
    # Collect checkpoint paths
    checkpoint_paths: list[Path] = []

    if args.checkpoints:
        for p in args.checkpoints:
            path = Path(p)
            if path.is_file():
                checkpoint_paths.append(path)
            else:
                print(f"Warning: {p} is not a file, skipping")

    if args.checkpoint_dir:
        base = Path(args.checkpoint_dir)
        discovered = discover_checkpoints(base)
        for m in discovered:
            checkpoint_paths.append(Path(m.checkpoint_path))

    # Parse tasks
    if args.tasks == "all":
        tasks = ["ppl", "algorithmic"]
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    # If only loading from results dir (no checkpoints)
    if args.results_only:
        print("Loading results from saved files (--results-only mode)...")
        aggregated = load_results_from_dir(Path(args.output_dir))
        if not aggregated:
            print(f"No results found in {args.output_dir}/")
            sys.exit(1)

        # Build ModelInfo stubs from result directory names
        models: list[ModelInfo] = []
        for key in aggregated:
            # Try to parse run name: {size}_{pe_type}_...
            parts = key.split("_")
            model_size = parts[0] if parts else "unknown"
            pe_type = parts[1] if len(parts) > 1 else "unknown"
            models.append(ModelInfo(
                checkpoint_path="",
                pe_type=pe_type,
                model_size=model_size,
                run_name=key,
            ))

        model_results = build_model_results(models, aggregated)
        report = generate_report(model_results)
        print(report)

        if args.save:
            _save_consolidated(model_results, Path(args.save))
        return

    if not checkpoint_paths:
        print("No checkpoints specified. Use positional args or --checkpoint-dir.")
        print("Run 'python scripts/analyze.py discover' to find available checkpoints.")
        sys.exit(1)

    # De-duplicate
    checkpoint_paths = list(dict.fromkeys(checkpoint_paths))

    print(f"\nAnalyzing {len(checkpoint_paths)} checkpoint(s)")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Device: {args.device}")
    print()

    # Extract model info
    models: list[ModelInfo] = []
    for cp in checkpoint_paths:
        try:
            info = extract_model_info(cp)
            models.append(info)
            print(f"  Found: {info.run_name} ({info.pe_type}, {info.model_size}, "
                  f"{_fmt_tokens(info.tokens_trained)} tokens)")
        except Exception as e:
            print(f"  Error loading {cp}: {e}")

    if not models:
        print("No valid checkpoints found.")
        sys.exit(1)

    # Run evaluations
    all_raw: dict[str, dict] = {}

    for i, info in enumerate(models, 1):
        key = info.run_name or Path(info.checkpoint_path).parent.name
        # Per-model output directory
        model_output_dir = str(Path(args.output_dir) / key)

        print(f"\n{'='*60}")
        print(f"  [{i}/{len(models)}] {key}")
        print(f"  PE: {info.pe_type}, Size: {info.model_size}")
        print(f"{'='*60}")

        results = run_single_eval(
            checkpoint_path=info.checkpoint_path,
            tasks=tasks,
            device=args.device,
            output_dir=model_output_dir,
            num_samples=args.num_samples,
        )
        all_raw[key] = results

    # Build comparison
    model_results = build_model_results(models, all_raw)
    report = generate_report(model_results)
    print(report)

    # Save consolidated results
    if args.save:
        _save_consolidated(model_results, Path(args.save))


def _save_consolidated(model_results: list[ModelResult], path: Path):
    """Save all results to a single JSON file for later analysis."""
    data = []
    for mr in model_results:
        entry = {
            "model": asdict(mr.info),
            "ppl": mr.ppl,
            "ppl_tokens": mr.ppl_tokens,
            "algo_id": mr.algo_id,
            "algo_ood": mr.algo_ood,
            "errors": mr.errors,
        }
        data.append(entry)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Consolidated results saved to {path}")


def cmd_report(args):
    """Load a consolidated results JSON and print the report."""
    path = Path(args.results_file)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    model_results: list[ModelResult] = []
    for entry in data:
        info = ModelInfo(**entry["model"])
        mr = ModelResult(
            info=info,
            ppl=entry.get("ppl"),
            ppl_tokens=entry.get("ppl_tokens", 0),
            algo_id=entry.get("algo_id", {}),
            algo_ood=entry.get("algo_ood", {}),
            errors=entry.get("errors", []),
        )
        model_results.append(mr)

    report = generate_report(model_results)
    print(report)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="analyze",
        description="PE-Explorer Model Analysis & Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- discover ---
    p_discover = subparsers.add_parser(
        "discover",
        help="Find and list all checkpoints",
    )
    p_discover.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Base directory to search (default: checkpoints)",
    )

    # --- compare ---
    p_compare = subparsers.add_parser(
        "compare",
        help="Evaluate and compare multiple models",
    )
    p_compare.add_argument(
        "checkpoints", nargs="*", default=[],
        help="Checkpoint file paths",
    )
    p_compare.add_argument(
        "--checkpoint-dir", default=None,
        help="Auto-discover checkpoints under this directory",
    )
    p_compare.add_argument(
        "--tasks", default="all",
        help="Comma-separated tasks: ppl, algorithmic, or 'all' (default: all)",
    )
    p_compare.add_argument(
        "--device", default="cuda",
        help="Device: cuda, mps, cpu (default: cuda)",
    )
    p_compare.add_argument(
        "--output-dir", default="eval_results",
        help="Directory for per-model results (default: eval_results)",
    )
    p_compare.add_argument(
        "--num-samples", type=int, default=None,
        help="Override number of eval samples",
    )
    p_compare.add_argument(
        "--save", default=None,
        help="Save consolidated results to this JSON file",
    )
    p_compare.add_argument(
        "--results-only", action="store_true",
        help="Skip evaluation; load results from --output-dir only",
    )

    # --- report ---
    p_report = subparsers.add_parser(
        "report",
        help="Print report from a saved consolidated results JSON",
    )
    p_report.add_argument(
        "results_file",
        help="Path to consolidated results JSON (from --save)",
    )

    args = parser.parse_args()

    if args.command == "discover":
        cmd_discover(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
