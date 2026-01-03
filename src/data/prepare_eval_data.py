"""
Evaluation Data Downloader

Downloads and prepares evaluation datasets:
- WikiText-103 (for perplexity)
- PG-19 (for long-context perplexity)
- LAMBADA (for last-word prediction)

Usage:
    python -m src.data.prepare_eval_data download --all
    python -m src.data.prepare_eval_data download --dataset wikitext103
    python -m src.data.prepare_eval_data info
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from functools import partial

from datasets import load_dataset
from tqdm import tqdm


EVAL_DATA_DIR = Path("data/eval")

DATASETS = {
    "wikitext103": {
        "hf_path": "wikitext",
        "hf_name": "wikitext-103-raw-v1",
        "splits": ["test", "validation"],
        "text_column": "text",
    },
    "pg19": {
        "hf_path": "pg19",
        "hf_name": None,
        "splits": ["test"],  # only need test split (~1GB vs 11GB full)
        "text_column": "text",
    },
    "lambada": {
        "hf_path": "lambada",
        "hf_name": None,
        "splits": ["test"],
        "text_column": "text",
    },
}


def download_split(args: tuple[str, str, dict, Path]) -> dict:
    """Download a single split of a dataset. Used by ThreadPoolExecutor."""
    name, split, config, output_dir = args
    
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    split_path = dataset_dir / f"{split}.jsonl"
    
    if split_path.exists():
        # count existing lines
        with open(split_path) as f:
            count = sum(1 for _ in f)
        return {"name": name, "split": split, "status": "exists", "count": count}
    
    # download from HuggingFace
    ds = load_dataset(
        config["hf_path"],
        config["hf_name"],
        split=split,
        trust_remote_code=True,
    )
    
    # save as JSONL 
    ds.to_json(split_path)
    
    return {"name": name, "split": split, "status": "downloaded", "count": len(ds)}


def download(datasets: list[str], output_dir: Path, num_workers: int = 4) -> None:
    """Download datasets in parallel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # build task list: (name, split, config, output_dir)
    tasks = []
    for name in datasets:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}, skipping")
            continue
        config = DATASETS[name]
        for split in config["splits"]:
            tasks.append((name, split, config, output_dir))
    
    if not tasks:
        print("No datasets to download")
        return
    
    print(f"Downloading {len(tasks)} splits with {num_workers} workers...")
    
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(download_split, tasks), total=len(tasks), desc="Downloading"):
            results.append(result)
            status = "skipped (exists)" if result["status"] == "exists" else "downloaded"
            print(f"  {result['name']}/{result['split']}: {result['count']:,} examples ({status})")
    
    # save manifest
    manifest = {
        "datasets": list(set(r["name"] for r in results)),
        "splits": {r["name"]: {r["split"]: r["count"] for r in results if r["name"] == r["name"]}
                   for r in results},
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nDone! Data saved to {output_dir}")


def info(output_dir: Path) -> None:
    """Show status of downloaded datasets."""
    print(f"Eval data directory: {output_dir}")
    print()
    
    for name, config in DATASETS.items():
        dataset_dir = output_dir / name
        print(f"{name}:")
        print(f"  Source: {config['hf_path']}")
        
        for split in config["splits"]:
            split_path = dataset_dir / f"{split}.jsonl"
            if split_path.exists():
                with open(split_path) as f:
                    count = sum(1 for _ in f)
                size_mb = split_path.stat().st_size / (1024 * 1024)
                print(f"  {split}: {count:,} examples ({size_mb:.1f} MB)")
            else:
                print(f"  {split}: not downloaded")
        print()


def main():
    parser = argparse.ArgumentParser(description="Download evaluation datasets")
    subparsers = parser.add_subparsers(dest="command")
    
    # download command
    dl = subparsers.add_parser("download", help="Download datasets from HuggingFace")
    dl.add_argument("--all", action="store_true", help="Download all datasets")
    dl.add_argument("--dataset", type=str, choices=list(DATASETS.keys()), 
                    help="Specific dataset to download")
    dl.add_argument("--output-dir", type=Path, default=EVAL_DATA_DIR)
    dl.add_argument("--workers", type=int, default=4, help="Number of download workers")
    
    # info command
    info_parser = subparsers.add_parser("info", help="Show download status")
    info_parser.add_argument("--output-dir", type=Path, default=EVAL_DATA_DIR)
    
    args = parser.parse_args()
    
    if args.command == "download":
        if args.all:
            datasets = list(DATASETS.keys())
        elif args.dataset:
            datasets = [args.dataset]
        else:
            parser.print_help()
            print("\nError: Specify --all or --dataset <name>")
            return
        download(datasets, args.output_dir, args.workers)
        
    elif args.command == "info":
        info(args.output_dir)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
