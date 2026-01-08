"""Download and tokenize FineWeb-Edu for training.

Usage:
    python -m src.data.prepare_fineweb download                   # Download shards
    python -m src.data.prepare_fineweb download --max-shards 100  # Download subset
    python -m src.data.prepare_fineweb tokenize                   # Tokenize to binary
    python -m src.data.prepare_fineweb tokenize --workers 16      # Parallel tokenization
    python -m src.data.prepare_fineweb info                       # Show status
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import time
from multiprocessing import cpu_count, Manager

import numpy as np
import pyarrow.parquet as pq
import tiktoken
import yaml
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

from src.data.tokenization import ShardWriter, Tokenizer

# Config
CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
CACHE_DIR = Path.home() / ".cache" / "fineweb-edu"
TOKENIZER_NAME = "gpt2"
SHARD_SIZE = 100_000_000  # 100M tokens per output shard (~200MB)
BUFFER_SIZE = 1_000_000   # 1M token buffer per worker (~2MB)

# vocab info for metadata (GPT-2 = 50257, fits in uint16)
VOCAB_SIZE = tiktoken.get_encoding(TOKENIZER_NAME).n_vocab

# overridden by configs/config.yaml if specified
DEFAULT_TRAINING_SHARDS = (0, 880)
DEFAULT_VALIDATION_SHARDS = (881, 890)
DEFAULT_FILLER_SHARDS = (891, 910)


def load_config() -> dict:
    """Load data.fineweb config with defaults and validation.
    
    Validates:
        - training, validation, and filler shards are disjoint
        - filler_shards is within downloaded range
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = (yaml.safe_load(f) or {}).get("data", {}).get("fineweb", {})
    else:
        cfg = {}
    
    config = {
        "repo": cfg.get("repo", "karpathy/fineweb-edu-100b-shuffle"),
        "total_shards": cfg.get("total_shards", 1822),
        "download_shards": cfg.get("download_shards", 911),
        "tokens_per_shard": cfg.get("tokens_per_shard", 55_000_000),
        "tokenized_path": cfg.get("tokenized_path", "data/fineweb_bin"),
        # shard partitions (inclusive bounds)
        "training_shards": cfg.get("training_shards", list(DEFAULT_TRAINING_SHARDS)),
        "validation_shards": cfg.get("validation_shards", list(DEFAULT_VALIDATION_SHARDS)),
        "filler_shards": cfg.get("filler_shards", list(DEFAULT_FILLER_SHARDS)),
    }
    
    # partitions must be disjoint: training < validation < filler
    train_end = config["training_shards"][1]
    val_start, val_end = config["validation_shards"]
    filler_start = config["filler_shards"][0]
    
    if train_end >= val_start:
        raise ValueError(f"Overlap: training ends at {train_end}, validation starts at {val_start}")
    if val_end >= filler_start:
        raise ValueError(f"Overlap: validation ends at {val_end}, filler starts at {filler_start}")
    
    # filler must be within download range
    filler_end = config["filler_shards"][1]
    download_max = config["download_shards"] - 1
    if filler_end > download_max:
        raise ValueError(
            f"Filler shards [{filler_start}, {filler_end}] exceed download range [0, {download_max}]. "
            f"Increase download_shards to at least {filler_end + 1}."
        )
    
    return config


def download_one(args: tuple[str, str, str]) -> str:
    repo, filename, output_dir = args
    return hf_hub_download(
        repo, 
        filename, 
        repo_type="dataset", 
        local_dir=output_dir,
        local_dir_use_symlinks=False
    )


def extract_shard_index(filepath: str) -> int | None:
    """Extract shard index from FineWeb filename (e.g., 'train-00042-of-01822.parquet' -> 42)."""
    # fineWeb-Edu uses: train-NNNNN-of-TOTAL.parquet
    if match := re.search(r"train-(\d+)-of-\d+\.parquet$", Path(filepath).name):
        return int(match.group(1))
    return None

def download(max_shards: int, num_workers: int = 8) -> None:
    """Download parquet shards from HuggingFace in parallel."""
    cfg = load_config()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Listing files in {cfg['repo']}...")
    files = sorted([f for f in list_repo_files(cfg["repo"], repo_type="dataset") if f.endswith(".parquet")])
    files = files[:max_shards]

    print(f"Downloading {len(files)}/{cfg['total_shards']} shards with {num_workers} workers...")
    
    args = [(cfg["repo"], f, str(CACHE_DIR)) for f in files]

    paths = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for path in tqdm(executor.map(download_one, args), total=len(files), desc="Downloading"):
            paths.append(path)

    with open(CACHE_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"shards": paths, "count": len(paths)}, f, indent=2)
    print(f"Done! {len(paths)} shards saved.")


def tokenize_worker(args: tuple) -> dict:
    """Worker function: tokenize input shards with streaming writes."""
    worker_id, input_shards, output_dir, queue = args
    
    # helper handles tiktoken init and optimized encoding
    tokenizer = Tokenizer(TOKENIZER_NAME)
    
    buffer = np.zeros(BUFFER_SIZE, dtype=tokenizer.dtype)
    buf_idx = 0
    total_tokens = 0
    total_docs = 0
    
    writer = ShardWriter(output_dir, f"train_w{worker_id:02d}", SHARD_SIZE)
    
    def flush_buffer():
        """Write buffer to shard(s) and reset buffer index."""
        nonlocal buf_idx
        if buf_idx == 0:
            return
        writer.write(buffer[:buf_idx])
        buf_idx = 0
    
    for shard_path in input_shards:
        # stream parquet in batches 
        parquet_file = pq.ParquetFile(shard_path)
        for batch in parquet_file.iter_batches(batch_size=1000, columns=["text"]):
            for item in batch["text"]:  # iterate pyarrow array directly
                text = item.as_py()     # convert one at a time
                
                # Optimized encoding directly to numpy with EOT
                arr = tokenizer.encode(text)
                if arr is None: 
                    continue
                    
                doc_len = len(arr)
                
                # handle oversized docs (larger than buffer)
                if doc_len > BUFFER_SIZE:
                    flush_buffer()  # write any pending data first
                    writer.write(arr)
                else:
                    # flush buffer if doc won't fit
                    if buf_idx + doc_len > BUFFER_SIZE:
                        flush_buffer()
                    # bulk copy into buffer
                    buffer[buf_idx:buf_idx + doc_len] = arr
                    buf_idx += doc_len
                
                total_tokens += doc_len
                total_docs += 1
        
        # Report progress for this shard
        if queue is not None:
            queue.put(1)
    
    flush_buffer()
    writer.close()
    
    return {
        "worker_id": worker_id,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "shard_paths": writer.shard_paths,
    }


def filter_shards_by_range(all_shards: list[str], shard_range: tuple[int, int]) -> list[str]:
    """Filter shard paths to those within the given index range."""
    start_idx, end_idx = shard_range
    return [p for p in all_shards
            if (idx := extract_shard_index(p)) is not None and start_idx <= idx <= end_idx]


def tokenize(num_workers: int = 8, split: str = "train") -> None:
    """Parallel tokenization for training or validation shards.
    
    Args:
        num_workers: Number of parallel workers
        split: 'train' or 'val' - determines which shard range to use
    """
    manifest_path = CACHE_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No shards found. Run 'download' first.")
        return

    with open(manifest_path, encoding="utf-8") as f:
        all_shards = json.load(f)["shards"]

    cfg = load_config()
    
    # select shard range based on split
    if split == "train":
        shard_range = tuple(cfg["training_shards"])
        output_subdir = "train"
    elif split == "val":
        shard_range = tuple(cfg["validation_shards"])
        output_subdir = "val"
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")
    
    selected_shards = filter_shards_by_range(all_shards, shard_range)
    
    if not selected_shards:
        print(f"No shards found for {split} in range {shard_range}.")
        return

    output_dir = Path(cfg["tokenized_path"]).expanduser() / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_workers = min(num_workers, len(selected_shards))
    chunks = [selected_shards[i::num_workers] for i in range(num_workers)]
    
    print(f"Tokenizing {len(selected_shards)} {split} shards with {num_workers} workers...")
    print(f"  Output: {output_dir}")
    
    with Manager() as manager:
        queue = manager.Queue()
        worker_args = [(i, chunks[i], output_dir, queue) for i in range(num_workers)]
        
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(tokenize_worker, args) for args in worker_args]
            
            with tqdm(total=len(selected_shards), desc="Tokenizing shards", unit="shard") as pbar:
                processed_count = 0
                while processed_count < len(selected_shards):
                    try:
                        while not queue.empty():
                            queue.get_nowait()
                            pbar.update(1)
                            processed_count += 1
                    except Exception:
                        pass
                    
                    if all(f.done() for f in futures):
                        while not queue.empty():
                            queue.get_nowait()
                            pbar.update(1)
                            processed_count += 1
                        break
                    
                    time.sleep(0.5)

            for f in futures:
                results.append(f.result())
    
    total_tokens = sum(r["total_tokens"] for r in results)
    total_docs = sum(r["total_docs"] for r in results)
    all_shard_paths = []
    for r in results:
        all_shard_paths.extend(r["shard_paths"])
    
    dtype_str = "uint16" if VOCAB_SIZE <= (np.iinfo(np.uint16).max + 1) else "uint32"
    meta = {
        "split": split,
        "shard_range": list(shard_range),
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "num_shards": len(all_shard_paths),
        "shard_size": SHARD_SIZE,
        "dtype": dtype_str,
        "vocab_size": VOCAB_SIZE,
        "shards": sorted([Path(p).name for p in all_shard_paths]),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone!")
    print(f"  Documents: {total_docs:,}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Shards: {len(all_shard_paths)}")


def info():
    """Show dataset status."""
    cfg = load_config()

    print(f"Config: {cfg['repo']}")
    print(f"  Total shards: {cfg['total_shards']}")
    print(f"  Default download: {cfg['download_shards']} shards")
    print()

    manifest = CACHE_DIR / "manifest.json"
    if manifest.exists():
        with open(manifest, encoding="utf-8") as f:
            count = json.load(f)["count"]
        print(f"Downloaded: {count} shards")
    else:
        print("Downloaded: 0 shards")

    output_dir = Path(cfg["tokenized_path"]).expanduser()
    train_meta = output_dir / "train" / "meta.json"
    val_meta = output_dir / "val" / "meta.json"

    if train_meta.exists() or val_meta.exists():
        for split, path in (("train", train_meta), ("val", val_meta)):
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
            print(f"Tokenized ({split}): {m['total_tokens']:,} tokens across {m['num_shards']} shards")
            print(f"  Path: {path.parent}")
            print(f"  Dtype: {m.get('dtype', 'uint16')} (vocab: {m.get('vocab_size', 'unknown')})")
    else:
        print(f"Tokenized: not yet (checked {output_dir})")


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu dataset")
    sub = parser.add_subparsers(dest="cmd")

    # download
    dl = sub.add_parser("download", help="Download shards from HuggingFace")
    dl.add_argument("--max-shards", type=int, default=cfg["download_shards"], help="Number of shards")
    dl.add_argument("--workers", type=int, default=8, help="Number of download workers")

    # tokenize
    tok = sub.add_parser("tokenize", help="Tokenize to binary shards")
    tok.add_argument("--workers", type=int, default=cpu_count(), help="Number of workers")
    tok.add_argument("--split", choices=["train", "val", "all"], default="all",
                     help="Which split to tokenize (default: all)")

    # info
    sub.add_parser("info", help="Show status")

    args = parser.parse_args()

    if args.cmd == "download":
        download(args.max_shards, args.workers)
    elif args.cmd == "tokenize":
        if args.split == "all":
            tokenize(args.workers, split="train")
            tokenize(args.workers, split="val")
        else:
            tokenize(args.workers, split=args.split)
    elif args.cmd == "info":
        info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
