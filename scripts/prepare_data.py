#!/usr/bin/env python3
"""Download and tokenize FineWeb-Edu for training.

Usage:
    python scripts/prepare_data.py download                   # Download shards
    python scripts/prepare_data.py download --max-shards 100  # Download subset
    python scripts/prepare_data.py tokenize                   # Tokenize to binary
    python scripts/prepare_data.py tokenize --workers 16      # Parallel tokenization
    python scripts/prepare_data.py info                       # Show status
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import pyarrow.parquet as pq
import tiktoken
import yaml
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

# Config
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"
CACHE_DIR = Path.home() / ".cache" / "fineweb-edu"
TOKENIZER_NAME = "gpt2"
SHARD_SIZE = 100_000_000  # 100M tokens per output shard (~200MB)
BUFFER_SIZE = 1_000_000   # 1M token buffer per worker (~2MB)

# vocab info for metadata (GPT-2 = 50257, fits in uint16)
VOCAB_SIZE = tiktoken.get_encoding(TOKENIZER_NAME).n_vocab


def load_config() -> dict:
    """Load data.fineweb config with defaults."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = (yaml.safe_load(f) or {}).get("data", {}).get("fineweb", {})
    else:
        cfg = {}
    return {
        "repo": cfg.get("repo", "karpathy/fineweb-edu-100b-shuffle"),
        "total_shards": cfg.get("total_shards", 1822),
        "download_shards": cfg.get("download_shards", 911),
        "tokens_per_shard": cfg.get("tokens_per_shard", 55_000_000),
    }


def download_one(args: tuple[str, str, str]) -> str:
    repo, filename, cache_dir = args
    return hf_hub_download(repo, filename, repo_type="dataset", cache_dir=cache_dir)

def download(max_shards: int, num_workers: int = 8) -> None:
    """Download parquet shards from HuggingFace in parallel."""
    cfg = load_config()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Listing files in {cfg['repo']}...")
    files = sorted([f for f in list_repo_files(cfg["repo"], repo_type="dataset") if f.endswith(".parquet")])
    files = files[:max_shards]

    print(f"Downloading {len(files)}/{cfg['total_shards']} shards with {num_workers} workers...")

    cache_dir = str(CACHE_DIR / "hf_cache")
    args = [(cfg["repo"], f, cache_dir) for f in files]

    paths = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for path in tqdm(executor.map(download_one, args), total=len(files), desc="Downloading"):
            paths.append(path)

    with open(CACHE_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"shards": paths, "count": len(paths)}, f, indent=2)
    print(f"Done! {len(paths)} shards saved.")


def tokenize_worker(args: tuple) -> dict:
    """Worker function: tokenize input shards with streaming writes."""
    worker_id, input_shards, output_dir = args
    
    # initialize inside worker for process safety
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    eot = tokenizer.eot_token
    dtype = np.uint16 if tokenizer.n_vocab <= (np.iinfo(np.uint16).max + 1) else np.uint32
    
    buffer = np.zeros(BUFFER_SIZE, dtype=dtype)
    buf_idx = 0
    shard_idx = 0
    shard_tokens = 0  # tokens written to current output shard
    total_tokens = 0
    total_docs = 0
    shard_paths = []
    current_file = None
    
    def open_new_shard():
        nonlocal current_file, shard_idx, shard_tokens
        if current_file:
            current_file.close()
        path = output_dir / f"train_w{worker_id:02d}_{shard_idx:04d}.bin"
        shard_paths.append(str(path))
        current_file = open(path, "wb")
        shard_idx += 1
        shard_tokens = 0
    
    def write_to_shards(arr: np.ndarray):
        """Write token array to shard(s), splitting at shard boundaries."""
        nonlocal current_file, shard_tokens
        pos = 0
        while pos < len(arr):
            if current_file is None:
                open_new_shard()
            space_left = SHARD_SIZE - shard_tokens
            to_write = min(len(arr) - pos, space_left)
            arr[pos:pos + to_write].tofile(current_file)
            shard_tokens += to_write
            pos += to_write
            # if shard is full, close it (next write opens a new one)
            if shard_tokens >= SHARD_SIZE:
                current_file.close()
                current_file = None
    
    def flush_buffer():
        """Write buffer to shard(s) and reset buffer index."""
        nonlocal buf_idx
        if buf_idx == 0:
            return
        write_to_shards(buffer[:buf_idx])
        buf_idx = 0
    
    for shard_path in input_shards:
        # stream parquet in batches 
        parquet_file = pq.ParquetFile(shard_path)
        for batch in parquet_file.iter_batches(batch_size=1000, columns=["text"]):
            for item in batch["text"]:  # iterate pyarrow array directly
                text = item.as_py()     # convert one at a time
                tokens = tokenizer.encode(text)
                tokens.append(eot)
                doc_len = len(tokens)
                arr = np.asarray(tokens, dtype=dtype)
                
                # check if doc would cross shard boundary - start new shard at doc boundary
                if shard_tokens + buf_idx + doc_len > SHARD_SIZE and (shard_tokens + buf_idx) > 0:
                    flush_buffer()
                    open_new_shard()
                
                # handle oversized docs (larger than buffer)
                if doc_len > BUFFER_SIZE:
                    flush_buffer()  # write any pending data first
                    write_to_shards(arr)
                else:
                    # flush buffer if doc won't fit
                    if buf_idx + doc_len > BUFFER_SIZE:
                        flush_buffer()
                    # bulk copy into buffer
                    buffer[buf_idx:buf_idx + doc_len] = arr
                    buf_idx += doc_len
                
                total_tokens += doc_len
                total_docs += 1
    
    flush_buffer()
    if current_file:
        current_file.close()
    
    return {
        "worker_id": worker_id,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "shard_paths": shard_paths,
    }


def tokenize(num_workers: int = 8) -> None:
    """Parallel tokenization: each worker handles a range of input shards."""
    manifest_path = CACHE_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No shards found. Run 'download' first.")
        return

    with open(manifest_path, encoding="utf-8") as f:
        input_shards = json.load(f)["shards"]

    output_dir = CACHE_DIR / "tokenized"
    output_dir.mkdir(exist_ok=True)
    
    # Split input shards across workers
    num_workers = min(num_workers, len(input_shards))
    chunks = [input_shards[i::num_workers] for i in range(num_workers)]
    
    print(f"Tokenizing {len(input_shards)} shards with {num_workers} workers...")
    print(f"  Output: {output_dir}")
    
    # Parallel tokenization
    worker_args = [(i, chunks[i], output_dir) for i in range(num_workers)]
    
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(tokenize_worker, worker_args), total=num_workers, desc="Workers"):
            results.append(result)
    
    # aggregate results
    total_tokens = sum(r["total_tokens"] for r in results)
    total_docs = sum(r["total_docs"] for r in results)
    all_shards = []
    for r in results:
        all_shards.extend(r["shard_paths"])
    
    # save metadata 
    dtype_str = "uint16" if VOCAB_SIZE <= (np.iinfo(np.uint16).max + 1) else "uint32"
    meta = {
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "num_shards": len(all_shards),
        "shard_size": SHARD_SIZE,
        "dtype": dtype_str,
        "vocab_size": VOCAB_SIZE,
        "shards": sorted(all_shards),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone!")
    print(f"  Documents: {total_docs:,}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Shards: {len(all_shards)}")


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

    meta = CACHE_DIR / "tokenized" / "meta.json"
    if meta.exists():
        with open(meta, encoding="utf-8") as f:
            m = json.load(f)
        print(f"Tokenized: {m['total_tokens']:,} tokens across {m['num_shards']} shards")
        print(f"  Dtype: {m.get('dtype', 'uint16')} (vocab: {m.get('vocab_size', 'unknown')})")
    else:
        print("Tokenized: not yet")


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

    # info
    sub.add_parser("info", help="Show status")

    args = parser.parse_args()

    if args.cmd == "download":
        download(args.max_shards, args.workers)
    elif args.cmd == "tokenize":
        tokenize(args.workers)
    elif args.cmd == "info":
        info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
