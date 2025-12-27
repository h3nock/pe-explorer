"""
Synthetic Algorithmic Data Generator for PE Benchmarks

Usage:
    python -m src.data.prepare_algorithmic generate --num_examples 1000000
    python -m src.data.prepare_algorithmic tokenize --input_dir ... --output_dir ...
"""

import random
import argparse
from pathlib import Path
from multiprocessing import Pool
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import yaml
import pandas as pd
import numpy as np
import tiktoken
import pyarrow.parquet as pq
from tqdm import tqdm

from src.data.tokenization import ShardWriter, Tokenizer


TOKENIZER_NAME = "gpt2"
SHARD_SIZE = 100_000_000  # 100M tokens per shard, matching prepare_data.py
BUFFER_SIZE = 1_000_000
CACHE_DIR = Path.home() / ".cache" / "algorithmic"


def load_config(config_path: str = "configs/data_generation.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

_worker_config = None

def _init_worker(config):
    """Initialize worker with config."""
    global _worker_config
    _worker_config = config

def _generate_chunk(args):
    """Generate a chunk of examples."""
    start, end, seed = args
    set_seed(seed + start)
    tasks = list(_worker_config["tasks"].keys())
    weights = [_worker_config["tasks"][t]["weight"] for t in tasks]
    data = []
    for _ in range(end - start):
        task = random.choices(tasks, weights=weights, k=1)[0]
        data.append(_generate_one(task, _worker_config["tasks"][task]["train_range"]))
    return data


def set_seed(seed: int):
    random.seed(seed)


def generate_addition(n: int) -> str:
    if n == 1:
        a, b = random.randint(0, 9), random.randint(0, 9)
    else:
        lo, hi = 10 ** (n - 1), 10 ** n - 1
        a, b = random.randint(lo, hi), random.randint(lo, hi)
    return f"{a}+{b}={a + b}"


def generate_sorting(n: int) -> str:
    nums = [random.randint(0, 99) for _ in range(n)]
    return f"sort:[{','.join(map(str, nums))}]->[{','.join(map(str, sorted(nums)))}]"


def generate_reversal(n: int) -> str:
    s = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=n))
    return f"rev:{s}->{s[::-1]}"


def generate_copy(n: int) -> str:
    s = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=n))
    return f"copy:{s}->{s}"


GENERATORS = {
    "addition": generate_addition,
    "sorting": generate_sorting,
    "reversal": generate_reversal,
    "copy": generate_copy,
}


def _generate_one(task: str, length_range: list) -> dict:
    length = random.randint(length_range[0], length_range[1])
    return {"text": GENERATORS[task](length), "task": task, "length": length}


def generate_train_dataset(num_examples: int, config: dict, seed: int) -> list[dict]:
    """Generate training dataset."""
    set_seed(seed)
    tasks = list(config["tasks"].keys())
    weights = [config["tasks"][t]["weight"] for t in tasks]
    data = []
    
    for _ in tqdm(range(num_examples), desc="Generating"):
        task = random.choices(tasks, weights=weights, k=1)[0]
        data.append(_generate_one(task, config["tasks"][task]["train_range"]))
    
    return data


def generate_train_dataset_parallel(num_examples: int, config: dict, seed: int, num_workers: int = 8) -> list[dict]:
    num_workers = min(num_workers, num_examples)  # avoid empty chunks
    chunk_size = num_examples // num_workers
    chunks = [(i * chunk_size, (i + 1) * chunk_size if i < num_workers - 1 else num_examples, seed)
              for i in range(num_workers)]
    
    print(f"Generating {num_examples:,} with {num_workers} workers...")
    with Pool(num_workers, initializer=_init_worker, initargs=(config,)) as pool:
        results = list(tqdm(pool.imap(_generate_chunk, chunks), total=len(chunks)))
    
    return [example for chunk in results for example in chunk]


def generate_eval_dataset(num_per_task: int, config: dict, seed: int) -> list[dict]:
    """Generate eval: 50% interpolation + 50% extrapolation per task."""
    set_seed(seed)
    n_interp, n_extrap = num_per_task // 2, num_per_task - num_per_task // 2
    data = []
    
    for task, cfg in tqdm(config["tasks"].items(), desc="Tasks"):
        for _ in range(n_interp):
            ex = _generate_one(task, cfg["train_range"])
            ex["split"] = "interpolation"
            data.append(ex)
        for _ in range(n_extrap):
            ex = _generate_one(task, cfg["eval_range"])
            ex["split"] = "extrapolation"
            data.append(ex)
    
    return data


def tokenize_worker(args):
    """Worker to tokenize a chunk of parquet file(s) into binary shards."""
    worker_id, input_files, output_dir = args
    
    # helper handles tiktoken init and optimized encoding
    tokenizer = Tokenizer(TOKENIZER_NAME)
    
    buffer = np.zeros(BUFFER_SIZE, dtype=tokenizer.dtype)
    buf_idx = 0
    total_tokens = 0
    total_docs = 0
    
    writer = ShardWriter(output_dir, f"algo_w{worker_id:02d}", SHARD_SIZE)

    def flush_buffer():
        nonlocal buf_idx
        if buf_idx == 0:
            return
        writer.write(buffer[:buf_idx])
        buf_idx = 0

    for file_path in input_files:
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
            for item in batch["text"]:
                text = item.as_py()
                
                # Optimized encoding directly to numpy with EOT
                arr = tokenizer.encode(text)
                if arr is None:
                    continue
                    
                doc_len = len(arr)
                
                if doc_len > BUFFER_SIZE:
                    flush_buffer()
                    writer.write(arr)
                else:
                    if buf_idx + doc_len > BUFFER_SIZE:
                        flush_buffer()
                    buffer[buf_idx:buf_idx + doc_len] = arr
                    buf_idx += doc_len
                
                total_tokens += doc_len
                total_docs += 1
                
    flush_buffer()
    writer.close()
        
    return {
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "shard_paths": writer.shard_paths,
    }

def tokenize_dataset(parquet_dir: Path | list[Path], output_dir: Path, num_workers: int = 8):
    """Tokenize parquet files to binary shards."""
    if isinstance(parquet_dir, list):
        parquet_files = parquet_dir
        input_desc = f"{len(parquet_files)} files"
    else:
        parquet_files = sorted(list(parquet_dir.glob("*.parquet")))
        input_desc = f"{parquet_dir}"

    if not parquet_files:
        print(f"No .parquet files found in {input_desc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split files among workers
    num_workers = min(num_workers, len(parquet_files))
    chunks = [parquet_files[i::num_workers] for i in range(num_workers)]
    
    print(f"Tokenizing {len(parquet_files)} files to {output_dir} with {num_workers} workers...")
    
    worker_args = [(i, chunks[i], output_dir) for i in range(num_workers)]
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(tokenize_worker, worker_args), total=num_workers):
            results.append(result)
            
    # Aggregate and save meta
    all_shards = []
    total_tokens = 0
    total_docs = 0
    
    for r in results:
        all_shards.extend(r["shard_paths"])
        total_tokens += r["total_tokens"]
        total_docs += r["total_docs"]
        
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    dtype_str = "uint16" if tokenizer.n_vocab <= (np.iinfo(np.uint16).max + 1) else "uint32"
    
    meta = {
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "num_shards": len(all_shards),
        "shard_size": SHARD_SIZE,
        "dtype": dtype_str,
        "vocab_size": tokenizer.n_vocab,
        # Store relative paths (filenames) for portability
        "shards": sorted([Path(p).name for p in all_shards]),
    }
    
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"\nDone! {total_tokens:,} tokens in {len(all_shards)} shards.")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    gen_parser.add_argument("--config", type=str, default="configs/data_generation.yaml")
    gen_parser.add_argument("--output_dir", type=str, default=None)
    gen_parser.add_argument("--num_examples", type=int, default=None)
    gen_parser.add_argument("--eval_per_task", type=int, default=None)
    gen_parser.add_argument("--parallel", action="store_true")
    gen_parser.add_argument("--num_workers", "--workers", type=int, default=8, dest="num_workers")

    # Tokenize command
    tok_parser = subparsers.add_parser("tokenize", help="Tokenize parquet to binary")
    tok_parser.add_argument("--input_dir", type=str, default=None, help="Directory containing .parquet files")
    tok_parser.add_argument("--output_dir", type=str, default="data/algorithmic_bin", help="Output directory for .bin files")
    tok_parser.add_argument("--workers", type=int, default=mp.cpu_count())

    args = parser.parse_args()

    if args.command == "tokenize":
        input_dir = Path(args.input_dir or CACHE_DIR)
        output_base = Path(args.output_dir)

        train_file = input_dir / "train.parquet"
        eval_file = input_dir / "eval.parquet"

        if train_file.exists():
            print(f"Tokenizing TRAIN data...")
            tokenize_dataset([train_file], output_base / "train", args.workers)

        if eval_file.exists():
            print(f"Tokenizing EVAL data...")
            tokenize_dataset([eval_file], output_base / "eval", args.workers)

        if not train_file.exists() and not eval_file.exists():
            print(f"Tokenizing ALL .parquet files in {input_dir}...")
            if not input_dir.exists():
                 print(f"Error: Input directory {input_dir} not found.")
                 return
            tokenize_dataset(input_dir, output_base, args.workers)

        return

    # Default to generate if no command or explicit generate
    if args.command == "generate" or args.command is None:
        if args.command is None:
            # support legacy usage without subcommand
            pass
            
        if args.num_workers < 1:
            parser.error("--num_workers must be >= 1")
        
        config = load_config(args.config)
        gen_cfg = config["generation"]
        
        # override from CLI or use config defaults
        # Default output to cache dir
        output_dir = Path(args.output_dir or CACHE_DIR)
        num_examples = args.num_examples or gen_cfg["default_train_examples"]
        eval_per_task = args.eval_per_task or gen_cfg["default_eval_per_task"]

        if num_examples < 1:
            parser.error("--num_examples must be >= 1")
        if eval_per_task < 1:
            parser.error("--eval_per_task must be >= 1")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # train
        print(f"\n{'='*50}\nTRAIN: {num_examples:,} examples\n{'='*50}")
        if args.parallel:
            train_data = generate_train_dataset_parallel(num_examples, config, gen_cfg["train_seed"], args.num_workers)
        else:
            train_data = generate_train_dataset(num_examples, config, gen_cfg["train_seed"])
        pd.DataFrame(train_data).to_parquet(output_dir / "train.parquet", index=False)
        
        # eval
        print(f"\n{'='*50}\nEVAL: {eval_per_task} per task\n{'='*50}")
        eval_data = generate_eval_dataset(eval_per_task, config, gen_cfg["eval_seed"])
        pd.DataFrame(eval_data).to_parquet(output_dir / "eval.parquet", index=False)
        
        print(f"\nDone! Files in {output_dir}/")


if __name__ == "__main__":
    main()
