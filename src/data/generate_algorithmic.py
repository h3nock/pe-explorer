"""
Synthetic Algorithmic Data Generator for PE Benchmarks

Usage:
    python src/data/generate_algorithmic.py --num_examples 1000000
    python src/data/generate_algorithmic.py --parallel --num_workers 8
"""

import random
import argparse
from pathlib import Path
from multiprocessing import Pool
import yaml
import pandas as pd
from tqdm import tqdm


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data_generation.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--eval_per_task", type=int, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num_workers must be >= 1")
    
    config = load_config(args.config)
    gen_cfg = config["generation"]
    
    # override from CLI or use config defaults
    output_dir = Path(args.output_dir or gen_cfg["output_dir"])
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
