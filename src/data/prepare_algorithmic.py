"""
Synthetic Data Generator for PE Benchmarks

Usage:
    python -m src.data.prepare_algorithmic generate --num_examples 1000000
    python -m src.data.prepare_algorithmic tokenize --input_dir ... --output_dir ...
"""

import argparse
import json
import multiprocessing as mp
import random
import string
import bisect
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from src.data.prepare_fineweb import DEFAULT_FILLER_SHARDS, extract_shard_index
from src.data.tokenization import (
    ShardWriter, Tokenizer, load_encoding, get_bos_token_id, get_dtype_str
)


SHARD_SIZE = 100_000_000  # 100M tokens per shard
BUFFER_SIZE = 1_000_000
CACHE_DIR = Path.home() / ".cache" / "algorithmic"
FINEWEB_CACHE_DIR = Path.home() / ".cache" / "fineweb-edu"
DIGITS = [str(i) for i in range(10)]
LETTERS = list(string.ascii_lowercase)

# Lazy-loaded encoder and constant tokens
_ENC = None
_PASSKEY_PROMPT_IDS = None
_COPY_PROMPT_IDS = None
_BOS_ID = None


def _get_enc():
    """Lazy load encoder on first use."""
    global _ENC, _PASSKEY_PROMPT_IDS, _COPY_PROMPT_IDS
    if _ENC is None:
        _ENC = load_encoding()
        _PASSKEY_PROMPT_IDS = _ENC.encode("The code is ")
        _COPY_PROMPT_IDS = _ENC.encode("PASTE: ")
    return _ENC


def _get_bos_id():
    """Cache BOS token id for consistent token-space outputs."""
    global _BOS_ID
    if _BOS_ID is None:
        _BOS_ID = get_bos_token_id(_get_enc())
    return _BOS_ID


def _prepend_bos(token_ids: list[int]) -> list[int]:
    bos = _get_bos_id()
    if bos is None:
        return list(token_ids)
    if token_ids and token_ids[0] == bos:
        return list(token_ids)
    return [bos] + list(token_ids)


def load_config(config_path: str = "configs/data_generation.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("content_sampling", {"digit": 0.5, "char": 0.5})
    cfg.setdefault("tasks", {})
    cfg.setdefault("filler", {})
    cfg.setdefault("generation", {})
    return cfg


class FillerPool:
    """Pre-tokenized filler pool for exact token-space sampling."""

    def __init__(self, pool_path: Path):
        df = pd.read_parquet(pool_path)
        token_ids_raw = df["token_ids"].tolist()
        self.token_ids = [list(x) for x in token_ids_raw]
        self.token_lens = df["token_len"].tolist()
        # sorted indices for bisect
        self.sorted_idx = sorted(range(len(self.token_lens)), key=lambda i: self.token_lens[i])
        self.sorted_lens = [self.token_lens[i] for i in self.sorted_idx]
        self.bos = get_bos_token_id(_get_enc())
        if self.bos is None:
            raise ValueError("Tokenizer missing <|bos|> special token")
        print(f"  Loaded filler pool: {len(self.token_ids):,} docs")

    def sample(self, target_tokens: int) -> list[int]:
        """Sample filler tokens. Returns list[int], not text."""
        if target_tokens <= 0:
            return []

        # try to find a single doc that is long enough
        pos = bisect.bisect_left(self.sorted_lens, target_tokens)

        if pos < len(self.sorted_idx):
            # pick one doc randomly 
            doc_idx = self.sorted_idx[random.randrange(pos, len(self.sorted_idx))]
            tokens = self.token_ids[doc_idx]
            return tokens[:target_tokens]

        # concatenate multiple docs if needed
        result: list[int] = []
        first_doc = True
        while len(result) < target_tokens:
            idx = random.randrange(len(self.token_ids))
            doc_tokens = self.token_ids[idx]
            if not first_doc and self.bos is not None:
                result.append(self.bos)
            result.extend(doc_tokens)
            first_doc = False
            
        return result[:target_tokens]

    def __len__(self) -> int:
        return len(self.token_ids)


def build_filler_pool(
    fineweb_dir: Path,
    shard_range: tuple[int, int],
    output_path: Path,
    min_tokens: int = 50,
    max_docs: int = 50000,
) -> None:
    """Build pre-tokenized filler pool from FineWeb shards."""
    fineweb_dir = fineweb_dir.expanduser()
    all_paths = sorted(fineweb_dir.rglob("*.parquet"))
    start_idx, end_idx = shard_range
    paths = [p for p in all_paths if (idx := extract_shard_index(str(p))) is not None and start_idx <= idx <= end_idx]

    if not paths:
        raise FileNotFoundError(f"No shards in range [{start_idx}, {end_idx}]")

    enc = load_encoding()
    token_ids_list, token_lens = [], []
    done = False

    print(f"Building filler pool from {len(paths)} shards...")
    for path in tqdm(paths, desc="Processing"):
        if done:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
            if done:
                break
            for item in batch["text"]:
                text = item.as_py()
                if text:
                    toks = enc.encode(text)
                    if len(toks) >= min_tokens:
                        token_ids_list.append(toks)
                        token_lens.append(len(toks))
                        if len(token_ids_list) >= max_docs:
                            done = True
                            break

    df = pd.DataFrame({"token_ids": token_ids_list, "token_len": token_lens})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df):,} docs to {output_path}")


def set_seed(seed: int):
    random.seed(seed)

def sample_length(length_range: list[int]) -> int:
    return random.randint(length_range[0], length_range[1])

def split_counts(total: int, n: int) -> list[int]:
    if n <= 0:
        return []
    base = total // n
    remainder = total % n
    return [base + (1 if i < remainder else 0) for i in range(n)]

def select_length_range(cfg: dict, mode: str) -> list[int]:
    length_cfg = cfg.get("length", {})
    if mode == "train":
        return length_cfg.get("train", length_cfg.get("eval_id"))
    if mode == "id":
        return length_cfg.get("eval_id", length_cfg.get("train"))
    if mode == "ood":
        return length_cfg.get("eval_ood", length_cfg.get("train"))
    raise ValueError(f"Unknown mode: {mode}")

def generate_passkey_example(cfg: dict, filler_pool: FillerPool, L: int, context_len: int, distance: float) -> dict:
    """Generate passkey example. Distance is fraction of usable reference space (L)."""
    code_len = random.randint(cfg["code_len"][0], cfg["code_len"][1])
    code = "".join(random.choice(DIGITS) for _ in range(code_len))

    prefix_ids = _get_enc().encode(f"The code is {code}. ")
    prompt_ids = _PASSKEY_PROMPT_IDS 
    target_ids = _get_enc().encode(code)

    usable_L_ref = L - len(prompt_ids) - len(target_ids)
    target_pos = int(distance * usable_L_ref)
    
    # calculate filler needed
    filler_needed = max(1, target_pos - len(prefix_ids))
    
    total = len(prefix_ids) + len(prompt_ids) + len(target_ids)
    max_filler = context_len - total
    if filler_needed > max_filler:
        filler_needed = max_filler

    filler_ids = filler_pool.sample(filler_needed)

    # Build full sequence (avoid intermediate lists)
    full_ids = prefix_ids + filler_ids
    full_ids.extend(prompt_ids)
    full_ids.extend(target_ids)
    
    prompt_ids_full = prefix_ids + filler_ids
    prompt_ids_full.extend(prompt_ids)

    return {
        "text": _get_enc().decode(full_ids),
        "prompt": _get_enc().decode(prompt_ids_full),
        "target": code,
        "task": "passkey",
        "code_len": code_len,
        "distance": distance,
        "total_tokens": len(full_ids),
        "token_ids": _prepend_bos(full_ids),
    }


def generate_copy_distance_example(cfg: dict, content_sampling: dict, filler_pool: FillerPool, L: int, context_len: int, distance: float) -> dict:
    """Generate copy_distance example. Distance is fraction of usable reference space (L)."""
    content_type = random.choices(list(content_sampling.keys()), weights=list(content_sampling.values()), k=1)[0]
    pool = DIGITS if content_type == "digit" else LETTERS
    items = [random.choice(pool) for _ in range(cfg["content_len"])]
    content = " ".join(items)

    # Token parts (use cached encoder)
    prefix_ids = _get_enc().encode(f"COPY: {content} ||| ")
    prompt_ids = _COPY_PROMPT_IDS  # Pre-encoded constant
    target_ids = _get_enc().encode(content)

    # Calculate distance relative to L (reference context)
    usable_L_ref = L - len(prompt_ids) - len(target_ids)
    target_pos = int(distance * usable_L_ref)
    
    # Calculate filler needed
    filler_needed = max(1, target_pos - len(prefix_ids))

    total = len(prefix_ids) + len(prompt_ids) + len(target_ids)
    max_filler = context_len - total
    if filler_needed > max_filler:
        filler_needed = max_filler

    filler_ids = filler_pool.sample(filler_needed)

    # Build full sequence (avoid intermediate lists)
    full_ids = prefix_ids + filler_ids
    full_ids.extend(prompt_ids)
    full_ids.extend(target_ids)
    
    prompt_ids_full = prefix_ids + filler_ids
    prompt_ids_full.extend(prompt_ids)

    return {
        "text": _get_enc().decode(full_ids),
        "prompt": _get_enc().decode(prompt_ids_full),
        "target": content,
        "task": "copy_distance",
        "content_type": content_type,
        "content_len": cfg["content_len"],
        "distance": distance,
        "total_tokens": len(full_ids),
        "token_ids": _prepend_bos(full_ids),
    }


def generate_length_task(task: str, cfg: dict, content_sampling: dict, mode: str, length_override: int | None = None) -> dict:
    """Unified generator for reverse, sort, simple_copy, no_carry_add."""
    length_cfg = cfg.get("length", {})
    if mode == "train":
        length_range = length_cfg.get("train", length_cfg.get("eval_id"))
    elif mode == "id":
        length_range = length_cfg.get("eval_id", length_cfg.get("train"))
    elif mode == "ood":
        length_range = length_cfg.get("eval_ood", length_cfg.get("train"))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    length = length_override if length_override is not None else random.randint(length_range[0], length_range[1])

    if task == "no_carry_add":
        digits_cfg = cfg.get("digits", [0, 4])
        min_d, max_d = int(digits_cfg[0]), int(digits_cfg[1])
        if min_d + min_d >= 10:
            raise ValueError("Digits range does not allow no-carry addition.")
        digits = list(range(min_d, max_d + 1))
        non_zero = [d for d in digits if d != 0]
        a_digits, b_digits = [], []
        for i in range(length):
            if i == 0 and non_zero:
                a, b = random.choice(non_zero), random.choice(non_zero)
            else:
                a, b = random.choice(digits), random.choice(digits)
            while a + b >= 10:
                a, b = random.choice(digits), random.choice(digits)
            a_digits.append(a)
            b_digits.append(b)
        a_str = "".join(str(d) for d in a_digits)
        b_str = "".join(str(d) for d in b_digits)
        sum_str = "".join(str(a + b) for a, b in zip(a_digits, b_digits))
        text = f"add: {a_str}+{b_str}={sum_str}"
        return {
            "text": text,
            "prompt": f"add: {a_str}+{b_str}=",
            "target": sum_str,
            "task": "no_carry_add",
            "length": length,
            "token_ids": _prepend_bos(_get_enc().encode(text)),
        }

    # Content-based tasks: reverse, sort, simple_copy
    content_type = random.choices(list(content_sampling.keys()), weights=list(content_sampling.values()), k=1)[0]
    pool = DIGITS if content_type == "digit" else LETTERS
    unique = (task == "sort")
    if unique:
        if length > len(pool):
            raise ValueError(f"Cannot sample {length} unique items from {content_type}")
        items = random.sample(pool, length)
    else:
        items = [random.choice(pool) for _ in range(length)]
    input_text = " ".join(items)

    if task == "reverse":
        output_text = " ".join(reversed(items))
        fmt = "reverse"
    elif task == "sort":
        if content_type == "digit":
            sorted_items = [str(i) for i in sorted(int(x) for x in items)]
        else:
            sorted_items = sorted(items)
        output_text = " ".join(sorted_items)
        fmt = "sort"
    else:  # simple_copy
        output_text = input_text
        fmt = "copy"

    return {
        "text": f"{fmt}: {input_text} -> {output_text}",
        "prompt": f"{fmt}: {input_text} ->",
        "target": output_text,
        "task": task,
        "content_type": content_type,
        "length": length,
        "token_ids": _prepend_bos(_get_enc().encode(f"{fmt}: {input_text} -> {output_text}")),
    }

def _generate_one(task: str, config: dict, mode: str, overrides: dict | None = None) -> dict:
    cfg = config["tasks"][task]
    content_sampling = config["content_sampling"]
    filler_pool = config.get("filler_pool")
    context_len = config.get("context_len", 1024)
    L = config.get("L", 1024)  # Reference unit (always 1024)

    if mode == "ood":
        # Use larger context for OOD extrapolation (2L)
        context_len = config.get("eval_context_len", context_len * 2)

    overrides = overrides or {}

    if task == "passkey":
        if not filler_pool:
            raise ValueError("FillerPool required for passkey generation.")
        mode_key = "eval_id" if mode == "id" else ("eval_ood" if mode == "ood" else "train")
        distances = cfg.get("distances", {}).get(mode_key, [0.25, 0.5, 0.75, 1.0])
        distance = overrides["distance"] if "distance" in overrides else random.choice(distances)
        return generate_passkey_example(cfg, filler_pool, L, context_len, distance)
    if task == "copy_distance":
        if not filler_pool:
            raise ValueError("FillerPool required for copy_distance generation.")
        mode_key = "eval_id" if mode == "id" else ("eval_ood" if mode == "ood" else "train")
        distances = cfg.get("distances", {}).get(mode_key, [0.25, 0.5, 0.75, 1.0])
        distance = overrides["distance"] if "distance" in overrides else random.choice(distances)
        return generate_copy_distance_example(cfg, content_sampling, filler_pool, L, context_len, distance)
    if task in ("reverse", "sort", "simple_copy", "no_carry_add"):
        return generate_length_task(task, cfg, content_sampling, mode, length_override=overrides.get("length"))
    raise ValueError(f"Unknown task: {task}")

_worker_config = None

def _init_worker(config):
    global _worker_config
    if "filler_pool_path" in config and "filler_pool" not in config:
        config = dict(config)
        config["filler_pool"] = FillerPool(Path(config["filler_pool_path"]))
    _worker_config = config

def _generate_chunk(args):
    start, end, seed = args
    set_seed(seed + start)
    tasks = list(_worker_config["tasks"].keys())
    weights = [_worker_config["tasks"][t]["weight"] for t in tasks]
    data = []
    for _ in range(end - start):
        task = random.choices(tasks, weights=weights, k=1)[0]
        data.append(_generate_one(task, _worker_config, mode="train"))
    return data


def generate_train_dataset(num_examples: int, config: dict, seed: int, chunk_size: int = 250_000, batch_size: int = 10_000):
    # batch_size = how much to generate at once (tune for RAM)
    # chunk_size = how many examples per output file
    set_seed(seed)
    tasks = list(config["tasks"].keys())
    weights = [config["tasks"][t]["weight"] for t in tasks]

    buffer = []
    generated = 0

    with tqdm(total=num_examples, desc="Generating") as pbar:
        while generated < num_examples:
            current_batch = min(batch_size, num_examples - generated)
            for _ in range(current_batch):
                task = random.choices(tasks, weights=weights, k=1)[0]
                buffer.append(_generate_one(task, config, mode="train"))
            generated += current_batch
            pbar.update(current_batch)

            # yield full chunks when ready
            while len(buffer) >= chunk_size:
                yield buffer[:chunk_size]
                buffer = buffer[chunk_size:]

    # yield any remaining data
    if buffer:
        yield buffer


def generate_train_dataset_parallel(
    num_examples: int,
    config: dict,
    seed: int,
    num_workers: int = 8,
    chunk_size: int = 250_000,
    batch_size: int = 10_000,
):
    # batch_size = how much each worker holds in memory at once 
    # chunk_size = how many examples per output file (tune this for I/O efficiency)
    num_batches = (num_examples + batch_size - 1) // batch_size

    batches = []
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_examples)
        batches.append((start, end, seed))

    print(f"Generating {num_examples:,} train examples with {num_workers} workers...")

    worker_config = config
    if "filler_sampler" in config and "filler_sampler_params" in config:
        worker_config = dict(config)
        worker_config.pop("filler_sampler", None)

    buffer = []
    with Pool(num_workers, initializer=_init_worker, initargs=(worker_config,)) as pool:
        for batch_data in tqdm(pool.imap(_generate_chunk, batches), total=len(batches), desc="Generating"):
            buffer.extend(batch_data)
            
            # yield full chunks when ready
            while len(buffer) >= chunk_size:
                yield buffer[:chunk_size]
                buffer = buffer[chunk_size:]
    
    # yield any remaining data
    if buffer:
        yield buffer


def generate_eval_dataset(num_per_task: int, config: dict, seed: int) -> list[dict]:
    set_seed(seed)
    data = []

    for task, cfg in tqdm(config["tasks"].items(), desc="Tasks"):
        id_count = num_per_task // 2
        ood_count = num_per_task - id_count

        if task in {"passkey", "copy_distance"}:
            id_distances = cfg["distances"]["eval_id"]
            ood_distances = cfg["distances"]["eval_ood"]
            for distance, count in zip(id_distances, split_counts(id_count, len(id_distances))):
                for _ in range(count):
                    ex = _generate_one(task, config, mode="id", overrides={"distance": distance})
                    ex["split"] = "id"
                    data.append(ex)
            for distance, count in zip(ood_distances, split_counts(ood_count, len(ood_distances))):
                for _ in range(count):
                    ex = _generate_one(task, config, mode="ood", overrides={"distance": distance})
                    ex["split"] = "ood"
                    data.append(ex)
            continue

        id_range = select_length_range(cfg, "id")
        ood_range = select_length_range(cfg, "ood")
        for _ in range(id_count):
            length = sample_length(id_range)
            ex = _generate_one(task, config, mode="id", overrides={"length": length})
            ex["split"] = "id"
            data.append(ex)
        for _ in range(ood_count):
            length = sample_length(ood_range)
            ex = _generate_one(task, config, mode="ood", overrides={"length": length})
            ex["split"] = "ood"
            data.append(ex)

    return data


def tokenize_worker(args):
    worker_id, input_files, output_dir = args

    tokenizer = Tokenizer()

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
        use_token_ids = "token_ids" in pf.schema.names
        columns = ["token_ids"] if use_token_ids else ["text"]
        for batch in pf.iter_batches(batch_size=1000, columns=columns):
            items = batch["token_ids"] if use_token_ids else batch["text"]
            for item in items:
                if use_token_ids:
                    token_list = item.as_py()
                    if not token_list:
                        continue
                    arr = np.asarray(token_list, dtype=tokenizer.dtype)
                else:
                    text = item.as_py()
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

    num_workers = min(num_workers, len(parquet_files))
    chunks = [parquet_files[i::num_workers] for i in range(num_workers)]

    print(f"Tokenizing {len(parquet_files)} files to {output_dir} with {num_workers} workers...")

    worker_args = [(i, chunks[i], output_dir) for i in range(num_workers)]
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(tokenize_worker, worker_args), total=num_workers):
            results.append(result)

    all_shards = []
    total_tokens = 0
    total_docs = 0

    for r in results:
        all_shards.extend(r["shard_paths"])
        total_tokens += r["total_tokens"]
        total_docs += r["total_docs"]

    enc = load_encoding()
    meta = {
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "num_shards": len(all_shards),
        "shard_size": SHARD_SIZE,
        "dtype": get_dtype_str(enc.n_vocab),
        "vocab_size": enc.n_vocab,
        "shards": sorted([Path(p).name for p in all_shards]),
    }

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! {total_tokens:,} tokens in {len(all_shards)} shards.")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build filler pool
    filler_parser = subparsers.add_parser("build-filler", help="Build pre-tokenized filler pool")
    filler_parser.add_argument("--fineweb_dir", type=str, default="~/.cache/fineweb-edu")
    filler_parser.add_argument("--output", type=str, default="data/filler_pool.parquet")
    filler_parser.add_argument("-n", "--num_examples", type=int, default=50000, help="Number of docs to include")
    filler_parser.add_argument("--min_tokens", type=int, default=50)

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    gen_parser.add_argument("--config", type=str, default="configs/data_generation.yaml")
    gen_parser.add_argument("--output_dir", type=str, default=None)
    gen_parser.add_argument("-n", "--num_examples", type=int, default=None, help="Number of training examples")
    gen_parser.add_argument("--eval_per_task", type=int, default=None)
    gen_parser.add_argument("--num_workers", type=int, default=1)
    gen_parser.add_argument("--batch_size", type=int, default=10_000, help="Examples per worker batch (lower = less RAM)")


    # Tokenize
    tok_parser = subparsers.add_parser("tokenize", help="Tokenize parquet to binary")
    tok_parser.add_argument("--input_dir", type=str, default=None)
    tok_parser.add_argument("--output_dir", type=str, default="data/algorithmic_bin")
    tok_parser.add_argument("--workers", type=int, default=mp.cpu_count())

    args = parser.parse_args()

    if args.command == "build-filler":
        main_config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        if main_config_path.exists():
            with open(main_config_path, encoding="utf-8") as f:
                shard_range = tuple(yaml.safe_load(f).get("data", {}).get("fineweb", {}).get("filler_shards", list(DEFAULT_FILLER_SHARDS)))
        else:
            shard_range = DEFAULT_FILLER_SHARDS
        build_filler_pool(
            fineweb_dir=Path(args.fineweb_dir),
            shard_range=shard_range,
            output_path=Path(args.output),
            min_tokens=args.min_tokens,
            max_docs=args.num_examples,
        )
        return

    if args.command == "tokenize":
        input_dir = Path(args.input_dir or CACHE_DIR)
        output_base = Path(args.output_dir)
        train_files = sorted(list(input_dir.glob("train_*.parquet")))
        if (input_dir / "train.parquet").exists():
            train_files.append(input_dir / "train.parquet")
        eval_file = input_dir / "eval.parquet"

        if train_files:
            print(f"Tokenizing TRAIN ({len(train_files)} files)...")
            tokenize_dataset(train_files, output_base / "train", args.workers)
        if eval_file.exists():
            print("Tokenizing EVAL...")
            tokenize_dataset([eval_file], output_base / "eval", args.workers)
        if not train_files and not eval_file.exists():
            parser.error(f"No train/eval parquet files found in {input_dir}")
        return

    # Generate command
    config = load_config(args.config)
    gen_cfg = config["generation"]
    filler_cfg = config.get("filler", {})

    if not config["tasks"]:
        parser.error("No tasks in config")

    # Context length for passkey/copy generation
    config["context_len"] = filler_cfg.get("context_len", 1024)
    config["eval_context_len"] = filler_cfg.get("eval_context_len", 2048)
    config["L"] = config["context_len"]  # Reference length (1024)

    # Load filler pool if needed
    needs_filler = any(t in config["tasks"] for t in ("passkey", "copy_distance"))
    if needs_filler:
        pool_path = Path(filler_cfg.get("pool_path", "data/filler_pool.parquet"))
        if not pool_path.exists():
            parser.error(f"Filler pool not found: {pool_path}. Run 'build-filler' first.")
        if args.num_workers > 1:
            config["filler_pool_path"] = str(pool_path)
        else:
            config["filler_pool"] = FillerPool(pool_path)

    output_dir = Path(args.output_dir or gen_cfg.get("output_dir") or CACHE_DIR)
    num_examples = args.num_examples or gen_cfg["num_train_examples"]
    eval_per_task = args.eval_per_task or gen_cfg["num_eval_per_task"]
    shard_size = gen_cfg.get("shard_size", 250000)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate train
    print(f"\n{'='*50}\nTRAIN: {num_examples:,} examples (context={config['context_len']})\n{'='*50}")
    if args.num_workers > 1:
        chunk_gen = generate_train_dataset_parallel(num_examples, config, gen_cfg["train_seed"], args.num_workers, shard_size, args.batch_size)
    else:
        chunk_gen = generate_train_dataset(num_examples, config, gen_cfg["train_seed"], shard_size, args.batch_size)
    for i, chunk in enumerate(chunk_gen):
        pd.DataFrame(chunk).to_parquet(output_dir / f"train_{i:04d}.parquet", index=False)

    # Generate eval
    if needs_filler and "filler_pool" not in config:
        config["filler_pool"] = FillerPool(Path(config["filler_pool_path"]))

    print(f"\n{'='*50}\nEVAL: {eval_per_task} per task\n{'='*50}")
    eval_data = generate_eval_dataset(eval_per_task, config, gen_cfg["eval_seed"])
    pd.DataFrame(eval_data).to_parquet(output_dir / "eval.parquet", index=False)
    print(f"\nDone! Files in {output_dir}/")

if __name__ == "__main__":
    main()
