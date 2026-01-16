"""Pre-tokenized binary dataset for fast LLM training.

Supports sharded binary files for multi-GPU training without file locking.
"""
import json
from pathlib import Path
import os
import resource
import bisect
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def worker_init_fn(worker_id: int) -> None:
    """Deterministic seeding for DataLoader workers.

    Uses the main process's torch seed (set via Generator) to derive
    reproducible seeds for each worker's RNG state.
    """
    # get base seed from PyTorch (set by DataLoader's generator)
    worker_seed = torch.initial_seed() % 2**32

    # seed all RNGs with worker-specific value
    np.random.seed(worker_seed)
    random.seed(worker_seed)

DEFAULT_TOKENIZED_DIR = Path("data") / "fineweb_bin"

class ShardDataset(Dataset):
    """Memory-mapped dataset for a single binary shard."""
    
    def __init__(self, shard_path: Path, seq_len: int, dtype=np.uint16):
        self.shard_path = shard_path
        self.seq_len = seq_len
        self.dtype = dtype
        self._data = None
        # calculate length without opening file
        file_size = os.path.getsize(shard_path)
        itemsize = np.dtype(dtype).itemsize
        self.num_samples = file_size // (itemsize * (seq_len + 1))
        
    @property
    def data(self):
        """Lazy-load memmap on first access."""
        if self._data is None:
            self._data = np.memmap(self.shard_path, dtype=self.dtype, mode="r")
        return self._data
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = idx * (self.seq_len + 1)
        chunk = self.data[offset : offset + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


class DeterministicSampler(Dataset):
    """Sampler that yields indices deterministically with offset resume support.

    Assigns indices to ranks using a strided pattern (0, 0+W, 0+2W, ...).
    Resume is handled by skipping the first `samples_seen` samples globally.
    """
    def __init__(self, total_samples: int, rank: int, world_size: int, samples_seen: int = 0):
        self.total_samples = total_samples
        self.rank = rank
        self.world_size = world_size
        self.samples_seen = samples_seen

    def __iter__(self):
        # align samples_seen to the next sample belonging to this rank
        remaining_offset = (self.rank - self.samples_seen) % self.world_size
        start_index = self.samples_seen + remaining_offset

        if start_index < self.total_samples:
            yield from range(start_index, self.total_samples, self.world_size)

    def __len__(self):
        offset = (self.rank - self.samples_seen) % self.world_size
        start = self.samples_seen + offset
        if start >= self.total_samples:
            return 0
        return (self.total_samples - start + self.world_size - 1) // self.world_size


class MemmapDataset(Dataset):
    """Dataset that loads from sharded binary files.
    
    Args:
        data_dir: Directory with tokenized shards
        seq_len: Sequence length for training
        token_budget: Maximum tokens to use (None = use all)
    """
    
    def __init__(
        self,
        data_dir: str | Path | None = None,
        seq_len: int = 2048,
        token_budget: int | None = None,
        shards: list[int] | None = None,
    ):
        if data_dir is None:
            # fallback to default cache if not provided
            self.data_dir = DEFAULT_TOKENIZED_DIR
        else:
            self.data_dir = Path(data_dir).expanduser()
            
        self.seq_len = seq_len
        
        # Load metadata
        meta_path = self.data_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Tokenized data not found: {self.data_dir}\n"
                "Run 'python -m src.data.prepare_fineweb download && python -m src.data.prepare_fineweb tokenize' first."
            )
        
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        
        # get dtype from metadata 
        dtype_str = self.meta.get("dtype", "uint16")
        self.dtype = np.uint16 if dtype_str == "uint16" else np.uint32
        
        # resolve shard paths (extract filename to support absolute paths) 
        all_shard_paths = [self.data_dir / Path(p).name for p in self.meta["shards"]]

        if shards is not None:
            if any(idx < 0 or idx >= len(all_shard_paths) for idx in shards):
                raise ValueError(f"Shard index out of range (total {len(all_shard_paths)})")
            shard_paths = [all_shard_paths[i] for i in shards]
        else:
            shard_paths = all_shard_paths

        # Check ulimit against shard count
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if len(shard_paths) > soft_limit - 100:  # buffer for other files
            print(f"WARNING: Dataset has {len(shard_paths)} shards but ulimit -n is {soft_limit}.")
            print("         Consider raising ulimit (ulimit -n 65535) or using fewer shards.")

        # Load ALL shards globally (DDP splitting handled by Sampler)
        self.datasets = [ShardDataset(p, seq_len, self.dtype) for p in shard_paths]
        
        # Calculate total samples
        self.total_samples = sum(len(d) for d in self.datasets)
        
        # cap dataset size to token budget for training tokens
        # we store (seq_len + 1) per sample, but model sees seq_len tokens.
        if token_budget:
            budget_samples = token_budget // seq_len
            self.total_samples = min(self.total_samples, budget_samples)
        
        # build cumulative index for fast lookup
        self.cumsum = []
        running = 0
        for d in self.datasets:
            self.cumsum.append(running)
            running += len(d)
        
        print(f"MemmapDataset: {self.data_dir}")
        print(f"  Shards: {len(shard_paths)}")
        print(f"  Total samples: {self.total_samples:,}")
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Binary search for shard index O(log N)
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range")
            
        shard_idx = bisect.bisect_right(self.cumsum, idx) - 1
        start = self.cumsum[shard_idx]
        return self.datasets[shard_idx][idx - start]


class InterleavedDataset(Dataset):
    """Deterministically mixes multiple datasets with integer weights.
    """
    def __init__(self, datasets: list[Dataset], weights: list[int]):
        assert len(datasets) == len(weights)
        assert all(w > 0 for w in weights)
        self.datasets = datasets
        self.weights = weights
        self.total_weight = sum(weights)
        
        # calculate max possible cycles determined by the most constrained dataset
        self.num_cycles = min(len(d) // w for d, w in zip(datasets, weights))
        self.total_samples = self.num_cycles * self.total_weight
        
        # precompute per-cycle offsets for O(1) random access
        # cum_weights: [w0, w0+w1, w0+w1+w2]
        self.cum_weights = []
        curr = 0
        for w in weights:
            curr += w
            self.cum_weights.append(curr)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        if idx >= self.total_samples:
            raise IndexError("Index out of bounds")
            
        cycle_idx = idx // self.total_weight
        offset_in_cycle = idx % self.total_weight
        
        # find which dataset this offset belongs to
        # e.g., weights [9, 1], offsets 0..8 -> ds0, offset 9 -> ds1
        dataset_idx = bisect.bisect_right(self.cum_weights, offset_in_cycle)
        
        # calculate local index within that dataset
        # local_offset = offset_in_cycle - sum(prev_weights)
        prev_weight_sum = self.cum_weights[dataset_idx-1] if dataset_idx > 0 else 0
        local_offset = offset_in_cycle - prev_weight_sum
        
        local_idx = cycle_idx * self.weights[dataset_idx] + local_offset
        return self.datasets[dataset_idx][local_idx]



def get_dataloader(
    mode: str,
    data_config: dict,
    seq_len: int,
    batch_size: int,
    token_budget: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    samples_seen: int = 0,
    seed: int = 42,
    num_workers: int = 4,
) -> DataLoader:
    """Create DataLoader for training or validation.

    Args:
        mode: "train" or "val"
        data_config: Data section from config.yaml
        token_budget: Max tokens to use (limits dataset size)
    """
    fineweb_cfg = data_config.get("fineweb", {})
    fineweb_path = fineweb_cfg.get("tokenized_path", DEFAULT_TOKENIZED_DIR)

    if mode == "train":

        if "mix_ratio" in data_config:
            # blend multiple sources
            ratio = data_config["mix_ratio"]
            multiplier = 1000

            datasets, weights = [], []
            for key, pct in ratio.items():
                w = int(pct * multiplier)
                if w == 0 and pct > 0:
                    raise ValueError(f"Ratio {pct} for {key} too small")
                weights.append(w)

                if key == "fineweb":
                    ds = MemmapDataset(fineweb_path, seq_len=seq_len)
                else:
                    path = data_config.get(key, {}).get("tokenized_path")
                    if not path:
                        raise ValueError(f"Missing 'tokenized_path' for '{key}'")
                    ds = MemmapDataset(path, seq_len=seq_len)
                datasets.append(ds)

            dataset = InterleavedDataset(datasets, weights)

            if token_budget:
                budget_samples = token_budget // seq_len
                if len(dataset) > budget_samples:
                    dataset = torch.utils.data.Subset(dataset, range(budget_samples))
        else:
            # single source training
            dataset = MemmapDataset(
                fineweb_path, seq_len=seq_len, token_budget=token_budget
            )
    else:
        # use validation data from separate directory
        val_path = fineweb_cfg.get("val_tokenized_path", fineweb_path)
        dataset = MemmapDataset(
            val_path, seq_len=seq_len, token_budget=token_budget
        )

    sampler = DeterministicSampler(
        total_samples=len(dataset),
        rank=rank,
        world_size=world_size,
        samples_seen=samples_seen,
    )

    # create generator with fixed seed for deterministic worker spawning
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
