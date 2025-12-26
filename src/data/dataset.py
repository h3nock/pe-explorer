"""Pre-tokenized binary dataset for fast LLM training.

Supports sharded binary files for multi-GPU training without file locking.
"""
import json
from pathlib import Path
import os
import resource
import bisect

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


DEFAULT_TOKENIZED_DIR = Path.home() / ".cache" / "fineweb-edu" / "tokenized"

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
    
    Assigns indices to ranks using a strided pattern (0, 0+W, 0+2W, ...) or contiguous.
    Resume is handled by skipping the first `samples_seen` samples globally.
    O(1) memory usage via streaming iterator.
    """
    def __init__(self, total_samples: int, rank: int, world_size: int, 
                 samples_seen: int = 0, shuffle: bool = False, seed: int = 42):
        self.total_samples = total_samples
        self.rank = rank
        self.world_size = world_size
        self.samples_seen = samples_seen
        self.shuffle = shuffle
        if self.shuffle:
            # For large datasets, shuffling requires careful O(N) index mapping.
            # For now, we enforce canonical order for determinism and O(1) streaming.
            raise NotImplementedError("Streaming shuffle not implemented. Use shuffle=False.")
        self.seed = seed
        
    def __iter__(self):
        # Calculate start offset for this rank
        # We want the first index `i >= samples_seen` where `i % world_size == rank`
        
        # 1. Align samples_seen to the next sample belonging to this rank
        remaining_offset = (self.rank - self.samples_seen) % self.world_size
        start_index = self.samples_seen + remaining_offset
        
        # 2. Yield indices with stride = world_size
        # 2. Yield indices with stride = world_size
        if start_index < self.total_samples:
            yield from range(start_index, self.total_samples, self.world_size)

    def __len__(self):
        # Remaining global samples after resume
        remaining = max(0, self.total_samples - self.samples_seen)
        # Exact length of range(start, stop, step) logic
        # First sample for this rank:
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
        
        # Get dtype from metadata (Llama 3 uses uint32 for 128k vocab)
        dtype_str = self.meta.get("dtype", "uint16")
        self.dtype = np.uint16 if dtype_str == "uint16" else np.uint32
        
        # Resolve shard paths relative to meta.json directory
        # Supports both legacy absolute paths (by taking .name) and new relative paths
        shard_paths = [self.data_dir / Path(p).name for p in self.meta["shards"]]
        
        # Check ulimit against shard count
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if len(shard_paths) > soft_limit - 100:  # buffer for other files
            print(f"WARNING: Dataset has {len(shard_paths)} shards but ulimit -n is {soft_limit}.")
            print("         Consider raising ulimit (ulimit -n 65535) or using fewer shards.")

        # Load ALL shards globally (DDP splitting handled by Sampler)
        self.datasets = [ShardDataset(p, seq_len, self.dtype) for p in shard_paths]
        
        # Calculate total samples
        self.total_samples = sum(len(d) for d in self.datasets)
        
        # apply token budget (global limit)
        # note: token_budget is "training tokens" (input tokens).
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
    seq_len: int,
    batch_size: int,
    data_dir: str | Path | None = None,
    data_config: dict | None = None,
    token_budget: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    samples_seen: int = 0,
    seed: int = 42,
    num_workers: int = 4,
) -> DataLoader:
    """Create DataLoader for sharded pre-tokenized data.
    
    Supports:
    1. Single dataset (via data_dir)
    2. Mixed dataset (via data_config with 'mix_ratio')
    """
    if data_config and "mix_ratio" in data_config:
        # mixed dataset mode
        ratio = data_config["mix_ratio"] # e.g. {"fineweb": 0.9, "algorithmic": 0.1}
        
        # we need integer weights for deterministic interleaving.
        # and use a large multiplier to handle small percentages (e.g. 0.5% = 0.005)
        multiplier = 1000
        
        keys = list(ratio.keys())
        
        final_weights = {}
        processed_datasets = []
        
        for k in keys:
            w = int(ratio[k] * multiplier)
            if w == 0 and ratio[k] > 0:
                raise ValueError(f"Ratio {ratio[k]} for {k} is too small for multiplier {multiplier}.")
            final_weights[k] = w
            
            sub_cfg = data_config.get(k, {}) if data_config else {}
            path_str = sub_cfg.get("tokenized_path")
            
            if path_str:
                d_dir = Path(path_str).expanduser()
            elif k == "fineweb":
                d_dir = DEFAULT_TOKENIZED_DIR
            else:
                raise ValueError(f"Missing 'tokenized_path' for dataset '{k}' in config.")

            if not d_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found for '{k}': {d_dir}")

            ds = MemmapDataset(data_dir=d_dir, seq_len=seq_len, token_budget=None)
            processed_datasets.append(ds)
            
        # create interleaved
        weights = [final_weights[k] for k in keys]
        dataset = InterleavedDataset(processed_datasets, weights)
        
        if token_budget:
            budget_samples = token_budget // seq_len
            if len(dataset) < budget_samples:
                raise ValueError(
                    f"Requested {budget_samples:,} samples ({token_budget} tokens) "
                    f"but interleaved dataset only has {len(dataset):,} samples. "
                )
            
            # enforce budget exact match (truncate)
            if len(dataset) > budget_samples:
                dataset = torch.utils.data.Subset(dataset, range(budget_samples))

    else:
        # fallback to single dataset
        dataset = MemmapDataset(
            data_dir=data_dir,
            seq_len=seq_len,
            token_budget=token_budget,
        )
    
    # Sampler handles global -> rank splitting and resume offset
    sampler = DeterministicSampler(
        total_samples=len(dataset),
        rank=rank,
        world_size=world_size,
        samples_seen=samples_seen,
        seed=seed,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
