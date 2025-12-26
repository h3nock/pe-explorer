import os
import random
from pathlib import Path
import torch 
import pandas as pd
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
import tiktoken 
from datasets import load_dataset, load_from_disk

# default local data paths (auto-detected)
LOCAL_DATA_PATHS = {
    "10BT": Path("data/fineweb-edu-10BT"),
    "100BT": Path("data/fineweb-edu-100BT"),
}

class FineWebEduDataset(IterableDataset):   
    """FineWeb-Edu dataset with DDP support."""
    def __init__(self, seq_len: int, variant: str = "10BT", split: str = "train", rank: int = 0, world_size: int = 1):
        self.seq_len = seq_len 
        self.split = split 
        self.variant = variant
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # check for local data
        self.local_path = LOCAL_DATA_PATHS.get(variant)
        self.use_local = self.local_path and self.local_path.exists()
        
        if self.use_local and rank == 0:
            print(f"Using local dataset: {self.local_path}")

    def __iter__(self):
        # load from local disk or stream from HF
        if self.use_local:
            dataset = load_from_disk(str(self.local_path))
            if "train" in dataset: dataset = dataset["train"] # handle DatasetDict
        else:
            variant_name = "sample-10BT" if self.variant == "10BT" else "sample-100BT"
            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu", 
                name=variant_name, 
                split=self.split, 
                streaming=True
            )

        # distributed Sharding (across GPUs)
        if self.world_size > 1:
            dataset = dataset.shard(num_shards=self.world_size, index=self.rank) 
            
        # worker Sharding (within single GPU's DataLoader)
        # essential if num_workers > 0, otherwise all workers stream identical data
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
             dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id) 

        buffer = []
        for example in dataset:
            tokens = self.tokenizer.encode(example["text"])
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1: 
                chunk = buffer[:self.seq_len + 1]  # +1 for target 
                buffer = buffer[self.seq_len + 1:]  # keep remainder

                x = torch.tensor(chunk[:-1], dtype=torch.long)  # input 
                y = torch.tensor(chunk[1:], dtype=torch.long)  # target (shifted by 1)
                yield x, y


class AlgorithmicDataset(IterableDataset):
    """Infinite stream of algorithmic examples from Parquet file."""
    
    def __init__(self, path: str, seq_len: int, col_name: str = "text", rank: int = 0, seed: int = 42):
        self.path = Path(path)
        self.seq_len = seq_len
        self.col_name = col_name
        self.rank = rank
        self.seed = seed
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        if not self.path.exists():
            raise FileNotFoundError(f"Algorithmic data not found: {self.path}")
        
        if rank == 0:
            print(f"Loading algorithmic data from {self.path}...")
        
        self.df = pd.read_parquet(self.path)
        self.texts = self.df[col_name].tolist()
        
        if rank == 0:
            print(f"Loaded {len(self.texts):,} algorithmic examples")
    
    def __iter__(self):
        # combine training seed + rank + worker id for variety across DDP and workers
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        worker_seed = self.seed + self.rank * 1000 + worker_id
        rng = random.Random(worker_seed)
        
        while True:  # infinite loop
            text = rng.choice(self.texts)
            tokens = self.tokenizer.encode(text)
            
            # pad short sequences by repeating
            needed = self.seq_len + 1
            while len(tokens) < needed:
                tokens = tokens + tokens
            
            chunk = tokens[:needed]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            yield x, y


class MixedIterableDataset(IterableDataset):
    """Interleaves multiple IterableDatasets based on sampling weights."""
    
    def __init__(self, datasets: list, weights: list, seed: int = 42, rank: int = 0):
        self.datasets = datasets
        self.weights = weights
        self.seed = seed
        self.rank = rank
    
    def __iter__(self):
        # combine training seed + rank + worker id for variety across DDP and workers
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        worker_seed = self.seed + self.rank * 1000 + worker_id
        rng = random.Random(worker_seed)
        
        iterators = [iter(d) for d in self.datasets]
        
        while True:
            # sample which dataset to pull from
            idx = rng.choices(range(len(iterators)), weights=self.weights, k=1)[0]
            try:
                yield next(iterators[idx])
            except StopIteration:
                # restart exhausted iterator (algorithmic is infinite, fineweb might end)
                iterators[idx] = iter(self.datasets[idx])
                yield next(iterators[idx])
    
def get_dataloader(
    seq_len: int, 
    batch_size: int, 
    variant: str = "10BT",
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    data_config: dict = None,
    seed: int = 42,
) -> StatefulDataLoader:
    """Create a StatefulDataLoader for training.
    
    If data_config contains mix_ratio, creates a mixed dataset of FineWeb-Edu
    and Algorithmic data. Otherwise, uses FineWeb-Edu only.
    
    Args:
        seq_len: Sequence length for training
        batch_size: Batch size
        variant: Dataset variant ("10BT" or "100BT")
        num_workers: Number of data loading workers
        rank: Distributed rank
        world_size: Distributed world size
        data_config: Optional data configuration with mix_ratio and algorithmic paths
    """ 
    fineweb = FineWebEduDataset(seq_len=seq_len, variant=variant, rank=rank, world_size=world_size)
    
    # check if mixing is enabled
    if data_config and "mix_ratio" in data_config and "algorithmic" in data_config:
        algo_cfg = data_config["algorithmic"]
        algo_path = algo_cfg.get("train_path", "data/algorithmic/train.parquet")
        
        # only create mixed dataset if algorithmic data exists
        if Path(algo_path).exists():
            algo = AlgorithmicDataset(
                path=algo_path,
                seq_len=seq_len,
                col_name=algo_cfg.get("col_name", "text"),
                rank=rank,
                seed=seed,
            )
            
            weights = [
                data_config["mix_ratio"].get("fineweb", 0.9),
                data_config["mix_ratio"].get("algorithmic", 0.1),
            ]
            
            if rank == 0:
                print(f"Mixed dataset: {weights[0]:.0%} FineWeb + {weights[1]:.0%} Algorithmic")
            
            dataset = MixedIterableDataset([fineweb, algo], weights, seed=seed, rank=rank)
        else:
            if rank == 0:
                print(f"Warning: Algorithmic data not found at {algo_path}, using FineWeb only")
            dataset = fineweb
    else:
        dataset = fineweb
    
    return StatefulDataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        timeout=30,
    )