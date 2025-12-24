import os
from pathlib import Path
import torch 
from torch.utils.data import IterableDataset, DataLoader 
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
    
def get_dataloader(
    seq_len: int, 
    batch_size: int, 
    variant: str = "10BT",
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create a DataLoader for FineWeb-Edu.
    
    Args:
        seq_len: Sequence length for training
        batch_size: Batch size
        variant: Dataset variant ("10BT" or "100BT")
        num_workers: Number of data loading workers
        rank: Distributed rank
        world_size: Distributed world size
    """ 
    dataset = FineWebEduDataset(seq_len=seq_len, variant=variant, rank=rank, world_size=world_size) 
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True 
    )