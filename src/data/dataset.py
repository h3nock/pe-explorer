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
    """FineWeb-Edu dataset.
    
    If local data exists at data/fineweb-edu-{variant}/, uses local disk.
    Otherwise, streams from HuggingFace Hub.
    """
    def __init__(self, seq_len: int, variant: str = "10BT", split: str = "train"):
        self.seq_len = seq_len 
        self.split = split 
        self.variant = variant
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # check for local data
        self.local_path = LOCAL_DATA_PATHS.get(variant)
        self.use_local = self.local_path and self.local_path.exists()
        
        if self.use_local:
            print(f"Using local dataset: {self.local_path}")

    def __iter__(self):
        # load from local disk or stream from HF
        if self.use_local:
            dataset = load_from_disk(str(self.local_path))
        else:
            variant_name = "sample-10BT" if self.variant == "10BT" else "sample-100BT"
            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu", 
                name=variant_name, 
                split=self.split, 
                streaming=True
            )

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
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader for FineWeb-Edu.
    
    Args:
        seq_len: Sequence length for training
        batch_size: Batch size
        variant: Dataset variant ("10BT" or "100BT")
        num_workers: Number of data loading workers
    """ 
    dataset = FineWebEduDataset(seq_len=seq_len, variant=variant) 
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True 
    )