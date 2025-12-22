import torch 
from torch.utils.data import IterableDataset, DataLoader 
import tiktoken 
from datasets import load_dataset

class FineWebEduDataset(IterableDataset):   
    def __init__(self, seq_len: int, split: str = "train", ): 
        self.seq_len = seq_len 
        self.split = split 
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def __iter__(self):
        dataset = load_dataset("HuggingFaceFW/FineWeb-Edu", name="sample-10BT", split=self.split, streaming=True)

        buffer = []
        for example in dataset:
            tokens = self.tokenizer.encode(example["text"])
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1: 
                chunk = buffer[:self.seq_len + 1] # +1 for target 
                buffer = buffer[self.seq_len + 1:] # keep remainder

                x = torch.tensor(chunk[:-1], dtype=torch.long) # input 
                y = torch.tensor(chunk[1:], dtype=torch.long) # target (shifted by 1)
                yield x, y
    
def get_dataloader(seq_len: int, batch_size: int, num_workers: int = 4) -> DataLoader:
    """Create a DataLoader for FineWeb-Edu.""" 
    dataset = FineWebEduDataset(seq_len=seq_len) 
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True 
    )
        