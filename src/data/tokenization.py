"""Tokenization and sharding utilities."""
import numpy as np
from pathlib import Path
from typing import BinaryIO
import tiktoken

class Tokenizer:
    """Optimized tokenizer wrapper for binary dataset creation.
    
    Handles:
    - Efficient dtype selection (uint16/uint32)
    - EOT token appending
    - Fast numpy array conversion
    """
    def __init__(self, model_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(model_name)
        self.eot = self.enc.eot_token
        # Determine smallest sufficient dtype
        self.dtype = np.uint16 if self.enc.n_vocab <= (np.iinfo(np.uint16).max + 1) else np.uint32
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str | None) -> np.ndarray | None:
        """Encode text to numpy array with appended EOT token.
        
        Returns None if text is None/empty to signal skipping.
        """
        if not text:
            return None
            
        tokens = self.enc.encode(text, allowed_special={'<|endoftext|>'})
        # only append EOT if not already present to avoid double-EOT
        if not tokens or tokens[-1] != self.eot:
            tokens.append(self.eot)
        return np.asarray(tokens, dtype=self.dtype)

class ShardWriter:
    """Writes token arrays to sharded binary files.
    
    Args:
        output_dir: Directory to write shards to
        file_prefix: Prefix for shard filenames (e.g., "train_w01")
        shard_size: Maximum tokens per shard
    """
    def __init__(self, output_dir: Path, file_prefix: str, shard_size: int):
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.shard_size = shard_size
        
        self.shard_idx = 0
        self.shard_tokens = 0      # tokens written to current shard
        self.shard_paths: list[str] = []
        self._current_file: BinaryIO | None = None
        
    def _open_new_shard(self):
        """Open a new binary shard file."""
        if self._current_file:
            self._current_file.close()
            
        fname = f"{self.file_prefix}_{self.shard_idx:04d}.bin"
        path = self.output_dir / fname
        self.shard_paths.append(str(path))
        
        self._current_file = open(path, "wb")
        self.shard_idx += 1
        self.shard_tokens = 0
        
    def write(self, arr: np.ndarray):
        """Write token array to shard(s), splitting at boundaries if needed."""
        pos = 0
        while pos < len(arr):
            if self._current_file is None:
                self._open_new_shard()
            
            space_left = self.shard_size - self.shard_tokens
            to_write = min(len(arr) - pos, space_left)
            
            # write chunk to current file
            arr[pos : pos + to_write].tofile(self._current_file)
            
            self.shard_tokens += to_write
            pos += to_write
            
            # if filled current shard, close it (next write opens new one)
            if self.shard_tokens >= self.shard_size:
                self._current_file.close()
                self._current_file = None

    def close(self):
        """Close any open file handles."""
        if self._current_file:
            self._current_file.close()
            self._current_file = None
