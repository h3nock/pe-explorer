"""Tokenization and sharding utilities.

Loads a nanochat-trained tiktoken Encoding from a pickle file.
"""
import pickle
from pathlib import Path

import numpy as np
import tiktoken

DEFAULT_TOKENIZER_PATH = Path(__file__).parent.parent.parent / "data" / "tokenizer.pkl"

SPECIAL_TOKENS = {
    "<|bos|>",           # document delimiter
    "<|user_start|>",    # user messages
    "<|user_end|>",
    "<|assistant_start|>",  # assistant messages
    "<|assistant_end|>",
    "<|python_start|>",  # python REPL tool
    "<|python_end|>",
    "<|output_start|>",  # python output
    "<|output_end|>",
}

BOS_TOKEN = "<|bos|>"  # default delimiter for pretraining

_enc: tiktoken.Encoding | None = None


def load_encoding(path: Path | None = None) -> tiktoken.Encoding:
    """Load and cache tiktoken.Encoding from pickle file."""
    global _enc
    if _enc is None:
        pkl_path = path or DEFAULT_TOKENIZER_PATH
        with open(pkl_path, "rb") as f:
            enc = pickle.load(f)
        if not isinstance(enc, tiktoken.Encoding):
            raise TypeError(f"Expected tiktoken.Encoding, got {type(enc)}")
        _enc = enc
    return _enc


def get_bos_token_id(enc: tiktoken.Encoding) -> int | None:
    """Get BOS token ID, or None if not defined."""
    try:
        return enc.encode_single_token(BOS_TOKEN)
    except KeyError:
        return None


def get_dtype_str(vocab_size: int) -> str:
    """Get dtype string based on vocab size ('uint16' or 'uint32')."""
    return "uint16" if vocab_size <= 65536 else "uint32"


class Tokenizer:
    """Tokenizer wrapper that encodes text and handles special token delimiters."""

    def __init__(self, tokenizer_path: Path | None = None):
        self.enc = load_encoding(tokenizer_path)
        self.bos = get_bos_token_id(self.enc)
        if self.bos is None:
            raise ValueError(f"Tokenizer missing {BOS_TOKEN} special token")
        self.vocab_size = self.enc.n_vocab
        self.dtype = np.uint16 if self.vocab_size <= 65536 else np.uint32

    def encode(self, text: str | None, prepend: str | None = BOS_TOKEN, append: str | None = None) -> np.ndarray | None:
        """Encode text with optional prepend/append of delimiter tokens.
        
        Args:
            text: Text to encode
            prepend: Special token to prepend (default: BOS_TOKEN).
            append: Special token to append (default: None).
        """
        if not text:
            return None
        tokens = self.enc.encode(text, allowed_special=SPECIAL_TOKENS)
        
        if prepend:
            try:
                pid = self.enc.encode_single_token(prepend)
                tokens = [pid] + tokens
            except KeyError:
                pass
            
        if append:
            try:
                aid = self.enc.encode_single_token(append)
                # avoid duplication if text already ends with this token
                if not tokens or tokens[-1] != aid:
                    tokens.append(aid)
            except KeyError:
                pass
                
        return np.asarray(tokens, dtype=self.dtype)


class ShardWriter:
    """Writes token arrays to sharded binary files."""

    def __init__(self, output_dir: Path, file_prefix: str, shard_size: int):
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.shard_size = shard_size
        self.shard_idx = 0
        self.shard_tokens = 0
        self.shard_paths: list[str] = []
        self._file = None

    def _open_new_shard(self):
        if self._file:
            self._file.close()
        path = self.output_dir / f"{self.file_prefix}_{self.shard_idx:04d}.bin"
        self.shard_paths.append(str(path))
        self._file = open(path, "wb")
        self.shard_idx += 1
        self.shard_tokens = 0

    def write(self, arr: np.ndarray):
        """Write token array to shard(s), splitting at boundaries if needed."""
        pos = 0
        while pos < len(arr):
            if self._file is None:
                self._open_new_shard()

            space_left = self.shard_size - self.shard_tokens
            chunk_size = min(len(arr) - pos, space_left)

            arr[pos : pos + chunk_size].tofile(self._file)
            self.shard_tokens += chunk_size
            pos += chunk_size

            if self.shard_tokens >= self.shard_size:
                self._file.close()
                self._file = None

    def close(self):
        """Close any open file handles."""
        if self._file:
            self._file.close()
            self._file = None
