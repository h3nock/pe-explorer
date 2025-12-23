"""Positional encoding registry: maps pe_type to PE class."""
from functools import partial
from typing import Callable, Dict, Type, Union

from src.encodings.base import PositionalEncoding
from src.encodings.none import NoPE
from src.encodings.sinusoidal import SinusoidalPE
from src.encodings.binary import BinaryPE
from src.encodings.decimal import DecimalPE


PE_REGISTRY: Dict[str, Union[Type[PositionalEncoding], Callable[..., PositionalEncoding]]] = {
    "none": NoPE,
    "sinusoidal": SinusoidalPE,
    "binary": partial(BinaryPE, normalize=False),
    "binary_norm": partial(BinaryPE, normalize=True),
    "decimal": partial(DecimalPE, normalize=False),
    "decimal_norm": partial(DecimalPE, normalize=True),
}


def get_pe(pe_type: str, d_model: int, max_seq_len: int, **kwargs) -> PositionalEncoding:
    """Create a positional encoding instance.
    
    Args:
        pe_type: type of positional encoding ("sinusoidal", "none", ...)
        d_model: model dimension
        max_seq_len: maximum sequence length
        **kwargs: additional PE specific parameters
    
    Returns:
        PositionalEncoding instance
    """
    if pe_type not in PE_REGISTRY:
        raise ValueError(f"Unknown PE type: {pe_type}. Available: {list(PE_REGISTRY.keys())}")
    
    pe_class = PE_REGISTRY[pe_type]
    return pe_class(d_model=d_model, max_seq_len=max_seq_len, **kwargs)
