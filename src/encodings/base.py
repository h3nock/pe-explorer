import torch 
import torch.nn as nn 
from abc import ABC, abstractmethod 

class PositionalEncoding(ABC, nn.Module):
    """Abstract base class for positional encodings."""
    def __init__(self, d_model: int, max_seq_len: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len 
    
    @property 
    @abstractmethod
    def adds_to_embedding(self) -> bool:
        """True if the PE is added to the token embeddings"""
        pass 

    @property 
    @abstractmethod
    def modifies_attention(self) -> bool: 
        """True if the PE modifies the attention mechanism"""
        pass 

    @property
    def requires_embedding_scaling(self) -> bool:
        """Whether to scale token embeddings by sqrt(d_model) before adding PE.
        
        required for fixed additive encodings (Sinusoidal, Binary, etc.) where
        PE magnitude >> initialized embedding magnitude.
        """
        return False 

    @abstractmethod 
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor: 
        """Generate positional encoding for absolute PEs.
        
        Returns: (1, seq_len, d_model) tensor to add to token embeddings
        """
        pass 

    def get_rope_fn(self): 
        """Returns a function(q, k) -> (q_rope, k_rope) or None if not using RoPE.""" 
        return None 
        