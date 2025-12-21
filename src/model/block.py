import torch 
import torch.nn as nn 
from typing import Optional, Callable 

from src.model.attention import MultiHeadAttention 
from src.model.mlp import MLP 

class TransformerBlock(nn.Module):
    """Single transformer block with pre-layer norm""" 
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float=0.0):
        super().__init__()  
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)   
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, rope_fn: Optional[Callable] = None) -> torch.Tensor:
        """Forward pass of the transformer block."""

        # pre-norm: normalize before sublayer, then residual connection 
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask, rope_fn=rope_fn))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x