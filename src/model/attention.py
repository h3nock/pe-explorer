import torch 
import torch.nn as nn
import math 
from typing import Optional, Callable

class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        
        # query, key, value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # output projection 
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, rope_fn: Optional[Callable] = None) -> torch.Tensor:
        """Forward pass of the multi-head attention module
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
            rope_fn: Optional function for rotary position embeddings
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """

        batch_size, seq_len, _ = x.shape

        # project inputs 
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape for multi-head 
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2) # (batch_size, n_heads, seq_len, d_head)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # apply rotary position embeddings if provided
        if rope_fn is not None:
            q, k = rope_fn(q, k)
        
        # uses FlashAttention on A100/H100, falls back to math on older GPUs 
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)



        
        
        