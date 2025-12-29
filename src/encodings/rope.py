"""Rotary Position Embedding (RoPE)."""
import torch
from src.encodings.base import PositionalEncoding
from typing import Callable, Tuple


class RoPE(PositionalEncoding):
    """Rotary Position Embedding.
    
    Args:
        d_model: model dimension
        max_seq_len: maximum sequence length to pre-compute
        n_heads: number of attention heads (used to compute d_head)
        theta: base for frequency computation (default: 10000.0)
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        n_heads: int,
        theta: float = 10000.0,
    ):
        super().__init__(d_model, max_seq_len)
        self.d_head = d_model // n_heads
        assert self.d_head % 2 == 0, f"d_head ({self.d_head}) must be even for RoPE rotation"
        self.theta = theta
        
        # use float32 for numerical stability
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for positions [0, seq_len).
        
        cache shape: (seq_len, d_head) for efficient broadcasting
        """
        # pos indices: (seq_len, 1)
        pos = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        
        freqs = torch.outer(pos, self.inv_freq)
        
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # cache as buffers (non-persistent to avoid checkpoint bloat)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def _extend_cache(self, new_seq_len: int) -> None:
        """Incrementally extend cache to new_seq_len without full rebuild."""
        old_len = len(self.cos_cached)
        
        # compute only new positions
        pos = torch.arange(old_len, new_seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # append to existing cache
        self.cos_cached = torch.cat([self.cos_cached, emb.cos()], dim=0)
        self.sin_cached = torch.cat([self.sin_cached, emb.sin()], dim=0)
    
    @property
    def adds_to_embedding(self) -> bool:
        return False
    
    @property
    def modifies_attention(self) -> bool:
        return True
    
    @property
    def requires_embedding_scaling(self) -> bool:
        return False
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return zeros since RoPE doesn't add to embeddings.
        
        This is for interface compatibility with additive PEs.
        """
        return torch.zeros(1, seq_len, self.d_model, device=device)
    
    def get_rope_fn(self) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Return the RoPE application function.
        
        Returns:
            callable that takes (q, k) and returns rotated (q, k).
        """
        def apply_rope(
            q: torch.Tensor,
            k: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Apply rotary position embeddings to query and key tensors.
            
            Args:
                q: Query tensor (batch, n_heads, seq_len, d_head)
                k: Key tensor (batch, n_heads, seq_len, d_head)
            
            Returns:
                Tuple of rotated (q, k) with same shapes
            """
            seq_len = q.shape[2]
            
            # dynamically extend cache if needed
            if seq_len > len(self.cos_cached):
                # grow to at least seq_len, or double cache size 
                new_max = max(seq_len, len(self.cos_cached) * 2)
                self._extend_cache(new_max)
            
            # shape: (seq_len, d_head) -> (1, 1, seq_len, d_head) for broadcasting
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            
            cos = cos.to(q.dtype)
            sin = sin.to(q.dtype)
            
            # apply rotation using the rotate_half
            q_rot = (q * cos) + (_rotate_half(q) * sin)
            k_rot = (k * cos) + (_rotate_half(k) * sin)
            
            return q_rot, k_rot
        
        return apply_rope


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input.
    
    for input [..., d], splits into [..., d//2] halves and returns
    [-x2, x1] concatenated. this implements the rotation matrix
    multiplication efficiently.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)