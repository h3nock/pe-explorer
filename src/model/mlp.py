import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feedforward network.

    Reference: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        # SwiGLU uses 3 projections without bias
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SwiGLU.
        """
        # gate controls information flow, up_proj provides the information
        return self.down_proj(self.dropout(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        ))
