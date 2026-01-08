import math
import torch
from src.encodings.base import PositionalEncoding


class SinOnlyPE(PositionalEncoding):
    """Sin-only positional encoding with dynamic extension for long contexts."""

    def __init__(self, d_model: int, max_seq_len: int, base: int = 10000, **kwargs):
        super().__init__(d_model, max_seq_len)
        self.base = base
        pe = self._create_pe_matrix(max_seq_len, d_model)
        self.register_buffer("pe", pe)

    @property
    def adds_to_embedding(self) -> bool:
        return True

    @property
    def requires_embedding_scaling(self) -> bool:
        return True

    @property
    def modifies_attention(self) -> bool:
        return False

    def _create_pe_matrix(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create a matrix of sin-only positional encodings."""
        pos = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model).float() * (-math.log(self.base) / d_model)
        )  # (d_model,)
        pe = torch.sin(pos * div_term)  # (max_len, d_model)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _extend_pe(self, seq_len: int) -> None:
        """Extend PE buffer to handle longer sequences."""
        if seq_len <= self.pe.size(1):
            return
        new_max_len = max(seq_len, self.pe.size(1) * 2)
        new_pe = self._create_pe_matrix(new_max_len, self.d_model)
        self.pe = new_pe.to(self.pe.device, dtype=self.pe.dtype)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len)
        return self.pe[:, :seq_len, :].to(device)  # (1, seq_len, d_model)

