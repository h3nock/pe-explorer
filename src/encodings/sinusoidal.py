import math 
import torch 
from src.encodings.base import PositionalEncoding 

class SinusoidalPE(PositionalEncoding): 
    """Sinusoidal positional encoding.""" 
    def __init__(self, d_model: int, max_seq_len: int, base: int = 10000, **kwargs): 
        super().__init__(d_model, max_seq_len) 
        self.base = base 
        pe = self._create_pe_matrix(max_seq_len, d_model)
        self.register_buffer("pe",pe)
    
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
        """Create a matrix of positional encodings.""" 
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(self.base)/d_model)) # (d_model//2, ) 
        pe[:,0::2] = torch.sin(pos * div_term) # even dimensions (max_len, d_model//2)
        pe[:,1::2] = torch.cos(pos * div_term) # odd dimensions (max_len, d_model//2)
        return pe.unsqueeze(0) # (1, max_len, d_model) 
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor: 
        return self.pe[:, : seq_len, :].to(device) # (1, seq_len, d_model)