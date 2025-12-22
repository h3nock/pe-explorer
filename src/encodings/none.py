import torch 
from src.encodings.base import PositionalEncoding 

class NoPE(PositionalEncoding): 
    """No positional encoding.""" 
    @property 
    def adds_to_embedding(self) -> bool: 
        return False 
    
    @property 
    def modifies_attention(self) -> bool: 
        return False 
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor: 
        return torch.zeros(1, seq_len, self.d_model, device=device)
    