import torch 
from src.encodings.base import PositionalEncoding 

class DecimalPE(PositionalEncoding): 
    """Position as decimal digits, zero-padded.
    
    example: 
        - if normalize:
            pos=42, d_model=8 → [-1,-1,-1,-1,-1,-1,-0.11, -0.56]
        - else:
            pos=42, d_model=8 → [0,0,0,0,0,0,4,2]
    """
    
    def __init__(self, d_model: int, max_seq_len: int, normalize: bool = True): 
        super().__init__(d_model, max_seq_len) 
        self.normalize = normalize
        pe = self._create_pe_matrix(max_seq_len, d_model)
        self.register_buffer("pe", pe)
    
    @property 
    def adds_to_embedding(self) -> bool: 
        return True 
    
    @property 
    def modifies_attention(self) -> bool: 
        return False 
    
    def _create_pe_matrix(self, max_len: int, d_model: int) -> torch.Tensor: 
        pe = torch.zeros(max_len, d_model)
        
        for pos in range(max_len):
            # pos=42 → "00000042" (zeros on left, digits on right)
            decimal = str(pos).zfill(d_model)
            for i, digit in enumerate(decimal):
                pe[pos, i] = float(digit)
        
        if self.normalize:
            pe = (pe - 4.5) / 4.5  # scales [0,9] roughly to [-1,1]
        
        return pe.unsqueeze(0)
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor: 
        return self.pe[:, :seq_len, :].to(device)
