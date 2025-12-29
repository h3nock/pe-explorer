import torch 
from src.encodings.base import PositionalEncoding 

class BinaryPE(PositionalEncoding): 
    """Position as binary, zero-padded to d_model.
    
    example: 
        - if normalize == False:
            pos=7, d_model=8 → [0,0,0,0,0,1,1,1]
        - else: 
            pos=7, d_model=8 -> [-1,-1,-1,-1,-1,1,1]
    """
    
    def __init__(self, d_model: int, max_seq_len: int, normalize: bool = False, **kwargs): 
        super().__init__(d_model, max_seq_len) 
        self.normalize = normalize
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
        pe = torch.zeros(max_len, d_model)
        
        for pos in range(max_len):
            # pos=7 → "00000111" 
            binary = format(pos, f'0{d_model}b')
            for i, bit in enumerate(binary):
                pe[pos, i] = float(bit)
        
        if self.normalize:
            pe = pe * 2 - 1
        
        return pe.unsqueeze(0)
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor: 
        return self.pe[:, :seq_len, :].to(device)
