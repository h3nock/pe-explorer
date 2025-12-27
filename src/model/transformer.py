import math
import torch 
import torch.nn as nn 
from typing import Optional 

from src.model.config import ModelConfig
from src.model.block import TransformerBlock
from src.encodings import get_pe

class Transformer(nn.Module):
    """Decoder-only Transformer language model.""" 
    def __init__(self, config: ModelConfig): 
        super().__init__() 
        self.config = config 

        # token embeddings 
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model) 

        # positional encoding
        self.pe = get_pe(config.pe_type, config.d_model, config.max_seq_len, **config.pe_params)  

        # initialize embeddings 
        self.token_embedding.weight.data.normal_(mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])

        # final layer norm 
        self.norm = nn.LayerNorm(config.d_model) 

        # lm head (ouput projection) 
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # weight tying
        if config.tie_embedding: 
            self.lm_head.weight = self.token_embedding.weight 
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        """
        Args: 
            x: (batch_size, seq_len) input tokens 
        Returns: 
            (batch_size, seq_len, vocab_size) logits 
        """
        # token embeddings 
        x = self.token_embedding(x) 
        
        # scale embeddings if required by PE type (e.g. Sinusoidal)
        if self.pe.requires_embedding_scaling:
            x = x * math.sqrt(self.config.d_model) 

        # add positional encoding (if this PE type adds to embeddings)
        if self.pe.adds_to_embedding: 
            x = x + self.pe(x.shape[1], x.device) 

        # get RoPE fn if using RoPE (returns None for non-RoPE types)
        rope_fn = self.pe.get_rope_fn() 

        for block in self.blocks:
            x = block(x, rope_fn=rope_fn)

        # final norm + projection to vocab  
        x = self.norm(x) 
        logits = self.lm_head(x) 
        return logits

            
         

