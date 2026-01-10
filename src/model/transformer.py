import math
import torch 
import torch.nn as nn 
from typing import Optional 

from src.model.config import ModelConfig
from src.model.block import TransformerBlock
from src.model.rmsnorm import RMSNorm
from src.encodings import get_pe

class Transformer(nn.Module):
    """Decoder-only Transformer language model.""" 
    def __init__(self, config: ModelConfig): 
        super().__init__() 
        self.config = config 

        # token embeddings 
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model) 

        # positional encoding
        self.pe = get_pe(config.pe_type, config.d_model, config.max_seq_len, n_heads=config.n_heads, **config.pe_params)  

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

        # final RMSNorm
        self.norm = RMSNorm(config.d_model)

        # lm head (output projection)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

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

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_token: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Args:
            prompt_tokens: (batch_size, seq_len) input token ids
            max_new_tokens: Maximum number of tokens to generate
            temperature: 0.0 for greedy, > 0 for sampling
            eos_token: Stop generation if this token is produced
            
        Returns:
            (batch_size, seq_len + generated) full sequence including prompt
        """
        tokens = prompt_tokens

        for _ in range(max_new_tokens):
            # use full context - PEs extend dynamically for long-context eval
            context = tokens

            # forward pass
            logits = self(context)
            next_logits = logits[:, -1, :]  # (batch, vocab)

            # sample or greedy
            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Check EOS
            if eos_token is not None and (next_token == eos_token).all():
                break
        
        return tokens

            
         

