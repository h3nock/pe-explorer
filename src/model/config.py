from dataclasses import dataclass, field 

@dataclass
class ModelConfig:
    # architecture
    d_model: int = 512 
    n_layers: int = 8 
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 1024
    vocab_size:int = 50257 # gpt2 tokenizer vocab size

    # pos encoding
    pe_type: str = "sinusoidal" # type of pos encoding 
    pe_params: dict = field(default_factory=dict)

    # training
    dropout: float = 0.1
    tie_embedding: bool = True # tie input and output embeddings 