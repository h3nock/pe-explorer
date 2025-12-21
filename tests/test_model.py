"""Simple tests for model components."""
import torch
from src.model.attention import MultiHeadAttention
from src.model.mlp import MLP
from src.model.block import TransformerBlock
from src.model.transformer import Transformer
from src.model.config import ModelConfig

# test config
BATCH_SIZE = 2
SEQ_LEN = 10
D_MODEL = 512
N_HEADS = 8
D_FF = 2048
VOCAB_SIZE = 1000  # small for testing
N_LAYERS = 2


def test_attention():
    """Test MultiHeadAttention module."""
    print("Testing MultiHeadAttention...")
    
    attn = MultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    out = attn(x)
    
    # shape test
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"Input: {x.shape}, Output: {out.shape} - Shape test passed!")
    
    # gradient test
    out.sum().backward()
    assert attn.q_proj.weight.grad is not None, "No gradient for q_proj"
    print("Gradient test passed!")
    
    # causal mask test
    x1 = torch.randn(1, 5, D_MODEL)
    x2 = x1.clone()
    x2[0, 1:, :] = torch.randn(4, D_MODEL)
    
    attn.eval()
    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)
    
    assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5), "Causal mask not working!"
    print("Causal mask test passed!")
    print("All attention tests passed!\n")


def test_mlp():
    """Test MLP module."""
    print("Testing MLP...")
    
    mlp = MLP(d_model=D_MODEL, d_ff=D_FF)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    out = mlp(x)
    
    # shape test
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"Input: {x.shape}, Output: {out.shape} - Shape test passed!")
    
    # gradient test
    out.sum().backward()
    assert mlp.fc1.weight.grad is not None, "No gradient for fc1"
    print("Gradient test passed!")
    print("All MLP tests passed!\n")


def test_transformer_block():
    """Test TransformerBlock module."""
    print("Testing TransformerBlock...")
    
    block = TransformerBlock(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    out = block(x)
    
    # shape test
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"Input: {x.shape}, Output: {out.shape} - Shape test passed!")
    
    # gradient test
    out.sum().backward()
    assert block.attn.q_proj.weight.grad is not None, "No gradient for attention"
    assert block.mlp.fc1.weight.grad is not None, "No gradient for mlp"
    print("Gradient test passed!")
    print("All TransformerBlock tests passed!\n")


def test_transformer():
    """Test full Transformer model."""
    print("Testing Transformer...")
    
    config = ModelConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        tie_embedding=True,
    )
    
    model = Transformer(config)
    
    # input is token IDs
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    logits = model(x)
    
    # shape test
    expected_shape = (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert logits.shape == expected_shape, f"Shape mismatch: {logits.shape} != {expected_shape}"
    print(f"Input: {x.shape}, Output: {logits.shape} - Shape test passed!")
    
    # weight tying test
    assert model.lm_head.weight is model.token_embedding.weight, "Weight tying failed!"
    print("Weight tying test passed!")
    
    # gradient test
    logits.sum().backward()
    assert model.token_embedding.weight.grad is not None, "No gradient for embeddings"
    print("Gradient test passed!")
    print("All Transformer tests passed!\n")


if __name__ == "__main__":
    test_attention()
    test_mlp()
    test_transformer_block()
    test_transformer()

