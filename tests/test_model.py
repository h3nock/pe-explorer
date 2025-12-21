"""Simple tests for model components."""
import torch
from src.model.attention import MultiHeadAttention
from src.model.mlp import MLP

def test_attention():
    """Test MultiHeadAttention module."""
    print("Testing MultiHeadAttention...")
    
    # config
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # create module
    attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    
    # test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # forward pass
    out = attn(x)
    
    # check output shape
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Shape test passed!")
    
    # check gradients flow
    loss = out.sum()
    loss.backward()
    
    # verify gradients exist
    assert attn.q_proj.weight.grad is not None, "No gradient for q_proj"
    assert attn.out_proj.weight.grad is not None, "No gradient for out_proj"
    print("Gradient test passed!")
    
    # check causal masking works
    # token at position 0 should have same output regardless of future tokens
    x1 = torch.randn(1, 5, d_model)
    x2 = x1.clone()
    x2[0, 1:, :] = torch.randn(4, d_model)  # change future tokens
    
    attn.eval()  # disable dropout for deterministic test
    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)
    
    # position 0 output should be identical (it can't see future)
    assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5), "Causal mask not working!"
    print("Causal mask test passed!")
    print("All attention tests passed!\n")


def test_mlp():
    """Test MLP module."""
    print("Testing MLP...")
    
    # config
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048
    
    # create module
    mlp = MLP(d_model=d_model, d_ff=d_ff)
    
    # test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # forward pass
    out = mlp(x)
    
    # check output shape
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Shape test passed!")
    
    # check gradients flow
    loss = out.sum()
    loss.backward()
    
    assert mlp.fc1.weight.grad is not None, "No gradient for fc1"
    assert mlp.fc2.weight.grad is not None, "No gradient for fc2"
    print("Gradient test passed!")
    print("All MLP tests passed!\n")


if __name__ == "__main__":
    test_attention()
    test_mlp()

