"""Tests for positional encodings."""
import pytest
import torch

from src.encodings import get_pe
from src.encodings.none import NoPE
from src.encodings.sinusoidal import SinusoidalPE
from src.encodings.sinonly import SinOnlyPE
from src.encodings.binary import BinaryPE
from src.encodings.decimal import DecimalPE
from src.encodings.rope import RoPE


class TestNoPE:
    def test_output_shape(self):
        pe = NoPE(d_model=128, max_seq_len=512)
        out = pe(seq_len=32, device=torch.device("cpu"))
        assert out.shape == (1, 32, 128)
    
    def test_output_is_zeros(self):
        pe = NoPE(d_model=64, max_seq_len=256)
        out = pe(seq_len=16, device=torch.device("cpu"))
        assert torch.all(out == 0)


class TestSinusoidalPE:
    def test_output_shape(self):
        pe = SinusoidalPE(d_model=128, max_seq_len=512)
        out = pe(seq_len=32, device=torch.device("cpu"))
        assert out.shape == (1, 32, 128)
    
    def test_values_in_range(self):
        pe = SinusoidalPE(d_model=64, max_seq_len=256)
        out = pe(seq_len=100, device=torch.device("cpu"))
        assert out.min() >= -1.0
        assert out.max() <= 1.0
    
    def test_different_positions_different_encodings(self):
        pe = SinusoidalPE(d_model=64, max_seq_len=256)
        out = pe(seq_len=10, device=torch.device("cpu"))
        # no two positions should have identical encodings
        for i in range(10):
            for j in range(i + 1, 10):
                assert not torch.allclose(out[0, i], out[0, j])


class TestSinOnlyPE:
    def test_output_shape(self):
        pe = SinOnlyPE(d_model=128, max_seq_len=512)
        out = pe(seq_len=32, device=torch.device("cpu"))
        assert out.shape == (1, 32, 128)
    
    def test_values_in_range(self):
        pe = SinOnlyPE(d_model=64, max_seq_len=256)
        out = pe(seq_len=100, device=torch.device("cpu"))
        assert out.min() >= -1.0
        assert out.max() <= 1.0
    
    def test_different_positions_different_encodings(self):
        pe = SinOnlyPE(d_model=64, max_seq_len=256)
        out = pe(seq_len=10, device=torch.device("cpu"))
        # no two positions should have identical encodings
        for i in range(10):
            for j in range(i + 1, 10):
                assert not torch.allclose(out[0, i], out[0, j])


class TestBinaryPE:
    def test_output_shape(self):
        pe = BinaryPE(d_model=128, max_seq_len=512, normalize=False)
        out = pe(seq_len=32, device=torch.device("cpu"))
        assert out.shape == (1, 32, 128)
    
    def test_binary_encoding_pos_0(self):
        pe = BinaryPE(d_model=8, max_seq_len=16, normalize=False)
        out = pe(seq_len=8, device=torch.device("cpu"))
        # pos 0 = 00000000
        expected = torch.zeros(8)
        assert torch.allclose(out[0, 0], expected)
    
    def test_binary_encoding_pos_7(self):
        pe = BinaryPE(d_model=8, max_seq_len=16, normalize=False)
        out = pe(seq_len=8, device=torch.device("cpu"))
        # pos 7 = 00000111
        expected = torch.tensor([0., 0., 0., 0., 0., 1., 1., 1.])
        assert torch.allclose(out[0, 7], expected)
    
    def test_binary_encoding_normalized(self):
        pe = BinaryPE(d_model=8, max_seq_len=16, normalize=True)
        out = pe(seq_len=8, device=torch.device("cpu"))
        # pos 7 = 00000111 -> normalized: [-1,-1,-1,-1,-1,1,1,1]
        expected = torch.tensor([-1., -1., -1., -1., -1., 1., 1., 1.])
        assert torch.allclose(out[0, 7], expected)
    
    def test_binary_unique_per_position(self):
        pe = BinaryPE(d_model=16, max_seq_len=100, normalize=False)
        out = pe(seq_len=100, device=torch.device("cpu"))
        for i in range(100):
            for j in range(i + 1, 100):
                assert not torch.allclose(out[0, i], out[0, j])


class TestDecimalPE:
    def test_output_shape(self):
        pe = DecimalPE(d_model=128, max_seq_len=512, normalize=False)
        out = pe(seq_len=32, device=torch.device("cpu"))
        assert out.shape == (1, 32, 128)
    
    def test_decimal_encoding_pos_0(self):
        pe = DecimalPE(d_model=8, max_seq_len=100, normalize=False)
        out = pe(seq_len=50, device=torch.device("cpu"))
        # pos 0 = 00000000
        expected = torch.zeros(8)
        assert torch.allclose(out[0, 0], expected)
    
    def test_decimal_encoding_pos_42(self):
        pe = DecimalPE(d_model=8, max_seq_len=100, normalize=False)
        out = pe(seq_len=50, device=torch.device("cpu"))
        # pos 42 = 00000042
        expected = torch.tensor([0., 0., 0., 0., 0., 0., 4., 2.])
        assert torch.allclose(out[0, 42], expected)
    
    def test_decimal_unique_per_position(self):
        pe = DecimalPE(d_model=8, max_seq_len=100, normalize=False)
        out = pe(seq_len=100, device=torch.device("cpu"))
        for i in range(100):
            for j in range(i + 1, 100):
                assert not torch.allclose(out[0, i], out[0, j])


class TestRoPE:
    """Tests for Rotary Position Embedding."""
    
    def test_forward_returns_zeros(self):
        """RoPE forward should return zeros (no embedding addition)"""
        pe = RoPE(d_model=128, max_seq_len=512, n_heads=8)
        out = pe(seq_len=32, device=torch.device("cpu"))
        assert out.shape == (1, 32, 128)
        assert torch.all(out == 0)
    
    def test_properties(self):
        """RoPE should modify attention, not embeddings."""
        pe = RoPE(d_model=64, max_seq_len=256, n_heads=4)
        assert pe.adds_to_embedding is False
        assert pe.modifies_attention is True
        assert pe.requires_embedding_scaling is False
    
    def test_get_rope_fn_returns_callable(self):
        """get_rope_fn should return a callable."""
        pe = RoPE(d_model=64, max_seq_len=256, n_heads=4)
        rope_fn = pe.get_rope_fn()
        assert callable(rope_fn)
    
    def test_rope_fn_output_shapes(self):
        """Rotated Q/K should have same shapes as input."""
        d_model, n_heads, seq_len, batch = 64, 4, 32, 2
        d_head = d_model // n_heads
        
        pe = RoPE(d_model=d_model, max_seq_len=256, n_heads=n_heads)
        rope_fn = pe.get_rope_fn()
        
        q = torch.randn(batch, n_heads, seq_len, d_head)
        k = torch.randn(batch, n_heads, seq_len, d_head)
        
        q_rot, k_rot = rope_fn(q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
    
    def test_rope_preserves_norm(self):
        """Rotation should approximately preserve vector norms."""
        d_model, n_heads, seq_len, batch = 64, 4, 32, 2
        d_head = d_model // n_heads
        
        pe = RoPE(d_model=d_model, max_seq_len=256, n_heads=n_heads)
        rope_fn = pe.get_rope_fn()
        
        q = torch.randn(batch, n_heads, seq_len, d_head)
        k = torch.randn(batch, n_heads, seq_len, d_head)
        
        q_rot, k_rot = rope_fn(q, k)
        
        # norms should be preserved (rotation is orthogonal)
        q_norm_orig = q.norm(dim=-1)
        q_norm_rot = q_rot.norm(dim=-1)
        assert torch.allclose(q_norm_orig, q_norm_rot, atol=1e-5)
    
    def test_rope_different_positions_different_rotations(self):
        """Different positions should have different rotations."""
        d_model, n_heads = 64, 4
        d_head = d_model // n_heads
        
        pe = RoPE(d_model=d_model, max_seq_len=256, n_heads=n_heads)
        rope_fn = pe.get_rope_fn()
        
        # create identical vector at 2 sequence positions
        # shape: (batch=1, heads=1, seq_len=2, d_head=16)
        vec = torch.randn(d_head)
        q = torch.stack([vec, vec]).unsqueeze(0).unsqueeze(0)  # same vec at pos 0 and pos 1
        k = torch.randn(1, 1, 2, d_head)
        
        q_rot, _ = rope_fn(q, k)
        
        # rotations at pos 0 and pos 1 should differ
        assert not torch.allclose(q_rot[0, 0, 0], q_rot[0, 0, 1])
    
    def test_rope_gradient_flow(self):
        """Gradients should flow through RoPE rotation."""
        d_model, n_heads, seq_len, batch = 64, 4, 16, 2
        d_head = d_model // n_heads
        
        pe = RoPE(d_model=d_model, max_seq_len=256, n_heads=n_heads)
        rope_fn = pe.get_rope_fn()
        
        q = torch.randn(batch, n_heads, seq_len, d_head, requires_grad=True)
        k = torch.randn(batch, n_heads, seq_len, d_head, requires_grad=True)
        
        q_rot, k_rot = rope_fn(q, k)
        loss = (q_rot.sum() + k_rot.sum())
        loss.backward()
        
        assert q.grad is not None
        assert k.grad is not None


class TestPERegistry:
    def test_get_none(self):
        pe = get_pe("none", d_model=64, max_seq_len=256)
        assert isinstance(pe, NoPE)
    
    def test_get_sinusoidal(self):
        pe = get_pe("sinusoidal", d_model=64, max_seq_len=256)
        assert isinstance(pe, SinusoidalPE)
    
    def test_get_binary(self):
        pe = get_pe("binary", d_model=64, max_seq_len=256)
        assert isinstance(pe, BinaryPE)
    
    def test_get_decimal(self):
        pe = get_pe("decimal", d_model=64, max_seq_len=256)
        assert isinstance(pe, DecimalPE)
    
    def test_get_rope(self):
        pe = get_pe("rope", d_model=64, max_seq_len=256, n_heads=4)
        assert isinstance(pe, RoPE)
    
    def test_invalid_pe_type(self):
        with pytest.raises(ValueError):
            get_pe("invalid", d_model=64, max_seq_len=256)

