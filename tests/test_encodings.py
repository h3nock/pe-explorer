"""Tests for positional encodings."""
import pytest
import torch

from src.encodings import get_pe
from src.encodings.none import NoPE
from src.encodings.sinusoidal import SinusoidalPE
from src.encodings.binary import BinaryPE
from src.encodings.decimal import DecimalPE


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
    
    def test_invalid_pe_type(self):
        with pytest.raises(ValueError):
            get_pe("invalid", d_model=64, max_seq_len=256)
