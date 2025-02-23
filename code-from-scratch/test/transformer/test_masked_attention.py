import torch

from src.transformer.masked_atten import MaskedAttention

def test_masked_attention():
    model = MaskedAttention(512)
    x = torch.rand(16, 10, 512)
    y = model(x)
    assert y.shape == (16, 10, 512)
