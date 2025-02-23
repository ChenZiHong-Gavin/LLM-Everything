import torch

from src.transformer.mul_head_atten import MultiHeadAttention

def test_multi_head_attention():
    model = MultiHeadAttention(512, 8, 768)
    x = torch.rand(16, 10, 512)
    y, _ = model(x)
    assert y.shape == (16, 10, 768)
