import torch

from src.transformer.atten.self_atten import SelfAttention

def test_self_atten():
    model = SelfAttention(512)
    x = torch.rand(16, 10, 512)
    y = model(x)
    assert y.shape == (16, 10, 512)
