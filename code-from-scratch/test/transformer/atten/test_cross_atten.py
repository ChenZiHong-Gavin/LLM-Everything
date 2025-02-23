import torch

from src.transformer.atten.cross_atten import CrossAttention

def test_cross_attention():
    model = CrossAttention(512)
    context = torch.rand(16, 10, 512)
    x = torch.rand(16, 10, 512)
    y = model(context, x)
    assert y.shape == (16, 10, 512)
