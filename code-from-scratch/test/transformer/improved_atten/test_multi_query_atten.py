import torch

from src.transformer.improved_atten.multi_query_atten import MultiQueryAttention

def test_multi_query_atten():
    sa = MultiQueryAttention(512, 8, 64, 64)
    x = torch.randn(16, 10, 512)
    y = sa(x)
    assert y.shape == (16, 10, 512)
