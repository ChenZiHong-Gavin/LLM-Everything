import torch

from src.transformer.improved_atten.group_query_atten import GroupedQueryAttention

def test_group_query_atten():
    sa = GroupedQueryAttention(512, 8, 8)
    x = torch.randn(16, 10, 512)
    y = sa(x)
    assert y.shape == (16, 10, 512)
