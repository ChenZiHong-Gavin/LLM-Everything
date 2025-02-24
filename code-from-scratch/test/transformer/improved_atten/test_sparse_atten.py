import torch

from src.transformer.improved_atten.sparse_atten import SparseAttention

def test_sparse_atten():
    sa = SparseAttention(embed_size=512, top_k=2)
    x = torch.randn(16, 10, 512)
    y = sa(x)
    assert y.shape == (16, 10, 512)

