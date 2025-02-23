import torch

from src.transformer.norm.deep_norm import DeepNorm

def test_batch_norm():
    dn = DeepNorm(alpha=1, normalized_shape=3)
    x = torch.randn(3, 16, 512)
    gx = torch.randn(3, 16, 512)
    output = dn(x, gx)
    assert output.shape == x.shape
