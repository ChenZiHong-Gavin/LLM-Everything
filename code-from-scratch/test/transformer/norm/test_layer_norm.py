import torch

from src.transformer.norm.layer_norm import LayerNorm

def test_layer_norm():
    ln = LayerNorm(normalized_shape=3)
    x = torch.randn(3, 16, 512)
    output = ln(x)
    assert output.shape == x.shape
