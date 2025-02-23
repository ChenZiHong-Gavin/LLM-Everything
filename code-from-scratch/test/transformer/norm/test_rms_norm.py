import torch

from src.transformer.norm.rms_norm import RMSNorm

def test_rms_norm():
    rmsn = RMSNorm(normalized_shape=3)
    x = torch.randn(3, 16, 512)
    output = rmsn(x)
    assert output.shape == x.shape
