import torch

from src.transformer.norm.batch_norm import BatchNorm2d

def test_batch_norm():
    bn = BatchNorm2d(num_features=3)
    x = torch.randn(2, 3, 4, 4)
    output = bn(x)
    assert output.shape == x.shape
