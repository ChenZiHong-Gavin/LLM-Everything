import torch

from src.transformer.ffn.ffn import FeedForward

def test_ffn():
    test_input = torch.randn(2, 10, 512)
    model = FeedForward(512, 512 * 4)
    output = model(test_input)
    assert output.shape == test_input.shape
