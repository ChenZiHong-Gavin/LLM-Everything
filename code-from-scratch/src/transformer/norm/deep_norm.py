import torch.nn as nn

from src.transformer.norm.layer_norm import LayerNorm


class DeepNorm(nn.Module):
    def __init__(self, alpha, normalized_shape, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.layer_norm = LayerNorm(normalized_shape, eps)

    def forward(self, x, gx):
        return self.layer_norm(x + self.alpha * gx)
