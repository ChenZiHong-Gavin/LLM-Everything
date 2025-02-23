import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
        self.beta = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        dims = (1, 2)
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        x_hat = (x-mean) / torch.sqrt(var + self.eps)

        gamma = self.gamma.view(-1, 1, 1)
        beta = self.beta.view(-1, 1, 1)
        return gamma * x_hat + beta
