import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x):
        # batch_size, seq_len, dim
        rms = torch.sqrt(
            x.pow(2).mean(dim=(1, 2), keepdim=True) + self.eps
        )

        gamma = self.gamma.view(-1, 1, 1)

        return gamma * (x / rms)
