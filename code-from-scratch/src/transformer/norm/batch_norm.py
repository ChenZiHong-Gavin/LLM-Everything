import torch
import torch.nn as nn

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features # 通道数
        self.eps = eps

        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))


    def forward(self, x):
        # N, C, H, W
        assert x.dim() == 4

        dims = (0, 2, 3)
        mean = x.mean(dim=dims, keepdim=True) # (1, C, 1, 1)
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        # 归一化 + 缩放平移
        x_hat = (x-mean) / torch.sqrt(var + self.eps)

        gamma = self.gamma.view(1, -1, 1, 1)  # 形状 (1, C, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)  # 形状 (1, C, 1, 1)

        return gamma * x_hat + beta
