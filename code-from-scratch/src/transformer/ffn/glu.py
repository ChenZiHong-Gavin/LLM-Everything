import torch
import torch.nn as nn
import torch.nn.functional as F

class GateLinearUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

        self.activation = nn.GELU() if activation == 'gelu' else activation

    def forward(self, x):
        # 主分支
        main = self.linear1(x)
        # 门控分支计算
        gate = self.activation(self.linear2(x))
        return main * gate
