import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self,
                 input_dim,
                 bottleneck_dim=64,
                 non_linearity='relu',
                 dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.non_linearity = nn.ReLU() if non_linearity == 'relu' else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        residual = x
        x = self.down_proj(x)
        x = self.non_linearity(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x + residual  # 残差连接
