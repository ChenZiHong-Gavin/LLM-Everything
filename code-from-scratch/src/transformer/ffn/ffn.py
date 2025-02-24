import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, activation = "gelu", dropout_rate=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.activation = F.gelu if activation == 'gelu' else activation

        # 可选
        self.dropout = nn.Dropout(dropout_rate)

        # 关键点：初始化参数
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')  # 添加下划线
        nn.init.zeros_(self.linear1.bias)

        nn.init.xavier_normal_(self.linear2.weight, gain=0.02)  # 添加下划线
        nn.init.zeros_(self.linear2.bias)


    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x
