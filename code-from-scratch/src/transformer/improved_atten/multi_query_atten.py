import torch
import torch.nn as nn

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.q_embedding = nn.Linear(d_model, num_heads * d_k)
        self.k_embedding = nn.Linear(d_model, d_k)
        self.v_embedding = nn.Linear(d_model, d_v)

        self.output = nn.Linear(num_heads * d_v, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 处理查询（Q）
        q = self.q_embedding(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, d_k]

        # 处理键（K）并扩展头部维度
        k = self.k_embedding(x).unsqueeze(1)  # [batch, 1, seq_len, d_k]
        k = k.transpose(-1, -2)  # [batch, 1, d_k, seq_len]

        # 处理值（V）并扩展头部维度
        v = self.v_embedding(x).unsqueeze(1)  # [batch, 1, seq_len, d_v]

        # 计算注意力得分
        scores = torch.matmul(q, k) / (self.d_k ** 0.5)  # [batch, num_heads, seq_len, seq_len]
        attention_weights = torch.softmax(scores, dim=-1)

        # 应用注意力权重到值（V）
        output = torch.matmul(attention_weights, v)  # [batch, num_heads, seq_len, d_v]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.output(output)

        return output
