import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups

        assert num_heads % num_groups == 0

        self.group_size = num_heads // num_groups  # 每个组的头数
        self.d_k = d_model // num_heads # 每个头的维度

        self.q_embedding = nn.Linear(d_model, d_model)
        self.k_embedding = nn.Linear(d_model, self.d_k * num_groups)
        self.v_embedding = nn.Linear(d_model, self.d_k * num_groups)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 处理查询（Q）
        q = self.q_embedding(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        q = q.transpose(1, 2)

        k = self.k_embedding(x).view(batch_size, seq_len, self.num_groups, self.d_k)
        k = k.transpose(1, 2)

        v = self.v_embedding(x).view(batch_size, seq_len, self.num_groups, self.d_k)
        v = v.transpose(1, 2) # [batch, num_groups, seq_len, d_k]

        q = q.view(batch_size, self.num_groups, self.group_size, seq_len, self.d_k)

        scores = torch.matmul(
            q, k.unsqueeze(2).transpose(-1, -2)
        )
        scores = scores / (self.d_k ** 0.5)

        attention_weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights,
                                v.unsqueeze(2))

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return output
