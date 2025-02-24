import torch
import torch.nn as nn


class SparseAttention(nn.Module):
    def __init__(self, embed_size, top_k=2):
        super().__init__()
        self.embed_size = embed_size
        self.top_k = top_k

        self.q_embedding = nn.Linear(embed_size, embed_size)
        self.k_embedding = nn.Linear(embed_size, embed_size)
        self.v_embedding = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        q = self.q_embedding(x)
        k = self.k_embedding(x)
        v = self.v_embedding(x)

        scores = torch.matmul(q, k.transpose(1, 2))

        # 获取每行的top_k个值以及索引
        topk_values, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        sparse_scores = torch.zeros_like(scores)
        sparse_scores.scatter_(dim=-1, index=topk_indices, src=topk_values)

        # 归一化
        attention_weights = sparse_scores / (sparse_scores.sum(dim=-1, keepdim=True) + 1e-10)

        output = torch.matmul(attention_weights, v)

        return output
