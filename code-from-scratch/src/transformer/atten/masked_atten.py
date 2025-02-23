import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedAttention(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.embedding_size = embedding_size

        self.q_embedding = nn.Linear(embedding_size, embedding_size, bias=False)
        self.k_embedding = nn.Linear(embedding_size, embedding_size, bias=False)
        self.v_embedding = nn.Linear(embedding_size, embedding_size, bias=False)

        self.norm = 1 / math.sqrt(self.embedding_size)

    def forward(self, x, mask=None):
        # batch_size x seq_len x embedding_size
        batch_size, seq_len, _ = x.shape

        q = self.q_embedding(x)
        k = self.k_embedding(x)
        v = self.v_embedding(x)

        scores = torch.matmul(q, k.transpose(1, 2))
        scores = scores * self.norm

        if mask is None:
            # diagonal=1代表主对角线右边的位置为1，转bool后为True，即要mask的地方
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.to(x.device)
            scores = scores.masked_fill(mask, float('-inf'))

        else:
            scores = scores.masked_fill(mask, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)

        return out

