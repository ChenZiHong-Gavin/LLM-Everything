import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.q_embedding = nn.Linear(embedding_size, embedding_size, bias=False)
        self.k_embedding = nn.Linear(embedding_size, embedding_size, bias=False)
        self.v_embedding = nn.Linear(embedding_size, embedding_size, bias=False)

        self.norm = 1 / math.sqrt(self.embedding_size)

    def forward(self, context, x):
        # context: batch_size x seq_ctx_length x embedding_size
        # x: batch_size x seq_length x embedding_size
        batch_size, seq_len, _ = x.shape

        q = self.q_embedding(x)
        k = self.k_embedding(context)
        v = self.v_embedding(context)

        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention * self.norm
        attention = F.softmax(attention, dim=-1)

        score = torch.matmul(attention, v)

        return score
