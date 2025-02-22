import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.q_embedding = nn.Linear(embedding_size, embedding_size, bias=False)
        self.k_embedding = nn.Linear(embedding_size, embedding_size, bias=False)
        self.v_embedding = nn.Linear(embedding_size, embedding_size, bias=False)

        self.d_model = embedding_size
        self.norm = 1 / math.sqrt(self.d_model)

    def forward(self, x):
        # batch_size x seq_len x embedding_size
        batch_size, seq_len, _ = x.shape

        q = self.q_embedding(x)
        k = self.k_embedding(x)
        v = self.v_embedding(x)

        attention = torch.matmul(q, k.transpose(1, 2)) * self.norm # batch_size x seq_len x seq_len
        attention = F.softmax(attention, dim=-1)

        score = torch.matmul(attention, v)
        return score
