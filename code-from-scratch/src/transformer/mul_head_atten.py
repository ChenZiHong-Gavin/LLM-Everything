import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, attention_head_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.attention_head_size = attention_head_size

        assert self.attention_head_size % self.num_heads == 0
        self.head_dim = self.attention_head_size // self.num_heads

        self.q_embedding = nn.Linear(self.embedding_size, self.attention_head_size, bias=False)
        self.k_embedding = nn.Linear(self.embedding_size, self.attention_head_size, bias=False)
        self.v_embedding = nn.Linear(self.embedding_size, self.attention_head_size, bias=False)

        self.norm = 1 / math.sqrt(self.head_dim)

    def forward(self, x):
        # batch_size x seq_len x embedding_size
        batch_size, seq_len, _ = x.shape

        q = self.q_embedding(x)
        k = self.k_embedding(x)
        v = self.v_embedding(x)

        # split multi heads
        # batch_size x num_heads x seq_len x head_dim
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores * self.norm

        attention = F.softmax(scores, dim=-1)
        attention = torch.matmul(attention, v)

        # batch_size x seq_len x atten_size
        out = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return out, attention
