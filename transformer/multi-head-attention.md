# Multi-Head Attention

## 1 Multi-Head Attention概述

一段文字可能蕴含了不同维度的特征，比如情感、时间、逻辑等，为了能够从不同的维度抓取信息的重点，会使用Multi-Head Attention。Multi-Head Attention 是由多个 Self-Attention 组合形成。每个self-attention会产生一个维度上的输出特征，当使用多头注意力时，允许模型从不同的子空间捕捉不同级别的特征和信息，使模型从不同的角度理解数据。

<figure><img src="../.gitbook/assets/image (27).png" alt=""><figcaption></figcaption></figure>

在这里，V,K,Q三个矩阵通过h个线性变换，分别得到h组V,K,Q，每一组经过Attention公式计算，得到h个Attention Score进行拼接，最后通过一个线性变换得到输出。

## 2 Multi-Head Attention 例子

输入词：X=\[‘图’, ’书’, ’馆’]，句子长度为3，词向量的维度为4。

这里将词向量分为2个头，线性变换后得到2组$$(V_0, K_0, Q_0)$$和$$(V_1, K_1, Q_1)$$。每组$$(V, K, Q)$$进行Self-Attention计算得到两个Score即$$Z_0$$和$$Z_1$$，将$$Z_0$$和$$Z_1$$进行拼接Concat后进行线性变换得到输出向量Z，其维度与输入矩阵维度相同。

<figure><img src="../.gitbook/assets/image (28).png" alt=""><figcaption></figcaption></figure>

```python
import torch
import torch.nn as nn
import numpy as np
import torch.nn.funcational as F

class MultiHeadAttention(nn.module):
	def __init__(self, embeded_size, num_heads, attention_head_size):
		super().__init__()
		self.num_heads = num_heads
		self.attention_head_size = attention_head_size
		self.embeded_size = embeded_size
		
		self.W_query = nn.Linear(embeded_size, attention_head_size)
		self.W_key = nn.Linear(embeded_size, attention_head_size)
		self.W_value = nn.Linear(embeded_size, attention_head_size)
	
	def forward(self, x):
		batch_size, seq_len, _ = x.size()
		querys = self.W_query(x) # (batch_size, sequence_len, attention_head_size)
		keys = self.W_keys(x)
		values = self.W_values(x)
		
		assert self.attention_head_size % self.num_heads == 0
		split_size = self.attention_head_size // self.num_heads
		
		querys = torch.view(self.num_heads, batch_size, seq_len, split_size) # (h, batch_size, sequence_len, split_size)
		keys = torch.view(self.num_heads, batch_size, seq_len, split_size)
		values = torch.view(self.num_heads, batch_size, seq_len, split_size)
		
		scores = torch.matmul(querys, keys.transpose(2, 3))
		scores = scores / (split_size ** 0.5)
		
		scores = F.softmax(scores, dim=-1)
		
		out = torch.matmul(scores, values) 	# (h, batch_size, sequence_len, split_size)
		out = out.transpose(0, 1) 	# (batch_size, h, sequence_len, split_size)
		out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) 
		
		return out, scores
		
		
	
```

`contiguous()`:

* 这个操作是为了确保张量在内存中是连续存储的，这样我们才能使用 `view()` 方法进行张量的重塑。

## 3 MQA & GQA

MHA是一种基础的注意力机制，它通过将输入分割成多个头来并行计算注意力，每个头学习输入的不同部分，最终将结果合并，以捕获序列的不同方面信息 。

MQA则是一种优化的注意力机制，它通过让所有头共享相同的键和值，减少了参数量和计算量，从而加快了推理速度，但可能会牺牲一些精度 。

GQA作为MHA和MQA的折中方案，它将查询头（query heads）分组，每组共享一个键和值，而不是所有头都共享。这样，GQA能够在减少计算量的同时，保持更多的多样性，从而在推理速度和模型精度之间取得平衡 。

* GQA-1：一个单独的组，等同于 Multi-Query Attention (MQA)。
* GQA-H：组数等于头数，基本上与 Multi-Head Attention (MHA) 相同。
* GQA-G：一个中间配置，具有G个组，平衡了效率和表达能力。

具体来说，GQA通过分组的方式，减少了需要缓存的键和值的数量，从而减少了内存的使用，同时由于不是所有头都共享键和值，它能够比MQA更好地保持MHA的多样性和精度 。例如，如果GQA使用2个头的键和值，那么每个组包含4个查询头，这样在保持速度的同时，精度损失会比MQA小 。

```
class  MultiQueryAttention(Attention):
    r"""
    https://arxiv.org/pdf/1911.02150.pdf
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64, n_query:int=8) -> None:
        super().__init__(word_size, embed_dim)
        self.n_query = n_query
        self.proj = nn.Linear(in_features=embed_dim * n_query,
                              out_features=embed_dim, bias=False)
        delattr(self, 'query')
        self.querys = nn.ModuleList([
            nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
            for _ in range(n_query)
        ])
        self.key = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.value = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)

    def forward(self, x: Tensor, mask:Optional[BoolTensor]=None) -> Tensor:
        K = self.key(x)
        V = self.value(x)
        Z_s = torch.cat([
            self.self_attention(query(x), K, V, mask) for query in self.querys
        ], dim=1)
        Z = self.proj(Z_s)
        return Z


class  GroupedQueryAttention(Attention):
    r"""
    https://arxiv.org/pdf/2305.13245.pdf
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64,
                 n_grouped: int = 4, n_query_each_group:int=2) -> None:
        super().__init__(word_size, embed_dim)
        delattr(self, 'query')
        delattr(self, 'key')
        delattr(self, 'value')

        self.grouped = nn.ModuleList([
            MultiQueryAttention(word_size, embed_dim, n_query=n_query_each_group)
            for _ in range(n_grouped)
        ])
        self.proj = nn.Linear(in_features=embed_dim * n_grouped,
                              out_features=embed_dim, bias=False)

    def forward(self, x: Tensor, mask:Optional[BoolTensor]=None) -> Tensor:
        Z_s = torch.cat([head(x, mask) for head in self.grouped], dim=1)
        Z = self.proj(Z_s)
        return Z
```

## 4 总结

* 多头注意力的作用
* 手撕多头注意力
* contiguous的作用是什么

## 参考

1. [一文搞定自注意力机制（Self-Attention）-CSDN博客](https://blog.csdn.net/weixin_42110638/article/details/134016569)
2. [https://zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/30483365941)[/p/30483365941](https://zhuanlan.zhihu.com/p/30483365941)
3. [https://blog.51cto.com/u\_16163453/12100632](https://blog.51cto.com/u_16163453/12100632)
