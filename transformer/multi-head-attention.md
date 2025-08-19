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

## 3 总结

* 多头注意力的作用
* 手撕多头注意力
* contiguous的作用是什么

## 参考

1. [一文搞定自注意力机制（Self-Attention）-CSDN博客](https://blog.csdn.net/weixin_42110638/article/details/134016569)
