# FeedForward

### 1 FeedForward原理

FeedForward的输入是Multi-Head Attention的输出做了残差连接和Norm之后的数据。

<figure><img src="../.gitbook/assets/image (2).png" alt=""><figcaption></figcaption></figure>

FeedForward做了两次线性线性变换，为的是更加深入的提取特征。每次线性变换都引入了非线性激活函数。在Multi-Head Attention中，对于输入主要是进行矩阵乘法进行线性变换，而线性变换的学习能力不如非线性变换的学习能力强。

<figure><img src="../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>

#### 1.1 激活函数的选择

*   RELU

    * 原始Transformer使用，计算高效，但是存在“死区”，模仿人脑，半饱和，可以有效对抗梯度爆炸/消失的问题

    <figure><img src="../.gitbook/assets/image (10).png" alt=""><figcaption></figcaption></figure>
* GELU

将不重要的激活信息规整为0。对于每一个输入x，其服从标准的正太分布，它会乘上一个伯努利分布。

<figure><img src="../.gitbook/assets/image (11).png" alt=""><figcaption></figcaption></figure>

早期FFN层的激活函数用ReLU，现在BERT、GPT等主流模型多用GELU。GELU可以看作是ReLU的平滑近似，它在负值区不是完全置零，而是有一个平滑的曲线，近似神经元的随机正则化效果，被认为在处理自然语言任务时能提供更好的性能，因为它允许更丰富的梯度信息流动。

#### 1.2 Position-wise

Position-wise 代表对每个 token 应用同样的变换，但相互独立。它和注意力层形成功能互补：注意力负责“横向”的token间信息流动，FFN负责“纵向”的单个token信息深化。

### 2 FeedForward作用

通过线性变换和非线性激活函数，先将数据映射到高纬度的空间再映射到低纬度的空间，提取了更深层次的特征；通过激活函数引入非线性变换，增强模型对复杂模式的拟合能力。

如果没有FFN提供的非线性，那么多层Transformer堆叠在一起，其表达能力将大打折扣。因为多层线性变换的叠加本质上仍然是一个线性变换。非线性使得模型能够学习和拟合更加复杂的函数和模式。

**简单来说？**

* **提供“思考空间”**：升维操作可以被看作是给模型一个更大的“特征空间”或“思考空间”。在这个高维空间里，原始空间中线性不可分的特征可能变得更容易被分开和处理。

### 3 FFN层的参数量

在Transformer模型中，FFN层占据了绝大部分的参数。

* **举例**：以BERT-base为例， $$d_{model} = 768，d_{ff} = 3072$$
  * FFN层的参数量约等于：$$d_{model} \times d_{ff} + d_{ff} \times d_{model} = 2 \times 768 \times 3072 \approx 4.7 M$$
  * 而自注意力层中，Q/K/V的投影矩阵参数量为 $$3 \times d_{model} \times d_{model} = 3 \times 768 \times 768 \approx 1.8 M$$
* 模型的大部分“知识”或者说模式记忆，实际上是存储在FFN的权重中的。因此，FFN也被认为是Transformer中实现“记忆”功能的重要部分。
  * 后续的工作如**MoE (Mixture of Experts)**，就是通过将一个巨大的FFN层替换为多个稀疏激活的“专家”FFN网络，来在不显著增加计算量的情况下，极大扩展模型参数量。

### 4 FFN层的实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, activation: str = "gelu", dropout: float = 0.1):
        """
        Args:
            d_model: 输入/输出维度（即 Transformer 的隐藏层维度）
            d_ff:    中间层维度（默认扩展为 4*d_model）
            activation: 激活函数，支持 "gelu" 或 "relu"
            dropout:  Dropout 概率
        """
        super().__init__()
        d_ff = d_ff or 4 * d_model  # 默认扩展比例为4倍

        # 定义两个线性层
        self.linear1 = nn.Linear(d_model, d_ff)  # 扩展维度
        self.linear2 = nn.Linear(d_ff, d_model)  # 压缩回原维度
        
        # 激活函数选择
        self.activation = F.gelu if activation == "gelu" else F.relu
        
        # Dropout 层（可选）
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 参数初始化（关键！）
        self._init_weights()

    def _init_weights(self):
        # 使用 He/Kaiming 初始化（适合 ReLU）
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)
        
        # 缩小输出层的初始化范围
        nn.init.xavier_normal_(self.linear2.weight, gain=0.02)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入形状: (batch_size, seq_len, d_model)
        输出形状: (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)       # (batch, seq, d_ff)
        x = self.activation(x)    # 非线性激活
        x = self.dropout(x)       # 可选 Dropout
        x = self.linear2(x)       # (batch, seq, d_model)
        return x
```



### 参考

1. [前馈神经网络（Feed-Forward Neural Network）-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/2511089)
2. [对Transformer中FeedForward层的理解\_feedforward层的作用-CSDN博客](https://blog.csdn.net/weixin_51756104/article/details/127250190)



