# Add & Norm

Transformer的Encoder层和Decoder层中都用到了Add\&Norm操作。

<figure><img src="../.gitbook/assets/image (29).png" alt=""><figcaption></figcaption></figure>

## 1 Add

残差连接就是把网络的输入和输出相加，即网络的输出为$$F(x)+x$$

在网络结构比较深的时候，网络梯度反向传播更新参数时，容易造成梯度消失的问题（每一层都乘以一个小于1的数，几十次一乘就几乎变成了0），但是如果每层的输出都加上一个$$x$$的时候，就变成了$$F(x)+x$$，对$$x$$求导结果为1，所以就相当于每一层求导时都加上了一个常数项 1 ，有效解决了梯度消失问题。

## 2 Norm

### 2.1 Norm的作用

当我们使用梯度下降法做优化时，随着网络深度的增加，输入数据的特征分布会不断发生变化，为了**保证数据特征分布的稳定性**，会加入Normalization。从而可以使用更大的学习率，**加速模型的收敛速度**。同时，Normalization也有一定的**抗过拟合**作用，使训练过程更加平稳。&#x20;

BN（BatchNorm）和LN（LayerNorm）是两种最常用的Normalization的方法，它们都是将输入特征转换为均值为0，方差为1的数据，它们的形式是：

<p align="center"><span class="math">\begin{equation} B N\left(x_i\right)=\alpha \times \frac{x_i-\mu_b}{\sqrt{\sigma_B^2+\epsilon}}+\beta \end{equation} </span></p>

<p align="center"><span class="math">\begin{equation}L N\left(x_i\right)=\alpha \times \frac{x_i-\mu_L}{\sqrt{\sigma_L^2+\epsilon}}+\beta\end{equation} </span></p>

<figure><img src="../.gitbook/assets/image (30).png" alt=""><figcaption></figcaption></figure>

BatchNorm一般用于CV，LayerNorm一般用于NLP

<figure><img src="../.gitbook/assets/image (31).png" alt=""><figcaption></figcaption></figure>

### 2.2 BatchNorm

假设把中国的收入水平进行标准化（变成标准正态分布），这时中国高收入人群的收入值接近3，中收入人群的收入值接近0，低收入人群接近-3。不难发现，标准化后的相对大小是不变的，即中国富人的收入水平在标准化前和标准化后都比中国穷人高。 **把中国的收入水平看成一个分布的话，我们可以说一个分布在标准化后，分布内的样本还是可比较的。**

假设把中国和印度的收入水平分别进行标准化，这时中国和印度的中收入人群的收入值都为0，但是这两个0可比较吗？印度和中国的中等收入人群的收入相同吗？不难发现，中国和印度的收入水平在归一化后，两国间收入值已经失去了可比性。 **把中国和印度的收入水平各自看成一个分布的话，我们可以说，不同分布分别进行标准化后，分布间的数值不可比较**

<figure><img src="../.gitbook/assets/image (32).png" alt=""><figcaption></figcaption></figure>

BatchNorm把一个batch中同一通道的所有特征（如上图红色区域）视为一个分布（有几个通道就有几个分布），并将其标准化。这意味着:

* 不同图片同一通道的相对关系是保留的，即不同图片的同一通道的特征是可比较的
* 同一图片的不同通道的特征失去了可比性

feature的每个通道都对应一种特征（如低纬特征的颜色、纹理、亮度等，高纬特征的人眼、鸟嘴等）。BatchNorm之后，颜色特征是可以相互比较的，但是颜色特征与纹理特征其实没有必要比较。

#### 2.2.1 BatchNorm代码实现

```python
import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # 判断 2-D 还是 4-D
        if x.dim() == 2:                       # (B, C)
            dims = (0,)
        else:                                  # (B, C, H, W)
            dims = (0, 2, 3)

        mean = x.mean(dim=dims, keepdim=True)
        var  = x.var(dim=dims, keepdim=True, unbiased=False)
        x_hat = (x - mean) / (var + self.eps).sqrt()

        # 广播 γ, β
        return self.weight.view_as(mean) * x_hat + self.bias.view_as(mean)
```

### 2.3 LayerNorm

<figure><img src="../.gitbook/assets/image (33).png" alt=""><figcaption></figcaption></figure>

同一句子中词义向量（上图中的V1, V2, …, VL）的相对大小是保留的

考虑两个句子，“教练，我想打篮球！” 和 “老板，我要一打包子。”。通过比较两个句子中 “打” 的词义我们可以发现，词义并非客观存在的，而是由上下文的语义决定的。 因此进行标准化时不应该破坏同一句子中不同词义向量的可比性，而LayerNorm是满足这一点的，BatchNorm则是不满足这一点的。且不同句子的词义特征也不应具有可比性，LayerNorm也是能够把不同句子间的可比性消除。

#### 2.3.1 LayerNorm代码实现

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
	def __init__(self, dim, eps=1e-6):
		super().__init__()
		self.eps = eps
		self.weight = nn.Parameter(torch.ones(dim))
		self.bias = nn.Parmaeter(torch.zeros(dim))
	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True, unbiased=False)
		return self.weight * (x - mean) / (std + self.eps) + self.bias
```

### 2.4 RMSNorm

虽然LayerNorm很好，但是它每次需要计算均值和方差。RMSNorm的思想就是移除(1)式中$$\mu$$的计算部分。

<figure><img src="../.gitbook/assets/image (34).png" alt=""><figcaption></figcaption></figure>

相当于仅使用$$x$$的均方根来对输入进行归一化，它简化了层归一化的计算，变得更加高效。

```python
import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm(nn.Module):
  def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(hidden_size))
  
  def _norm(self, hidden_states: Tensor) -> Tensor:
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + self.eps)
  
  def forward(self, hidden_states: Tensor) -> Tensor:
    return self.weight * self._norm(hidden_states.float()).type_as(hidden_states)
    

```

RMSNorm使用示例：

```python
import torch
import torch.nn as nn
from torch import Tensor

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(in_features=10, out_features=5)
        self.rmsnorm = RMSNorm(hidden_size=5)

    def forward(self, x):
        x = self.linear(x)
        x = self.rmsnorm(x)
        return x

net = SimpleNet()

input_data = torch.randn(2, 10)  # 2个样本，每个样本包含10个特征

output = net(input_data)

print("Input Shape:", input_data.shape)
print("Output Shape:", output.shape)

```

### 2.5 DeepNorm

DeepNorm用于超深层 Transformer 的稳定训练，是 **LayerNorm 的改进版本**，具体来说，**把残差分支的结果先放大 α 倍，再做 LayerNorm，从而在千层网络里保持梯度量级恒定，训练更稳。**

$$DeepNorm(x) = LayerNorm(α·x + Sublayer(x))$$&#x20;

α>1，由总层数 N 决定。

#### 2.5.1 DeepNorm代码实现

```python
import torch
import torch.nn as nn
import math

class DeepNorm(nn.Module):
    """单层的 DeepNorm（Post-Norm 版）"""
    def __init__(self, d_model: int, N: int):  # N = 模型总层数
        super().__init__()
        self.alpha = (2 * N) ** 0.25
        self.beta  = (8 * N) ** -0.25
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x, sublayer):
        """
        x:    残差输入 (B, L, d)
        sublayer: 一个 nn.Module，例如 Attention 或 FFN
        """
        return self.norm(self.alpha * x + sublayer(x * self.beta))
```

### 2.6 Post-norm & Pre-norm

论文 _On Layer Normalization in the Transformer Architecture_ 提出了两种Layer Normalization方式并进行了对比。

把Transformer架构中传统的**Add\&Norm**做layer normalization的方式叫做Post-LN，并针对Post-LN，模型提出了Pre-LN，即把layer normalization加在残差连接之前，如下图所示：

<figure><img src="../.gitbook/assets/image (35).png" alt=""><figcaption></figcaption></figure>

归一化的位置也有区别，分为后归一化（PostNorm）和前归一化（PreNorm），其中PostNorm在操作后进行归一化，而PreNorm在操作前进行归一化。PreNorm相较于Postnorm无需warmup,模型的收敛速度更快,但是实际应用中一般PreNorm效果不如PostNorm，因为PreNorm多层叠加的结果更多是增加宽度而不是深度。

## 总结

1. 残差连接的作用是什么？
2. norm的作用是什么？
3. LN和BN的区别
4. 手撕LN和BN
5. 手撕RMSNorm
6. RMS Norm 相比于 Layer Norm 有什么特点？
7. 手撕Deep Norm
8. Deep Norm 有什么优点？
9. LN在LLMs中的不同位置有什么区别吗？
10. LLMs各模型分别用了哪种LN

## 参考

1. [https://www.xiaohongshu.com/explore/6648b5680000000005005849](https://www.xiaohongshu.com/explore/6648b5680000000005005849)
2. [对Transformer中Add\&Norm层的理解-CSDN博客](https://blog.csdn.net/weixin_51756104/article/details/127232344)
3. [BERT用的LayerNorm可能不是你认为的那个Layer Norm？ (qq.com)](https://mp.weixin.qq.com/s/HNCl6MPS_hjTVHNt7UkYyw)
4. [Llama改进之——均方根层归一化RMSNorm-CSDN博客](https://blog.csdn.net/yjw123456/article/details/138139970)
5. [https://arxiv.org/abs/2203.00555](https://arxiv.org/abs/2203.00555)
