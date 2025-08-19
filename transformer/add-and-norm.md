# Add & Norm

在Encoder层和Decoder层中都用到了Add\&Norm操作

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/4cc6e659-51de-4788-a12f-40e2f862c8c2/08dfdae1-b39a-432b-9ff7-c49654099f56/image.png)

## 1 Add

残差连接就是把网络的输入和输出相加，即网络的输出为F(x)+x

在网络结构比较深的时候，网络梯度反向传播更新参数时，容易造成梯度消失的问题，但是如果每层的输出都加上一个x的时候，就变成了F(x)+x，对x求导结果为1，所以就相当于每一层求导时都加上了一个常数项‘1’，有效解决了梯度消失问题

## 2 Norm

### 2.1 Norm的作用

当我们使用[梯度下降法](https://so.csdn.net/so/search?q=%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95\&spm=1001.2101.3001.7020)做优化时，随着网络深度的增加，输入数据的特征分布会不断发生变化，为了**保证数据特征分布的稳定性**，会加入Normalization。从而可以使用更大的[学习率](https://so.csdn.net/so/search?q=%E5%AD%A6%E4%B9%A0%E7%8E%87\&spm=1001.2101.3001.7020)，从而**加速模型的收敛速度**。同时，Normalization也有一定的**抗过拟合**作用，使训练过程更加平稳。 **LayerNorm & BatchNorm**

BN（BatchNorm）和LN（LayerNorm）是两种最常用的Normalization的方法，它们都是将输入特征转换为均值为0，方差为1的数据，它们的形式是：

\$$

\begin{equation} B N\left(x\_i\right)=\alpha \times \frac{x\_i-\mu\_b}{\sqrt{\sigma\_B^2+\epsilon\}}+\beta \end{equation} \$$

\$$ \begin{equation}L N\left(x\_i\right)=\alpha \times \frac{x\_i-\mu\_L}{\sqrt{\sigma\_L^2+\epsilon\}}+\beta\end{equation} \$$

![image.png](attachment:2342507f-a53d-4356-856f-4890abea0710:image.png)

BatchNorm一般用于CV，LayerNorm一般用于NLP

![image.png](attachment:7583a55f-c0ac-4d92-9ef6-618d50d5426b:image.png)

### 2.2 BatchNorm

假设把中国的收入水平进行标准化（变成标准正态分布），这时中国高收入人群的收入值接近3，中收入人群的收入值接近0，低收入人群接近-3。不难发现，标准化后的相对大小是不变的，即中国富人的收入水平在标准化前和标准化后都比中国穷人高。 把中国的收入水平看成一个分布的话，我们可以说一个分布在标准化后，分布内的样本还是可比较的

假设把中国和印度的收入水平分别进行标准化，这时中国和印度的中收入人群的收入值都为0，但是这两个0可比较吗？印度和中国的中等收入人群的收入相同吗？不难发现，中国和印度的收入水平在归一化后，两国间收入值已经失去了可比性。 **把中国和印度的收入水平各自看成一个分布的话，我们可以说，不同分布分别进行标准化后，分布间的数值不可比较**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/4cc6e659-51de-4788-a12f-40e2f862c8c2/c1eded30-4237-4910-9079-fff8dec98251/image.png)

BatchNorm把一个batch中同一通道的所有特征（如上图红色区域）视为一个分布（有几个通道就有几个分布），并将其标准化。这意味着:

* 不同图片同一通道的相对关系是保留的，即不同图片的同一通道的特征是可比较的
* 同一图片的不同通道的特征失去了可比性

feature的每个通道都对应一种特征（如低纬特征的颜色、纹理、亮度等，高纬特征的人眼、鸟嘴等）。BatchNorm之后，颜色特征是可以相互比较的，但是颜色特征与纹理特征其实没有必要比较。

### 2.3 LayerNorm

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/4cc6e659-51de-4788-a12f-40e2f862c8c2/d8d23ec3-e1ff-4bd8-b0a2-93b064a51f37/image.png)

同一句子中词义向量（上图中的V1, V2, …, VL）的相对大小是保留的

考虑两个句子，“教练，我想打篮球！” 和 “老板，我要一打包子。”。通过比较两个句子中 “打” 的词义我们可以发现，词义并非客观存在的，而是由上下文的语义决定的。 因此进行标准化时不应该破坏同一句子中不同词义向量的可比性，而LayerNorm是满足这一点的，BatchNorm则是不满足这一点的。且不同句子的词义特征也不应具有可比性，LayerNorm也是能够把不同句子间的可比性消除。

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
	def __init__(self, dim, eps=1e-6):
		super().__init__()
		self.eps = eps
		self.weight = nn.Parameter(torch.ones(dim))
		self.boas = nn.Parameter(torch.zeros(dim))
	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True, unbiased=False)
		return self.weight * (x - mean) / (std + self.eps) + self.bias
```

### 2.4 RMSNorm

虽然LayerNorm很好，但是它每次需要计算均值和方差。RMSNorm的思想就是移除(1)式中μ \muμ的计算部分

![image.png](attachment:38f1f196-6011-4ae6-843f-c59ab2835b53:image.png)

相当于仅使用x的均方根来对输入进行归一化，它简化了层归一化的计算，变得更加高效

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

DeepNet: Scaling Transformers to 1,000 Layers

用于超深层 Transformer 的稳定训练

DeepNorm 是 **LayerNorm 的改进版本**，通过**残差路径权重缩放**和**前置归一化**组合，解决超深层网络的梯度消失/爆炸问题。

### 2.6 Post-norm & Pre-norm

论文《On Layer Normalization in the Transformer Architecture》提出了两种Layer Normalization方式并进行了对比

把Transformer架构中传统的**Add\&Norm**做layer normalization的方式叫做Post-LN，并针对Post-LN，模型提出了Pre-LN，即把layer normalization加在残差连接之前，如下图所示：

![image.png](attachment:26f2b2d6-8b6b-4e9c-ae88-2a13a29336e4:image.png)

归一化的位置也有区别，分为后归一化（PostNorm）和前归一化（PreNorm），其中PostNorm在操作后进行归一化，而PreNorm在操作前进行归一化。PreNorm相较于Postnorm无需warmup,模型的收敛速度更快,但是实际应用中一般PreNorm效果不如PostNorm，因为PreNorm多层叠加的结果更多是增加宽度而不是深度。

Pre-LN在底层的梯度往往大于顶层

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

### 参考

1. [https://www.xiaohongshu.com/explore/6648b5680000000005005849](https://www.xiaohongshu.com/explore/6648b5680000000005005849)
2. [对Transformer中Add\&Norm层的理解-CSDN博客](https://blog.csdn.net/weixin_51756104/article/details/127232344)
3. [BERT用的LayerNorm可能不是你认为的那个Layer Norm？ (qq.com)](https://mp.weixin.qq.com/s/HNCl6MPS_hjTVHNt7UkYyw)
4. [Llama改进之——均方根层归一化RMSNorm-CSDN博客](https://blog.csdn.net/yjw123456/article/details/138139970)
