# Self Attention

<figure><img src="../.gitbook/assets/image (23).png" alt=""><figcaption></figcaption></figure>

## 1 Self-Attention的概念

NLP任务中，输入的文本序列充当了上下文的作用。

人类在处理长文本时，需要知道一个单词与其它单词之间的相关性，也就是“语境”。**相关性**是人类对两个词在语义或语法上有多大关联”的直觉判断。比如，“apple”和“fruit”相关性强，“apple”和“car”相关性弱。

自注意力机制将单个序列的不同位置关联起来，用来计算同一序列的注意力权重。**注意力权重**是LLM通过训练任务（如语言建模、翻译、分类等）**自动学到的数值**，表示在某个特定上下文中，一个token对另一个token的“依赖程度”，和人类所说的“相关性”类似。

在一般的注意力机制中（Encoder-Decoder框架），query和key是不同来源的。例如在中译英模型中，query是中文单词特征，而key是英文单词特征。但self-attention的query和key都来自同一组元素，相互之间做注意力汇聚。

Self Attention可以捕获同一个句子中单词之间的一些句法特征或者语义特征。

引入Self Attention后会**更容易捕获句子中长距离的相互依赖的特征**。如果是RNN/LSTM，需要按照次序依次计算，对于远距离相互依赖的特征，要经过若干步的信息累计才能将两者联系起来，而距离越远，有效捕捉的可能性越小。

## 2 Self-Attention V.S. CNN

如果把Self-Attention用于图像：

* 优点
  * 可以建立全局的依赖关系，扩大图像的感受野。相比于CNN，其感受野更大，可以获取更多上下文信息。在全局建模能力上，Self-Atten有明显优势，它可以显式捕捉序列中任意两个元素之间的关系，不论它们的距离，在处理长距离依赖和全局信息方面非常强大
* 缺点
  * **自注意力机制是通过筛选重要信息，过滤不重要信息实现的，这就导致其有效信息的抓取能力会比CNN小一些**
  * 无法利用图像本身具有的尺度，平移不变性，以及图像的特征局部性这些先验知识，**只能通过大量数据进行学习。这就导致自注意力机制只有在大数据的基础上才能有效地建立准确的全局关系，而在小数据的情况下，其效果不如CNN**

## 3 Self-Attention原理

Self-Attention的**架构**

<figure><img src="../.gitbook/assets/image (1) (2).png" alt=""><figcaption></figcaption></figure>

Attention函数将query和一系列键值对mapping到一个output。

<figure><img src="../.gitbook/assets/image (23) (1).png" alt=""><figcaption></figcaption></figure>

主要步骤如下：

1. 将query和每个key进行**相似度计算**得到权重，相似度函数有点积、拼接、感知机等
2. 使用softmax函数将这些权重进行归一化
3. 将权重和对应的value相乘得到最后的attention

在Self-Atten中，输入序列中的每个词的**key**和**value**实际上是同一组线性变换后的表示。Key是用于检索的索引，类似于搜索引擎的关键词，value是被检索的内容，即搜搜结果的具体信息。而在Self-Atten中，每个词**既是检索者，也是被检索者。**

* 键值对形式的Attention计算公式

<figure><img src="../.gitbook/assets/image (24).png" alt=""><figcaption></figcaption></figure>

**为什么要除以** $$\sqrt{d_k}$$ **?**

* 压缩softmax输入值，将方差重新缩放到 1，避免输入值过大，进入饱和区，导致梯度值太小（求导之后发现）难以训练

在Self-Atten中，公式中的Q、K、V的来源都是输入矩阵X与矩阵的乘积，本质上是X的线性变换：

<figure><img src="../.gitbook/assets/image (25).png" alt=""><figcaption></figcaption></figure>

## 4 代码实现

```python
import torch
import torch.nn as nn
import torch.function as F

class Attention(nn.module):
	def __init__(self, embeded_size):
		super().__init__()
		self.linear_q = nn.Linear(embeded_size, embeded_size, bias=False)
		self.linear_k = nn.Linear(embeded_size, embeded_size, bias=False)
		self.linear_v = nn.Linear(embeded_size, embeded_size, bias=False)
		
		self.norm = 1/math.sqrt(embeded_size)
	
	def forward(self, x):
		# x: batch_size, seq_len, embeded_size
		batch_size, seq_len, _ = x
		q = self.linear_q(x) # (batch_size, seq_len, embed_size)
		k = self.linear_k(x)
		v = self.linear_v(x)
		
		# dot production
		attention = torch.bmm(q, k.transpose(1,2)) * self.norm
		attention = F.softmax(attention, dim=-1) # (batch_size, seq_len, seq_len)
		
		out = torch.bmm(attention, values)
		return out
		
```

**为什么bias=False**

* 由于 softmax 函数会将输入转换成一个概率分布，其输出值的范围在0到1之间，并且各输出值的和为1，这减少了偏置项对模型性能的影响。因此，在这种情况下，省略偏置项可以减少模型的参数数量，提高训练速度，简化模型复杂度，并且有助于避免过拟合，提高模型的泛化能力。

## 5 Self-Attention例子

输入：

```python
”The animal didn't cross the street because it was too tired”
```

这句话中的单词"it"指的是什么呢？它是指“street”还是“animal”呢？

我们当然知道它是指”animal”，但是对于计算机而言，很难辨别。而自注意力机制允许寻找单词之间的相关性，帮助模型将“it”和“animal”放在一起处理。

<figure><img src="../.gitbook/assets/image (26).png" alt=""><figcaption></figcaption></figure>

1.  我们将query和key点乘后得到注意力score。以单词“Thinking”为例，我们需要计算出当前单词“Thinking”的其它所有单词的score。score通过计算query和其它各个单词key的内积得到，因此query和key必须拥有相同的维度。

    例如，在处理单词“Thinking”时，在其自身“Thinking”上的Score为112，而在单词“Machines”上的Score则为96。
2. 将得到的score除以key向量的维度开方，这一步会使得模型在训练时有着更稳定的梯度
3. 对上面的结果进行softmax处理，softmax使得上面的score都为正值，并相加等于1。这里得到的是在处理当前单词的时候每个单词的重要程度
4. 最后，用soft score和value相乘，这一步中，我们保持了那些需要注意的单词的完整性，并冲淡了那些与单词关联性不强的单词

## 6 Self-Attention的问题

1. **缺乏位置信息**：Self-Attention虽然考虑了所有的输入向量，但是没有考虑到向量的位置信息。可以通过位置编码来解决这个问题，将位置信息添加到输入序列中。
2.  **计算复杂度高**：计算复杂度为$$n^2$$，对于长序列，计算成本显著增加

    可以使用优化算法和更高效的硬件（GPU）来加速计算，稀疏注意力等也能减小成本
3.  **对长序列处理能力有限**：虽然自注意力可以处理长序列，但是在实际应用中，可能需要对非常长的序列进行有效处理

    可以使用分段注意力机制或者层次化注意力结构
4. 对小数据集泛化能力有限

## 7 总结

* 解释self-attention
* self-atten和CNN做对比（NLP和CV）
* self-atten和CNN的本质区别
* self-atten的公式
* **为什么要除以$\sqrt{d\_k}$**
* **为什么bias=False**
* self-atten的问题
* 手撕self-atten

## 参考

1. [一文搞定自注意力机制（Self-Attention）-CSDN博客](https://blog.csdn.net/weixin_42110638/article/details/134016569)
2. [attention各种形式总结\_attention公式-CSDN博客](https://blog.csdn.net/qq_41058526/article/details/80783925)
3. [CNN与注意力机制的本质区别 (qq.com)](https://mp.weixin.qq.com/s/XJy0EMFp7HvwBaOmkn5_Yg?poc_token=HLx0wWajBnIDELzlapiDLDrP8dmMlQRmT-qAxtfc)

