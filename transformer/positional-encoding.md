# Positional Encoding

在任何一门语言中，词语的位置和顺序对句子意思表达都是至关重要的。传统RNN模型天然有序，在处理句子时，以序列的模式逐个处理句子中的词语，这使得词语的顺序信息在处理过程中被天然的保存下来，并不需要额外的处理。

由于Transformer模型没有RNN或CNN结构，句子中的词语都是同时进入网络进行处理，所以没有明确的关于单词在源句子中位置的相对或绝对的信息。为了让模型理解序列中每个单词的位置（顺序），Transformer论文中提出了使用一种叫做 Positional Encoding（位置编码） 的技术。这种技术通过为每个单词添加一个额外的编码来表示它在序列中的位置，这样模型就能够理解单词在序列中的相对位置。

## 1 位置敏感性

位置敏感性（position-insensitive）的定义是：

* 如果模型的输出会随着输入文本数据顺序的变化而变化，那么这个模型就是关于位置敏感的，反之则是位置不敏感的。
  * RNN和textCNN是位置敏感的
  * 无位置编码的Transformer是位置不敏感的

Positional Encoding的引入，就是用来表征文本中词与词之间的顺序关系，解决Transformer的位置敏感性问题。

## 2 Positional Encoding的概念

Positional Encoding就是将位置信息嵌入到Embedding词向量中，让Transformer保留词向量的**位置信息**，可以提高模型对序列的理解能力。

理想的位置编码应该满足：

1. 为每个时间（序列中token的位置）输出唯一的编码
2. 不同长度的句子之间，任意两个字的差值应该保持一致
3. 外推性：编码值应该是有界的，应该能在不付出任何努力的条件下泛化到更长的句子中

## 3 Positional Encoding的分类

最朴素的想法是为每个token赋予一个数字，如，第一个字为1，第二个字为2。

但是这个数字会随着文本长度变长而变得很大。在推理时，如果没有看过某个特定长度的样本，会导致外推性很弱。

### 3.1 可学习位置编码

`Learned Positional Embedding`方法直接对不同的位置随机初始化一个position embedding，加在word enbedding上输出模型，作为参数进行训练。

<figure><img src="../.gitbook/assets/image (18).png" alt=""><figcaption></figcaption></figure>

BERT使用的位置编码就是可学习位置编码。

### 3.2 sinusoidal位置编码

使用sin，cos交替创建positional encoding，用正余弦表示绝对位置，通过内积来表征相对位置。

$$
PE_{(pos, 2i)} = sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$

$$
PE_{(pos, 2i+1)} = cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$



pos代表token在序列中的位置，假设句子的长度为L，则 $$pos=0,1,\dots,L-1$$。i的范围可以从0取到 $$\frac {d_{model}}{2}$$，最终$$PE$$的维度和$$d_{model}$$一样。

$$
\begin{equation}\overrightarrow{PE}=\left[\begin{array}{c}\sin \left(\omega_1 \cdot pos\right) \\\cos \left(\omega_1 \cdot pos\right) \\\sin \left(\omega_2 \cdot pos\right) \\\cos \left(\omega_2 \cdot pos\right) \\\vdots \\\vdots \\\sin \left(\omega_{d / 2} \cdot pos\right) \\\cos \left(\omega_{d / 2} \cdot t\right)\end{array}\right]_{d \times 1}\end{equation}
$$

* `10000` 是一个经验值，其作用类似于**调节频率范围的超参数**
  * 较小的分母会导致波长较短，过度关注局部信息，难以建立长依赖
  * 较大的分母会导致波长较长，丢失局部细节

**这样的正余弦的组合如何表示一个位置或者顺序？**

假设用一个二进制的形式表示数字：

<figure><img src="../.gitbook/assets/image (19).png" alt=""><figcaption></figcaption></figure>

可以观察到不同位之间变化的速度：最低位之间每个数字交替变化，第二低位每两个数字翻转一次。但是用二进制值会在浮点世界里造成空间浪费。所以反过来，我们可以使用它们对应的浮点连续值——三角函数。实际上，它们等价于交替变换的位。而且，通过降低它们的频率，我们也可以实现从红色的位走到橙色的位。

三角位置编码的另一个特性是，它允许模型毫无费力地把握相对位置。这是一段原论文的引用：

<figure><img src="../.gitbook/assets/image (20).png" alt=""><figcaption></figcaption></figure>

公式：

$$sin(\alpha+\beta)=sin\alpha \cdot cos\beta + cos\alpha\cdot sin\beta$$

$$cos(\alpha+\beta) = cos\alpha\cdot cos\beta - sin\alpha\cdot sin\beta$$

对于位置pos+k的positional encoding：

$$PE_{(pos+k, 2i)} = sin(w_{i}\cdot (pos+k)) = sin(w_{i}pos)cos(w_{i}k)+cos(w_{i}pos)sin(w_{i}k)$$

$$PE_{(pos+k, 2i+1)} = cos(w_{i}\cdot (pos+k)) = cos(w_{i}pos)cos(w_{i}k)-sin(w_{i}pos)sin(w_{i}k)$$

其中，$$w_{i} = \frac{1}{10000^{2i/d_{model}}}$$

则：

$$PE_{(pos+k, 2i)} = cos(w_{i}k)PE_{(pos, 2i)}+sin(w_{i}k)PE_{(pos, 2i+1)}$$

$$PE_{(pos+k, 2i+1)} = cos(w_{i}k)PE_{(pos,2i+1)}-sin(w_{i}k) PE_{(pos, 2i)}$$

因此：

$$\left [\begin{matrix} PE_{(pos+k,2i)} \\ PE_{(pos+k, 2i+1)} \end{matrix} \right]=\left [\begin{matrix} u & v \\ -v & u \end{matrix} \right]\times\left [\begin{matrix} PE_{(pos,2i)} \\ PE_{(pos, 2i+1)} \end{matrix} \right]$$

其中$$u = cos(w_{i}\cdot k), v = sin(w_{i} \cdot k)$$为常数

所以$$PE_{pos+k}$$可以被$$PE_{pos}$$线性表示

计算两者的内积，可以发现：

$$\begin{align} PE_{pos} \cdot PE_{pos+k} &= \sum_{i=0}^{\frac{d}{2}-1}sin(w_ipos)\cdot sin(w_i(pos+k)) + cos(w_ipos) \cdot cos(w_i(pos+k)) \\ &=\sum_{i=0}^{\frac{d}{2}-1} cos(w_i(pos-(pos+k)) \\ &=\sum_{i=0}^{\frac{d}{2}-1} cos(w_ik) \end{align}$$

内积会随着相对位置的递增而减小，从而表征位置的相对距离。

但是由于距离的对称性，虽然能反映相对位置的距离关系，但是无法区分方向。

**位置嵌入还有一种更直觉的解释，就是把它想象成一个钟（因为余弦和正弦就是单元圆的概念）。位置编码的每两个维度可以看成是钟的针（时针、分针、秒针等）。它们从一个位置移动到下一个位置就是在不同的频率下旋转它们的针。所以，尽管没有公式推导，它也能立刻告诉你为什么那个旋转矩阵存在。**







