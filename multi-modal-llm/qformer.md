# Qformer

传统的视觉-语言预训练（VLP）成本极高，因为需要端到端训练视觉和文本模型。BLIP-2 提出了一种通用且高效的预训练策略，利用现成的冻结预训练图像编码器和冻结大语言模型（LLM），只训练一个轻量级的"桥梁"模块来连接两者，即Qformer（Querying Transformer）。

<figure><img src="../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

Qformer是一个轻量级Transformer，作为冻结图像模型和冻结LLM之间的信息瓶颈：

* 使用32个可学习的查询向量（Queries）从冻结的图像编码器中提取视觉特征
* 采用"瓶颈"设计（32×768维），远小于原始图像特征（如257×1024维），强制提取与语言最相关的视觉信息
* 包含图像Transformer（与冻结编码器交互）和文本Transformer（处理文本）

### 1 Qformer架构

<figure><img src="../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

Q-Former模块可以实例化为可学习的查询嵌入集合 $$\mathbf{Q}\in\mathbb{R}^{m\times d}$$，其中 $$m$$ 为查询数量，$$d$$ 为模型维度。每个Q-Former块由以下顺序层组成：

* **多头自注意力**（作用于查询，用于查询间上下文关联）
* **多头交叉注意力**（从查询指向主干网络特征，即ViT、语言词元）
* **前馈网络**（MLP）
* **层归一化和残差连接**（每个核心操作后）

过程如下：&#x20;

$$
\begin{align*} \mathbf{Q}^1 &= \mathbf{Q}^0 + \mathrm{MSA}(\mathrm{LN}(\mathbf{Q}^0)) \\ \mathbf{Q}^2 &= \mathbf{Q}^1 + \mathrm{CrossAttn}(\mathrm{LN}(\mathbf{Q}^1),\mathbf{E}) \\ \mathbf{Z} &= \mathbf{Q}^2 + \mathrm{FFN}(\mathrm{LN}(\mathbf{Q}^2)) \end{align*}
$$

### 2 两阶段训练

Q-Former并非直接接入LLM，而是通过**渐进式训练**避免灾难性遗忘：

**阶段一：表征学习（Bootstrapping Vision-Language Representation）**

* 仅训练Q-Former，冻结视觉编码器（ViT-G/14等）
* 使用图文对数据集（COCO, Visual Genome等）
* 学习视觉-文本对齐的**紧凑表征**（32个查询向量）

Q-Former在阶段一同时优化三个互补任务：

| 目标      | 全称                              | 作用        | 关键细节                                      |
| ------- | ------------------------------- | --------- | ----------------------------------------- |
| **ITC** | Image-Text Contrastive Learning | 对齐视觉和文本表征 | 计算查询向量与文本\[CLS] token的相似度，使用MoCo机制维护动量编码器 |
| **ITM** | Image-Text Matching             | 细粒度对齐     | 二分类任务，查询向量通过Bi-Attention与文本交互，判断图文是否匹配    |
| **ITG** | Image-Grounded Text Generation  | 生成能力      | 因果LM目标，查询向量作为视觉前缀，自回归生成文本（仅阶段二使用）         |

**阶段二：生成学习（Bootstrapping Vision-to-Language Generative Learning）**

* 将Q-Former连接到**冻结的LLM**（OPT, Flan-T5等）
* 添加全连接层将Q-Former输出投影到LLM的嵌入维度
* 训练目标变为：让LLM基于Q-Former提取的视觉特征生成文本描述



### 参考

1. [https://www.emergentmind.com/articles/transformer-based-adapters-q-former](https://www.emergentmind.com/articles/transformer-based-adapters-q-former)
2. [\[2301.12597\] BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)

<br>
