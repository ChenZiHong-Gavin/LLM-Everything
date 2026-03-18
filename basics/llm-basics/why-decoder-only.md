# 为什么现在的LLM都是decoder-only架构

### 先搞清楚三种架构

Transformer原论文提出的是Encoder-Decoder架构，后来演化出三条路线：

| 架构                  | 代表模型                         | 注意力机制                       | 典型任务          |
| ------------------- | ---------------------------- | --------------------------- | ------------- |
| **Encoder-Only**    | BERT, RoBERTa                | 双向注意力（每个token能看到所有token）    | 分类、NER、语义理解   |
| **Encoder-Decoder** | T5, BART, Flan-T5            | Encoder双向 + Decoder因果       | 翻译、摘要、seq2seq |
| **Decoder-Only**    | GPT系列, LLaMA, Qwen, DeepSeek | 因果注意力（每个token只能看到它之前的token） | 文本生成、对话、推理    |

三者的核心区别在于**注意力掩码**：Encoder-Only用全连接注意力，Decoder-Only用下三角因果掩码，Encoder-Decoder两者都用。

### Decoder-Only为什么赢了

#### 1. 训练效率：Next Token Prediction的暴力美学

Decoder-Only的训练目标极其简单——给定前面所有token，预测下一个token。这个目标有一个关键优势：**序列中的每一个位置都是一个训练样本**。

一条长度为 N 的序列，一次前向传播就能产生 N−1 个训练信号。而Encoder-Decoder架构中，Encoder端的表示学习依赖Decoder的梯度回传，信号传递路径更长，Encoder部分的参数利用效率相对较低。

BERT那种Masked Language Model（随机遮15%的token去预测）每条样本只有15%的位置产生损失，训练信号的密度远低于自回归目标。

#### 2. 扩展性：大力出奇迹的最佳载体

Scaling Law（Kaplan et al., 2020; Hoffmann et al., 2022）反复验证了一件事：**模型越大、数据越多、算力越多，性能就越好**，而且这个关系是可预测的幂律。

Decoder-Only架构在scaling上有天然优势：

* **架构极简**。只有一个组件（堆叠的Transformer Decoder Block），没有Encoder-Decoder之间的cross-attention，超参数更少，调参空间更小，更容易找到最优配置。
* **参数全部集中在一个模块**。不存在"Encoder分多少层、Decoder分多少层"的分配问题。Raffel et al.（T5论文）做过大量实验，发现Encoder-Decoder在总参数量相同时，并不比Decoder-Only有明显优势，但调参成本高得多。
* **训练基础设施更容易优化**。单一结构意味着更规整的计算图，更容易做tensor并行、pipeline并行、序列并行等分布式策略。

当你要训练万亿参数的模型时，架构的简洁性直接决定工程可行性。

#### 3. 涌现能力：Zero-shot和In-context Learning

GPT-2/GPT-3揭示了一个关键现象：**足够大的自回归语言模型天然具备in-context learning能力**——不需要微调，只需要在prompt里给几个例子，模型就能学会新任务。

这种能力对Decoder-Only特别友好，因为：

* 自回归本身就是一个序列建模过程，prompt里的few-shot examples和实际任务的boundary是模糊的，模型可以无缝地从"理解例子"过渡到"执行任务"。
* BERT类模型的\[MASK]预测范式很难自然地做生成式的in-context learning。
* Encoder-Decoder架构虽然也能做（Flan-T5表现不错），但需要额外的工程设计来处理输入输出的边界。

当in-context learning成为LLM最重要的能力之一后，Decoder-Only就成了天然的最佳选择。

#### 4. 统一范式：一切皆生成

Decoder-Only把所有任务都转化为文本生成：

* 分类 → 生成类别名
* 翻译 → 生成目标语言文本
* 问答 → 生成答案
* 推理 → 生成推理链 + 答案
* 代码 → 生成代码

这种**统一性**意味着：

* 一个模型、一套训练流程、一个推理接口就能覆盖几乎所有NLP任务
* 不需要为每个任务设计特定的head或loss
* 多任务训练变得trivial——只要把不同任务的数据混在一起训就行

相比之下，BERT时代每换一个任务就要加一个分类头再微调，这在LLM时代已经显得笨拙。

#### 5. 推理效率：KV Cache的天然适配

自回归生成时，Decoder-Only可以使用**KV Cache**——已生成token的Key和Value缓存下来，每一步只需要计算新token的注意力，复杂度从 O(n2) 降到 O(n)。

这个优化对Decoder-Only来说是天然的，因为因果掩码保证了已有token的表示不会因为后续token的出现而改变。Encoder-Only或Encoder-Decoder的双向注意力部分无法直接使用这个trick，因为每个新token理论上会改变所有已有token的表示。

在实际部署中，KV Cache是让LLM能以可接受延迟逐token流式输出的关键技术。

#### 6. 数据红利：互联网就是最大的自回归语料库

Next Token Prediction的训练数据就是**原始文本**——不需要标注、不需要配对、不需要人工设计mask策略。互联网上有数万亿token的文本，全都可以直接拿来训练。

Encoder-Decoder架构虽然也能用无监督数据预训练（T5用的是span corruption），但其天然适合的场景是有明确输入输出对的任务（翻译、摘要），这类配对数据的规模远小于纯文本。

当预训练数据从GB级别涨到TB级别时，能最高效利用纯文本数据的架构就是赢家。

### 一些常见疑问

#### Encoder-Decoder真的不行吗？

不是不行。Flan-T5、UL2等模型证明Encoder-Decoder在同等参数量下的表现可以很好，某些任务上甚至更好。但问题是：

1. 工程复杂度更高，scaling更难
2. 社区生态已经完全倾向Decoder-Only，工具链、推理框架、对齐技术（RLHF/DPO）都是为Decoder-Only设计的
3. 在绝对性能上，靠堆参数量的Decoder-Only最终还是会追上来

这是一个**技术选择 + 生态效应**共同导致的结果。

#### 双向注意力不是信息更丰富吗？

是的。理论上双向注意力能捕获更多上下文信息。但实践表明：

* 当模型足够大时，因果注意力通过建模足够长的上下文，也能隐式地学到"双向"的信息
* 在预训练阶段，自回归目标的训练信号密度更高（每个位置都有loss），弥补了单向性的信息损失
* 对于生成任务来说，因果注意力是天然匹配的，强行引入双向注意力反而需要额外处理

#### 有没有可能未来架构会变？

完全可能。一些值得关注的方向：

* **Mamba / State Space Models**：用线性复杂度替代注意力，但核心仍然是自回归生成
* **Mixture of Experts (MoE)**：如DeepSeek-V2/V3、Mixtral，在Decoder-Only基础上做稀疏激活，不改变基础范式
* **Diffusion-based LLM**：用扩散过程替代自回归，理论上可以并行生成，但目前还不成熟
* **Hybrid架构**：比如在Decoder-Only中引入少量双向attention层（部分研究在探索）

但至少目前，Decoder-Only的地位还非常稳固。

### 总结

Decoder-Only赢在**简洁、易扩展、与数据和任务的天然适配**。它不一定是理论上最优的架构，但它是工程实践中最实用的架构。在AI发展的当前阶段——数据充足、算力增长、scaling law有效——这种"大力出奇迹"的路线恰好需要一个最简洁的载体，Decoder-Only就是那个载体。

核心逻辑其实就一句话：**当你有足够的数据和算力时，最简单的方法往往最有效。**
