# 📃 前言

![](.gitbook/assets/Gemini_Generated_Image_nvoawnnvoawnnvoa.png)

## LLM-Everything

**从零开始，系统掌握大语言模型的一切。**

[![GitBook](https://img.shields.io/static/v1?message=Documented%20on%20GitBook\&logo=gitbook\&logoColor=ffffff\&label=%20\&labelColor=5c5c5c\&color=3F89A1)](https://chenzihong.gitbook.io/llm-everything) [![知乎](https://img.shields.io/static/v1?message=%E7%9F%A5%E4%B9%8E%E4%B8%93%E6%A0%8F\&logo=zhihu\&logoColor=ffffff\&label=%20\&labelColor=5c5c5c\&color=0084FF)](https://www.zhihu.com/column/c_1931824303218885390)

***

### ✨ 为什么是这个项目？

市面上不缺 LLM 教程，但缺的是**真正讲明白**的。

* 🎯 **不复制粘贴** — 每篇文章精心打磨，用生动的方式拆解复杂概念
* 🔨 **从零实现代码** — 不只讲理论，带你亲手写出来，在实战中理解原理
* 🗺️ **体系化路线** — 从基础到前沿，完整的学习路径，不再迷路

***

### 📚 知识地图

#### 🎚️ 基础部分

**🐍 Python 基础**

* [logging 模块](basics/python-basics/logging.md)
* [import 模块](basics/python-basics/import.md)
* [multiprocessing 模块](basics/python-basics/multiprocessing.md)

**🐘 机器学习基础**

* 文本表示模型
  * [Bag-of-Words](basics/machine-learning-basics/feature-extraction/text-representation-models/bag-of-words.md)
  * [Topic Model](basics/machine-learning-basics/feature-extraction/text-representation-models/topic-model.md)
  * [Static Word Embeddings](basics/machine-learning-basics/feature-extraction/text-representation-models/static-word-embeddings.md)

**🪿 深度学习基础**

* 🚧 持续更新中...

**🐬 LLM 基础**

* [思考模式切换](basics/llm-basics/switch-thinking.md)
* [为什么现在的LLM都是decoder-only架构](basics/llm-basics/why-decoder-only.md)

#### 🐬 Prompt Engineering

* [Tree of Thoughts](prompt-engineering/tree-of-thoughts.md)

#### 🦖 Transformer 架构

> 逐模块拆解 Transformer

Tokenizer

* [tokenizer.md](transformer/tokenizer.md)

Embeddings

* [ELMo](transformer/embeddings/elmo.md)&#x20;
* [BERT](transformer/embeddings/bert.md)
* [GPT](transformer/embeddings/gpt.md)

Positional Encoding

* [positional-encoding.md](transformer/positional-encoding.md)

Self Attention

* [self-attention.md](transformer/self-attention.md)

Multi-Head Attention

* [multi-head-attention.md](transformer/multi-head-attention.md)

Add & Norm

* [add-and-norm.md](transformer/add-and-norm.md)

FeedForward

* [feedforward.md](transformer/feedforward.md)

Linear & Softmax

* [linear-and-softmax.md](transformer/linear-and-softmax.md)

Decoding Strategy

* [decoding-strategy.md](transformer/decoding-strategy.md)

#### 🎄 LLM 训练

**显存需求**

* [LLM 精度问题](train/llm-vram-needs/llm-precision.md)
* [训练需要多少显存](train/llm-vram-needs/vram_needs_for_llm_training.md)

**分布式并行**

* [数据并行](train/distributed-training-parallelism/data-parallelism.md)&#x20;
* [模型并行](train/distributed-training-parallelism/model-parallelism.md)&#x20;
* [优化器并行](train/distributed-training-parallelism/optimizer-parallelism.md)
* [异构系统并行](train/distributed-training-parallelism/heterogeneous-system-parallelism.md)

**训练流程**

* **数据准备**
  * [课程学习](train/data-preparation/curriculum-learning.md)
* [预训练](train/pre-train.md)
  * [data-engineering.md](train/pre-train/data-engineering.md "mention")
  * [hyper-param.md](train/pre-train/hyper-param.md "mention")
  * [long-text-extension.md](train/pre-train/long-text-extension.md "mention")
  * [evaluation\_and\_engineering.md](train/pre-train/evaluation_and_engineering.md "mention")
* [监督微调](train/sft/)
  * [data\_engineering.md](train/sft/data_engineering.md "mention")
  * [peft.md](train/sft/peft.md "mention")
  * [training-strategy.md](train/sft/training-strategy.md "mention")
  * [evaluation.md](train/sft/evaluation.md "mention")
* [reinforce-learning](train/reinforce-learning/ "mention")
  * [rlhf-basics.md](train/reinforce-learning/rlhf-basics.md "mention")

#### 🐒 MoE（混合专家模型）

* [专家并行](moe/expert-parallelism.md)

#### 🪿 LLM 应用

* [info-retrieval](llm-application/info-retrieval/ "mention")
  * [similarity.md](llm-application/info-retrieval/similarity.md "mention")
  * [text-representation.md](llm-application/info-retrieval/text-representation.md "mention")
* [RAG](llm-application/rag.md)
* [Graph RAG](llm-application/graph-rag.md)

#### 🐢 多模态大模型

* [QFormer](multi-modal-llm/qformer.md)

#### 🔒 LLM 安全

* 🚧 持续更新中...

***

### 🤝 参与贡献

本项目正在快速迭代中，欢迎：

* 🐛 提 Issue 指出错误或疑问
* 🔀 提 PR 补充内容
* ⭐ 觉得有用就给个 Star，这是最大的鼓励

***

**如果这个项目帮到了你，请点个 ⭐ Star 支持一下！**
