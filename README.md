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

| <p><strong>🐍 Python 基础</strong></p><ul><li><a href="basics/python-basics/logging.md">logging 模块</a></li><li><a href="basics/python-basics/import.md">import 模块</a></li><li><a href="basics/python-basics/multiprocessing.md">multiprocessing 模块</a></li></ul> | <p><strong>🐘 机器学习基础</strong></p><ul><li><p>文本表示模型</p><ul><li><a href="basics/machine-learning-basics/feature-extraction/text-representation-models/bag-of-words.md">Bag-of-Words</a></li><li><a href="basics/machine-learning-basics/feature-extraction/text-representation-models/topic-model.md">Topic Model</a></li><li><a href="basics/machine-learning-basics/feature-extraction/text-representation-models/static-word-embeddings.md">Static Word Embeddings</a></li></ul></li></ul> |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <p><strong>🪿 深度学习基础</strong></p><ul><li>🚧 持续更新中...</li></ul>                                                                                                                                                                                                 | <p><strong>🐬 LLM 基础</strong></p><ul><li><a href="basics/llm-basics/switch-thinking.md">思考模式切换</a></li><li><a href="basics/llm-basics/why-decoder-only.md">为什么现在的LLM都是decoder-only架构</a></li></ul>                                                                                                                                                                                                                                                                                          |

#### 🐬 Prompt Engineering

* [Tree of Thoughts](prompt-engineering/tree-of-thoughts.md)

#### 🦖 Transformer 架构

> 逐模块拆解 Transformer，从输入到输出，一个不落。

| 模块                   | 链接                                                                                                                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tokenizer            | [tokenizer.md](transformer/tokenizer.md)                                                                                                                                               |
| Embeddings           | <ul><li><a href="transformer/embeddings/elmo.md">ELMo</a> </li><li><a href="transformer/embeddings/bert.md">BERT</a></li><li><a href="transformer/embeddings/gpt.md">GPT</a></li></ul> |
| Positional Encoding  | [positional-encoding.md](transformer/positional-encoding.md)                                                                                                                           |
| Self Attention       | [self-attention.md](transformer/self-attention.md)                                                                                                                                     |
| Multi-Head Attention | [multi-head-attention.md](transformer/multi-head-attention.md)                                                                                                                         |
| Add & Norm           | [add-and-norm.md](transformer/add-and-norm.md)                                                                                                                                         |
| FeedForward          | [feedforward.md](transformer/feedforward.md)                                                                                                                                           |
| Linear & Softmax     | [linear-and-softmax.md](transformer/linear-and-softmax.md)                                                                                                                             |
| Decoding Strategy    | [decoding-strategy.md](transformer/decoding-strategy.md)                                                                                                                               |

#### 🎄 LLM 训练

| 主题        | 内容                                                                                                                                                                                                                                                                                                                                                                                           |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **显存需求**  | <ul><li><a href="train/llm-vram-needs/llm-precision.md">LLM 精度问题</a></li><li><a href="train/llm-vram-needs/vram_needs_for_llm_training.md">训练需要多少显存</a></li></ul>                                                                                                                                                                                                                            |
| **分布式并行** | <ul><li><a href="train/distributed-training-parallelism/data-parallelism.md">数据并行</a> </li><li><a href="train/distributed-training-parallelism/model-parallelism.md">模型并行</a> </li><li><a href="train/distributed-training-parallelism/optimizer-parallelism.md">优化器并行</a></li><li><a href="train/distributed-training-parallelism/heterogeneous-system-parallelism.md">异构系统并行</a></li></ul> |
| **训练流程**  | <ul><li><p><a href="train/pre-train.md">预训练</a></p><ul><li><a data-mention href="train/pre-train/data-engineering.md">data-engineering.md</a></li><li><a data-mention href="train/pre-train/hyper-param.md">hyper-param.md</a></li></ul></li><li><a href="train/sft.md">监督微调</a></li><li>强化学习 🚧</li></ul>                                                                                   |
| **数据准备**  | [课程学习](train/data-preparation/curriculum-learning.md)                                                                                                                                                                                                                                                                                                                                        |

#### 🐒 MoE（混合专家模型）

* [专家并行](moe/expert-parallelism.md)

#### 🪿 LLM 应用

* [RAG](llm-application/rag.md)
* [Graph RAG](llm-application/graph-rag.md)

#### 🐢 多模态大模型

* [QFormer](multi-modal-llm/qformer.md)

#### 🔒 LLM 安全

* 🚧 持续更新中...

***

### 🛣️ 推荐学习路线

```
基础部分 → Prompt Engineering → LLM 应用
   ↓
Transformer 架构 → LLM 训练
                      ↓
                MoE / 多模态 / 安全
```

***

### 🤝 参与贡献

本项目正在快速迭代中，欢迎：

* 🐛 提 Issue 指出错误或疑问
* 🔀 提 PR 补充内容
* ⭐ 觉得有用就给个 Star，这是最大的鼓励

***

**如果这个项目帮到了你，请点个 ⭐ Star 支持一下！**
