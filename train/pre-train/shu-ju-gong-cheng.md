# 数据工程

### 1 如何获取**大规模预训练数据？**

Common Crawl、GitHub、书籍、论文、代码等多源数据。

目前也有很多开源的pretrain数据可以使用：

| 数据集名称                 | 数据类型  | 官方链接                                                                                                                                           | 规模              | 说明/备注                                   |
| --------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | --------------------------------------- |
| **FineWeb**           | 英文网页  | [https://huggingface.co/datasets/HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)                                 | 15T tokens      | HuggingFace 高质量清洗，2013-2024 CC 数据       |
| **FineWeb-2**         | 多语言网页 | [https://huggingface.co/datasets/HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)                             | 1000+ 语言        | FineWeb 多语言版，覆盖稀缺语种                     |
| **FineWeb-Edu**       | 教育网页  | [https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)                         | 1.3T tokens     | 教育内容筛选，适合数学/推理训练                        |
| **RedPajama-V1**      | 综合    | [https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)       | 1.2T tokens     | Together 出品，模仿 LLaMA 配方（CC+GitHub+书+论文） |
| **RedPajama-V2**      | 网页    | [https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)       | 30T tokens（清洗后） | 100T 原始，提供 40+ 质量注释信号                   |
| **SlimPajama**        | 综合    | [https://huggingface.co/datasets/cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B)                           | 627B tokens     | Cerebras 深度清洗版，基于 RedPajama-V1          |
| **RefinedWeb**        | 网页    | [https://huggingface.co/datasets/tiiuae/falcon-refinedweb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)                           | 5T tokens       | TII/Falcon 出品，90% 淘汰率高质量纯网页             |
| **DCLM-Pool**         | 原始网页  | [https://huggingface.co/datasets/mlfoundations/dclm-pool](https://huggingface.co/datasets/mlfoundations/dclm-pool)                             | 340TB 原始        | DeepMind 开源最大原始网页池                      |
| **DCLM-Baseline**     | 网页    | [https://huggingface.co/datasets/mlfoundations/dclm-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline)                     | 7T tokens       | DCLM 清洗后基准版                             |
| **Dolma**             | 综合    | [https://huggingface.co/datasets/allenai/dolma](https://huggingface.co/datasets/allenai/dolma)                                                 | 3T tokens       | AI2 出品，含代码、论文、书籍、百科、Reddit              |
| **Dolma-2**           | 综合    | [https://huggingface.co/datasets/allenai/dolma-v2](https://huggingface.co/datasets/allenai/dolma-v2)                                           | 3.5T tokens     | Dolma 升级版                               |
| **The Pile**          | 综合    | [https://huggingface.co/datasets/EleutherAI/pile](https://huggingface.co/datasets/EleutherAI/pile)                                             | 825GB / 1.4B 文档 | EleutherAI 经典数据集，22 个子集（学术/代码/书）        |
| **The Pile-T5**       | 综合    | [https://huggingface.co/datasets/EleutherAI/pile-t5-base](https://huggingface.co/datasets/EleutherAI/pile-t5-base)                             | 去重版             | T5 训练优化版，已全局去重                          |
| **The Stack**         | 代码    | [https://huggingface.co/datasets/bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack)                                         | 6.4TB，358 语言    | BigCode 出品，permissive 许可证过滤             |
| **The Stack v2**      | 代码+   | [https://huggingface.co/datasets/bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2)                                   | 30TB+           | 含 GitHub Issues/PRs/Kaggle notebooks    |
| **StarCoderData**     | 代码    | [https://huggingface.co/datasets/bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)                                 | 783GB，86 语言     | 含 Issues、Jupyter notebooks、Commits      |
| **The Stack Smol**    | 代码子集  | [https://huggingface.co/datasets/bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol)                               | 每种语言 1 万样本      | 小型实验用子集                                 |
| **Fineweb-Code**      | 代码网页  | [https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-code-corpus](https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-code-corpus) | 55B tokens      | 从 FineWeb 召回的高质量代码相关网页                  |
| **OpenWebMath**       | 数学    | [https://huggingface.co/datasets/open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math)                     | 14.7B tokens    | 数学网页提取自 CC                              |
| **Fineweb-Math**      | 数学    | [https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-math-corpus](https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-math-corpus) | 数学相关            | 从 FineWeb 召回的数学内容                       |
| **FineMath**          | 数学    | [https://huggingface.co/datasets/HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath)                               | -               | HuggingFace 数学数据集                       |
| **Proof-Pile-2**      | 数学证明  | [https://huggingface.co/datasets/EleutherAI/proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2)                             | -               | EleutherAI 数学与形式化证明数据                   |
| **MegaMath**          | 数学    | [https://huggingface.co/datasets/batch-dit/MegaMath](https://huggingface.co/datasets/batch-dit/MegaMath)                                       | 370B tokens     | 史上最大数学预训练集（2014-2024）                   |
| **Algebraic Stack**   | 数学代码  | [https://huggingface.co/datasets/EleutherAI/proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2)                             | -               | 数学论文、形式化代码、教材                           |
| **CCI** (中文)          | 中文网页  | [https://huggingface.co/datasets/BAAI/CCI-Data](https://huggingface.co/datasets/BAAI/CCI-Data)                                                 | 104GB           | 智源研究院中文互联网语料库                           |
| **CCI2** (中文)         | 中文网页  | [https://huggingface.co/datasets/BAAI/CCI2-Data](https://huggingface.co/datasets/BAAI/CCI2-Data)                                               | 501GB           | CCI 升级版                                 |
| **CCI3** (中文)         | 中文网页  | [https://huggingface.co/datasets/BAAI/CCI3-Data](https://huggingface.co/datasets/BAAI/CCI3-Data)                                               | -               | CCI 最新版（2023.01-06）                     |
| **ChineseWebText**    | 中文网页  | [https://huggingface.co/datasets/CASIA-LM/ChineseWebText2.0](https://huggingface.co/datasets/CASIA-LM/ChineseWebText2.0)                       | -               | 中科院自动化所高质量中文网页                          |
| **MAP-CC**            | 多语言中文 | [https://huggingface.co/datasets/m-a-p/MAP-CC](https://huggingface.co/datasets/m-a-p/MAP-CC)                                                   | 2000 语言+中文      | M-A-P 多语言中文语料                           |
| **SkyPile-150B**      | 中文    | [https://huggingface.co/datasets/Skywork/SkyPile-150B](https://huggingface.co/datasets/Skywork/SkyPile-150B)                                   | 150B tokens     | Skywork 中文预训练数据                         |
| **TigerBot-pretrain** | 中文    | [https://huggingface.co/datasets/TigerResearch/tigerbot-pretrain](https://huggingface.co/datasets/TigerResearch/tigerbot-pretrain)             | 2TB（开源 100GB）   | TigerBot 中文预训练集（书+百科+网页）                |
| **WanJuan**           | 中文综合  | [https://opendatalab.org.cn/OpenDataLab/WanJuan1.0](https://opendatalab.org.cn/OpenDataLab/WanJuan1.0)                                         | -               | 上海 AI Lab 万卷数据集（需官网下载）                  |
| **TeleChat-PTD**      | 中文    | [https://huggingface.co/datasets/Tele-AI/TeleChat-PTD](https://huggingface.co/datasets/Tele-AI/TeleChat-PTD)                                   | -               | 电信 AI 中文预训练数据                           |
| **PeS2o**             | 学术论文  | [https://huggingface.co/datasets/allenai/PeS2o](https://huggingface.co/datasets/allenai/PeS2o)                                                 | 30B tokens      | AI2 学术论文数据集（OpenAlex 衍生）                |
| **WebStories**        | 故事/创意 | [https://huggingface.co/datasets/HuggingFaceFW/webstories](https://huggingface.co/datasets/HuggingFaceFW/webstories)                           | -               | HuggingFace 创意写作数据集                     |
| **Dolma-Flan**        | 指令数据  | [https://huggingface.co/datasets/allenai/dolma-flan](https://huggingface.co/datasets/allenai/dolma-flan)                                       | -               | Dolma 配套指令微调数据                          |
| **CulturaX**          | 多语言   | [https://huggingface.co/datasets/uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)                                               | 6.3TB，167 语言    | 多语言预训练数据集                               |

**选多少数据量比较合适？**

根据 DeepMind 在2022年的Chinchilla模型实验，在给定的计算预算（FLOPs）下，模型参数量（N）与训练数据量（D）应以相同比例缩放，具体比例为1:20，即每 1B 参数需匹配 20BB  的训练数据。

### 2 启发式过滤 数据过滤

#### 2.1 启发式过滤



* **质量控制**：启发式过滤、基于模型的质量打分（如 FastText/Perplexity 过滤）、隐私脱敏（PII 移除）

### 数据去重

去重策略（MinHash/LSH）

### 数据配比

不同领域数据的最优混合比例（DoReMi、课程学习）

### Tokenizer **再训练**

* **Tokenization 再训练**：大规模语料下 BPE/SentencePiece 的训练与词汇表设计考量
