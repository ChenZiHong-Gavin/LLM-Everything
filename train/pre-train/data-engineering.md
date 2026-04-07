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

根据 DeepMind 在2022年的Chinchilla模型实验，在给定的计算预算（FLOPs）下，模型参数量（N）与训练数据量（D）应以相同比例缩放，具体比例为1:20，即每 1B 参数需匹配 20B  的训练数据。

**不同文本的知识密度是有差异的**

在 RedPajama-V2 等数据集中，知识密度通过 **Wikipedia 训练的 n-gram 语言模型困惑度** 来衡量，

* **Head（头部）**：低困惑度（perplexity 低），语言规范、信息密集（如 Wikipedia、教科书）
* **Middle（中部）**：中等困惑度，一般网页内容
* **Tail（尾部）**：高困惑度（perplexity 高），低质量文本（广告、垃圾信息、随机字符串）

### 2 数据过滤

#### 2.1 启发式过滤

典型过滤条件有：

* 语言
* 长度：太短文本（如导航栏、页脚），过长文本（可能是抓取错误）
* 符号与格式：符号占比过高（如乱码、表情符号堆叠、大量停用词、HTML格式残留）

#### 2.2 基于模型过滤

基于模型的过滤分为标签过滤和分数过滤，标签过滤就是去掉你不想要的文本类别（比如代码类），分数过滤就是去掉低质量数据，或按分数加权采样。

标签过滤的方式：

* 使用 FastText 模型分类：FastText 是微软开源的模型，将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类。
* 自己训练分类器：通常使用 BERT 之类的 encoder-decoder 模型，训练一个分类器

**标签体系设计建议**

* **粗粒度**（5-10类）：如 `academic`, `code`, `news`, `forum`, `novel`, `ads`, `spam`
* **细粒度**（20+类）：如 `arxiv_cs`, `arxiv_math`, `github_python`, `github_js`, `wikipedia`, `textbook`, `patent`, `law`, `medical`

分数过滤的方式

* Perplexity 过滤：计算文档困惑度作为质量指标。困惑度越低，说明文本越符合自然语言分布规律（通常为高质量书写文本）。**PPL越低越好**：表示模型对下一个词的预测越准确。
  * PPL（Perplexity，**困惑度**）的标准公式为：

$$\text{PPL}(W) = P(w_1, w_2, \ldots, w_N)^{-\frac{1}{N}} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i|w_1\ldots w_{i-1})\right)$$

* 自己训练一个打分器，训练方法同上文的标签过滤
* LLM-as-Judge：最简单也是最烧钱的方法，用更好的LLM作为打分器

#### 2.3 隐私脱敏

**PII** = **Personally Identifiable Information**（个人身份信息），指任何可用于识别、定位或联系特定个人的数据。

常见的需要脱敏的数据如下：

| 类别         | 示例                     | 检测方法                |
| ---------- | ---------------------- | ------------------- |
| **基础标识**   | 姓名、身份证号、手机号、邮箱、地址      | 正则表达式 + 字典匹配        |
| **金融信息**   | 银行卡号、信用卡CVV、交易记录       | 校验位算法（Luhn算法）+ 模式匹配 |
| **网络标识**   | IP地址、MAC地址、Cookie、设备ID | 正则 + 网络协议解析         |
| **生物特征**   | 指纹、面部识别数据、DNA序列        | 文件头检测（通常以二进制存储）     |
| **上下文PII** | "我住在朝阳区"（需结合说话人身份）     | NER模型 + 依存句法分析      |

### 3 数据去重

训练语料中的重复数据会导致训练效率下降、模型过拟合重复文本，降低泛化能力。

预训练中数据去重的级别一版分为：

* **文档级：**&#x6574;篇文档的完全或近似重复
* **段落级：**&#x6587;档内或跨文档的段落重复
* **子串级：**&#x8FDE;续字符串有重复

除了硬匹配和一些常见的相似度检测算法，去重常用的算法是 MinHash-LSH：

#### &#x20;MinHash 算法原理

**数学基础：Jaccard 相似度估计**

对于两个集合 $$A$$ 和 $$B$$，Jaccard 相似度为： $$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$$

直接计算 Jaccard 需要 $$O(|A|+|B|)$$ 的存储和计算，**MinHash** 通过**概率 sketch** 将复杂度降至 $$O(1)$$。

**算法步骤**：

1.  **Shingling（分片）**：将文档切分为连续的 $$k$$-gram 集合

    ```python
    # 5-gram 字符级 shingles（推荐代码/网页数据）
    def get_shingles(text, k=5):
        return {text[i:i+k] for i in range(len(text)-k+1)}
    # 示例："hello world" → {'hello', 'ello ', 'llo w', 'lo wo', 'o wor', ' worl', 'world'}
    ```
2.  **Hash 签名生成**： 使用 $$n$$ 个独立的哈希函数 $$h_1, h_2, ..., h_n$$，对每个 shingle 计算哈希值。

    MinHash 签名 $$sig(A)$$ 的第 $$i$$ 个分量是： $$sig_i(A) = \min_{x \in A} h_i(x)$$

    即：对每个哈希函数，取集合 $$A$$ 中所有元素哈希值的**最小值**。
3. **相似度估计**： $$J_{est}(A,B) = \frac{\text{matches}}{n}$$ 其中 matches 是 $$sig(A)$$ 和 $$sig(B)$$ 中相等位置的数量。

**概率保证**：估计误差为 $$O(1/\sqrt{n})$$，通常 $$n=128$$ 或 $$256$$ 可提供足够精度。

**LSH（局部敏感哈希）加速**

**问题**：当文档数量达到十亿级时，两两比较 MinHash 签名（$$O(N^2)$$）不可行。

**LSH 策略**：将签名分桶，仅比较同一桶内的文档。

### 4 数据配比

前面我们训练的分类器，再这里也能够派上用场。

大部分中文模型的数据配比为：中：英：code = 4:4:2。

也可以使用一些方法来优化数据配比，例如DoReMi。

#### 4.1 DoReMi

DoReMi 是 Google DeepMind 开源的一套自动调整数据配比的方法，简单来说就是**多采样模型相对学得不好的数据。**

DoReMi 通过比较参考模型（Reference Model）**与**生产模型（Proxy Model）的性能差距（Excess Loss）来动态调整权重：

$$\text{Excess Loss}_d = \mathbb{E}_{x \sim P_d}[-\log Q_\alpha(x)] - \mathbb{E}_{x \sim P_d}[-\log Q_{ref}(x)]$$

其中：

* $$Q_{ref}$$：在**均匀分布**数据上训练的小模型（捕捉基线能力）
* $$Q_\alpha$$：在当前权重 $$\alpha$$ 下训练的小模型（验证者）
* **核心洞察**：若某领域的 Excess Loss > 0（生产模型比参考模型差），说明该领域训练不足，需**增加采样权重**

#### 算法流程

**Step 1 - 训练参考模型**：在均匀采样（各领域等概率）的数据上训练一个小规模参考模型（通常为主模型 1/10 规模，如 280M 参数），记录各领域验证集上的基线损失 $$L_{ref}^{(d)}$$。

**Step 2 - 迭代优化权重**（通常 3-5 轮）：

1. 按当前权重 $$\alpha$$ 采样数据，训练同等规模的代理模型（Proxy Model）
2. 计算各领域 Excess Loss（代理模型损失 - 参考模型损失）
3. **权重更新**：对 Excess Loss 大的领域增加权重（Multiplicative Weights Update）： $$\alpha_d^{t+1} \propto \alpha_d^t \cdot \exp(\eta \cdot \text{Excess Loss}_d)$$
4. 归一化权重并迭代，直至权重收敛（最大变化 < 1%）或 Max Excess Loss 不再降低

**Step 3 - 应用最优权重**：将收敛后的权重应用于大规模模型（如 7B/70B）的实际训练。



### 参考

1. [https://arxiv.org/abs/2305.10429](https://arxiv.org/abs/2305.10429)
