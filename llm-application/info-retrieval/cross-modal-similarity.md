# 跨模态相似度

### 1 跨模态相似度

现实世界的信息不只有文字。你在淘宝搜"红色连衣裙"，返回的是图片；你在 Google 看到一张猫的照片，脑子里自动浮现"猫"这个概念。**人类天然具备跨模态对齐的能力**，我们能轻松地在文字和图像之间建立语义关联。

对机器来说，这件事一点也不轻松：

文本可以被编码成离散的 token 序列，在语言语义空间中表征；图像则是连续的像素矩阵，在视觉特征空间中表征。文本编码器输出的向量和图像编码器输出的向量，**天生就不在同一个空间里，**&#x76F4;接算余弦相似度毫无意义。

### 2 CLIP

**CLIP**（Contrastive Language-Image Pre-training）是 OpenAI 在 2021 年发布的模型。CLIP **用对比学习的方式，预训练一个语言-图像模型**。

#### 2.1 双塔架构

<figure><img src="../../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

* **Image Encoder**：接收一张图片，输出一个 d 维向量
* **Text Encoder**：接收一段文本，输出一个 d 维向量
* 两个向量**投影到同一个 d 维空间**，然后直接算余弦相似度

两个编码器互不干扰，但最终的输出被**对齐到同一个语义空间**。这就像两个翻译官，一个懂图像，一个懂文字，但他们把翻译结果都写成同一种"通用语言"。

CLIP 的训练数据叫 **WIT**（WebImageText），包含从互联网上爬取的约 **4 亿个图文配对**。

这些是网页上自然存在的图文关系：

* 图片的 alt 文本
* 图片附近的描述文字
* 社交媒体上图片配的文案

**CLIP 利用互联网的天然监督信号来学习**。数据虽然有噪声，但胜在量大、覆盖面广。

#### 2.2 对比学习：InfoNCE Loss

训练的核心是**对比损失**（Contrastive Loss），具体使用的是 InfoNCE Loss 的变体。

假设一个 batch 里有 N 个图文对。每个图文对 (I\_i, T\_i) 是正确的配对。训练目标是：

> **让正确配对的相似度高，让错误配对的相似度低。**

用一个 N×N 的相似度矩阵来理解：

```
              T_1    T_2    T_3    ...    T_N
    I_1    [ 0.9   0.1    0.05   ...   0.02 ]   ← 对角线是正样本
    I_2    [ 0.05  0.85   0.1    ...   0.03 ]
    I_3    [ 0.02  0.08   0.92   ...   0.01 ]
    ...
    I_N    [ 0.01  0.03   0.02   ...   0.88 ]
```

* **对角线**上的值（I\_i 和 T\_i 的相似度）应该尽量**大**
* **非对角线**上的值（I\_i 和 T\_j 的相似度，i≠j）应该尽量**小**

损失函数同时从两个方向优化：

**图像→文本方向**（给定一张图，找对的文本）：

$$\mathcal{L}_{i2t} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}$$

**文本→图像方向**（给定一段文本，找对的图像）：

$$\mathcal{L}_{t2i} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(T_i, I_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(T_i, I_j) / \tau)}$$

最终损失取平均：

$$\mathcal{L} = \frac{1}{2}(\mathcal{L}_{i2t} + \mathcal{L}_{t2i})$$

其中 τ（temperature）是一个可学习的温度参数，控制分布的"尖锐程度"。

**直觉理解**：这其实就是一个 N 类分类问题。给定图像 I\_i，在 N 个候选文本中把正确的 T\_i "分类"出来；反过来也一样。batch size 越大，负样本越多，学到的表示越好——CLIP 原始训练的 batch size 是 **32768**。

#### 2.4 矩阵化的高效实现

对比学习的美妙之处在于它的计算可以高度并行化：

```python
# 伪代码
# 1. 编码
image_features = image_encoder(images)        # (N, d)
text_features = text_encoder(texts)           # (N, d)

# 2. L2 归一化
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 3. 计算相似度矩阵
logits = (image_features @ text_features.T) * exp(log_temperature)  # (N, N)

# 4. 对称的交叉熵损失
labels = torch.arange(N)                      # [0, 1, 2, ..., N-1]
loss_i2t = cross_entropy(logits, labels)       # 每行的目标是对角线位置
loss_t2i = cross_entropy(logits.T, labels)     # 每列的目标是对角线位置
loss = (loss_i2t + loss_t2i) / 2
```

一个矩阵乘法 `@` 就算出了所有 N² 个相似度——这就是 batch 对比学习的效率所在。

***

### 三、图像编码器：ViT 的结构与工作方式

CLIP 的图像编码器有两个版本：ResNet 和 **ViT**（Vision Transformer）。ViT 版本效果更好，也是后续被广泛采用的版本。

#### 3.1 ViT 的核心想法

ViT 的想法出奇地简单：**把图像当成一句话来处理**。

具体做法：

1. **切分 patch**：把一张 224×224 的图像切成 16×16 的小块（patch），得到 (224/16)² = **196 个 patch**
2. **线性投影**：每个 patch 被展平后通过一个线性层，映射成一个 d 维向量——这就是图像的 "token"
3. **加上位置编码**：告诉模型每个 patch 在图像中的位置
4. **加上 \[CLS] token**：在序列开头插入一个特殊的分类 token
5. **送入 Transformer**：标准的多层 Transformer encoder 处理这 197 个 token
6. **取 \[CLS] 输出**：最终 \[CLS] token 的输出向量就是整张图像的表示

```
原始图像 (224×224×3)
    │
    ▼  切分为 16×16 的 patch
[p1] [p2] [p3] ... [p196]
    │
    ▼  线性投影 + 位置编码
[CLS] [e1] [e2] [e3] ... [e196]    ← 197 个 token
    │
    ▼  Transformer Encoder × L 层
[CLS'] [e1'] [e2'] [e3'] ... [e196']
    │
    ▼  取 CLS token
  图像向量 (d 维)
```

#### 3.2 不同尺寸的 ViT

CLIP 提供了多种规格：

| 模型           | patch 大小         | 层数 | 隐藏维度 | 参数量    |
| ------------ | ---------------- | -- | ---- | ------ |
| ViT-B/32     | 32×32            | 12 | 768  | \~86M  |
| ViT-B/16     | 16×16            | 12 | 768  | \~86M  |
| ViT-L/14     | 14×14            | 24 | 1024 | \~304M |
| ViT-L/14@336 | 14×14 (输入 336px) | 24 | 1024 | \~304M |

patch 越小，图像被切得越细，信息保留越多，效果越好——但计算量也更大（token 数量是 patch 大小平方的反比）。

***

### 四、文本编码器：Transformer + BPE

#### 4.1 结构

CLIP 的文本编码器是一个标准的 **Transformer**（类似 GPT 的 decoder-only 结构，使用 causal attention mask）：

* **Tokenizer**：使用 BPE（Byte Pair Encoding）分词，词表大小约 49152
* **最大长度**：77 个 token（包括 \[SOT] 和 \[EOT]）
* **输出**：取 \[EOT] token 位置的隐藏状态作为文本的整体表示

```
"a photo of a cat"
    │
    ▼  BPE Tokenizer
[SOT] a photo of a cat [EOT] [PAD] [PAD] ...
    │
    ▼  Token Embedding + Position Embedding
    │
    ▼  Transformer × L 层 (causal mask)
    │
    ▼  取 [EOT] 位置的输出
  文本向量 (d 维)
```

#### 4.2 投影到共享空间

Image Encoder 和 Text Encoder 的输出维度可能不一样。CLIP 在两者之后各加了一个**线性投影层**（projection head），把它们投影到同一个 d 维空间（比如 512 维或 768 维），然后做 L2 归一化，再算余弦相似度。

```
Image Encoder (输出 1024 维) ──▶ Linear(1024, 512) ──▶ L2 Norm ──▶ 图像向量 (512 维)
Text Encoder  (输出 768 维)  ──▶ Linear(768, 512)  ──▶ L2 Norm ──▶ 文本向量 (512 维)
```

***

### 五、CLIP 能做什么

CLIP 训练完之后，你得到的是两个对齐的编码器。这意味着你可以**自由组合**图像和文本来做各种事。

#### 5.1 零样本图片分类（Zero-Shot Classification）

这是 CLIP 最惊艳的能力：**不需要任何标注数据，就能对图片进行分类**。

传统的图片分类是这样的：

1. 收集大量标注数据（比如 1000 类 ImageNet，每类 1000+ 张图）
2. 训练一个分类器
3. 只能识别这 1000 个类别

CLIP 的做法完全不同：

```
假设你要分类一张图片，候选类别是 [猫, 狗, 鸟]

1. 构造文本 prompt：
   "a photo of a cat"
   "a photo of a dog"
   "a photo of a bird"

2. 用 Text Encoder 编码这 3 个 prompt → 3 个文本向量

3. 用 Image Encoder 编码待分类图片 → 1 个图像向量

4. 计算图像向量与 3 个文本向量的余弦相似度

5. 相似度最高的那个就是分类结果
```

```python
import torch
from PIL import Image
import clip

# 加载模型
model, preprocess = clip.load("ViT-B/32")

# 准备图像
image = preprocess(Image.open("cat.jpg")).unsqueeze(0)

# 准备候选标签
texts = clip.tokenize(["a photo of a cat", "a photo of a dog", "a photo of a bird"])

# 编码
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(texts)

# 归一化 + 计算相似度
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (image_features @ text_features.T).softmax(dim=-1)

print(similarity)
# tensor([[0.92, 0.05, 0.03]])  → 猫！
```

这意味着：

* **类别可以随时更改**——不需要重新训练
* **类别可以是自然语言描述**——不局限于单个词
* 在 ImageNet 上，CLIP 的零样本准确率可以和专门训练的 ResNet-50 媲美

#### 5.2 文搜图 / 图搜文

**文搜图**：用户输入文本 query，在图片库中找最相关的图片。

```
流程：
1. 离线：把所有图片用 Image Encoder 编码成向量，存入向量数据库
2. 在线：用户输入文本 → Text Encoder 编码 → 在向量库中检索最近邻

"sunset over the ocean"  →  [0.23, -0.15, 0.87, ...]
                                    │
                                    ▼  向量检索
                              返回最相似的图片
```

**图搜文**：给定一张图片，找最相关的文本描述。流程完全对称。

这正是前几篇聊的向量检索技术的**跨模态版本**——只是现在查询和文档可以是不同模态了。

#### 5.3 图文匹配打分

给定任意一对 (图片, 文本)，CLIP 可以输出一个相似度分数，表示它们的语义匹配程度。

应用场景：

* **内容审核**：判断图片和标题是否一致
* **推荐系统**：衡量用户搜索词和商品图的匹配度
* **数据清洗**：在图文数据集中过滤掉不匹配的样本
* **生成模型的评估**：CLIP Score 被广泛用于评价文本生成图像的质量

***

### 六、Chinese-CLIP：适配中文场景

原始 CLIP 主要在英文数据上训练，对中文的理解有限。**Chinese-CLIP** 是由阿里达摩院提出的中文适配版本。

#### 6.1 核心改动

|       | 原始 CLIP        | Chinese-CLIP                 |
| ----- | -------------- | ---------------------------- |
| 文本编码器 | 英文 Transformer | **RoBERTa-wwm-ext**（中文预训练模型） |
| 图像编码器 | ViT（从头训练）      | ViT（**用 CLIP 原始权重初始化**）      |
| 训练数据  | 4 亿英文图文对       | \~2 亿中文图文对                   |
| 训练策略  | 端到端            | **两阶段**：先锁图训文，再联合微调          |

#### 6.2 两阶段训练

这是 Chinese-CLIP 的一个巧妙设计：

**第一阶段（Lock-image Tuning）**：

* 冻结 Image Encoder（保留 CLIP 学到的强视觉特征）
* 只训练 Text Encoder，让中文文本对齐到已有的视觉空间
* 这比从头训练高效得多

**第二阶段（联合微调）**：

* 解冻两个编码器，端到端微调
* 在中文数据上进一步优化

#### 6.3 使用示例

```python
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
import torch
from PIL import Image

# 加载模型
model, preprocess = load_from_name("ViT-B-16", device="cuda")
model.eval()

# 编码图像
image = preprocess(Image.open("猫.jpg")).unsqueeze(0).to("cuda")

# 编码中文文本
texts = clip.tokenize(["一只猫", "一条狗", "一辆车"]).to("cuda")

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(texts)
    # 归一化
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # 相似度
    similarity = (image_features @ text_features.T).softmax(dim=-1)

print(similarity)
# tensor([[0.95, 0.03, 0.02]])  → 一只猫！
```

其他中文/多语言变体还有：

* **AltCLIP**（百川）：基于 XLMR 的多语言文本编码器
* **Taiyi-CLIP**（IDEA 研究院）：中文 CLIP
* **multilingual-CLIP**：社区版多语言适配

***

### 七、CLIP 的局限与后续发展

CLIP 虽然强大，但远非完美。

#### 7.1 CLIP 的局限

**1. 细粒度理解不足**

CLIP 擅长全局语义匹配，但对细节不敏感：

```
"一个红色杯子在蓝色盘子上"  vs  "一个蓝色杯子在红色盘子上"
→ CLIP 可能给出相近的分数，因为它主要捕捉了"杯子""盘子""红""蓝"的共现
```

属性绑定（attribute binding）和空间关系是 CLIP 的弱项。

**2. 组合理解能力弱**

CLIP 对否定、数量、组合推理的理解有限：

* "没有猫的房间" vs "有猫的房间" → 相似度可能很接近
* "两只猫" vs "三只猫" → 难以区分

**3. 文本长度限制**

最大 77 个 token，无法处理长文本。对于需要复杂描述的场景力不从心。

**4. 训练数据的偏差**

互联网爬取的数据存在各种偏差（文化、语言、地域），CLIP 会继承这些偏差。

**5. 只能做检索，不能生成**

CLIP 是一个判别模型——它能告诉你图文是否匹配，但不能根据文本生成图像（不过它可以给生成模型当"裁判"）。

#### 7.2 后续发展

CLIP 之后，社区涌现了大量改进工作：

**BLIP / BLIP-2（Salesforce）**

```
CLIP：双塔，各编各的，只在最后比相似度
BLIP：双塔 + 交叉注意力，图文之间有深度交互
```

* 引入了 Image-Text Matching (ITM) 和 Image-grounded Text Generation 等额外任务
* BLIP-2 更进一步，用 Q-Former 架构桥接冻结的视觉模型和语言模型
* 既能做检索，也能做图文生成

**SigLIP（Google）**

```
CLIP：softmax-based InfoNCE Loss（需要全局归一化）
SigLIP：sigmoid loss（每对独立计算，无需全局归一化）
```

* 把 N 类分类问题变成 N² 个二分类问题
* 好处：可以更灵活地扩展 batch size，训练更稳定
* 在同等计算量下效果优于 CLIP

**EVA-CLIP（BAAI / 智源）**

* 用更强的预训练 ViT（EVA 系列）作为图像编码器
* 通过更好的初始化和训练策略，在更少数据上达到更好效果
* EVA-02-CLIP 在多个基准上刷新了 CLIP 类模型的记录

**其他重要工作**

| 模型                            | 核心改进                                 |
| ----------------------------- | ------------------------------------ |
| OpenCLIP                      | CLIP 的开源复现，支持更多架构和数据集                |
| MetaCLIP                      | 用 CLIP 的元数据策略重新构建高质量训练集              |
| DFN (Data Filtering Networks) | 用模型自身来过滤训练数据，提升数据质量                  |
| LiT (Locked-image Tuning)     | 冻结图像编码器，只训练文本端（Chinese-CLIP 也用了这个思路） |
| ALIGN (Google)                | 类似 CLIP 但用了更大规模的噪声数据（18 亿对）          |

#### 7.3 CLIP 在生成模型中的角色

虽然 CLIP 本身不生成图像，但它在生成模型生态中扮演了关键角色：

* **DALL·E 2**：用 CLIP 的图像向量作为生成的条件信号
* **Stable Diffusion**：用 CLIP 的文本编码器来理解用户 prompt
* **CLIP Score**：评估生成图像与文本的匹配度的标准指标
* **CLIP Guidance**：在生成过程中用 CLIP 引导图像向目标语义靠近

可以说，**CLIP 是连接语言和视觉世界的桥梁，而这座桥上跑着各种各样的应用**。

***

### 八、总结

让我们回顾整条线索：

```
第 1 篇：向量是什么？为什么万物皆可向量化？
第 2 篇：余弦相似度——如何衡量向量间的"语义距离"
第 3 篇：文本嵌入——Sentence-BERT 如何把文本变成向量
第 4 篇：向量数据库与语义搜索——大规模检索的实现
第 5 篇：CLIP——打破模态壁垒，让文本和图像住进同一个空间 ← 你在这里
```

CLIP 的核心贡献是证明了一件事：

> **通过对比学习，在足够大的数据上，两个不同模态的编码器可以学会把语义相同的东西映射到向量空间中的相近位置。**

这不仅解决了图文匹配的问题，更重要的是，它提供了一种**通用的跨模态对齐范式**。同样的思路可以扩展到音频、视频、3D 模型等任何模态——只要你有配对数据和对比损失。

从向量、到相似度、到文本嵌入、到向量检索、到跨模态对齐——**这条路的终点不是 CLIP，而是一个万物互联的统一语义空间**。而 CLIP，是这条路上最关键的一步。

***

_本文是"向量与相似度"系列第 5 篇。如果你从第 1 篇看到这里，恭喜——你已经理解了从基础向量到跨模态 AI 的完整知识链。_

