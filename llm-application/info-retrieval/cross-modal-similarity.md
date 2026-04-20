# 跨模态相似度

#### 第 5 篇：CLIP 与跨模态相似度

**标题建议：** _CLIP 原理详解：如何让文本和图像住进同一个向量空间_

**内容：**

* **跨模态相似度的挑战**：文本和图像是完全不同的信号，怎么比较？
* **CLIP 的核心思想**
  * 对比学习框架：image encoder + text encoder，对齐到同一空间
  * 训练数据：4 亿图文对，从互联网自然爬取
  * 对比损失（InfoNCE Loss）：正确的图文配对拉近，错误的推远
  * 矩阵化的高效实现
* **图像编码器**：ViT（Vision Transformer）的结构与工作方式
* **文本编码器**：Transformer + BPE tokenizer
* **CLIP 能做什么**
  * 零样本图片分类（不需要标注数据！）
  * 文搜图 / 图搜文
  * 图文匹配打分
* **Chinese-CLIP 等变体**：如何适配中文场景
* **CLIP 的局限与后续发展**（BLIP、SigLIP、EVA-CLIP 等）
