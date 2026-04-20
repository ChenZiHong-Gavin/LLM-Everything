# 词向量

#### 第 3 篇：词向量与语义相似度

**标题建议：** _从 Word2Vec 到 Sentence Embedding：让机器真正"理解"语义相似_

**内容：**

* **为什么需要语义级别的表示**：TF-IDF 的天花板（"快乐"和"开心"余弦为 0）
* **Word2Vec**
  * CBOW 与 Skip-gram 两种架构
  * 负采样的巧妙设计
  * 词向量的涌现能力："king - man + woman ≈ queen"
  * 从词向量到句子向量的简单策略（平均池化、TF-IDF 加权平均）及其局限
* **Sentence Embedding：SentenceBERT**
  * BERT 为什么不能直接做句子相似度（CLS 向量的问题）
  * Siamese Network 结构：两个句子分别编码 → 比较向量
  * 训练方式：NLI 数据集 + Softmax 或 Cosine Loss
* **CoSENT（Cosine Sentence）**
  * 对 SentenceBERT 损失函数的改进
  * 排序思想：正例对的余弦分数应该高于负例对
  * Circle Loss 的连接
  * 为什么 CoSENT 在中文语义匹配上效果更好
* **预训练模型的选择逻辑**：中文用什么、英文用什么、多语言怎么办
