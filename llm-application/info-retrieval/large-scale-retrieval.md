# 大规模向量检索

#### 第 6 篇：大规模向量检索——ANN 算法原理

**标题建议：** _亿级向量怎么搜？Faiss、Annoy、HNSW 三大 ANN 算法原理详解_

**内容：**

* **问题定义**：当向量有几亿条时，暴力遍历不可行，怎么办？
* **近似最近邻（ANN）的核心思想**：用少量精度换巨大的速度提升
* **Faiss（Facebook AI Similarity Search）**
  * IndexFlatL2 / IndexFlatIP：暴力搜索基线
  * IVF（倒排文件索引）：先聚类后搜索
  * PQ（Product Quantization）：向量压缩，内存降几十倍
  * IVF + PQ 组合：大规模检索的主力方案
  * nprobe 参数的含义：搜多少个聚类中心
* **Annoy（Approximate Nearest Neighbors Oh Yeah）**
  * 随机投影树的构造过程
  * 多棵树的集成：用空间换精度
  * 优势：只读索引、mmap 友好、轻量
* **HNSW（Hierarchical Navigable Small World）**
  * 小世界网络的直觉："六度分隔"
  * 多层图的构建过程：高层快速定位 → 低层精确搜索
  * 为什么 HNSW 召回率最高（接近暴力搜索）
  * 代价：内存占用大、构建索引慢
* **三者对比总结**：速度 / 内存 / 召回率 / 适用场景一张表
