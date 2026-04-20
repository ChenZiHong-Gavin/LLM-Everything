# 相似度度量

> 在推荐系统、搜索引擎、RAG、图像检索等场景中，"相似度计算"是最底层、最核心的操作之一。衡量两个向量之间"有多像"，这个问题看似简单，但选错度量方法，整个系统的效果可能天差地别。

### 1 为什么"相似度"需要量化？

人类判断相似性靠直觉——你一眼就能看出"猫"和"狗"比"猫"和"卡车"更像。但机器没有直觉，它只能依靠数字来判断。

在这里，文本、图像、音频等各种数据，都会先被编码成**向量**（embedding）。一段话是一个 768 维向量，一张图是一个 512 维向量。两个东西"像不像"，就变成了一个纯粹的数学问题：<mark style="color:$primary;">**两个向量之间的"距离"或"相似度"是多少？**</mark>

#### 什么是好的相似度度量？

一个好的度量函数应该满足几个直觉：

* **自身最相似**：一个向量和自己的相似度应该是最大值
* **对称性**：A 和 B 的相似度 = B 和 A 的相似度
* **区分度**：真正相似的向量得分高，不相似的得分低
* **计算高效**：在亿级数据中，这个函数要被调用几亿次，不能太慢

### 2 余弦相似度

以下是完整的全文，所有数学公式均已采用标准 LaTeX 表示：

***

### 2.1 几何直觉

余弦相似度的核心思想极其直观：**只看方向，不看长度**。

想象二维平面上的两个箭头。如果它们指向几乎相同的方向，无论长短，我们都认为它们"相似"。余弦相似度衡量的就是这两个箭头之间夹角的余弦值。

* 夹角 $$= 0^\circ$$（完全同向） $$\rightarrow \cos(0^\circ) = 1$$（最相似）
* 夹角 $$= 90^\circ$$（正交，毫无关系） $$\rightarrow \cos(90^\circ) = 0$$
* 夹角 $$= 180^\circ$$（完全反向） $$\rightarrow \cos(180^\circ) = -1$$（最不相似）

### 2.2 数学公式

给定两个向量 $$\mathbf{a} = (a_1, a_2, \ldots, a_n)$$ 和 $$\mathbf{b} = (b_1, b_2, \ldots, b_n)$$，余弦相似度定义为：

$$
\cos\_sim(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \cdot \sqrt{\sum_{i=1}^{n} b_i^2}}
$$

其中：

*   $$\mathbf{a} \cdot \mathbf{b}$$ 是向量点积：

    $$
    \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + \cdots + a_nb_n
    $$
*   $$|\mathbf{a}|$$ 是向量 $$\mathbf{a}$$ 的 L2 范数（模长）：

    $$
    \|\mathbf{a}\| = \sqrt{\sum_{i=1}^{n} a_i^2} = \sqrt{a_1^2 + a_2^2 + \cdots + a_n^2}
    $$

**取值范围**：$$[-1, 1]$$

实际上，在大多数 embedding 模型的输出中（如 Sentence-BERT、text2vec），向量的各维度都是正数居多，所以实际范围通常在 $$[0, 1]$$ 附近。

### 2.3 为什么它对向量长度不敏感

这是余弦相似度最重要的特性。看公式，分母是两个向量的模长之积，这本质上是在做归一化。

举个例子：

$$
\mathbf{a} = (1, 2, 3), \quad \mathbf{b} = (2, 4, 6) \quad (\mathbf{b} = 2\mathbf{a}，方向完全相同，长度翻倍)
$$

$$
\begin{aligned}
\cos\_sim(\mathbf{a}, \mathbf{b}) &= \frac{1 \times 2 + 2 \times 4 + 3 \times 6}{\sqrt{14} \times \sqrt{56}} \\
&= \frac{28}{28} \\
&= 1.0 \quad \leftarrow \text{完全相似}
\end{aligned}
$$

向量 $$\mathbf{b}$$ 只是 $$\mathbf{a}$$ 的两倍长，但余弦相似度认为它们完全一样。这在文本场景中特别有用：一篇长文章和一篇短文章，如果讨论的是同一个话题，它们的 embedding 方向应该一致，只是"强度"（模长）不同。余弦相似度不会因为文档长度不同而给出不公平的分数。

如果向量已经做过 L2 归一化（即 $$|\mathbf{a}| = |\mathbf{b}| = 1$$），那么：

$$
\cos\_sim(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b}
$$

余弦相似度退化为点积。这就是为什么很多系统会先归一化，再直接算点积——更快，结果一样。

### 2.4 Python 实现

```python
import numpy as np

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# 示例
a = np.array([1, 2, 3])
b = np.array([2, 4, 6])   # 与 a 同方向
c = np.array([3, -1, 2])  # 与 a 方向不同

print(cosine_similarity(a, b))  # 1.0（完全相似）
print(cosine_similarity(a, c))  # 0.41（有一定相似性）
```

### 3 点积 <a href="#id-3-e7-82-b9-e7-a7-afdot-product-inner-product" id="id-3-e7-82-b9-e7-a7-afdot-product-inner-product"></a>

#### 3.1 从余弦相似度到点积

点积是最"朴素"的向量运算：

$$
\text{dot}(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + \cdots + a_nb_n
$$

回顾余弦相似度的公式，我们知道：

$$
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \cdot \|\mathbf{b}\| \cdot \cos(\theta)
$$

也就是说：

$$
\text{点积} = \text{余弦相似度} \times \text{两个向量的模长之积}
$$

这意味着点积同时编码了两个信息：

1. **方向的一致性**（$$\cos(\theta)$$ 部分）
2. **向量的"强度"**（模长部分）

<mark style="color:$primary;">当模长本身携带有意义的信息时，点积比余弦更合适。</mark>

一个经典的例子是推荐系统。假设：

* 用户向量的模长代表用户的活跃度（活跃用户向量更长）
* 物品向量的模长代表物品的热门程度（热门物品向量更长）

如果用余弦相似度，一个冷门物品和一个热门物品只要方向一致，得分就一样。但实际上，你可能希望热门物品天然有更高的分数（因为用户更可能点击）。这时候点积就能自然地做到这一点。

再举一个更具体的例子：

$$
\begin{aligned}
\mathbf{user} &= (0.5, 0.3) \quad \text{\# 不太活跃的用户，偏好方向：}(5, 3) \\
\mathbf{item}\_A &= (4.0, 2.4) \quad \text{\# 热门物品，方向与用户一致} \\
\mathbf{item}\_B &= (0.5, 0.3) \quad \text{\# 冷门物品，方向与用户一致}
\end{aligned}
$$

$$
\cos\_sim(\mathbf{user}, \mathbf{item}\_A) = 1.0 \quad \leftarrow \text{余弦看来完全一样}
$$

$$
\cos\_sim(\mathbf{user}, \mathbf{item}\_B) = 1.0
$$

$$
\mathbf{user} \cdot \mathbf{item}\_A = 2.0 + 0.72 = 2.72 \quad \leftarrow \text{点积能区分热门和冷门}
$$

$$
\mathbf{user} \cdot \mathbf{item}\_B = 0.25 + 0.09 = 0.34
$$

#### 3.3 内积搜索（MIPS）问题

在大规模检索中，如果你用点积作为度量，需要找到与 query 向量点积最大的 top-k 向量，这就是 Maximum Inner Product Search (MIPS) 问题。

MIPS 比最近邻搜索（NNS）更棘手，因为：

* 欧氏距离最近邻搜索有成熟的数据结构（KD-Tree、Ball Tree 等）
* 但点积不满足三角不等式，传统方法不能直接用

解决方案通常是把 MIPS 转化为 NNS。一个经典技巧：给向量增加一个额外维度，使得点积最大等价于欧氏距离最小。具体来说，对于向量 $$\mathbf{x}$$，构造：

$$
\mathbf{x}' = \left(\mathbf{x}, \sqrt{M^2 - \|\mathbf{x}\|^2}\right) \quad \text{其中 } M = \max\_{\mathbf{x}} \|\mathbf{x}\| \text{ 对所有 } \mathbf{x}
$$

这样就可以用 Faiss 等工具的 L2 索引来做 MIPS 了。

实际工程中，Faiss 的 `IndexFlatIP` 直接支持内积搜索，底层已经做了类似的优化。

#### 3.4 点积的取值范围

没有固定范围，取决于向量的模长和维度。这是点积的一个"缺点"——不同 query 的点积分数不可直接比较。如果需要可比较的分数，要么归一化后用余弦，要么对分数做后处理（如 softmax）。

#### 3.5 Python 实现

```python
import numpy as np

def dot_product(a, b):
    """计算两个向量的点积"""
    return np.dot(a, b)

# 示例：点积能区分模长
user = np.array([0.5, 0.3])
item_hot = np.array([4.0, 2.4])   # 热门物品
item_cold = np.array([0.5, 0.3])  # 冷门物品

print(dot_product(user, item_hot))   # 2.72
print(dot_product(user, item_cold))  # 0.34
```

### 4 欧氏距离

#### 4.1 最符合直觉的"距离"

欧氏距离就是日常生活中"两点之间的直线距离"，推广到 $$n$$ 维空间：

$$
d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2} = \sqrt{(a_1-b_1)^2 + (a_2-b_2)^2 + \cdots + (a_n-b_n)^2}
$$

也叫 L2 距离。注意，欧氏距离是距离（越小越相似），而不是相似度（越大越相似）。

实际计算中，经常省略开方，直接用平方欧氏距离，因为：

* 开方不改变排序关系
* 省掉开方运算更快

$$
d^2(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n} (a_i - b_i)^2 = \|\mathbf{a} - \mathbf{b}\|^2
$$

#### 4.2 与余弦相似度的数学等价

这是一个非常重要的结论：**当向量经过 L2 归一化后，欧氏距离和余弦相似度是等价的。**

证明如下。设 $$|\mathbf{a}| = |\mathbf{b}| = 1$$，则：

$$
\begin{aligned}
\|\mathbf{a} - \mathbf{b}\|^2 &= (\mathbf{a} - \mathbf{b}) \cdot (\mathbf{a} - \mathbf{b}) \\
&= \mathbf{a}\cdot\mathbf{a} - 2\mathbf{a}\cdot\mathbf{b} + \mathbf{b}\cdot\mathbf{b} \\
&= \|\mathbf{a}\|^2 - 2(\mathbf{a}\cdot\mathbf{b}) + \|\mathbf{b}\|^2 \\
&= 1 - 2(\mathbf{a}\cdot\mathbf{b}) + 1 \\
&= 2 - 2\cos\_sim(\mathbf{a}, \mathbf{b}) \\
&= 2(1 - \cos\_sim(\mathbf{a}, \mathbf{b}))
\end{aligned}
$$

所以：

$$
\|\mathbf{a} - \mathbf{b}\|^2 = 2(1 - \cos\_sim(\mathbf{a}, \mathbf{b}))
$$

这意味着：**对归一化向量来说，用余弦相似度排序和用欧氏距离排序，结果完全一样。**&#x20;

这就是为什么 Faiss 中 `IndexFlatL2`（L2 距离）和 `IndexFlatIP`（内积）在归一化向量上给出相同的检索结果。

#### 4.3 维度灾难：高维空间中欧氏距离的失效

这是一个反直觉但极其重要的现象。

在高维空间中，随机采样的点之间的欧氏距离趋于集中。也就是说，最近的点和最远的点之间的距离差异变得非常小。

数学上可以证明，对于 $$n$$ 维空间中的随机向量：

$$
\lim_{n \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0
$$

#### 4.4 Python 实现

```python
import numpy as np

def euclidean_distance(a, b):
    """计算两个向量的欧氏距离"""
    return np.linalg.norm(a - b)

def squared_euclidean_distance(a, b):
    """计算平方欧氏距离（省去开方，更快）"""
    diff = a - b
    return np.dot(diff, diff)

# 示例
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.1, 2.1, 3.1])  # 很近
c = np.array([5.0, 0.0, -1.0]) # 很远

print(euclidean_distance(a, b))  # 0.173（很近）
print(euclidean_distance(a, c))  # 5.099（很远）

# 验证归一化后的等价关系
a_norm = a / np.linalg.norm(a)
b_norm = b / np.linalg.norm(b)
cos_sim = np.dot(a_norm, b_norm)
l2_sq = np.sum((a_norm - b_norm) ** 2)
print(f"2(1 - cos_sim) = {2 * (1 - cos_sim):.6f}")
print(f"L2² = {l2_sq:.6f}")
```

***

### 5 汉明距离

汉明距离衡量的是两个等长序列中，对应位置上不同元素的个数。

对于二进制向量（0/1 向量），就是不同 bit 的个数：

```
a = 1 0 1 1 0 1 0 0
b = 1 0 0 1 1 1 0 0
    ✓ ✓ ✗ ✓ ✗ ✓ ✓ ✓

Hamming(a, b) = 2（第 3 位和第 5 位不同）
```

汉明距离的计算可以用两个硬件原语完成：

$$
\text{Hamming}(\mathbf{a}, \mathbf{b}) = \text{popcount}(\mathbf{a} \oplus \mathbf{b})
$$

* **XOR（异或）**：相同位得 0，不同位得 1。一条 CPU 指令
* **popcount（位计数）**：统计结果中 1 的个数。现代 CPU 有专用指令（POPCNT）

对于一个 64-bit 的二进制向量，整个汉明距离计算只需要 2 条 CPU 指令。相比之下，余弦相似度在 64 维向量上需要约 200 次浮点运算。

更重要的是，二进制向量的存储极其紧凑：

* 一个 768 维的 float32 embedding = 3072 字节
* 同样信息量，用 768-bit 的二进制哈希 = 96 字节（压缩 32 倍）

#### 5.1 与 SimHash / 局部敏感哈希（LSH）的关系

但你可能会问：文本 embedding 是浮点数向量，怎么变成二进制向量？

这就是**局部敏感哈希（Locality-Sensitive Hashing, LSH）** 的工作。LSH 的核心思想是：

> 设计一个哈希函数，使得相似的输入以高概率映射到相似的哈希值（汉明距离小），不相似的输入映射到不相似的哈希值。

**SimHash** 就是一种经典的 LSH 方法，常用于文本去重。它的流程是：

1. 对文本进行分词，提取特征（如 n-gram）
2. 每个特征通过普通哈希函数映射到一个固定长度的二进制串
3. 根据特征的权重（如 TF-IDF），对每一位进行加权投票
4. 最终每一位：正数票多 → 1，负数票多 → 0

```
文本 A → SimHash: 10110100...（64 bit）
文本 B → SimHash: 10100100...（64 bit）
```

两个文本的相似度判断就变成了：汉明距离 $$\leq$$ 阈值（通常取 3）就认为近似重复。

Google 早期用 SimHash 做网页去重，在几十亿网页中快速发现重复内容。64-bit SimHash + 汉明距离 $$\leq 3$$ 的方案，可以在毫秒级别完成判断。

#### 5.2 汉明距离的局限

* **信息损失大**：从 float32 到 1-bit 是极端的量化，精度会有明显下降
* **只适合粗筛**：通常作为第一步快速过滤，后续再用余弦等精确度量进行重排
* **阈值敏感**：汉明距离是离散的整数值，对阈值的选择很敏感

#### 5.3 Python 实现

```python
def hamming_distance_bin(a: int, b: int) -> int:
    """计算两个整数的二进制汉明距离"""
    return bin(a ^ b).count('1')

# 示例
a = 0b10110100
b = 0b10100100
print(hamming_distance_bin(a, b))  # 2

# 用 numpy 处理二进制向量
import numpy as np

def hamming_distance_vec(a, b):
    """计算两个 0/1 向量的汉明距离"""
    return np.sum(a != b)

a = np.array([1, 0, 1, 1, 0, 1, 0, 0])
b = np.array([1, 0, 0, 1, 1, 1, 0, 0])
print(hamming_distance_vec(a, b))  # 2
```

***

### 6. 横向对比总结

#### 一张表说清楚

| 度量    | 类型         | 取值范围                   | 是否关注模长   | 计算复杂度      | 典型场景                 |
| ----- | ---------- | ---------------------- | -------- | ---------- | -------------------- |
| 余弦相似度 | 相似度（越大越相似） | $$[-1, 1]$$            | ❌ 不关注    | $$O(n)$$   | 文本 embedding 比较、语义搜索 |
| 点积    | 相似度（越大越相似） | $$(-\infty, +\infty)$$ | ✅ 关注     | $$O(n)$$   | 推荐系统、模长有意义的场景        |
| 欧氏距离  | 距离（越小越相似）  | $$[0, +\infty)$$       | ✅ 关注     | $$O(n)$$   | 低维几何距离、聚类、归一化后检索     |
| 汉明距离  | 距离（越小越相似）  | $$[0, n]$$（$$n$$ 为维度）  | N/A（二进制） | $$O(1)^*$$ | 大规模去重、粗筛、LSH         |

\*$$O(1)$$ 指的是利用硬件 XOR + POPCNT 指令，对固定长度的二进制串，计算时间与"维度"无关。

### 参考

1. [https://github.com/shibing624/similarities](https://github.com/shibing624/similarities)
