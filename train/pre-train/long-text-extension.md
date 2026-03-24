# 长文本拓展

### 1 渐进式扩展

直接从短上下文（如 4K）扩展到超长上下文（128K 或更高）会导致训练不稳定和灾难性遗忘。现代 LLM 采用渐进式课程学习（Progressive Curriculum）策略。

典型的三阶段扩展路径如下 ：

| 阶段          | 上下文长度 | 训练目标   | 关键技术               |
| ----------- | ----- | ------ | ------------------ |
| **Stage 1** | 4K/8K | 基础预训练  | 标准预训练，建立基础能力       |
| **Stage 2** | 32K   | 文件级理解  | 长文本初始化，数据混合调整      |
| **Stage 3** | 128K+ | 长程依赖捕获 | 长文档、多文件推理、Agent 轨迹 |

**具体流程如下**（参考 InCoder-32B 和 TeleChat2）：

* **Stage 2.1 (8K→32K)**：直接训练至 32K 长度，无需位置插值，使用 Cosine 学习率衰减，数据混合强调代码、推理 QA 和 Agent 轨迹。
* **Stage 2.2 (32K→128K)**：采用\*\*渐进式热身（Graduated Warm-up）\*\*策略——长序列样本（>32K）比例从 10% 线性增加至 50%，防止训练初期的不稳定。
* **数据回放（Data Replay）**：在每个阶段保留 5-20% 的上阶段数据，缓解分布偏移，稳定specialization。

长上下文扩展需重新调整学习率：

* **重置与衰减**：每进入新阶段，学习率重置为该阶段初始值（如从 4e-4 降至 4e-5），再执行 Cosine Annealing。如果不重置学习率，继续使用上一阶段结束时的极小学习率，模型几乎没有"学习能力"去适应新的位置编码（RoPE）分布。重置相当于给模型一次"重新热身"的机会，让它用相对较高的学习率快速适应新长度。
* **ABF（Attention Base Frequency）技术**：在扩展阶段同步增大 RoPE 基础频率（base frequency），如从 1×10⁶（32K 阶段）提升至 4×10⁷（256K 阶段），以缓解长程衰减。

***

### 2 位置编码适配

位置编码是决定模型长度泛化能力的核心。当前主流方案分为 **外推（Extrapolation）与插值（Interpolation）**&#x4E24;大类。

#### 1. ALiBi

**ALiBi（Attention with Linear Biases）** 通过在注意力分数中添加**线性衰减的负偏置**实现位置编码： $$\text{Attention}(Q, K) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} - m \cdot |i-j|\right)$$ 其中 $$m$$ 为预定义的斜率（head-specific slope），$$|i-j|$$ 为 token 间的绝对距离。

**特点**：

* **训练免费的外推**：在短文本上训练，可直接推断更长序列（Train Short, Test Long）。
* **局限性**：长程精度有限，超过 4 倍训练长度后性能急剧退化。

#### 2. RoPE 及其扩展家族

**RoPE（Rotary Position Embedding）**&#x901A;过旋转矩阵注入相对位置信息： $$f(x_m, m) = \Theta_m x_m, \quad \Theta_m = \text{diag}(\{e^{i m \theta_j}\}_{j=1}^{d/2})$$ 其中 $$\theta_j = b^{-2j/d}$$，$$b$$ 为基础频率（base frequency，通常 10000）。

**(1) 位置插值**

将位置索引 $$m$$ 按比例 $$s = L'/L$$ 压缩到预训练范围内： $$m' = m \cdot \frac{L}{L'}$$ **缺陷**：对所有频率均匀压缩，导致高频信息（局部细节）损失。

**(2) NTK-aware 插值**

基于 Neural Tangent Kernel 理论，**高频特征难以学习**，应减少高频压缩。通过非均匀缩放因子： $$s_j = \left(\frac{L'}{L}\right)^{2j/(d-2)}$$ 高频（$$j$$ 大）缩放少，低频（$$j$$ 小）缩放多，保留局部分辨率。

**(3) YaRN（Yet another RoPE extensioN）**

将 RoPE 维度分为三组处理：

* **高频分量**：不缩放（$$s_j = 1$$），保留局部敏感性
* **低频分量**：PI 式完全缩放（$$s_j = L'/L$$），缓解 OOD 问题
* **中间频率**：线性过渡插值

**动态缩放（Dynamic Scaling）**：在自回归生成中，根据当前序列长度动态调整 $$s = \max(1, l'/L)$$，实现平滑退化而非突然崩溃。

**(4) LongRoPE 与 LongRoPE2**

采用**困惑度引导的搜索策略**，为每个频率维度寻找最优缩放因子，而非预设公式。LongRoPE2 进一步实现**近无损扩展**，在 128K 长度上性能超越传统方法。

#### 3. 动态位置编码

**Dynamic NTK** 在推理时根据当前序列长度实时调整 base frequency： $$b' = b \cdot \left(\frac{l'}{L}\right)^{d/(d-2)}$$ 实现训练后无需微调的长度泛化。

位置编码的详细推导可见之前的文章 [positional-encoding.md](../../transformer/positional-encoding.md "mention")

### 3 稀疏注意力

长上下文训练的**计算复杂度瓶颈**在于 Self-Attention 的 $$O(L^2)$$ 复杂度。稀疏注意力通过**选择性计算**降低开销。

#### 3.1 Sliding Window Attention（局部滑动窗口）

仅计算每个 token 与其左右窗口大小为 $$w$$ 的邻域内的注意力，复杂度降至 $$O(L \cdot w)$$。

**预训练集成策略**：

* **分阶段扩展窗口**：初期使用较小窗口（如 4K），逐步扩大至全局长度。
* **全局-局部混合**：保留少量全局 attention heads（如 1-2 个），其余使用滑动窗口，平衡局部细节与全局结构。

#### 3.2 FlashAttention 系列：IO 感知的精确注意力

FlashAttention 通过**分块计算（Tiling）和重计算（Recomputation）策略，减少 HBM 访问次数，实现精确的稀疏注意力**（非近似）：

**FlashAttention-3 优化**：

* **Warp-specialized 调度**：针对 Hopper 架构优化异步拷贝与 Tensor Core 重叠。
* **低精度支持**：支持 FP8 训练，进一步加速长序列处理。

**预训练集成**：

* 在 128K 预训练中，FlashAttention 将内存占用从 $$O(L^2)$$ 降至 $$O(L)$$，使得单卡可承载更长序列。
* 与序列并行（Sequence Parallelism）结合，将长序列分片到多卡计算。



### 参考

1. [https://arxiv.org/pdf/2501.15383](https://arxiv.org/pdf/2501.15383)
