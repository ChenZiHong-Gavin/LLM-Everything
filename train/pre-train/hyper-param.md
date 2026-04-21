# 超参配置

### 0 快速决策表

<table data-header-hidden><thead><tr><th>超参数</th><th width="243.666748046875">大规模预训练 （多机多卡）</th><th>中等规模/单卡预训练（单机多卡/单机单卡）</th><th>LoRA微调</th></tr></thead><tbody><tr><td><strong>学习率调度</strong></td><td><strong>WSD</strong><br>• Warmup: 1-2%<br>• Stable: 88-89%<br>• Decay: 10% 步数 → 0.1×Peak<br>Peak: 3e-4 ~ 1e-4（随模型参数增大降低）</td><td><strong>WSD</strong><br>比例同上，但可延长 Warmup 至 5%<br>Peak: 1e-4 ~ 5e-5</td><td><strong>WSD</strong><br>• Warmup: 10% 步数<br>• Stable: 80%<br>• Decay: 10%<br><strong>Peak: 2e-4</strong></td></tr><tr><td><strong>Batch Size</strong></td><td><strong>最小可行</strong><br>Global Batch: 4M-8M tokens<br>（约 16-32 样本 × 序列长度 × GPU数）<br>不使用梯度累积</td><td><strong>最小可行</strong><br>大约 1-16<br>不使用梯度累积</td><td><strong>16-32</strong><br>可通过梯度累积实现</td></tr><tr><td><strong>优化器</strong></td><td><strong>AdamW /</strong> <strong>Muon</strong>（实验性，省50%显存）</td><td><strong>AdamW</strong></td><td><strong>AdamW</strong></td></tr><tr><td><strong>AdamW 参数</strong></td><td>β1=0.9<br><strong>β2=0.95</strong>（预训练关键）<br>ε=1e-5（LLM稳定值）<br>Weight Decay: <strong>0.01</strong> (GPT) / <strong>0.1</strong> (LLaMA)</td><td>β1=0.9<br>β2=0.99~0.999（按 half-life 缩放）<br>ε=1e-5<br>Weight Decay: 0.01-0.1</td><td>β1=0.9<br>β2=0.999<br>ε=1e-6<br>Weight Decay: <strong>0.01</strong>（通常更低）</td></tr><tr><td><strong>Epoch/步数</strong></td><td><strong>1 epoch</strong><br>（Single-pass，数据量 > 1T tokens）</td><td><strong>1-2 epochs</strong></td><td><strong>3-5 epochs</strong><br>+ Early Stopping</td></tr><tr><td><strong>梯度裁剪</strong></td><td>1.0</td><td>1.0</td><td><strong>1.0-5.0</strong><br>配合 2e-4 学习率</td></tr><tr><td>其它</td><td>• 禁用 Dropout 提升下游任务性能<br>• 优先用最大 batch size 而非梯度累积</td><td><ul><li>学习率需按 √B 缩放</li></ul></td><td>• rank 通常 8-64<br>• alpha = 2×rank<br>• target_modules: q_proj, v_proj</td></tr></tbody></table>



### 1 学习率调度

> "学习率越大，微调速度越快，但你必须在速度与跳过最优解或导致训练不稳定的风险之间取得平衡。"

当前 LLM 预训练的首选 **Warmup-Stable-Decay (WSD)**

分为三个阶段：

* **Warmup**：线性预热（通常占总步数的1-2%），从0升至峰值学习率（如 $$6\times10^{-4}$$）
* **Stable**：长时间保持恒定学习率
* **Decay**：在选定checkpoint后，使用快速衰减（如线性或余弦衰减至峰值的10%）

**核心优势**：

1. **连续训练友好**：无需预先确定总步数，可从stable阶段的任意checkpoint分支进行decay实验
2. **loss动态独特**：stable阶段loss下降缓慢（高于Cosine schedule），但decay阶段会出现**sharp drop**（DeepSeek、OLMo、MiniMax 等训练日志中均有记录），最终性能往往优于Cosine

<figure><img src="../../.gitbook/assets/image (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

**Cosine Decay with Restarts vs. Linear Decay**

* **Cosine Annealing**：需要预先确定总训练步数，学习率平滑下降。虽然稳定，但缺乏灵活性，无法轻松扩展训练
* **Linear Decay**：简单直接，但可能导致后期学习率下降过快，不如Cosine平滑

**如何调整学习率**

* 密切关注训练损失。如果出现突然的峰值或"NaN"（非数字）值，这是危险信号——立即降低学习率。一个经验法则是将预热步数设置为总训练步数的10%。

***

### 2 如何设置 Batch Size

&#x20;Batch Size 越大，学习率肯定越大

苏剑林在《当 Batch Size 增大时，学习率该如何随之变化？》 中系统梳理了几种主流视角：

#### 一、方差视角（平方根缩放）

最早的理论认为，batch size 扩大到 $$n$$ 倍时，学习率应扩大到 $$\sqrt{n}$$ 倍。推导基于**保持 SGD 增量的方差不变**：

* Batch size 增大 → 梯度协方差矩阵缩小为 $$1/B$$ → 为保持噪声强度，学习率需满足 $$\eta^2/B = \text{常数}$$ → $$\eta \propto \sqrt{B}$$

#### 二、直面损失视角（单调有界缩放）

OpenAI《An Empirical Model of Large-Batch Training》提出更本质的二阶近似分析：

* 最优学习率$$\eta^* \approx \frac{\eta_{max}}{1 + \mathcal{B}_{\text{noise}}/B}$$
* **关键结论**：学习率随 batch size 增加而**单调递增但有上界** $$\eta_{max}$$
* 当 $$B \ll \mathcal{B}_{\text{noise}}$$（小 batch）：近似**线性缩放** $$\eta \propto B$$
* 当 $$B \gg \mathcal{B}_{\text{noise}}$$（大 batch）：学习率趋于饱和，**不应继续增大**

其中 $$\mathcal{B}_{\text{noise}} = \frac{\text{tr}(\Sigma)}{g^\top g}$$ 是**信噪比倒数**，反映梯度噪声强度与信号强度的比值。

前沿 LLM（GPT、Llama 等）通常在**大规模集群**上使用极大的 batch size（数百万 token），因为：

* 梯度估计更稳定，可使用更大学习率加速收敛
* 最大化 GPU 吞吐量（throughput），提高训练效率
* 配合 Adam/AdamW 等复杂优化器

所以，通常会使用**硬件能支持的最大 batch size。**

#### 梯度累积有什么陷阱？

当显存不足时，常见做法是通过梯度累积模拟大 batch。但需要注意的是，梯度累积不会提供吞吐量收益，只是通过"以时间换空间"增加训练 wall-clock 时间，且可能使超参调优复杂化。

***

### 3 如何选择优化器

#### **AdamW**

解耦权重衰减与L2正则化，避免Adam中自适应梯度导致的权重衰减效果衰减问题，与 Adam 的 L2 正则化的区别：

* **Adam 的 L2**：梯度更新为 $$\theta \leftarrow \theta - \eta(\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon} + \lambda\theta)$$，其中自适应项 $$\frac{1}{\sqrt{\hat{v}}}$$ 会**削弱**权重衰减效果（梯度大的参数权重衰减小）
* **AdamW**：直接将权重衰减应用于参数更新步骤 $$\theta \leftarrow \theta - \eta\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon} - \eta\lambda\theta$$，**不受梯度尺度影响**，所有参数以相同速率 $$\lambda$$ 衰减

**参数设置**：

* **β1（一阶矩衰减）**：0.9
* **β2（二阶矩衰减）**：0.95（预训练）或 0.999（微调）。**0.95在预训练中表现更好，**&#x8F83;小的二阶矩衰减率意味着更快的"遗忘"速度，使优化器对数据分布的非平稳性（预训练数据在不同阶段有不同特征）更敏感，适应更快。0.999在SFT任务上表现更好，微调时数据分布稳定，需要长期累积二阶统计信息以获得更平滑的梯度估计。
* **ε（数值稳定性）**：$$1\times10^{-5}$$ 到 $$1\times10^{-8}$$。较大的ε（如1e-5）在LLM训练中更稳定，防止除以极小值，在梯度稀疏的大模型中避免数值爆炸
* **权重衰减**：0.01（GPT架构）到 0.1（LLaMA/T5架构）

#### **Muon**

**核心机制：梯度正交归一化**

由 Keller Jordan 提出（2024），针对**矩阵参数**（如线性层权重 $$W \in \mathbb{R}^{m \times n}$$）设计：

1. 对梯度矩阵 $$G$$ 进行 SVD 分解：$$G = U \Sigma V^T$$
2. **正交化梯度**：$$G_{orth} = U V^T$$（丢弃奇异值 $$\Sigma$$，仅保留旋转分量）
3. 更新规则：$$W \leftarrow W - \eta \cdot G_{orth} + \text{AdamW-style 权重衰减}$$

**为什么有效**：

* 正交矩阵的谱范数为 1，防止更新步长过大导致的训练不稳定
* 强制梯度在正交流形上更新，保持参数矩阵的良好条件数，改善梯度传播
* **内存效率**：不需要存储二阶矩（无 $$v$$ 统计量），仅需存储梯度本身，比 AdamW 节省约 50% 优化器状态内存
* 适合**大规模预训练**作为 AdamW 的替代，在 Cerebras 和部分开源 LLM 训练中表现出色。

#### **Adafactor**

**优势**：

* **内存效率**：通过因子分解二阶动量矩阵（$$v \approx R \times C$$），将显存占用从Adam的**12 bytes/参数**降至**4-8 bytes/参数**
* 适合千亿参数模型或单卡训练（如24GB显存训练13B模型）

**代价**：

* 需要更精细的超参调优（学习率通常需比Adam大2-10倍）
* 收敛速度略慢，下游任务性能可能略低于AdamW

| 优化器           | 内存占用       | 计算开销      | 推荐场景           | 关键权衡                               |
| ------------- | ---------- | --------- | -------------- | ---------------------------------- |
| **AdamW**     | 高（12B/参数）  | 基准        | 通用首选，特别是有充足显存时 | 最稳定，但内存压力大                         |
| **Muon**      | 中（\~6B/参数） | 低（+SVD开销） | 大规模矩阵参数训练      | 需要实现 SVD，可能对非矩阵参数（如 Embedding）效果一般 |
| **Adafactor** | 低（4-8B/参数） | 低         | 显存受限（单卡/长序列）   | 需精细调参，收敛稍慢                         |

***

### 4 权重衰减与正则化

**为什么 LLM 预训练中不需要 Dropout 了（p=0.0）**

大规模语料上的单轮预训练很少产生过拟合，而Dropout引入的噪声会阻碍收敛速度，不使用Dropout的模型在下游任务（BLiMP、SQuAD、MNLI）上表现更好。

**权重衰减（Weight Decay）**

* **推荐值**：0.01-0.1（AdamW中）
* **μP（Maximal Update Parametrization）**：使用μP时，学习率和权重 decay 需按宽度进行缩放，可实现超参数从小模型向大模型的零样本迁移

***

### 5 超参搜索

确定了关键超参数后，下一个挑战是如何有效地微调它们。自动化优化方法可以通过系统地探索不同配置来简化这一过程。

<figure><img src="../../.gitbook/assets/image (45).png" alt=""><figcaption></figcaption></figure>

#### 5.1 网格搜索与随机搜索

网格搜索涉及测试超参数的每一种可能组合。虽然这能保证对搜索空间进行全面探索，但缺点是计算成本高昂。添加的参数越多，成本就呈指数级增长。相比之下，随机搜索从预定义的分布中随机采样超参数值。这种方法可能是有效的，因为只有少数超参数对性能有重大影响。此外，由于每次试验都是独立的，可以并行运行多个实验。

#### 5.2 贝叶斯优化

贝叶斯优化采用更智能的方法，将超参数调优视为一个需要学习的问题。它使用概率模型（通常是高斯过程）基于先前结果预测哪些超参数组合可能表现良好。这种方法在探索新可能性和聚焦有前景的配置之间取得平衡。

#### 数学框架

**目标**：在超参数空间 $$\mathcal{X}$$ 中找到最小化 $$f(x)$$（如模型验证损失）的配置 \$$x^\*\$$

$$\displaystyle x^* = \underset{x \in \mathcal{X}}{\arg\min} f(x)$$

其中 $$f(x)$$ 是**计算昂贵且不可导**的（需要完整训练一个模型）

#### 采集函数

**Upper Confidence Bound (UCB)**： $$\alpha_{\text{UCB}}(x) = \mu(x) + \beta \sigma(x)$$

* $$\beta$$ 控制探索权重，最大化UCB意味着选择"可能很好或不确定性很高"的点

**Expected Improvement (EI)**： $$\alpha_{\text{EI}}(x) = \mathbb{E}[\max(0, f(x^+) - f(x))]$$

* $$x^+$$ 是当前最佳观测值，EI最大化预期提升量

#### **流程**：

1. 用当前数据集 $$D_{1:t}$$ 拟合高斯过程 → 得后验 $$p(f|D)$$
2. 优化采集函数找到下一个评估点 $$x_{t+1} = \arg\max_x \alpha(x)$$
3. 评估 $$f(x_{t+1})$$（训练模型），更新 $$D_{t+1} = D_t \cup (x_{t+1}, f(x_{t+1}))$$

#### 5.3 基于种群的训练（PBT）

将超参数优化视为**在线学习问题**，而非静态搜索。利用**部分训练好的模型作为" warm-start"**，通过演化算法动态调整超参数，最大化**计算资源利用率**而非样本效率。

#### 演化机制原理

**种群结构**：

* 维持 $$N$$ 个模型（个体）同时训练，每个个体 $$i$$ 拥有：
  * 当前权重 $$\theta_i$$
  * 超参数配置 $$h_i$$（学习率、dropout率等）
  * 当前性能指标 $$P_i$$（验证准确率）

**异步更新循环**（每 $$T$$ 步执行一次）：

```
对于种群中每个个体 i：
    如果 performance(i) < 种群性能阈值（底部20%）：
        # 1. 淘汰（Exploitation）
        随机选择一个表现更好的个体 j（顶部20%）
        将 i 的权重复制为 j 的权重：θ_i ← θ_j
        将 i 的超参数复制为 j 的超参数：h_i ← h_j
        
        # 2. 变异（Exploration）
        对 h_i 进行随机扰动（如学习率 ×0.8 或 ×1.2）
        重置优化器状态（如Adam的moment估计）
```



### 参考

1. [https://arxiv.org/pdf/2507.17634](https://arxiv.org/pdf/2507.17634)
2. [https://spaces.ac.cn/archives/10542](https://spaces.ac.cn/archives/10542)
3. [https://latitude.so/blog/fine-tuning-llms-hyperparameter-best-practices](https://latitude.so/blog/fine-tuning-llms-hyperparameter-best-practices)
4. [https://arxiv.org/pdf/2507.07101](https://arxiv.org/pdf/2507.07101)
