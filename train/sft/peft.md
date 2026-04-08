# 参数高效微调

### 1 Adapter

<figure><img src="../../.gitbook/assets/image (50).png" alt=""><figcaption></figcaption></figure>

Adapter（2019年提出）在Transformer子层（注意力层+FFN层）后插入瓶颈式（Bottleneck）适配模块，通过下投影→非线性激活→上投影的方式，在不改变原始参数的情况下引入任务特定知识。

**数学表达**： 对于输入特征 $$\mathbf{h} \in \mathbb{R}^d$$，Adapter层的输出为： $$\mathbf{h}' = \mathbf{h} + f(\mathbf{h}W_{down})W_{up}$$ 其中：

* $$W_{down} \in \mathbb{R}^{d \times m}$$：下投影矩阵（$$m \ll d$$，通常 $$m = 64$$）
* $$W_{up} \in \mathbb{R}^{m \times d}$$：上投影矩阵
* $$f(\cdot)$$：非线性激活（通常使用ReLU或GELU）
* **残差连接**：原始特征 $$\mathbf{h}$$ 被保留，Adapter学习残差变换 $$\Delta\mathbf{h}$$

**标准Adapter配置**：

* **插入位置**：每个Transformer层的两个子层后（Post-Attention + Post-FFN）
* **瓶颈维度**：$m$ 通常为原始维度 $$d$$ 的1/64到1/4（如BERT-base $$d=768$$，则 $$m=48-192$$）
* **参数占比**：仅增加约 **2-4%** 的可训练参数

**主要局限**：

* **推理时必须加载Adapter**：无法像LoRA那样将权重合并回基座模型，增加部署复杂度
* **深层信息瓶颈**：当 $$m$$ 过小时，信息压缩可能导致语义损失

### 2 Prefix-Tuning

<figure><img src="../../.gitbook/assets/image (51).png" alt=""><figcaption></figcaption></figure>

Prefix-Tuning (2021年提出）在Transformer的**每一层**的 Key 和 Value 矩阵前，前置可学习的连续向量（Soft Prompts），而非修改模型参数或添加输入层提示。这相当于在注意力计算中引入了**虚拟的"锚点 token"**。

**数学机制**： 对于第 $$i$$ 层自注意力机制，原始 Key 为 $$K_i \in \mathbb{R}^{n \times d}$$，Prefix-Tuning将其扩展为： $$K'_i = [P^K_i; K_i], \quad V'_i = [P^V_i; V_i]$$ 其中 $$P^K_i, P^V_i \in \mathbb{R}^{p \times d}$$ 为可学习前缀向量，$$p$$ 为前缀长度（通常10-50 tokens）。

**架构特点**：

* **层级干预**：在**所有层**（不仅是输入层）添加前缀，深度影响注意力模式
* **重参数化技巧**：使用小型MLP生成前缀参数，避免直接优化不稳定： $$P_i = \text{MLP}(P'_i)$$ 训练完成后丢弃MLP，仅保留生成的 $$P_i$$

**为什么有效**：

* **全局偏置注入**：这些前缀向量参与**所有位置**的注意力计算（因为每个真实 token 都会 attend 到它们），相当于在每一层都施加了任务特定的**注意力偏置（attention bias）**
* **细粒度控制**：不同于输入层提示（仅影响第一层表征），深层前缀可以直接干预高层语义特征的聚合方式，**逐层调整**模型对任务的理解

**超参数配置**：

* 前缀长度 $$p$$：10-100（任务越复杂，所需前缀越长）
* 可训练参数占比：**0.1-0.5%**

### 3 LoRA、QLoRA

#### 3.1 LoRA

<figure><img src="../../.gitbook/assets/image (48).png" alt=""><figcaption></figcaption></figure>

**核心假设**： 模型微调时的参数更新 $$\Delta W$$ 具有**低内在秩**（Low Intrinsic Rank）。对于预训练权重 $$W_0 \in \mathbb{R}^{d \times k}$$，LoRA将其更新约束为低秩矩阵乘积：

$$W' = W_0 + \Delta W = W_0 + BA$$

其中 $$B \in \mathbb{R}^{d \times r}$$，$$A \in \mathbb{R}^{r \times k}$$，且秩 $$r \ll \min(d, k)$$。

**初始化策略**：

* **矩阵A**：使用Kaiming均匀初始化（类似神经网络标准初始化）
* **矩阵B**：零初始化（确保训练开始时 $$\Delta W = 0$$，保持预训练知识）

**为什么要这样设计？**

这种**非对称初始化**（A随机、B零）是LoRA训练稳定性的关键设计，包含两个个层面的考量：**训练起点控制**、**梯度流动机制**。

* 零初始化：确保"暖启动"（Warm Start）
  * 训练开始时 $$\Delta W = BA = 0$$，使模型初始状态**严格等价于**预训练模型。
* A 的Kaiming初始化：保持梯度流动与方向探索
  * 虽然B初始为0，但A**必须非零**（Kaiming/He初始化或标准高斯），原因涉及**梯度反向传播机制**：
    * **梯度计算**： 对于损失函数 $$\mathcal{L}$$，B的梯度为： $$\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial (BA)} \cdot A^T = \frac{\partial \mathcal{L}}{\partial \Delta W} \cdot A^T$$
  * **如果A也初始化为0**：
    * $$\frac{\partial \mathcal{L}}{\partial B} = 0$$（梯度消失）
    * B将永远无法更新（被困在0点）
    * LoRA路径 $$BA$$ 永远保持0，失去微调能力

**缩放机制**： 引入缩放系数 $$\frac{\alpha}{r}$$ 控制LoRA输出的影响范围： $$h = W_0x + \frac{\alpha}{r} \cdot BAx$$

**目标模块选择策略**：

* **`q_proj, v_proj`**：参数效率最高（仅占0.5%参数），可达全微调98%性能
* **`q_proj, k_proj, v_proj, o_proj`**：标准配置，平衡性能与显存
* **`all_linear`**（注意力+MLP）：领域适配场景，接近全微调效果

#### 3.2 QLoRA

<figure><img src="../../.gitbook/assets/image (49).png" alt=""><figcaption></figcaption></figure>

**4-bit NormalFloat (NF4) 量化**：

* 理论基础：神经网络权重通常服从零均值正态分布 $$N(0, \sigma^2)$$
* **信息论最优**：NF4将4-bit量化值映射到正态分布的分位数，相对于均匀分布（FP4）最大化信息熵
* **分块量化**：每64个参数独立量化，配备独立的量化常数 $c$，减少异常值影响

**双量化（Double Quantization）**： 对量化常数 $$c$$（32-bit）进行二次量化至8-bit：

* 第一层：$$c_{32} \rightarrow c_{8}$$（节省显存）
* 第二层：反量化 $$c_{8} \rightarrow c_{32}$$（计算时恢复精度）

**分页优化器（Paged Optimizers）**： 利用CPU内存作为GPU显存的页缓冲区，当GPU OOM时自动将优化器状态分页到CPU，实现单卡训练超大模型（如65B模型在48GB显存上训练）。

**QLoRA配置清单**：

```python
# 量化配置
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 启用4-bit
    bnb_4bit_quant_type="nf4",           # NF4优于FP4约15-20%
    bnb_4bit_use_double_quant=True,      # 双量化节省~1GB（70B模型）
    bnb_4bit_compute_dtype=bfloat16      # 计算时反量化至bf16
)

# LoRA配置
lora_config = LoraConfig(
    r=64,                # QLoRA可使用更高秩（量化节省的显存可用于增大r）
    lora_alpha=16,       # α/r = 0.25，保守更新（量化模型需稳定训练）
    target_modules="all_linear",
    lora_dropout=0.05,
    use_rslora=True      # 可选：秩稳定LoRA，解决高秩训练不稳定问题
)
```

### 参考

1. [https://www.aidoczh.com/adapterhub/methods.html#lora](https://www.aidoczh.com/adapterhub/methods.html#lora)
