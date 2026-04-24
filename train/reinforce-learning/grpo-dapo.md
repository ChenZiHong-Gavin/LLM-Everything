# GRPO

GRPO（Group Relative Policy Optimization，群组相对策略优化） 是一种基于强化学习的策略优化算法 ，旨在提升大语言模型在复杂任务（如数学推理、编程）中的表现。

GRPO 最早在 _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_ 这篇论文中提出。

<figure><img src="../../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

传统 PPO 需要维护**四个模型**：策略模型（Policy）、参考模型（Reference）、奖励模型（Reward）和价值模型（Critic）。其中 Critic 模型通常与策略模型同等规模，带来显著的显存和计算开销。

| 模型                         | 作用                         |
| -------------------------- | -------------------------- |
| 策略模型（Policy Model）         | 当前正在训练的语言模型                |
| 参考模型（Reference Model）      | 冻结的旧策略，用于 KL 约束            |
| 奖励模型（Reward Model）         | 对输出质量打分                    |
| 价值模型（Critic / Value Model） | 估计状态价值函数，用于计算优势（Advantage） |

GRPO 的关键创新在于**移除 Critic 模型**，改为对每个问题采样**一组输出**，用组内奖励的均值作为 baseline 来估计优势（Advantage）：

* **采样阶段**：对于每个问题 $$q$$，从旧策略 $$\pi_{\theta_{old}}$$ 中采样 $$G$$ 个输出 $${o_1, o_2, \cdots, o_G}$$
* **奖励计算**：用奖励模型为每个输出打分，得到 $$G$$ 个奖励 $${r_1, r_2, \cdots, r_G}$$
* **优势估计**：对每个输出的奖励进行组内归一化，作为该输出中所有 token 的优势值：

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_1, \cdots, r_G\})}{\text{std}(\{r_1, \cdots, r_G\})}$$

这种"群组相对"的方式天然契合奖励模型的比较性质（奖励模型通常基于同一问题的输出比较进行训练）。

### 1 奖励模型

DeepseekMath的原始论文中，研究团队使用了预训练的 **过程奖励模型（Process Reward Model, PRM）。**

PRM 要对推理过程的**每一步**给出分数（比如 0\~1，表示"这一步是否正确/合理"）。难点在于：**步骤级别的标注从哪来？** 人工对每一步打标签成本极高。

OpenAI 在 _Let's Verify Step by Step_（Lightman et al., 2023）中采用纯人工标注：

* 让模型生成多步推理（数学题）
* 标注员对每一步打三档标签：**正确 / 中性 / 错误**
* 累计标注了 **80 万** 条步骤级标签 → PRM800K 数据集

DeepSeekMath 用的是 _Math-Shepherd_（Wang et al., 2024）提出的**自动化标注**思路，核心是用"蒙特卡洛 rollout"代替人工：

**步骤：**

1. 对一道有标准答案的数学题，让基础模型生成完整解答 $$s_1, s_2, \ldots, s_n$$（按推理步骤切分）
2. 对每个中间步骤 $$s_i$$，**从该步骤的状态出发**，让模型继续 rollout $$N$$ 次（比如 8 次或 16 次），看最终答案对不对
3. 把"从 $$s_i$$ 出发能得到正确答案的概率"作为该步骤的软标签：

$$\text{label}(s_i) = \frac{\#\{\text{正确的 rollout}\}}{N}$$

如果某一步是正确的，从它继续推理大概率能得到正确答案；如果这一步走偏了，后续 rollout 很难再回到正确答案。**最终答案的对错反向给中间步骤打分。**&#x8FD9;样就把"昂贵的步骤标注"转化为"便宜的最终答案验证"——只要题目有标准答案，就能自动生成大量步骤级标签。

数据准备好后，训练就比较直接了：

**模型结构：** 在预训练 LLM 上加一个 token 级的分类头（或回归头）

**输入格式：**

```
Question: ...
Step 1: xxx <step_tag>
Step 2: yyy <step_tag>
...
```

在每个特殊 token `<step_tag>` 的位置预测该步骤的标签。

**损失函数：**

* 软标签（如 Math-Shepherd）：用 MSE 或交叉熵回归到概率值 $$\mathcal{L} = \sum_i \text{BCE}\big(\hat{p}(s_i),\ \text{label}(s_i)\big)$$
* 硬标签：标准的二分类交叉熵

### 2 目标函数与算法流程

GRPO 的目标函数如下：

$$J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min\left( \frac{\pi_\theta}{\pi_{\theta_{old}}} \hat{A}_{i,t}, \text{clip}(\frac{\pi_\theta}{\pi_{\theta_{old}}}, 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right) - \beta D_{KL}[\pi_\theta \| \pi_{ref}] \right) \right]$$

其中：

* 第一项是带裁剪的策略梯度，与 PPO 类似
* 第二项是 KL 散度惩罚，**直接加到 loss 中**而非奖励里，避免 advantage 计算复杂化
* $$\epsilon$$ 和 $$\beta$$ 是超参数

算法流程：

1. 将当前策略设为参考模型 $$\pi_{ref} \leftarrow \pi_\theta$$
2. 对每个训练步骤：
   * 采样 batch 数据，更新旧策略 $$\pi_{\theta_{old}} \leftarrow \pi_\theta$$
   * 对每个问题采样 $$G$$ 个输出
   * 计算奖励和组相对优势 $$\hat{A}_{i,t}$$
   * 进行 $$\mu$$ 次 GRPO 迭代更新策略
3. 用 replay 机制持续更新奖励模型

### 3 为什么 GRPO 有效

GRPO 的策略梯度本质上是一种 **U-Statistic**，具有以下性质：

1. **Oracle 等价性**：GRPO 渐近等价于一个拥有完美价值函数的 Oracle 策略梯度算法
2. **最优性**：在广泛的策略梯度算法类中，GRPO 能达到渐近最优性能
3. **可扩展性**：存在通用的群组大小缩放规律，可用于指导最优组大小选择

相比 PPO，GRPO 避免了训练 Critic 带来的方差和不稳定性；相比 DPO，GRPO 支持在线探索和迭代训练，更适合复杂推理任务。



### 参考

1. [https://blog.csdn.net/Eternity\_\_Aurora/article/details/149080119](https://blog.csdn.net/Eternity__Aurora/article/details/149080119)
2. [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
