# RLHF基础

RLHF（Reinforcement Learning from Human Feedback）是将人类偏好引入大语言模型训练的核心技术范式。与传统的 SFT 不同，RLHF 不再追求"模仿人类书写的正确答案"，而是学习"生成人类更偏好的结果"。这一范式的完整流程可分为两个紧密衔接的阶段：

### 阶段一：训练奖励模型

Reward Model（奖励模型，简称 RM）是 RLHF 的"裁判系统"。其核心任务是为任意给定的（Prompt, Response）对输出一个标量奖励值，量化人类对该回复的偏好程度。

RM 通常基于预训练语言模型（如 LLaMA、GPT 系列）进行改造。具体实现上，在模型的最后一层隐藏状态之上添加一个**回归头**，将高维表示映射为单一标量值：

$$r_\theta(x, y) \in \mathbb{R}$$

其中 $$x$$ 表示输入提示（Prompt），$$y$$ 表示模型生成的回复（Response），$$\theta$$ 为 RM 的可训练参数。

架构选择上有两种主流方案：

* **共享参数方案**：复用预训练模型的全部参数，仅替换最后的输出层。优势在于保持强大的语义理解能力，劣势是训练成本较高。
* **冻结部分层方案**：冻结模型的底层参数（通常占 70%\~80%），仅微调顶层 Transformer 层和回归头。这能在保持 RM 判别能力的同时显著降低显存占用和训练时间。

**训练目标：Pairwise Ranking Loss**

RM 的训练数据由成对比较构成：对于同一 Prompt $$x$$，人类标注者从两个候选回复 $$y_w$$（Win，被选中的）和 $$y_l$$（Lose，被拒绝的）中选出更优者。但神经网络需要数值才能优化。

Bradley-Terry 模型的核心洞察是：

> <mark style="color:$success;">如果 A 比 B 好，那么 A 的"质量分数"应该显著高于 B 的分数。</mark>

$$\mathcal{L} = -\log \sigma\bigl(r_\theta(x, y_w) - r_\theta(x, y_l)\bigr)$$

**符号解释：**

* $$y_w$$：Win（被人类选中的更好回复）
* $$y_l$$：Lose（被人类拒绝的较差回复）
* $$r_\theta(x, y)$$：奖励模型输出的标量分数（越高越好）
* $$\sigma$$：Sigmoid 函数，$$\sigma(z) = \frac{1}{1+e^{-z}}$$

对损失函数求导（链式法则），你会发现梯度为：

$$\frac{\partial \mathcal{L}}{\partial \theta} = \underbrace{\bigl[1 - \sigma(r_w - r_l)\bigr]}_{\text{误差项}} \cdot \underbrace{\bigl[\frac{\partial r_w}{\partial \theta} - \frac{\partial r_l}{\partial \theta}\bigr]}_{\text{奖励值对参数的梯度}}$$

**含义解读：**

1. **误差项** $$[1 - \sigma(r_w - r_l)]$$：
   * 当 $$r_w \gg r_l$$ 时，Sigmoid 接近 1，误差接近 0 → **梯度很小，几乎不再更新**
   * 当 $$r_w \approx r_l$$ 时，误差接近 0.5 → **梯度较大，大力调整**）
2. **调整方向**：
   * 增加获胜回复 $$y_w$$ 的奖励值（$$\frac{\partial r_w}{\partial \theta}$$ 为正）
   * 降低失败回复 $$y_l$$ 的奖励值（$$-\frac{\partial r_l}{\partial \theta}$$ 为负）

模型像一个跷跷板，一端抬起 Win，一端压低 Lose。

**Margin** $$\delta$$ **的作用**

带 Margin 的版本： $$\mathcal{L} = -\log \sigma\bigl(r_w - r_l - \delta\bigr)$$

**为什么要加？**

想象两个场景：

* **场景 A**：$$r_w = 0.8, r_l = 0.7$$，差值 0.1
* **场景 B**：$$r_w = 0.8, r_l = -0.5$$，差值 1.3

没有 Margin 时，只要 $$r_w > r_l$$ 并且 $$r_w \ r_l$$ 的距离拉大，损失就会变小。

但问题在于，**微小的正差距（如 0.1）对应的损失（0.64）虽然比** $$r_w = r_l$$ 的 **0.69 略低，但这个优化信号较弱**，模型可能缺乏动力继续大幅拉开差距。

设置 $$\delta = 0.5$$ 后：

* 场景 A：$$0.8 - 0.7 - 0.5 = -0.4$$，Sigmoid(-0.4) ≈ 0.4，损失很大，模型必须继续拉大差距
* 场景 B：$$0.8 - (-0.5) - 0.5 = 0.8$$，Sigmoid(0.8) ≈ 0.69，损失较小

Margin 强迫模型不仅要知道"谁好"，还要明确"好多少"，避免对相近质量的回复给出暧昧的分数。

如果你熟悉分类任务，这个损失其实是**二元交叉熵的特例**：

标准的二元交叉熵： $$\mathcal{L}_{BCE} = -\bigl[y \cdot \log(p) + (1-y) \cdot \log(1-p)\bigr]$$

在我们的设定中：

* 标签 $$y = 1$$（始终假设 $$y_w$$ 应该赢）
* 概率 $$p = \sigma(r_w - r_l)$$

代入得到： $$\mathcal{L} = -\log \sigma(r_w - r_l) - 0 = -\log \sigma(r_w - r_l)$$

完全一样！所以你可以把它理解为：**在比较两个回复时，预测"左边比右边好"这个二元事件的概率，并最大化该概率**。



### 阶段二：强化学习策略优化

获得训练好的 RM 后，进入 RLHF 的第二阶段：使用强化学习算法（这里以 PPO 为例，Proximal Policy Optimization）优化语言模型策略 $$\pi_\phi$$，使其生成能获得更高 RM 分数的回复。

**优化目标**

$$\max_{\phi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\phi(y|x)} \left[ r_\theta(x, y) \right] - \beta \mathbb{D}_{\text{KL}} \left[ \pi_\phi(y|x) \| \pi_{\text{ref}}(y|x) \right]$$

目标函数包含两个关键项：

* **奖励最大化项**：鼓励策略生成高 RM 分数的回复
* **KL 散度约束项**：防止策略 $$\pi_\phi$$ 偏离参考模型 $$\pi_{\text{ref}}$$（通常是 SFT 模型）太远，维护语言模型的基础能力和输出分布稳定性

$$\beta$$ 为 KL 惩罚系数，是 RLHF 中最重要的超参数之一，通常需要在 0.01\~0.2 之间精细调优。

**Actor-Critic 架构**

PPO 算法采用经典的 Actor-Critic 结构：

* **Actor（策略模型）**：即被训练的语言模型，负责生成回复。其参数通过策略梯度更新，朝着高奖励方向优化。
* **Critic（价值模型）**：估计给定状态（已生成的 Token 序列）的期望累计回报，用于计算 Advantage（优势函数），降低梯度估计的方差。

Critic 通常从 RM 或 SFT 模型初始化，其输出头被改造为价值输出。价值模型的损失函数为： $$\mathcal{L}_{\text{critic}} = \left( R_t - V_\psi(x, y_{<t}) \right)^2$$

其中 $$R_t$$ 为时刻 $$t$$ 的折扣回报。

**关键算法组件**

1. **Generalized Advantage Estimation（GAE）**：平衡偏差与方差的优势估计方法 $$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$ 其中 $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$ 为 TD 误差，$$\gamma$$ 为折扣因子，$$\lambda \in [0,1]$$ 控制偏差-方差权衡。$$\lambda=1$$ 时无偏但方差大，$$\lambda=0$$ 时有偏但方差小。
2. **Clipped Surrogate Objective**：PPO 的核心创新，限制单步策略更新幅度 $$L^{\text{CLIP}}(\phi) = \mathbb{E}_t \left[ \min \left( r_t(\phi) \hat{A}_t, \text{clip}(r_t(\phi), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$ 其中 $$r_t(\phi) = \frac{\pi_\phi(a_t|s_t)}{\pi_{\phi_{\text{old}}}(a_t|s_t)}$$ 为概率比，$$\epsilon$$（通常 0.1\~0.2）为裁剪阈值。









