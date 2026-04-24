# DAPO

**DAPO**（**D**ecoupled Clip and Dynamic s**A**mpling **P**olicy **O**ptimization）是字节跳动 Seed 团队与清华 AIR 在 2025 年 3 月提出的一套大规模 LLM 强化学习算法与系统，论文为 _DAPO: An Open-Source LLM Reinforcement Learning System at Scale_（Yu et al., 2025, arXiv:2503.14476）。

DAPO 在 GRPO 的基础上引入了 **4 项关键技术改进**，用 Qwen2.5-32B 作为基座模型，在 AIME 2024 数学竞赛上取得了 **50 分**，超过了同期 DeepSeek-R1-Zero-Qwen-32B 的 47 分，且训练步数仅为后者的 **50%**。

### 1 动机

vanilla GRPO 训练 32B 模型时观察到 4 个严重问题：

| 问题                        | 具体表现                                                        |
| ------------------------- | ----------------------------------------------------------- |
| **熵崩塌（Entropy Collapse）** | 策略熵迅速下降，模型早期就锁定到少数生成模式，丧失探索能力                               |
| **梯度无效（Zero Gradient）**   | 一组采样中所有输出全对或全错时，组内奖励标准差为 0，优势全为 0，整 batch 梯度白算              |
| **长样本被惩罚**                | 在长 CoT 场景下（数万 token），GRPO 的样本级 loss 平均使长输出中的低质量 token 难以被惩罚 |
| **奖励噪声**                  | 超长输出被截断后仍按"错误答案"给负奖励，引入虚假信号                                 |

DAPO 的 4 项技术分别针对性地解决这 4 个问题。

### 2 四大核心技术

#### 2.1 Clip-Higher

PPO/GRPO 的 clip 操作用对称区间 $$[1-\epsilon,\ 1+\epsilon]$$（通常 $$\epsilon=0.2$$）。这对**低概率 token 的提升**非常不利——一个概率 0.01 的 token，最多只能提升到 $$0.01 \times 1.2 = 0.012$$，几乎无法被强化。

但是，当我们把上下界拆开：

$$\text{clip}\left(\rho_{i,t},\ 1-\epsilon_{\text{low}},\ 1+\epsilon_{\text{high}}\right)$$

实践中 $$\epsilon_{\text{low}}=0.2$$ 保持不变，$$\epsilon_{\text{high}}$$ 放大到 **0.28**。这样允许低概率 token 有更大的上升空间，**保留探索**，避免熵崩塌。

> 直觉：下界控制"别太激进地降低坏 token 的概率"，上界控制"别太激进地拔高好 token 的概率"。两者的合理范围本来就不对称。

#### 2.2 Dynamic Sampling

GRPO 对每个 prompt 采样 $$G$$ 个回答（如 $$G=16$$）。对于太简单或太难的题目，常出现"全对"或"全错"——此时组内归一化后所有优势都是 0，**这些样本对梯度没有任何贡献**，浪费算力还稀释了 batch 的有效信号。

**DAPO使用过采样 + 动态过滤**。持续采样直到 batch 中**正确率严格在 (0, 1) 之间**的 prompt 数量达到目标值：

$$0 < \left| \{o_i \mid \text{is\_correct}(q, o_i)\} \right| < G$$

也就是说，每个被采纳的 prompt 都至少有 1 个对、1 个错的回答，确保组内优势非零。

**收益：** 训练效率显著提升——虽然单步采样成本增加，但有效梯度密度更高，整体收敛更快。

#### 2.3 Token-Level Policy Gradient Loss

GRPO 原始 loss 是**样本级平均**：

$$\mathcal{L}_{\text{GRPO}} = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\ell_{i,t}$$

注意每个样本先内部按长度归一化，再在样本间平均。这意味着：**一个 100 token 的样本和一个 10000 token 的样本对梯度的贡献相同**。在长 CoT 场景下，长输出里的每个 token 被严重稀释，无意义的重复、胡言乱语难以被有效惩罚。

DAPO的做法是改为**所有 token 一起平均**：

$$\mathcal{L}_{\text{DAPO}} = \frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\ell_{i,t}$$

长样本里的每个 token 现在和短样本里的 token **同等权重**。这样：

* 长输出中的高质量推理 token 能获得更大权重 → 鼓励有效长链推理
* 长输出中的退化模式（重复、乱码）也能被有效惩罚 → 抑制低质量长输出

#### 2.4 Overlong Reward Shaping

GRPO 训练时给生成长度设置了硬上限（如 20480 token）。被截断的样本即使推理过程正确，也因没输出最终答案被判错，得到 -1 奖励——这是**虚假的负信号**，会误导模型。

DAPO采用了两种策略：

**(a) Overlong Filtering**：直接把被截断的样本从 loss 中屏蔽，不参与梯度计算。

**(b) Soft Overlong Punishment**：在硬上限之前设置一个软缓冲区，长度超过软阈值后逐步线性扣分，超过硬上限才完全屏蔽：

$$R_{\text{length}}(y) = \begin{cases} 0 & |y| \le L_{\text{max}} - L_{\text{cache}} \\ \frac{(L_{\text{max}} - L_{\text{cache}}) - |y|}{L_{\text{cache}}} & L_{\text{max}} - L_{\text{cache}} < |y| \le L_{\text{max}} \\ -1 & |y| > L_{\text{max}} \end{cases}$$

这样模型被**温和地告知"该收敛了"**，而不是粗暴地把"思考太久"和"答错"混为一谈。

### 3 目标函数

整合 4 项改进后的完整目标函数：

$$\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\ \{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{\sum_i |o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|} \min\Big(\rho_{i,t}\hat{A}_{i,t},\ \text{clip}(\rho_{i,t},\ 1-\epsilon_{\text{low}},\ 1+\epsilon_{\text{high}})\hat{A}_{i,t}\Big) \right]$$

约束条件：

$$\text{s.t.}\quad 0 < \big|\{o_i \mid \text{is\_equivalent}(a, o_i)\}\big| < G$$

注意几个细节：

* **没有 KL 惩罚项**：作者发现在长 CoT 推理任务中，策略会大幅偏离初始模型，KL 约束反而限制了探索。直接去掉。
* **没有 Critic**：与 GRPO 一致，仍用组内归一化估计优势
* **优势仍是 token 级共享**：$$\hat{A}_{i,t} = \tilde{r}_i$$，组内 z-score

### 4 算法流程

```
输入：初始策略 π_θ，奖励函数 R，训练数据 D
重复直到收敛：
    1. 采样 batch B = {}
    2. While |B| < batch_size:
         a. 从 D 采样若干 prompt
         b. 对每个 prompt q 采样 G 个输出 {o_i}
         c. 用 R 给每个输出打分，计算正确率
         d. 仅保留 0 < 正确率 < 1 的 prompt 加入 B
    3. 对 B 中每组计算优势 Â_{i,t}（组内归一化）
    4. 应用超长奖励整形修正 r_i
    5. 进行 μ 次梯度更新（用 token 级 loss + 解耦 clip）
```



### 参考

1. [https://arxiv.org/pdf/2503.14476](https://arxiv.org/pdf/2503.14476)

