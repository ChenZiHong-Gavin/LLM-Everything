# GRPO

GRPO（Group Relative Policy Optimization，群组相对策略优化） 是一种基于强化学习的策略优化算法 ，旨在提升大语言模型在复杂任务（如数学推理、编程）中的表现。

GRPO 最早在 _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_ 这篇论文中提出。

<figure><img src="../../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

传统 PPO 需要维护**四个模型**：策略模型（Policy）、参考模型（Reference）、奖励模型（Reward）和价值模型（Critic）。其中 Critic 模型通常与策略模型同等规模，带来显著的显存和计算开销。

GRPO 的关键创新在于**移除 Critic 模型**，改为对每个问题采样**一组输出**，用组内奖励的均值作为 baseline 来估计优势（Advantage）：

* **采样阶段**：对于每个问题 $$q$$，从旧策略 $$\pi_{\theta_{old}}$$ 中采样 $$G$$ 个输出 $${o_1, o_2, \cdots, o_G}$$
* **奖励计算**：用奖励模型为每个输出打分，得到 $$G$$ 个奖励 $${r_1, r_2, \cdots, r_G}$$
* **优势估计**：对每个输出的奖励进行组内归一化，作为该输出中所有 token 的优势值：

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_1, \cdots, r_G\})}{\text{std}(\{r_1, \cdots, r_G\})}$$

这种"群组相对"的方式天然契合奖励模型的比较性质（奖励模型通常基于同一问题的输出比较进行训练）。

### 奖励模型

DeepseekMath的原始论文中，研究团队使用了预训练的 **过程奖励模型（Process Reward Model, PRM）。**



### 目标函数与算法流程

GRPO 的目标函数如下：

$$J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min\left( \frac{\pi_\theta}{\pi_{\theta_{old}}} \hat{A}_{i,t}, \text{clip}(\frac{\pi_\theta}{\pi_{\theta_{old}}}, 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right) - \beta D_{KL}[\pi_\theta \| \pi_{ref}] \right) \right]$$

其中：

* 第一项是带裁剪（clip）的策略梯度，与 PPO 类似
* 第二项是 KL 散度惩罚，**直接加到 loss 中**而非奖励里，避免 advantage 计算复杂化
* $\epsilon$ 和 $\beta$ 是超参数

算法流程（迭代式 GRPO）：

1. 将当前策略设为参考模型 $\pi\_{ref} \leftarrow \pi\_\theta$
2. 对每个训练步骤：
   * 采样 batch 数据，更新旧策略 $\pi\_{\theta\_{old\}} \leftarrow \pi\_\theta$
   * 对每个问题采样 $G$ 个输出
   * 计算奖励和组相对优势 $\hat{A}\_{i,t}$
   * 进行 $\mu$ 次 GRPO 迭代更新策略
3. 可选：用 replay 机制持续更新奖励模型

***

### 监督信号类型

GRPO 支持两种监督方式：

| 类型                      | 说明         | 优势计算                             |
| ----------------------- | ---------- | -------------------------------- |
| **Outcome Supervision** | 只在输出结束时给奖励 | 所有 token 共享同一个归一化奖励 $\hat{r}\_i$ |
| **Process Supervision** | 对每个推理步骤给奖励 | 使用每个步骤的归一化奖励                     |

***

### 为什么 GRPO 有效？

根据 2026 年 3 月的最新理论分析，GRPO 的策略梯度本质上是一种 **U-Statistic**，具有以下性质：

1. **Oracle 等价性**：GRPO 渐近等价于一个拥有完美价值函数的 Oracle 策略梯度算法
2. **最优性**：在广泛的策略梯度算法类中，GRPO 能达到渐近最优性能
3. **可扩展性**：存在通用的群组大小缩放规律，可用于指导最优组大小选择

相比 PPO，GRPO 避免了训练 Critic 带来的方差和不稳定性；相比 DPO，GRPO 支持在线探索和迭代训练，更适合复杂推理任务。

***

### 实际应用与影响

GRPO 已成为当前 LLM 推理训练的主流方法：

* **DeepSeekMath**：在 MATH 竞赛题上达到 51.7% 准确率，接近 GPT-4 水平
* **DeepSeek-R1**：使用 GRPO 进行多轮强化学习，培养模型的自我验证和迭代反思能力
* **低成本训练**：配合 LoRA 技术，甚至可在 **16GB 显存**下将 1B 参数模型训练为推理模型

后续研究也基于 GRPO 发展出多种改进：

* **DAPO**：解耦裁剪和动态采样策略
* **Scaf-GRPO**：通过分层提示（scaffolding）解决"学习悬崖"问题（过难题目导致零奖励信号）
* **Dr. GRPO**：针对医学推理等场景，消除奖励与输出长度的耦合

***

### 总结

GRPO 通过**群组相对优势估计**替代了传统 PPO 中的 Critic 模型，将显存开销降低近 **50%**，同时保持甚至提升了复杂推理任务的训练效果。它代表了 RLHF 技术向**更简单、更高效**方向演进的趋势，使得在资源受限环境下训练具备深度推理能力的模型成为可能。





### 参考

1. [https://blog.csdn.net/Eternity\_\_Aurora/article/details/149080119](https://blog.csdn.net/Eternity__Aurora/article/details/149080119)
2. [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
