# 强化学习

#### 1. **RLHF 基础框架**

* **两阶段流程详解**：Reward Model（RM）训练 + 强化学习策略优化（PPO）
* **偏好数据构造**：Bradley-Terry 模型假设、成对比较数据标注（Chosen vs Rejected）、Elo 评分体系在数据筛选中的应用

#### 2. **Reward Model（奖励模型）**

* **模型架构**：基于预训练模型的输出头改造（Regression Head）、共享参数 vs 冻结部分层
* **损失函数**：Pairwise Ranking Loss、Margin 设置、In-batch Negative Sampling
* **RM 的陷阱与局限**：Reward Hacking（奖励作弊）、分布外（OOD）泛化能力差、长度偏好（Length Bias）问题

#### 3. **PPO（Proximal Policy Optimization）算法深度解析**

* **Actor-Critic 架构**：策略模型（Actor）与价值模型（Critic）的初始化与更新策略
* **关键组件**：
  * KL 散度约束（KL Penalty）：参考模型（Reference Model）的固定与约束系数（β）调优
  * Generalized Advantage Estimation（GAE）：λ 参数设置与回报计算
  * Clipped Surrogate Objective：PPO 核心裁剪机制（ε 参数）
* **训练稳定性**：Value Loss 爆炸、Entropy Collapse、KL 散度突然增大的诊断与修复

#### 4. **Direct Preference Optimization（DPO）及变体**（现代主流，必须包含）

* **DPO 原理**：跳过显式 RM，直接偏好优化，交叉熵损失与 Bradley-Terry 模型的等价推导
* **DPO 的改进变体**：
  * **IPO（Identity Preference Optimization）**：解决 DPO 的过拟合问题
  * **KTO（Kahneman-Tversky Optimization）**：无需成对偏好，仅需二元好坏标签
  * **RPO（Robust Preference Optimization）**：处理噪声偏好数据
  * **SimPO（Simple Preference Optimization）**：去除参考模型，降低显存占用
* **DPO vs PPO**：显存效率对比、数据效率对比、何时选择何种算法

#### 5. **进阶 RL 方法（前沿必备）**

* **RLAIF（AI Feedback）**：Constitutional AI（CAI）流程，LLM 作为标注者生成 Critique 与修订
* **Self-Play 与对抗训练**：SPIN（Self-Play Fine-Tuning）、Gouda（基于博弈论的方法）
* **过程奖励模型（PRM）与结果奖励模型（ORM）**：OpenAI O1 背后的技术，Math/代码推理场景中的 Step-by-step 奖励建模
* **迭代优化（Iterative RL）**：拒绝采样微调（Rejection Sampling Fine-Tuning, RFT）、在线 DPO（Online DPO）、Iterative DPO

#### 6. **RL 工程实践与稳定性**

* **长度控制**：Length Penalty 设计，防止模型生成冗长无意义内容骗取高分
* **重复抑制**：针对重复生成（Repetition）的奖励塑形（Reward Shaping）
* **多轮 RL 的数据构造**：从 SFT 模型采样 → RM 打分 → 筛选高价值样本 → 继续训练（STaR、Vicuna 方法）
