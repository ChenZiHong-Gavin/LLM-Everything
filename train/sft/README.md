# 监督微调

#### 1. **数据工程**

* Base 模型与 Instruct 模型的差异
* **数据拼接策略**：Pack vs Truncation，Attention Mask 的正确处理
* **指令多样性设计**：多轮对话构造、工具调用数据格式
* **数据质量过滤**：基于 Reward Model 或困惑度的数据筛选，指令复杂度评估（Instag、Self-Instruct 方法）

#### 2. **参数高效微调**

* **LoRA/QLoRA**：低秩适配原理、秩（r）与缩放参数（α）的选择、目标模块选择（q\_proj/v\_proj/all）、4-bit/8-bit 量化与双量化（NF4/FP4）
* **高级 PEFT 方法**：DoRA（Weight-Decomposed LoRA）、PiSSA、Prefix-Tuning、P-Tuning v2、Adapter 的对比与选型
* **全参数微调 vs PEFT**：显存占用对比、收敛速度差异、何时必须全参数微调（领域知识大幅更新场景）

#### 3. **训练策略与稳定性**

* **灾难性遗忘（Catastrophic Forgetting）缓解**：混合预训练数据（Replay Buffer）、LISA（Layerwise Importance Sampling）、LFQA（Learning with Forgettable Examples）
* **长上下文微调**：从预训练的 4K/8K 扩展到 32K/128K 的渐进式微调（Positional Interpolation、YaRN 在 SFT 阶段的应用）
* **多轮对话位置编码处理**：NTK-aware 扩展在对话场景中的具体实现

#### 4. **评估**

* **指令跟随能力评估**：MT-Bench、AlpacaEval、IFEval（指令遵循专项测试）
* **安全性对齐初步**：SFT 阶段的安全数据注入（Refusal Training）、毒性内容过滤策略



### 参考

1. [https://zhuanlan.zhihu.com/p/809229182](https://zhuanlan.zhihu.com/p/809229182)
