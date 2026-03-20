# 预训练

**预训练**是 LLM 能力形成的基石阶段。作为自监督学习在海量无标注文本上的规模化实践，预训练通过训练目标 Next Token Prediction 将分布式语义知识压缩至模型参数中，构成后续监督微调（SFT）与强化学习（RL）对齐阶段的初始化基础。

{% stepper %}
{% step %}
#### **数据工程**

* **大规模语料采集**：Common Crawl、GitHub、书籍、论文、代码等多源数据
* **质量控制**：启发式过滤（长度 / 重复 / 符号比）、基于模型的质量打分（如 FastText/Perplexity 过滤）、去重策略（MinHash/LSH）、隐私脱敏（PII 移除）
* **数据配比**：不同领域数据的最优混合比例（DoReMi、课程学习）
* **Tokenization 再训练**：大规模语料下 BPE/SentencePiece 的训练与词汇表设计考量
{% endstep %}

{% step %}
#### **预训练策略与超参配置**

* **学习率调度**：Warmup-Stable-Decay（WSD）、Cosine decay with restarts、Linear decay 的具体选择
* **Batch Size 调度**：从 small batch 逐渐增大到 full batch（LARGE BATCH TRAINING）
* **优化器细节**：AdamW（β1/β2/ε 的设置）、Adafactor、Sophia 等新型优化器对比
* **权重衰减与正则化**：Dropout 在预训练中的使用（现代 LLM 通常不用，但需要说明）
{% endstep %}

{% step %}
#### **训练稳定性与效率技巧**

* **混合精度训练**：FP16/BF16 的选择标准、Loss scaling 机制、GradScaler 原理
* **激活重计算（Gradient Checkpointing）**：时间与显存的 trade-off
* **梯度裁剪（Gradient Clipping）**：global norm 阈值设置
* **损失尖峰（Loss Spike）处理**：检测机制与回滚策略
* **确定性训练（Deterministic Training）**：保证可复现性的种子设置与 CUDA 确定性配置
{% endstep %}

{% step %}
### **长上下文扩展策略**

* **渐进式扩展**：从 4k → 32k → 128k 的多阶段预训练
* **位置编码适配**：ALiBi、RoPE 外推/内插（NTK-aware, YaRN）、动态位置编码
* **稀疏注意力**：Sliding Window、FlashAttention 在预训练中的集成
{% endstep %}

{% step %}
### **评估与监控体系**

* **中间评估（Mid-training Eval）**：PPL 之外，阶段性的下游任务 zero-shot/few-shot 评测（HellaSwag、MMLU 等）
* **训练动态监控**：梯度范数、激活值分布、权重矩阵的奇异值监控（防止坍塌）
* **Scaling Law 验证**：损失曲线与模型规模/数据规模的幂律关系验证
{% endstep %}

{% step %}
### **工程化与容错**

* **断点续训（Checkpointing Strategy）**：高频 checkpoint 与异步保存策略
* **故障自动恢复**：节点掉线时的弹性训练（Elastic Training）
* **数据加载优化**：Megatron-LM/DeepSpeed 中的数据并行加载与预取
{% endstep %}
{% endstepper %}



### 参考

1. [https://zhuanlan.zhihu.com/p/718354385](https://zhuanlan.zhihu.com/p/718354385)

