# 评估与工程化

* **中间评估（Mid-training Eval）**：PPL 之外，阶段性的下游任务 zero-shot/few-shot 评测（HellaSwag、MMLU 等）
* **训练动态监控**：梯度范数、激活值分布、权重矩阵的奇异值监控（防止坍塌）
* **Scaling Law 验证**：损失曲线与模型规模/数据规模的幂律关系验证
* **断点续训（Checkpointing Strategy）**：高频 checkpoint 与异步保存策略
* **故障自动恢复**：节点掉线时的弹性训练（Elastic Training）
* **数据加载优化**：Megatron-LM/DeepSpeed 中的数据并行加载与预取
