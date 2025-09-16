# 分布式训练并行

LLM的参数量已达到前所未有的级别。这种指数级的增长使得传统的单机单卡训练模式在计算能力和显存容量上捉襟见肘，难以满足日益增长的模型训练需求。因此，单机多卡和多机多卡的分布式训练成为日常。

分布式训练的核心在于如何有效地将计算任务和模型数据分配到多个计算设备上。使用的并行策略主要可以分为以下几类：

* [data-parallelism.md](data-parallelism.md "mention") (Data Parallelism)：通过将数据集划分为多个子集，让每个设备独立处理一部分数据，同时保持模型副本的完整性。
* [model-parallelism.md](model-parallelism.md "mention") (Model Parallelism)：当模型本身过于庞大，无法容纳于单张显卡时，可以将模型拆分到多个设备上，包括张量并行 (Tensor Parallelism)（层内并行）和流水线并行 (Pipeline Parallelism)（层间并行）。
* [optimizer-parallelism.md](optimizer-parallelism.md "mention") (Optimizer Parallelism)：针对优化器状态、梯度和模型参数等占用大量显存的部分进行分片，如 ZeRO 优化器系列。
* [heterogeneous-system-parallelism.md](heterogeneous-system-parallelism.md "mention") (Heterogeneous System Parallelism)：利用 CPU 内存甚至 NVMe 磁盘的巨大容量，将模型的部分数据卸载，以容纳更大的模型。
* [multi-dimensional-hybrid-parallelism.md](multi-dimensional-hybrid-parallelism.md "mention") (Multi-dimensional Hybrid Parallelism)：将数据并行、模型并行（张量并行、流水线并行）等多种策略融合，以应对超大规模模型的训练挑战。
* [auto-parallelism.md](auto-parallelism.md "mention")(Auto Parallelism)：旨在自动化模型切分和并行策略的选择，降低开发者的使用门槛。
* [expert-parallelism.md](../../moe/expert-parallelism.md "mention") (Mixture-of-Experts Parallelism)：针对稀疏激活的模型架构，将计算任务分配给不同的“专家”模型。

