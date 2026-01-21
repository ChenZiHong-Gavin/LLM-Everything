# 优化器并行

随着模型越来越大，单张 GPU 的显存（即使是 80GB 的 A100/H100）通常无法装下完整的模型，所以需要想办法对占显存的地方进行优化。

通常来说，模型训练的过程中，GPU上需要进行存储的参数包括了模型本身的参数、优化器状态、激活函数的输出值、梯度以及一些临时的Buffer。各种数据的占比如下图所示：

<figure><img src="../../.gitbook/assets/image (44).png" alt=""><figcaption></figcaption></figure>

* **模型状态（Model States）：** 包含：
  * **模型参数（Parameters）：** 模型本身的权重。
  * **梯度（Gradients）：** 反向传播计算出的梯度。
  * **优化器状态（Optimizer States）：** 例如 Adam 优化器中的 Momentum（动量）和 Variance（方差）。
* **激活值（Activations）：** 前向传播过程中各层的输出，需要在反向传播时使用。
* **临时缓冲区（Temporary Buffers）：** 算子计算时的临时空间。
* **碎片（Fragmentation）：** 显存分配不连续导致的浪费。

假设我们使用 Adam 优化器训练一个 $$\Phi$$ 参数量的模型：&#x20;

* **模型参数 (FP16):** $$2\Phi$$ bytes&#x20;
* **梯度 (FP16):** $$2\Phi$$ bytes&#x20;
* **优化器状态 (FP32):** Adam 需要存储一份参数副本、Momentum 和 Variance，通常全是 FP32，共 $$4\Phi + 4\Phi + 4\Phi = 12\Phi$$ bytes。

{% hint style="info" %}
**结论：** 模型状态总计约 $$16\Phi$$ bytes。其中，**优化器状态占据了绝大部分（约 75%）**。这是我们需要优化的核心目标。
{% endhint %}

而优化器相关的并行就是一种去除冗余数据的并行方案，目前这种并行最流行的方法是 [ZeRO](https://zhida.zhihu.com/search?content_id=221180389\&content_type=Article\&match_order=1\&q=ZeRO\&zhida_source=entity) **(Zero Redundancy Optimizer)**，由微软 DeepSpeed 团队提出。

针对模型状态的存储优化（去除冗余），ZeRO使用的方法是分片，即每张卡只存 1/N 的模型状态量，这样系统内只维护一份模型状态。ZeRO有三个不同级别，对模型状态进行不同程度的分片：

* ZeRO-1 : 对优化器状态分片（Optimizer States Sharding）
  * **原理：** 将显存占用最大的“优化器状态”切分到 $$N$$ 个 GPU 上。每张卡只负责更新自己分到的那 $$1/N$$ 的参数。
  * **效果：** 显存节省显著，且几乎不增加通信开销。
* ZeRO-2 : 对优化器状态和梯度分片（Optimizer States & Gradients Sharding）
  * **原理：** 在 ZeRO-1 的基础上，进一步把“梯度”也切分了。在反向传播计算梯度时，不同 GPU 计算出的梯度通过 Reduce-Scatter 操作聚合到对应的分片 GPU 上，而不是每张卡都持有完整梯度。
  * **效果：** 进一步降低显存占用。
* ZeRO-3 : 对优化器状态、梯度分片以及模型权重参数分片（Optimizer States & Gradients & Parameters Sharding）
  * **原理：** 连“模型参数”本身都切分了。每张 GPU 上只存放 $$1/N$$ 的模型权重。
    * **前向传播时：** 当需要计算某一层时，持有该层参数的 GPU 通过 **All-Gather** 广播给所有 GPU。大家计算完该层后，立即释放这些参数（除了自己持有的那部分）。
    * **反向传播时：** 同理，需要参数时动态拉取，用完即扔。**效果：** 显存占用极低，显存消耗与 GPU 数量 $$N$$ 成反比。理论上可以训练无限大的模型（只要卡够多）。
    * **代价：** 会增加额外的通信开销（参数需要反复的 All-Gather），通常会比 ZeRO-1/2 慢一些，但在超大模型上是唯一的选择。

<figure><img src="../../.gitbook/assets/unnamed (1).jpg" alt=""><figcaption></figcaption></figure>

| **策略**          | **分片内容**        | **显存优化程度**    | **通信开销**    | **适用场景**       |
| --------------- | --------------- | ------------- | ----------- | -------------- |
| **Standard DP** | 无               | 低 (冗余严重)      | 低           | 小模型            |
| **ZeRO-1**      | 优化器状态           | **中 (性价比高)**  | 低 (基本无增量)   | 中大模型通用首选       |
| **ZeRO-2**      | 优化器状态 + 梯度      | **高**         | 低           | 显存较紧张时         |
| **ZeRO-3**      | 优化器状态 + 梯度 + 权重 | **极高 (线性扩展)** | 中 (增加约 50%) | 超大模型 / 单卡显存不足时 |
