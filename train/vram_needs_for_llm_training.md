# LLM训练需要多少显存

## 1 大模型训练RAM构成

首先，精度会对参数所需内存有影响：

* fp32，float point，一个参数需要32bits，即4bytes
* fp16,，一个参数需要 16 bits, 2 bytes
* int8 精度，一个参数需要 8 bits, 1 byte

大模型训练的显存占用主要由下面几部分构成：

* 模型参数
  * P = 参数量 \* 每个参数所需内存
* 梯度
  * 与模型参数类似，G = 参数量 \* 每个梯度参数所需内存
* 优化器状态
  * 不同的优化器所储存的参数量不同。
    * 例如，AdamW 需维护一阶动量（m）和二阶动量（v），因此需要存储两倍的模型参数
* 激活值（前向传播产生的中间张量）
  * A = B \* S \* E \* C
    * B: Batch Size (批处理大小)
    * S: Sequence Length (序列长度，或称上下文长度，即输入和输出的总token数)
    * E: Embedding Dimension (嵌入维度，或模型的隐藏层大小)
    * C: 常数因子，取决于具体的模型架构和实现细节。对于Transformer模型，这个因子通常大于1，因为它需要存储多个层的激活，并且某些操作（如MLP层）会创建维度为 4×E 的中间张量。
* CUDA内核开销
  * CUDA kernel 也会占据一些 RAM，大概 1.3GB 左右

## 2 显存计算示例

以 LLaMA-6B 为例，所需的显存占用如下：

* 模型状态
  * 模型参数 6GB
  * 梯度 6GB
  * 优化器（AdamW）12GB
* 运行时内存
  * CUDA Kernel 1.3 GB
* 激活值
  * 基于参数架构
    * hidden\_size = 4096, intermediate\_size =11008, num\_hidden\_layers = 32, context\_length = 2048
  *   每个实例：

      (4096 +11008) \* 2048 \*32 \* 1byte = 990MB
  * batch size=50时占用 48.3GB

### 参考

1. [https://zhuanlan.zhihu.com/p/716806882](https://zhuanlan.zhihu.com/p/716806882)

