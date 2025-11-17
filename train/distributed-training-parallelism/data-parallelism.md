# 数据并行

PyTorch现在已经提供了成熟、高效的分布式训练数据并行解决方案，因此本篇就以 PyTorch 的实现来详解数据并行，从DP到DDP再到FSDP。



在PyTorch的分布式训练中，有几个核心概念需要了解：

1. **进程组（Process Group）**：进程组是分布式训练的基本单位，由一组协同工作的进程组成。每个进程都可以通过进程组进行通信和同步操作。进程组负责协调各个计算节点的工作，确保训练过程的顺利进行。
2. **后端（Backend）**：后端是实现进程组通信的具体方法。PyTorch提供了多种后端选择，如TCP、Gloo和MPI等，以适应不同的分布式环境和需求。选择合适的后端可以优化通信效率，提高训练速度。



### 1 DP (Data Parallel)

DP 是 PyTorch 最早提供的数据并行方案，旨在使用单个进程在单台机器的多个 GPU 上实现数据并行训练。

#### 1.1 工作机制

1. **数据分割（Scatter）**：将一个 **batch** 的数据平均分割（Scatter）到各个 GPU。
2. **模型复制（Replication）**：将模型复制（Replication）到每个 GPU 上。
3. **独立计算**：每个 GPU **独立**计算前向传播和损失。
4. **梯度聚合与同步（Gather & Update）**：
   * 在 **主 GPU**（通常是 **GPU 0**）上收集（Gather）所有 GPU 计算出的损失并计算梯度。
   * 在 **主 GPU** 上更新模型参数。
   * 在下一次前向传播之前，将最新的模型参数再次复制到所有其他 GPU。

#### 1.2 局限性

虽然 DP 使用简单，但它存在几个显著的缺点，使其不适合大规模或高性能训练：

* **GPU 0 负载不均**：所有的梯度计算、参数更新和模型同步都集中在 **GPU 0** 上。这导致 GPU 0 成为**瓶颈**，使得其他 GPU 经常处于等待状态，造成 GPU 利用率不均衡。
* **单进程限制**：DP 只使用了**一个 Python 进程**。由于 Python 的 **GIL (Global Interpreter Lock)** 限制，无法充分利用多核 CPU 的并行优势，且不能进行真正的**多机分布式训练**。
* **模型同步开销**：每次迭代都需要将完整的模型参数或梯度在 GPU 0 和其他 GPU 之间传输，通信开销较大。

因此，PyTorch **官方已不推荐**在生产环境中使用 DP，并建议使用 **DDP**。

### 2 DDP (Distributed Data Parallel)

DDP 是 PyTorch 官方推荐的成熟、高效的数据并行方案，用于单机多卡或多机多卡的分布式训练。它解决了 DP 的所有主要问题。

<figure><img src="../../.gitbook/assets/image (37).png" alt=""><figcaption></figcaption></figure>

#### 2.1 DP VS DDP

| **特性**     | **DP**                    | **DDP**               | 优点                                   |
| ---------- | ------------------------- | --------------------- | ------------------------------------ |
| **进程**     | 单进程，多线程                   | 多进程（每个 GPU 独立一个进程）    | 避免 **GIL 瓶颈**，充分利用多核 CPU，支持**多机训练**。 |
| **通信机制**   | Gather/Scatter（集中在 GPU 0） | **All-Reduce**（点对点通信） | **负载均衡**，没有主GPU瓶颈，通信效率高。             |
| **梯度同步时机** | 反向传播结束后，在 GPU 0 集中计算梯度    | **梯度计算过程中**           | 减少等待时间。                              |
| **模型存储**   | GPU 0 存储原始模型，其他 GPU 存副本   | 每个进程存储一个**完整的模型副本**   | **内存消耗均衡**，更新独立且一致。                  |

当用户用 `DDP(model)` 封装模型时，DDP 会在模型的 **Hooks** 上进行操作，从而控制反向传播的行为，**`_rebuild_parameters`** 方法在 DDP 初始化时，检查模型的所有参数，并为每个参数注册一个 **`grad_acc_hook`**。这是 DDP 区别于 DP 的核心优势。DDP 不等待所有梯度计算完毕再进行一次大规模的同步，而是利用 **All-Reduce** 算法，并在**反向传播过程中**实现**计算与通信的重叠。当反向传播开始时，PyTorch 的自动微分引擎 Autograd 从输出层向输入层逐步计算梯度。一旦一个参数的梯度被计算出来，它所注册的 `grad_acc_hook` 就会被触发，DDP 会检查这个梯度是否属于当前正在被收集的梯度块 (Bucket)**。**如果**一个完整的梯度块中的所有梯度都已计算完毕，DDP 会立即在这个块上异步地启动一个 **All-Reduce** 通信操作。

当一个进程正在对一个梯度块执行 All-Reduce 时，Autograd 引擎可以同时继续计算模型中更深层次的参数的梯度**计算**。

通过将通信操作分解并与计算并行执行，DDP 大幅减少了通信等待时间，显著提高了 GPU 利用率。

### 3 FSDP (Fully Sharded Data Parallel)

尽管 DDP 解决了计算和通信的效率问题，但它仍然存在一个根本的限制：

> 内存冗余：在 DDP 中，每个 GPU 都需要存储完整的 PGO (Parameters, Gradients, Optimizer States) 副本。

随着模型规模（如 **GPT-3**、**LLaMA** 等大型语言模型）的不断增大，模型参数、优化器状态和激活的显存占用可能**超过单个 GPU 甚至整个机器的显存容量**。在这种情况下，传统的数据并行（DDP）就无能为力了。

**FSDP** (Fully Sharded Data Parallel)，是 PyTorch **1.11** 版本后引入的**高效**且**可扩展**的并行策略，它基于 **ZeRO (Zero Redundancy Optimizer)** 思想。

它将模型的 PGO 沿维度切片，每个 GPU 只负责存储和管理它被分配到的那一部分。

#### 3.1 DDP VS FSDP

| **内存项**                      | **DDP 存储方式**               | **FSDP 存储方式 (Sharding Level)**         |
| ---------------------------- | -------------------------- | -------------------------------------- |
| **P**arameters (参数)          | 每个 GPU 存储 $$\text{100\%}$$ | 每个 GPU 只存储 $$\text{1/N}$$ (N 为 GPU 总数) |
| **G**radients (梯度)           | 每个 GPU 存储 $$\text{100\%}$$ | 每个 GPU 只存储 $$\text{1/N}$$              |
| **O**ptimizer States (优化器状态) | 每个 GPU 存储 $$\text{100\%}$$ | 每个 GPU 只存储 $$\text{1/N}$$              |

在 PyTorch 中，FSDP 主要通过 `torch.distributed.fsdp.FullyShardedDataParallel` 类实现。

#### 3.2 FSDP过程

**前向传播**

FSDP 在前向传播中引入了 **All-Gather** 操作：

1. **分片参数**：初始状态下，每个 GPU 只拥有它负责的**参数分片**（$$\text{Param}_\text{shard}$$）。
2. **All-Gather**：当前向传播需要某个层的参数时，FSDP 会自动触发一个 **All-Gather** 操作。
   * 所有的 GPU 进程会将它们持有的该层参数分片汇集起来，**临时**重建出该层的**完整参数**（$$\text{Param}_\text{full}$$）。
   * $$\text{Param}_\text{full}$$仅在该层计算期间存在于每个 GPU 的内存中。
3. **计算**：使用完整的参数进行前向计算。
4. **释放内存 (Shard & Free)**：计算完成后，完整参数会被**立即释放**，GPU 内存只保留其分片。

> 这个 Fetch-Compute-Free 策略是 FSDP 节省显存的关键：完整参数不会在整个训练过程中持续存在。

**反向传播**

反向传播中，FSDP 类似 DDP，实现了计算与通信的重叠，但使用 Reduce-Scatter 操作进行梯度同步。

1. **局部梯度计算**：当 Autograd 计算出某层参数的梯度时，由于每个 GPU 仅持有该层的**分片参数**，因此它也只会计算出**分片梯度**（$$\text{Grad}_\text{shard}$$）。
2. **Reduce-Scatter**：FSDP 会触发一个 **Reduce-Scatter** 操作。
   * 这个操作将所有 GPU 上的**分片梯度**（$$\text{Grad}_\text{shard}$$）进行求和（Reduce），然后将求和后的完整梯度**再次分片**（Scatter），发送回各个 GPU。
   * 最终，每个 GPU 进程会收到它负责的那个**已同步的梯度分片**。



### 参考

1. [大模型分布式训练并行技术（二）-数据并行 - 知乎](https://zhuanlan.zhihu.com/p/650002268)
2. [PyTorch 分布式训练：从历史到概述-百度开发者中心](https://developer.baidu.com/article/details/3272751)
