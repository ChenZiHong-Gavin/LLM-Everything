# 专家并行

**核心思想：专家并行 = 专家分布 + 动态路由 + All2All 通信**

### 1 专家并行概述

专家并行的目标是将一个 MoE 层中的众多专家分布到不同的设备上，每个设备负责一部分专家。如果某个设备上的计算需要其他设备的专家，可以通过All2All通信实现。

专家并行的思想来自于论文：GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding，如下图所示。

<figure><img src="../.gitbook/assets/image (7).png" alt=""><figcaption></figcaption></figure>

具体来说，MoE模型通常使用 Gating 模块来决定每个输入数据样本应该由哪些专家来处理。假设有一个输入数据样本位于设备 A 上，而 Gating 模块决定该样本应该由设备 B 和设备 C 上的专家来处理，那么就需要将该数据样本从设备 A 传输到设备 B 和设备 C。

### 2 All2All通信

All2All是**分布式计算、并行计算和高性能计算**中的一种核心通信模式。

> **在一个由 N 个节点组成的群体中，每一个节点都需要向其他&#x20;**_**所有**_**&#x20;N-1 个节点发送一份不同的数据，同时也需要从其他&#x20;**_**所有**_**&#x20;N-1 个节点接收一份不同的数据。**

即“所有人给所有人发消息”。如下图所示，看起来很像一个矩阵的转置操作。

<figure><img src="../.gitbook/assets/image (8).png" alt=""><figcaption></figcaption></figure>

#### 1.1 标准 All2All

标准All2All即均匀发送和接受数据，发送到和接受自不同设备的数据量相同。

可以使用 `torch.distributed` 实现。

```python
import torch
import torch.distributed as dist

def run_standard_all2all():
    rank = dist.get_rank() # rank代表进程
    size = dist.get_world_size() # size代表分布式组中参与进程的总数
    device = torch.device(f"cuda:{rank}")
    # 在每个rank上创建对应的input_tensor
    input_tensor = torch.ones(size, dtype=torch.int32, device=device) * rank
    print(f"Rank {rank} before all2all, input_tensor: {input_tensor.tolist()}")
    
    output_tensor = torch.empty(size, dtype=torch.int32, device=device)
    dist.all_to_all_single(output_tensor, input_tensor)
    print(f"Rank {rank} after all2all, output_tensor: {output_tensor.tolist()}")
    
def main():
    dist.init_process_group(backend='nccl')  # 初始化分布式环境
    run_standard_all2all()
    dist.destroy_process_group()  # 销毁分布式环境
    
if __name__ == "__main__":
    main()
```

假设文件名为 [`test.py`](http://test.py) ，启动命令如下：

```python
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 test.py
```

输出结果如下：

```python
Rank 0 before all2all, input_tensor: [0, 0, 0, 0]
Rank 2 before all2all, input_tensor: [2, 2, 2, 2]
Rank 1 before all2all, input_tensor: [1, 1, 1, 1]
Rank 3 before all2all, input_tensor: [3, 3, 3, 3]
Rank 1 after all2all, output_tensor: [0, 1, 2, 3]
Rank 2 after all2all, output_tensor: [0, 1, 2, 3]
Rank 3 after all2all, output_tensor: [0, 1, 2, 3]
Rank 0 after all2all, output_tensor: [0, 1, 2, 3]
```

#### 1.2 非标准 All2All

实际上有些场景并非均匀发送和接收，有可能发送到不同设备的数据量不同，从不同设备接收的数据量也可能不同。Pytorch 的 `torch.distributed.all_to_all_single` 提供了 `input_split_sizes` 和 `output_split_sizes` 参数来支持：

* `input_split_sizes` 表示向每个设备发送的数据量。
* `output_split_sizes` 表示从每个设备接收的数据量。

假设有4个GPU，每个GPU包含10个数据：

* 4 个 GPU 都向 GPU k 发送 k+1 个数据
  * 即，都向 GPU 0 发送 1 条数据，向 GPU 3 发送 4 条数据
* GPU k 从其余每个 GPU 都接收 k+1 个数据
  * 即，GPU 0 从其余每个 GPU 接收 1 条数据，共接收 3 条

如下图所示：

<figure><img src="../.gitbook/assets/image (9).png" alt=""><figcaption></figcaption></figure>

代码实现如下：

```python
import torch
import torch.distributed as dist

def run_nonstandard_all2all():
    rank = dist.get_rank()
    size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    input_splits = [i+1 for i in range(size)]
    input_tensor = torch.ones(sum(input_splits), dtype=torch.int32, device=device) * rank
    print(f"Rank {rank} before all2all, input_tensor: {input_tensor.tolist()}")

    output_splits = [rank + 1] * size
    output_tensor = torch.empty(sum(output_splits), dtype=torch.int32, device=device)
    dist.all_to_all_single(output_tensor, input_tensor, output_splits, input_splits)
    print(f"Rank {rank} after all2all, output_tensor: {output_tensor.tolist()}")

def main():
    dist.init_process_group(backend='nccl')  # 初始化分布式环境
    run_nonstandard_all2all()
    dist.destroy_process_group()  # 销毁分布式环境

if __name__ == "__main__":
    main()
```

输出如下：

```python
Rank 3 before all2all, input_tensor: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
Rank 1 before all2all, input_tensor: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Rank 2 before all2all, input_tensor: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
Rank 0 before all2all, input_tensor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Rank 2 after all2all, output_tensor: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
Rank 3 after all2all, output_tensor: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
Rank 0 after all2all, output_tensor: [0, 1, 2, 3]
Rank 1 after all2all, output_tensor: [0, 0, 1, 1, 2, 2, 3, 3]
```

#### 1.3 两次 All2All

上述非标准 All2All 中有个问题：有些时候当前设备只知道要向其他设备发送多少数据，而并不知道需要从其他设备接收多少数据。

这个问题可以通过 2 次 all2all 来解决：

* 第一次 all2all 交换要传输的数据量信息，这是一个标准的 all2all 操作。
* 第二次 all2all 根据上述获取的数据量信息来执行真正的数据传输，此时是一个非标准 all2all 操作。

代码如下：

```python
def run_all2al_twice():
    rank = dist.get_rank()
    size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # 第一次 all_to_all
    input_splits = [i + 1 for i in range(size)]
    input_tensor = torch.ones(sum(input_splits), dtype=torch.int32, device=device) * rank
    print(f"Rank {rank} before first all2all, input_tensor: {input_tensor.tolist()}")

    input_splits_pt =  torch.tensor(input_splits, dtype=torch.int32, device=device)
    output_splits_pt = torch.empty(size, dtype=torch.int32, device=device)
    dist.all_to_all_single(output_splits_pt, input_splits_pt)
    output_splits = output_splits_pt.tolist()
    print(f"Rank {rank} after first all2all, output_splits: {output_splits}")

    output_tensor = torch.empty(sum(output_splits), dtype=torch.int32, device=device)
    dist.all_to_all_single(output_tensor, input_tensor, output_splits, input_splits)
    print(f"Rank {rank} after first all2all, output_tensor: {output_tensor.tolist()}")

def main():
    dist.init_process_group(backend='nccl')  # 初始化分布式环境
    run_all2al_twice()
    dist.destroy_process_group()  # 销毁分布式环境

if __name__ == "__main__":
    main()
```

输出如下：

```python
Rank 3 before first all2all, input_tensor: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
Rank 1 before first all2all, input_tensor: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Rank 0 before first all2all, input_tensor: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Rank 2 before first all2all, input_tensor: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
Rank 0 after first all2all, output_splits: [1, 1, 1, 1]
Rank 1 after first all2all, output_splits: [2, 2, 2, 2]
Rank 3 after first all2all, output_splits: [4, 4, 4, 4]
Rank 2 after first all2all, output_splits: [3, 3, 3, 3]
Rank 3 after first all2all, output_tensor: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
Rank 0 after first all2all, output_tensor: [0, 1, 2, 3]
Rank 1 after first all2all, output_tensor: [0, 0, 1, 1, 2, 2, 3, 3]
Rank 2 after first all2all, output_tensor: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
```

### 3 专家并行分片示例

参考DeepSeek的PEFT相关工作： [ESFT/train\_ep.py at main · deepseek-ai/ESFT](https://github.com/deepseek-ai/ESFT/blob/main/train_ep.py)

并行组划分：

```python
def init_parallel_groups(ep_size=1):
    dist.init_process_group("nccl")
    world_size = int(os.getenv("WORLD_SIZE", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    ep_group = edp_group = None
    for i in range(0, world_size, ep_size):
        ranks = list(range(i, i + ep_size))
        group = dist.new_group(ranks)
        if local_rank in ranks:
            ep_group = group
    edp_group = None
    for i in range(ep_size):
        ranks = list(range(i, world_size, ep_size))
        group = dist.new_group(ranks)
        if local_rank in ranks:
            edp_group = group
    dist.all_reduce(torch.zeros(1, device="cuda"), group=ep_group)
    dist.all_reduce(torch.zeros(1, device="cuda"), group=edp_group)
    return world_size, local_rank, ep_group, edp_group
```

1. **`world_size`**：全局 GPU 总数（所有节点）
2. **`local_rank`**：当前 GPU 在节点内的本地编号（0\~N-1）
3. `ep_group` ：专家并行组（Expert Parallelism Group）
4. **`edp_group`**：专家数据并行组（Expert Data Parallelism Group）

**通信组划分逻辑**

假设有 8 个 GPU（world\_size=8），ep\_size=2（每个专家组包含 2 个 GPU）：

*   **专家并行组（ep\_group）划分**

    ```python
    for i in range(0, world_size, ep_size):
        ranks = list(range(i, i + ep_size))
    ```

    * **组0**：GPU \[0, 1] → 共同处理专家A
    * **组1**：GPU \[2, 3] → 共同处理专家B
    * **组2**：GPU \[4, 5] → 共同处理专家C
    * **组3**：GPU \[6, 7] → 共同处理专家D
*   **专家数据并行组（edp\_group）划分**

    ```python
    for i in range(ep_size):
        ranks = list(range(i, world_size, ep_size))
    ```

    * **组0**：GPU \[0, 2, 4, 6] → 专家A的不同数据分片
    * **组1**：GPU \[1, 3, 5, 7] → 专家B的不同数据分片
*   通信验证

    ```python
    dist.all_reduce(torch.zeros(1), group=ep_group)
    dist.all_reduce(torch.zeros(1), group=edp_group)
    ```

    * 测试两个通信组是否正常工作
    * 确保所有 GPU 都能在各自组内通信

### 4 发展

| **时间** | **研究团队**      | **技术名称**            | **主要改进点**               | **适用场景**      | **参考文献**                                       |
| ------ | ------------- | ------------------- | ----------------------- | ------------- | ---------------------------------------------- |
| 2021年  | Google        | Switch Transformers | 更高效的并行方式                | Google的TPU    | [2101.03961](https://arxiv.org/pdf/2101.03961) |
| 2021年  | 清华大学          | FastMoE             | 适用于GPU和PyTorch的实现       | GPU和PyTorch环境 | [2103.13262](https://arxiv.org/pdf/2103.13262) |
| 2022年  | 微软            | Tutel               | 优化了All2All通信性能          | 通信性能优化场景      | [2206.03382](https://arxiv.org/pdf/2206.03382) |
| 2022年  | 斯坦福大学、微软和谷歌联合 | MegaBlocks          | 解决1个GPU上多个专家时负载不均衡导致的问题 | 单GPU多专家场景     | [2211.15841](https://arxiv.org/pdf/2211.15841) |

### 参考

1. [https://blog.csdn.net/xx\_nm98/article/details/142422761](https://blog.csdn.net/xx_nm98/article/details/142422761)
2. [17. 使用Alltoall通讯 — python-parallel-programming-cookbook-cn 1.0 文档](https://python-parallel-programmning-cookbook.readthedocs.io/zh-cn/latest/chapter3/17_Collective_communication_using_Alltoall.html)
3. [2006.16668](https://arxiv.org/pdf/2006.16668)
4. [DeepSeek R1专家并行-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/2499657)
