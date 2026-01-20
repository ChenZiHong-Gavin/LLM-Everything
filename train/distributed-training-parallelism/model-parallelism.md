# 模型并行

在数据并行训练中，一个明显的特点是每个 GPU 持有整个模型权重的副本。这就带来了冗余问题，因为最终你只需要一个更新后的模型权重。因此可以使用模型并行的训练方式，即模型被分割并分布在一个设备阵列上。

通常有两种类型的模型并行：张量并行和流水线并行。

* 张量并行是在一个操作中进行并行计算，如矩阵乘法。将单层内的张量操作拆分到多个 GPU 上，减少单卡内存压力。
* 流水线并行是在各层之间进行并行计算。将模型的不同层分布在不同设备上，通过 micro-batching 和交错调度减少 pipeline bubble（设备空闲时间）。

因此，从另一个角度来看，张量并行可以被看作是层内并行，流水线并行可以被看作是层间并行。

<figure><img src="../../.gitbook/assets/image (38).png" alt=""><figcaption></figcaption></figure>

## 1 张量并行

张量并行的数学本质是对矩阵乘法 $$Y=XA$$ 进行分块计算。主要分为两种基础切分方式：

## 1.1 行并行和列并行

### 1.1.1 行并行 (Row Parallelism)

* 切分逻辑：
  * 权重 $$A$$ 按**行**切分 ( $$A_1, A_2$$)。
  * 输入 $$X$$ 需按**列**切分 ( $$X_1, X_2$$) 以匹配维度。
* 计算过程：
  * GPU0: $$Y_1 = X_1 A_1$$
  * GPU1: $$Y_2 = X_2 A_2$$
  * 最终结果： $$Y = Y_1 + Y_2$$ (需要 **All-Reduce** 求和)

### 1.1.2 列并行 (Column Parallelism)

* 切分逻辑：
  * 权重 $$A$$ 按**列**切分 ( $$A_1, A_2$$)。
  * 输入 $$X$$ 在所有 GPU 上保持一致（复制/广播）。
* 计算过程：
  * GPU0: $$Y_1 = X A_1$$
  * GPU1: $$Y_2 = X A_2$$
  * 最终结果： $$Y = [Y_1, Y_2]$$ (拼接/Concat，需要 **All-Gather**)

## 2 张量并行的实现

### 2.1 1D 张量并行

1D的意思是张量全部按照某一维度进行划分（横切或者竖切）。1D张量并行是目前 Transformer 架构大模型最主流的方案，由 Megatron-LM 提出。

以一个线性层为例：

考虑一个线性层的 GEMM 运算：

$$
Y = XA
$$

给定2个处理器，我们将权重矩阵 $$A$$ 按**列**划分为：

$$
A = [A_1 \ A_2]
$$

在每个处理器上独立计算：

$$
Y_i = XA_i
$$

最终得到：

$$
[Y_1 \ Y_2] = [XA_1 \ XA_2]
$$

这种划分方式称为**列并行方式**。

当第二个线性层 $$Z = YB$$ 接在上述列并行层之后时，我们需要将 $$B$$ 按**行**划分：

$$
B = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}
$$

为计算：

$$
Z = [Y_1 \ Y_2] \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}
$$

计算过程分为两步（使用 stepper 表示顺序流程）：

{% stepper %}
{% step %}
### 在每个处理器上本地计算

计算本地乘积：$$Y_i B_i$$
{% endstep %}

{% step %}
### 使用 all-reduce 汇总结果

得到：

$$
Z = Y_1B_1 + Y_2B_2
$$
{% endstep %}
{% endstepper %}

在后向计算中，列并行线性层需要聚合输入张量 $$X$$ 的梯度。原因在于：

在每个处理器 $$i$$ 上，我们仅持有部分输出梯度 $$\dot{Y}_i$$，因此本地计算得到：

$$
\dot{X}_i = \dot{Y}_i A_i^T
$$

为获得完整的输入梯度，必须在各处理器间执行 **all-reduce** 操作：

$$
\dot{X} = \dot{Y}A^T = \dot{Y}_1 A_1^T + \dot{Y}_2 A_2^T
$$

***

### 2.2 2D 张量并行

1D张量并行没有对 **activations** 进行划分，就大规模模型而言，这也会消耗大量的内存。为了平均分配计算和内存负荷，在 SUMMA（可扩展的通用矩阵乘法算法）的基础上，**2D张量并行**被引入。

还是以线性层为例：

$$
Y = XA
$$

给定 **P = q × q** 个处理器（必要条件），如 **q = 2**，我们把输入 **X** 和权重 **A** 都划分为：

$$
\begin{bmatrix}
X_{00} & X_{01} \\
X_{10} & X_{11}
\end{bmatrix}
\quad \text{and} \quad
\begin{bmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{bmatrix}
$$

该计算包括 q 步（使用 stepper 展示每个时刻的操作）：

{% stepper %}
{% step %}
### 第1步（t = 1）

* $$X_{i0}$$ 在其行中被广播
* $$A_{0j}$$ 在其列中被广播

得到：

$$
\begin{bmatrix}
X_{00}, A_{00} & X_{00}, A_{01} \\
X_{10}, A_{00} & X_{10}, A_{01}
\end{bmatrix}
$$

在每个处理器 $$(i, j)$$ 上将 $$X_{i0}$$ 和 $$A_{0j}$$ 相乘为：

$$
\begin{bmatrix}
X_{00}A_{00} & X_{00}A_{01} \\
X_{10}A_{00} & X_{10}A_{01}
\end{bmatrix}^{(1)}
$$
{% endstep %}

{% step %}
### 第2步（t = 2）

* $$X_{i1}$$ 在其行中被广播
* $$A_{1j}$$ 在其列中被广播

得到：

$$
\begin{bmatrix}
X_{01}A_{10} & X_{01}A_{11} \\
X_{11}A_{10} & X_{11}A_{11}
\end{bmatrix}^{(2)}
$$
{% endstep %}
{% endstepper %}

通过将第1步和第2步的结果相加，我们得到：

$$
Y = XA =
\begin{bmatrix}
X_{00}A_{00} + X_{01}A_{10} & X_{00}A_{01} + X_{01}A_{11} \\
X_{10}A_{00} + X_{11}A_{10} & X_{10}A_{01} + X_{11}A_{11}
\end{bmatrix}
$$

### 2.3 2.5D 张量并行

与一维张量并行相比，二维并行降低了内存成本，但可能引入更多的通信。因此，**2.5D张量并行**在2.5D SUMMA的基础上被提出，它通过使用更多的设备来减少通信开销。

以线性层为例：

$$Y = XA$$

给定 **P = q × q × d** 个处理器（必要条件），如 **q = d = 2**：

我们把输入 **X** 划分为 **d × q** 行和 **q** 列：

$$
\begin{bmatrix}
X_{00} & X_{01} \\
X_{10} & X_{11} \\
X_{20} & X_{21} \\
X_{30} & X_{31}
\end{bmatrix}
$$

它可以被重塑为 **d** 层：

$$
\begin{bmatrix}
X_{00} & X_{01} \\
X_{10} & X_{11}
\end{bmatrix}
\quad \text{and} \quad
\begin{bmatrix}
X_{20} & X_{21} \\
X_{30} & X_{31}
\end{bmatrix}
$$

权重 **A** 被分割为：

$$
\begin{bmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{bmatrix}
$$

对于 **X** 相关的每一层，我们使用 SUMMA 算法将 **X** 与 **A** 相乘。然后，我们得到输出：

* 第1层输出：

$$
\begin{bmatrix}
Y_{00} = X_{00}A_{00} + X_{01}A_{10} & Y_{01} = X_{00}A_{01} + X_{01}A_{11} \\
Y_{10} = X_{10}A_{00} + X_{11}A_{10} & Y_{11} = X_{10}A_{01} + X_{11}A_{11}
\end{bmatrix}
$$

* 第2层输出：

$$
\begin{bmatrix}
Y_{20} = X_{20}A_{00} + X_{21}A_{10} & Y_{21} = X_{20}A_{01} + X_{21}A_{11} \\
Y_{30} = X_{30}A_{00} + X_{31}A_{10} & Y_{31} = X_{30}A_{01} + X_{31}A_{11}
\end{bmatrix}
$$

***

### 2.4 3D 张量并行

**3D张量并行**是一种将神经网络模型的计算并行化，以期望获得最佳通信成本优化的方法。

以线性层为例：

$$
Y = XA
$$

给定 **P = q × q × q** 个处理器（必要条件），如 **q = 2**，我们把输入 **X** 和权重 **A** 划分为：

$$
\begin{bmatrix}
X_{000} & X_{001} \\
X_{010} & X_{011} \\
X_{100} & X_{101} \\
X_{110} & X_{111}
\end{bmatrix}
\quad \text{and} \quad
\begin{bmatrix}
A_{000} & A_{001} \\
A_{010} & A_{011} \\
A_{100} & A_{101} \\
A_{110} & A_{111}
\end{bmatrix}
$$

其中每个 $$X_{ijl}$$ 和 $$A_{lji}$$ 都被存储在处理器 $$(i,j,l)$$ 上，如下图所示：

<figure><img src="../../.gitbook/assets/image (39).png" alt=""><figcaption></figcaption></figure>

前向传播时的流程可以用 stepper 表示：

{% stepper %}
{% step %}
### 数据收集阶段

* 在 $$(i, 0...q, l)$$ 上收集 $$X_{ijl}$$，得到 $$X_{il}$$
* 在 $$(0...q, j, l)$$ 上收集 $$A_{lji}$$，得到 $$A_{lj}$$
{% endstep %}

{% step %}
### 本地计算

在每个处理器 $$(i, j, l)$$ 上计算乘积 $$X_{il}A_{lj}$$
{% endstep %}

{% step %}
### 结果归约

在 $$(i, j, 0...q)$$ 对结果进行 **reduce-scatter** 得到 $$Y_{ijl}$$，最终形成：

$$
Y =
\begin{bmatrix}
Y_{000} & Y_{001} \\
Y_{010} & Y_{011} \\
Y_{100} & Y_{101} \\
Y_{110} & Y_{111}
\end{bmatrix}
$$
{% endstep %}
{% endstepper %}

反向传播时：

* 需要 **all-gather** 梯度 $$\dot{Y_{ijl}}$$
* 然后 **reduce-scatter** 梯度：
  * $$X$$ 的梯度：$$\dot{X_{il}} = \dot{Y_{ij}} A_{lj}^T$$
  * $$A$$ 的梯度：$$\dot{A_{lj}} = X_{il}^T \dot{Y_{ij}}$$

## 3 流水线并行

流水线并行的核心思想是，模型按层分割成若干块，每块都交给一个设备。

* 在前向传播过程中，每个设备将中间的激活传递给下一个阶段。
* 在后向传播过程中，每个设备将输入张量的梯度传回给前一个流水线阶段。

这允许设备同时进行计算，从而增加训练的吞吐量。

<figure><img src="../../.gitbook/assets/image (40).png" alt=""><figcaption></figcaption></figure>

## 4 流水线并行的实现

### 4.1 朴素流水线并行

**原理：**

假设我们有 4 个 GPU（GPU0 - GPU3），模型也被切分为 4 段。下面使用 stepper 展示数据在各 GPU 间的流动：

{% stepper %}
{% step %}
### 步骤 1

GPU0 接收一个完整的 Batch 数据，计算完第一部分层，将输出传给 GPU1。
{% endstep %}

{% step %}
### 步骤 2

GPU1 接收数据，计算第二部分，传给 GPU2。
{% endstep %}

{% step %}
### 步骤 3

以此类推，直到 GPU3 完成前向传播，计算 Loss，然后开始反向传播，数据流反向回传。
{% endstep %}
{% endstepper %}

<figure><img src="../../.gitbook/assets/image (42).png" alt=""><figcaption></figcaption></figure>

流水线并行训练的一个明显缺点是训练设备容易出现空闲状态（因为后一个阶段需要等待前一个阶段执行完毕），导致计算资源的浪费，加速效率没有数据并行高。

<figure><img src="../../.gitbook/assets/image (43).png" alt=""><figcaption></figcaption></figure>

### 4.2 微批次流水线并行

为了解决“朴素并行”的空泡问题，引入了微批次（Micro-batch）的概念。

* 原理： 将一个大的全局 Batch（Global Batch）切分成多个小的 Micro-batch。
  * 例如：Global Batch = 1024，切分成 4 个 Micro-batch，每个大小 256。
* 作用： GPU0 处理完 Micro-batch 1 后，立刻发送给 GPU1，紧接着 GPU0 可以马上处理 Micro-batch 2，而不需要等待。
  * 这样，流水线就像工厂的传送带一样流动了起来，多个 GPU 可以同时工作。

基于“微批次”思想，可以设计不同的“调度策略”来平衡显存占用和计算效率：

| 特性          | GPipe           | PipeDream Flush (1F1B)           | PipeDream (Original) | PipeDream-2BW |
| ----------- | --------------- | -------------------------------- | -------------------- | ------------- |
| 调度模式        | F … F → B … B   | 稳定期交替 1F → 1B                    | 持续循环，无 Flush         | 限制版本的持续循环     |
| 显存占用 (激活值)  | 极高 (存 M 份)      | 低 (存 N 份，及时释放)                   | 低                    | 低             |
| 显存占用 (模型权重) | 低 (1 份)         | 低 (1 份)                          | 高 (多版本 stashing)     | 中 (固定 2 份)    |
| 权重更新语义      | 同步 SGD (准确)     | 同步 SGD (准确)                      | 异步 SGD (有误差)         | 异步 SGD (有误差)  |
| 流水线空泡       | 较大 (取决于 M/N 比例) | 与 GPipe 相同 (但在同显存下可跑更大 M，从而稀释空泡) | 极小 / 无               | 极小            |
| 适用场景        | 教学、小模型          | 当前大模型训练主流 (LLaMA, GPT-3)         | 研究性质，追求极致吞吐          | 显存受限的异步训练     |

## 参考

1. [(3 封私信 / 1 条消息) 大模型分布式训练并行技术（四）-张量并行 - 知乎](https://zhuanlan.zhihu.com/p/657921100)
2. [(3 封私信) 大模型分布式训练并行技术（三）-流水线并行 - 知乎](https://zhuanlan.zhihu.com/p/653860567)
3. [1D 张量并行 | Colossal-AI](https://colossalai.org/zh-Hans/docs/features/1D_tensor_parallel/)
4. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://deepakn94.github.io/assets/papers/megatron-sc21.pdf)
5. [An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/pdf/2104.05343)
6. [Tesseract: Parallelize the Tensor Parallelism Efficiently](https://arxiv.org/pdf/2105.14500)
7. [2105.14450](https://arxiv.org/pdf/2105.14450)
