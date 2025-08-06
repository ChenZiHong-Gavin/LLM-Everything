# LLM精度问题

FP32，FP16，BF16是LLM中常见的精度类型。

**数字在计算机中是如何存储的？**

一个浮点数的值由符号位（S）、指数位（E）、尾数位（F）共同决定：

$$
Value=(-1)^S \times (1.F)_2 \times 2^{E-127}
$$

* **符号位（sign）**
  * 表示数字正负
* **指数位（Exponent）**
  * 控制数值的**数量级（范围）**，相当于科学计数法中的“10的幂次”。
* **尾数位（Fraction/Mantissa）**
  * 存储数值的**有效数字（精度）**，相当于科学计数法中的“1.23”。

### 1 FP32 （单精度浮点数）

传统深度学习训练（高精度，但计算慢、内存占用高）

<figure><img src="../../.gitbook/assets/image (2) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

* 位分配
  * 1位符号位
  * 8位指数位
  * 23位尾数位
* 范围
  * 最小正数：$$2^{-126} \approx 1.18 \times 10^{-38}$$
  * 最大正数：$$\left(2-2^{-23}\right) \times 2^{127} \approx 3.40 \times 10^{38}$$
  * 精度：约6-9位有效十进制数字
* 使用方式：`torch.float32` 或者 `torch.float`

### 2 FP16 （半精度浮点数）

深度学习领域正呈现出一种从 FP32 转向使用 FP16 的趋势，因为对于神经网络而言，低精度计算似乎并不关键。额外的精度并无实际益处，反而会使运算速度变慢、占用更多内存且降低通信速度。

<figure><img src="../../.gitbook/assets/image (1) (1) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

* 位分配
  * 1位符号位
  * 5位指数位
  * 10位尾数位
* 范围
  * 最小正数：$$2^{-14} \approx 6.10 \times 10^{-5}$$
  * 最大正数：$$\left(2-2^{-10}\right) \times 2^{15} \approx 65,504$$
  * 精度：约3-4位有效十进制数字
* 使用方式：`torch.float16` 或 `torch.half`&#x20;

### 3 BF16 （**Brain Floating Point**）

最初设计的FP16并未考虑深度学习应用需求，其动态范围过于狭窄。而BF16解决了这个问题，提供了与FP32完全相同的动态范围。

<figure><img src="../../.gitbook/assets/image (2) (1) (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

* **位分配**：
  * 符号位：**1位**
  * 指数位：**8位**（与FP32相同，Bias = 127）
  * 尾数位：**7位**
* **范围**
  * 最小正数：$$2^{-126} \approx 1.18 \times 10^{-38}$$ （同FP32）
  * 最大正数：$$\left(2-2^{-7}\right) \times 2^{127} \approx 3.39 \times 10^{38}$$
  * 精度：约2-3位有效十进制数字
* 使用方式：`torch.bfloat16`&#x20;

### 总结

| 属性       | FP32                                   | FP16                               | BF16                                   |
| -------- | -------------------------------------- | ---------------------------------- | -------------------------------------- |
| **总位数**  | 32位                                    | 16位                                | 16位                                    |
| **指数位**  | 8位（范围大）                                | 5位（范围小）                            | 8位（同FP32）                              |
| **尾数位**  | 23位（精度高）                               | 10位（精度低）                           | 7位（精度最低）                               |
| **动态范围** | $$\sim 10^{-38} \text { to } 10^{38}$$ | $$\sim 10^{-8} \text { to } 10^4$$ | $$\sim 10^{-38} \text { to } 10^{38}$$ |
| **典型场景** | 传统训练                                   | 推理、轻量训练                            | 现代大模型训练                                |

<figure><img src="../../.gitbook/assets/image (3).png" alt=""><figcaption></figcaption></figure>

### 参考

1. [FP64, FP32, FP16, BFLOAT16, TF32, and other members of the ZOO | by Grigory Sapunov | Medium](https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407)
2. [LLM大模型之精度问题（FP16，FP32，BF16）详解与实践 - 知乎](https://zhuanlan.zhihu.com/p/657886517)
