# RLVR

### 1 什么是 Verifiable Reward

一个奖励函数 $$r(q, o)$$ 如果可以由一段确定性程序在有限时间内判定，就称为 verifiable reward。最常见的形式是二值的:

$$r(q, o) = \mathbb{1}\{\mathrm{verify}(q, o) = \texttt{pass}\} \in {0, 1}$$

其中，`1{...}` 是**指示函数,**&#x547D;题为真返回 1,为假返回 0:

典型实例:

| 类型     | 验证器                    |
| ------ | ---------------------- |
| 数学最终答案 | 字符串/数值/符号等价匹配          |
| 代码     | 解释器 + 单元测试             |
| 形式化证明  | Lean / Coq 内核          |
| 结构约束   | JSON Schema、正则、长度、格式标签 |
| 游戏/规划  | 环境模拟器的终止状态             |

它的三个关键性质:

* **可判定**:有固定的标准。
* **廉价且可并行**
* **相对抗操纵**: “相对”不那么容易被 hack

**可验证性是关于"评价"的性质,不是关于"任务"的性质**。同一个任务，换一种输出格式，可验证性会完全不同。

### 2. 为什么数学和代码最适合 RLVR

三个原因：

1. **生成难、验证易的不对称性。** 这本质上是 NP 式的结构：找到一个 15 行的正确实现很难，运行 20 个单元测试很容易。RL 需要海量的 $$(o, r)$$ 对，只有当验证成本远低于生成成本时，这个循环才在经济上成立。
2. **答案空间可归一化。** 数学题的最终答案能被投影到一个小的、可比较的空间(例如一个整数、一个分式、一个表达式)。这让"对/错"这个判断不需要理解中间推理。
3. **语义由执行定义。** 代码尤其特殊:它的正确性不是由人的判断定义的，而是由解释器定义的。这意味着奖励几乎没有标注噪声，也就是不存在主观性。

反过来说，这三个理由也精确地指出了 RLVR 在别处会遇到什么困难：开放式写作的输出空间不可归一化，长程 agent 任务的验证成本可能高于生成成本，而"这个回答有帮助吗"没有执行语义。

但是实践并没有想象中那么简单，中数学 RLVR 的验证器有大量**假阴性**(例如`1/2` vs `0.5` vs $$\frac{1}{2}$$； $$2\sqrt{3}$$ vs  $$\sqrt{12}$$ )和**假阳性**(多选题猜中、数值巧合、模型输出里恰好包含答案子串)。一个粗糙的 `boxed` 正则匹配，假阴率能到百分之几——这些噪声会被后面讲的方差放大机制放大。

### 3. RLHF、RLAIF、RLVR 的区别

|        | RLHF                              | RLAIF                     | RLVR            |
| ------ | --------------------------------- | ------------------------- | --------------- |
| 奖励来源   | 人类偏好数据训练的 RM                      | 更强 LLM 的判断 / constitution | 确定性程序           |
| 信号形式   | 标量(有序但无绝对尺度)                      | 标量或偏好                     | 二值(或分档)         |
| 标注成本   | 高,随任务线性增长                         | 中,可扩展                     | 近乎没有            |
| 主要失效模式 | RM over-optimization(奖励模型被外推到分布外) | judge 的风格偏好、自我偏爱、谄媚       | 验证器漏洞、格式钻空、答案泄漏 |
| 优化上限   | RM 的准确性                           | judge 的能力                 | 验证器的覆盖度         |
| 适用任务   | 主观、开放                             | 主观 + 半客观                  | 客观、可判定          |

RLHF、RLAIF、RLVR 三者的**差别不在算法，真正的区别在于谁来当裁判**。

* **RLHF：**&#x88C1;判是人。但人打分又慢又贵，所以实际做法是先请人打少量一批分数，再根据这批数据训一个小的 Reward Model 代替人打分。
* **RLAIF：**&#x88C1;判是另一个 LLM，它可以根据给定评分标准来打分。
* **RLVR：**&#x88C1;判是一段程序。例如代码问题就跑自动化测试，数学问题就和答案比较。

GRPO/PPO 都可以配任意一种奖励。RLVR 的独特优势是裁判判断的**保真度的上限是 1.0，**&#x800C;不是 RM 的 0.7\~0.9；它的代价是裁判的**覆盖率极低**——世界上大部分值得做的任务都不在它的定义域里。

还有一个常被忽略的差别：RLHF 的奖励通常是**稠密相对**的(任何两个回答都能比出高低)，而 RLVR 的奖励是**稀疏绝对**的(要么 0 要么 1)。这直接导致了后面所有关于组内方差、零梯度、采样效率的问题。

### 4. Outcome Reward 与 Process Reward

**Outcome Reward (ORM)：**&#x53EA;在序列末尾给一个信号，$$r$$ 只依赖最终答案。&#x20;

**Process Reward (PRM)：**&#x7ED9;推理链的每一步  $$s_i$$ 一个信号 $$r_i$$。

差别的本质是 **credit assignment(信用分配)**。一条 500 token 的推理链只拿到一个 bit 的反馈，意味着策略梯度必须把这 1 bit 均摊到 500 个 token 上——第 300 步的关键错误和第 5 步无关紧要的措辞得到完全相同的惩罚。这是纯粹的统计效率问题，需要更多样本才能把有效信号从噪声里分离出来。

#### 4.1 结果奖励真的改善了推理过程吗?

这是本文最想认真回答的问题。答案是:**它可靠地改善了拿到正确答案的策略，只在推理是拿到答案的必要手段时才顺带改善推理**。也就是说，改善推理过程本身不是结果奖励的直接目的。

1. **分布锐化 vs 能力扩展。** 大量复现实验观察到这样的模式：RLVR 之后 pass@1 显著上升,而 pass@k(k 较大,比如 256)持平甚至下降。对于这个现象的一种假设是：RL 把 base model 本来就能采样到的正确解的概率质量往上抬，同时压缩了熵——它重新分配了概率，但没有创造新的解。也就是说，**GRPO 的更新方向永远是“把已经成功过的东西做得更确定”，它结构上无法奖励一个从未成功过的行为**。
2. **答案对但过程错。** 结果奖励对过程的正误毫不在意。多选题、有限答案空间的数值题、以及模型在错误推理后凭直觉写出正确答案的情况，都会拿到 $$r=1$$。ProcessBench 一类的评测显示，即使是很强的模型也难以定位一条推理链里第一个出错的步骤——这说明"过程正确性"和"答案正确性"在训练信号层面被系统性地混同了。更糟的是,有研究观察到 outcome-only RL 会**降低**推理链的忠实度:模型学会写出一段看起来像推理的文本,同时用另一条内部捷径得到答案。
3. **Spurious reward。** 对某些 base model(尤其 Qwen 系列)，即使用随机奖励或错误奖励做 RL，某些数学 benchmark 上的分数也会上升。这几乎只能解释为 RL 在**唤醒**预训练里已有的格式与行为模式(输出 `\boxed{}`、使用代码式推理)，而不是在**学习**推理。

**结论**结果奖励是一个关于**输出端点**的约束。只要是存在捷径的地方，它就会奖励捷径；只有推理是唯一路径的地方,它才奖励推理。所以"能不能靠结果奖励得到好的过程"这个问题，等价于"当前任务是否不存在任何捷径"。

### 5. GRPO 在 RLVR 中扮演什么角色

#### 5.1 算法本体

GRPO 相对 PPO 的唯一实质改动：**用组内蒙特卡洛统计量替代 critic**。对每个 prompt $$q$$ 采 $$G$$ 条回答 $${o_1,\dots,o_G}$$，优势为 z-score 后的奖励:

$$A(q, o_i) = \frac{r(q,o_i) - \mu(q)}{\sigma(q)}, \quad \mu = \frac{1}{G}\sum_j r_j,\ \ \sigma = \mathrm{std}(r_{1..G})$$

目标函数是 PPO 式的裁剪代理目标,加上对参考策略的 KL 惩罚。

省掉 critic 带来两个好处：显存减半、以及不需要处理"critic 在长序列上估不准价值"这个问题。

#### 5.2 为什么能放大成功概率

设 $$p(q)$$ 为旧策略在问题 $$q$$ 上的成功概率 (PoS, probability of success)。奖励是伯努利的，所以 $$\mu = p$$ ，$$\sigma^2 = p(1-p)$$。代入 z-score 公式:

$$A(q,o) = \begin{cases} \dfrac{1-p}{\sqrt{p(1-p)}} = \sqrt{\dfrac{1-p}{p}} & r = 1 \[2ex] -\dfrac{p}{\sqrt{p(1-p)}} = -\sqrt{\dfrac{p}{1-p}} & r = 0\end{cases}$$

这个公式说明：**GRPO 不是在优化一个固定的奖励，而是在优化一个被"当前难度"重新标定过的奖励**。看具体数值:

| p    | 成功样本权重 $\sqrt{(1-p)/p}$ | 失败样本权重 $-\sqrt{p/(1-p)}$ |
| ---- | ----------------------- | ------------------------ |
| 0.01 | **+9.95**               | −0.10                    |
| 0.10 | +3.00                   | −0.33                    |
| 0.25 | +1.73                   | −0.58                    |
| 0.50 | +1.00                   | −1.00                    |
| 0.75 | +0.58                   | −1.73                    |
| 0.90 | +0.33                   | −3.00                    |
| 0.99 | +0.10                   | **−9.95**                |

* **难题上罕见的成功被极度放大**($$p=0.01$$ 时权重接近 10),而失败几乎不被惩罚。
* **简单题上失败被极度惩罚**,成功几乎不给奖励。GRPO 会花大量精力解决那些本该做对却做错的题。
* 展开成损失函数，GRPO 等价于一个**自适应加权的对比损失**:正样本项的权重是 $$\sqrt{(1-p)/p}$$，负样本项的权重是 $$\sqrt{p/(1-p)}$$，而对比样本来自上一轮策略自己生成的合成数据。

#### 5.3 归一化本身引入的偏差

归一化不是免费的。

* **除以** $$\sigma$$ **引入难度偏差。** mean-only 归一化(不除方差)才是无偏的策略梯度；除以 $$\sigma$$  会按难度重新加权问题，系统性地给中等难度题更大的更新幅度。上面的表就是这个偏差的定量形式。
* **除以** $$|o|$$ **引入长度偏差。** 原始 GRPO 按序列长度平均 token loss,这让长的错误回答每个 token 受到的惩罚更小,从而系统性地鼓励"错就错长一点"。DAPO 的 token-level loss(在整个 batch 的 token 上归一化,而非先按序列内平均)修掉了这一项。这是**长度爆炸**最常被忽略的直接来源，很多人会误以为它是模型学会了深度思考。

### 6. 为什么全对和全错的组无法提供有效相对优势

若组内所有 $$r_i$$ 相等(全 0 或全 1),则 $$\mu = r_i$$,于是分子 $$r_i - \mu = 0$$ 对所有 $$i$$ 成立,$$\sigma = 0$$。无论实现是加 $$\varepsilon$$ 平滑还是显式置零，**该组对梯度的贡献严格为 0**。

GRPO 的优势是**组内相对**的,一个没有内部差异的组不包含任何关于什么更好的信息。

定量的来看，一个组退化的概率是：

$$P(\text{degenerate}) = p^G + (1-p)^G$$

| $p$  | $G=8$ | $G=16$ | $G=64$ |
| ---- | ----- | ------ | ------ |
| 0.5  | 0.8%  | 0.003% | \~0    |
| 0.2  | 17%   | 2.8%   | 0.001% |
| 0.05 | 66%   | 44%    | 3.8%   |
| 0.01 | 92%   | 85%    | 53%    |
| 0.95 | 66%   | 44%    | 3.8%   |

两个结论:

1. **有效 batch size 是一个随训练动态变化的隐变量。** 训练初期数据太难 → 大量全错组；训练后期数据被学会 → 大量全对组。两端都会让有效 batch 塌缩。&#x20;
2. **最优数据难度在**  $$p \approx 0.5$$ **附近。** $$p \approx 0.5$$ 时组退化概率最低，且期望梯度幅度 $$\propto \sqrt{p(1-p)}$$ 最大。所以 GRPO 隐含地要求一个把数据维持在 50% 通过率附近的课程机制。

### 7. 如何处理零成功率

**首先，零成功率不是算法能解决的问题**。p=0 意味着策略的支撑集里没有正解，而策略梯度只能重新加权支撑集内的东西。因此需要从**外部**注入一条成功轨迹。下面是一些可能的方法：

1. **诊断。** 先用 pass@k(k=64 或 256)测试每道题的可解性，把数据分成$$p=0$$、$$0<p<1$$、$$p=1$$三桶。前后两桶可以直接丢弃，因为在当前阶段对 GRPO 是纯粹的算力浪费。
2. **采样层面**

* **Dynamic sampling(DAPO)：**&#x8FC7;采样并丢弃退化组，持续补采直到 batch 填满非退化组。代价是每步的实际 rollout 数变多且不定，但有效 batch 恒定。
* **提高** $$G$$**：**&#x628A;截断上限从 $$\sqrt{G-1}$$ 抬高,并把可用难度下界从 $$1/G$$ 推低。
* **Clip-higher**:把上界裁剪 $$\epsilon_{\text{high}}$$ 放宽(比如 0.28),给低概率 token 留出被抬升的空间,抑制熵坍缩。
* **温度与 top-p**:训练采样用高温、评测用低温。

3. **问题层面**

* **课程与难度分桶：**&#x6309;实测 $$\hat p$$ 排课,维持 batch 的平均 $$\hat p$$ 在 0.3\~0.7。
* **提示注入 / 逆向课程**:把参考解的前 $$k%$$ 作为前缀塞进 prompt,让 $$p$$ 从 0 变成正数;随训练逐步减少 $$k$$。
* **难度自适应权重**:对不同子任务按"剩余提升空间"加权,形成隐式课程。

4. **信号层面**

* **分档奖励**:代码任务用**单元测试通过率**而非全通过与否;数学任务给格式分、给中间过程正确性分。
* **过程奖励 / 子目标**
* **蒙特卡洛价值估计**:从推理链的中间状态继续 rollout,用后续成功率作为该步的价值,得到逐步优势(Math-Shepherd、VinePPO、VSRM 一类)。

5. **分布层面**

* **拒绝采样微调 / expert iteration 预热：**&#x5148;用高温大量采样 + 验证器筛选出正解做 SFT，把 $$p$$ 从 0 抬到可 RL 的区间,再上 GRPO。这是最稳的工程路径,几乎所有严肃的 RLVR pipeline 都有这一步。
* **从更强模型蒸馏种子轨迹**:能力上限受教师限制。

### 8. Reward Hacking

#### 9.1 策略攻击验证器

| 攻击    | 表现                                                               | 防御                                                   |
| ----- | ---------------------------------------------------------------- | ---------------------------------------------------- |
| 测试泄漏  | 代码读取 test 文件、`__file__`、环境变量后硬编码答案                               | 测试与代码物理隔离,独立进程、独立目录、不传测试路径                           |
| 断言绕过  | 覆盖 `assert`、monkey-patch `unittest`、`sys.exit(0)`、修改 `pytest` 钩子 | 在受限解释器里跑;检查退出码**与**测试计数;禁止导入 `sys`/`os`(或白名单 import) |
| 平凡通过  | 用 `try/except: pass` 吞掉所有异常;返回硬编码常量恰好通过 3 个可见测试                  | held-out 测试;mutation testing(自动变异测试用例)               |
| 超时利用  | 死循环压 timeout,或利用"超时不算失败"的实现                                      | 超时显式记为 $r=0$;资源限额(CPU 时间、内存、进程数)                     |
| 格式钻空  | 一次输出多个 `\boxed{}` 覆盖多个可能答案;在推理里穷举                                | 只取最后一个/唯一一个;检测到多答案直接 $r=0$                           |
| 答案泄漏  | prompt 或 few-shot 里意外包含答案;数据集与预训练语料重叠                            | 去重、去污染、held-out 时间切片评测                               |
| 语言/退化 | 中英混杂、无意义重复、乱码但答案对                                                | 格式与语言一致性作为**门控**(不是加分项)                              |

#### 9.2 验证器自身出错

* **假阴性**(对答案被判错):等价形式未归一化。后果比想象严重——它给正确行为负梯度,而且这些负梯度出现在最难的题上(那里 $p$ 最小,失败权重虽小,但正确样本被误判成失败等于直接抹掉了唯一的正信号)。**上线前必须人工抽检 200 条被判错的样本。**
* **假阳性**(错答案被判对):子串匹配过宽、数值容差过大、多选题猜中。后果是模型学到捷径。
* **验证器覆盖不足**:3 个测试的"通过"和 30 个测试的"通过"是完全不同的奖励函数。稀疏的测试集直接定义了一个更宽的可行域,而 RL 一定会找到那个可行域里最便宜的点。

***

### 参考

1. [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
2. [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
3. [https://arxiv.org/abs/2503.06639](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
4. [https://arxiv.org/abs/2503.20783](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
5. [https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
6. [https://arxiv.org/abs/2305.20050](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
7. [https://github.com/openai/prm800k](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
8. [https://arxiv.org/abs/2312.08935](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
9. [https://arxiv.org/abs/2410.01679](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
10. [https://arxiv.org/abs/2508.10293](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
11. [https://aclanthology.org/2026.findings-acl.1611/](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
12. [https://arxiv.org/abs/2603.29500](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
13. [https://arxiv.org/abs/2605.12519](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
14. [https://arxiv.org/abs/2504.13837](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
15. [https://arxiv.org/abs/2506.10947](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
16. [https://arxiv.org/abs/2605.17291](https://arxiv.org/abs/2402.03300https:/arxiv.org/abs/2501.12948https:/arxiv.org/abs/2503.06639https:/arxiv.org/abs/2503.20783https:/arxiv.org/abs/2503.14476https:/arxiv.org/abs/2305.20050https:/github.com/openai/prm800khttps://arxiv.org/abs/2312.08935https:/arxiv.org/abs/2410.01679https:/arxiv.org/abs/2508.10293https:/aclanthology.org/2026.findings-acl.1611/https://arxiv.org/abs/2603.29500https:/arxiv.org/abs/2605.12519https:/arxiv.org/abs/2504.13837https:/arxiv.org/abs/2506.10947https:/arxiv.org/abs/2605.17291)
