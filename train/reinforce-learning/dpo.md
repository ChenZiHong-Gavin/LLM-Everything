# DPO

**DPO**（Direct Preference Optimization）的论文标题：

> _"Your Language Model is Secretly a Reward Model"_ （你的语言模型其实就是一个奖励模型）

**意思是，其实不需要像PPO那样显式地训练一个奖励模型，然后再做强化学习。你可以直接用人类偏好数据来优化语言模型本身。**

回到 RLHF 的优化目标：

$$\max_{\pi} \; \mathbb{E}_{y \sim \pi} \left[ r(x, y) \right] - \beta \cdot \mathbb{KL}\left[ \pi(y|x) \;\|\; \pi_{\text{ref}}(y|x) \right]$$

这个优化问题有一个**闭式解**（closed-form solution）：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{r(x,y)}{\beta}\right)$$

其中 Z(x) 是归一化常数，确保概率和为 1：

$$Z(x) = \sum_y \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{r(x,y)}{\beta}\right)$$

**对于这个公式的直觉理解是：**&#x6700;优策略就是在参考模型的基础上，按照奖励的指数比例"重新加权"——奖励高的回答概率增大，奖励低的概率缩小。

把上面的最优解反过来解 r(x, y)：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

**也就是说：奖励可以用"最优策略相对于参考策略的对数概率比"来表示**。

Z(x) 只依赖于 x（prompt），和 y（回答）无关。这意味着当我们比较两个回答的奖励差时，Z(x) 会被消掉。

**代入 Bradley-Terry 偏好模型**

人类偏好数据的格式是：给定 prompt x，人类认为回答 y\_w（winner）优于回答 y\_l（loser）。

Bradley-Terry 模型假设人类选择 y\_w 优于 y\_l 的概率为：

$$p(y_w \succ y_l | x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

其中 σ 是 sigmoid 函数。

把第二步的奖励表达式代入：

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$$

注意 **Z(x) 被消掉了。**

所以：

$$p(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

这里用 π\_θ 替代了 π\*，因为我们要训练的模型就是对最优策略的近似。

最终的 DPO 损失函数就是对上面偏好概率取负对数似然：

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right) \right]$$

**拆开来看**：

```
对每一条偏好数据 (x, y_w, y_l)：

1. 计算胜出回答的"隐式奖励"：
   r_w = β × log[ π_θ(y_w|x) / π_ref(y_w|x) ]

2. 计算失败回答的"隐式奖励"：
   r_l = β × log[ π_θ(y_l|x) / π_ref(y_l|x) ]

3. 希望 r_w > r_l，即：
   loss = -log σ(r_w - r_l)
```

**直觉理解**：

* **π\_θ(y\_w|x) / π\_ref(y\_w|x)** ：模型相对于参考模型，对好回答的概率提升了多少
* **π\_θ(y\_l|x) / π\_ref(y\_l|x)** ：模型相对于参考模型，对坏回答的概率提升了多少
* DPO 要求前者的提升**大于**后者的提升——即模型应该"更偏向好回答"

这和交叉熵损失的形式完全一样，可以直接用标准的梯度下降来优化。利用一个**简洁的分类损失代替采样、强化学习和奖励模型。**

对 DPO 损失求梯度，可以更清晰地看到它在做什么：

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \cdot \mathbb{E} \Big[ \underbrace{\sigma(-\hat{r})}_{\text{权重}} \Big( \underbrace{\nabla_\theta \log \pi_\theta(y_w|x)}_{\text{增大好回答概率}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l|x)}_{\text{减小坏回答概率}} \Big) \Big]$$

其中 $$\hat{r} = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$$

**解读**：

1. **增大好回答的概率，减小坏回答的概率**——这是方向
2. **权重 σ(-r̂)**——这是力度。当模型已经正确区分了好坏回答（r̂ 很大），权重接近 0，梯度变小；当模型还分不清（r̂ 接近 0），权重接近 0.5，梯度最大。这是一个**自适应的加权机制**，让模型把注意力集中在它还没学好的样本上。



### 参考

1. [https://www.cnblogs.com/GreenOrange/p/18798910](https://www.cnblogs.com/GreenOrange/p/18798910)
