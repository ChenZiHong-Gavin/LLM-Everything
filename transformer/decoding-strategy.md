# Decoding Strategy

Transformer 的 decoder 在生成文本时，并非一次性输出完整的句子，而是自回归逐个生成 token。在生成每个 token 时，模型会基于已经生成的 token 和编码器对输入信息的理解，计算出词汇表中所有可能 token 的概率分布。

如何从这个概率分布中选择下一个 token？

* 每次选择概率最高的 token——恭喜你，发明了 Greedy Search
* 每次保留概率最高的几个 token，再选择一条总概率最高的路径——恭喜你，发明了 Beam Search
* 如果不想要每次都走选最正确的路，希望有一点随机性——恭喜你，发明了 Sampling

## **1 Greedy Search**

Greedy Search 有时也称作启发式搜索，在 LLM 中，通常指在每个 _**t**_ 时刻选择下一个词时，根据  _**wt=argmaxwP(w|w1:t−1)**_ 选择概率最高的词。

<figure><img src="../.gitbook/assets/image (36).png" alt=""><figcaption></figcaption></figure>

Greedy Search 是一种合理的策略，但也有一些缺点，例如：

* 输出可能会卡在重复循环中。就像智能输入法给出的下一词建议，当你不断选择建议最高的单词时，它可能会变成重复的句子。

## **2 Beam Search**

与 Greedy Search 不同，Beam Search 为了避免错过隐藏的高概率词，通过参数 _**num\_beams**_ 的配置，可以在每个时刻，记录概率最高的前 _num\_beams_ 个路径，在下一个时刻可以有多个基础路径同时搜索。

如上图，当 t=1 时，最大概率的路径是（“The”、“nice”），Beam Search 同时也会记录概率排第二的路径（“The”、“dog”）；当t=2时，集束搜索也会发现路径（“The”、“dog”、“has”）有0.36的概率超过了路径（“The”、“nice”、“women”）的概率0.2。因此，两条路径中，找到了概率最高的路径，得到了更为合理的答案。 不过，在开放领域生成任务的时候，Greedy Search 和 Beam Search 都不是很好的解码方式，因为可能会导致缺乏创造性、趣味性、多样性。

### 2.1 Beam Search 代码实现

```python
class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        self.max_length = max_length - 1   # ignoring bos_token
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9
 
    def __len__(self):
        return len(self.beams)
 
    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            # 可更新的情况：数量未饱和或超过最差得分
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                # 数量饱和需要删掉一个最差的
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
 
    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        相关样本是否已经完成生成。
        best_sum_logprobs是新的候选序列中的最高得分。
        """
 
        if len(self) < self.num_beams:
            return False
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            # 是否最高分比当前保存的最低分还差
            ret = self.worst_score >= cur_score
            return ret

# 建立beam容器，每个样本一个
generated_hyps = [
    BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
    for _ in range(batch_size)
]
 
# 每个beam容器的得分，共batch_size*num_beams个
beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=encoder_input_ids.device)
beam_scores = beam_scores.view(-1)
 
# 每个样本是否完成生成，共batch_size个
done = [False for _ in range(batch_size)]
 
# 为了并行计算，一次生成batch_size*num_beams个序列
# 第一步自动填入bos_token
input_ids = torch.full(
    (batch_size*num_beams, 1),
    bos_token_id,
    dtype=torch.long,
    device=next(self.parameters()).device,
)
 
# 当前长度设为1
cur_len = 1

while cur_len < max_length:
    # 将编码器得到的上下文向量和当前结果输入解码器，即图中1
    output = decoder.decode_next_step(context, input_ids)
    # 输出矩阵维度为：(batch*num_beams)*cur_len*vocab_size
    
    # 取出最后一个时间步的各token概率，即当前条件概率
    # (batch*num_beams)*vocab_size
    scores = next_token_logits = output[:, -1, :]
 
    ###########################
    # 这里可以做一大堆操作减少重复 #
    ###########################
 
    # 计算序列条件概率的，因为取了log，所以直接相加即可。得到图中2矩阵
    # (batch_size * num_beams, vocab_size)
    next_scores = scores + beam_scores[:, None].expand_as(scores)
 
    # 为了提速，将结果重排成图中3的形状
    next_scores = next_scores.view(
            batch_size, num_beams * vocab_size
        )  # (batch_size, num_beams * vocab_size)
 
    # 取出分数最高的token（图中黑点）和其对应得分
    # sorted=True，保证返回序列是有序的
    next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
 
    # 下一个时间步整个batch的beam列表
    # 列表中的每一个元素都是三元组
    # (分数, token_id, beam_id)
    next_batch_beam = []
 
    # 对每一个样本进行扩展
    for batch_idx in range(batch_size):
 
        # 检查样本是否已经生成结束
        if done[batch_idx]:
            # 对于已经结束的句子，待添加的是pad token
            next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
            continue
 
        # 当前样本下一个时间步的beam列表
        next_sent_beam = []
 
        # 对于还未结束的样本需要找到分数最高的num_beams个扩展
        # 注意，next_scores和next_tokens是对应的
        # 而且已经按照next_scores排好顺序
        for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
            zip(next_tokens[batch_idx], next_scores[batch_idx])
        ):
            # get beam and word IDs
            # 这两行可参考图中3进行理解
            beam_id = beam_token_id // vocab_size
            token_id = beam_token_id % vocab_size
 
            effective_beam_id = batch_idx * num_beams + beam_id
 
            # 如果出现了EOS token说明已经生成了完整句子
            if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                # if beam_token does not belong to top num_beams tokens, it should not be added
                is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                if is_beam_token_worse_than_top_num_beams:
                    continue
                # 往容器中添加这个序列
                generated_hyps[batch_idx].add(
                    input_ids[effective_beam_id].clone(), beam_token_score.item(),
                )
            else:
                # add next predicted word if it is not eos_token
                next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
 
            # 扩展num_beams个就够了
            if len(next_sent_beam) == num_beams:
                break
 
        # 检查这个样本是否已经生成完了，有两种情况
        # 1. 已经记录过该样本结束
        # 2. 新的结果没有使结果改善
        done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
            next_scores[batch_idx].max().item(), cur_len=cur_len
        )
 
        # 把当前样本的结果添加到batch结果的后面
        next_batch_beam.extend(next_sent_beam)
 
    # 如果全部样本都已经生成结束便可以直接退出了
    if all(done):
        break
    
    # 把三元组列表再还原成三个独立列表
    beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
    beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
    beam_idx = input_ids.new([x[2] for x in next_batch_beam])
 
    # 准备下一时刻的解码器输入
    # 取出实际被扩展的beam
    input_ids = input_ids[beam_idx, :]
    # 在这些beam后面接上新生成的token
    input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
 
    # 更新当前长度
    cur_len = cur_len + 1
    # end of length while
```

## **3 Sampling**

### **3.1 Top-K 固态采样**

Beam Search 每次会选择在 Beam 中最大概率的词汇，Top-k 采样是对 Greedy Search 的优化，它从排名前 k 的 token 中进行抽样，允许其他概率较高的 token 也有机会被选中，也就是有一定机率不选最大概率的词，而其引入的随机性有助于在许多情况下提高生成质量。

<figure><img src="../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>

#### **3.1.1 代码实现**

```python
import torch
import torch.nn.functional as F

def top_k_sampling(logits, k, temperature=1.0):
    """
    对给定的 logits 执行 Top-K 采样。

    Args:
        logits (torch.Tensor): 模型输出的原始 logits，形状为 (vocab_size,)。
        k (int): 只从概率最高的 k 个词元中采样。
        temperature (float): 温度参数，用于调整概率分布的平滑度。

    Returns:
        int: 被采样的词元的 ID。
    """
    # 1. 对 logits 应用 temperature
    if temperature != 1.0:
        logits = logits / temperature

    # 2. 找到 top k 的 logits 和它们的索引
    # torch.topk 会返回 (values, indices)
    top_k_values, top_k_indices = torch.topk(logits, k)
    
    # 3. 创建一个新的 logits 张量，只保留 top k 的值，其余设为负无穷
    # 这是一种高效的过滤方法，因为 e^(-inf) = 0
    filtered_logits = torch.full_like(logits, -float('Inf'))
    filtered_logits[top_k_indices] = top_k_values
    
    # 4. 对过滤后的 logits 应用 softmax，得到一个合法的概率分布
    # 那些被设为 -inf 的位置，其概率会变为 0
    probabilities = F.softmax(filtered_logits, dim=-1)
    
    # 5. 从新的概率分布中随机采样一个词元
    # torch.multinomial 要求输入是概率，而不是 logits
    next_token_id = torch.multinomial(probabilities, num_samples=1)
    
    return next_token_id.item()
```

### **3.2 Top-P 动态采样**

Top-P方法可以动态设置token候选列表的大小。这种方法也称为 _Nucleus Sampling（核采样）_，通过选择可能性总和不超过特定值的高概率token来创建候选名单。可以看下面的示意图：

<figure><img src="../.gitbook/assets/image (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

#### 3.2.1 代码实现

```python
import torch
import torch.nn.functional as F

def top_p_sampling(logits, p, temperature=1.0):
    """
    对给定的 logits 执行 Top-p (Nucleus) 采样。

    Args:
        logits (torch.Tensor): 模型输出的原始 logits，形状为 (vocab_size,)。
        p (float): 累积概率阈值，介于 0 和 1 之间。
        temperature (float): 温度参数。

    Returns:
        int: 被采样的词元的 ID。
    """
    # 1. 对 logits 应用 temperature 并计算概率
    if temperature != 1.0:
        logits = logits / temperature
    probabilities = F.softmax(logits, dim=-1)

    # 2. 对概率进行降序排序，并保留原始索引
    # sorted_probs: 排序后的概率值
    # sorted_indices: 排序后概率值对应的原始索引
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

    # 3. 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 4. 找到需要保留的词元的索引（构建 Nucleus）
    # 我们移除那些累积概率超过 p 的词元。
    # 注意：我们至少保留一个概率最高的词元。
    sorted_indices_to_remove = cumulative_probs > p
    # 将第一个词元（概率最高的）的移除标记设为 False，确保至少有一个词被保留
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # 5. 创建一个只包含 Nucleus 中词元的新概率分布
    # 创建一个全零的分布
    filtered_probs = torch.zeros_like(probabilities)
    # 使用 scatter_ 将要保留的词的原始概率放回新分布中
    # sorted_indices[~sorted_indices_to_remove] 得到要保留的词的原始索引
    # sorted_probs[~sorted_indices_to_remove] 得到这些词的概率值
    indices_to_keep = sorted_indices[~sorted_indices_to_remove]
    probs_to_keep = sorted_probs[~sorted_indices_to_remove]
    filtered_probs.scatter_(dim=-1, index=indices_to_keep, src=probs_to_keep)

    # 6. 对新分布进行重新归一化
    renormalized_probs = filtered_probs / torch.sum(filtered_probs)

    # 7. 从重新归一化的分布中采样
    next_token_id = torch.multinomial(renormalized_probs, num_samples=1)
    
    return next_token_id.item()
```

### **3.3 Temperature 强化采样**

如上面的代码所示，Temperature 在模型计算出最终的词汇表概率分布（通过 Softmax 函数）之前，对原始的 `logits`进行缩放。

公式为：

$$
Probabilities=softmax( \frac{logits}{temperature})
$$

高 Temperature 意味着更多的随机性。这可以帮助模型提供更多创造性的输出。

通常我们是将 top-k、top-p、Temperature 联合起来使用。使用的先后顺序是`top-k->top-p->Temperature` 。

### 总结

1. 有哪些采样策略
2. 手撕
3. topp, topk, temperature是如何协同工作的

### 参考

1. [大模型（LLM）解码：从Greedy Search到Top-P\_大模型解码策略-CSDN博客](https://blog.csdn.net/Mike0010/article/details/138326166)
2. [https://zh.d2l.ai/chapter\_recurrent-modern/beam-search.html](https://zh.d2l.ai/chapter_recurrent-modern/beam-search.html)
3. [https://www.cnblogs.com/nickchen121/p/15499576.html](https://www.cnblogs.com/nickchen121/p/15499576.html)
