# 数据工程

### 1 Base 模型和 Instruct 模型的区别

我们经常在开源模型权重中看到base和instruct后缀，但其实这两个不是两个不同的模型。Base模型和Instruct模型在模型结构上完全相同，差异仅体现在训练数据的组织方式和损失函数的计算范围。

* Base 模型
  * 在海量无标注文本上做 Next Token Prediction
  * 输入纯文本序列（`"巴黎是法国的首都..."`）
  * 除第一个 token 外的**所有位置都计算 loss**
* Instruct 模型
  * 通常基于 Base 模型继续训练
  * 输入结构化对话模板（`<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！`）
  * 仅 **Assistant 回复部分计算 loss**
  * 包含 `im_start`, `im_end`, `eot_id`, `tool_call` 等角色标记

简单来说，就是 Base 模型的输入是文本块，Instruct 模型的输入是问答对。通常我们命名只经历 pretrain 阶段的模型为 xxx-base，命名经历了 SFT 阶段后的模型为 xxx-instruct。

### 2 训练数据拼接策略

在 SFT 阶段，数据样本的长度往往呈现**极端长尾分布，也就是短文本很多，长文本很少。**

如果简单使用 **Truncation（截断）** 或 **Padding（填充）** 到固定长度，会遇到两个问题：

* **Truncation**：短样本没问题，长样本被砍断（丢失关键上下文）
* **Padding**：短样本带来 70%+ 的无效计算（GPU 并行计算时，padding位置会有计算浪费）

<mark style="color:$warning;">Packing 策略将多个独立样本</mark> <mark style="color:$warning;"></mark><mark style="color:$warning;">`[A][B][C]`</mark> <mark style="color:$warning;"></mark><mark style="color:$warning;">拼接为连续序列，一次性计算，减少 padding 浪费。</mark>

但是，如果直接拼接而不处理 Attention Mask，会导致 **Cross-sample Contamination（样本间交叉注意力）**：模型在预测样本 B 的 token 时，能看到样本 A 的内容，而 A 中可能会包含答案或者提示，训练 loss 会异常降低，但模型学到的是虚假依赖。

将多条样本拼接后，通过设置 `attention_mask` 为块对角矩阵（或使用 `cu_seqlens` 机制），确保样本 A 的 token 无法 attend 到样本 B 的内容，同时仍保持因果语言模型的自回归特性（每个 token 只能看到同一样本内的前序 token）。

* **实现方式**：主流框架（如 Megatron、HuggingFace 的 `pack=True` 模式）支持传入 `position_ids` 与 `attention_mask`，将拼接序列在逻辑上分割为多个独立“虚拟序列”，既提高计算效率（减少 padding 占比），又保证训练信号纯净。

### **3 指令多样性设计**

SFT 阶段的指令覆盖度直接影响模型的泛化能力与任务适配性。

* **多轮对话构造**：单纯堆叠单轮样本会使模型缺乏多轮上下文理解能力。需要构造真实的多轮对话数据，通过 `user-assistant` 交替序列模拟对话历史，并在每轮中仅对当前轮次的 `assistant` 回复部分计算 loss，前序轮次作为上下文但不更新梯度，从而训练模型的多轮连贯性。
* **工具调用数据格式**：为增强模型使用外部工具的能力，需将工具调用设计为结构化的特殊 token 序列。常见格式包括 `<|tool_call|>function_name\narguments<|/tool_call|>` 或 JSON 形式的函数调用块，并在训练时明确区分“自然语言回复”与“工具调用输出”的 loss 计算范围，避免模型混淆生成模式。

### **4 数据质量过滤**

SFT 阶段，低质量或重复的指令数据会引入噪声甚至导致模型能力退化。

* **困惑度过滤**：利用预训练的 Base 模型计算每条样本的困惑度（PPL），剔除 PPL 过高（可能为异常、跨语言或逻辑混乱）和 PPL 过低（可能为简单重复）的数据，保留中等困惑度、信息量适中的样本。
* **Reward Model 筛选**：若有现成的偏好模型或质量评估模型，可对生成式样本进行质量打分，过滤掉低分回复。
* **指令复杂度评估**：通过 **Instag** 等方法对指令进行多维度标注（如推理难度、多约束条件、专业知识等），筛选高复杂度、高多样性的指令；或使用 **Self-Instruct** 方法通过种子指令 + 模型自生成的方式扩充指令集，再结合聚类与去重保证数据覆盖度与均匀性。

