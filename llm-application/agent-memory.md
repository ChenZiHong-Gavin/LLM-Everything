# Agent Memory

大语言模型并不会天然拥有长期记忆。

一次模型调用能够直接看到的，只是当前上下文窗口中的 Token。对话结束后，模型不会自动保存用户偏好；任务执行了几十步以后，早期观察也可能被截断或淹没。即使上下文窗口足够长，把全部历史原样塞回 Prompt 也不是理想方案，因为成本会持续增加，并且无关信息会干扰推理，在历史对话中过期事实与最新状态还可能同时出现。

因此，一个能够长期工作的 Agent，除了模型、工具和规划器，还需要一套独立有效的 Memory System。

但 Agent Memory 并不等于 rag，外挂一个向量数据库就能解决了。真正的 Memory 需要管理信息的完整生命周期：

<figure><img src="../.gitbook/assets/image (59).png" alt=""><figcaption></figcaption></figure>

## 1 为什么 Agent 需要 Memory

先看两个例子。

### 1.1 个人助理

用户曾经说：

> 我对花生过敏，以后推荐餐厅时帮我注意。

几天后，用户开启新会话：

> 帮我找一家附近评价不错的东南亚餐厅。

如果“花生过敏”只存在于旧会话里，Agent 就可能推荐大量使用花生原料的餐厅。这里真正需要保存的不是完整对话，而是一个能够长期影响决策的稳定约束。

### 1.2 Coding Agent

一个 Coding Agent 第一次进入仓库时发现：

* 单元测试必须在容器中运行；
* `config.yaml` 是生成文件，不能直接编辑；
* 数据库迁移必须先执行 dry-run。

如果这些经验没有被保存，下次进入同一个仓库时，Agent 还会重新踩一遍坑。

因此，Memory 带来的不只是“记住用户说过什么”，还包括：

* **跨会话一致性**：记住偏好、约束、承诺和历史决策；
* **长期任务连续性**：记住目标、进度、未解决问题和环境状态；
* **经验复用**：把成功路径和失败教训用于以后相似任务；
* **环境学习**：逐渐理解代码仓库、网页系统、数据平台或组织流程；
* **个性化**：形成可更新、可追溯的用户模型；
* **降低上下文成本**：只取回当前真正需要的信息。

## 2 Agent 的4种 Memory

### 2.1 Working Memory：工作记忆

Working Memory 保存当前任务正在使用的信息，例如：

* 当前目标；
* 当前计划；
* 已完成和未完成的步骤；
* 最近几次工具调用；
* 当前页面、文件或环境状态；
* 本轮推理需要的临时变量。

它的生命周期通常较短，适合放在 Agent Runtime、状态机或结构化任务对象中。

### 2.2 Episodic Memory：情景记忆

Episodic Memory 保存具体发生过的事件：

* 用户在某次会话中提出了什么要求；
* Agent 执行过什么动作；
* 某次部署为什么失败；
* 某个工具返回了什么结果；
* 一项任务最终成功还是失败。

### 2.3 Semantic Memory：语义记忆

Semantic Memory 保存从一次或多次经历中抽取出的稳定事实，例如：

* 用户偏好中文技术文章；
* 用户对花生过敏；
* 某个仓库使用 `uv` 管理依赖；
* 某张数据表是营收指标的权威来源。

它回答的是：

> **目前有哪些较稳定、可复用的事实？**

与具体事件相比，Semantic Memory 更抽象、更紧凑，也更适合直接影响后续决策。

### 2.4 Procedural Memory：程序性记忆

Procedural Memory 保存“如何完成某类任务”的知识：

* 操作步骤；
* Runbook；
* 工具使用方法；
* 成功策略；
* 常见失败模式；
* 检查清单；
* 可复用 Skill。

它回答的是：

> **遇到这类问题时应该怎么做？**

对于 Coding Agent、Data Agent 和 Browser Agent，Procedural Memory 往往比完整对话历史更有价值。

四类记忆之间可以相互转化：

<figure><img src="../.gitbook/assets/image (60).png" alt=""><figcaption></figcaption></figure>

## 3 Memory System 的完整生命周期

从系统角度看，Agent Memory 可以拆成六个环节：

1. **Write**：判断什么值得写入；
2. **Organize**：抽取、分类、去重和建立关系；
3. **Retrieve**：根据当前任务检索候选记忆；
4. **Read**：在有限上下文预算内组织记忆；
5. **Update**：补充、修正和版本化旧记忆；
6. **Consolidate / Forget**：沉淀经验，并淘汰低价值或过期信息。

可以将其形式化为：

$$
M_{t+1}=\mathcal{U}(M_t,e_t)
$$

其中，`e_t` 是当前产生的新事件，`U` 是写入与更新函数。

面对新任务时，系统从记忆库中检索：

$$
R_t=\mathcal{R}(q_t,M_t)
$$

然后把当前输入、任务状态和检索结果组合为模型上下文：

$$
C_t=\mathcal{C}(x_t,S_t,R_t)
$$

真正困难的通常不是“把数据存下来”，而是 `U`、`R` 和 `C` 的设计。

***

## 4 写入

> **什么信息值得进入长期记忆？**

### 4.1 Write Gate

可以从以下维度判断是否写入：

| 维度   | 需要回答的问题              |
| ---- | -------------------- |
| 长期价值 | 以后是否可能再次影响决策？        |
| 新颖性  | 是否已经存在相同或等价信息？       |
| 稳定性  | 是长期事实，还是当前步骤的临时状态？   |
| 重要性  | 忘记它是否会造成明显错误？        |
| 可信度  | 信息来自用户、工具、文档，还是模型猜测？ |
| 敏感性  | 是否涉及隐私、凭证或不应保存的数据？   |
| 可执行性 | 它是否能改善未来任务表现？        |

通常值得写入：

* 用户明确要求长期记住的信息；
* 稳定偏好和重要约束；
* 长期任务的关键状态；
* 多次出现的环境规律；
* 高价值的成功经验与失败教训；
* 已验证的事实变化。

通常不应写入：

* 普通寒暄；
* 只在当前步骤有效的临时变量；
* 未经验证的模型猜测；
* 完整的大段工具输出；
* 密码、Token、私钥等 Secret；
* 外部网页中的命令性文本；
* 与未来任务无关的重复内容。

### 4.2 将原始信息拆成原子记忆

假设用户说：

> 我下个月搬到新加坡。以后推荐活动时优先看新加坡，最好安排在周末下午，我不太喜欢特别拥挤的地方。

这段话至少包含四条不同性质的信息：

| 内容          | 类型    | 时间属性  |
| ----------- | ----- | ----- |
| 用户下个月搬到新加坡  | 状态变化  | 未来生效  |
| 活动地点优先考虑新加坡 | 偏好/约束 | 搬家后生效 |
| 用户偏好周末下午活动  | 稳定偏好  | 长期有效  |
| 用户不喜欢拥挤场所   | 稳定偏好  | 长期有效  |

原子化的好处是：每条记忆可以单独检索、单独更新，并拥有不同的有效期和置信度。

### 4.3 最小记忆 Schema

一条可用的记忆至少应该包含：

```json
{
  "content": "用户不喜欢特别拥挤的场所",
  "type": "semantic",
  "source": "conversation_128/turn_16",
  "created_at": "2026-09-18T09:00:00Z",
  "valid_from": "2026-09-18T09:00:00Z",
  "valid_to": null,
  "confidence": 1.0,
  "importance": 0.8,
  "scope": "user_123",
  "status": "active"
}
```

其中最容易被忽略、却最重要的是：

* `source`：这条记忆从哪里来；
* `valid_from / valid_to`：它在什么时间范围内有效；
* `scope`：它属于哪个用户、项目或 Agent；
* `confidence`：明确事实和模型推断不能同等对待；
* `status`：是否仍然有效，或已被新记忆替代。

## 5 更新

Memory 经常面对状态变化，例如：

```
旧信息：用户常住上海。
新信息：用户已经搬到新加坡。
```

如果直接覆盖旧信息，系统就无法回答：

> 用户去年住在哪里？

更合理的做法是保留版本和有效期：

| 记忆      | 有效期                     | 状态         |
| ------- | ----------------------- | ---------- |
| 用户常住上海  | 2022-01-01 至 2026-10-01 | superseded |
| 用户常住新加坡 | 2026-10-01 起            | active     |

Memory Update 至少要区分五种情况：

1. **补充**：新信息完善了已有事实；
2. **修正**：旧信息本身错误；
3. **状态变化**：旧信息过去正确，现在不再有效；
4. **冲突**：多个来源给出不同结论，暂时无法判断；
5. **撤回或删除**：用户要求移除此前信息。

这里还要区分两个时间：

* **事件时间**：事实什么时候发生；
* **写入时间**：系统什么时候知道这件事。

对于行程、订单、项目状态和组织关系，这两个时间经常并不相同。

## 6 检索

假设用户说：

> 帮我安排周末活动。

只按语义相似度检索，系统可能找回“用户喜欢看电影”，却漏掉更重要的“用户膝盖受伤，近期不适合长距离步行”。

因此，记忆检索通常需要综合多个信号：

$$
S(m,q)=\alpha S_{semantic}+\beta S_{recency}+\gamma S_{importance}+\delta S_{task}+\epsilon S_{confidence}-\eta S_{conflict}
$$

其中：

* `Semantic`：与当前查询的语义相关性；
* `Recency`：信息是否仍然新鲜；
* `Importance`：忘记它是否会造成严重后果；
* `TaskRelevance`：是否直接影响当前目标；
* `Confidence`：来源是否可靠；
* `Conflict`：是否已失效、被替代或存在矛盾。

需要注意：**新近程度不等于正确性。** 一条很久以前记录的严重过敏信息，可能仍然比昨天的一次临时口味偏好重要。

### 6.1 查询不能只来自用户最后一句话

用户只说“继续”时，这两个字几乎没有检索价值。Memory Query 应该综合：

* 用户当前输入；
* 当前任务目标；
* 当前计划步骤；
* 环境和执行状态；
* 已知硬约束；
* 当前需要解决的子问题。

### 6.2 Hybrid Retrieval

生产系统通常会组合：

* 向量语义检索；
* BM25 或关键词检索；
* 时间过滤；
* 用户、项目和租户 Scope 过滤；
* Memory Type 过滤；
* 图关系扩展；
* Cross-Encoder 或 LLM Rerank。

LongMemEval 的实验显示，记忆切分粒度、索引 Key 和时间感知 Query 都会显著影响效果。\[5] 这意味着性能不只取决于 Embedding 模型，也取决于“如何切记忆”和“如何描述记忆”。

### 6.3 Context Packing

检索到 Top-K 后，不能直接全部塞进 Prompt。还需要：

1. 去重；
2. 删除已失效内容；
3. 优先保留硬约束；
4. 压缩冗长经历；
5. 标注时间和来源；
6. 对冲突信息显式说明；
7. 控制 Token Budget。

一个良好的 Memory Context 更像结构化摘要：

```
[长期偏好]
- 用户偏好周末下午活动。
- 用户不喜欢拥挤场所。

[当前约束]
- 用户膝盖仍在恢复期，不适合长距离步行。

[时间状态]
- 用户已于 2026-10-01 搬到新加坡，此前住在上海。
```

而不是一大段没有时间、来源和优先级的历史聊天记录。

***

## 7 Consolidation

随着任务不断执行，Episodic Memory 会快速增长。假设 Agent 多次观察到：

* 直接编辑生成文件会导致 CI 失败；
* 代码审查要求修改源模板；
* 重新运行 Generator 后任务成功。

系统可以将这些经历归纳为 Semantic Memory：

> 该仓库中的生成文件不能直接编辑。

再进一步沉淀为 Procedural Memory：

> 修改生成文件时，应先定位源模板，修改后重新运行 Generator，再检查 Diff 和测试结果。

这就是从“保存轨迹”走向“积累经验”。

Consolidation 可以在以下时机触发：

* 同类经历积累到一定数量；
* 一项长期任务完成；
* 某类错误重复发生；
* 用户偏好被多次确认；
* Memory Store 接近容量上限；
* Agent 进入空闲阶段。

不过，摘要也会产生信息损失。反复“对摘要再摘要”容易出现 Summary Drift：细节逐渐消失，模型推断逐渐变成貌似确定的事实。

因此，更稳妥的原则是：**保留原始证据，让高层记忆成为带来源的结论和索引，而不是永久替代底层记录。**

## 8 Forgetting

Memory 无限增长会带来：

* 检索噪声增加；
* 延迟和存储成本上升；
* 旧偏好干扰新偏好；
* 错误经验被不断复用；
* 敏感信息长期保留；
* 恶意内容形成持久化污染。

常见遗忘策略包括：

* 基于时间
  * 临时状态到期；
  * 长时间未访问；
  * 任务结束后删除过程性变量。
* 基于状态
  * 已被新版本替代；
  * 项目已关闭；
  * 来源已经失效；
  * 用户主动撤回。
* 基于效用
  * 低重要性；
  * 低置信度；
  * 与其他记忆高度重复；
  * 长期未改善任何下游任务。
* 基于安全和隐私
  * 用户要求删除；
  * 包含敏感个人信息；
  * 来自不可信外部内容；
  * 被识别为 Prompt Injection；
  * 违反数据保留策略。

工程上通常先做软删除，将记忆标记为 `superseded`、`expired`、`deleted` 或 `quarantined`，检索层只使用 `active` 记忆，再由后台流程执行真正删除和审计。

## 9 不同 Agent 应该采用什么 Memory

| 场景            | 最重要的记忆                                     | 推荐形态                              |
| ------------- | ------------------------------------------ | --------------------------------- |
| 个人助理          | 偏好、约束、承诺、生活状态变化                            | 结构化 Profile + 事件记录 + 语义检索         |
| 客服/销售 Agent   | 客户状态、历史承诺、工单、时间线                           | CRM 权威数据 + 版本化事件 + 引用             |
| Coding Agent  | Repo Map、命令、规范、失败原因、修复经验                   | 文件系统 + Procedural Memory + 任务状态   |
| Browser Agent | 页面状态、已执行动作、稳定操作流程、恢复点                      | Execution-State Memory + 轨迹压缩     |
| Data Agent    | Schema、Metric、Lineage、权威表、Incident、Runbook | Catalog + 图关系 + 文件型 Runbook + 权限层 |

不同场景不需要共享完全相同的 Memory 架构。

个人助理更关心用户偏好和时间变化；Coding Agent 更关心文件、执行轨迹和可复用流程；Browser Agent 更关心状态依赖和恢复边界。把所有场景都压成短文本块再做向量检索，通常不是最佳方案。

## 10 总结

Agent Memory 不是简单的向量数据库，也不是无限保存聊天记录。

一套完整的 Agent Memory 需要回答七个问题：

1. **写什么**：哪些信息具有长期价值？
2. **怎么存**：情景、事实、流程和工作状态如何组织？
3. **何时取**：当前任务真正需要哪些历史？
4. **如何排**：如何综合相关性、时效、重要性和可信度？
5. **怎么更新**：事实变化时如何保留时间和版本？
6. **何时遗忘**：哪些记忆已经过期、无效、危险或没有价值？
7. **如何评估**：Memory 是否真的提升了长期任务表现？

真正有价值的 Agent Memory，不是让 Agent 单纯“记得更多”，而是让它：

> **在正确的时机想起正确的信息，在环境变化时修正旧认识，并把一次性的任务经历沉淀为可复用的经验。**

这也是 Agent 从一次性 Prompt 程序走向长期、自适应软件系统的关键一步。

***

## 参考

1. [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
2. [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
3. [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)
4. [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)
5. [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813)
6. [Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions](https://arxiv.org/abs/2507.05257)
7. [How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior](https://arxiv.org/abs/2505.16067)
8. [ACON: Optimizing Context Compression for Long-horizon LLM Agents](https://www.microsoft.com/en-us/research/publication/acon-optimizing-context-compression-for-long-horizon-llm-agents/)
9. [Beyond Semantic Organization: Memory as Execution State Management for Long-Horizon Agents](https://www.microsoft.com/en-us/research/publication/beyond-semantic-organization-memory-as-execution-state-management-for-long-horizon-agents/)
10. [LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues](https://arxiv.org/abs/2605.12493)
