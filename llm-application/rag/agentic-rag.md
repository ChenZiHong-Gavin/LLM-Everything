# Agentic RAG

### 1 为什么需要 Agentic RAG

RAG 的核心思想是，把语言模型的**参数化记忆**和外部知识库的**非参数化记忆**结合起来，从而在知识密集型任务上获得更好的事实性和可更新性。

我们之所以需要RAG，是因为大模型本身有几个天然限制：

1. 模型参数里的知识会过时
2. 模型很难给出可靠出处
3. 领域私有知识通常不在预训练语料里

RAG 给大模型接了一套外部知识访问机制。但早期或者最常见的 Naive RAG 有一个很大的假设：**用户的问题可以通过一次检索解决。**

典型流程是：

```
Query -> Embedding -> Vector Search Top-K -> 拼接 Prompt -> LLM Answer
```

这在简单场景里够用，但在复杂场景里不够。

比如用户问：

> “我们最近几个客户流失案例里，最主要的共同原因是什么？和产品文档、客服记录、销售反馈能不能对上？”

这个问题不是一次向量检索能解决的。它至少包含：

* 需要识别“最近几个客户流失案例”
* 需要分别查 CRM、客服工单、销售记录、产品文档
* 需要对多类证据做归因
* 需要判断哪些信息可靠、哪些只是个案
* 需要最后给出可追溯的结论

而这，就是 Agentic RAG 要解决的问题。

### 2 Agentic RAG 是什么？

**Agentic RAG 是一种把 Agent 的规划、工具调用、反思、验证能力引入 RAG 流程的架构，让系统能够动态决定何时检索、检索什么、去哪里检索、是否继续检索，以及如何验证最终答案。**

Agentic RAG 是一次动态循环：

```
Plan, retrieve, evaluate, revise, retrieve again, verify, then answer.
```

这和 ReAct 的思想很接近。ReAct 论文提出让语言模型交错地产生 reasoning traces 和 actions：推理帮助模型规划、跟踪、更新行动，行动则让模型访问外部知识库或环境获取额外信息。

放到 RAG 里，action 就可以是：

* 搜向量库
* 搜关键词索引
* 查 SQL 数据库
* 查知识图谱
* 调用内部 API
* 访问网页
* 读取代码仓库
* 查日志系统
* 读取历史对话
* 调用 reranker
* 调用 fact checker
* 触发二次检索

总结来说，Agentic RAG 的关键是让模型拥有一个**检索决策循环**。

### 3 Agentic RAG 的典型架构

一个工程上可落地的 Agentic RAG 系统，可以画成这样：

<figure><img src="../../.gitbook/assets/image (58).png" alt=""><figcaption></figcaption></figure>

这张图的关键是两个循环：

第一个循环是：

```
Planner -> Retriever -> Evidence Evaluator -> Planner
```

解决的问题是：Evidence 不够怎么办？

第二个循环是：

```
Generator -> Verifier -> Planner
```

解决的问题是：答案不可靠怎么办？

### 4 Agentic RAG 的核心能力

#### 4.1 判断是否需要检索

不是所有问题都需要 RAG。

例如：“帮我把这句话润色一下。”不需要检索，模型直接回答就行了。

但如果用户问：“我们公司最新的报销政策是什么？”就必须检索。

普通 RAG 往往默认所有问题都检索，这会带来两个问题：

1. 浪费成本和延迟。
2. 引入不相关上下文，反而干扰模型。

Agentic RAG 的一个重要思想就是按需检索，而不是对每个输入都固定检索。

工程上可以做一个轻量级 router：

* 无需检索：写作、翻译、格式转换、通用解释
* 需要检索：事实查询、企业知识、时间敏感问题、需要引用的问题
* 需要多步检索：比较、归因、综合分析、多源交叉验证

#### 4.2 查询改写与问题拆解

用户的问题通常不是一个好的搜索 query。

用户会问：“这个方案为什么之前没推下去？”

Agentic RAG 需要先把用户问题转成一组可检索的子问题。

例如：

```
1. A 客户最近一次续约沟通记录是什么？
2. CRM 中记录的流失原因是什么？
3. 客服工单中是否出现高频问题？
4. 销售是否提到竞品或价格因素？
5. 产品使用数据是否显示活跃度下降？
6. 这些证据之间是否一致？
```

#### 4.3 工具选择

很多 RAG 系统一开始就把所有问题都丢给向量库。但真实业务知识往往分布在多种系统里：

| 数据类型     | 更适合的检索方式              |
| -------- | --------------------- |
| 文档、手册、制度 | 向量检索 + 关键词检索          |
| 数值报表     | SQL / BI 查询           |
| 实体关系     | Knowledge Graph       |
| 工单、日志    | 关键词检索 + 聚合统计          |
| 代码仓库     | 代码搜索 + AST / repo map |
| 实时信息     | API / Web Search      |
| 跨文档主题    | Graph RAG / 聚类摘要      |

Agentic RAG 的 router 应该根据问题选择工具，而不是无脑搜向量库。

#### 4.4 多轮检索与证据充分性判断

Agentic RAG 最重要的能力之一，是判断现在的证据够不够回答.

在真实系统里，检索结果可能有四种情况：

| 检索结果     | 系统应该怎么做       |
| -------- | ------------- |
| 证据充分且一致  | 直接生成答案        |
| 证据相关但不完整 | 继续检索          |
| 证据互相冲突   | 查更权威来源或标注冲突   |
| 证据不相关    | 改写 query 或换工具 |

一个普遍的做法是引入轻量级检索评估器，对检索结果质量打分，并根据置信度触发不同动作。

工程上可以设计一个 evidence evaluator，让它输出结构化结果：

```json
{
  "relevance": 0.82,
  "coverage": 0.65,
  "conflict": true,
  "missing_evidence": [
    "缺少客户最后一次续约会议记录",
    "缺少产品使用数据"
  ],
  "next_action": "retrieve_more"
}
```

这个 evaluator 不应该只问“大模型你觉得够了吗”。更可靠的方式是结合规则、检索分数、引用覆盖率、来源优先级和 LLM judge。

#### 4.5 生成后的验证

Agentic RAG 不应该在生成答案后就结束。因为即使检索结果正确，模型仍然可能：

* 错误归因；
* 把弱证据说成强结论；
* 引用了不存在的来源；
* 把多个来源混在一起；
* 忽略时间顺序。

RAG 并不能彻底消除幻觉。RAG 评估研究也强调，RAG 系统评估不能只看最终答案，还要同时评估检索质量、事实一致性、安全性和计算效率。

所以生产级 Agentic RAG 至少需要三类验证：

| 验证类型 | 要回答的问题              |
| ---- | ------------------- |
| 引用验证 | 答案里的每个关键断言是否能被引用支持？ |
| 事实验证 | 答案是否忠实于检索内容？        |
| 任务验证 | 是否真正回答了用户的问题？       |

像 ARES 等 RAG 自动评估框架把评估拆成 context relevance、answer faithfulness、answer relevance 等维度，这种拆法很适合落到 Agentic RAG 的 verifier 中。

### 5. Agentic RAG 的工程难点

#### 5.1 延迟变高

Agentic RAG 的多轮流程天然增加延迟。

工程上可以做几件事：

* 并行检索多个数据源；
* planner 用小模型；
* evaluator 用小模型或规则模型；
* 高频问题走缓存；
* 简单问题走普通 RAG；
* 复杂问题才进入 Agentic RAG；
* 对 Top-K 先粗排，再对少量候选做 rerank。

#### 5.2 上下文越来越长

Agentic RAG 在多轮检索和迭代推理之后，往往会积累大量候选材料。但上下文越长，并不意味着答案质量越高。

一方面，模型的注意力和利用能力并不是均匀分布在整个输入上的。_Lost in the Middle_ 的研究表明，长上下文模型对输入中间位置的信息利用并不稳定：当关键信息出现在开头或结尾时，模型更容易正确使用；而当相关信息被埋在中间时，性能可能明显下降。

因此，Agentic RAG 不能简单地把所有检索结果都塞进 prompt。真正重要的不是“检索到了多少”，而是“最终放进上下文的内容是否足够相关、足够紧凑、足够可引用”。

更合理的流程应该是：

> **retrieve broad, rerank carefully, compress selectively, cite precisely.**

也就是说，检索阶段可以尽量放宽，保证召回足够多的潜在证据；但进入上下文之前，必须经过严格重排、去重和筛选。对于冗长材料，还需要进行有选择的压缩，只保留回答问题所必需的论点、事实和出处。最终生成答案时，也要精确引用证据，而不是笼统地依赖一整段未经整理的上下文。

#### 5.3 Agent 过度行动

Agentic RAG 最大的问题之一，是 Agent 可能过度行动。

它可能：

* 不停改写 query；
* 查无关数据源；
* 在证据足够时仍继续检索；
* 被工具输出里的 prompt injection 影响；
* 把低质量来源当成权威来源；
* 为了完成任务而“编造”中间结论。

所以 Agentic RAG 必须有边界，比如设置工具白名单、数据源权限控制、最大循环次数等等。生产系统里的 Agentic RAG，最好不是“完全自治智能体”，而是“受控的检索状态机”。

### 6 Agentic RAG 的典型模式

#### 6.1 Router RAG

先判断问题类型，再选择路径。‘

* 简单事实问题 -> 普通 RAG
* 复杂分析问题 -> Agentic RAG
* 结构化数据问题 -> Text-to-SQL
* 全局总结问题 -> Graph RAG
* 无需外部知识 -> 直接 LLM

适合大多数生产系统。

#### 6.2 Multi-Hop RAG

把问题拆成多个子问题，逐步检索。

例如问题：A 公司为什么收购 B 公司？

子问题：

1. A 公司最近战略方向是什么？
2. B 公司核心资产是什么？
3. 双方业务是否互补？
4. 市场和财务动机是什么？
5. 管理层公开表述是什么？

适合研究、投研、法律、医疗、科研问答。

#### 6.3 Corrective RAG

先检索，再判断检索结果质量。如果质量不够，就纠错。

```
retrieve
-> grade documents
-> if irrelevant: rewrite query
-> if incomplete: retrieve more
-> if conflicting: search authoritative source
-> generate
```

#### 6.4 Self-Reflective RAG

模型不仅生成答案，还评估：

```
我是否需要检索？
检索内容是否相关？
我的答案是否被证据支持？
```

Self-RAG 是这一方向的代表，它通过自我反思机制提升事实性和引用质量。

#### 6.5 Graph Agentic RAG

当问题涉及实体关系、跨文档归因、全局主题时，可以把知识图谱和 Agentic RAG 结合起来。

```
文档 -> 实体抽取 -> 关系图谱 -> 社区摘要 -> 图检索 -> 多跳推理
```

GraphRAG 适合回答“整批文档说明了什么”这类问题，而不只是“哪一段文档包含答案”。

### 参考

1. [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511?utm_source=chatgpt.com)
2. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401?utm_source=chatgpt.com)
3. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629?utm_source=chatgpt.com)
4. [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983?utm_source=chatgpt.com)
5. [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
6. [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403)
7. [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)
8. [Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey](https://arxiv.org/abs/2504.14891)
