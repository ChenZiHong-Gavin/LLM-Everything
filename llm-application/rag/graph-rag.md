# Graph RAG

## Graph RAG

Graph RAG 入门：从文本块到知识图谱，解决全局问题

很多人第一次听到 Graph RAG，会以为它就是“RAG + Neo4j”：把文档切块之后，再抽几个实体和关系，存进图数据库，查询时做一下图遍历。

这个理解不算错，但只说到了一半。

Graph RAG 真正重要的地方，不是“用了图数据库”，而是它改变了 RAG 的默认假设。

普通 RAG 默认答案藏在某几个 chunk 里，所以只要把和问题最相似的 Top-K chunk 找出来，再交给 LLM 总结就行。Graph RAG 认为：很多问题的答案并不藏在某几个 chunk 里，而是藏在整个语料的结构里。你必须先理解文档之间的实体、关系、主题和社区，才能回答这类问题。

这篇文章讲清楚 Graph RAG 解决什么问题、核心流程是什么、它和普通 RAG / Agentic RAG 的区别，以及实际落地时最容易踩的坑。

#### 1 普通 RAG 的盲区：全局问题

普通 RAG 最擅长的是局部问题。

比如：

> “合同第 8 条关于提前终止怎么写？”
>
> “Chamomile 有什么疗效？”
>
> “Python SDK 里 create\_session 参数怎么用？”

这些问题通常有一个明确答案，答案也大概率出现在某个文档片段里。只要检索能命中对应 chunk，后面的生成就比较简单。

但真实业务里经常出现另一类问题：

> “这批客户流失的共同原因是什么？”
>
> “过去两周的产品反馈里，最值得关注的风险是什么？”
>
> “这家公司内部文档反映出的主要组织问题是什么？”
>
> “最近几个安全事故之间有没有共同模式？”

这些问题不是在问“某个事实在哪里”，而是在问“整个数据集呈现出什么模式”。

这就是 Graph RAG 论文里强调的 **global question**。它不是普通检索问题，而更像 query-focused summarization：围绕一个问题，对整个语料做有目标的总结。

普通向量 RAG 处理这类问题会出现三个问题。

**1.1 Top-K chunk 不代表整体**

如果用户问“主要主题是什么”，向量检索会找出和“主题”“主要”这类词语最相似的 chunk。但这些 chunk 只是局部相似，不一定代表整个语料。

这就像你想了解一本书的核心思想，却只翻到了几页出现“总结”“观点”“主题”的段落。它们可能很相关，但不能代表全书。

**1.2 调大 Top-K 也不能真正解决**

很多人遇到全局问题时，会直接把 Top-K 从 5 改成 20、50、100。

这会带来两个副作用：

1. 召回内容变多，但信息密度下降；
2. 上下文里混入大量重复、过期、冲突或弱相关信息。

最后模型不是更聪明了，而是在一堆噪声里“强行写报告”。

**1.3 多跳关系很难靠相似度发现**

有些答案需要跨文档串联关系。

比如客户流失分析里，一个客户在客服工单里抱怨“响应慢”，在销售记录里提到“竞品报价更低”，在产品反馈里又反复提到“权限体系不够灵活”。单看任意一个 chunk，都不一定能看出流失原因；把这些实体和事件连起来，才会看到完整模式。

普通向量检索擅长找相似文本，不擅长显式表示“谁和谁有关、通过什么关系有关”。

Graph RAG 要解决的，正是这类结构性问题。

#### 2 Graph RAG 是什么？

一句话：

> **Graph RAG 是一种把非结构化文本先组织成知识图谱和社区摘要，再基于图结构做检索与生成的 RAG 架构。**

它不是简单地把文档丢进图数据库，而是把 RAG 的索引对象从“孤立 chunk”升级成“语义网络”。

普通 RAG 的索引对象主要是：

```
chunk text + embedding + metadata
```

Graph RAG 的索引对象会更丰富：

```
TextUnit
Entity
Relationship
Claim
Community
Community Report
Embedding
Source Reference
```

其中最关键的是两个东西：

1. **知识图谱**：表示实体之间的关系；
2. **社区摘要**：表示一组紧密相关实体共同构成的主题。

知识图谱让系统能“连点成线”，社区摘要让系统能“从局部看到整体”。

所以 Graph RAG 不是为了替代向量检索，而是为了给 RAG 增加一种新的信息组织方式：从文本相似度，升级到语义结构。

#### 3 索引阶段：把文本变成图

Graph RAG 的第一阶段是离线索引。

可以把它理解成：在用户提问之前，先让系统读完整个语料，抽取里面的重要对象和关系，并预先建立一张“语义地图”。

典型流程如下：

```
Documents
  -> TextUnits
  -> Entities / Relationships / Claims
  -> Entity Graph
  -> Community Detection
  -> Community Reports
  -> Embeddings + Metadata + Source References
```

**3.1 TextUnit：不是为了回答，而是为了建图**

普通 RAG 切 chunk 是为了检索和回答。Graph RAG 里的 TextUnit 也可以理解成 chunk，但它的第一目标是支持后续的信息抽取。

切得太小，关系会断；切得太大，实体和关系会混杂。

比如：

```
A 公司在 2024 年 Q3 取消续约。客户成功记录显示，
主要原因是集成成本高、权限配置复杂，同时竞品 B 提供了更低报价。
```

这里至少包含几个实体：

```
A 公司
2024 年 Q3
客户成功记录
集成成本
权限配置
竞品 B
报价
```

以及几条关系：

```
A 公司 -> 取消续约 -> 2024 年 Q3
A 公司 -> 流失原因 -> 集成成本高
A 公司 -> 流失原因 -> 权限配置复杂
竞品 B -> 提供 -> 更低报价
```

如果切分把这些信息拆散，后面建出来的图就会丢失关键关系。

**3.2 实体抽取：图的节点**

实体可以是人、组织、产品、地点，也可以是业务概念、技术模块、风险类型、流程节点。

在企业知识库里，实体不一定都是“专有名词”。比如：

```
报销政策
提前解约
权限体系
SLA
数据同步
客户流失
安全审计
```

这些概念如果在多个文档里反复出现，就值得成为图里的节点。

实体抽取最难的不是“抽出来”，而是“抽干净”。

同一个实体可能有很多写法：

```
OpenAI
Open AI
OpenAI 公司
开放人工智能公司
```

如果不做归一化，图会被切成多个碎片，看起来节点很多，实际关系很散。

**3.3 关系抽取：图的边**

关系是 Graph RAG 相比普通 RAG 最关键的增量。

普通 RAG 知道两个 chunk 语义相似，但不知道它们为什么相关。Graph RAG 希望显式表示：

```
客户 A -> 投诉 -> 响应慢
客户 A -> 使用 -> 产品 X
客户 A -> 转向 -> 竞品 Y
产品 X -> 存在问题 -> 权限复杂
```

关系最好带上来源证据，而不是只存一个三元组。

更好的结构是：

```
{
  "source": "客户 A",
  "relation": "投诉",
  "target": "响应慢",
  "description": "客户 A 在 2024-05 的工单中多次提到响应慢影响上线进度",
  "source_text": "...原文片段...",
  "document_id": "...",
  "confidence": 0.82
}
```

这样做的好处是，最后生成答案时可以追溯到原始证据，而不是只相信图里的抽取结果。

**3.4 社区发现：把图聚成主题**

当实体和关系足够多之后，图会变得很复杂。

Graph RAG 会对图做社区发现，把联系紧密的一组节点聚在一起。一个社区通常对应一个主题、事件簇、业务问题或知识模块。

比如客户流失数据里，可能聚出几个社区：

```
社区 1：价格敏感型客户
社区 2：权限和安全诉求
社区 3：上线实施阻塞
社区 4：竞品迁移
社区 5：客服响应慢
```

社区发现的意义在于：它把大量零散实体组织成不同层次的语义主题。

这一步很重要，因为后面的全局回答不是直接遍历所有原文，而是更多依赖这些社区摘要。

**3.5 Community Report：提前写好的“局部综述”**

社区发现之后，Graph RAG 会让 LLM 为每个社区生成摘要，也就是 Community Report。

它不是普通摘要，而是围绕社区里的实体和关系，写出这个社区的主题、关键实体、内部关系、重要发现和可能风险。

你可以把 Community Report 理解成系统提前写好的“局部研究报告”。

普通 RAG 是用户问了之后才临时总结 Top-K chunk。Graph RAG 则是在索引阶段就先总结好语料的结构。

这也是 Graph RAG 能回答全局问题的关键：当用户问“主要风险是什么”时，系统不用从原文里临时大海捞针，而是可以读取不同层级的社区报告，再做有目标的综合。

#### 4 查询阶段：Graph RAG 不是一种检索器，而是一组检索策略

Graph RAG 的查询方式不止一种。

不同问题应该走不同路径。

**4.1 Global Search：回答整体性问题**

Global Search 适合这种问题：

```
这批文档的主要主题是什么？
过去两周有哪些关键风险？
这些客户流失案例的共同原因是什么？
这个组织的主要协作瓶颈是什么？
```

这类问题需要理解整个语料，而不是定位某个片段。

Global Search 的典型流程是 map-reduce：

```
User Query
  -> 读取某一层级的 Community Reports
  -> Map：每批社区报告独立生成中间答案，并给要点打分
  -> Filter / Rank：保留重要中间要点
  -> Reduce：综合所有中间答案，生成最终回答
```

这个流程的核心是：先让模型在不同社区上分别思考，再把局部观点合并成整体答案。

它比普通 Top-K 检索更适合全局问题，因为社区报告本身已经覆盖了语料的结构。

**4.2 Local Search：回答实体相关问题**

Local Search 适合这种问题：

```
客户 A 为什么流失？
产品 X 被哪些客户投诉过？
某个功能和哪些风险有关？
某个研究者在数据集中和哪些主题相关？
```

这类问题围绕某个实体展开。

Local Search 的流程大致是：

```
User Query
  -> 找到语义相关实体
  -> 沿着实体关系向外扩展
  -> 找到相关关系、相邻实体、原始 TextUnits、社区报告
  -> 排序和过滤
  -> 拼成上下文
  -> 生成答案
```

它和普通向量检索的区别在于：普通检索只找相似 chunk，Local Search 会把实体当成入口，沿着关系网络找证据。

这就像你不是在全文搜索“客户 A”，而是在看客户 A 的关系图谱：它和哪些问题、产品、人员、竞品、事件连在一起。

**4.3 DRIFT Search：在全局和局部之间折中**

Global Search 全面，但可能贵。Local Search 具体，但可能视野太窄。

DRIFT Search 的思路是把二者结合起来：

```
先用社区报告获得大方向
  -> 生成更具体的追问
  -> 用 Local Search 继续深入
  -> 得到更细粒度证据
  -> 汇总成最终答案
```

它适合很多真实问题，因为真实问题往往既需要整体背景，也需要局部证据。

比如：

> “最近客户流失的主要原因是什么？请给出几个典型案例。”

这不是纯全局问题，也不是纯局部问题。你既要总结总体原因，也要落到具体客户案例。DRIFT 这类方法的目标就是在成本和质量之间做折中。

**4.4 Basic Search：不要为了图而图**

不是所有问题都需要 Graph RAG。

如果用户只是问：

```
“报销系统入口在哪里？”
“API token 怎么创建？”
“第 3 条规则原文是什么？”
```

普通向量检索、关键词检索或 hybrid search 可能更简单、更快、更准。

Graph RAG 是增强工具，不是默认答案。

#### 5 一个例子：用 Graph RAG 分析客户流失

假设公司里有这些数据：

```
CRM 记录
销售跟进记录
客服工单
客户成功会议纪要
产品反馈
竞品分析
合同续约记录
```

用户问：

> “最近流失客户的共同原因是什么？”

普通 RAG 可能会检索到几个包含“流失”“续约失败”“竞品”的 chunk，然后让模型总结。

这很容易漏掉信息。因为有些客户记录里可能没有出现“流失”这个词，而是写：

```
客户暂停推进
客户转向其他方案
客户认为集成周期不可控
客户暂缓预算
```

Graph RAG 会先在索引阶段形成类似这样的图：

```
客户 A -> 取消续约 -> 产品 X
客户 A -> 投诉 -> 集成复杂
客户 A -> 提到 -> 竞品 Y
客户 B -> 延迟上线 -> 权限配置问题
客户 B -> 投诉 -> 客服响应慢
客户 C -> 预算收缩 -> 降级套餐
客户 C -> 对比 -> 竞品 Y
产品 X -> 常见问题 -> 权限配置复杂
竞品 Y -> 优势 -> 报价更低
```

然后社区发现可能得到：

```
社区 1：集成和上线复杂度
社区 2：权限与安全配置
社区 3：价格与竞品压力
社区 4：客服响应与客户成功流程
```

当用户问“共同原因”时，Graph RAG 可以先看这些社区报告，再综合出答案：

```
最近流失客户的原因主要集中在四类：

1. 集成和上线成本高
   多个客户在上线阶段遇到接口联调、权限配置、数据同步问题。

2. 权限体系复杂
   企业客户对细粒度权限、审计和安全合规有更强诉求，但现有产品配置成本高。

3. 竞品价格压力
   部分客户在续约前对比竞品，竞品 Y 经常以更低报价切入。

4. 客户成功响应不足
   多个高风险客户在流失前都有响应慢、问题升级不及时的记录。
```

这类答案不是来自某一个 chunk，而是来自整个语料的结构。

#### 6 Graph RAG、Naive RAG、Agentic RAG 的区别

很多人会把这些概念混在一起。

可以这样理解：

| 架构           | 核心问题        | 主要能力                                                   | 典型适用场景          |
| ------------ | ----------- | ------------------------------------------------------ | --------------- |
| Naive RAG    | 模型不知道外部知识   | 找相似 chunk                                              | 简单事实问答          |
| Advanced RAG | 检索质量不够好     | Query Rewrite、Hybrid Search、Rerank、Context Compression | 生产级知识库问答        |
| Graph RAG    | 知识缺少结构      | 实体、关系、社区、全局总结                                          | 多跳关系、全局主题、复杂归因  |
| Agentic RAG  | 系统不知道如何获取知识 | 规划、工具选择、多轮检索、验证                                        | 复杂任务、多源数据、研究型问题 |

Graph RAG 和 Agentic RAG 不是替代关系。

Graph RAG 更偏“知识组织方式”：

```
把数据组织成图和社区摘要
```

Agentic RAG 更偏“控制流程”：

```
让 Agent 决定什么时候检索、检索哪里、是否继续追问、如何验证答案
```

二者可以组合。一个 Agentic RAG 系统的工具列表里，可以同时有：

```
vector_search()
keyword_search()
graph_local_search()
graph_global_search()
sql_query()
web_search()
rerank()
fact_check()
```

Agent 负责判断走哪条路，Graph RAG 负责提供结构化知识入口。

#### 7 Graph RAG 的工程难点

Graph RAG 听起来很美，但落地并不轻松。

它的问题主要不在“能不能跑通 demo”，而在“能不能稳定、低成本、可追溯地维护一张有用的图”。

**7.1 索引成本高**

Graph RAG 的离线索引通常需要大量 LLM 调用：抽实体、抽关系、抽 claims、生成社区报告。

这比普通 embedding 入库贵很多。

所以实际落地时通常要：

1. 从小数据集开始，不要一上来索引全公司文档；
2. 做 LLM cache，避免重复抽取；
3. 区分高价值文档和低价值文档；
4. 对低价值文档使用便宜模型或轻量抽取；
5. 对 prompt 做领域适配，不要直接套默认模板。

Graph RAG 官方仓库也提醒过：索引可能消耗较多 LLM 资源，应该先从小规模数据开始。

**7.2 图会被脏实体污染**

如果实体抽取质量差，图会很快变脏。

常见问题包括：

```
同义实体没有合并
无意义实体太多
实体粒度不一致
临时短语被当成实体
关系方向混乱
关系描述过于泛化
```

比如“权限系统”“权限模块”“RBAC”“权限配置”到底是不是一个实体，要根据业务语境决定。

图不是越大越好。图越脏，检索越难，摘要越容易失真。

**7.3 关系可能幻觉**

LLM 抽取关系时，可能会把原文没有明确支持的关系补出来。

比如原文只是说：

```
客户 A 在续约前提到了竞品 B。
```

模型可能抽成：

```
客户 A -> 转向 -> 竞品 B
```

但“提到竞品”不等于“转向竞品”。

所以生产系统里，关系最好要有：

```
source_text
document_id
confidence
extraction_prompt_version
created_at
```

并且最终答案必须能回溯到原文证据。

**7.4 更新和权限比普通 RAG 更复杂**

普通 RAG 更新一个 chunk，重新 embedding 就行。

Graph RAG 更新一批文档，可能影响：

```
实体
关系
社区结构
社区报告
实体 embedding
报告 embedding
图统计特征
```

如果文档频繁变化，维护完整 Graph RAG 的成本会很高。

权限也是一样。普通 RAG 可以在 chunk 级别做权限过滤；Graph RAG 还要考虑社区报告是否混入了用户无权查看的信息。

如果一个社区报告总结了多个部门的文档，那么不同权限的用户能不能看到这个报告？这是一个非常实际的问题。

**7.5 评估更难**

普通 RAG 可以评估：

```
Recall@K
MRR
nDCG
答案忠实度
引用准确性
```

Graph RAG 还要额外评估：

```
实体抽取是否正确
关系抽取是否正确
社区划分是否合理
社区报告是否忠实
全局答案是否覆盖主要主题
答案是否引用到了真实证据
```

尤其是全局问题，往往没有标准答案，只能用人工评审、LLM-as-judge、成对比较等方式评估。这会带来新的偏差。

因此，Graph RAG 的评估不能只看“答案看起来更丰富”，还要看它是否真的更准确、更可追溯。

#### 8 什么时候该用 Graph RAG？

适合用 Graph RAG 的场景：

```
1. 问题经常涉及多跳关系
2. 用户需要整体总结和主题分析
3. 数据来自大量非结构化文本
4. 文档之间存在大量实体复用
5. 答案需要解释“为什么”和“关联是什么”
6. 组织愿意为高质量索引支付额外成本
```

典型场景：

```
客户流失归因
投研报告分析
舆情主题发现
安全事件分析
科研文献综述
企业知识治理
法律/合同风险分析
医疗病例或指南关系分析
```

不适合用 Graph RAG 的场景：

```
1. 文档数量很小
2. 问题主要是简单事实定位
3. 数据变化极其频繁
4. 没有明显实体和关系
5. 成本、延迟比答案深度更重要
6. 无法接受 LLM 抽取带来的不确定性
```

如果只是做 FAQ 问答，Graph RAG 大概率过重。

#### 9 一个生产级 Graph RAG 架构

一个更完整的架构大概是这样：

```
离线索引链路：

Document Ingestion
  -> Parse / Clean
  -> Chunk into TextUnits
  -> Entity Extraction
  -> Relationship / Claim Extraction
  -> Entity Normalization
  -> Graph Construction
  -> Community Detection
  -> Community Report Generation
  -> Embedding
  -> Store Graph / Vector / Reports / Metadata


在线查询链路：

User Query
  -> Intent Classification
  -> Query Rewrite
  -> Choose Search Mode
       - Basic Search
       - Local Search
       - Global Search
       - DRIFT Search
  -> Build Context
  -> Generate Answer
  -> Attach Citations
  -> Log Trace
  -> Evaluate / Feedback
```

最关键的是 `Choose Search Mode`。

不要让所有问题都走 Graph RAG。更合理的做法是：

```
简单事实问题 -> Basic / Hybrid Search
实体相关问题 -> Local Search
整体总结问题 -> Global Search
既要整体又要细节 -> DRIFT Search
复杂多源任务 -> Agentic RAG 调度多个工具
```

好的 RAG 系统不是某一种方法打天下，而是根据问题选择合适的上下文构造方式。

#### 10 Graph RAG 的发展方向

Graph RAG 的一个明显问题是成本。

完整 Graph RAG 需要在索引阶段做大量 LLM 抽取和摘要，所以后续研究主要围绕两个方向优化。

**10.1 Dynamic Community Selection**

早期 Global Search 会在某个固定社区层级上处理大量社区报告，这样很全面，但成本高。

Dynamic Community Selection 的思路是：不要一开始就把所有社区报告送进 map-reduce，而是从高层社区开始，让较便宜的模型判断某个社区和问题是否相关。如果不相关，就剪掉这个社区及其子社区；如果相关，再向下探索。

它的本质是：

```
先判断相关性，再决定是否深入
```

这能减少无关社区报告进入后续生成流程，从而降低成本。

**10.2 LazyGraphRAG**

LazyGraphRAG 更进一步：它试图把 Graph RAG 和 vector RAG 结合起来，并推迟 LLM 的使用。

完整 Graph RAG 是“先花很多成本建好图和社区摘要，再查询”。

LazyGraphRAG 则更像：

```
先用轻量方法建立概念图
查询时再根据问题逐步深入
只对相关部分使用 LLM
```

它的方向很值得关注，因为生产环境里最常见的矛盾就是：我们想要 Graph RAG 的全局理解能力，但又不想承担完整图索引的高成本。

不过从工程角度看，不要因为看到了 LazyGraphRAG 就立刻否定完整 Graph RAG。完整 Graph RAG 的社区报告本身也有价值：它可以被人阅读、审计、分享，并作为数据资产长期存在。

#### 11 总结

Graph RAG 的核心不是“用了图”，而是把 RAG 的上下文工程从文本片段级别推进到了语义结构级别。

普通 RAG 问的是：

```
哪些 chunk 和问题最相似？
```

Graph RAG 问的是：

```
这个语料里有哪些实体？
实体之间有什么关系？
这些关系形成了哪些主题社区？
用户的问题应该从哪个层级理解？
答案需要哪些局部证据和全局摘要？
```

所以，Graph RAG 最适合解决的不是“找到某个答案”，而是“理解一批文档”。

一句话总结：

> **Graph RAG 把 RAG 从相似片段检索系统，升级成了可总结、可遍历、可追溯的语义地图。**

#### 参考

1. [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)
2. [Microsoft GraphRAG Docs](https://microsoft.github.io/graphrag/)
3. [Microsoft GraphRAG GitHub](https://github.com/microsoft/graphrag)
4. [GraphRAG Global Search](https://microsoft.github.io/graphrag/query/global_search/)
5. [GraphRAG Local Search](https://microsoft.github.io/graphrag/query/local_search/)
6. [GraphRAG DRIFT Search](https://microsoft.github.io/graphrag/query/drift_search/)
7. [GraphRAG: Improving global search via dynamic community selection](https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/)
8. [LazyGraphRAG: Setting a new standard for quality and cost](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
