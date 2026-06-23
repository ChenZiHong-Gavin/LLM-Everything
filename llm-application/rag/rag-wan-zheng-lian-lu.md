# RAG 完整链路

第一节：**为什么很多 RAG demo 看起来能用，实际上不能用**\
讲典型问题：召回不到、召回太多、上下文污染、答案幻觉、引用不准、成本暴涨。

第二节：**Naive RAG 的最小闭环**\
文档切分 → Embedding → 向量库 → Top-K 检索 → 拼接 Prompt → LLM 生成。这里可以配一张流程图。

第三节：**RAG 的核心不是“向量库”，而是“上下文工程”**\
重点讲 chunk size、overlap、metadata、query rewrite、hybrid search、rerank、context compression。

第四节：**为什么 RAG 需要评估**\
把评估拆成四类：检索评估、生成评估、端到端评估、线上监控。AI Engineering Report 2025 也提到，AI 系统进入生产后需要监控与可观测性，受访者常结合标准 observability、离线 eval 和人工 review 来评估系统表现。

第五节：**什么时候普通 RAG 不够，需要 Graph RAG**\
普通 RAG 擅长回答局部事实问题，例如“某个合同第几条是什么”。但对于“这批文档的主要主题是什么”“多个事件之间有什么关系”这类全局问题，Graph RAG 更合适。Microsoft Research 的 GraphRAG 论文就指出，传统 RAG 对整个语料的全局问题并不擅长，而 GraphRAG 试图通过图结构和社区摘要来处理这类 query-focused summarization 问题。

第六节：**生产级 RAG 的架构图**\
可以画成：

用户问题\
→ Query Rewrite\
→ Hybrid Retrieval\
→ Reranker\
→ Context Builder\
→ LLM\
→ Citation / Verification\
→ Evaluation / Logging

第七节：**总结：RAG 的本质是把“不确定的生成”变成“可追踪的回答”**
