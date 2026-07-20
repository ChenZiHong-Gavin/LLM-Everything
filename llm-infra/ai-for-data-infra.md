# AI for data infra

明白，我把常见的技术术语保留成英文——像 pipeline、catalog、schema、lineage、hooks、agent、PR、backfill 这类词，中文数据圈本来也大多直接说英文。下面是调整后的版本：

***

## 面向数据基础设施的 AI：2026 数据工程 Agent 完全指南

**AI for Data Infra**（面向数据基础设施的 AI）是一种工程实践——让自主 agent 直接运行在数据平台内部（pipeline、catalog、data quality、governance、incident），由 agent 来完成过去平台团队手动处理的工作。它取代的是那些"跟你的数据聊天"式的玩具产品，转而构建真正用于生产环境的 agent 集群，底层依托 Claude Code、MCP 以及现代 lakehouse 架构。

如果你曾尝试用 LLM 处理数据，却被幻觉 SQL、被搞崩的 pipeline、或是看不到你 catalog 的 agent 折腾得够呛，那么这份指南就是面向下一代技术的实战手册。它涵盖四层工程体系、MCP 协议、作为 runtime 的 Claude Code、与 Snowflake / Databricks / dbt 的集成、合规与 governance、评估方法，以及 Data Workers 如何用开源方式实现其中每一环。

读完之后，你会知道如何搭建一套能安全落地生产、不至于把系统搞垮的 AI-for-data-infra 技术栈，如何评估厂商的宣传话术，以及如何把 Data Workers 的 14-agent 集群映射到你现有的数据平台上。下面这篇约 2500 字的完整讲解是本文核心参考——文中每个链接都指向本资源合集里更深入的专题。

### 什么是 AI for Data Infra？

它是一门专门用于构建和运营自主 agent 的技术，这些 agent 负责管理数据平台本身：pipeline、warehouse、catalog、质量检查、成本控制、迁移以及故障响应。它与"跟数据聊天"的 BI 机器人有本质区别，因为这些 agent 作用于基础设施本身，而非 dashboard——它们会提 PR、执行 migration、呼叫值班人员、回滚失败的部署。

这个品类在 2025 和 2026 年崛起，背后有三股力量交汇推动。第一，LLM 在代码和 SQL 生成上跨过了可靠性门槛。第二，Model Context Protocol（MCP）统一了 agent 访问数据系统的方式。第三，Claude Code 及类似的 coding agent 为平台提供了一个能跨会话记住项目的持久 runtime。三者结合，让 data infra agent 不仅成为可能，更能真正部署上线。

一句话概括：AI for data infra 意味着"用 agent 取代工单"。过去每一次需要开 Jira 卡片的交接——加一列字段、backfill 一张维度表、排查一条 freshness 告警、审批一次 schema 变更——现在都变成了向 agent 发出的一条 prompt，而这个 agent 具备完成任务所需的 context、工具和安全护栏。agent 并不是取代数据团队，而是取代数据团队原本要一件件处理的那个任务队列。

### 为什么早期"跟数据聊天"的 agent 失败了

第一代数据 agent——Text2SQL 演示、自然语言 BI、跟 warehouse 对话——在 2023 和 2024 年陆续推出，几乎无一例外地在实战中远逊于演示效果。一旦深入架构，失败模式其实早有预兆：它们把 warehouse 当成一个扁平的 schema，直接把原始表名喂给 LLM，指望它能猜出背后的业务含义。它猜不出来。

有三个具体的短板扼杀了第一代产品。**Context 短板**：agent 对业务如何使用数据毫无长期记忆。**隐性知识短板**：那些让一条查询正确的规则（该排除哪个 user\_id、哪一列营收是扣除退款后的净额、哪个日期字段才是权威来源）都藏在 Slack 讨论串和资深工程师的脑子里，从未进入 schema。**权威表短板**：大多数 warehouse 里有三张营收表，到底该用哪张取决于问题本身——LLM 选了第一个匹配项，然后就错了。

* 会话之间没有持久的项目记忆——每个问题都从零开始
* 扁平的 schema prompt，忽略了 metric 定义和业务逻辑
* 无法访问 lineage、ownership，也接触不到 catalog 的语义层
* Text-to-SQL 是在 Spider 这类 benchmark 上评估的，而非在脏乱的真实 warehouse 上
* 没有写入路径——agent 只能建议，永远无法真正落地修复

这里的教训不是"LLM 不擅长数据"，而是数据工作高度依赖 context，而第一波产品没有在 context 上下功夫。第二代——也就是我们所说的 AI for data infra——从 context 层出发，再向外扩展。这段历史的详细版本见我们的《Autonomous Data Engineering》入门篇（/resources/autonomous-data-engineering）。

### 面向数据的四层 AI 工程体系

一套用于生产的 AI-for-data-infra 技术栈包含四层，缺任何一层 agent 都会崩。这四层从底到顶依次是：项目记忆（CLAUDE.md）、可复用 skills、执行 hooks、编排 agents。每一层都是你提交到 repo 里的具体产物，而不是某个厂商的功能卖点。

| 层级        | 产物                     | 用途               | 示例                                          |
| --------- | ---------------------- | ---------------- | ------------------------------------------- |
| 1. Memory | CLAUDE.md              | 跨会话的持久项目 context | Warehouse DSN、权威表、品牌口吻、负责人                  |
| 2. Skills | /.claude/skills/\*.md  | agent 可调用的可复用手册  | run-dbt、backfill-dim、on-call-triage         |
| 3. Hooks  | settings.json 里的 hooks | 护栏与自动化           | 未经审批禁止写生产、保存时跑测试                            |
| 4. Agents | Subagent + MCP 工具      | 带工具的自主执行         | pipeline agent、catalog agent、incident agent |

CLAUDE.md 是承重文件。它用持久的项目记忆取代了一次性 prompt——模型在每次会话开始时都会读它，所以你不必每次对话都重新解释一遍你的 warehouse。最好的 CLAUDE.md 文件在 500 到 2000 行之间，读起来像一份给新员工的入职文档：规范、表、SLA、谁负责什么。完整模板见我们面向数据团队的《CLAUDE.md 指南》（/resources/claude-md-data-engineering）。

Skills 是第二层，因为它把操作手册打包起来。一个 skill 就是一个 Markdown 文件，说明如何完成一项重复性任务——跑一次 dbt build、backfill 一张 dimension、triage 一次故障、轮换一个凭证。agent 按名称调用 skill，这意味着不论是人还是 agent 在操作，同一份手册都会以相同方式执行。这正是产出可复现的原因。

Hooks 是第三层，因为它把策略变成了代码。一个 hook 是注册在 settings.json 里的小脚本，会在 agent 动作之前或之后运行：没有 PR 就禁止写生产、每次保存都跑 dbt 测试、agent 一碰 PII 表就往 Slack 发消息。hooks 是你敢把生产交给 agent 的底气所在——不是因为 agent 完美无缺，而是因为 hooks 能兜住那 1% 的失误。

Agents 是最顶层。每个 agent 都是一个 Claude Code subagent，拥有一套受限的工具集：pipeline agent 有 dbt、Airflow 和 git；catalog agent 有 OpenMetadata 和 DataHub；cost agent 有 Snowflake 查询历史和 Databricks 账单。agent 之间通过 MCP 相互调用来协同，因此一次 pipeline 故障会触发 incident agent，后者会呼叫 owner 并开出工单——中间无需人工介入。

### Context Engineering：Prompt Engineering 的继任者

Prompt engineering 把 LLM 当成一个需要你去"骗"它听话的黑盒；context engineering 则把它当成一个需要你好好带教的新员工。这个转变，是从"巧妙的句子"转向"持久的产物"——是 agent 每次会话都会读的文件，而不是你每次都要重新想一遍的 prompt。

面向 data infra 的 context 栈有四个面：**code plane**（repo 内容）、**data plane**（schema、样本行、lineage）、**runbook plane**（incident、SLA、ownership）、**history plane**（过往决策、上次 migration 为何失败）。只看得到 code plane 的 agent，会写出语法正确但语义错误的 SQL；能看到全部四个面的 agent，写出的 SQL 会是你的资深工程师愿意上线的水平。

* **Code plane**——repo 文件、dbt 模型、Airflow DAG、Terraform
* **Data plane**——schema、样本行、lineage 图、metric 定义
* **Runbook plane**——incident 历史、SLA、on-call 轮换、升级路径
* **History plane**——决策日志、过往 migration 记录、review 意见
* **Human plane**——Slack 对话、设计文档、PRD 存档

Data Workers 会自动构建这套 context 栈——catalog agent 爬取元数据，observability agent 构建 lineage，insights agent 为决策历史建立索引，一切都通过 MCP 工具对外呈现。完整架构见我们的《Context Engineering 手册》（/resources/context-engineering-for-data-agents）。

### 面向数据的多 Agent 技术部门

一个"包打天下"的单一 agent 在规模化时必然崩溃。真正能在生产中跑通的替代方案，是一个多 agent 的"技术部门"：由多个各管一摊的专门 subagent 组成，再由一个 planning agent 来统一协调。这种模式模仿的正是真实的工程组织：Architect 思考这次变更，Builder 去实现它，Reviewer 去检查它，Release agent 负责上线。

数据工作中的三个核心角色是 **Architect**、**Builder** 和 **Reviewer**。Architect agent 规划变更（新增一列、一张表、一条 pipeline）并撰写技术 spec。Builder agent 实现该 spec——提一个包含 dbt 模型、测试、文档和 catalog 条目的 PR。Reviewer agent 运行测试、检查 lineage 影响、标记 SLA 风险，然后要么批准、要么打回要求修改。只有当 Reviewer 升级上报时，人类才会介入。

| Subagent  | 负责    | 工具                              | 产出         |
| --------- | ----- | ------------------------------- | ---------- |
| Architect | 规划与设计 | catalog、lineage、决策日志            | 技术 spec 文档 |
| Builder   | 实现    | repo、dbt、git、MCP writer         | 带测试的 PR    |
| Reviewer  | 质量关卡  | test runner、lineage diff、SLA 检查 | 批准或要求修改    |
| Release   | 生产部署  | CI/CD、rollback、监控               | 部署或回滚      |

这套模式可以直接映射到 Data Workers 的 14 个 agent 上：pipeline、incident、catalog、schema、quality、governance、cost、migration、insights、observability、streaming、orchestration、connectors、usage-intelligence。每一个都可以根据任务扮演 Architect、Builder 或 Reviewer 的角色。完整映射见我们的《多 Agent 编排模式指南》（/resources/multi-agent-orchestration-data）。

### MCP 层：面向数据领域的 Model Context Protocol

MCP（Model Context Protocol）是让 agent 调用数据系统、又不被厂商锁定的接口。在 MCP 之前，每个 agent 针对每个系统都有自己的 connector，意味着每接一个新 warehouse 就要写一个新集成。有了 MCP 之后，一个 warehouse 只需暴露一个 MCP server，所有会说 MCP 的 agent 都能用它。这个类比很直白：MCP 之于数据，正如 LSP 之于 IDE。

对于 AI for data infra，MCP 之所以是承重协议，是因为它统一了三件事：tool catalog（agent 能做什么）、resource catalog（agent 能看到什么数据）、以及授权模型（这个 agent 被允许碰什么）。没有 MCP，你最后只能把 agent 用胶带硬贴到各种 API 上；有了 MCP，你得到的是一套可移植、可审计、可按 tier 设卡的工具集。

* **Tools**——可执行的操作（跑 query、开 PR、触发 dbt）
* **Resources**——可读取的 context（schema、文档、lineage、metric）
* **Prompts**——server 提供给 agent 的可复用模板
* **Sampling**——由 server 发起的 LLM 调用，用于多步工作流
* **Authorization**——OAuth 2.1 + tier 分级设卡（community、pro、enterprise）

Data Workers 在 14 个 agent 上提供了 212 多个 MCP 工具，并通过单个 Claude Code 插件统一暴露。这些工具在框架层面按 tier 设卡，因此 community 用户看到的是一套，enterprise 用户看到的是另一套，而 PII middleware 加审计日志则挡在每一次调用的前面。深入内容见我们的《面向数据工程师的 MCP 指南》（/resources/mcp-data-engineering-guide）和《MCP Server 对比》（/resources/mcp-server-comparison-data）。

### Claude Code 作为编排引擎

Claude Code 是让上述一切在实践中跑起来的 runtime。它是一个 terminal 原生的 agent，可运行在工程师的笔记本或 CI 里，会读取 CLAUDE.md、加载 skills、执行 hooks、编排 subagent。它之所以成为 AI for data infra 的默认 runtime，关键在于持久性——会话可恢复、记忆能留存，昨天上线一次变更的那个 agent，今天照样能上线新的变更。

其他候选方案——定制的 LangChain 脚本、自研 agent 框架、厂商托管的 UI——都栽在同一个点上：它们把每次 agent 运行都当成一次全新对话。而数据工作往往是长周期的：一次 migration 要好几周，一次 schema rollout 要好几天，一次 incident 排查可能跨越多个 on-call 班次。你需要一个能记事的 runtime。Claude Code 记得住。

Data Workers 是 Claude Code 原生的——14 个 agent 以 Claude Code 插件形式安装，skills 放在 .claude/skills，hooks 放在 settings.json，CLAUDE.md 则是项目 context。详见我们的《面向数据工程师的 Claude Code 指南》（/resources/claude-code-data-engineering）以及《Claude Code 对比 LangChain deep agents》（/resources/dataworkers-vs-langchain-deep-agents）。

### 与 Snowflake、Databricks 和 dbt 的集成

对 AI for data infra 来说，最要紧的三个系统是 Snowflake、Databricks 和 dbt——真正的工作就在这里发生。agent 层必须同时集成这三者，又不能自己变成一层新的 lock-in。正确的做法是每个系统配一个 MCP server，让 agent 框架充当 client。

| 系统         | MCP server     | Agent 能力                          | 护栏                    |
| ---------- | -------------- | --------------------------------- | --------------------- |
| Snowflake  | snowflake-mcp  | query、schema、cost、RBAC            | 行级访问、成本预算、masking     |
| Databricks | databricks-mcp | SQL warehouse、Unity Catalog、job   | 集群预算、Unity ACL、PII 扫描 |
| dbt        | dbt-mcp        | 模型、测试、文档、lineage                  | CI 强制执行、生产分支保护        |
| BigQuery   | bq-mcp         | query、INFORMATION\_SCHEMA、billing | slot 预算、dataset ACL   |
| Iceberg    | iceberg-mcp    | 表、snapshot、compaction             | 分支保护、snapshot 保留      |

Data Workers 对这五者都提供了 first-party 集成。实现细节见《Claude Code Snowflake 集成指南》（/resources/claude-code-snowflake-integration-guide）、《Databricks Unity Catalog Agent 指南》（/resources/databricks-unity-catalog-agent）、《dbt Cloud AI Agent 指南》（/resources/dbt-cloud-ai-agent）以及《BigQuery 自主 Agent 指南》（/resources/bigquery-autonomous-agent-guide）。

### 合规与 Governance Agent

任何碰生产数据的 agent，迟早都会碰到受监管的数据。这不是你要去规避的 bug，而是你要提前规划的设计约束。在 AI for data infra 里，合规与 governance 不是事后补上的一层，而是横在每一次 agent 调用和每一个数据系统之间的 middleware。PII 检测、审计日志、策略执行、region pinning，全都在 agent 看到任何一行数据之前就先跑一遍。

2026 年最要紧的四套监管体系是 GDPR（欧盟）、《欧盟人工智能法案》（针对高风险 AI 系统）、BCBS 239（银行业风险汇总），以及美国各州拼凑的隐私法（CCPA、CPRA 等等）。一套合规的 AI-for-data-infra 技术栈，必须对每一次请求回答三个问题：agent 看到了什么数据、它做了什么、是谁授权的。如果你在审计中无法把这三个问题都答清楚，那你手里就不是合规，而是一笔隐患。

* **PII middleware**——在敏感字段到达 LLM 之前检测并 mask
* **防篡改审计日志**——每一个 agent 动作的 SHA-256 hash chain
* **带 JWT 的 OAuth 2.1**——对每次请求做机器可验证的授权
* **Region pinning**——把欧盟数据留在欧盟、美国数据留在美国
* **Policy as code**——agent 无法绕过的 Open Policy Agent 规则

Data Workers 把这五项都作为 core/enterprise middleware，接入了每一个 MCP agent。针对各监管体系的细节，见《面向数据 Agent 的 AI Governance 指南》（/resources/ai-governance-data-agents）、《欧盟 AI 法案合规手册》（/resources/eu-ai-act-data-agents）和《用 AI Agent 满足 BCBS 239 指南》（/resources/bcbs-239-ai-agents）。

### 开发者效率与 Human-in-the-Loop

AI for data infra 的效率提升故事并不是 10x。对于 agent 能自主完成的工作，它更接近 3x 到 5x；而对于它做不了的工作，则是 0x——所以真正的杠杆在于分清这两类工作，并让人集中在后者。insights agent 是大多数团队会忽略的一环：它盯着其他 agent 在做什么，把需要人工 review 的决策标记出来，让团队知识不断累积，而不是被消耗掉。

Human-in-the-loop 是关键枢纽。它有四种模式：**完全自主**（低风险、可逆的动作，比如更新文档）、**部署前 review**（PR）、**逐动作审批**（写生产）、以及**有人监督**（人主导、agent 建议）。一套生产技术栈会同时用上这四种，按任务类别切换模式。要是把所有动作都塞进同一个关卡，agent 要么变得没用，要么变得危险。

详见《Human-in-the-Loop 数据 Agent 指南》（/resources/human-in-the-loop-data-agents）、《AI Agent 开发者效率报告》（/resources/developer-productivity-ai-agents）以及《Insights Agent 深度解析》（/resources/insights-agent-deep-dive）里的效率账。

### 评估：Agent-as-a-Judge 与决策追踪 Context Graph

评估 agent 比评估模型更难。模型可以用固定 benchmark 打分；而 agent 面对的是开放式任务，有很多种合理结果。生产中真正管用的有两种方法：**Agent-as-a-Judge**（用第二个 agent 给第一个打分），以及**决策追踪 context graph**（记录 agent 做出的每一个决策，之后可以 replay）。两者都无法取代人，但结合起来，它们抓 regression 的速度比任何测试套件都快。

* **Golden queries**——200 条已知正确的 prompt，附带预期输出
* **Agent-as-a-Judge**——一个打分 agent 审查每一个生产动作
* **Decision trace**——存下每一次 tool call 的输入、输出和理由
* **Replay harness**——用今天的 agent 重跑昨天的 incident
* **人工抽检**——抽取 1% 的 agent 动作做人工复核

Data Workers 为 catalog agent 提供了一套 200 条 golden query 的评估套件，并为 pipeline、quality、incident agent 提供了 Agent-as-a-Judge 评测框架。方法论见我们的《数据工程 Agent 评估手册》（/resources/agent-evaluation-data-engineering）和《Agent-as-a-Judge 指南》（/resources/agent-as-a-judge-data）。

### Data Workers 如何实现 AI for Data Infra

Data Workers 是本指南所述技术栈的开源参考实现。它提供 14 个专门 agent、212 多个 MCP 工具、CLAUDE.md / Skills / Hooks / Agents 的四层架构、企业级 PII 与审计 middleware，以及对 Snowflake、Databricks、dbt、BigQuery 和 Iceberg 的 first-party 集成。它运行在 Claude Code 上，以插件形式安装，community tier 完全免费。

这 14 个 agent 分别是：pipeline、incident、catalog、schema、quality、governance、cost、migration、insights、observability、streaming、orchestration、connectors、usage-intelligence。每个 agent 都按 tier 设卡（community、pro、enterprise），每个工具都被审计，每个动作都可逆。repo 在 GitHub 上，文档在 dataworkers.io/docs，社区在 Discord。

如果你正在评估厂商，有三个问题要问：它是否运行在 Claude Code 上、是否暴露 MCP、agent 代码是否开源。Data Workers 是唯一对这三点都回答"是"的技术栈。详见《Data Workers 架构概览》（/resources/dataworkers-architecture-overview）、《14 Agent 参考》（/resources/data-workers-14-agents）和《开源数据 Agent 对比》（/resources/open-source-data-agents-comparison）。

### 常见问题

**AI for data infra 和"跟数据聊天"有什么区别？** "跟数据聊天"是让人用自然语言对着 dashboard 提问。AI for data infra 则是运行自主 agent 去管理底层平台——pipeline、catalog、quality、incident。前者是个 BI 功能；后者是一门工程实践。chatbot 只读，agent 会写。

**AI for data infra 能上生产了吗？** 对合适的任务，能。对于 schema 变更、backfill、文档生成、catalog 同步、成本优化和 incident triage，agent 已经可以上生产。但它还不适合从零做架构设计，也不适合在含糊的业务逻辑上做判断。让 agent 去做那 80% 可重复的工作，把那 20% 不可重复的留给人。

**用 AI for data infra 一定要用 Claude Code 吗？** 严格说不是，但实际上基本是。Claude Code 是那个能把 CLAUDE.md、skills、hooks 和 subagent 拧成一个系统的持久 runtime。其他 runtime 也有，但在持久记忆和 MCP 集成上没有能比得上 Claude Code 的。Data Workers 是 Claude Code 原生的——你要是换别的 runtime，就得把这些底层管道重新实现一遍。

**这和 LangChain 或 CrewAI 有什么不同？** LangChain 和 CrewAI 是通用的 agent 框架。AI for data infra 是一个垂直方向——开箱即带有观点鲜明的 agent、数据专用的 MCP 工具和合规 middleware。你当然可以在 LangChain 上把它搭出来，但那等于重写一遍 Data Workers 已经提供好的东西。详见《Data Workers 对比 LangChain deep agents》（/resources/dataworkers-vs-langchain-deep-agents）。

**要花多少钱？** Data Workers community tier 免费且开源。pro 和 enterprise tier 增加了托管 middleware、SSO 和 SLA。agent 推理成本取决于你选的模型——配合 prompt caching 和 Claude Code 的会话复用，重度日常使用下，典型数据团队每月在模型调用上大约花 50 到 500 美元。

**agent 能碰生产吗？** 能，配合 hooks 就行。模式是：agent 开 PR，CI 跑测试，一个人（或 reviewer agent）批准，CI 部署。agent 绝不会不经关卡就直接写生产。对于低风险动作（更新文档、catalog 同步），agent 可以完全自主运行。按动作类别去调关卡的松紧就好。

**在这个领域里，我该怎么评估厂商的宣传？** 问四个数：agent 数量、MCP 工具数量、测试数量，以及 repo 地址。如果厂商这几样都拿不出来，那就是营销，不是工程。Data Workers 这四样都公开：14 个 agent、212 多个工具、3342 多个测试、github.com/DhanushAShetty/data-workers。

**它能配我的 warehouse 用吗？** 如果你的 warehouse 是 Snowflake、Databricks、BigQuery、Redshift 或 Iceberg lakehouse，能。Data Workers 还支持 Postgres、MySQL、Trino、DuckDB 以及 35 多个企业 connector。完整清单见《Connector 指南》（/resources/data-workers-connectors）。

**部署要多久？** 从全新 clone repo 到第一次 MCP tool call，不到 60 秒。而一次完整的生产上线——配好 CLAUDE.md、skills、hooks 和 14 个 agent——一个资深工程师专心搞一个下午就能完成。分步说明见《部署指南》（/resources/data-workers-deployment-guide）。

**我的数据会发给 Anthropic 吗？** 只有 agent 完成任务所需的数据会发出去，而且如果你开启了 PII masking，还会先脱敏。Data Workers 的 PII middleware 会在每次 LLM 调用之前运行，region pinning 则把欧盟数据留在欧盟。政策详见《LLM 数据披露说明》（/resources/llm-data-disclosure）。

AI for data infra 是数据工程的下一个十年，而这套打法已经在生产中跑起来了。先从 CLAUDE.md 起步，再叠加 skills 和 hooks，部署 Data Workers 的 14 个 agent，让这个 agent 集群去运营平台，你的团队则专注于那些 agent 做不了的判断题。想看这套完整技术栈在你自己 warehouse 上跑起来的样子，可以预约一次演示（/book-demo）。



### 参考

1. [https://dataworkers.io/resources/ai-for-data-infra/](https://dataworkers.io/resources/ai-for-data-infra/)

