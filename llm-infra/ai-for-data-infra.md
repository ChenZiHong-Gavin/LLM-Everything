# AI for data infra

**AI for Data Infra 让** agent 可以直接运行在数据平台内部（pipeline、catalog、data quality、governance、incident），由 agent 来完成过去团队手动处理的工作。它取代的是那些聊天式的玩具产品，转而构建真正用于生产环境的 agent 集群，底层依托 Claude Code、MCP 以及现代 lakehouse 架构。

这篇文章讲的是如何搭建一套能安全落地生产的 AI-for-data-infra 技术栈。

### 1 什么是 AI for Data Infra？

它是一门构建 agent 的技术，这些 agent 负责管理数据平台本身：pipeline、warehouse、catalog、质量检查、成本控制、迁移以及故障响应。它与聊天式的 BI 机器人有本质区别，因为这些 agent 有自主行动能力——它们会提 PR、执行 migration、呼叫值班人员、部署等等。

能做成这件事，有几个前提：

1. LLM 在代码生成上已然非常可靠。
2. MCP 统一了 agent 访问数据系统的方式。
3. Claude Code 及类似的 coding agent 为平台提供了一个能跨会话记住项目的持久 runtime

三者结合，让 data infra agent 不仅成为可能，更能真正部署上线。

### 2 为什么早期聊天式的数据 agent 失败了

2023到2024年，各个公司陆续推出了第一代 data agent，大部分是 Text2SQL、自然语言 BI 等的 demo。但是无一例外，它们的演示价值大于实际使用价值。我们深入它们的架构，就能看到失败模式其实早有预兆：它们把 warehouse 当成一个扁平的 schema，直接把原始表名喂给 LLM，指望它们能猜出背后的业务含义。结果就是它们猜不出来。

**有三个致命弱点：**

1. **Context 短板**：agent 对业务如何使用数据毫无长期记忆。
2. **隐性知识短板**：一些和查询规则相关的知识（例如该排除哪个 user\_id、哪一列营收是扣除退款后的净额、哪个日期字段才是权威来源）都藏在聊天讨论和工程师的脑子里，没有被文档化。
3. **权威表短板**：大多数 warehouse 里有三张以上营收表，到底该用哪张取决于问题本身。

还有若干个虽不致命但也很麻烦的弱点：

1. 会话之间没有持久的项目记忆
2. 扁平的 schema prompt，忽略了 metric 定义和业务逻辑
3. 无法访问 lineage、ownership，也接触不到 catalog 的语义层
4. Text-to-SQL 是在 Spider 这类 benchmark 上评估的，而非在脏乱的真实 warehouse 上
5. 没有写入权限——agent 只能建议，永远无法真正落地修复

LLM 并非不擅长数据，而是数据工作高度依赖 context，而第一代的玩具 agent 没有在 context 上下功夫。第二代——也就是我们所说的 AI for data infra——从 context 层出发，再向外扩展。

### 3 面向数据的四层 AI 工程体系

一套用于生产的 AI-for-data-infra 技术栈包含四层，缺任何一层 agent 都会崩。这四层自下而上依次是：项目记忆（CLAUDE.md）、skills、执行 hooks、编排 agents。

| 层级        | 产物                     | 用途               | 示例                                          |
| --------- | ---------------------- | ---------------- | ------------------------------------------- |
| 1. Memory | CLAUDE.md              | 跨会话的持久项目 context | Warehouse DSN、权威表、负责人                       |
| 2. Skills | /.claude/skills/\*.md  | agent 可调用的可复用手册  | run-dbt、backfill-dim、on-call-triage         |
| 3. Hooks  | settings.json 里的 hooks | 护栏与自动化           | 未经审批禁止写生产、保存时跑测试                            |
| 4. Agents | Subagent + MCP 工具      | 带工具的自主执行         | pipeline agent、catalog agent、incident agent |

CLAUDE.md 用持久的项目记忆取代了一次性 prompt。模型在每次会话开始时都会读它，所以你不必每次对话都重新解释一遍你的 warehouse。CLAUDE.md 文件最好在 500 到 2000 行之间，类似一份给新员工的入职文档：规范、表、SLA、谁负责什么。

Skills 是第二层。一个 skill 就是一个 Markdown 文件，说明如何完成一项重复性任务，例如跑一次 dbt build、backfill 一张 dimension、triage 一次故障、轮换一个凭证。agent 按名称调用 skill，这意味着不论是人还是 agent 在操作，同一份手册都会以相同方式执行。这也是产出可复现的原因。

Hooks 是第三层，因为它把策略变成了代码。一个 hook 是注册在 settings.json 里的小脚本，会在 agent 动作之前或之后运行：没有 PR 就禁止写生产、每次保存都跑 dbt 测试、agent 一碰 PII 表就往 Slack 发消息。有了 hooks 兜底，你就可以放心把生产任务交给 agent。

Agents 是最顶层。每个 agent 都是一个 Claude Code subagent，拥有一套受限的工具集：pipeline agent 有 dbt、Airflow 和 git；catalog agent 有 OpenMetadata 和 DataHub；cost agent 有 Snowflake 查询历史和 Databricks 账单。agent 之间通过 MCP 相互调用来协同，因此一次 pipeline 故障会触发 incident agent，后者会呼叫 owner 并开出工单——中间无需人工介入。

### 4 Context Engineering

面向 data infra 的 context engineering需要涵盖 4 个方面：

* **Code plane**——repo 文件、dbt 模型、Airflow DAG、Terraform
* **Data plane**——schema、样本行、lineage 图、metric 定义
* **Runbook plane**——incident 历史、SLA、on-call 轮换、升级路径
* **History plane**——决策日志、过往 migration 记录、review 意见
* **Human plane**——Slack 对话、设计文档、PRD 存档

子 agent 会自动构建这套 context 栈——catalog agent 爬取元数据，observability agent 构建 lineage，insights agent 为决策历史建立索引，一切都通过 MCP 工具对外呈现。

### 5 Multi-agent

单个 agent 在规模化时必然崩溃。真正能在生产中跑通的替代方案，是一个 multi agent 的"技术部门"：由多个各司其职的 subagent 组成，再由一个 planning agent 来统一协调。这种模式模仿的是真实的工程组织：Architect 思考这次变更，Builder 去实现它，Reviewer 去检查它，Release agent 负责上线。

数据工作中的三个核心角色是 **Architect**、**Builder** 和 **Reviewer**。

* Architect agent 规划变更（新增一列、一张表、一条 pipeline）并撰写技术 spec
* Builder agent 实现该 spec——提一个包含 dbt 模型、测试、文档和 catalog 条目的 PR。
* Reviewer agent 运行测试、检查 lineage 影响、标记 SLA 风险，然后要么批准、要么打回要求修改。只有当 Reviewer 升级上报时，人类才会介入。

| Subagent  | 负责    | 工具                              | 产出         |
| --------- | ----- | ------------------------------- | ---------- |
| Architect | 规划与设计 | catalog、lineage、决策日志            | 技术 spec 文档 |
| Builder   | 实现    | repo、dbt、git、MCP writer         | 带测试的 PR    |
| Reviewer  | 质量关卡  | test runner、lineage diff、SLA 检查 | 批准或要求修改    |
| Release   | 生产部署  | CI/CD、rollback、监控               | 部署或回滚      |

综上，可以有14 个 agent：pipeline、incident、catalog、schema、quality、governance、cost、migration、insights、observability、streaming、orchestration、connectors、usage-intelligence。每一个都可以根据任务扮演 Architect、Builder 或 Reviewer 的角色。

### 6 MCP

MCP 是让 agent 调用数据系统的接口。在 MCP 之前，每个 agent 针对每个系统都有自己的 connector，意味着每接一个新 warehouse 就要写一个新集成。有了 MCP 之后，一个 warehouse 只需暴露一个 MCP server，所有会说 MCP 的 agent 都能用它。

对于 AI for data infra，MCP 统一了三件事：tool catalog（agent 能做什么）、resource catalog（agent 能看到什么数据）、以及授权模型（这个 agent 被允许碰什么）。没有 MCP，你只能在 agent 流程内部用各种 API 实现；有了 MCP，你得到的是一套可移植、可审计的工具集。

* **Tools**——可执行的操作（跑 query、开 PR、触发 dbt）
* **Resources**——可读取的 context（schema、文档、lineage、metric）
* **Prompts**——server 提供给 agent 的可复用模板
* **Sampling**——由 server 发起的 LLM 调用，用于多步工作流
* **Authorization**——OAuth 2.1 + tier 分级设卡（community、pro、enterprise）

### 7 Claude Code 作为编排引擎

Claude Code 是让上述一切在实践中跑起来的 runtime。它是一个 terminal 原生的 agent，可运行在工程师的笔记本或 CI 里，会读取 CLAUDE.md、加载 skills、执行 hooks、编排 subagent。它之所以成为 AI for data infra 的默认 runtime，关键在于持久性——会话可恢复、记忆能留存，昨天上线一次变更的那个 agent，今天照样能上线新的变更。

### 8 与 Snowflake、Databricks 和 dbt 的集成

对 AI for data infra 来说，最要紧的三个系统是 Snowflake、Databricks 和 dbt。正确的做法是每个系统配一个 MCP server，让 agent 框架充当 client。

| 系统         | MCP server     | Agent 能力                          | 护栏                    |
| ---------- | -------------- | --------------------------------- | --------------------- |
| Snowflake  | snowflake-mcp  | query、schema、cost、RBAC            | 行级访问、成本预算、masking     |
| Databricks | databricks-mcp | SQL warehouse、Unity Catalog、job   | 集群预算、Unity ACL、PII 扫描 |
| dbt        | dbt-mcp        | 模型、测试、文档、lineage                  | CI 强制执行、生产分支保护        |
| BigQuery   | bq-mcp         | query、INFORMATION\_SCHEMA、billing | slot 预算、dataset ACL   |
| Iceberg    | iceberg-mcp    | 表、snapshot、compaction             | 分支保护、snapshot 保留      |

### 9 Human-in-the-Loop

AI for data infra 的效率提升故事并不是 10x。对于 agent 能自主完成的工作，它更接近 3x 到 5x；而对于它做不了的工作，则是 0x——所以真正的杠杆在于分清这两类工作，并让人集中在后者。insights agent 是大多数团队会忽略的一环：它盯着其他 agent 在做什么，把需要人工 review 的决策标记出来，让团队知识不断累积。

Human-in-the-loop 有四种模式：

1. **完全自主**（低风险、可逆的动作，比如更新文档）
2. **部署前 review**（PR）
3. **逐动作审批**（写生产）
4. **有人监督**（人主导、agent 建议）。

一套生产技术栈会同时用上这四种，按任务类别切换模式。要是把所有动作都塞进同一个关卡，agent 要么变得没用，要么变得危险。

### 10 评估

评估 agent 比评估模型更难。模型可以用固定 benchmark 打分；而 agent 面对的是开放式任务，有很多种合理结果。生产中真正管用的有两种方法：**Agent-as-a-Judge**（用第二个 agent 给第一个打分），以及**决策追踪 context graph**（记录 agent 做出的每一个决策，之后可以 replay）。两者都无法取代人，但结合起来，它们抓 regression 的速度比任何测试套件都快。

* **Golden queries**——200 条已知正确的 prompt，附带预期输出
* **Agent-as-a-Judge**——一个打分 agent 审查每一个生产动作
* **Decision trace**——存下每一次 tool call 的输入、输出和理由
* **Replay harness**——用今天的 agent 重跑昨天的 incident
* **人工抽检**——抽取 1% 的 agent 动作做人工复核

### 参考

1. [https://dataworkers.io/resources/ai-for-data-infra/](https://dataworkers.io/resources/ai-for-data-infra/)

