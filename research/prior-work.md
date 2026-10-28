# DeltaWeave：最近邻与机制来源审计

核验日期：2026-10-28。只把本轮实际读取的官方文档、源码或论文段落列为证据；访问网页不等于复现软件，论文作者的性能数字不作为本项目结果。下列仓库链接指向读取时的分支；实际复用代码前仍需固定 commit 与许可证版本。

## 1. 当前创新性判断

“记录依赖—检测更新—局部重算—输出不变时停止”已有成熟先例。空查询与新匹配行的失效也有数据库理论背景。当前 DeltaWeave 应被称为 **面向版本化 Agent 工具观测的增量更新框架与验证研究**，不称新通用增量算法。

最初提出的可检验差异包括：工具能否给出完整候选域、缺失查询与适用条件的可审计证书；模型、工具、证据和训练样例是否服从同一快照；在计入证书成本之后是否有用。后续有限域实测见 [结果页](../docs/results.md)。新增的反例集、代码或工程组合不能自行证明论文级新颖性。

## 2. 最接近的计算机制

### Build Systems à la Carte，Mokhov、Mitchell、Peyton Jones，ICFP 2018

已读原论文第 2.3 节、4.2.2—4.2.3 节及相关示例。Shake 通过旧依赖图发现变化，重执行时发现新依赖；输出未变可 early cutoff。Verifying traces 保存依赖及其值/哈希，用当前依赖校验已有结果；constructive traces 进一步保存可复用结果。这直接覆盖 DeltaWeave 的大部分调度与复用思想。[论文 PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/03/build-systems-final.pdf)、[作者代码](https://github.com/snowleopard/build)

**对设计的约束**：不得把“观测 digest + 动态依赖 + 停止传播”声称为首创；模型/工具版本也属于任务描述依赖。可比较的是 Agent 查询域的正确封装与维护代价。本文没有复现该论文全部调度组合，也没有证明 DeltaWeave 达到其 minimality 定义。

### Adapton，Hammer 等，PLDI 2014

已读作者论文的执行图说明、正确性讨论、Algorithm 1 及第 5 节相关段落。其 demanded computation graph 区分 dirtying 与按需求传播；按读取顺序检查边；发生不一致时清空旧 outgoing edges，并由重执行建立新边。该机制已经处理动态换支、按需执行和缓存复用。[作者页面](https://matthewhammer.org/adapton/)、[论文 PDF](https://matthewhammer.org/adapton/adapton-pldi2014.pdf)

**对设计的约束**：节点输出未变时，仍须更新已重执行节点的读集；不能为了维护旧图而执行已经消失的分支。我们目前只给受控工具与有限终止执行的条件证明，不继承 Adapton 形式系统的全部证明结论。

### Salsa 与 rustc 的 red-green / backdating

已读 Salsa 官方 `book/src/reference/algorithm.md`。输入/查询记录 revision；输入变化后重新执行；输出等值则 backdate，使依赖者不必重跑。Rust 官方增量编译说明也明确以纯函数、显式执行 DAG 为基础，并使用 stable fingerprints 检查结果。它们是 U4 的直接工程最近邻。[Salsa 文档源码](https://github.com/salsa-rs/salsa/blob/master/book/src/reference/algorithm.md)、[Rust 官方说明](https://rustc-dev-guide.rust-lang.org/queries/incremental-compilation-in-detail.html)

**对设计的约束**：如果数据库集合读取已被编码为受追踪输入，经典正依赖跟踪也可以正确捕获插入。我们批评的是仅追踪返回行 ID 的薄弱实现，不能暗示所有已有依赖跟踪都漏掉新节点。Salsa 不直接提供 PyPI 数据完整性或模型随机性的保证；把它嵌入框架也需要这些接口条件。

### DBToaster，Ahmad 等，PVLDB 2012

已读原论文摘要、引言与开头两个 delta 例子。Viewlet transforms 对查询递归作差分，将查询及高阶 delta 物化并互相增量维护；不是每次重新执行查询后再比较结果。其代表比“触发重查询”更强的代数式增量维护路线。[论文 PDF](https://vldb.org/pvldb/vol5/p968_yanifahmad_vldb2012.pdf)、[项目研究入口](https://dbtoaster.github.io/home_research.html)

**对设计的约束**：若工具主要是 SQL 过滤、连接和聚合，应比较成熟 IVM/物化视图的能力，不能只与低效扫描重算比较。首版重查询观测的成本必须实报；不把高阶 delta 的性能性质移植给本项目。没有读完其全部优化和实验，因此不作适用 SQL 全覆盖判断。

### 数据库 phantom 与谓词依赖

PostgreSQL 官方隔离文档将 phantom 定义为重新执行同一条件查询时匹配集合改变；Serializable 的 predicate locks 用于检测本会改变先前读取结果的写入，并可以因内存成本合并为更粗粒度。此机制服务并发事务序列化，不是长期答案缓存，但证明“未返回行也能影响查询”并非新的问题。[官方文档](https://www.postgresql.org/docs/current/transaction-iso.html)

**对设计的约束**：证书订阅与谓词锁的目的、生命周期不同，不能宣称等价实现；可借鉴保守覆盖及粒度/维护成本权衡。单个查询看到一致数据不意味着多个 Agent 工具调用、外部向量库和模型配置自动属于同一快照。

## 3. 动态 RAG / 图 / 流水线的四个实现参考

| 项目与已读证据 | 已有机制 | DeltaWeave 的边界与借鉴 |
|---|---|---|
| [LightRAG](https://github.com/HKUDS/LightRAG)，README 及 [Core API 指南](https://github.com/HKUDS/LightRAG/blob/main/docs/ProgramingWithCore.md) 的删除/恢复段落 | 文档删除后清理 chunks，删除失去全部支持的实体/关系，并从其他来源重建共享对象；图、追踪表、向量采用分阶段更新及恢复 | 来源支持集和局部重建已存在。指南说明向量 flush 失败可遗留旧记录，需要重建修复，不能把“调用删除 API 成功的一部分”当整体新快照就绪。用于文档 RAG 对照，不作精确依赖求解器 |
| [Graphiti](https://github.com/getzep/graphiti)，README 及 [`graphiti.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/graphiti.py) 的 `add_episode` / `remove_episode` | 时间有效事实、episode 来源与增量更新；当前删除实现保留仍受其他 episode 支持的边，并重建剩余节点摘要 | 当前源码摘要重建失败分支会记录错误并恢复旧摘要，因此不能推导整体删除操作的原子性。README 提醒小模型结构化输出风险；不应让 1.5B 模型从已有精确结构重新猜测依赖。未复现软件 |
| [Pathway](https://github.com/pathwaycom/pathway)，[连接器说明](https://pathway.com/developers/user-guide/connect/connectors-overview/) 与 [`data_index.py`](https://github.com/pathwaycom/pathway/blob/main/python/pathway/stdlib/indexing/data_index.py) | 数据流以插入/撤回传播更新；`query` 可随索引变化更新既有查询结果，`query_as_of_now` 保留请求时结果 | 是动态派生数据的强近邻；旧 chat 答案与 SFT 数据不因使用流式框架就自动重验证。核心当前 BSL 1.1，README 说明转换安排，不应统称 MIT/Apache 项目；本轮不做许可法律结论 |
| [Debezium](https://github.com/debezium/debezium)，[PostgreSQL 连接器官方文档](https://debezium.io/documentation/reference/stable/connectors/postgresql.html) | WAL 增删改、事务元数据、快照、删除 tombstone；恢复可能重复交付，下游需处理至少一次投递 | 提供事件入口，不提供派生答案/图/训练数据失效。PyPI 远端 API 不等于我们的 WAL；首版只能声明本地观测快照差分。迁移自有 PostgreSQL 时才考虑 CDC |

这些项目已覆盖增量输入、图维护或动态查询中的多个层次。我们的候选差异是跨工具合同的依赖闭合与可重放验证，而不是仅把四个库连接起来。是否需要其中任一库，应由本地实验范围决定；小规模版本存储不必为展示组件而引入完整流处理集群。

## 4. PyPI 场景能提供的真值及不能提供的真值

### 文件级元数据，不能拿 release 摘要当全版本事实

PyPI 官方 JSON API 明确说明上传元数据未必匹配文件内容，同一 release 首次存入的元数据不会被后续上传更新。Index API 提供文件级 Core Metadata 标识与摘要；Core Metadata 规范定义 `Requires-Dist`、`Requires-Python` 及 marker。由此推导：同一版本应保留发行文件和环境范围，不能无条件合并为单一依赖行。[JSON API](https://docs.pypi.org/api/json/)、[Index API](https://docs.pypi.org/api/index-api/)、[Core Metadata](https://packaging.python.org/en/latest/specifications/core-metadata/)

本地可以声明“在冻结候选域、已验证的文件元数据、给定环境和策略下约束可满足”。不能声明任意平台实际可运行；也不能从当前快照自动恢复历史删除/yank 的精确时刻。

### Yank、版本约束与求解器职责

File Yanking 规范定义文件级、可撤销状态，并允许精确 pin/锁文件相关策略；因此 yank 不等于删除。`packaging` 的 `Requirement`、`SpecifierSet`、`Marker` 是解析与条件求值组件；pip 官方说明完整求解借助 resolvelib 的回溯过程。[Yanking](https://packaging.python.org/en/latest/specifications/file-yanking/)、[packaging](https://packaging.pypa.io/en/stable/requirements.html)、[pip 求解机制](https://pip.pypa.io/en/latest/topics/more-dependency-resolution/)

真值应由有限域的独立 solver 或枚举器给出；LLM 生成的解释、内部推理和生成数据的自身标签都不是独立真值。组织策略的新增/修改是可控合成事件，须与真实抓取的发布元数据清楚分开。

## 5. 更贴近领域的研究，当前只核实摘要

以下三项已打开 arXiv 原始摘要页，未做全文机制复现；不能据此声称它们“没有”某项机制，只能用来排除宽泛首创说法。

| 研究 | 摘要直接支持的最近邻关系 | 对选题的影响 |
|---|---|---|
| [MEMRES，2026](https://arxiv.org/abs/2604.16941) | Agent Python 依赖恢复；结合记忆、知识库与置信度级联，把 LLM 放在末级 | “Agent + 依赖修复”已存在；本项目不能只改底座模型就当新机制 |
| [SMTpip，2026](https://arxiv.org/abs/2608.15886) | PyPI 依赖知识图与解释器条件编码为 SMT，联合求解版本与解释器 | “PyPI 图 + 约束求解”已存在；我们的独立 solver 是验证基座，不声称新求解算法 |
| [CHRONO-RESOLUTION，2026](https://arxiv.org/abs/2607.15315) | 在发布时点构建 npm/PyPI/crates.io 的依赖解析数据 | “时间化依赖图”已存在；本项目要区分真实观测历史与合成更新回放 |

若后续要写论文级创新声明，应先阅读全文及作者代码，尤其检查动态缓存失效、候选集合订阅、版本快照和评估设置。目前这些差异保持未决，不能从摘要未提及推导不存在。

## 6. 必须保留的竞争解释

1. **普通 query memo 已足够**：只要工具把整个项目候选集注册为受追踪输入，Salsa 式 backdating 就能产生同样效果；复杂谓词证书的额外价值要通过成本与粒度对照证明。
2. **所有收益都来自精确工具**：如果规则 solver 已经决定全部行为，模型只改写答案，不能把系统正确率提升归因于 SFT 或 Agent 推理。应单独测模型选择工具、提出必要补充问题与解释的价值。
3. **数据规模让全量重算更便宜**：减少模型调用但大量扫描候选域、维护订阅，不等于总更新成本更低。
4. **“停止传播”只是投影收窄**：若 U4 模型少看到信息而 U3 模型看到原文，差异来自输入变更，不能称同任务上的增量算法优势。四路线必须共享公开工具合同。

最初优先选择 [update-semantics.md](update-semantics.md) 中的有限状态回放、两步动态换支反例和全成本比较。最终实际实现的是共享索引与相等截断下的 full/coarse/predicate/adaptive 对照，不能将其写成已完成全部理论 U3/U4 调度组合的消融。详细范围及负面结果见 [结果页](../docs/results.md)。
