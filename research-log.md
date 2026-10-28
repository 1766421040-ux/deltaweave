# Research log

## 2026-10-28 — Scope and initial branches

The user asks for a new project on database-update-driven collaboration between RAG, Agent and SFT, using a DeepSeek-source model. They prioritize careful reasoning about an effective and efficient update framework over implementation polish, authorize local model runs, and request a validated GitHub publication.

Confirmed constraints: current Mac only, no cloud GPU or paid model APIs; official small distilled DeepSeek accepted; no specific professional domain or database is supplied, and the assistant may choose the data. Public software dependency metadata is a provisional domain choice, not a user-prescribed domain.

The initial portfolio separates complete recomputation, positive-dependency incremental propagation, applicability-aware change propagation, and parameter updates. The first discriminating case is a newly applicable fact or exception with no edge in the previous proof graph. A second is a transaction or delayed index update that otherwise permits answers mixing versions.

Historical proposals on a shared policy-and-answer model and dependency state are treated as proposals, not implemented or experimentally supported results. The prior EvidenceGate project is complete and will not be modified.
# 2026-10-28 首轮证据与调整

终审进一步复现并修复共享 Engine 的结果/last_receipt 竞争窗口，增加同锁 `run_with_receipt` 与并发回归。计时审查发现 v3 把 oracle 初始副本算入主时间；该开销各模式共有，但与排除评分开销的口径不符。将其移出计时，保留 v3 精确源码与原报告，按同一冻结负载重跑 v4；不改变阈值或选有利样本。

已完成最近邻审查，确认动态依赖、集合失效和相等截断有经典先例；将主张限定为 Agent 工具合同、训练数据来源管理及公平实测。

CPU 开发测量否定“更细的谓词必然更快”。加入所有模式共享的等值索引后，coarse 获得主要收益，predicate 在高扇出明显亏损。保留初始扫描版与索引版报告及源码归档；默认采用 coarse，再冻结 .25 完整事件密度的安全 full 回退进行 v3 验证。这个阈值是 AI 开发期决策，不是用户指定，也不是已证明最优。

本地 DeepSeek 可行性已验证。正式首个 dev8 正常结束但 0/8 严格成功；开始相同监督 token 预算的 QA/行为 LoRA。模型接口实验是有限合同控制，规则基线可以达到合同正确；不得借 SFT 分数声称新知识被写入参数，或宣称一般问答提升。

## Confirmation freeze and interface failure

Both adapters are frozen after the original supervision-token budget: QA 16,522 tokens / 157 steps, behavior 16,534 tokens / 174 steps. No second training stage is selected. QA dev is 20/32; behavior dev is 32/32. Before reading any test/future predictions, freeze the original model, learning-data contract, seeds and protocol-v1 configuration for all three model routes on both 32-case confirmation splits.

The actual long-evidence SQLite development demo completed only 1/6 sessions with the behavior adapter. Three fresh generations produced malformed/incomplete answer JSON, while two later sessions reused a bad cached output. The original report and five source files are retained. This is evidence against generalizing the short-state dev score to real long payloads.

The subsequent intervention is an interface experiment, not additional SFT: compare full input/output copying, full factual context with a compact bound observation reference, and compressed context with the same reference output. The host supplies the expanded factual answer; raw model actions remain separately scored and never repaired. Reference transport does not establish factual truth or general reasoning skill. The original state test/future still evaluate the unchanged full-copy contract. A separate reference confirmation protocol must be frozen before model execution.

The original confirmation is now complete: base/QA/behavior score 4/20/32 out of 32 on test and 2/20/24 on synthetic future combinations. Behavior fails all four stale-conflict and all four combined-not-published states despite valid JSON. The adapters remain frozen; no test-informed retraining was performed.

Reference protocol v2 first removed a genuine contradictory conflict-output instruction before confirmation. Development full/reference_context/reference completed 1/1/4 out of 6. The subsequent independently frozen 16-chain confirmation completed 12/12/23 out of 48, while all rule views completed 48/48. References reduce generation burden, but do not establish reliable model control. All model answers actually emitted passed the finite oracle; blocked failures remain in completion denominators. No rejected prediction was repaired or replaced for scoring.

An actual full-engine model demo produced exactly the same generated IDs as the coarse demo: both used 5 fresh model calls and 2 ordinary semantic-cache hits. Thus this case does not support a special model-call saving from the incremental graph. The final selection is rule/full observation policy with coarse execution by default; model/reference paths remain reproducible optional experiments.
