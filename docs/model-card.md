# DeltaWeave 本地状态控制 adapter 模型卡

状态：本页区分已观察到的短状态 dev、原始 full 闭环失败与后续引用接口。训练与评测是否完成、实际预算、原始输出和结论，以 [`experiments/`](../experiments/) 中对应记录为准。尚无记录的指标应视为未测，不能从可运行代码、训练 loss 或导出成功推断模型达标。

## 模型与许可证

共同底座为官方 [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)，固定 revision 为 `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`。该稠密模型来自 Qwen2.5-Math-1.5B，架构为 `Qwen2ForCausalLM`；不应称为本地训练了完整 DeepSeek-R1 MoE。DeepSeek 说明其蒸馏模型用 R1 整理的样本微调，本项目只在发布底座上进行标准 LoRA SFT。[官方模型来源](https://github.com/deepseek-ai/DeepSeek-R1#deepseek-r1-distill-models)

DeepSeek 仓库与模型权重采用 MIT，原始 Qwen 来源为 Apache 2.0。使用及再分发应保留这两层来源和适用许可证；本卡不重新声明底座或上游代码的所有权。[DeepSeek 许可证说明](https://github.com/deepseek-ai/DeepSeek-R1#7-license)、[Qwen 原始许可证](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/blob/main/LICENSE)

本机下载工具固定模型 revision，校验源文件 SHA256。每个训练 run 再次校验完整七文件，并记录 tokenizer、配置、权重、数据与训练源码哈希。adapter 不包含完整底座；推理仍需相同固定底座及 tokenizer。

## 预期用途与明确边界

模型扮演只读软件依赖知识助手的状态控制器。在已给出的公开规则和可信 resolver 观测下，选择 `wait`、`ask`、`refresh`、`answer`、`abstain`，并在可以回答时正确复制完整状态、版本选择和引用载荷。

这是**闭合接口、合成状态上的行为适配实验**。prompt 明确给出优先级规则；resolver 的 `status`、`selection`、`citations` 和有效性标记是合法工具输入。模型没有被要求独立求解依赖约束。目标由公开状态合同确定，不能视为一般问答数据或由模型自行验证的真值。规则控制器是强基线：如果规则已经满足任务需求，模型结果不能证明 SFT 是必要组件。

当前状态数据不评估任意自然语言需求、恶意提示注入防御、完整 PyPI 的依赖可解性或真实运行时兼容性。有限元数据上的 resolver/oracle 验证属于另一个实验层；它们不因模型能复制合法 observation 就自动获得验证。真实更新闭环中的模型调用、规则拦截、工具重算和内容缓存也必须单独报告。

## 已观察到的结果与失效边界

行为 adapter 的[短状态 dev](../experiments/behavior-dev.json)为 32/32 严格成功，四个世界全部观测状态正确。这是小型合成闭合接口结果，不能等同于长证据复制能力或真实问答能力。

冻结模型随后完成的同证据确认如下，训练未因保留集结果追加：

| 模型 | test 严格完整输出 | future 严格完整输出 |
|---|---:|---:|
| 官方底座 M0 | 4/32 | 2/32 |
| 证据条件 QA Mqa | 20/32 | 20/32 |
| 更新行为 Mbeh | 32/32 | 24/32 |

两条 SFT 在共享四态的两个 split 均为 16/16，差异集中于更新控制状态；训练状态和动作覆盖的干预不能被解释成新事实推理能力。Mbeh 的 future 八错全部格式合法：四个过期 conflict 直接 answer，四个未发布且同时缺 Python/observation 的状态误选 refresh。future 四个世界均未实现整世界全对。保留该组合泛化失败，不用 JSON 合法率掩盖，不再通过保留集调参。[完整结果与逐状态解释](results.md#4-冻结模型的同证据对照)

在随后[原始 full 接口演示](../experiments/demo-behavior-coarse/report.json)的六个会话中，仅一个完成。三次新生成正常到达 EOS，但输出有额外 JSON 内容/闭合结构，严格解析失败；两个后续会话复用已完成的错误预测，再次被合同检查阻断。最后一次成功输出为 `abstain`，不是完整版本选择答案。这里有五次真实模型调用：一次 refresh、三次无效完整答案、一次 abstain；另有两次错误缓存命中，不能记成两次新的生成失败，也不能把它们从会话分母删除。

失败报告和[对应 full 源码](../experiments/full-copy-source/manifest.json)保留，不用后续接口的结果覆盖。现有观察证明短 dev 成功不足以保证长载荷闭环成功；它尚未单独隔离输入长度、输出长度、事实分布和序列化复杂度各自的作用。

后续可选的 `reference` 接口让模型输出当前 observation 的精确 `answer_ref` 与引用标识，Host 核验 query、当前快照、注册版本和执行记录后展开原始工具载荷。`reference_context` 保留同一短引用输出，同时提供只读完整证据作为输入；`full` 继续使用原合同和完整载荷输出。它们是不同响应接口，不能直接合并为同一准确率。

引用提示 v2 显式替换原合同第 5 条：resolved 和 conflict 均复制引用对象，去除了原 full 的 conflict 空 selection 要求与版本复制示例；full 提示逐字保持不变。错误、未知或过期引用不被修复，Host 展开也不替模型选择动作。展开后的版本和完整 citations 来自工具记录，**不是模型自行生成的事实**。引用成功率、Host 绑定校验、独立求解正确性及原始模型动作成功率应分别报告。[接口与信任边界](architecture.md#61-完整载荷与绑定引用是不同的响应接口)

在同一六事件开发演示上，[`reference_context` 为 1/6](../experiments/demo-behavior-reference_context/report.json)，[`reference` 为 4/6](../experiments/demo-behavior-reference/report.json)。缺失 status、错误 JSON 括号和误复制完整证据 selection 仍会导致阻断。引用是观察到 full 失败后提出的可选接口实验；该回放不能视为独立确认，也没有证明引用路线已可靠修复问题。CLI 默认仍是规则策略与 full 视图，模型及两种引用视图都需显式启用。

后续独立冻结的 16 条链确认中，full / reference_context / reference 分别完成 12/48、12/48、23/48 会话；三个规则视图都为 48/48。引用传输仍有坏 JSON 和动作优先级错误，因此不作为可靠模型方案推荐。没有继续使用确认结果调提示或训练。完整分母及错误归因见 [结果页](results.md) 和 [派生汇总](../experiments/reference-confirmation-summary.json)。

## 三个比较对象与数据干预

| 对象 | 训练条件 | 可以回答的问题 |
| --- | --- | --- |
| M0 | 同一官方底座，无本项目 adapter | 底座在同一提示、证据和解码预算下如何遵守接口 |
| Mqa | 终端、开放证据条件的 QA 状态 SFT | 证据回答、冲突说明、缺元数据与缺 Python 的接口适配 |
| Mbeh | 包含上述共享状态及更新状态的 SFT | 增加过期/缺观测、未发布、认证复用后，状态控制行为如何变化 |

共享四态为 `resolved`、`conflict`、`incomplete`、`missing_python`；新增四态为 `stale`、`no_observation`、`not_published`、`certified_reuse`。Mqa 只有 `answer`、`abstain`、`ask` 目标，Mbeh 另有 `refresh` 和 `wait`。因此二者比较同时改变状态覆盖、目标动作覆盖和样本分布；不能把总体分数之差单独归因于“理解了数据库更新机制”。报告应分别列共享四态、新增四态、各 action 的完整 payload 成功和回退情况。

同一训练世界不跨 split；Mqa 与 Mbeh 使用相同训练世界集合，但训练行数不同。具体数量、种子和 SHA 见 [`data/learning/manifest.json`](../data/learning/manifest.json)。`future` 是未见合成实体与先决条件组合，例如多个条件同时失败时的动作优先级，不是模型训练之后发生的真实 PyPI 时间序列。

## 训练合同与预算

预登记配置见 [`research/protocol-v1.md`](../research/protocol-v1.md)：BF16 底座、FP32 adapter、rank 4、alpha 8、dropout 0.05，覆盖 q/k/v/o/gate/up/down 七类投影；AdamW 学习率 `2e-4`，梯度裁剪 `1.0`，batch 1、累积 4，seed 42。这是标准 LoRA，未提出新训练算法。

以监督 completion token 预算比较训练路线，在达到预算的首个完整 optimizer window 后停止；实际 token 超额不超过该最后窗口。每个窗口按照全部有效监督 token 归一，而非平均四个不同长度目标的样本 loss。`--steps` 是硬上限；若达到上限仍未满足 token 预算，run 失败且不导出成功 adapter。

按固定种子生成长度桶次序，每个完整 epoch 覆盖全部训练行，最后一个部分 epoch 不保证全部行或世界被访问。相同监督 token 预算不意味着同样多的输入 token、样本、世界覆盖、训练轮数或墙钟时间。训练报告必须保留实际 exposure、step、样本次数、输入/监督 token、等效 epoch、最后窗口超额与耗时，不用目标预算替代真实消耗。

输入最长 1024 tokens；超长直接拒绝。`id`、`group`、`kind`、`split` 和监督 target 不进入模型条件输入；训练条件仅使用 `prompt`。这不代表 prompt 没有公开规则：规则和可信工具观测是本任务设计的一部分，不能据此推断隐藏知识泛化。

## R1 模板与短监督

[`research/model-protocol.md`](../research/model-protocol.md) 记录九个 tokenizer fixture。真实审计观察到官方完整 assistant 模板会剥去 thinking 前段，且拼接文本重新分词可能改变边界；因此本项目使用显式 `prompt_ids + completion_ids`。prompt 取官方 generation template，保留 `<think>\n` 起始；completion 为 `</think>\n{动作JSON}` 加一个 EOS，prompt labels 为 `-100`，整个 completion 和 EOS 参与监督。

PAD 与 EOS 相同，attention mask 必须按长度构造。真实角色标记和 thinking 标记并未全部被 tokenizer 标成 special，因此不能只依赖 `all_special_tokens` 判断控制串。九个 fixture 验证的是分词、边界和 mask，不是模型业务能力。

本轮使用空 rationale 的短动作监督，可能使模型更快结束 thinking。这是行为干预及其待测代价，不能声称保留或增强了 R1 的一般推理能力。训练平均 loss 低也不保证自由生成、动作判别或载荷复制正确。

## 推理与评测

正式协议使用同一 public prompt、相同底座/tokenizer、配对 prompt-derived seed，temperature `0.6`、top-p `0.95`、top-k `0`。当前预算为最多 2048 个新 token、每例 120 秒，包含 thinking；任何后续变更须留存版本及理由，不得在看到保留集后调整并继续称为原确认结果。官方 R1 报告采用过更长生成预算，本项目结果仅适用于本机明确的资源预算，不能代表底座完整推理能力。

主分数要求正常 EOS 结束、未超时/截断，并严格解析唯一 `</think>` 后的单一 JSON object，逐键比较 action 和完整 payload。多余键、错误版本或引用、重复 JSON key、尾随说明均不通过。列表保持顺序，未额外声明引用无序等价。

辅助分数只额外容忍最终段为一个单独的 JSON Markdown 围栏，不从多个候选中挑答案、不移除任意前言、不改 action 或载荷值、不用 gold 修复输出。主分数与辅助分数必须分开。即使部分输出可解析，未正常完成的样本仍失败，不能从分母移除。

逐例保留 raw text、generated IDs、停止原因、输入/输出 token、耗时、种子及来源哈希。恢复运行需全部数据、源码、模型、adapter 和解码设置完全一致；缓存 receipt 会重新解码、重算分数并核验绑定。MPS 与当前 RSS 是离散采样，采样最大值不能写成真实峰值显存。

## 统计与解释限制

test 与 future 各只有四个独立世界，每个世界八种相关状态。逐状态成功和整世界成功都应报告，不能把同一世界的状态当成独立任务估计精度。当前使用单一训练种子；随机生成虽按 prompt 固定种子，也不证明跨硬件、内核的逐位可复现，不能继承确定性工具引擎的重跑等价定理。

若 Mbeh 的收益主要来自格式、动作词表或见过的状态，结论应限定为该接口分布上的适配。若公共规则已经满足需求，优先使用可审计规则，adapter 可作为可选行为组件。关于跨真实专业领域、长期连续更新、记忆新事实、避免遗忘或恶意输入稳健性的结论，都需要额外独立数据与实验。

## 本机运行与发布

仅使用本机 Mac M4 Pro 24GB；无云 GPU 或付费模型 API。已有预检不保证所有上下文、batch 或长输出都适用，应以真实 run 的内存采样、终止和耗时为准。下列命令在项目根目录、已安装所需本机 ML 依赖的 Python 环境中运行；数据和模型均已在本地，入口不会自动下载。

```bash
# CPU 分词与来源审计，不加载模型权重；输出目录必须全新。
python scripts/train_adapter.py --data data/learning/train_qa.jsonl \
  --output runs/qa-dry --steps 512 --target-supervised-tokens 16384 --dry-run

# 两条路线分别新建目录；不要并行占用 MPS。
python scripts/train_adapter.py --data data/learning/train_qa.jsonl \
  --output runs/qa --steps 512 --target-supervised-tokens 16384
python scripts/train_adapter.py --data data/learning/train_behavior.jsonl \
  --output runs/behavior --steps 512 --target-supervised-tokens 16384

# 无 --adapter 时评测 M0；已完成输出拒绝覆盖。
python scripts/evaluate_model.py --data data/learning/dev.jsonl \
  --adapter runs/behavior/adapter --output experiments/behavior-dev.json \
  --max-new-tokens 2048 --max-seconds 120
```

只在未完成评测时，使用完全相同命令并加 `--resume` 复用已验证 receipt。诊断子集 `--limit-dev` 只允许 dev，不能截取 test/future 后冒充完整确认。训练没有精确优化器断点恢复能力；导出的 adapter 用于推理，不包含 optimizer/RNG checkpoint。

发布必须核验完成的 summary、run manifest、loss log、adapter manifest 与实际文件。导出保持权重、配置和训练 manifest 的原始字节；如果配置或元数据含本机绝对路径，拒绝发布并显式处理，不能静默改写配置破坏原有哈希。导出和哈希一致仅证明产物与记录一致，不是质量验收。实际模型结果与其限制请查看 [`experiments/`](../experiments/)。

```bash
# 两条路线都完成后：先只读核验，再导出到全新目录。
python scripts/export_adapters.py --runs runs --output artifacts --dry-run
python scripts/export_adapters.py --runs runs --output artifacts
```

缺少完成记录时返回 `pending`（退出码 2），不会创建目标目录。导出目录各包含原始 `adapter_model.safetensors`、`adapter_config.json`、`training_manifest.json`；根 `manifest.json` 保存两条路线的完成记录、loss log、覆盖计数与来源哈希。此命令不上传 GitHub，不执行模型，也不将训练完成改写为质量通过。
