# Findings

## Current evidence

- User constraints and research priorities are recorded in research-state.yaml.
- The official DeepSeek 1.5B distilled checkpoint is published under deepseek-ai and uses a Qwen2 architecture; it is not a locally trainable copy of the large DeepSeek MoE. Source/tokenizer audits and actual local training are complete.
- Dynamic retrieval, temporal knowledge graphs and incremental deletion are already present in existing projects. Combining these labels is not itself an innovation.

## Main open question

Can a conservative applicability-aware dependency certificate preserve full-recomputation correctness while avoiding unnecessary retrieval and model inference after a data change, including newly introduced dependencies? What happens when the applicability description itself is learned or incomplete?

## Learning boundary

We must separately measure gains from refreshed knowledge, a changed inference policy, and SFT parameter adaptation under identical knowledge and tools. A system can answer fresh facts using retrieval while its parameters remain unchanged; this is not evidence of knowledge stored in the model weights.

## Results

Current verified results are indexed in [docs/results.md](docs/results.md). The following development observations are historical, not a replacement for final confirmation.
# 2026-10-28 本机首轮结果（开发阶段）

- 官方 DeepSeek-R1-Distill-Qwen-1.5B 已校验下载并在 MPS 上完成生成与 8 步 LoRA 可行性预检。预检不属于质量基准。
- 精确有限状态回放包含 384 个单步转换：full/coarse/predicate 均 0 次不一致；只追踪已返回行的 positive 出现 82 次不一致。反例主要涉及空/未返回集合成员进入有效域。
- 共享索引后，960 行低变化场景 full 累计约 .066–.072 秒，coarse 约 .029 秒；这是开发期微测中位数，非生产吞吐承诺。高扇出时 full .340、coarse .370、predicate .789 秒，细粒度方案反而更慢。数据见 `experiments/updates-indexed-v2.json`。
- 因此将默认收束到 coarse，保留 predicate 作为细粒度对照；新增透明的高密度 full 回退候选，在另一组规模/种子上再验证，协议见 `research/protocol-cost-v3.md`。不声称全局最小前沿。
- 域求解首轮 74 链/157 状态，full/coarse/predicate 与独立有限 oracle 一致；positive 144/157。coarse 与 predicate 求解次数同为 126，不能把微小查询次数差异宣传成机制大胜。报告 `experiments/domain-v1.json`。
- 原始模型的首个 8 状态开发诊断严格成功 0/8、格式有效 4/8；所有样本正常 EOS，没有因 2048-token 上限截断。错误包括误用旧证据、字段抄错、忽略发布前置条件。这只是一个世界，不可外推为全部 DeepSeek 能力。QA 与行为 SFT 对照正在执行，尚无保留集结论。

## 冻结确认与模型接口反例

- 最终 v4 共享索引成本在 240/2400 行、新 seed 93、5 次重复下完成；四个安全模式通过 384 转换，positive 错 82。高扇出时细谓词维护成本超过重算。详见最终报告，不以先前开发测量替代。
- 原始模型/QA/行为 adapter 的 test 严格成功为 4/32、20/32、32/32；future 为 2/32、20/32、24/32。future 行为失败是条件组合优先级，不能声称已解决一般更新行为。
- 真实六事件长载荷 demo 中，full 为 1/6 完成；短引用保留全部事实上下文仍是 1/6，压缩上下文为 4/6。短输出节省生成 token，但并未解决全部错误；公开输入也变化的结果不得归因为增量算法。
- 默认保留规则控制器。模型路线错误被阻断，不以规则答案替代后计为模型成功。引用展开是宿主工作，不是模型事实生成。
