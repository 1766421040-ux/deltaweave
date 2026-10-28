# DeepSeek R1 蒸馏模型：tokenizer 与生成协议

本文件保留首次 tokenizer 审计时的设计记录；在当时只完成 CPU 分词、模板与标签边界检查，尚未加载模型。随后已完成官方权重校验、真实本机 LoRA 和模型生成评测。最终参数以 [protocol-v1.md](protocol-v1.md) 为准，已发生的结果见 [结果页](../docs/results.md) 与 [模型卡](../docs/model-card.md)，不要把下文的历史待定项当成当前执行状态。

## 1. 底座与来源

- 模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`。
- 固定 revision：`ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`。
- 架构：`Qwen2ForCausalLM`，来源为 Qwen2.5-Math-1.5B；本地实际 tokenizer 类名为 `LlamaTokenizerFast`，不能据此将模型称为 Llama 架构。
- DeepSeek 官方模型/代码为 MIT，来源 Qwen 为 Apache 2.0；公开 adapter 必须保留来源、revision 和许可证说明。
- 此蒸馏底座使用 DeepSeek-R1 整理的样本微调。本项目后续若执行 LoRA SFT，应描述为对官方蒸馏底座的适配，不能描述为复现完整 R1 训练或重新完成其蒸馏过程。

来源：[固定模型快照](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/tree/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562)、[官方来源与许可证](https://github.com/deepseek-ai/DeepSeek-R1#7-license)。

## 2. 已实测的模板问题

官方 generation prompt 以 `<｜Assistant｜><think>\n` 结束，但完整 assistant 消息模板会移除 `</think>` 及其前面的内容。因此，完整会话编码后的前若干 token 不等于真实推理 prompt，不能用 `len(prompt_ids)` 在完整会话编码上猜测监督边界。

九个审计组合覆盖三个 user 输入（英文、版本约束与多行、中文及 Unicode）和三个 completion 前缀（空理由、简短理由、前导换行）：

| 检查 | 实际结果 |
| --- | --- |
| full-chat 编码是否以 generation prompt IDs 开头 | 0/9 一致，风险已复现 |
| 分别编码后拼 IDs 是否等于拼文本后重新编码 | 6/9 一致；带前导换行的 3/9 不同 |
| 显式拼 IDs 后 prompt 全屏蔽、completion 与 EOS 全监督 | 9/9 通过 |
| 显式 prompt/completion 解码与原始文本往返 | 9/9 通过 |
| 最终 action JSON 按固定分界严格解析 | 9/9 通过 |

已测 token ID：BOS `151646`，EOS 与 PAD 均为 `151643`，闭合 thinking token `151649`。`</think>` 虽是 added token，但在这个固定 tokenizer 中没有标为 special；实测 `skip_special_tokens=True` 不会移除此分界。user/assistant 角色标记也没有被标为 special，输入检查不能只依赖 `all_special_tokens`。该解码选项仍会隐藏 EOS 等控制符，因此审计和评分采用 `skip_special_tokens=False`，按 ID 单独移除终止 EOS。

训练 fixture 的推理理由用于边界测试，不是专业任务的教师数据或模型质量证据。

本次小文件 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `config.json` | `37bd455e9679d2959536270fed49d25cc7c290a64f6e52abb97c71345a9cee41` |
| `tokenizer.json` | `88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773` |
| `tokenizer_config.json` | `8ac8c85fb242563c2260baec0909debd69d718af6a0b3d90e6cab62b4d341cd5` |
| `generation_config.json` | `72cbec1015da9ed03ad025483005cbf6481403abf947adfd54e504d1a66a2126` |

当次检查时权重下载尚未完成，`download-manifest.json` 尚未出现；审计报告明确为 `pending_download_manifest`，不声称全模型下载已验证。下载完成后重跑会核对模型 ID、revision 以及这些文件的 manifest 哈希。上述模板文本的 SHA256 为 `56a1447ad31926fdc21fb07e56e5642bd9c850c4f52d8c8af7bbe5f079a84f5f`。

## 3. 训练与推理共用的 token 合同

实现位于 [`scripts/audit_tokenizer.py`](../scripts/audit_tokenizer.py)。它只依赖 tokenizer，动作集合由任务层另行验证。

1. 将所有实际可观察的指令和证据放在单条 user 消息中，用官方模板 `add_generation_prompt=True` 得到 `P = prompt_ids`。不把 target、gold、split、样本分型等标签元数据拼入 user 输入。
2. 明确写出 completion 文本：`[可选短理由与换行]</think>\n{紧凑动作JSON}`。以 `add_special_tokens=False` 单独编码为 `C`，末尾追加一次 EOS ID。
3. 输入是 `P + C`，标签是 `[-100] * len(P) + C`。不重新分词拼接文本，不调用完整 assistant chat template 生成训练目标。
4. 标准 causal LM 内部 shift 后，prompt 的最后一个 token 预测 `C[0]`；不能再手动把 completion 的第一个 token 屏蔽。EOS 也参与训练。
5. 训练右 padding 的 labels 为 `-100`；attention mask 依据真实长度构造。不能使用 `input_ids != pad_id`，因为真实 EOS 与 PAD 相同。批量生成左 padding，同样提供显式 attention mask。
6. 超长样本直接报错，先按完整数据预检再选择上下文长度；不静默截断 prompt、理由或 JSON。
7. 模型输入中的特殊控制串必须拒绝或在上游按记录的规则转义；不能允许数据伪造 user/assistant 边界。禁止把数据处理时的转义当作修复模型输出。

可供训练模块复用的接口：

```python
prompt_ids = encode_prompt(tokenizer, user_content)
example = encode_training_example(
    tokenizer, user_content, target, rationale="", max_length=chosen_max_length
)
# example: input_ids, attention_mask, labels, prompt_length,
#          completion_ids, completion_text
```

空 `rationale` 意味着模型被监督立即生成 `</think>`，这可能教会短动作输出并改变底座的推理行为。该行为及其质量/时延代价必须测量，不能因保留 `<think>` 起始就声称“保留了 R1 推理能力”。若加入理由，只使用有来源、可检验的简短依据或符号证明；理由不是额外真值，不采用未经验证的长推理充当标签。短监督和带依据监督若成为实验分支，必须预登记并给基座相同可观察证据。

## 4. 自由生成、最长预算与公平比较（待预实验冻结）

生成必须从 `encode_prompt` 返回的完全相同输入 IDs 开始，禁止候选打分替代自由生成，也不通过规则把错误动作改成正确动作。保存完整 generated IDs、raw continuation、停止原因及耗时。generation 输出不含已有 prompt 的 `<think>` 起始；评分只对 continuation 操作。

官方建议温度 0.5–0.7（推荐 0.6）、将指令放在 user、保留 thinking 起始。其报告的最大生成长度为 32,768 tokens；这不代表本项目的 24GB 机器能够经济地采用该预算。本项目不得用 72-token 短上限压低基座，再把格式/截断差异当作推理提升。[官方使用与评测设置](https://github.com/deepseek-ai/DeepSeek-R1#usage-recommendations)

**预实验方案而非已验证配置：**先在少量 train/dev 深推理实例（版本范围、多跳依赖、新增适用条件）上测 base 的吞吐、长度和终止分布，初次探测可用 `max_new_tokens=4096`、每例 `300s`。它们是可调整的探测上限，没有证据表明足够或最优，不可直接宣布为正式预算。参考官方采样设置 `temperature=0.6, top_p=0.95`；配对随机种子记录到结果。若另做 greedy，则明确作为独立设置，不能与采样结果混合。

正式比较前，按预实验测得的吞吐和截断原因锁定全体模型相同的最大生成 tokens、墙钟预算、上下文上限、停止条件、解码设置和种子；冻结 test 前保存配置及 SHA。若任一分支在 dev 上超过 5% 被 token/time cap 截断，先诊断原因：必要时对所有分支统一增加预算并重跑 dev；本机无法承担时明确将结论限定为该资源预算，不能声称基座充分推理后仍然失败。更长预算并不自动更公平或更有效，最终选择必须有 dev 证据。不得依据 test 输出调整这些参数。

墙钟超时应由单样本停止条件检查，记录从推理开始到实际终止的总时间和超时超出量；仅 `generate(max_time=...)` 可能在当前一步完成后才退出，不能宣称硬实时截止。批量 throughput 另报；公平单例预算比较不能因不同批次组成而改变超时归因。

至少报告以下项，使用所有预登记实例作为分母：

- 严格任务成功率、动作有效率、版本/证据一致性；另报格式有效率，不能用格式替代语义成功。
- EOS 终止率、达到 token 上限率、时间上限率、OOM/异常率，以及缺少 `</think>`、缺少/多余 JSON、重复 key 等格式错误。
- 总生成 tokens、thinking tokens、最终 JSON tokens，p50/p95 延迟及模型/工具调用次数。分 token 的边界用实际 token IDs 或 tokenizer offsets 审计，不按字符串长度估算。
- 使用同一证据条件的 paired base/adapter 差异；采样比较至少公开种子与原始输出，是否增加重复次数取决于冻结预算。

达到 cap 或超时的样本保留原始输出并计为未完成，即便截断前恰好出现一个可解析 JSON 也不能悄悄算完成。格式解析只接受唯一闭合 `</think>` 后的一个完整 JSON object，拒绝代码围栏、尾随说明、重复键、NaN/Infinity；随后仍须任务层严格验证 action 与参数 schema。模型输出不要求非空 thinking，但必须区分“无推理直接动作”和“有依据后动作”的长度/质量结果。

## 5. 已验证与待验证边界

已运行命令（在项目根目录，使用已有本机 Python 环境）：

```bash
python scripts/audit_tokenizer.py --model .cache/deepseek-r1-distill-qwen-1.5b
python -m unittest discover -s tests -p test_tokenizer_contract.py -v
```

`python` 需安装 Transformers/tokenizers；审计的 `local_files_only=True` 禁止自动联网下载，且 `trust_remote_code=False`。可用 `--output <新文件>` 保存 JSON，已有报告拒绝覆盖。离线单测包含六项不依赖 Transformers 的合同测试和三项真实本地 tokenizer 测试；2026-10-28 本机全部九项通过。没有本地 tokenizer 或 Transformers 时明确跳过真实集成测试，不能将该状态称为真实 tokenizer 审计通过。

待验证：完整权重来源验收；MPS BF16 的 forward/训练峰值与吞吐；teacher-forced 标签是否改善自由生成；短动作监督对推理、更新泛化及旧能力的影响；正式生成预算与时间上限；业务动作 schema 与独立真值验证器。tokenizer 审计通过不等价于其中任何一项通过。
