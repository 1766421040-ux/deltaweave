# 发布验收记录

2026-10-28，在 macOS 27.0.1 / arm64、CPython 3.12.14 上完成独立 CPU 验收。使用 `/private/tmp` 下新建的虚拟环境，`include-system-site-packages = false`；执行命令时清除 `PYTHONPATH` 并设置 `PYTHONNOUSERSITE=1`。没有借用项目旧环境或相邻项目的测试依赖。

最初独立环境验收通过 **105 个测试、57 个子测试；5 个可选模型测试跳过**。增加引用合同和发布包校验后，又把将发布的源码（排除所有 Git 忽略文件）复制到新临时目录，使用现有 CPU 测试环境完成 **133 passed、5 skipped、66 subtests passed**。后一次是干净源码检查，不冒称另建了全新环境。随后增加分片传输与中断恢复测试，当前源码本机全量测试为 **136 passed、5 skipped、72 subtests passed**。CPU 框架验收不替代模型实验；远端 CI 收据单列于下。

## 安装与命令

在源码仓库根目录执行以下流程；输出必须使用尚不存在的路径。下列环境路径写法便于复现，实际验收使用独立临时目录。

```bash
python3.12 -m venv /tmp/deltaweave-release-env
source /tmp/deltaweave-release-env/bin/activate
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python -m pip install -e '.[test]'
python -m pip check
python -m pytest -q
python -m deltaweave demo --output /tmp/deltaweave-demo-new --mode adaptive
python scripts/benchmark_domain.py --output /tmp/deltaweave-domain-new.json
```

安装成功，`pip check` 无损坏依赖。实际安装版本为 DeltaWeave 0.1.0、packaging 26.3、pytest 9.2.0、pip 25.0.1；packaging 和 pytest 均来自新虚拟环境的 site-packages。首次受限网络下载失败后，仅为安装声明的构建/CPU 测试依赖重试，未下载模型。

| 检查 | 实际结果 |
| --- | --- |
| 初次独立环境 `python -m pytest -q` | 105 passed、5 skipped、57 subtests passed |
| 最终待发布源码副本 `python -m pytest -q` | 133 passed、5 skipped、66 subtests passed；无底座、权重、训练 JSONL 或旧运行目录 |
| 分片分发改动后本机全量 `python -m pytest -q` | 136 passed、5 skipped、72 subtests passed；使用现有 CPU 虚拟环境 |
| CI 同款 SQLite demo，`adaptive` | 6 个状态全部完成，最终不完整目录正确拒答 |
| CI 同款域基准 | 74 条链、157 个状态；4 个安全模式均与 oracle 一致 |
| 已安装的 `deltaweave --help` | 正常列出 `demo` / `resolve` |
| 离开源码工作目录后执行 `deltaweave resolve 'requests>=2.31'` | `resolved`，requests 2.32.5；证明入口不依赖当前工作目录 |
| 训练/评测脚本 `--help` 与 README 参数 | 参数匹配；没有执行训练或模型评测 |
| README 本地链接与脚本路径 | 检查时全部存在 |

这与 [CI 配置](../.github/workflows/ci.yml) 的 CPU 命令一致，但运行平台是本机 macOS。Ubuntu / Python 3.10 与 3.12 的远端矩阵结果需以实际 GitHub Actions 收据为准。

5 个跳过项有明确原因：3 个 tokenizer 集成测试缺 Transformers，1 个伪模型 CPU 循环缺可选 ML 库，1 个梯度检查缺 PyTorch。新环境中没有 torch、transformers、peft、accelerate 或 safetensors；纯合同测试仍运行。跳过不能视为模型功能通过。

## 数据库到回答与训练清单

最终产物为 [demo report](../experiments/demo-rule-final/report.json)、[独立审计结果](../experiments/demo-rule-final/independent-audit.json) 和 [审计程序](../experiments/demo-rule-final/audit_replay.py)。演示实际写入本地 SQLite；初始数据来自已记录的公开文件元数据，后续事件均为合成实验覆盖层。

| 状态 | requests 选择 / 动作 | 训练样例变化 |
| --- | --- | --- |
| 初始目录 | 2.32.5 / answer | 建立初始样例 |
| 新增组织限制 | 2.31.0 / answer | 重标 |
| 无关项目策略 | 2.31.0 / answer | 无变化 |
| 撤销组织限制 | 2.32.5 / answer | 重标 |
| 撤回选中文件 | 2.32.3 / answer | 重标 |
| 删除完整性证书 | incomplete / abstain | 重标 |

独立审计从公开 catalog 及声明事件重建每个状态，不调用生产 catalog 转换器、Store.commit、SQLite loader 或求解器。6/6 状态均通过有限枚举 oracle / witness 核验；source identity、本地/来源 revision、快照指纹、执行版本、样例内容、manifest、工具收据、最终数据库及连续 outbox 全部一致。每步 `raw_model_success = null`、新鲜模型调用数为 0，规则完成不计为模型成功。

审计共享 packaging 的 PEP 440/508 解析器，因此解析器缺陷仍是共同风险。引用检查覆盖记录存在性及所选版本支持，不声称最小证明或完整 unsat core。SQLite 文件是本地生成物、被 Git 忽略；从新输出目录运行 demo 可重建数据库。归档审计程序依赖该目录的数据库，拒绝覆盖已有审计 JSON。

## 域回放与源码绑定

[最终域基准](../experiments/domain-final.json) 包含 122 个 resolved、29 个 conflict、6 个 incomplete 状态。full、coarse、predicate、adaptive 均为 **157/157 状态一致、122/122 resolved witness 有效、74/74 整链通过**。仅跟踪已返回记录的 positive 消融为 144/157 状态一致、63/74 整链通过；它不能作为安全部署模式。

这是有限候选集合、受控更新与固定随机图的验证，不是所有 PyPI 或运行时兼容性保证。此次验收只使用计数和正确性结论，不从并行开发期间的域回放时间推导性能优势。

最终引用协议 v2 的关键源码 SHA-256（初次 105 项验收后的 Session 已有版本变化）：

| 文件 | SHA-256 |
| --- | --- |
| `engine.py` | `adad4b58f9fddcc5a2b1f554ba51f3abe8ae60847b650ada2f34b53fe76f154d` |
| `session.py` | `f59731de25e14c28da185cb14bda4d1f1169aec8a093d5a9f0ddec90dd277551` |
| `lineage.py` | `1a2b6dabe3317a0826005deb3882de914dbc749594ab2eea26e6d6f9054432e7` |
| `evidence.py` | `a500113219525111c321e1d422cf851d54ba9403b0aef409b84b2a8b1210c687` |

最终测试覆盖原子 `run_with_receipt` 与并发收据回归。历史 v3 成本报告对应 [source-v3 归档](../experiments/source-v3/)；不能把旧源码摘要写成当前源码。独立完成的 [最终成本 v4](../experiments/updates-final-v4.json) 已绑定当前 engine/store/runner 摘要；其成本与计时口径见结果报告，不能与这里的正确性检查混为同一次计时。

## 版本与分发边界

对最终 `src/`、`scripts/`、`tests/` 共 41 个 Python 文件使用 Python 3.10 语法模式解析，零语法错误。**本机没有运行 Python 3.10**，静态检查不证明其运行时和依赖组合通过。

本次构建成功的 wheel 包含 Python 模块，却不包含 `data/public/pypi-catalog.json` 或仓库根目录的 `scripts/`。因此当前受支持的复现入口是 **保留完整仓库的源码 checkout + editable 安装**；不能把该 wheel 当成自带数据和训练工具的独立分发包。没有测试从 sdist 或独立 wheel 完成全部演示，也没有发布 PyPI 包。

本页 CPU 验收步骤未运行模型或调用 GPU。整个项目另已实际完成本机训练/评测；其行为与成本必须读取独立模型收据，不能从 CPU 测试通过推断。GitHub 发布结果以仓库与 Release 的实际回读核验为准。

## GitHub 源码与远端 CI

源码已发布到 [1766421040-ux/deltaweave](https://github.com/1766421040-ux/deltaweave)，实验发布提交为 `e5877dd13ef655fece6faa73d40bd0d6b00f477e`。179 个文件逐一核对 Git blob 摘要，远端树与本地发布索引均为 `18ecc3333d1cda17370b427124b547d45ef75924`；随后实际 fetch 回读一致。

[GitHub Actions 40638152693](https://github.com/1766421040-ux/deltaweave/actions/runs/40638152693) 已完成，Ubuntu 的 Python 3.10 与 3.12 两个 job 均为 success。它们执行 CPU 测试、SQLite demo 和独立领域基准，不下载或运行模型。模型运行证据来自本机单独保存的收据。

加入分片恢复、其回归测试及完整结果说明的提交为 `d7d25ad2d429855ce716738b7292ccca6542c09f`，源码树 `f725cc4f806d6756fbfec2b6e2c2050bef2e3853` 已实际 fetch 比对。[该版本 CI 40640055032](https://github.com/1766421040-ux/deltaweave/actions/runs/40640055032) 的 Python 3.10 / 3.12 均已通过。

## 权重与训练数据分发

完整 ZIP 为 34,295,413 字节，SHA-256 为 `b9078114f7df68acd8edc36022f3c51f75bbf0a0fb1eaa683daf3600098d86fb`。首次整包上传被服务端中断，因此改为 9 个最大 4 MiB 的分片；拆分没有改变原始 ZIP、两个 adapter 或任何训练数据字节。仓库内 [release manifest](../artifacts/release.json) 绑定分片顺序、长度、摘要与 13 个文件的摘要。

`fetch_release.py` 只访问固定 GitHub Release URL，逐片校验并缓存，合并后再次核对完整 ZIP，随后核对成员白名单与解压摘要；既有目标文件不相同则拒绝覆盖。11 个分发测试包含模拟网络中断、已完成分片复用、错误顺序/URL/大小、分片损坏、合并摘要不一致和解压路径攻击。公开下载的实际验收状态以 manifest 的 `publication_status` 和后续验证收据为准。

[v0.1.0 研究预览版](https://github.com/1766421040-ux/deltaweave/releases/tag/v0.1.0) 已公开发布，标签绑定 `d7d25ad2d429855ce716738b7292ccca6542c09f`。从不带认证的公开下载链接出发，在全新缓存和全新恢复目录内实际完成九片下载、整包合并及全部 13 个文件校验，共恢复 40,943,664 字节。GitHub 记录的每个已上传附件摘要也与 manifest 一致。重新恢复时跳过全部 13 个一致文件；两个 adapter 均通过底座摘要、训练配置与监督 token 预算绑定检查。详见 [完整公开下载收据](../experiments/release-verification.json)。这只验证传输、文件及来源一致，不新增模型质量结论。
