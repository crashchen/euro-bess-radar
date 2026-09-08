# Step 1b 交接：剩余策略入口与 uplift 的交割网格校验

**历史快照**：本文记录 `0b5d718`／`a5a8f6ad…` 的首轮实现与验收证据，不代表当前 PR 状态。用户随后提供的 CC 复审确认其测试结果，但指出孤立报价导致的 0→1 可用性反转；该问题已按日期排除修正。#86 已合并，#87 已转向 main。当前行为、增量补丁与验证见 [第二轮交接](2026-09-08-step1b-r2-handoff.md)。本文以下内容保留为首轮历史记录，包括当时较严格的整窗拒绝规则。

2026-09-08。承接用户提供的 CC Step 1 验收意见及“继续”授权。

Step 1 已发布为 [PR #86](https://github.com/crashchen/euro-bess-radar/pull/86)，提交 `3823acd5bdb23f65607cce91894dbfcbbb1f4e4f`，GitHub CI 已通过。Step 1b 在独立分支 `codex/replay-market-grid-step1b` 开发；其 PR 以 Step 1 分支为基线，便于分别复核。两项均未自动合并，未调用 CC。

**CI 状态**：#86 已通过。`.github/workflows/ci.yml` 仅对 main push／目标为 main 的 PR 触发，#87 当前以 Step 1 分支为基线，因此没有排队中的 CI run。先合并 #86，再将 #87 转向 main 并验证 CI 后合并；不能将当前本地通过误写为 #87 的远端 CI 已通过。

## 复核对象

- 基线：`3823acd5bdb23f65607cce91894dbfcbbb1f4e4f`。
- [源码／测试冻结补丁](2026-09-08-evidence/step1b-source.patch)：`a5a8f6adb1ca308b8bed9b95242b8c52cc48343bb4a14cbf2e3344278ed40f12`（SHA-256）。
- 该 hash 覆盖 3 个源码文件及 2 个测试文件；提交另包含 `docs/design/replay-input-guards.md` 的行为说明更新。
- 提交：`0b5d7181c4d2ec7a6e8e85f7fdd36f8cedf2cd97`；[草稿 PR #87](https://github.com/crashchen/euro-bess-radar/pull/87)，基线为 Step 1 分支。

同 hash 复核命令（两条输出应一致）：

```sh
git diff --binary --full-index 3823acd5bdb23f65607cce91894dbfcbbb1f4e4f 0b5d7181c4d2ec7a6e8e85f7fdd36f8cedf2cd97 -- src/ tests/ | shasum -a 256
shasum -a 256 docs/audits/2026-09-08-evidence/step1b-source.patch
```

审核材料本身不包含在该源码 hash 内。发布后已核对 commit 与冻结补丁一致。

## 行为与边界

1. 五条公共路径（reserve ceiling、sequential、sequential reserve、stochastic、stochastic triple）共用 `_merge_matching_da_id_day`。原连续普通 DA/IDA 也使用该 helper，维持 Step 1 行为。inner join 必须保留两侧原始行数，然后才走原有 NaN／完整规则日检查。错网格仍计作 missing，不调用 solver；没有新的 failure status。
2. forecast/scenario 仍在原生 IDA 时间戳上构造，只在市场网格通过后对齐。未改变 walk-forward、场景、reserve skip 次序、SoC、dt 或最优化目标。
3. 普通单日回放的报错先说 coverage incomplete，再说明可能的 resolution mismatch，并保留 DA／IDA／merged 三个行数，落实 CC 的非阻塞意见。
4. Revenue uplift 对每个有两侧数据的本地日核对推断 cadence 和时间戳相位。重复时间戳、cadence 不同、偏离网格或某侧少于两个观测时，**整个窗口不可用**，不悄悄换成一个更小的有效窗口。可确认同网格的稀疏样本保留；两侧一起在日界改变 cadence 的窗口可以使用。
5. 新增 `model_available`／`reason` 及两侧 `da_coverage_pct`／`ida_coverage_pct`。后两者为有限价格匹配数除以各自原始输入行数；可用样本以两者较低值折算年度 uplift。不可用时 `n_periods=0`，页面显示原因，不展示收益指标。原始重叠比即使非零，也不代表市场模型可用。
6. 补充堵住该诊断的旧缓存非有限值入口：仅 finite price pairs 进入 uplift 和直方图，并反映到覆盖率。既有 importer／solver 守卫不变。

覆盖率仍以区间数量计，不是时长或能量覆盖率。非常稀疏的记录若无法确认 cadence，会保守拒绝；若要在这类记录中区分“缺失报价”和“更长交割周期”，需后续引入明确的 delivery-duration 元数据。

## 验证证据

新增 **72 个参数展开后的用例**，合计 **37 个有基线失败证据、35 个为兼容控制**。首批 68 个在未改源码的 Step 1 基线运行：**34 FAIL / 34 PASS，3.71 秒**。之后三个公共适配器拒绝用例在完整的 `git archive 3823acd` 临时基线检出中全部 FAIL；新增的 2026 年春季 DST 兼容用例在同一基线 PASS。失败用例包含新增披露字段／措辞契约，不能全部称作已有经济计算错误。

| 范围 | 数量 | 基线结果 |
|---|---:|---|
| 五条公共路径、两个网格方向，含正／零 reserve | 16 | FAIL |
| 同网格完整／缺行／NaN／15 分钟公共路径 | 32 | PASS（兼容） |
| uplift 两个方向和窗口内单日不匹配 | 3 | FAIL |
| 重复、相位偏移、孤立观测的不可用状态 | 4 | FAIL |
| 两侧覆盖率、稀疏 DA/IDA、日界 cadence 切换 | 4 | FAIL（含新增披露契约） |
| 两个价格列各 ±inf | 4 | FAIL |
| 普通回放 coverage-first 报错 | 1 | FAIL |
| 实际 Revenue 面板不可用提示及两侧覆盖率 | 2 | FAIL |
| Berlin 前拨／回拨日 uplift | 2 | PASS（兼容） |
| Project Case 原生异网格拒绝 | 3 | FAIL |
| 2026 春季 DST 原生同网格容量结算 | 1 | PASS（兼容） |

公共路径测试使用真实 forecast、真实本地 solver、两个生成场景；包含正 reserve，可用路径也实际求解，没有 fake 成功结果。UI AppTest 使用真实面板与真实 analytics，提前放入 synthetic session cache，不访问网络或真实市场缓存。

- [基线红测输出](2026-09-08-evidence/step1b-red-tests.txt)。命令：`.venv/bin/python -m pytest tests/test_market_grid_guards.py -q`。
- 新文件修后 **68/68**；包含 simulation、analytics、ui_theme 的[针对性回归](2026-09-08-evidence/step1b-targeted-tests.txt) **261/261，7.12 秒**。
- `.venv/bin/ruff check src/ app.py tests/`、`git diff --check` 通过。
- 首轮全套：**1740 PASS / 3 FAIL / 2 SKIP**；[保留原始失败记录](2026-09-08-evidence/step1b-first-full-tests.txt)。失败均为下述依赖已禁止行为的 Project Case 用例，未放宽生产守卫来恢复旧结果。
- 新增适配器拒绝的[基线输出](2026-09-08-evidence/step1b-adapter-red-tests.txt)：3 FAIL；新增 spring DST 的[基线兼容输出](2026-09-08-evidence/step1b-dst-control-baseline.txt)：1 PASS。全代码由 `git archive` 检出，非 fake runner 或仅替换一个模块。
- 调整后适配器全文件：[30/30 PASS](2026-09-08-evidence/step1b-adapter-final-tests.txt)。
- 最终全套：**1747 PASS / 2 SKIP / 1749 collected，264.66 秒**；[完整输出](2026-09-08-evidence/step1b-full-tests.txt)。本次包含全部 **22 个 slow 用例**（新增的 2026 春季 DST 用例使 slow 从 21 增为 22），未重复单独运行 slow。Ruff、语法编译和 diff check 均通过；最终源码／测试补丁 hash 已再次核对。
- [修后残留探针输出](2026-09-08-evidence/step1b-residual-probe.json)：原五条公共路径全部 `valid_days=0 / excluded_days=1 / model_available=False`，记录到的 solver 调用均为空。uplift 为不可用、0 个有效比较区间，两侧原始 timestamp overlap 为 DA 100%／IDA 25%。复跑脚本沿用 [Step 1 探针](2026-09-08-evidence/step1-residual-probe.py)。

## 三个旧集成用例的明确调整

首轮全套查出 DE_LU / FR 的 non-UTC 正常求解用例用的是 2025 年 6 月：完整原生 DA 为小时、IDA 为 15 分钟，其成功依赖 lossy join。现在改用 2025-11-01..03 原生同网格；保留真实 solver、首日 walk-forward 缺失、后两日有效、有限现金和 fingerprint 断言。没有把 IDA 降采样或把 DA 补成细网格。

2025 春季 DST 的 DA+IDA+reserve 用例也依赖异网格丢行，现在断言原生行数确为 4 倍并要求 `AdapterUnavailableError`。同一用例中的 DA-only reserve **€480** 断言仍执行；秋季同网格 triple **€480** 也保留。另加 **2026 春季 DST 同网格 triple €480**，使春季容量结算的正向端到端覆盖仍在。新增三条 public-adapter 负向测试独立证明基线曾错误返回可指纹化结果。

因此本轮不声称“旧测试完全未改”；上述三条行为／夹具变更是已宣布的输入契约收紧，文件 diff 可独立复核。

## 仍然保留的后续工作

- Step 2：BG / EE / FI / GR / LT / LV / PT / RO 日内切换的 dt 契约；FI joint capacity 的 €110.4375 对 €114 残差仍未修。
- `compute_forecast_skill` 的 DA baseline 匹配点数／覆盖率披露另列后续事项；本轮没有改 forecast 生成器或其评分。
- F4 均线、F8 依赖下界，以及后续 F6 留存、F7 KPI、F5 DST 口径披露。
- Step 4：README／CLAUDE.md／Vault housekeeping；明确 `n_cadence_splits` 是实际切分次数。Vault 未修改。

给 CC 的复核重点：在 Step 1 基线上只复制新增测试能否得到同样红／兼容计数；五条入口是否在丢行之前拒绝；正 reserve 与非 UTC 场景是否保留；uplift 原始重叠比例是否与可用性明确区分；同网格稀疏与 DST 行为是否保持。用户自行调用 CC，本任务未发起外部复核。
