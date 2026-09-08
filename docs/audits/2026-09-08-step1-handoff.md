# Step 1 交接：连续回放分辨率切段、DA/IDA 双侧校验、非有限输入

历史验收记录，范围锁定于 `3823acd`。#86 已于 2026-09-08 合并，merge commit 为 `850343def9d69a0774570a4112db6d6498002dce`。下文 Step 1b 的“待修”描述是当时发现的残留；当前状态见 [证据索引](README.md)。

2026-09-08。实现依据：[CC Step 1 规格](2026-09-08-step1-spec.md)。

规格内改动已完成；F1 和 F2 均不能标记为全局关闭。下文分别列出 Step 2 的日内切换残留，以及本轮补查确认的其他 DA/IDA 消费路径残留。README / CLAUDE.md / Vault housekeeping 按规格留到 Step 4。

## CC 独立验收：通过

用户于 2026-09-08 提供 CC 的复核报告，结论为 **Step 1 通过，可以接受**。以下为该报告提供的独立验证结果，与后文 Codex 自测记录分别归属：

- 在基线 worktree 中仅复制 7 个新测试文件，得到 **86 FAIL / 4 PASS**；T1 和 T2 的旧数值均复现。
- 独立全套 **1675 passed / 2 skipped，281.9 秒**；Ruff、diff check 通过。7 个测试文件仅新增 **634 行**，没有删除或改写旧测试。
- 独立核对源码补丁 hash、五条剩余 dispatch 入口以及 uplift 单侧覆盖率问题。
- 额外探针：Berlin 前拨／回拨窗口均未误计 cadence split；四类 parser 的 header-only CSV 和整数列行为保持兼容。

收到报告后，Codex 再次直接核对本地工作树与冻结补丁，两者仍是下列 `c2e70d3…b9c681`，14 个源码／测试文件一致。本次仅补充审核记录和 [PR 描述草稿](2026-09-08-step1-pr.md)，源码和测试未改，因此没有重复运行已独立通过的全套。

三条非阻塞意见的处置：

1. **Step 1b**：将行数不匹配报错中的 incomplete coverage 提到 resolution mismatch 前面，减少缺一格时的误导；继续保留三侧行数和 `invalid_input`。
2. **Step 4**：文档明确 `n_cadence_splits` 统计实际因 cadence 变化发生的连续段切分。若缺口／NaN／稀疏日已截断连续段，计数可以为 0；它不是任意跨分辨率窗口的布尔标识。
3. 私有 finite helper 的跨模块引用保持现有实现；这一命名意见不构成本轮新增重构任务。

交付顺序：先将已验收的 Step 1 独立走分支／PR，再进行 Step 1b 的剩余五条策略入口及 Revenue uplift 修复，之后进入 Step 2。

## 复核对象

- 基线 commit：`8f9eab71253b357e96379e225851af458e5f653a`。
- Step 1 已提交为 `3823acd5bdb23f65607cce91894dbfcbbb1f4e4f`，并创建 [PR #86](https://github.com/crashchen/euro-bess-radar/pull/86)。源码和测试保持 CC 验收的补丁 hash；额外提交了可随仓库发布的行为说明。未调用 CC。
- [冻结源码与测试补丁](2026-09-08-evidence/step1-source.patch) 包含 14 个文件；审核说明和诊断材料不包含在补丁内。
- 补丁 SHA-256：`c2e70d354cf86e01029fe77c93570063af3e477759b7aa48fdf7667a2b89c681`。

同 hash 复核前，在项目根目录执行以下两条，结果应相同。第一条核对已发布的 Step 1 commit，第二条核对冻结补丁。当前开发分支已进入 Step 1b，因此不再用其工作树代表 Step 1。

```sh
git diff --binary --full-index 8f9eab71253b357e96379e225851af458e5f653a 3823acd5bdb23f65607cce91894dbfcbbb1f4e4f -- src/ tests/ | shasum -a 256
shasum -a 256 docs/audits/2026-09-08-evidence/step1-source.patch
```

## 实际行为变化

1. **F1 / C1**：`simulation._group_clean_runs` 在完整、规则日之间遇到 cadence 变化时，沿用 size-cap 的 soft reset：flush 后保留 `previous_date`，当日正常进入新段，SoC 继续结转，段尾重新施加 terminal-neutral 等式。普通 DA 和连续 DA+IDA 共用此修复。没有改变 dt 推断、规则日校验或 576 区间上限。
2. **披露**：所有 `simulate_replay_batch` 返回值均提供整数 `n_cadence_splits`，空窗口、单日、per-day 模式也有稳定的 0。日期缺口和单纯的 size-cap 切段不算市场分辨率切换。多日面板只在计数大于 0 时，在 KPI 后显示说明；常量文字和实际按钮触发后的 caption 有独立 pin。
3. **F2 / C2**：单日普通 DA+IDA 在 inner join 后检查两侧原始行数；Revenue 的 `calculate_two_stage_da_id_dispatch` 增加 IDA 侧的 90% 覆盖闸门，继续接受原有 23/24 小时稀疏样本。缺失仍计入 missing，错误状态继续使用 `invalid_input`。
4. **对规格的最小补全**：默认连续 DA+IDA 绕过单日函数，也有同样的 join 漏口，因此在 `build_da_id_day` 同步加入双侧行数检查。没有 forward-fill DA，也没有扩大到 forecast / reserve / stochastic 路径。
5. **F3 / C3**：11 个既有价格入口改为 finite 检查，±inf 返回原有 typed failure；合法的无限 rebid cap、容量向量清洗和零容量价格 skip 次序保持不变。IDA、capacity、activation、imbalance 四个 CSV parser 复用一个私有 helper，在原有丢行流程前将非有限必填数值转为 NaN，按行计数记录 warning。既有全丢弃时的异常／空帧及元数据校验次序保持不变。

## 关键结果

| 案例 | 旧代码 | 修后 |
|---|---|---|
| T1：5 个小时日 + 1 个 15 分钟日 | €3,792 / 96 MWh 吞吐 / 48 FCE | €948 / 24 MWh 吞吐 / 12 FCE；6 天全部保留 |
| T2：1 个小时日 + 1 个 15 分钟日 | 小时日 €237 / 6 MWh | €316 / 8 MWh，与 standalone 相等 |
| T4/T5：24 DA / 96 IDA | join 剩 24 行仍被接受 | 普通回放拒绝；Revenue two-stage 排除该日 |
| 稀疏 IDA 兼容控制 | Revenue 接受 23/24 行 | 保持接受；cockpit 仍拒绝不完整日 |
| T6：IDA CSV 有 1 行 inf | 24 行落库，含 inf | 23 行落库，0 个 inf，warning 记录 1 行 |
| T7：直接输入 ±inf | 可能抛 SciPy ValueError 或返回无效成功值 | 现有 `invalid_input` typed failure |

T1 同时钉住两种连续回放模式、每日物理吞吐量上界和不丢日；T2 同时对照真实单日求解结果。上述数值来自 synthetic 本地真实求解，不是 live 市场收益。

CC 对严重性和修复范围的纠正成立：F1 可以高估 300%，不只低估 25%；本轮可用切段先修，无需先完成全局 dt 向量工程。另作单位澄清：T1 的 96 MWh 是充放电总吞吐量，不是单独放电量；旧结果对应放电 48 MWh。

## 红测与验证

新增 **90 个参数展开后的用例**：**86 个有旧代码失败证据，4 个为旧代码也通过的兼容控制**。兼容控制不冒充 mutation proof。

| 对应 finding / 要求 | 新增用例 | 旧代码结果 | 修后针对性检查 |
|---|---:|---|---|
| simulation：T1–T4、T7、切段计数 | 23 | 首批 16 FAIL；后补计数 5 FAIL，NaN 兼容 2 PASS | simulation 全文件 81 PASS |
| analytics：T5 | 2 | 细粒度 IDA 1 FAIL，23/24 兼容 1 PASS | analytics 全文件 99 PASS |
| UI：C1 caption | 3 | 常量和真实面板渲染 2 FAIL，不切段兼容 1 PASS | ui_theme 全文件 13 PASS |
| solver：C3-a / T7 | 48 | ±inf 48 FAIL | 48 PASS；相关非 slow 216 PASS |
| import：C3-b / T6 | 14 | ±inf 14 FAIL | 14 PASS；两文件 307 PASS |

首批 simulation、analytics、UI、solver、import 红测均在对应源码修改前运行。后补的 5 个计数测试使用 `git show HEAD:src/simulation.py` 的隔离模块复跑，未回滚共享工作树；它们因旧实现缺少披露 attr 而失败。两条 NaN 控制在同一隔离运行中通过。

- [红测原始输出](2026-09-08-evidence/step1-red-tests.txt)：保留断言、节点和计数，仅将本机项目绝对路径替换为 `<repo>`。
- [修后诊断输出](2026-09-08-evidence/step1-results.txt)：保留诊断结果及 numeric-drop warning，省略 Streamlit bare-mode 噪声；原始 baseline `results.txt` 未覆盖。
- 诊断命令：`PYTHONPATH=. .venv/bin/python docs/audits/2026-09-08-evidence/reproduce.py`。脚本仍不进入 CI；曾漏写 `PYTHONPATH=.` 导致 import 失败，使用报告中的完整命令后成功。
- 全套：`.venv/bin/python -m pytest tests/ -q` → **1675 passed / 2 skipped / 1677 collected，296.15 秒**。[完整输出](2026-09-08-evidence/step1-full-tests.txt)。两个 skip 是既有 opt-in 图表渲染测试。
- `.venv/bin/ruff check src/ app.py tests/` 和 `git diff --check`：通过。
- 独立 slow 门禁：`.venv/bin/python -m pytest tests/ -m slow -q` → **21 passed / 1656 deselected，290.48 秒**。[完整输出](2026-09-08-evidence/step1-slow-tests.txt)。最终门禁后再次核对工作树与冻结补丁 hash，保持一致。

## 未修残留及建议顺序

### 先补 Step 1b：其他 F2 消费面

本轮只读扩展检查确认，另外五个公共 batch 路径仍接受 **24 DA / 96 IDA**：

- `simulate_da_id_reserve_ceiling_batch`
- `simulate_sequential_da_id_batch`
- `simulate_sequential_da_id_reserve_batch`
- `simulate_stochastic_da_id_batch`
- `simulate_stochastic_triple_batch`

[可复跑探针](2026-09-08-evidence/step1-residual-probe.py) 与 [输出](2026-09-08-evidence/step1-residual-probe.json) 显示：均返回 `valid_days=1`、`excluded_days=0`、`model_available=True`，实际求解器只收到 24 点、dt=1h，75% IDA 区间丢失。探针使用真实 forecast/scenario 和真实本地 solver；包装器只记录长度/dt，然后原样调用原函数，没有替换经济结果。该夹具各路径现金结果均为 0，不据此声称已量化所有策略的收益偏差。

探针使用两天 Berlin 数据，前一天作为 walk-forward 历史；reserve 为 0/None，stochastic 为 S=1、seed=7。因此它证明这些公共路径可接受错网格，不单独量化正 reserve / 多场景的经济影响。复跑：`PYTHONPATH=. .venv/bin/python docs/audits/2026-09-08-evidence/step1-residual-probe.py`。

Revenue 页还调用 `calculate_intraday_uplift`：相同两日 48 DA / 192 IDA 输入只留下 48 个区间，但报告 `coverage_pct=100%`。这不是本轮已修的 two-stage 函数，仍需修正覆盖分母与不匹配网格处理。

建议在这些入口对 join 前后的双侧交割网格做明确校验，并为每条公共路径加入先红后绿回归；沿用 missing/invalid_input，不补齐 DA 网格。此项应优先于新功能及下一轮视觉工作。forecast/scenario 构造本身仍保留 96 点；`compute_forecast_skill` 的 DA baseline 只有 24 个匹配点，属于另一个需要披露基准覆盖率的问题，不能误报为 forecast 已丢掉细网格。

### Step 2：日内切换、时间窗口和依赖

BG / EE / FI / GR / LT / LV / PT / RO 八区切换点落在本地日内部；本轮保留规则日拒绝策略。`solve_joint_capacity_batch` 的 FI 诊断仍输出 **€110.4375，预期 €114**。per-interval dt 留在独立实现中，不能因本轮切段修复而关闭这项。

同时处理 F4 的 720 行均线／最小样本时间语义和 F8 的 Streamlit 依赖下界。

### Step 3–4：交互、视觉与 housekeeping

F6 结果留存复用现有 fingerprint 模式；F7 优先让 Project Case 分位标识在 KPI 中可辨认；F5 优先明确既有 DST 容量结算口径。最后按已验证源码更新 README、CLAUDE.md、测试命令／计数和 Vault 项目笔记。本轮没有修改这些正文或 Vault。

PDF 验证的措辞也应在文档轮修正：`_render_or_skip` 确实会把渲染异常转成 skip，但“两个测试结构上不可能失败”过强；渲染后仍有 PDF/PNG 签名和大小断言。问题是渲染错误可能被跳过掩盖，不能将默认全套通过视为图表 PDF 已验收。

## 原 CC 复核请求（已完成）

当时的复核范围：先核对本文件的基线和补丁 SHA-256，再按 Step 1 规格检查 cadence 分段是否沿用 soft reset 且没有丢日、T1 是否有真实红测、Revenue 对称闸门是否保留 23/24、finite 守卫是否保持失败契约，以及 caption 是否真实显示。Step 1b 和日内切换残留与本轮回归分开判断；本实现不宣称 F1/F2 已全局修复。验收结果见本文开头。
