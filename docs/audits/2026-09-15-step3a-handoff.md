# Step 3A：不可验证时长的披露、索引诊断与兼容性门禁命名

2026-09-15 首轮，2026-09-16 按 Codex 复核意见修订。
基线 `95f6d09e0bfc99f1022e45c72fbdebcd3465d6a0`（#88 合并后的 main）。
本轮由 CC 实施，交由 Codex 独立复核；未调用其他 reviewer，未自动合并。

## 复核轮次

| 轮次 | head | 结论 |
|---|---|---|
| 首次提交 | `e05a08a4f13caccc086fc27426cb039ee6f8ea6e` | Codex：需修改。一个 P2 加三个小项 |
| 本次修订 | 见 PR #89 的 Checks | 待复核 |

Codex 首轮的 P2 与三个小项，以及本轮的处理：

| 问题 | 处理 |
|---|---|
| **P2**：47 个正价小时 + 一行 `NaT / -50`，三端都显示确定的零小时。`groupby(prices.index.date)` 默认丢弃 NaT 键，该行永远进不到交割网格验证 | 在分组**之前**识别交割时刻未设定的记录，返回不可用时长与明确原因；区间计数、比例、均值与极值全部保留。补统计／页面／Excel／PDF 四端回归 |
| Excel 新原因行标签（40 字符）被 30 字宽的 A 列裁切，B 列非空 | `_write_kv_pair(wrap_key=True)` 对该行换行并按行数抬高行高；宽度上限与每行高度提为 `_MAX_COLUMN_WIDTH` / `_ROW_HEIGHT_PER_LINE` |
| `git diff --check 95f6d09 e05a08a` 实际 19 处空白告警，全部来自冻结补丁；首轮交接称“无输出”不成立 | 按前两轮证据目录的惯例加 `docs/audits/2026-09-15-step3a-evidence/.gitattributes`（`*.patch -text whitespace=-blank-at-eol`），补丁字节不变、hash 不变；审计索引补一句说明 |
| 合同引用了不存在的 `negative_price_hours_display` | 改为实际的 `negative_price_hours_reason`，并写明共享的是“是否可用 + 原因”，各消费端保留自身数值格式 |
| 首轮交接把基线页面写成“显示 `n/a` 但无原因” | 更正：基线页面已有一条通用 caption（“delivery interval grid cannot be verified”）。本轮新增的是**共享的具体原因**（首个不可验证的本地日期／未设定时刻的条数）与观测计数的同句披露；从 NaN／字面 `nan` 改成 `n/a` + 原因的是 Excel 和 PDF |

## 问题与最终行为

| 触发情况 | 基线 `95f6d09` | Step 3A |
|---|---|---|
| 负价小时的交割网格无法验证 | 页面显示 `n/a`，caption 只说“网格无法验证”，不指出是哪一天；Excel 写入 NaN，读回是**空单元格**（`None`，数字型）；PDF 打印字面 `nan` | 三处共享同一条原因（含首个不可验证的本地日期），统一显示 `n/a`；Excel 增加原因行，PDF 增加原因行 |
| 负价行的交割时刻未设定（NaT） | 三端均显示确定的 **0 小时**、无原因——该行被 `groupby` 静默丢弃 | 时长不可用，原因写明未设定交割时刻的条数；1 个区间、2.08%、均值与极值仍是真实数字 |
| 同一情况下的区间计数 | 保留 | 保留：区间数、比例、均值与极值均为真实数字，未被置零 |
| 价格索引有重复时间戳 | `render()` 抛 `ValueError`，整页裸异常 | 原始价格线照常绘制（不去重、不排序），仅均线不可用并以 `st.warning` 说明 |
| 价格索引未排序或含 NaT | 同上，整页裸异常 | 不绘制价格图，`st.error` 说明原因，并清除 `report_figures["price_ts"]`，旧图不会进入 PDF |
| Excel 原因行的标签 | —（无此行） | 标签换行并抬高行高，不被 B 列裁切 |
| CI 第二个 job 的命名 | `minimum-ui` / “minimum supported Python / Streamlit”，读起来像全依赖最低版本测试 | `Python 3.11 / Streamlit 1.55.0 compatibility check`，并在配置中注明其余依赖仍按 manifest 解析 |
| reserve 平均功率的时长权重 | 顺序／随机路径用算术平均，前提未记录 | 合同与代码注释记录 uniform-only 前提及未来向量化的前置条件；**本轮不改任何模型或现金** |

以上均为 synthetic 已知答案，不是市场收益预测。

## 设计决定

- `describe_price_index_issue` 是“可用价格索引”的**唯一定义**，
  `time_weighted_rolling_price_mean` 继续严格拒绝并复用同一条原因，
  页面在调用前先分类而不是捕获它的异常——因此无关程序错误不会被伪装成数据不可用。
- 页面按**图表边界**分级：重复时间戳仍可按交割顺序如实绘制，只停掉均线；
  未排序或 NaT 会让“按行序连线”失真，因此整张图不出，并清除对应导出图。
  任何分支都不排序、不去重、不补点、不填零。
- 交割时刻未设定的记录与“某一天网格无法验证”走**同一条口径**：只要窗口里存在
  无法确定物理时长的记录，负价小时就是不可用，而不是把它当成零贡献。这与基线
  已有的“任一天不可验证即整窗不可用”保持一致，没有引入第二套判定。
- `negative_price_hours_reason` 只集中“是否可用 + 原因”。各消费端保留原有数值
  格式（页面一位小数、Excel 数字单元格、PDF 原字符串），所以本轮不引入任何
  可用数值的格式回归；Excel 仍是数字型单元格，未被改成字符串。
- 两条原因串都在横向 A4 的 177 mm 值列内（151.6 mm 与 157.2 mm），有实测断言，
  因此 PDF 不需要换行；Excel 的裁切是列宽问题，只对该行换行处理。

## 边界（本轮明确未做）

- 未放宽 DA/IDA 网格守卫，未改 SoC、段末中性、576 点上限、FEC、未来数据可见性，
  未触及 Project Case 公共 schema/fingerprint 与现金流合同。
- 未改整体 Avg Price 口径（Step 3C）、未做结果持久化（Step 3B）、
  未做 DST 结算披露（Step 3D）、未做 Vault housekeeping（Step 4）。
- 未泛化顺序／随机策略的 vector dt；F4 只落文档与注释。
- 未写入真实市场缓存；未删除或改写既有测试。
- 未改变“任一天／任一条记录不可验证即整窗时长不可用”的既有保守口径：
  即便不可验证的那一天本身没有负价，也仍然报不可用。收紧到“只在不可验证的
  范围内确实出现负价时才不可用”属于口径变更，不夹带在本展示 PR。

## 冻结与独立复现

- 基线：`95f6d09e0bfc99f1022e45c72fbdebcd3465d6a0`。分支 `step3a-display-contract-repairs`。
- [冻结代码补丁](2026-09-15-step3a-evidence/step3a-code.patch)，SHA-256：
  `64989c7c1c5ac116a6d799b82310c169fb9b557a6ea7dcc24ef267090183c560`。
- hash 范围为 `src/ tests/ .github/workflows/ci.yml`；文档与日志不在代码 hash 内。
  新测试文件已加入版本控制，不会因 untracked 漏出补丁。
  Step 1、1b、1b-R2、Step 2 的原有补丁保持原字节。
- 证据目录的 `.gitattributes` 只抑制 `*.patch` 的空白告警，补丁本身一字节未改。

在本分支检出执行：

```sh
git diff --binary --full-index 95f6d09e0bfc99f1022e45c72fbdebcd3465d6a0 HEAD -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-15-step3a-evidence/step3a-code.patch
```

## 验证

```sh
.venv/bin/python -m pytest tests/test_step3a_display_contract.py -q
.venv/bin/python -m pytest tests/test_analytics.py tests/test_export.py \
    tests/test_delivery_durations.py tests/test_step2_views.py \
    tests/test_market_grid_guards.py tests/test_project_case_export.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff --check 95f6d09 HEAD
PYTHONPATH=. .venv/bin/python docs/audits/2026-09-15-step3a-evidence/step3a-probe.py
```

新增 27 个用例：在基线 `95f6d09` 上 **20 个失败、7 个通过**；本轮 27 个全通过。
那 7 个即兼容对照。基线红测见
[baseline failures](2026-09-15-step3a-evidence/step3a-baseline-failures.txt)，
通过记录见 [green tests](2026-09-15-step3a-evidence/step3a-green-tests.txt)。

探针对照（同一脚本，两个版本）：
[基线输出](2026-09-15-step3a-evidence/step3a-probe-baseline.txt) 含三次
`Uncaught app execution`、PDF 的字面 `nan`，以及 `[1b]` 的 `negative_hours = 0.0`；
[本轮输出](2026-09-15-step3a-evidence/step3a-probe-head.txt) 无异常，`[1b]` 为
不可用加原因，`[1c]` 显示标签换行与 30.0 的行高。
探针读取真实生成的 `.xlsx` 与 `.pdf` 字节，写在临时目录，退出即删除。

兼容对照覆盖：可验证的 48 负价小时、零负价小时、干净索引的图表与均线。
另有三个用例断言坏索引的原始行仍然完整写入 Excel 的 `Hourly Prices` 页
（逐值相等），证明本轮只是不画图，没有隐藏数据。

本轮实测（Python 3.13 / Streamlit 1.55.0 / pandas 2.3.3 / SciPy 1.17.1）：

| 命令 | 结果 |
|---|---|
| `pytest tests/ -q`（含 slow） | **1871 passed / 2 skipped，242.02 秒** |
| `pytest tests/ -q --collect-only` | 1873 collected，其中 22 个 slow |
| 针对性回归（6 个相关测试文件） | 329 passed / 2 skipped |
| `ruff check src/ app.py tests/` | 通过 |
| `git diff --check 95f6d09 HEAD` | 无输出 |
| 兼容性 job 的本地等价选择 | 12 passed / 89 deselected |

基线 `95f6d09` 的最近完整验收为 1844 passed / 2 skipped（1846 collected）；
本轮 +27 个新用例，与 1871 passed / 1873 collected 一致。
那 2 个 skip 是既有的 opt-in 图表渲染跳过，**不代表渲染通过**。

## 未完成与限制

- 兼容性 job 的名字变化只影响该 check 的显示名；main 的分支保护只要求 `test`，
  已核对，不影响门禁。改名后的远端运行状态以本 PR 的 Checks 为准。
- PDF 可读性断言是按 fpdf2 字体度量计算的行宽，不是渲染截图。Codex 首轮已用
  Poppler 对本轮 summary 的原因行做了实际渲染核对；带图 PDF 未做渲染验收，
  两项 opt-in 图表渲染跳过仍未证明 PDF 图表渲染成功。
- Excel 的换行断言读的是保存后文件的 `wrap_text` 与行高，不是在原生 Excel
  应用里的视觉验收。
- 真实浏览器验证留给 Step 3C 的布局项；本轮的页面验证是 AppTest 驱动真实面板。
