# Step 3A：不可验证时长的披露、索引诊断与兼容性门禁命名

2026-09-15。基线 `95f6d09e0bfc99f1022e45c72fbdebcd3465d6a0`（#88 合并后的 main）。
本轮由 CC 实施，交由 Codex 独立复核；未调用其他 reviewer，未自动合并。

## 问题与最终行为

| 触发情况 | 基线 `95f6d09` | Step 3A |
|---|---|---|
| 负价小时的交割网格无法验证 | 页面显示 `n/a` 但无原因；Excel 写入 NaN，读回是**空单元格**（`None`，数字型）；PDF 打印字面 `nan` | 三处统一显示 `n/a`，并给出同一条原因（含首个不可验证的本地日期）；Excel 增加原因行 |
| 同一情况下的区间计数 | 保留 | 保留：24 个区间、88.89%、均值与极值均为真实数字，未被置零 |
| 价格索引有重复时间戳 | `render()` 抛 `ValueError`，整页裸异常 | 原始价格线照常绘制（不去重、不排序），仅均线不可用并以 `st.warning` 说明 |
| 价格索引未排序或含 NaT | 同上，整页裸异常 | 不绘制价格图，`st.error` 说明原因，并清除 `report_figures["price_ts"]`，旧图不会进入 PDF |
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
- `negative_price_hours_reason` 只集中“是否可用 + 原因”。各消费端保留原有数值
  格式（页面一位小数、Excel 数字单元格、PDF 原字符串），所以本轮不引入任何
  可用数值的格式回归；Excel 仍是数字型单元格，未被改成字符串。
- 原因串在横向 A4 的 177 mm 值列内可读（151.6 mm），有实测断言，未做换行改动。

## 边界（本轮明确未做）

- 未放宽 DA/IDA 网格守卫，未改 SoC、段末中性、576 点上限、FEC、未来数据可见性，
  未触及 Project Case 公共 schema/fingerprint 与现金流合同。
- 未改整体 Avg Price 口径（Step 3C）、未做结果持久化（Step 3B）、
  未做 DST 结算披露（Step 3D）、未做 Vault housekeeping（Step 4）。
- 未泛化顺序／随机策略的 vector dt；F4 只落文档与注释。
- 未写入真实市场缓存；未删除或改写既有测试。

## 冻结与独立复现

- 基线：`95f6d09e0bfc99f1022e45c72fbdebcd3465d6a0`。分支 `step3a-display-contract-repairs`。
- [冻结代码补丁](2026-09-15-step3a-evidence/step3a-code.patch)，SHA-256：
  `88c1837b6f0adff7b3def14d73cba3e6f96b7ce49410ab9ab0b6a678a2474962`。
- hash 范围为 `src/ tests/ .github/workflows/ci.yml`；文档与日志不在代码 hash 内。
  新测试文件已加入版本控制，不会因 untracked 漏出补丁。
  Step 1、1b、1b-R2、Step 2 的原有补丁保持原字节。

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

新增 21 个用例：在基线 `95f6d09` 上 **14 个失败、7 个通过**；本轮 21 个全通过。
那 7 个即兼容对照。基线红测见
[baseline failures](2026-09-15-step3a-evidence/step3a-baseline-failures.txt)，
通过记录见 [green tests](2026-09-15-step3a-evidence/step3a-green-tests.txt)。

探针对照（同一脚本，两个版本）：
[基线输出](2026-09-15-step3a-evidence/step3a-probe-baseline.txt) 含三次
`Uncaught app execution` 与 PDF 的字面 `nan`；
[本轮输出](2026-09-15-step3a-evidence/step3a-probe-head.txt) 无异常。
探针读取真实生成的 `.xlsx` 与 `.pdf` 字节，写在临时目录，退出即删除。

兼容对照覆盖：可验证的 48 负价小时、零负价小时、干净索引的图表与均线。
另有三个用例断言坏索引的原始行仍然完整写入 Excel 的 `Hourly Prices` 页
（逐值相等），证明本轮只是不画图，没有隐藏数据。

本轮实测（Python 3.13 / Streamlit 1.55.0 / pandas 2.3.3 / SciPy 1.17.1）：

| 命令 | 结果 |
|---|---|
| `pytest tests/ -q`（含 slow） | **1865 passed / 2 skipped，294.29 秒** |
| `pytest tests/ -q --collect-only` | 1867 collected，其中 22 个 slow |
| 针对性回归（6 个相关测试文件） | 329 passed / 2 skipped |
| `ruff check src/ app.py tests/` | 通过 |
| `git diff --check 95f6d09 HEAD` | 无输出 |
| 兼容性 job 的本地等价选择 | 12 passed / 89 deselected |

基线 `95f6d09` 的最近完整验收为 1844 passed / 2 skipped；本轮 +21 个新用例，
与 1865 一致。那 2 个 skip 是既有的 opt-in 图表渲染跳过，**不代表渲染通过**。

## 未完成与限制

- 兼容性 job 的名字变化只影响该 check 的显示名；main 的分支保护只要求 `test`，
  已核对，不影响门禁。改名后的首次远端运行状态以本 PR 的 Checks 为准。
- PDF 可读性断言是按 fpdf2 字体度量计算的行宽，不是渲染截图；
  两项 opt-in 图表渲染跳过仍未证明 PDF 图表渲染成功。
- 真实浏览器验证留给 Step 3C 的布局项；本轮的页面验证是 AppTest 驱动真实面板。
