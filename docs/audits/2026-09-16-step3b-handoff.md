# Step 3B：批量 replay／forecast 结果持久化、内容指纹与过期拦截

2026-09-16 首轮。基线 `e60a74f3ad075c1cdb567893ef418eee7e943582`（#89 合并后的 main）。
本轮由 CC 实施，交由 Codex 独立复核；未调用其他 reviewer，未自动合并。

## 复核轮次

| 轮次 | head | 结论 |
|---|---|---|
| 首次提交 | 见 PR 的 Checks | 待复核 |

## 问题与最终行为

入口是 `src/pages/simulation_cockpit.py` 的两个按钮面板：**Multi-day replay summary**
与 **Forecast-driven IDA policy**。基线两者都是 `if not run: return`。

| 触发情况 | 基线 `e60a74f` | Step 3B |
|---|---|---|
| 点 Run | 求解并显示 | 不变：求解一次并显示；forecast 面板的 sequential、reserve co-opt、9.2b triple、stochastic 全部在这一次里算完 |
| 无关 rerun（其他控件、其他面板） | 结果消失，回到 “click Run” 提示 | 结果保留，**不重算**任何 solver |
| 点 Excel 下载（Streamlit 1.55 默认触发一次 rerun） | 同上：点下载后结果消失 | 结果保留，不重算；导出内容就是屏幕上的那次结果 |
| 图表主题变化 | 同上 | 用新主题重画，不重算，不算过期 |
| 同长度价格更正（DA、IDA 或 reserve 价格改值，行数不变） | 结果本就不保留；若恰好在同一次 rerun 里点 Run 则重算 | 显示 “Inputs changed since the last run”，隐藏图表、表格和下载，直到再次点 Run |
| 未评分日期上的训练历史更正（LOO／walk-forward 用到的日子） | 同上 | 同样过期；指纹对完整 DA／IDA 取 hash，不只 hash 目标窗口 |
| 参数切换（日期集合、资产、模式、SoC 连续、deadband、气候分桶、预测模式、reserve 产品／来源、stochastic 开关与 cap） | 同上 | 过期并拦截导出；改回原值时，已存结果原样恢复，不重算 |
| 样本选项变化但实际日期集合不变（如只载入 4 天时 7 天与 30 天） | — | 不算过期 |
| 过期后导出 | — | 下载按钮不渲染 |
| 结果之后 assumptions 表变化（仅这些面板用不到的行） | 导出总是配当次 rerun 的 assumptions | 导出配**计算那次**的 assumptions 快照，旧结果不会配新 assumptions |
| solver 全部失败／无有效日 | 点 Run 那次显示错误或警告，下一次 rerun 即消失 | 契约不变，照原样分别披露；之后的 rerun 从存储的结果重放，不重算 |
| reserve 联合 MILP 失败警告（求解中发出） | 只在点 Run 那次出现 | 求解时收集进结果包，每次渲染重放 |

## 设计决定

- **计算与渲染分离。** `_compute_multi_day_bundle` / `_compute_forecast_policy_bundle`
  调用全部 solver，返回完整结果包（含 reserve、triple、stochastic 派生结果、导出表、
  导出 assumptions）；`_render_*_bundle` 只读结果包，不调用任何 solver。forecast 计算在
  sequential 模型不可用或无有效日时就停下，与基线在这些情况下不再求解派生策略一致。
- **身份 = 求解器实际读取的内容。** `src/content_fingerprint.py` 参照 Project Case
  请求指纹的内容 hash：`pd.util.hash_pandas_object(index=True)` 的逐行 hash 再加列名、
  dtype，并**额外**纳入索引 dtype（含时区）、索引名与行数，用 PC-CBOR 规范编码后取
  SHA-256。没有复制 frontier 的首日／末日／行数指纹。
  - 传给 solver 的是完整 frame，所以 hash 完整 frame。forecast 面板必须如此（训练历史）。
    multi-day 面板因此偏保守：改动已载入但不在回放样本中的日子也会判过期，只会多报、不会漏报。
  - multi-day 仅在 DA + IDA1 模式纳入 IDA 内容：DA-only 路径从不读取 `intraday_df`
    （已核对 `simulate_replay_batch` 的三个分支）。
  - forecast 面板仅在选了 reserve 产品时纳入解析后的 capacity frame、来源标签与产品；
    没有产品就没有任何 reserve solver 运行。rebid cap 仅在勾选 stochastic 时纳入。
  - 身份还含面板标识（`cockpit-multi-day-replay/v1`、`cockpit-forecast-policy/v1`）与
    模型常量：VOM、容量可用率、连续回放点数上限、scenario 数与 seed。Streamlit 热重载会
    保留 session state，常量或面板语义变化必须让旧结果失效。
  - 主题和 assumptions 表都不在签名里（有测试检查签名）。
- **assumptions 是快照，不是身份。** 全局 assumptions 表里有这两个面板用不到的行
  （sidebar capture rate、dispatch model、Revenue 页的 rebid share、activation／imbalance
  capture share）。把它们作为身份会迫使昂贵的无谓重算；与面板相关的 deadband／模式／
  分桶本来就是身份输入。导出一律用计算时的快照。
- **过期即隐藏，不置灰。** 与 Project Case 和 frontier 的现有做法一致：过期时只显示警告，
  不显示旧数字，也不渲染下载。旧结果仍留在 session 中，输入改回即可恢复；点 Run 时先删除
  旧结果再计算，重算若抛异常，旧结果不会被当成当前结果显示。
- `_render_solver_failure_notice` 与计算阶段共用 `_batch_model_unavailable` 判定，避免
  “计算停下”与“渲染停下”两套条件分叉。`_reserve_coopt_total` 新增可选的 `warn` 回调，
  默认仍为 `st.warning`，既有调用与测试不变。
- `_render_multi_day_summary` 新增可选参数 `primary_zone`（`render()` 传入并纳入身份）；
  设为可选是为了不改动既有测试的调用。

## 边界（本轮明确未做）

- 未改任何 solver、收益、SoC、段末中性、576 点上限、FEC、未来数据可见性、DA/IDA 网格守卫；
  未触及 Project Case 公共 schema／fingerprint 与现金流合同，也未改 Project Case 代码。
- 未改 frontier 与 contracted-floor 面板的既有指纹。frontier 的首日／末日／行数键仍漏检
  同长度数据更正，已在 CLAUDE.md 该条目注明，留作后续。
- 页面顶部的单日 replay 仍在每次 rerun 时求解（单个 MILP，不在本项范围）。
- 未做 Step 3C 的均价口径与布局、Step 3D 的 DST 结算披露、Step 4 的 Vault 整理。
- 未写入真实市场缓存；未删除或改写既有测试。

## 冻结与独立复现

- 基线：`e60a74f3ad075c1cdb567893ef418eee7e943582`。分支 `step3b-batch-result-persistence`。
- [冻结代码补丁](2026-09-16-step3b-evidence/step3b-code.patch)，SHA-256：
  `034b0b88c3011715d4bc650fbcbfe35555c41dd648461cb2da4065af29b3ef53`
- 代码提交 `6b6d436`；其后提交只改文档与证据，不改变上述 hash。
- hash 范围为 `src/ tests/`；文档、日志与探针不在代码 hash 内。新测试文件已加入版本控制。
  Step 1、1b、1b-R2、Step 2、Step 3A 的原有补丁保持原字节。
- 证据目录沿用 `.gitattributes`（`*.patch -text whitespace=-blank-at-eol`）。

```sh
git diff --binary --full-index e60a74f3ad075c1cdb567893ef418eee7e943582 HEAD -- src/ tests/ | shasum -a 256
shasum -a 256 docs/audits/2026-09-16-step3b-evidence/step3b-code.patch
```

## 验证

```sh
.venv/bin/python -m pytest tests/test_step3b_batch_result_persistence.py -q
.venv/bin/python -m pytest tests/test_strategy_compare.py tests/test_ui_theme.py \
    tests/test_simulation_cockpit_frontier.py \
    tests/test_simulation_cockpit_contracted_floor.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff --check e60a74f HEAD
PYTHONPATH=. .venv/bin/python docs/audits/2026-09-16-step3b-evidence/step3b-probe.py
```

新增 67 个用例：在基线 `e60a74f` 上 **58 个失败、9 个通过**；本轮 67 个全通过。
那 9 个是兼容对照，基线与本轮都通过：

- multi-day：点 Run 前只提示、不求解；点 Run 后渲染结果，`simulate_replay_batch` 调用一次。
- multi-day：点 Run 当次，solver 全部失败显示 unavailable、无有效日显示 warning（2 个参数）。
- forecast：点 Run 当次，渲染 7 行策略表和 4 个 sheet 的导出；sequential、joint capacity、
  9.2b reserve、stochastic triple 各调用一次。
- forecast：点 Run 当次的失败／无有效日披露（2 个参数），且不求解任何派生策略。
- forecast：点 Run 当次重放 reserve 联合 MILP 失败警告。
- 图表 spec 是真实 Plotly JSON（主题断言的前提）。

基线的 58 个失败都落在该失败的断言上：结果在 rerun 后消失、没有过期警告、披露在 rerun
后消失；纯身份函数的用例在基线上因为新函数不存在而失败。见
[baseline failures](2026-09-16-step3b-evidence/step3b-baseline-failures.txt) 与
[green tests](2026-09-16-step3b-evidence/step3b-green-tests.txt)。

solver 计数包住 cockpit 模块导入的**全部** `simulate_*`／`solve_*`／`compute_*` 入口
（夹具断言至少包住 8 个）并转发到真实 solver，所以派生策略若在 rerun 时被重算也会被计到。
forecast 面板用真实 solver，scenario 数在测试里降为 2（它本身也在指纹里）。

探针对照（同一脚本，两个版本，真实 solver，合成数据，缓存指向临时目录）：
[基线输出](2026-09-16-step3b-evidence/step3b-probe-baseline.txt) 中，点 Run 之后的
每一次 rerun 都是 `tables=0 downloads=0 stale=False`，更正后也没有任何提示；
[本轮输出](2026-09-16-step3b-evidence/step3b-probe-head.txt) 中，无关 rerun 与主题变化
保持结果、调用计数不变；deadband 改动、同长度 reserve 价格更正、未评分日的 IDA 历史更正
均为 `stale=True` 且无表格、图表、下载；deadband 改回即恢复；再次 Run 后四个 solver
各变为 2 次。

本轮实测（Python 3.13.9 / Streamlit 1.55.0 / pytest 8.4.2）：

| 命令 | 结果 |
|---|---|
| `pytest tests/ -q`（含 slow） | **1938 passed / 2 skipped，300.84 秒** |
| `pytest tests/ -q --collect-only` | 1940 collected，其中 28 个 slow（原 22 个 + 本轮 6 个 forecast 真实 solver 用例） |
| `pytest tests/ -q -m "not slow"` | 1910 passed / 2 skipped / 28 deselected，44.02 秒 |
| `tests/test_step3b_batch_result_persistence.py` | 67 passed，20.77 秒 |
| 针对性回归（strategy compare、ui theme、frontier、contracted floor） | 135 passed |
| `ruff check src/ app.py tests/` | 通过 |
| `git diff --check e60a74f HEAD` | 无输出（依赖证据目录的 `.gitattributes`） |
| 兼容性 job 的本地等价选择 | 12 passed / 89 deselected |

基线 `e60a74f` 的最近完整验收为 1871 passed / 2 skipped（1873 collected，22 slow）；
本轮 +67 个新用例，与 1938 passed / 1940 collected 一致。
那 2 个 skip 是既有的 opt-in 图表渲染跳过，**不代表渲染通过**。

指纹开销：一整年 15 分钟 DA（35 136 行）取一次内容 hash 约 0.54 ms，存在结果时
每次 rerun 重算指纹的成本可以忽略。

## 未完成与限制

- **AppTest 不能点击下载按钮。** 测试与探针用一次无输入变化的 rerun 代替，依据是
  Streamlit 1.55 `download_button` 的默认 `on_click="rerun"`。真实浏览器中的下载点击
  **尚未验证**（需要实际下载文件，未经用户许可未执行）；已在
  `docs/runbooks/manual-ui-smoke.md` 增加第 28–31 项供人工验收。
- 结果会一直显示到指纹变化为止。热重载时若只改了 solver 内部实现，而没有改常量或面板
  标识，同一 session 仍会显示旧结果；新 session 不受影响。改变面板语义时应同步提升面板标识。
- 输入 frame 的 `.attrs` 不在 hash 内：已核对各 solver 只写输出 frame 的 attrs，不读输入的。
- 逐行 hash 为 64 位；同长度更正恰好碰撞的概率可以忽略，但不是零。
- 过期结果留在 session 内存中，直到下次 Run 或会话结束。
- 原先 forecast 面板会先画出 sequential 结果，再去算 reserve／stochastic；现在全部算完才
  渲染（期间显示 spinner）。数值与顺序不变，只是不再渐进显示。
- 真实浏览器布局验证留给 Step 3C。
