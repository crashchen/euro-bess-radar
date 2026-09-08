# Step 1b 第二轮：孤立报价的按日排除

2026-09-08。用户提供的 CC 复审接受 #86，并复现 Step 1b 首轮的
`a5a8f6ad…` hash、37 FAIL / 34 PASS（新文件 71 例）、1747 PASS / 2 SKIP
及五条路径的 solver 前拒绝；同时指出增加一个 IDA 报价会使整窗不可用。
本轮直接修复该反转，用户提供的 CC 验证不延伸为对本轮新代码的验收。

## 行为

- 某本地日任一来源少于两个时间戳，无法确认 cadence：排除当天的比较点，继续核对其他日期。
- 原始 DA 和 IDA 行数继续作为各自覆盖率分母；排除日不贡献收益或样本数。没有一个可确认日时仍不可用。
- 实际 cadence 不同、相位偏移、重复时间戳仍拒绝整窗。严格 dispatch 守卫不变。
- 收益统计和实际 Revenue 直方图共用 `_intraday_uplift_price_pairs`，防止只改判断后把孤立高价混进图中。页面说明按日排除规则及原始分母。

五个 Berlin 小时日的 synthetic 样本，第三天缺失或只保留一个极端报价：

| 第三天 IDA | `0b5d718` 首轮 | 本轮 |
|---|---|---|
| 0 个报价 | 可用，96 点，€511.35 | 可用，96 点，€511.35 |
| 1 个报价 | 不可用，0 点 | 可用，96 点，€511.35 |

本轮两者总体覆盖均为 80%。单点情形 IDA 分母仍是 97，IDA 覆盖为
96/97 = 99.0%（显示精度），而非删掉单点后报告 100%。DA 侧的同类稀疏
情况保持对称。数值不是实际市场收益。

## 冻结范围与复现

- 基线：`0b5d7181c4d2ec7a6e8e85f7fdd36f8cedf2cd97`。
- [本轮增量补丁](2026-09-08-evidence/step1b-r2-source.patch) 覆盖 2 个源码文件和 1 个测试文件。
- SHA-256：`022f22efb435078e1e37f7cabbababf13030c25f588409f5b94b1c2e8e47f788`。
- CC 已核对的 [首轮补丁](2026-09-08-evidence/step1b-source.patch) 未覆盖、未改名。

在本轮源码版本检出中核对（文档不计入源码 hash）：

```sh
git diff --binary --full-index 0b5d7181c4d2ec7a6e8e85f7fdd36f8cedf2cd97 c46aaf9054cc00723d4314d462b46d293da7f375 -- src/ tests/ | shasum -a 256
shasum -a 256 docs/audits/2026-09-08-evidence/step1b-r2-source.patch
PYTHONPATH=. .venv/bin/python docs/audits/2026-09-08-evidence/step1b-r2-probe.py
```

基线红测独立使用完整 `git archive 0b5d718` 检出，仅复制当前
`tests/test_market_grid_guards.py`，执行：

```sh
.venv/bin/python -m pytest tests/test_market_grid_guards.py -k 'singleton_day_matches or excludes_singleton' -q
```

结果：[7 FAIL / 71 deselected](2026-09-08-evidence/step1b-r2-red-tests.txt)。
同一探针在 [旧源码](2026-09-08-evidence/step1b-r2-baseline-probe.json)
和 [本轮源码](2026-09-08-evidence/step1b-r2-probe.json) 的 JSON 均保留。
其余首轮历史基线失败与兼容控制见 [首轮记录](2026-09-08-step1b-handoff.md)。

## 验证与发布

- 新增 7 例：DA/IDA × 本地日首／中／末位置共 6 例，以及真实 Revenue 面板的 1 例。面板测试记录实际传给 Plotly 的数据，同时执行真实图表构建；断言 24 个价差均为 €10、覆盖率 50.0%/96.0%、headline €320。
- [针对性回归](2026-09-08-evidence/step1b-r2-targeted-tests.txt)：177 PASS。
- 完整门禁 **1754 PASS / 2 SKIP / 1756 collected，288.60 秒**，包含全部 **22 个 slow**；见 [全套输出](2026-09-08-evidence/step1b-r2-full-tests.txt)。Ruff、语法编译和 diff check 通过。
- #86 已合并：`850343def9d69a0774570a4112db6d6498002dce`。#87 已转向 main；远端结果以 [PR Checks](https://github.com/crashchen/euro-bess-radar/pull/87/checks) 的具体 head 为准。
- 审计证据、冻结补丁、复现脚本和交接记录随本次修正提交进仓库。原始工作站项目笔记不作为 PR 附件发布。

覆盖率仍是区间计数，不是时长或能量覆盖。两个以上但仍不足以推断真实
delivery duration 的稀疏记录，需要后续元数据契约；本轮没有猜测或重采样。
下一步顺序见 [证据索引](README.md)，日内 dt、视觉和完整 housekeeping 尚未完成。
