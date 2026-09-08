# Step 2：原生 DA 时长、覆盖披露与均线

2026-09-08。本轮叠在 #87 的孤立报价修正之上，供用户将 Step 1b 与
Step 2 一并交给 CC 复审。#87 尚未合并，本轮未调用 CC 或其他外部评审。

## 结果与范围

| 触发情况 | Step 1b 基线 | Step 2 |
|---|---|---|
| FI 切换日 93 个原生 DA 产品，24 物理小时 | replay 拒绝；容量平价样本 EUR 110.4375 | DA replay 保留；容量 EUR 114 |
| BG/EE/FI/GR/LT/LV/PT/RO 的本地日内切换 | 统一 dt 不能表达真实交付时长 | 按每个原生产品时长计收益、能量、SoC、FEC、VWAP 和事件时长 |
| 96 个 forecast 点、24 个 DA 对照点 | headline 未披露 DA 子样本覆盖 | 主评分仍 96 点；DA 对照明确 24/96、25%，同时披露子样本的两项 MAE |
| 30 天低价后 10 天高价，小时价转 15 分钟价 | 720 行均线实际只覆盖 7.5 天 | 720 物理小时、按时长加权，synthetic 末值 EUR 40/MWh |
| UI 依赖下界与实际使用 API 不一致 | 安装声明允许更旧的 Streamlit | 两份 manifest 下界统一为 1.55，新增 Python 3.11 / Streamlit 1.55.0 CI 面板门禁 |

以上为 synthetic 已知答案，不是市场收益预测。完整边界见
[时长契约](../design/delivery-duration-v1.md)。改动涉及 8 个源码文件、
2 个新测试文件、2 份依赖声明和 1 份 CI 配置；没有重写或删除已有测试。

DA-only 和 joint-capacity 两个底层优化器接收 scalar/vector dt，批量 DA、
continuous DA、cycle frontier 和既有 Project Case DA 入口已接通。
Step 1 的逐日主 cadence 分段、SoC 交接、段末中性和 576 点上限保持原策略。
严格 DA/IDA 守卫未放宽；随机策略和顺序 DA/IDA 的 vector API 未泛化。
FI Project Case reserve 仍未注册支持，不能将 screening 的 EUR 114
修复解释为新增 reserve profile。

## 冻结与独立复现

- 基线：`c46aaf9054cc00723d4314d462b46d293da7f375`（#87 当前 head）。
- 分支：`codex/replay-interval-durations-step2`，PR 以 #87 分支为 base。
- [冻结代码补丁](2026-09-08-step2-evidence/step2-code.patch)，SHA-256：
  `fda57842530b8cfd0f84b9666d1fa6dd7e290e4f40e1c64538ea8002c42b7ba1`。
- hash 范围包含 `src/ tests/ requirements.txt pyproject.toml .github/workflows/ci.yml`；
  文档和日志不在代码 hash 范围。Step 1、1b、1b-R2 的原有补丁保持原字节。

在本轮分支检出执行：

```sh
git diff --binary --full-index c46aaf9054cc00723d4314d462b46d293da7f375 HEAD -- src/ tests/ requirements.txt pyproject.toml .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-08-step2-evidence/step2-code.patch
python -m pytest tests/test_delivery_durations.py tests/test_step2_views.py -q
PYTHONPATH=. python docs/audits/2026-09-08-evidence/reproduce.py
python -m pytest tests/ -q
ruff check src/ app.py tests/
git diff --check c46aaf9 HEAD
```

基线使用完整 `git archive c46aaf9` 干净解包，仅复制上述两个新测试文件，
以相同 Python 和依赖执行，得到 [82 FAIL / 8 PASS](2026-09-08-step2-evidence/baseline-red-tests.txt)。
8 个兼容控制也保留在补丁内。可按以下方式重建（从本轮仓库根目录执行）：

```sh
step2_repo=$PWD
step2_baseline=$(mktemp -d)
git archive c46aaf9054cc00723d4314d462b46d293da7f375 | tar -x -C "$step2_baseline"
cp tests/test_delivery_durations.py tests/test_step2_views.py "$step2_baseline/tests/"
(cd "$step2_baseline" && "$step2_repo/.venv/bin/python" -m pytest tests/test_delivery_durations.py tests/test_step2_views.py -q)
```

## 验证证据

- 新增 **90 例**：80 例时长／物理约束／8 区原生网格／实际 Project Case DA
  适配器与 fingerprint；10 例覆盖分母和真实 Streamlit 面板／图表。
- [针对性验证](2026-09-08-step2-evidence/step2-targeted-tests.txt)：103 PASS。
- [已有核心回归](2026-09-08-step2-evidence/core-first-regression.txt)：424 PASS。
- [完整测试](2026-09-08-step2-evidence/full-tests.txt)：
  **1844 PASS / 2 SKIP / 1846 collected，288.77 秒**，包含全部 **22 个 slow**。
  本地 Python 3.13.9、Streamlit 1.55.0。两个显式 opt-in 图表渲染跳过不代表 PDF 验证通过。
- [UI 门禁命令](2026-09-08-step2-evidence/ui-runtime-tests.txt)：本地 12 PASS / 89 deselected。
  本地不是 Python 3.11；新增远端 `minimum-ui` job 才验证这个组合，具体状态以本轮 PR Checks 为准。
- [原审计探针重跑](2026-09-08-step2-evidence/residual-probe.txt)：FI 原生 93 点未补行、
  replay 成功、容量 EUR 114、实际图表均线 EUR 40。保留 bare-mode Streamlit 警告。
- Ruff 通过；日志只规范化工作站路径、终端颜色和行尾空白，失败断言与计数保留。

目录中 `duration-*`、`frontier-red-tests`、`views-*` 是实现过程的分组记录；
它们的样本数会较少。最终新测试的完整基线证据是 `baseline-red-tests.txt`，
最终整套证据是 `full-tests.txt`，不能把过程记录误当成最终文件数量。

## 建议 CC 重点独立检查

1. 先核对补丁 hash，再在干净基线重跑 82 个红测及 8 个兼容控制。
2. FI/PT 的真实本地日边界、三日连续 replay 的逐日 dt 切片、SoC 和容量现金，
   以及 cycle frontier 的已知答案；确认没有把缺失区间拉长成产品。
3. 核对二次优化的时长权重、事件结束时间与物理能量；检查 uniform 路径回归。
4. 对照实际面板数据：DA 基准 24/96 与主评分 96；均线切换前后末值 40，
   warm-up 是 24 小时，左端截断小时和 NaN 未凭空制造覆盖。
5. #87 的孤立报价修正与这轮保持分开复核；两份冻结补丁均可从仓库取得。

## 后续

Step 3 继续处理结果持久化、金额 KPI／Project Case 分位数可读性和 DST
结算披露。原探针仍显示德国容量：Project Case EUR 456，cockpit 春季 EUR 437、
秋季 EUR 475，这是尚需明确展示的两种结算约定，本轮没有统一它们。
Step 4 再完成 README/CLAUDE 和 Vault 全量 housekeeping；本轮只同步
运行门槛、可执行测试命令、当前 replay 契约和审计证据索引。
