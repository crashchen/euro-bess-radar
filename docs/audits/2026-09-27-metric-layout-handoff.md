# Remaining metric pages — review handoff (2026-09-27)

**Base:** `47a8495` (#97 ordinary merge). **Code:** `4b4df23`. Draft PR and
final documentation head are recorded in the PR. Do not merge automatically;
the user owns CC invocation and any merge instruction.

Read [current verification](../validation/current.md), the
[browser measurements and fixture](2026-09-27-metric-layout-evidence/README.md),
then inspect the exact production diff. The frozen `src/ tests/ CI` patch is
[code.patch](2026-09-27-metric-layout-evidence/code.patch), SHA-256
`72b6589d1eb843630f35ff7957a1637fb1d419113d50567ad53d6d5d5723cf6c`.
The #97 `current.md` was copied byte-for-byte to
[its dated archive](../validation/2026-09-23-strategy-export.md)
(SHA-256 `bd26a372b571742d5780bdb748125b097b924c0e7c454de1bf669f7cd772a253`).

## What changed

Three production pages use the already-merged `metric_columns()` layout helper
for pure KPI rows. Revenue's fixed metric rows now wrap inside their actual
container, including the joint MILP expander. Renewable's four labels and
Spread Uplift value no longer clip at 1280px. Forward's external-benchmark
metrics wrap; two annual average values and the model-minus-benchmark delta
show shorter euro amounts, and a visible caption states that those amounts
are EUR/MW/yr. The ratio and CAGR keep their separate units.

Data Trust was audited on the same clean baseline at 1280/1440/390px. Its
loaded four-card row already fitted, so its production code was not changed.
Revenue's non-metric input and chart columns were also left alone. No solver,
source, data transformation, cash, session cache, Excel/PDF or Project Case
code changed.

## Independent review requests

1. Recompute both patch hashes and verify the source change is limited to the
   three pages. The code commit is `4b4df23`; later commits should be docs-only.
2. On a clean `47a8495` archive, run the synthetic browser harness and inspect
   the actual label/value `<p>` nodes at 1280/1440/390px with sidebar expanded
   on desktop. Re-run on the candidate. For Forward, verify the mobile amount
   after merely widening cards was still 294px of text in a 280px value area;
   the shorter display plus explicit group unit is needed.
3. Verify `€134,567`, `€191,756`, `+18.0%/yr`, `70.2%` and the signed gap
   remain derived from the same benchmark summary fields, and the ratio cannot
   be read as a capture rate. Check table and export still use the original
   numeric reconciliation, not display strings.
4. Check nested expander behavior, including Revenue joint cards and Forward
   benchmark. Existing AppTest/solver tests are useful behavior controls;
   browser measurements are the layout evidence. Challenge any claim beyond
   the loaded synthetic cases. Conditional Revenue branches, extreme values,
   actual provider imports and download interactions are still operator checks.
5. Check README, CLAUDE, audit index, validation archive and
   [manual items 48–50](../runbooks/manual-ui-smoke.md) for correct merged vs
   candidate status. Do not edit historical handoffs or frozen patches.

## Local commands

```sh
git diff --binary --full-index 47a8495 4b4df23 -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-27-metric-layout-evidence/code.patch
.venv/bin/python -m pytest tests/test_trader_benchmark.py tests/test_step3d_export_disclosure.py tests/test_data_trust.py tests/test_revenue_estimation_decay.py tests/test_market_grid_guards.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff --check
```

The targeted set passed 161 tests. The full suite completed after code commit
`4b4df23`: 2112 passed / 2 skipped, including 35 slow cases, in 341.01 seconds.
The two opt-in chart-render skips do not certify PDF rendering. Remote CI must
be checked against the final PR head, not inferred from this local result.
