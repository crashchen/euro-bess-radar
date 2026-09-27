# Current verification snapshot

Verified **2026-09-27** against code commit `4b4df23` and final PR head
`7060149`. The remaining-page metric-layout change was independently reviewed
and [#98](https://github.com/crashchen/euro-bess-radar/pull/98) was ordinarily
merged as `3ab98c5`. Base `47a8495` is the ordinary merge of #97. Steps 1–4
and follow-ups #94–#98 are merged. The pre-merge candidate snapshot is archived
[verbatim](2026-09-27-metric-layout.md); the documentation status update does
not constitute a new code validation run.

[Browser measurements and reproducible synthetic harness](../audits/2026-09-27-metric-layout-evidence/README.md)
· [Remaining work](follow-ups.md).

## Checks and scope

| Check | Result |
|---|---|
| Browser at 1280/1440/390px | The loaded Revenue main/joint, Renewable and Forward benchmark metric labels/values fit at the reviewed head. The baseline defects and explicit limits are in the evidence README. Data Trust was checked and left unchanged. The user's external reviewer independently repeated these checks on the same head. |
| Targeted regression | 161 passed: benchmark, Step 3D display, Data Trust, Revenue decay and market-grid guards. |
| Full local suite | 2112 passed / 2 skipped, 2114 collected, including 35 slow cases; 341.01 seconds, 38 warnings. Run completed after code commit `4b4df23`. |
| Ruff and whitespace | `.venv/bin/ruff check src/ app.py tests/`, harness Ruff and `git diff --check` passed. |
| Remote CI | At exact PR head `70601497468e8c9af14892a935e3b86e968228f7`, both `test` and `Python 3.11 / Streamlit 1.55.0 compatibility check` concluded SUCCESS before merge. |

The code changes only Streamlit metric placement in Revenue Estimation,
Renewable Correlation and Forward Scenarios. In the Forward external benchmark
panel, the two average revenue values and gap use a shorter `€` display, with
their shared `EUR/MW/yr` unit stated in a visible caption. Parsing,
reconciliation values, chart/table/export data, solver cash and model contracts
are unchanged. The browser fixture uses synthetic data; no live provider,
download interaction or exhaustive numeric-width validation was run. The
Forward benchmark chart still has a separate pre-existing fractional-year
x-axis tick formatting issue, tracked in [follow-ups](follow-ups.md).

## Repeat

```sh
.venv/bin/python -m pytest tests/test_trader_benchmark.py tests/test_step3d_export_disclosure.py tests/test_data_trust.py tests/test_revenue_estimation_decay.py tests/test_market_grid_guards.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff --check
git diff --binary --full-index 47a8495 4b4df23 -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-27-metric-layout-evidence/code.patch
```

Both hash commands return
`72b6589d1eb843630f35ff7957a1637fb1d419113d50567ad53d6d5d5723cf6c`.
