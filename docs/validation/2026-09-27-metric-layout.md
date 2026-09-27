# Current verification snapshot

Verified **2026-09-27** against code commit `4b4df23`, the remaining-page
metric-layout candidate. Base `47a8495` is the ordinary merge of independently
reviewed [#97](https://github.com/crashchen/euro-bess-radar/pull/97). This
candidate is **awaiting independent review**, not merged. Steps 1–4 and
follow-ups #94–#97 are merged. The #97 verification snapshot is archived
[verbatim](2026-09-23-strategy-export.md); later documentation does not make
its tests evidence for this new code.

[Browser measurements and reproducible synthetic harness](../audits/2026-09-27-metric-layout-evidence/README.md)
· [Remaining work](follow-ups.md).

## Checks and scope

| Check | Result |
|---|---|
| Browser at 1280/1440/390px | The loaded Revenue main/joint, Renewable and Forward benchmark metric labels/values fit in the candidate. The baseline defects and explicit limits are in the evidence README. Data Trust was checked and left unchanged. |
| Targeted regression | 161 passed: benchmark, Step 3D display, Data Trust, Revenue decay and market-grid guards. |
| Full local suite | 2112 passed / 2 skipped, 2114 collected, including 35 slow cases; 341.01 seconds, 38 warnings. Run completed after code commit `4b4df23`. |
| Ruff and whitespace | `.venv/bin/ruff check src/ app.py tests/`, harness Ruff and `git diff --check` passed. |
| Remote CI | Check the draft PR at its exact final head; not inferred from local checks. |

The code changes only Streamlit metric placement in Revenue Estimation,
Renewable Correlation and Forward Scenarios. In the Forward external benchmark
panel, the two average revenue values and gap use a shorter `€` display, with
their shared `EUR/MW/yr` unit stated in a visible caption. Parsing,
reconciliation values, chart/table/export data, solver cash and model contracts
are unchanged. The browser fixture uses synthetic data; no live provider,
download interaction or exhaustive numeric-width validation was run.

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
