# Current verification snapshot

Verified **2026-09-28** against Forward chart code commit `49a2d77`, based on
`76dc758` (the ordinary merge of documentation [#99](https://github.com/crashchen/euro-bess-radar/pull/99)).
The chart change is a **review candidate**, not merged. Steps 1–4 and
follow-ups #94–#99 are merged. The prior #99 current snapshot is archived
[verbatim](2026-09-28-pre-forward-chart.md); its tests are not evidence for
this chart change.

[Forward chart evidence and frozen patch](../audits/2026-09-28-forward-chart-evidence/README.md)
· [Remaining work](follow-ups.md).

## Checks and scope

| Check | Result |
|---|---|
| Clean baseline contrast | The new `test_trader_benchmark.py` on a clean `76dc758` archive: 2 failed / 32 passed. Both failures are the missing chart builder; the unit-caption compatibility check passes. |
| Candidate targeted tests | 34 passed in `tests/test_trader_benchmark.py`; the chart tests pin integer labels, sparse long-horizon labels and unchanged x/y traces. AppTest confirms the abbreviated euro KPIs still have their EUR/MW/yr caption. |
| Browser | The existing full Forward benchmark renderer on the base showed fractional/comma-formatted year ticks. A chart-only Streamlit harness calling the production builder showed whole-year ticks at 390/1280/1440 CSS px. At those widths the legend sits below the x-axis title and inherits readable light-theme text color. This does not certify the full Forward page at every width. |
| Full local suite | On a clean archive of code commit `49a2d77`: 2115 passed / 2 skipped, 29 warnings in 298.35 seconds; pytest and its corrected shell wrapper exited 0. The warnings are the previously tracked 23 Streamlit attrs and 6 pandas concat warnings. |
| Ruff and whitespace | Targeted Ruff and `git diff --check` passed. |
| Remote CI | Verify against the PR's exact final head after publication; local checks are not remote CI. |

The candidate changes only the Forward external benchmark chart's tick labels,
legend placement and text-color inheritance. It preserves comparison x/y
values, parsed benchmark data, reconciliation, exports and solver cash. A
separate Cockpit export-copy correction is being prepared for review; it is
not part of this code commit or these checks. No live provider or production
cache was used for browser evidence.

## Repeat

```sh
.venv/bin/python -m pytest tests/test_trader_benchmark.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff --check
git diff --binary --full-index 76dc758 49a2d77 -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-28-forward-chart-evidence/code.patch
```

Both hash commands return
`61d15bed759c368793ac24402584f02853fbcd81a9a642653b8c46c364dc449f`.
