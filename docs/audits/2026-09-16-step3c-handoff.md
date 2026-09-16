# Step 3C — duration-weighted Avg Price and readable KPI cards

Status: implementation complete, prepared for independent review; do not merge
automatically. CC implemented the average-price checkpoint `c703013`; Codex
reviewed that checkpoint and completed the cadence guards, export disclosure
and layout in `97520f751bb0fbb8515dab1da83437165a80247a`. This handoff is an
implementation record, not a claim that the final increment has passed CC review.

Base: `cf913747d4f27e490545f96b35f1f23e7b956cf4` (normal merge of #90).
Branch: `step3c-duration-weighted-avg-price`. Later commits contain documentation
and evidence only. GitHub Checks must be read against the PR's latest head.

## Result and scope

- Market Overview, Zone Comparison, Excel and PDF share
  `calculate_average_price`: sum(finite price × verified delivery hours) divided
  by finite covered hours. The 30 hourly days at 10 followed by 10 quarter-hour
  days at 100 example now yields **32.50**, rather than the row mean **61.43**.
  The trailing 720-hour chart mean remains **40**, a different window.
- The whole input grid is verified before excluding non-finite prices.
  A real DatetimeIndex and supported native 1h/30min/15min products are required;
  mixed products still need the registered SDAC cutover. Gaps, duplicates,
  NaT, unsorted timestamps, singleton and unknown regular 2h/daily samples do
  not acquire invented durations. Finite coverage is disclosed.
- Unavailable averages have a reason. Summary exports use `n/a`, while valid
  workbook averages remain numeric. Zone Comparison keeps its numeric table
  and sorting: a missing numeric cell is explained by the named zone's `n/a`
  caption below; its Excel file uses literal `n/a`. The new Excel reason column
  wraps and gets enough row height to remain readable beside numeric columns.
- Market and Project Case metrics, plus the Cockpit batch/forecast/reserve/
  stochastic/frontier/floor metrics, wrap in native horizontal containers.
  Cards request 360 px, capped by the actual parent width. Project Case keeps
  full economic section titles and short P10/P50/P90/probability card labels;
  help retains the full context and all amounts retain their original format.
- Cockpit's custom cards use a container-driven CSS grid with a 240 px minimum;
  its health entries use 150 px. They no longer depend on a viewport breakpoint
  that ignores the open sidebar. Metric inputs and control rows are unchanged.

The shared duration inference, dispatch, simulation and Project Case core
modules are byte-identical to base. No DA/IDA guard, SoC, segment-end equality,
576-interval limit, FEC, future-data visibility, Project Case schema/fingerprint
or cash-flow model changed. Renewable-conditioned and forward averages are
outside this increment. Step 3D DST settlement disclosure and Step 4 Vault
housekeeping remain pending; this is not a full Vault freshness claim.

## Frozen code and reproducible checks

[Frozen patch](2026-09-16-step3c-evidence/step3c-code.patch), SHA-256:

`82f63e0113352765fd3ad031c2f2e1cffe8fe0bc820252e583b139849fdd9bfa`

Scope: `src/ tests/ .github/workflows/ci.yml` (workflow itself is unchanged).
All six older patches remain byte-identical; their hashes are recorded in
[historical-patch-hashes.json](2026-09-16-step3c-evidence/historical-patch-hashes.json).

```sh
git diff --binary --full-index cf913747d4f27e490545f96b35f1f23e7b956cf4 HEAD -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-16-step3c-evidence/step3c-code.patch
.venv/bin/python -m pytest tests/test_step3c_average_price.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff --check cf913747d4f27e490545f96b35f1f23e7b956cf4 HEAD
PYTHONPATH=. .venv/bin/python docs/audits/2026-09-16-step3c-evidence/independent_probe.py
```

The final 37-case file collects cleanly on a `git archive` of base:
**31 failed / 6 passed**. Copy only that test file into the archive. The six
compatibility controls pass on both revisions, and are named in the
[baseline summary](2026-09-16-step3c-evidence/final-baseline-summary.md).
The same final file on CC's `c703013` checkpoint is **5 failed / 32 passed**,
isolating unsupported 2h and 24h cadence, numeric indexes, clipped workbook
reasons and the missing explicit zone `n/a` marker.

Local full suite: **1975 passed / 2 skipped**, 1977 collected, including all
28 slow tests, 332.76 seconds. The last zone-caption wording and its assertion
were completed during that run; afterwards the final code commit passed all
**80** average-price and contracted-floor cases. The final 37-case file was also
rerun on base. See [full suite](2026-09-16-step3c-evidence/full-suite.txt),
[final targeted run](2026-09-16-step3c-evidence/final-targeted-tests.txt),
[baseline](2026-09-16-step3c-evidence/final-base-tests.txt) and
[checkpoint failures](2026-09-16-step3c-evidence/final-c703013-tests.txt).
Ruff passed. Local runtime: Python 3.13.9, Streamlit 1.55.0, pandas 2.3.3,
NumPy 2.4.3 and SciPy 1.17.1. Remote CI is a separate exact-head check.

The independent randomized probe compares 100 mixed windows, with NaN and both
infinities, to the same physical prices expanded onto 15-minute products.
Maximum mean difference is `1.0658141036401503e-14`; finite and total hours
match exactly. Removing products before, at or after the seam is rejected.
[Probe output](2026-09-16-step3c-evidence/updated-independent-results.json).

Existing Project Case tests now check both full section titles, each group's
short labels, original values and full-context help without collapsing duplicate
short labels into a dictionary. The pre-existing floor layout assertion retains
its 2/2/3 semantic groups using wrapping containers; it is not visual proof.

## Actual browser and export evidence

The [synthetic harness](2026-09-16-step3c-evidence/layout_harness.py) invokes real
production renderers, redirects caches into a temporary directory and does not
fetch market data. It includes a real Project Case fixture and a 250 MW Cockpit
replay. Its separate KPI gallery supplies synthetic presentation fixtures to
seven actual Cockpit renderers: all 37 related metric cards, without solvers.

```sh
PYTHONPATH=. .venv/bin/streamlit run docs/audits/2026-09-16-step3c-evidence/layout_harness.py --server.address 127.0.0.1 --server.port 8617 --browser.gatherUsageStats false
```

In a real browser, open the four sidebar sections at 1280×900, 1440×900 and
390×844 CSS pixels. Desktop sidebar is expanded. At 390 px Streamlit uses an
overlay: its expanded state is recorded, then it is closed to inspect the main
content. We do not claim that content underneath an open mobile overlay is
visible. This is browser rendering, not AppTest string inspection.

All 12 section/width combinations had no overflowing metric text or custom
card contents in the measured fixtures. Measurements include actual
`innerWidth`, sidebar state, widths, full strings and overflow results:
[JSON](2026-09-16-step3c-evidence/browser/measurements.json).
Screenshots are in the same directory, including:

- [Old market, 1280](2026-09-16-step3c-evidence/browser/baseline-market-1280.png): three clipped value/unit strings.
- [Market, 1280](2026-09-16-step3c-evidence/browser/market-1280.png) and [390](2026-09-16-step3c-evidence/browser/market-390.png).
- [Project Case, 1440](2026-09-16-step3c-evidence/browser/project-case-1440.png) and [390](2026-09-16-step3c-evidence/browser/project-case-390.png): negative eight-digit and positive nine-digit euro amounts.
- [Cockpit, 1440](2026-09-16-step3c-evidence/browser/cockpit-1440.png) and [390](2026-09-16-step3c-evidence/browser/cockpit-390.png).
- [KPI gallery, 1280](2026-09-16-step3c-evidence/browser/gallery-1280.png) and [390](2026-09-16-step3c-evidence/browser/gallery-390.png): `EUR 286,341/MW/yr` is complete.
- [Mobile sidebar overlay](2026-09-16-step3c-evidence/browser/mobile-sidebar-open.png).

Browser: Codex in-app browser, Streamlit 1.55.0, screenshots at device scale 1.
This validates the named fixtures and widths, not every amount, zoom, browser,
Plotly chart or table. The baseline layout is the unchanged layout at `c703013`.

Four actual summary XLSX/PDF pairs cover the 32.50 mixed window, partial finite
coverage (958.50/960 hours, 99.8%), missing products and unknown cadence.
Saved Excel cell types were checked with openpyxl; workbooks were imported and
rendered read-only with Artifact Tool. Every PDF page was rendered with Poppler.
The comparison workbook was re-rendered after its clipped reason was fixed.
[Export report, scripts, actual files and images](2026-09-16-step3c-evidence/exports/review.md).
This is not a native Excel application check. The partial-finite PDF continues
two existing summary rows onto page 2; both are readable. These are actual
summary renders, not a claim that the two opt-in chart-render tests passed.

## Review focus

1. Recompute the frozen hash and repeat the 31/6 baseline versus 37/0 head test.
2. Verify the native-duration mean, unsupported cadence refusal and finite
   coverage, especially the distinction from the trailing 720-hour mean.
3. Open actual export files and inspect the PDF and workbook reason rows.
4. Repeat the three viewport checks with the synthetic harness; keep the mobile
   overlay qualification and verify full PC economic context and Cockpit units.
5. Confirm exact-head CI and unchanged model/core files before considering merge.
   Do not start Step 3D or merge this PR as part of this review.
