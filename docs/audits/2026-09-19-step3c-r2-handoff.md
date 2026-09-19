# Step 3C review follow-up — remaining Cockpit cards and statistic labels

The user's independent reviewer accepted `242843e` and suggested two
non-blocking presentation fixes. Both are implemented in
`559f33a3938f0868960da0a4c6f9227cf990b1a6`. Draft PR #91 remains open for review;
this record does not authorize merging or starting Step 3D.

Review baseline: `242843e7ab0b5991e9b750ab2de6b429a21373c7`.
PR base: `cf913747d4f27e490545f96b35f1f23e7b956cf4`.
Later commits contain evidence and documentation only. Read remote CI against
the latest PR head, separately from the local results below.

## Changes

- The three remaining Cockpit metric rows (reserve forecast skill, activation
  overlay and imbalance overlay) use the existing wrapping container helper.
  Only the three layout calls changed. Calculations, labels, values and controls
  are unchanged. The same production panels return the same eight metric
  label/value pairs on both revisions, with no AppTest exceptions.
- Summary XLSX/PDF now say **Median Price (row-based, EUR/MWh)**. The Zone
  Comparison page and workbook say **Std Dev (row-based)**, with page help
  explaining equal weight per price row. Median and sample standard deviation
  retain their existing calculations; neither is duration-weighted. Numeric
  Excel cell types and two-decimal formats are preserved. The longer summary
  label uses the existing wrap/row-height support.
- The successful Avg Price caption uses an em dash before its basis statement,
  removing the doubled colon. CLAUDE.md, the duration contract and smoke guide
  distinguish statistic bases and explicitly scope layout acceptance to Market
  Overview, Project Case (including its cockpit mirror) and Simulation Cockpit.
  The four other metric-bearing pages are not included in this layout claim.

No analytics calculation, solver, SoC rule, DA/IDA guard, segment-end condition,
576-interval limit, FEC, future-data visibility, Project Case model/schema/
fingerprint or cash-flow convention changed in this follow-up. The core files
listed by the previous review remain byte-identical to the PR base.

## Frozen increment and checks

The original seven frozen patches are unchanged, including Step 3C's
`82f63e01…9bfa`. [Their verified hashes](2026-09-19-step3c-r2-evidence/historical-patch-hashes.json)
and the two new patch references are recorded separately in the
[manifest](2026-09-19-step3c-r2-evidence/manifest.json).

- [Increment from reviewed head](2026-09-19-step3c-r2-evidence/step3c-r2-code.patch):
  `e0dcce755ec46f1eac3b21e9495326316bc55ce4b6c5aaf731ada3a1e6804dce`.
- [Aggregate from PR base](2026-09-19-step3c-r2-evidence/step3c-r2-aggregate-code.patch):
  `984818209aacbcacc47d7fb0b5c6acfd945108101aeaf0e6714dca4df1534d23`.

Both use the existing `src/ tests/ .github/workflows/ci.yml` scope. CI workflow
and check names are unchanged.

```sh
git diff --binary --full-index 242843e7ab0b5991e9b750ab2de6b429a21373c7 HEAD -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
git diff --binary --full-index cf913747d4f27e490545f96b35f1f23e7b956cf4 HEAD -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-19-step3c-r2-evidence/*code.patch
.venv/bin/python -m pytest tests/test_step3c_average_price.py tests/test_export.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff --check cf913747d4f27e490545f96b35f1f23e7b956cf4 HEAD
```

The 37-case average-price file retains all previous assertions and strengthens
four existing cases for the new labels, numeric values/types/formats and page
help. Copying this final file into a clean archive of `242843e` gives
**4 failed / 33 passed**; all four failures are the missing median/std
disclosures. [Baseline log](2026-09-19-step3c-r2-evidence/base-r2-tests.txt).
The current file gives **37 passed**; including the export suite gives
**59 passed / 2 skipped** on the final code commit.
[Final targeted log](2026-09-19-step3c-r2-evidence/final-targeted-tests.txt).

The local full suite gives **1975 passed / 2 skipped**, including 28 slow tests,
in 310.68 seconds. The final assertion additions were completed during that
run; the final code commit was then checked by the targeted run above.
[Full-suite log](2026-09-19-step3c-r2-evidence/full-suite.txt). Ruff and diff check
pass. The two skipped opt-in chart-render tests remain skips; the actual
summary PDF rendering below is separate evidence.

## Browser and saved-file verification

The [synthetic harness](2026-09-19-step3c-r2-evidence/panel_harness.py) invokes
the three production renderers with in-memory synthetic imports. Only the two
cache readers are replaced; the skill and overlay calculations are real. It
does not fetch or write production market data. The
[AppTest probe](2026-09-19-step3c-r2-evidence/panel_probe.py) records
[baseline](2026-09-19-step3c-r2-evidence/panel-base.json) and
[current](2026-09-19-step3c-r2-evidence/panel-working.json) outputs: eight
identical labels/values, including `EUR 123.45/MW/h`, `EUR 123,456,789` and
`EUR -98,765,432`. This establishes value compatibility, not visual readability.

```sh
PYTHONPATH=. .venv/bin/streamlit run docs/audits/2026-09-19-step3c-r2-evidence/panel_harness.py --server.address 127.0.0.1 --server.port 8621 --browser.gatherUsageStats false
```

Real-browser acceptance uses Streamlit 1.55.0 in the Codex in-app browser at
1280×900, 1440×900 and 390×844 CSS pixels. All three expanders are open.
Desktop sidebar is expanded. On mobile it is recorded as an overlay, then
closed before checking main content. The
[baseline screenshot](2026-09-19-step3c-r2-evidence/browser/baseline-reserve-1280.png)
shows three clipped reserve values: their text needs 264/246/264 px inside
150 px text boxes. The current boxes are 329 px on desktop and 280 px on mobile;
all eight metric values and labels fit at all three widths. Measurements use
the actual value paragraph's `scrollWidth`/`clientWidth`, not only its outer
container, and record expander/sidebar state. Closed baseline overlays in the
JSON are not presented as visual acceptance evidence.

- [1280 px](2026-09-19-step3c-r2-evidence/browser/working-1280.json),
  [1440 px](2026-09-19-step3c-r2-evidence/browser/working-1440.json),
  [390 px](2026-09-19-step3c-r2-evidence/browser/working-390.json).
- [Reserve desktop](2026-09-19-step3c-r2-evidence/browser/working-reserve-1280.png),
  [reserve mobile](2026-09-19-step3c-r2-evidence/browser/working-reserve-390.png),
  [activation mobile](2026-09-19-step3c-r2-evidence/browser/working-activation-390.png),
  [imbalance mobile](2026-09-19-step3c-r2-evidence/browser/working-imbalance-390.png).

Actual production [summary XLSX](2026-09-19-step3c-r2-evidence/exports/mixed.xlsx),
[comparison XLSX](2026-09-19-step3c-r2-evidence/exports/comparison.xlsx) and
[PDF](2026-09-19-step3c-r2-evidence/exports/mixed.pdf) are included. Poppler
rendered the PDF; Artifact Tool imported and rendered the saved workbooks
read-only. Labels and values are readable in all three renders. The synthetic
mixed window still has Avg Price **32.50**, row median **100.00**, and row
sample standard deviation **44.55**. Workbook cells remain numeric with
`#,##0.00`; the median label wraps and has a 30-point row. See the
[inspection JSON](2026-09-19-step3c-r2-evidence/exports/disclosure-inspection.json)
and adjacent scripts/images. This is not a native Excel application check.

## Remaining scope and review focus

1. Review this small increment against `242843e`; original Step 3C evidence and
   its 31-red/6-green baseline remain frozen in the earlier handoff.
2. Recheck the actual export labels, preserved numbers, and three added metric
   rows at the listed widths. This does not certify every amount, browser,
   zoom, table or Plotly chart.
3. Confirm both CI jobs on the latest PR head before considering merge.

Step 3D remains the separate DST capacity-settlement disclosure task; it must
retain both existing economic conventions. Step 4 owns README/Vault freshness
reconciliation and prioritization of metric layouts on Revenue Estimation,
Forward Scenarios, Renewable Correlation and Data Trust. No broad housekeeping
completion is claimed here.

Two reviewer observations remain recorded, without widening this patch:
`calculate_average_price` does not normalize arbitrary object-typed price
columns, and its reason for a naive mixed-cadence index can obscure the missing
timezone. Normal ingestion supplies numeric prices and aware timestamps; a
future input-contract hardening task should assess those alternate inputs.
CC was not invoked automatically. The review verdict came from the user.
