# Remaining-page metric layout evidence — 2026-09-27

Code base `47a8495` (#97 ordinary merge); candidate code `4b4df23`.
The [frozen code patch](code.patch) is scoped to `src/`, `tests/` and CI; its
SHA-256 is `72b6589d1eb843630f35ff7957a1637fb1d419113d50567ad53d6d5d5723cf6c`.
No solver, source data, export workbook, model input or cash calculation changed.

## Browser check

Chrome/Streamlit 1.55 was tested at 1280, 1440 and 390 CSS-pixel viewport
widths. Desktop checks kept the 300px sidebar expanded. The same synthetic
inputs and production renderers were used on a `git archive 47a8495` baseline
and the candidate. In browser DOM, a label or value counts as clipped when its
rendered `<p>` has `scrollWidth > clientWidth + 1`; Streamlit gives these nodes
`overflow: hidden` and `text-overflow: ellipsis`. The displayed text and visual
screens were also inspected. This is a fixture-bound measurement, not a claim
that every possible currency magnitude or conditional Revenue branch fits.

| Loaded panel | Base at 1280px | Candidate at 1280px | 1440/390px boundary |
|---|---|---|---|
| Revenue Estimation, joint MILP | `Avg Reserve Commitment` displays `100% of po...` in the [earlier browser capture](../2026-09-20-step3d-evidence/browser/revenue-1280.png) | Three joint cards wrap inside their expander; `100% of power` and `€239,421/yr` are complete | The eight visible main/joint cards have no label or value overflow at 1440 or 390 |
| Renewable Correlation | Two Avg Price labels and the Spread Uplift label clip; `€-22.0/MWh` clips | Four cards wrap 2+2; all four labels and values fit | At 1440 the baseline still clips the Spread Uplift label; candidate fits. At 390 both versions stack, candidate fits |
| Forward external benchmark | The `EUR 134,567/MW/yr` and `EUR 191,756/MW/yr` values clip, along with several labels/CAGR in four narrow cards | Cards wrap 2+2; labels and values fit | Baseline still clips both values at 1440 and 390; candidate uses `€134,567`/`€191,756` plus a visible `EUR/MW/yr` group caption and fits at both sizes |
| Data Trust | All four labels and values fit at 1280, 1440 and 390 | No production-code change | Four fixed columns remain |

The Revenue browser check used the existing [synthetic real-solver page](../2026-09-20-step3d-evidence/browser/revenue_harness.py).
The other three pages use this directory's [harness](harness.py). It calls the
production renderers with synthetic price/generation/benchmark rows; only the
generation fetch, benchmark file input and Data Trust provenance sidecars are
replaced locally. It neither downloads nor writes to the production cache.
Forward reconciliation still runs its real parser and comparison calculation.

The shared `metric_columns()` helper was already present before this change.
This increment applies it only to pure metric rows in Revenue, Renewable and
Forward. Input fields and paired charts keep their original columns. Data Trust
was deliberately left alone after the clean-base visual check found no clipping.

## Repeat

```sh
git diff --binary --full-index 47a8495 4b4df23 -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-27-metric-layout-evidence/code.patch
PYTHONPATH=. .venv/bin/streamlit run docs/audits/2026-09-27-metric-layout-evidence/harness.py
PYTHONPATH=. .venv/bin/streamlit run docs/audits/2026-09-20-step3d-evidence/browser/revenue_harness.py
```

For a clean comparison, archive base `47a8495` to a separate temporary
directory and copy `harness.py` into it; use its checkout as `PYTHONPATH`.
The baseline/candidate measurement did not use live imports or fetched market
data. Direct click/download flows and all Revenue parameter combinations remain
in the manual smoke checklist rather than being certified by this check.
