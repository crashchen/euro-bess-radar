# Forward benchmark chart handoff — 2026-09-28

Base: `76dc758` (merged #99). Code commit: `49a2d77`. This is a review
candidate; do not merge on the strength of this handoff. The frozen
`src/ tests/ .github/workflows/ci.yml` [patch](2026-09-28-forward-chart-evidence/code.patch)
has SHA-256
`61d15bed759c368793ac24402584f02853fbcd81a9a642653b8c46c364dc449f`.
The [current snapshot](../validation/current.md) records check status and
the [evidence directory](2026-09-28-forward-chart-evidence/README.md) records
the synthetic browser boundary.
The full suite on a clean code-commit archive exited 0 with **2115 passed /
2 skipped**, 29 previously tracked warnings, in 298.35 seconds; targeted
Ruff and whitespace checks passed.

## Change

The Forward external benchmark comparison still plots the exact annual
benchmark and model points. Its x-axis now labels actual whole calendar
years, with no more than eight labels for long curves. The legend no longer
overlaps the x-axis title at 390px and inherits the active theme's text
color instead of forcing a pale color on a light chart. Chart height grows
from 360px to 420px to reserve bottom space for that legend. No comparison value,
table, workbook, solver, input or cache path changed. A small AppTest fixes
the `EUR/MW/yr` caption that gives the shortened benchmark KPI values their
unit.

## Independent review requests

1. Put the new test file on a clean `76dc758` archive: expect **2 failed /
   32 passed**. Both failures must be missing `_benchmark_comparison_figure`;
   the unit-caption compatibility check must pass. On the candidate, expect
   **34 passed**.
2. Recompute both patch hashes. Compare the figure's numeric x/y trace data
   before and after; only axis/legend presentation should differ.
3. Run the production Forward benchmark section with a synthetic upload and
   inspect actual ticks. The existing #98 page harness reproduces the
   fractional baseline at 1280px. The new chart-only harness plus fixed-width
   iframe page checks 390/1280/1440px. Confirm the legend is readable in a
   light theme and inspect a dark theme separately; the recorded browser
   measurements do not claim a live provider or full-page responsive pass.
4. Check that the PR changes only Forward presentation and its tests/docs;
   the separate Cockpit export-copy follow-up must not leak into this diff.

The browser fixture has two years, so long-horizon label density is pinned
by the figure-level test rather than a rendered 25-year screenshot. The
operator's full-page, live-data checks remain in the manual smoke list.
