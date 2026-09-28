# Forward benchmark chart evidence — 2026-09-28

Base `76dc758` (#99 ordinary merge); code commit `49a2d77`. The frozen
[code patch](code.patch) covers `src/`, `tests/` and CI. Its SHA-256 is
`61d15bed759c368793ac24402584f02853fbcd81a9a642653b8c46c364dc449f`.

The original Forward benchmark chart gives Plotly a numeric calendar-year
axis without tick constraints. With 2027/2028 synthetic annual data, the real
renderer showed `2027, 2,027.2, 2,027.4, 2,027.6, 2,027.8, 2028` at a
1280px viewport. The candidate labels only actual integer years. For longer
curves it chooses at most eight labels, including the first and last year;
both price/revenue traces retain their original x/y values.

The [browser measurements](browser-measurements.json) come from two surfaces:
the existing #98 page harness for the 1280px baseline and a synthetic
[chart-only Streamlit harness](chart_harness.py) for candidate widths 390,
1280 and 1440 CSS px. The latter calls the production chart builder but does
not exercise the full Forward page. At 390px the old fixed light legend color
and default placement made the legend faint and overlapped the x-axis title.
The candidate inherits the active chart theme's text color and places the
legend below the axis title; measured separation is at least 33px at all
three widths. The chart gains 60px of height and a 120px bottom margin for
the legend. The light-theme screenshot was inspected, but no live forward
curve, benchmark upload, download, or every possible year-range was tested.

The new test file on a clean base archive yields **2 failed / 32 passed**:
the two chart tests cannot import the new builder, while the pre-existing
tests and new unit-caption compatibility check pass. At the code commit it
yields **34 passed**. The caption test runs the actual benchmark section with
a synthetic upload and asserts that the shortened euro amounts retain their
`EUR/MW/yr` explanation. This does not claim AppTest verifies pixel layout.
The complete suite on a clean archive of code commit `49a2d77` exited 0:
**2115 passed / 2 skipped**, 29 previously tracked warnings, 298.35 seconds.
The first attempt produced the same pytest pass summary but its surrounding
zsh command used the shell's read-only `status` variable and exited 1 after
pytest; the corrected wrapper rerun supplies the clean process exit.

Repeat the code and chart checks:

```sh
git diff --binary --full-index 76dc758 49a2d77 -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-28-forward-chart-evidence/code.patch
.venv/bin/python -m pytest tests/test_trader_benchmark.py -q
PYTHONPATH=. .venv/bin/streamlit run docs/audits/2026-09-28-forward-chart-evidence/chart_harness.py --server.address 127.0.0.1 --server.port 8613
```

For the fixed-width browser comparison, serve only this evidence directory
on localhost with `.venv/bin/python -m http.server 8614 --bind 127.0.0.1
--directory docs/audits/2026-09-28-forward-chart-evidence`, then open
`http://127.0.0.1:8614/widths.html`. The three iframes target port 8613;
they use synthetic values and do not read the production cache. Stop both
local servers after inspection.
