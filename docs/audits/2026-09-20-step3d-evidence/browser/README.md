# Step 3D browser evidence

Production renderers on synthetic, localhost-only Streamlit 1.55.0 harnesses,
checked in the Codex in-app browser on 2026-09-20. These are real screenshots,
not AppTest screenshots or mockups. No production cache is read or written.

- 1280 × 900 CSS px: sidebar expanded.
- 390 × 844 CSS px: sidebar becomes an overlay; close it before inspecting
  main content. All captions wrap without horizontal overflow (measurements.json).
- PC: full result plus expanded read-only Cockpit mirror. True public adapter
  and RunResult, DE_LU / FCR [symmetric], 2026-03-29, capacity cash EUR 456.
- Cockpit: actual persisted bundle renderer, comparison chart/table/download.
  Solver results are explicitly stubbed by the display-test factory; four
  unrelated subpanels are suppressed. This is not a numerical solver probe.
  Capacity basis and scope are accessible by horizontal table scrolling;
  mobile does not fit both 200px columns simultaneously. Full caption remains
  visible above the chart. The desktop/mobile download buttons were clicked,
  and the table/caption persisted. Solver no-recompute guarantees are covered
  separately by tests; no browser solver-call counter was installed.
- Revenue: real aggregate FCR + aFRR Up capacity and joint solver; energy-only
  mFRR Up excluded. Unrelated subpanels mocked as in the display test.
  Expand Joint MILP co-optimization estimate to see the caption.

Run one harness from the repository root, then inspect the two viewport sizes:

```sh
PYTHONPATH=. .venv/bin/streamlit run docs/audits/2026-09-20-step3d-evidence/browser/project_case_harness.py --server.address 127.0.0.1 --server.port 8623 --browser.gatherUsageStats false
PYTHONPATH=. .venv/bin/streamlit run docs/audits/2026-09-20-step3d-evidence/browser/cockpit_harness.py --server.address 127.0.0.1 --server.port 8624 --browser.gatherUsageStats false
PYTHONPATH=. .venv/bin/streamlit run docs/audits/2026-09-20-step3d-evidence/browser/revenue_harness.py --server.address 127.0.0.1 --server.port 8625 --browser.gatherUsageStats false
```

The PC/Cockpit harness also writes a synthetic workbook in the system temporary
directory. Reruns retain the Cockpit bundle. The production PC mirror is opened
via its expander; no cached real project is used.

Measurements read the rendered caption's `p` element: textContent, clientWidth,
scrollWidth, clientHeight; they never modify the DOM. Screenshots plus these
measurements establish fit of the listed disclosure strings, not arbitrary
products, every chart/table, all browsers, or the full manual smoke checklist.
Existing Revenue desktop `100% of power` metric clipping and existing small
screen chart density are outside 3D; Step 4 retains layout prioritization.
