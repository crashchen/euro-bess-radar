# Current verification snapshot

Verified **2026-09-22** on code commit
[`e11eacad27766e1f305d9444279ae82d261da8a1`](https://github.com/crashchen/euro-bess-radar/commit/e11eacad27766e1f305d9444279ae82d261da8a1),
the Frontier content-identity follow-up. This code is **awaiting independent
review**, not merged. Main is `6d986e3`, the ordinary merge of Step 4 [#93](https://github.com/crashchen/euro-bess-radar/pull/93);
Steps 1–4 (#86–#93) are merged. Later commits on this branch record documentation
and evidence; they do not change this tested code.

[Machine-readable result](2026-09-22.json) ·
[Handoff](../audits/2026-09-22-frontier-handoff.md) ·
[Previous dated snapshot](2026-09-20.md).
The previous snapshot is preserved verbatim as a dated record, not a new run.

## Executed checks

| Check | Result | Evidence |
|---|---|---|
| Full local suite | **2100 passed / 2 skipped**, 2102 collected, all 35 slow cases, 283.62 seconds | [Full log](../audits/2026-09-22-frontier-evidence/full-suite.txt), [JUnit](../audits/2026-09-22-frontier-evidence/full-suite.xml) |
| Collection | 57 test files, 1562 AST test definitions, 1562 collected callables | [Inventory and node IDs](../audits/2026-09-22-frontier-evidence/test-inventory.json) |
| New regression file | 43 passed; the same file on clean `git archive 6d986e3` gives 25 failed / 18 passed | [Baseline failures](../audits/2026-09-22-frontier-evidence/baseline-tests.txt), [compatibility case list](../audits/2026-09-22-frontier-evidence/verification.json) |
| Related frontier/floor suites | 150 passed | [Targeted log](../audits/2026-09-22-frontier-evidence/targeted-tests.txt) |
| Ruff | Passed for `src/ app.py tests/` | [Output](../audits/2026-09-22-frontier-evidence/ruff.txt) |

The 35 slow cases all passed. The non-slow subset is **2065 passed / 2 skipped**
(2067 collected), derived from the same full run, **not separately executed**.
The two opt-in PDF chart-render tests remain skipped; they require
`BESS_PULSE_RUN_KALEIDO_TESTS=1` and working Kaleido/Chrome. This increment changes
no PDF or workbook layout and makes no fresh rendering claim.

[Runtime](../audits/2026-09-22-frontier-evidence/runtime.json): Python 3.13.9,
Streamlit 1.55.0. [Warnings](../audits/2026-09-22-frontier-evidence/warning-summary.json):
23 Streamlit DataFrame-attrs warnings and 6 pandas concat FutureWarnings.
They are unchanged in this run; no NumPy scalar-conversion warning occurred.
Remote CI is a separate gate: consult the review PR's **exact-head Checks**.
The recorded green #93/main runs belong to their earlier heads, not this code.

## Behavior and evidence boundaries

Frontier now hashes the full DA frame, every selected date in order, relevant
solver/economic controls, consumed constants and a panel version. A same-shape
price/index correction hides its result/download and passes no merchant baseline
to the contracted-floor panel. Changes outside selected dates conservatively
invalidate too. Restoring original inputs restores the saved frontier without
solving; the floor's existing cache-clearing behavior requires its own Run.
The frontier export keeps the assumptions captured when that sweep ran.

AppTest drives real production panels and real solvers. It counts both the
sweep entry and every daily solver call, checks stale/reversion/download-button
visibility, and reads actual generated Excel bytes with openpyxl to verify the
assumption snapshot. It does not click a browser download or certify layout.

[Chrome smoke](../audits/2026-09-22-frontier-evidence/browser-check.json) used
synthetic hourly prices and the real panels. Run, floor propagation, price
correction, restoration, a direct download-button click and chart-template
change were exercised. Counts remained one sweep/four daily solves on unrelated
reruns. **Browser download completion/file contents were not confirmed**: event
observation timed out and the internal downloads page was blocked by the browser
tool. No workaround was attempted. Browser observations are a transcribed tool
record, not archived screenshots or responsive-layout acceptance.

No live API data, ESS execution, market cache mutation, or other-page layout
acceptance was added. [Scope evidence](../audits/2026-09-22-frontier-evidence/scope.json)
confirms 121 existing runtime/source/test/configuration files and all ten older
patches are unchanged; only the cockpit module and an existing test fixture
changed, plus one new regression file. All 26 real-cache file contents remained
unchanged. The core dispatch/frontier algorithms, Project Case and JSON wire
remain unchanged.

[Remaining work](follow-ups.md) retains remaining layouts, export clipping,
manual/ESS acceptance and optional version assertions. The separate floor export
still composes global assumptions at display time; it does not inherit this
frontier snapshot guarantee. Historical browser/export acceptance retains the
bounds in the [previous snapshot](2026-09-20.md).

## Reproduction

From the repository root:

```sh
.venv/bin/python -m pytest tests/test_frontier_result_identity.py tests/test_simulation_cockpit_frontier.py tests/test_simulation_cockpit_contracted_floor.py tests/test_cycle_frontier.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
```

The full run additionally used the existing read-only
[collection/outcome plugin](../audits/2026-09-20-step4-evidence/snapshot_plugin.py).
The handoff records exact commands, frozen patch, baseline procedure and limits.
