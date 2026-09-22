# Current verification snapshot

Verified on **2026-09-20**, against code
[`fb72dbf529e9e06ba1716aa66120b2f4f68fe4de`](https://github.com/crashchen/euro-bess-radar/commit/fb72dbf529e9e06ba1716aa66120b2f4f68fe4de),
the ordinary merge of reviewed Step 3D [#92](https://github.com/crashchen/euro-bess-radar/pull/92).
Steps 1, 1b, 2 and 3A–3D (#86–#92) are merged. Step 4 changes documentation and
verification records only. [Machine-readable snapshot](2026-09-20.json).

This is the current entry point for README, agent guidance and project notes.
Historical audit counts belong to their recorded commits; do not treat them
as fresh runs. Updating this entry never rewrites their handoffs or patches.

## Executed automated checks

| Check | Result | Evidence |
|---|---|---|
| Full local suite on the merged code | **2057 passed / 2 skipped**, 2059 collected, 308.86 seconds, all 35 slow cases included | [Full output](../audits/2026-09-20-step4-evidence/full-suite.txt), [JUnit](../audits/2026-09-20-step4-evidence/full-suite.xml) |
| Collection inventory | 56 test files, **1545 test function definitions**, 1545 collected callables, 2059 parameterized cases | [Per-file counts and node IDs](../audits/2026-09-20-step4-evidence/test-inventory.json) |
| Slow subset of that full run | 35 passed | [Run outcomes](../audits/2026-09-20-step4-evidence/run-results.json) |
| Non-slow subset of that full run | 2022 passed / 2 skipped, 2024 collected | Derived from the same outcomes; **not a separately executed fast command** |
| Corrected focused selector | `tests/test_analytics.py::TestDailySpreads`: 8 passed | [Execution record](../audits/2026-09-20-step4-evidence/focused-command.txt) |
| Ruff | Passed: `ruff check src/ app.py tests/` | [Output](../audits/2026-09-20-step4-evidence/ruff.txt) |
| Exact merged-head CI | Both `test` and Python 3.11 / Streamlit 1.55.0 compatibility check succeeded | [Main run](https://github.com/crashchen/euro-bess-radar/actions/runs/35506713370), [recorded job results](../audits/2026-09-20-step4-evidence/main-ci.json) |

Local runtime: Python 3.13.9, pytest 8.4.2, Streamlit 1.55.0. Package versions
are [recorded](../audits/2026-09-20-step4-evidence/runtime.json), not inferred
from the dependency ranges. CI's full job runs on Python 3.13; its second job
pins Python 3.11 and Streamlit 1.55.0 only. Other packages still resolve from
manifest ranges. That second job runs its selected real-panel smoke subset,
not all tests or the minimum allowed version of every dependency.

The two skips are the opt-in PDF chart-render cases in `test_export.py`.
`BESS_PULSE_RUN_KALEIDO_TESTS=1` and a working Kaleido/Chrome environment are
required to execute them. Existing separately rendered PDFs do not turn those
skips into passes.

Warnings in this run: **23 Streamlit warnings** about serializing DataFrame
attrs containing another DataFrame, plus **6 pandas concat FutureWarnings**
at `src/ancillary.py:1036` (two messages, three occurrences each).
[Classification](../audits/2026-09-20-step4-evidence/warning-summary.json).
The earlier NumPy scalar-conversion deprecation warning was **not reproduced**
in this run. No warning source was changed during housekeeping.

## Browser, export and consumer evidence

Step 4 did not launch a browser, fetch live market data or execute the ESS
consumer. The references below retain their original fixture, revision and
coverage; unchanged source permits reuse as historical evidence, not relabelling
it as a new manual run.

| Surface | Existing evidence | Limit |
|---|---|---|
| Batch/forecast run state | [Step 3B](../audits/2026-09-16-step3b-handoff.md): real AppTest panels and solver counters for rerun, stale and export assumptions | Use its exact disclosed browser/download scope; not every UI permutation |
| Avg Price and main metric layouts | [Step 3C](../audits/2026-09-16-step3c-handoff.md) and [3C follow-up](../audits/2026-09-19-step3c-r2-handoff.md): actual synthetic 1280/1440/390px browser checks and saved-file renders | Scoped to Market Overview, Project Case/mirror and Cockpit; not every amount or every page |
| DST capacity disclosure | [Step 3D](../audits/2026-09-20-step3d-handoff.md): actual PC/mirror/Cockpit/Revenue captions at 1280/390px; five XLSX, two PDFs; real download clicks on synthetic Cockpit | Workbook renders use Artifact Tool, not native Excel; Cockpit presentation economics are explicitly stubbed; stochastic DST numerical acceptance is not claimed |
| Radar annual-revenue JSON | Producer and golden-fixture tests in this full run; ESS control names/CPI wording and the matching fixture were read in sibling source `e7cdac0` | No fresh ESS test/UI/import/valuation run; no ESS file changed |
| Live imports and full manual suite | [47-item procedure](../runbooks/manual-ui-smoke.md) | Not a completed report. New JSON checks 44–47 are unexecuted; earlier evidence covers named subsets only |

Keep the cache distinction explicit: multi-day/forecast restore their stored
bundle when inputs return to the original values. Project Case deletes a stale
cache, so restoring its inputs still requires another Run. The frontier still
uses the older first/last-selected-day plus selected-day-count identity.

[Remaining work](follow-ups.md) contains the frontier content identity, remaining
page layout inventory, long XLSX strategy names, warnings, manual acceptance and
bounded defensive/model-contract follow-ups. Documentation completion does not
mean those features or checks are complete.

## Reproduce or refresh this record

Use the repository environment, from its root:

```sh
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/ --collect-only -q
.venv/bin/python -m pytest tests/test_analytics.py::TestDailySpreads -q
.venv/bin/ruff check src/ app.py tests/
```

The dated run additionally loaded the read-only
[collection/outcome recorder](../audits/2026-09-20-step4-evidence/snapshot_plugin.py)
and wrote JUnit. To regenerate those artifacts into a new temporary directory:

```sh
STEP4_OUTPUT=$(mktemp -d)
export STEP4_OUTPUT
PYTHONPATH=docs/audits/2026-09-20-step4-evidence:. .venv/bin/python -m pytest tests/ -q -p snapshot_plugin --junitxml="$STEP4_OUTPUT/full-suite.xml"
```

These commands must be run again before claiming a later code head is verified.
Keep fast/slow selection, full-suite results, skips, warnings, CI and browser
coverage distinct. The source inventory uses AST `test_*` definitions and
collected callable identities separately; their equality here is measured.

[Runtime scope](../audits/2026-09-20-step4-evidence/runtime-scope.json) confirms
all 123 tracked app/source/test/script/dependency/CI files are byte-identical to
`fb72dbf`. [Ten historical patches](../audits/2026-09-20-step4-evidence/historical-patches.json)
retain their original SHA-256 values. Step 4 has no code patch or baseline red
suite because it changes no product behavior.
