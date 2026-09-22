# Current verification snapshot

Verified **2026-09-22** against code commit
[`886b2d5605c770fb34645ba7636f9324d49ec6b4`](https://github.com/crashchen/euro-bess-radar/commit/886b2d5605c770fb34645ba7636f9324d49ec6b4),
the frontier/floor export-provenance and failed-retry follow-up. This code is
**awaiting independent review**, not merged. Main is
`bd4bb888775fb775b61de10dc6715b9673b1401c`, the ordinary merge of
reviewed frontier [#94](https://github.com/crashchen/euro-bess-radar/pull/94).
Steps 1–4 (#86–#93) and #94 are merged. Later commits on this branch record
documentation and evidence only; they do not change this tested code.

[Machine-readable result](2026-09-22-floor-followup.json) ·
[Review handoff](../audits/2026-09-22-floor-followup-handoff.md) ·
[Reproduction evidence](../audits/2026-09-22-floor-followup-evidence/README.md) ·
[Archived #94 snapshot](2026-09-22-frontier.md) ·
[Archived Step 4 snapshot](2026-09-20.md).
Earlier counts and limitations remain bound to their own revisions; this
entry does not retroactively verify them on the current code.

## Executed checks

| Check | Result | Evidence |
|---|---|---|
| Full local suite | **2105 passed / 2 skipped**, 2107 collected, all 35 slow cases, 308.83 seconds, 29 warnings | [Full output](../audits/2026-09-22-floor-followup-evidence/full-suite.txt) |
| Collection | 2107 collected; 35 selected by the slow marker | `pytest tests/ --collect-only -m slow -q` |
| New regression file | 5 passed on candidate; the same file on clean `git archive bd4bb88` gives 4 assertion failures / 1 pass | [Baseline](../audits/2026-09-22-floor-followup-evidence/baseline-tests.txt) |
| Related frontier/floor suites | 155 passed | [Targeted log](../audits/2026-09-22-floor-followup-evidence/targeted-tests.txt) |
| Ruff | Passed for `src/ app.py tests/` | [Output](../audits/2026-09-22-floor-followup-evidence/ruff.txt) |

The two opt-in PDF chart-render checks remain skipped by default unless
`BESS_PULSE_RUN_KALEIDO_TESTS=1` and working Kaleido/Chrome are available.
This increment changes no PDF or workbook layout and makes no new PDF-render
claim. Remote CI is a separate gate: consult the draft PR's **exact-head
Checks** after it runs. A green #94 run is not a test of this candidate.
The warnings were the previously observed 23 Streamlit DataFrame-attrs
serialization warnings and 6 pandas concat FutureWarnings; no NumPy scalar
conversion warning appeared. The non-slow subset is **2070 passed / 2 skipped**
by subtraction from this full run, not by a separate fast test command.

## Behavior and evidence boundaries

The merged #94 content fingerprint still protects the full DA frame and exact
selected dates. On this review branch, frontier/floor Excel assumption copies
label the actual DA-only MILP, no sidebar DA-slippage capture and inherited
linear-wear CapEx. The floor saves its export assumptions at successful Run;
later unrelated sidebar changes do not rewrite the workbook or rerun the
solver. Both panels clear the previous success immediately before a valid-input
explicit retry; a caught failure therefore cannot make that success reappear
on the next ordinary rerun. Merely reverting changed frontier inputs without
pressing Run retains #94's restore-from-cache behavior.

AppTest drives the actual production panels and real local solvers. It reads
the generated XLSX bytes with openpyxl, checks corrected labels and numeric
cell type, and verifies the saved floor Assumptions/result sheets across an
in-place metadata edit and a fresh Run. This is **saved-file inspection**, not
a browser download, native Excel render or responsive-layout acceptance.
No live provider data, production cache mutation, ESS execution or Project
Case model/wire change was added.

[Remaining work](follow-ups.md) retains separate metric-page layouts, existing
strategy-name clipping, manual/ESS acceptance, warnings and versioned
settlement questions. The #94 snapshot is archived verbatim at
[2026-09-22-frontier.md](2026-09-22-frontier.md); its test count is not a
fresh run on this branch.

## Reproduction

From the repository root:

```sh
.venv/bin/python -m pytest tests/test_floor_frontier_followups.py tests/test_frontier_result_identity.py tests/test_simulation_cockpit_contracted_floor.py tests/test_simulation_cockpit_frontier.py tests/test_cycle_frontier.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff --binary --full-index bd4bb88..886b2d5 -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
```

The frozen patch SHA-256 is
`d378f1571698bda7cd767dd221adb5fcf7cf19a100da639b74ee9170c3cc333f`.
The handoff records the clean-baseline procedure and known limits.
