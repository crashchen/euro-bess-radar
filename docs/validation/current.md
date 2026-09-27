# Current verification snapshot

Verified **2026-09-23** against code commit
[`2bb14f7`](https://github.com/crashchen/euro-bess-radar/commit/2bb14f7),
the Cockpit strategy-name XLSX readability candidate. This code is **awaiting
independent review**, not merged. Main is
`175c6a7fba90e0b6e8774cc51f6cf03241eb7610`, the ordinary merge of
reviewed [#96](https://github.com/crashchen/euro-bess-radar/pull/96).
Steps 1–4 (#86–#93) and follow-ups #94–#96 are merged. Later documentation
commits must not be read as new numerical validation. The #96 snapshot is
archived [verbatim](2026-09-22-cockpit-export.md).

[Review handoff](../audits/2026-09-23-strategy-export-handoff.md) ·
[Saved-file evidence](../audits/2026-09-23-strategy-export-evidence/README.md) ·
[Remaining work](follow-ups.md).

## Executed checks

| Check | Result | Evidence |
|---|---|---|
| New regression on clean base | 1 assertion failure: long strategy cell lacks `wrap_text` | [Evidence](../audits/2026-09-23-strategy-export-evidence/README.md) |
| New regression on candidate | 1 passed | `tests/test_export.py::test_cockpit_strategy_names_are_readable_without_changing_values` |
| Related export/disclosure suites | 60 passed / 2 opt-in chart-render skips | `tests/test_export.py`, `test_step3d_export_disclosure.py`, `test_step3d_cockpit_disclosure.py`, `test_cockpit_export_provenance.py` |
| Full local suite | 2112 passed / 2 skipped, 2114 collected, including 35 slow cases; 297.84 seconds, 29 warnings | `.venv/bin/python -m pytest tests/ -q` |
| Ruff and diff check | Passed; no whitespace errors | `.venv/bin/ruff check src/ app.py tests/`; `git diff --check` |
| Saved-XLSX render | Both previously clipped strategy names visible in candidate; baseline clips them | [Before](../audits/2026-09-23-strategy-export-evidence/baseline.png), [after](../audits/2026-09-23-strategy-export-evidence/candidate.png) |
| Saved-cell comparison | All populated values, data types and number formats equal; only strategy wrapping and row heights differ | [Base](../audits/2026-09-23-strategy-export-evidence/baseline.xlsx), [candidate](../audits/2026-09-23-strategy-export-evidence/candidate.xlsx) |

Remote CI is a separate exact-head gate; check the draft PR's current head.
The two opt-in Kaleido/Chrome chart-render skips are not evidence of successful
chart rendering. The saved-XLSX image is an Artifact Tool read-only render,
not a screenshot from Microsoft Excel. The synthetic fixture has stubbed
Cockpit solvers and does not test market revenue economics.

## Behavior and scope

Only the `Strategy comparison` sheet in `cockpit_tables_to_excel` gains wrapped
strategy-name cells and row heights sufficient to display long labels inside
the existing width-30 column. The source strings, table values, numeric cell
types and `#,##0.00` formats are unchanged. No solver, cash, session cache,
assumptions or Project Case code is touched. Normal app downloads have not
been clicked in a browser during this increment; the production exporter and
saved workbook bytes were tested directly.

## Reproduction

```sh
.venv/bin/python -m pytest tests/test_export.py::test_cockpit_strategy_names_are_readable_without_changing_values -q
.venv/bin/python -m pytest tests/test_export.py tests/test_step3d_export_disclosure.py tests/test_step3d_cockpit_disclosure.py tests/test_cockpit_export_provenance.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff --binary --full-index 175c6a7 2bb14f7 -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-23-strategy-export-evidence/code.patch
```

Both hash commands return
`cd1cb1ca7abd4f8a0c4a6141a8730da196c7e64bda0d654c20de123242ac97d1`.
