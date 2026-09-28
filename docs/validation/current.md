# Current verification snapshot

Verified **2026-09-28** against Forward chart follow-up code commit `d6f101c`,
based on released `main` `87a9533` (ordinary merge of #101). #100 was
ordinarily merged as `dfcf736`; its second parent was reviewed head `938cf03`.
#101 was ordinarily merged as `87a9533`; its second parent was reviewed head
`af24781`. Both merge trees matched their reviewed heads. The preceding #101
snapshot is archived [verbatim](2026-09-28-cockpit-export-copy.md). This
follow-up remains an unmerged review candidate.

[Handoff](../audits/2026-09-28-forward-chart-followup-handoff.md) ·
[Frozen patches and browser record](../audits/2026-09-28-forward-chart-followup-evidence/README.md) ·
[Remaining work](follow-ups.md).

## Checks and scope

| Check | Result |
|---|---|
| Clean baseline red checks | Apply only the new test cases to a clean `87a9533` archive: **5 failed / 34 passed**. The saved red-test patch keeps the base's leaking helper so the isolation failure remains observable. |
| Candidate focused tests | `tests/test_trader_benchmark.py`: **39 passed**. Ruff and whitespace checks passed. |
| Browser presentation | Offline production figure at 390 CSS px: 17/20/26/32-year samples all render first and last whole-year labels. This is a chart-only fixture, not the complete Forward page. |
| Full local suite | Clean archive of code commit `d6f101c`: **2121 passed / 2 skipped / 29 warnings**, including slow tests, in 291.01s; command exit 0. Warnings are the tracked Streamlit DataFrame-attrs and pandas empty-concat warnings. |
| Remote CI | Check the [draft PR #102](https://github.com/crashchen/euro-bess-radar/pull/102) at its exact final head. The local full-suite result above is separate from remote CI. |

The change restores `DeltaGenerator.file_uploader` after the AppTest panel
render and avoids adding a crowded final tick adjacent to the preceding one.
No curve coordinates, comparison values, export workbook, solver, cache,
Project Case schema or market-data input changes. The chart browser probe
checks four synthetic horizons; it does not certify every viewport or live
provider response.

## Repeat

```sh
.venv/bin/python -m pytest tests/test_trader_benchmark.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff --check
git diff --binary --full-index 87a9533 d6f101c -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-28-forward-chart-followup-evidence/code.patch
```

Both code-patch hash commands return
`b11a2085888244d9e2722e63ce004ea1bc93528101f07ff4d2845d4375e1fa04`.
