# Current verification snapshot

Verified **2026-09-28** against Cockpit export-copy code commit `7ec5302`,
stacked on `938cf03`, the documentation head of draft Forward chart PR #100.
Neither candidate is merged; released `main` remains `76dc758` (#99 ordinary
merge) until separately authorized merges. The preceding chart snapshot is
archived [verbatim](2026-09-28-forward-chart.md), and the #99 pre-chart
[snapshot](2026-09-28-pre-forward-chart.md) remains revision-bound.

[Export-copy evidence and frozen patch](../audits/2026-09-28-cockpit-export-copy-evidence/README.md)
· [Forward chart evidence](../audits/2026-09-28-forward-chart-evidence/README.md)
· [Remaining work](follow-ups.md).

## Checks and scope

| Check | Result |
|---|---|
| Clean baseline contrast | New `test_cockpit_export_provenance.py` on a clean `938cf03` archive: 4 failed / 3 passed. Failures are assertion mismatches in the old export copy and missing frontier CapEx row; no collection error. |
| Candidate focused tests | The new file: 7 passed. With frontier identity and Step 3B batch-persistence suites: 117 passed, 23 previously tracked Streamlit attrs warnings. Ruff and whitespace checks passed. |
| Saved-artifact probe | Exact-base and code-commit JSON files show only the multi-day/forecast-policy export descriptions and one frontier missing-CapEx row change. Global rows, a numeric workbook value/type/format and the `None` frontier path are identical. |
| Full local suite | Clean archive of code commit `7ec5302`: **2116 passed / 2 skipped / 29 warnings**, including slow tests, in 292.02s; command exit 0. Warnings were the previously tracked Streamlit DataFrame-attrs and pandas empty-concat warnings. |
| Remote CI | Verify separately at each draft PR's exact final head; local checks are not remote CI. |

This candidate changes the export-assumption copy and saved-result version IDs
for three Cockpit panels. Older session bundles need a fresh Run before their
Excel provenance is shown. No solver, cash result, Project Case schema,
forward reconciliation or global sidebar assumption table changes. The
synthetic probe does not certify a full workbook render or live provider.

The parent chart candidate has its own [frozen review handoff](../audits/2026-09-28-forward-chart-handoff.md).
Its clean code-commit suite was 2115 passed / 2 skipped, 29 previously tracked
warnings. That result does not substitute for testing this export-copy code.

## Repeat

```sh
.venv/bin/python -m pytest tests/test_cockpit_export_provenance.py tests/test_frontier_result_identity.py tests/test_step3b_batch_result_persistence.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff --check
git diff --binary --full-index 938cf03 7ec5302 -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-28-cockpit-export-copy-evidence/code.patch
```

Both hash commands return
`57a410fd99f48e8aca269d777dd6d059ec08f4236f129add6adec9274203bd14`.
