# Current verification snapshot

Verified **2026-09-22** against code commit
[`e39968dd93eb72f8c792c11049035f2210b8c5eb`](https://github.com/crashchen/euro-bess-radar/commit/e39968dd93eb72f8c792c11049035f2210b8c5eb),
the multi-day replay and forecast-policy Excel provenance correction. This
code is **awaiting independent review**, not merged. Main is
`cc6b1e24c527d5eec5454ffbdb2cfcc07cabaff0`, the ordinary merge of
reviewed [#95](https://github.com/crashchen/euro-bess-radar/pull/95).
Steps 1–4 (#86–#93) and follow-ups #94–#95 are merged. Later commits on this
branch document the tested code; they must not be read as new numerical
validation. The #95 snapshot is archived [verbatim](2026-09-22-floor-followup.md).

[Review handoff](../audits/2026-09-22-cockpit-export-handoff.md) ·
[Reproduction evidence](../audits/2026-09-22-cockpit-export-evidence/README.md) ·
[Remaining work](follow-ups.md).

## Executed checks

| Check | Result | Evidence |
|---|---|---|
| Full local suite | **2111 passed / 2 skipped**, 2113 collected, including the 35 slow cases, 296.83 seconds, 29 warnings | [Full output](../audits/2026-09-22-cockpit-export-evidence/full-suite.txt) |
| New regression file | 6 passed on candidate; the same file on clean `git archive cc6b1e2` gives 5 assertion failures / 1 pass | [Baseline output](../audits/2026-09-22-cockpit-export-evidence/baseline-tests.txt) |
| Related non-slow suites | 106 passed, 6 slow cases deselected | [Targeted output](../audits/2026-09-22-cockpit-export-evidence/targeted-tests.txt) |
| Ruff | Passed for `src/ app.py tests/` | [Output](../audits/2026-09-22-cockpit-export-evidence/ruff.txt) |
| Synthetic baseline/candidate probe | Same multi-day gross revenue, degradation cost and valid days in both replay modes; same forecast comparison revenue and valid days; same global sidebar assumptions. The diff contains only the corrected export rows | [Baseline](../audits/2026-09-22-cockpit-export-evidence/baseline-probe.json), [candidate](../audits/2026-09-22-cockpit-export-evidence/candidate-probe.json) |

Remote CI is a separate gate. Verify the draft PR's **exact-head Checks**;
the prior #95 checks do not validate this increment. The same pre-existing
warning categories appear: Streamlit DataFrame-attrs serialization and pandas
concat FutureWarnings. The two opt-in Kaleido/Chrome PDF tests remain skipped
by default. This increment changes no PDF output.

## Behavior and evidence boundaries

On Run, the multi-day replay saves an Excel assumptions copy that identifies
either DA-only MILP or two-stage DA+IDA1 MILP, its own capture haircut, and the
sidebar CapEx actually passed to its ex-post linear degradation calculation.
That CapEx does not affect dispatch or gross revenue. The forecast-policy
export identifies its sequential DA+IDA1 MILP; existing rows separately
describe optional reserve, triple and stochastic variants. Neither correction
changes the global Data Trust table, solvers, cash, result-table numbers or
Excel numeric cell types. Panel-version bumps make older session snapshots
stale until Run creates an export with current labels.

The new tests inspect actual generated XLSX bytes with openpyxl and exercise
both multi-day solver modes. Existing AppTest suites exercise session
persistence and stale-result gating. There is no new browser-download,
native-Excel rendering, live-provider, ESS-consumer or full responsive-layout
acceptance claim. Global assumptions may be absent in direct helper calls;
the established `None` passthrough remains, while normal `app.py` supplies a
nonempty table.

## Reproduction

From the repository root:

```sh
.venv/bin/python -m pytest tests/test_cockpit_export_provenance.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff --binary --full-index cc6b1e2..e39968d -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-22-cockpit-export-evidence/code.patch
```

Both hash commands return
`b295b4bf8e648c3e262d375f6cbc92a52001af87c456e7235539185c90b051ae`.
