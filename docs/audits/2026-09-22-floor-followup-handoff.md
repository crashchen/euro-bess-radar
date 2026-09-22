# Frontier/floor export provenance and failed retries: review handoff

## Result and revisions

Reviewed frontier identity/snapshot [#94](https://github.com/crashchen/euro-bess-radar/pull/94)
was ordinarily merged as `bd4bb888775fb775b61de10dc6715b9673b1401c`.
This separate follow-up is a **review candidate**, not a merge request.

- Base: `bd4bb888775fb775b61de10dc6715b9673b1401c`.
- Tested code: `886b2d5605c770fb34645ba7636f9324d49ec6b4`.
- Frozen patch scope: `src/ tests/ .github/workflows/ci.yml`.
- Patch SHA-256: `d378f1571698bda7cd767dd221adb5fcf7cf19a100da639b74ee9170c3cc333f`.
- [Code patch](2026-09-22-floor-followup-evidence/code.patch),
  [reproduction evidence](2026-09-22-floor-followup-evidence/README.md),
  [current validation snapshot](../validation/current.md).

Only `src/pages/simulation_cockpit.py` changes in production code. The new
test file exercises real Streamlit panels, solvers and exported workbook bytes.
No dispatch/frontier/floor algorithm, Project Case schema or cash equation
changes. Later documentation/evidence commits leave the frozen code patch
unchanged. Historical patches and handoffs are not rewritten.

## Before and after

The floor cached its computed result, but rebuilt the Excel Assumptions sheet
from the current sidebar table on every rerender. A later metadata edit could
therefore place new assumptions next to old cash figures without recomputing.
Even on the original Run, the shared sidebar table could say `70%` DA capture,
`Greedy single-cycle` dispatch and `CapEx: Payback period only`, although the
frontier merchant baseline is raw DA-only MILP net of linear wear. The frontier
workbook had the same dispatch/CapEx contradiction.

Both exports now adapt **copies** of the global table to the actual DA-only
frontier basis: capture not applied, dispatch `DA-only MILP multi-cycle`, and
CapEx inherited for linear wear. The global Data Trust table is unchanged.
Frontier still saves its assumptions at successful Run; its panel version is
bumped so old in-session export snapshots require a new Run. Floor now builds
and saves its completed assumptions at successful Run, then downloads from
that snapshot. A separate floor panel-version gate rejects old bundles before
accessing the new field. `None` global assumptions retain the established
panel-local floor rows and the frontier's established `None` behavior.

With valid inputs, an explicit frontier or floor Run clears that panel's
previous success immediately before computing. A caught `ValueError` leaves
no previous table/chart/download to
reappear on the next ordinary rerun. Merely changing and restoring a frontier
input **without** pressing Run still restores its saved result as before.
Floor source invalidation continues to clear its own cache and require Run.

## Independent reproduction

The new five-case file runs against a clean `git archive bd4bb88` checkout:
**4 assertion failures / 1 compatibility pass**, with no import or collection
errors. On the tested code it passes all five cases. It covers panel-correct
frontier/floor workbook values, floor snapshot isolation from an in-place
global-table edit, a new explicit Run taking new metadata, old-session gates,
and success → failed retry → ordinary rerun for both panels.

```sh
git archive bd4bb88 | tar -x -C <empty-baseline-directory>
cp tests/test_floor_frontier_followups.py <empty-baseline-directory>/tests/
# In that directory, using this repository's virtualenv:
<venv-python> -m pytest tests/test_floor_frontier_followups.py -q
```

The five related suites pass **155 cases** on the tested code. The full suite
passes **2105 / 2 skipped** (2107 collected, all 35 slow cases) in 308.83
seconds, with 29 previously observed warnings. The [current snapshot](../validation/current.md)
and [full log](2026-09-22-floor-followup-evidence/full-suite.txt) bind those
counts to the tested code. Ruff and
`git diff --check` are separate checks. Remote CI must be read from the draft
PR's **exact head** once it runs; a prior green #94 check is not this PR's CI.

The workbook assertions inspect actual saved XLSX bytes with openpyxl,
including numeric cell type and the unchanged result sheet on unrelated
reruns. They do not certify native Excel rendering or a browser-downloaded
file. No live provider fetch, production cache mutation, ESS execution, or
whole-app visual acceptance was performed.

## Review focus

1. Confirm the copied global rows correctly describe the fixed DA-only MILP,
   no sidebar capture haircut and inherited linear-wear CapEx in **both**
   workbooks; check that the original global table remains unchanged.
2. Confirm floor result/cash and fingerprinted numerical inputs are unchanged, while
   both version gates prevent older session bundles from reaching a missing
   export field or retaining misleading export metadata.
3. Reproduce both failed-retry three-step UI paths and the existing
   input-change/reversion path without pressing Run.
4. Check docs mark #94 merged, this follow-up unmerged, and leave the
   archived #94 snapshot and historical evidence at their recorded revisions.

The remaining actual metric/strategy-name clipping, unexecuted manual checks,
dependency warnings and settlement-contract questions remain in
[follow-ups](../validation/follow-ups.md). The user invokes CC separately;
this task does not launch an external reviewer or authorize a merge.
