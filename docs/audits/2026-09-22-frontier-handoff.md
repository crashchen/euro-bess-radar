# Frontier content identity: review handoff

## Result and revisions

After the user's authorization, reviewed Step 4 [#93](https://github.com/crashchen/euro-bess-radar/pull/93)
was ordinarily merged as `6d986e30d45bb0fe932dd0477e228b361c4ca240`.
Its approved `c646877` head, both green checks and all 17 evidence hashes were
rechecked first. This separate Frontier follow-up is **awaiting independent
review**; no automatic merge is requested.

- Base: `6d986e30d45bb0fe932dd0477e228b361c4ca240`.
- Tested code: `e11eacad27766e1f305d9444279ae82d261da8a1`.
- Frozen patch scope: `src/ tests/ .github/workflows/ci.yml`.
- Patch SHA-256: `c11506a692f6244a117b9cec224e6ecbc0004269317997b0a4dcf30fabd73824`.
- [Code patch](2026-09-22-frontier-evidence/frontier-code.patch),
  [verification record](2026-09-22-frontier-evidence/verification.json),
  [current snapshot](../validation/current.md).

Later commits contain documentation and evidence only. The public Project Case
fingerprint, algorithms, grid rules, SoC, FEC, cash settlement and JSON wire are
unchanged. The fingerprint modified here is private session-state identity.

## Before and after

A price correction could preserve a frame's shape and selected-date endpoints,
so the old frontier cache still appeared current. Replacing a middle selected
date with another loaded day also escaped first/last/count identity. The old
export additionally rebuilt assumptions from current sidebar state beside the
saved result.

The new key includes the complete DA frame through the existing content-hash
helper, the exact ordered selected-date list, the active solver/economic knobs,
enabled liquidity volume/share, consumed VOM/annualisation/lifetime/tolerance
constants and a version prefix. It remains a tuple for downstream compatibility.
The full frame is hashed, so a correction outside selected dates conservatively
invalidates too. The solver does not read frame attrs, and attrs are not hashed.
Theme, IDA, reserve data and unrelated sidebar audit text are not frontier
solver inputs. Duplicate cap entries can conservatively invalidate; cap-order
permutation remains equivalent.

Run saves the completed export assumptions alongside the result. Ordinary
reruns/download rendering reuse them without solving. Changing an audit-only
assumption retains the original snapshot until an explicit new Run. On a key
mismatch no chart, table, metric, download or merchant context is returned.
Restoring original inputs restores the saved frontier without solving.

The floor's existing rule remains: absent frontier context clears the floor
cache. After restoring the frontier, the user must Run the floor again. Its
selected-row-value guard still protects a changed solved baseline after an
explicit rerun. Legacy frontier cache tuples fail the new version gate before
accessing the new snapshot field.

## Independent reproduction

The new 43-case file runs normally on the clean baseline (signature adaptation
is confined to test input setup). It produces **25 assertion failures / 18
passes**, with no import/collection errors. The 18 named compatibility controls
are listed in `verification.json`; they pass on both revisions.

```sh
git archive 6d986e30d45bb0fe932dd0477e228b361c4ca240 | tar -x -C <empty-baseline-directory>
cp tests/test_frontier_result_identity.py <empty-baseline-directory>/tests/
# Run from that clean directory, using this repository's installed environment:
<venv-python> -m pytest tests/test_frontier_result_identity.py -q
```

At the tested code head:

- **43 new cases pass**; related frontier/floor suites: **150 passed**.
- Full suite: **2100 passed / 2 skipped**, **2102 collected**, all **35 slow**
  cases, **283.62 seconds**. The skips remain opt-in PDF chart renders.
- Collection: **57 files / 1562 AST definitions / 1562 collected callables**.
- Ruff passed; `git diff --check` passed with the evidence `.gitattributes`
  preserving patch context bytes. No earlier patch was changed.
- 29 warnings: 23 Streamlit attrs serialization warnings and 6 pandas concat
  FutureWarnings. No NumPy scalar-conversion warning occurred in this run.

[Full logs, JUnit and environment](2026-09-22-frontier-evidence/README.md) are
normalized only for workstation paths/hostname and presentation whitespace.
Fast/slow subsets in the record are derived from the full run, not separate
commands. Remote CI must be checked at the review PR's exact head; local results
and the earlier #93 green CI are not substitutes.

AppTest drives both real panels, wraps the real sweep and daily solver, and
reads actual production-generated Excel bytes. For its three selected days and
two caps there are six daily solver entries per Run, zero added on theme/rerun/
stale/restoration/floor Run, and zero for empty or invalid windows. An explicit
new sweep adds exactly six. It verifies same-shape price/index corrections,
out-of-window corrections, middle-date replacement, legacy bundles, snapshot
copy isolation, download-button gating and actual new cash after correction.
The existing test file changes only to supply the now-required `primary_df`.

## Browser and other boundaries

[Chrome smoke record](2026-09-22-frontier-evidence/browser-check.json) and
[reproducible synthetic harness](2026-09-22-frontier-evidence/browser-harness.py)
cover initial Run, floor Run, price correction, restoration, direct Excel-button
click and theme rerun. One sweep/four real daily solver calls remained unchanged
through unrelated actions. **Browser download completion was not certified**:
the download-event wait timed out and browser policy blocked its internal
downloads page; no bypass was attempted. Saved-workbook assertions come from
AppTest, not a claim about a file retrieved from Chrome. No responsive-layout,
PDF-render or screenshot acceptance was added.

[Scope inventory](2026-09-22-frontier-evidence/scope.json) confirms 121 existing
runtime/source/test/config files and all ten historical patches are unchanged.
Only `src/pages/simulation_cockpit.py`, one existing test fixture and one new
test file differ. All 26 real-cache file hashes and the file set were unchanged.
No live API fetch or ESS execution/change occurred.

Bounded pre-existing follow-ups remain separate:

- Floor exports still compose the global assumptions table at display time;
  its numerical inputs and inherited merchant values remain guarded. A future
  floor export-snapshot change should also preserve the frontier's recorded
  provenance. This increment's snapshot guarantee is for frontier downloads.
- After an explicit failed retry with unchanged inputs, an earlier successful
  frontier can reappear on a later rerun. The failure message is shown on the
  failed attempt; result replacement/failure-state policy is unchanged here.
- CC's Step 4 wording correction is accepted: PC tests detect registry/profile
  v2 drift, but the screening-side DE_LU assertion lacks a direct version binding.
  The optional follow-up now states that distinction rather than claiming the
  screening side has complete protection.

README, agent guidance, the frontier contract and the current validation entry
are reconciled. Historical handoffs and the 2026-09-20 snapshot remain intact.
The later local-note synchronization receipt records private-note scope only;
no Vault text or paths are published. The user retains ownership of invoking CC.
