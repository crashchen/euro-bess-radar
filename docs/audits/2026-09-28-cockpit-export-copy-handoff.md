# Cockpit export-copy handoff — 2026-09-28

Base `938cf03` is the draft Forward chart PR #100 documentation head; code
commit `7ec5302` is stacked on it. Keep the two diffs separate during review
and do not merge either on this handoff alone. The frozen
[export-only code patch](2026-09-28-cockpit-export-copy-evidence/code.patch)
has SHA-256
`57a410fd99f48e8aca269d777dd6d059ec08f4236f129add6adec9274203bd14`.
See the [saved probe](2026-09-28-cockpit-export-copy-evidence/README.md) and
[current verification snapshot](../validation/current.md).

The multi-day DA+IDA1 Assumptions row now discloses that its replay uses
realised IDA prices. Forecast-policy Assumptions now distinguish the DA-only
baseline, sequential policy and ex-post ceiling. The direct frontier export
helper adds its CapEx row when a nonempty input table omitted it. Source
assumptions are copied, and `None` stays `None`. The three saved-result panel
versions advance because old session bundles contain the prior export copy.

Review these boundaries independently:

1. Place the new test file on a clean `938cf03` archive: expect **4 failed /
   3 passed**, with no collection errors. The candidate file has **7 passed**.
   Check the saved workbook numeric cell remains type `n`, value `123.45`,
   and that Dispatch model values fit their stored Excel column width.
2. Recompute both frozen-patch hashes. The production diff should touch only
   `src/pages/simulation_cockpit.py`; no solver, cash calculation, Project Case
   schema, CI or dependencies should change.
3. Compare `base.json` with `candidate.json`: all global assumption rows,
   the numeric export cell and the `None` frontier path are unchanged. Only
   the named export-copy descriptions and one missing-row case may differ.
4. Check each panel's saved-result fingerprint includes its new version ID,
   so an old session cannot re-export the old copy. The deliberate result
   invalidation requires Run; this is a UI/session behavior change, not a
   solver result change.

The clean code-commit full suite finished **2116 passed / 2 skipped /
29 warnings** (including slow tests) in 292.02s, exit 0. The warnings are
the previously tracked Streamlit attrs and pandas empty-concat warnings.
Check both remote CI jobs at the eventual exact PR head separately.

The probe does not render the whole Assumptions sheet or exercise a live
provider. Its source and exact base/candidate outputs are checked in for
independent reproduction.
