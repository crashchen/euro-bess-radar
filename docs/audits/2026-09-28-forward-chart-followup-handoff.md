# Forward chart follow-up handoff — 2026-09-28

Base `87a9533` is the ordinary merge of #101 into main. Code commit
`d6f101c` fixes two non-blocking findings from the user's independent #100
review. This is a new, separate review candidate; do not merge on the strength
of this handoff alone. The frozen [source/test/CI patch](2026-09-28-forward-chart-followup-evidence/code.patch)
has SHA-256
`b11a2085888244d9e2722e63ce004ea1bc93528101f07ff4d2845d4375e1fa04`.
The [evidence directory](2026-09-28-forward-chart-followup-evidence/README.md)
contains the separate red-test patch, focused baseline output and 390px
browser measurements; [current.md](../validation/current.md) records the
candidate's completed validation state.

A clean archive of the code commit passed the full suite, including slow tests:
**2121 passed / 2 skipped / 29 warnings** in 291.01s, exit 0. The warnings
are the tracked Streamlit DataFrame-attrs and pandas empty-concat warnings.
Remote CI is tracked separately at [draft PR #102](https://github.com/crashchen/euro-bess-radar/pull/102)
and must be checked at the final head.

The AppTest no longer leaves `DeltaGenerator.file_uploader` replaced for later
tests: `patch.object` scopes the synthetic upload to the panel render and
restores the original method even if that render raises. The chart now replaces
its penultimate tick when appending the last year would create a shorter final
interval than the preceding tick spacing. At 390px this retains the last-year
label for the 17/20/26/32-year examples. It does not change trace x/y arrays,
reconciled numbers, export data or the model.

Independent review should:

1. Apply only `baseline-red-tests.patch` to clean `87a9533` source. Expect
   **5 failed / 34 passed**, one `file_uploader` leak and four final-tick
   spacing failures. Applying the entire candidate test file also imports
   its fixed AppTest helper and therefore is not a valid five-red baseline.
2. Recompute the frozen code patch SHA-256 and verify production changes stay
   in `src/pages/forward_scenarios.py`; tests stay in
   `tests/test_trader_benchmark.py`. Check that `patch.object` restores the
   method on both normal and exception exits.
3. Inspect the figure's x/y trace arrays and reconciliation/export paths.
   Confirm only tick-label selection changes at the UI layer.
4. Render 17/20/26/32-year cases at 390px using `render_cases.py`; verify
   first and last years are present. The stored DOM/AX label observations are
   chart-only, not a complete Streamlit page or live-market smoke test.

After independent review, retain the explicit per-PR merge authorization rule
from `CLAUDE.md`; this handoff does not invoke CC.
