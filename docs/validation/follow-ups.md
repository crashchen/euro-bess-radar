# Work remaining after the September audit sequence

The [dated verification snapshot](current.md) records the current export/retry
candidate. Step 4 housekeeping is merged in #93 as `6d986e3`, and the frontier
content-identity/export-snapshot fix is merged in [#94](https://github.com/crashchen/euro-bess-radar/pull/94)
as `bd4bb88`. Its [archived snapshot](2026-09-22-frontier.md) and
[handoff](../audits/2026-09-22-frontier-handoff.md) remain revision-bound.
Historical findings remain in their original handoffs and patches; pending
review must not be presented as merged or as covered by an earlier test run.

| Priority | Work | Current evidence and required boundary |
|---|---|---|
| Implemented; awaiting review | Frontier/floor export provenance | Both Excel copies now label the actual DA-only MILP, no sidebar capture haircut and inherited wear CapEx. A successful floor Run saves a panel-correct assumptions snapshot with its result; later sidebar changes cannot rewrite that workbook. The frontier panel version invalidates old export snapshots. Numerical inputs, merchant baseline and cash are unchanged. See the [review handoff](../audits/2026-09-22-floor-followup-handoff.md) |
| Implemented; awaiting review | Explicit failed retries in frontier and floor | Both panels now clear their previous success before a valid-input solver retry; a caught `ValueError` leaves no result to reappear on a later rerun. Changed-input guards and the frontier's input-reversion-without-Run behavior remain. See the [review handoff](../audits/2026-09-22-floor-followup-handoff.md) |
| Next export pass | Multi-day replay and forecast-policy solver provenance | Both saved Excel assumptions tables correct the capture row but can retain the sidebar's `Dispatch model: Greedy single-cycle` when its switch is off, although the replay and forecast-policy paths use LP dispatch. The multi-day replay also uses CapEx in its degradation calculation while its inherited row says `Payback period only`. Audit each exported strategy's actual basis, then correct the Run-time export copies without changing the global Data Trust table, solver inputs, or cash. Pin the saved workbook rows and numeric outputs in focused tests. |
| Next visual pass | Remaining metric pages | Revenue Estimation has 48 metric call sites, Forward Scenarios 6, Renewable Correlation 4 and Data Trust 4. These are static call sites, including alternative branches, not simultaneously visible card counts. Audit actual loaded panels at 1280/1440/390px before expanding the Step 3C layout claim |
| Next export pass | Cockpit strategy names | Actual Step 3D XLSX rendering found long existing strategy names clipped in the width-30/no-wrap column. Full names remain in Assumptions. Test saved-file rendering with an appropriate wrap/width change; preserve numeric types/precision |
| Operator verification | Manual checks 1–47 | Existing browser evidence covers named synthetic panels and selected interactions. Live imports/fetches, every parameter combination and Radar→ESS consumer reconciliation have not all been run. Record each tested revision; do not mark the entire checklist passed |
| Dependency maintenance | Observed warnings | The dated Step 4 run emitted Streamlit DataFrame-attrs serialization warnings and pandas concatenation FutureWarnings. Address reproducible sources in small behavior-preserving PRs. The historical NumPy scalar-conversion warning did not recur; do not claim it is a current failure or fixed |
| Optional diagnostics | Explicit settlement-version assertions | Existing Project Case real-adapter page/provenance tests fail under simulated registry/profile v2 drift. The screening disclosure assertion lacks a direct binding to those producer version constants; add a focused binding check if tightening that contract. Keep this optional and distinguish PC drift coverage from screening assertion coverage |
| Optional defensive API handling | Null adapter/profile mappings | Calling the raw display helper with explicit `None` mappings raises AttributeError. Validated RunResult paths reject these payloads earlier; a future raw-mapping API hardening change can clarify that interface without widening current model claims |
| Optional export hardening | Shared assumption labels | `Dispatch model` and `CapEx` are duplicated string literals in the global assumptions builder and cockpit export adapter. Centralize them like the capture label and test renamed/missing-row behavior so a future label change cannot leave an old sidebar row beside a corrected panel row. |
| Before any model generalization | Sequential/stochastic nonuniform durations | Their current valid routes remain scalar/uniform-grid. Any vector-dt extension must first reconcile reserve average power weighting and add native-duration known answers; do not silently reuse an arithmetic mean on mixed durations |
| Before changing settlement cash | DST convention migration | Project Case's registered DE_LU profile retains nominal blocks; screening retains physical hours. The 456/437/475 examples explain implemented conventions, not externally verified billing rules. Any cash unification needs a new versioned contract |

The [layout inventory](../audits/2026-09-20-step4-evidence/layout-inventory.json)
records source locations. Existing Revenue desktop `100% of power` clipping
was visible in [Step 3D evidence](../audits/2026-09-20-step3d-evidence/browser/revenue-1280.png).
Forward/Renewable/Data Trust inventory alone establishes no visual defect.

Other bounded input-review observations from Step 3C remain optional follow-ups:
arbitrary object-typed prices are not normalized by `calculate_average_price`,
and a naive mixed-cadence index can produce a less specific unavailable reason.
Normal ingestion supplies numeric prices and aware timestamps. Do not label
these alternate-input paths as failures observed in normal use.

The user owns external CC review invocation. No automatic reviewer invocation,
merge, ESS change or new model work is authorized by this backlog.
