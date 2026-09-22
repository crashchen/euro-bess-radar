# Work remaining after the September audit sequence

The [dated verification snapshot](current.md) records the current Frontier candidate;
Step 4 housekeeping is merged in #93 as `6d986e3`. The frontier follow-up below
is implemented on a separate review branch, with its own [handoff](../audits/2026-09-22-frontier-handoff.md).
Historical findings remain in their original handoffs and patches; pending
review must not be presented as merged or as covered by an earlier test run.

| Priority | Work | Current evidence and required boundary |
|---|---|---|
| Implemented; awaiting review | Frontier result identity and export snapshot | Full DA frame content, every selected date, live consumed constants and a panel version now enter the tuple identity. Successful Run saves the export assumptions. Changed inputs withhold results/downloads and floor context; reverting restores the frontier without solving, while the cleared floor needs its own Run. See the [review handoff](../audits/2026-09-22-frontier-handoff.md) for revision-bound evidence |
| Bounded export follow-up | Contracted-floor global assumptions | Numerical floor knobs and the selected merchant baseline are fingerprinted. Its export still assembles the saved result with the current global assumptions table on rerender. If snapshotting this metadata, preserve the floor calculation and its current stale-cache behavior; the frontier change does not fix it |
| Optional retry-state follow-up | Previous successful frontier after failed same-input retry | A caught `ValueError` from an explicit rerun shows an error for that render but does not replace the previous successful cache. A later same-input rerender can show that earlier result. Changed inputs remain blocked; this is a pre-existing retry-state behavior, not new successful computation or proof of stale data escaping the guard |
| Next visual pass | Remaining metric pages | Revenue Estimation has 48 metric call sites, Forward Scenarios 6, Renewable Correlation 4 and Data Trust 4. These are static call sites, including alternative branches, not simultaneously visible card counts. Audit actual loaded panels at 1280/1440/390px before expanding the Step 3C layout claim |
| Next export pass | Cockpit strategy names | Actual Step 3D XLSX rendering found long existing strategy names clipped in the width-30/no-wrap column. Full names remain in Assumptions. Test saved-file rendering with an appropriate wrap/width change; preserve numeric types/precision |
| Operator verification | Manual checks 1–47 | Existing browser evidence covers named synthetic panels and selected interactions. Live imports/fetches, every parameter combination and Radar→ESS consumer reconciliation have not all been run. Record each tested revision; do not mark the entire checklist passed |
| Dependency maintenance | Observed warnings | The dated Step 4 run emitted Streamlit DataFrame-attrs serialization warnings and pandas concatenation FutureWarnings. Address reproducible sources in small behavior-preserving PRs. The historical NumPy scalar-conversion warning did not recur; do not claim it is a current failure or fixed |
| Optional diagnostics | Explicit settlement-version assertions | Existing Project Case real-adapter page/provenance tests fail under simulated registry/profile v2 drift. The screening disclosure assertion lacks a direct binding to those producer version constants; add a focused binding check if tightening that contract. Keep this optional and distinguish PC drift coverage from screening assertion coverage |
| Optional defensive API handling | Null adapter/profile mappings | Calling the raw display helper with explicit `None` mappings raises AttributeError. Validated RunResult paths reject these payloads earlier; a future raw-mapping API hardening change can clarify that interface without widening current model claims |
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
