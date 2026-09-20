# Work remaining after the September audit sequence

Baseline: `fb72dbf`, with [dated verification](current.md). Documentation
housekeeping reconciles status and evidence; it does not implement these items.
Historical findings remain in their original handoffs and patches.

| Priority | Work | Current evidence and required boundary |
|---|---|---|
| Next functional fix | Frontier result fingerprint | Its first/last-selected-day plus selected-day-count identity can miss a same-length price correction. Multi-day and forecast use full-content fingerprints; extend that protection in a separate behavior PR with stale/download/reversion and solver-call tests |
| Next visual pass | Remaining metric pages | Revenue Estimation has 48 metric call sites, Forward Scenarios 6, Renewable Correlation 4 and Data Trust 4. These are static call sites, including alternative branches, not simultaneously visible card counts. Audit actual loaded panels at 1280/1440/390px before expanding the Step 3C layout claim |
| Next export pass | Cockpit strategy names | Actual Step 3D XLSX rendering found long existing strategy names clipped in the width-30/no-wrap column. Full names remain in Assumptions. Test saved-file rendering with an appropriate wrap/width change; preserve numeric types/precision |
| Operator verification | Manual checks 1–47 | Existing browser evidence covers named synthetic panels and selected interactions. Live imports/fetches, every parameter combination and Radar→ESS consumer reconciliation have not all been run. Record each tested revision; do not mark the entire checklist passed |
| Dependency maintenance | Deprecation warnings | Classify the dated full-suite warnings before fixing source. Keep NumPy scalar-conversion and pandas concatenation cleanup in small behavior-preserving PRs, separate from this documentation change |
| Optional diagnostics | Explicit settlement-version assertions | The Step 3D reviewer suggested direct checks tying pinned display versions to the producer registry/profile. Existing real-adapter page/provenance tests already fail under simulated v2 drift; adding direct assertions improves failure locality, not a missing current safety boundary |
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
