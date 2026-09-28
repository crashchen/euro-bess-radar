# Work remaining after the September audit sequence

The [dated verification snapshot](current.md) records the Forward chart
follow-up candidate based on `87a9533`, the ordinary merge of #101. Step 4
housekeeping and follow-ups #94–#101 are merged. Historical snapshots remain
revision-bound: [#94](2026-09-22-frontier.md),
[#95](2026-09-22-floor-followup.md),
[#96](2026-09-22-cockpit-export.md),
[#97](2026-09-23-strategy-export.md),
[#98](2026-09-27-metric-layout.md),
[#99](2026-09-28-pre-forward-chart.md),
[#100](2026-09-28-forward-chart.md) and
[#101](2026-09-28-cockpit-export-copy.md).
Historical findings remain in their original handoffs and patches; a merge
does not extend a fixture-bound test to untested inputs.

| Priority | Work | Current evidence and required boundary |
|---|---|---|
| Implemented; awaiting review | Forward chart AppTest isolation and mobile terminal year | The [new candidate evidence](../audits/2026-09-28-forward-chart-followup-evidence/README.md) reproduces the leaked `file_uploader` stub and crowded final tick on clean merged #101, then fixes both without changing trace data. Its 390px browser pass covers 17/20/26/32-year static chart fixtures, not the complete Forward page |
| Operator verification | Manual checks 1–52 | Existing browser evidence covers named synthetic panels and selected interactions. Live imports/fetches, every parameter combination, remaining Revenue branches and Radar→ESS consumer reconciliation have not all been run. Record each tested revision; do not mark the entire checklist passed |
| Dependency maintenance | Observed warnings | The dated Step 4 run emitted Streamlit DataFrame-attrs serialization warnings and pandas concatenation FutureWarnings. Address reproducible sources in small behavior-preserving PRs. The historical NumPy scalar-conversion warning did not recur; do not claim it is a current failure or fixed |
| Optional diagnostics | Explicit settlement-version assertions | Existing Project Case real-adapter page/provenance tests fail under simulated registry/profile v2 drift. The screening disclosure assertion lacks a direct binding to those producer version constants; add a focused binding check if tightening that contract. Keep this optional and distinguish PC drift coverage from screening assertion coverage |
| Optional defensive API handling | Null adapter/profile mappings | Calling the raw display helper with explicit `None` mappings raises AttributeError. Validated RunResult paths reject these payloads earlier; a future raw-mapping API hardening change can clarify that interface without widening current model claims |
| Before any model generalization | Sequential/stochastic nonuniform durations | Their current valid routes remain scalar/uniform-grid. Any vector-dt extension must first reconcile reserve average power weighting and add native-duration known answers; do not silently reuse an arithmetic mean on mixed durations |
| Before changing settlement cash | DST convention migration | Project Case's registered DE_LU profile retains nominal blocks; screening retains physical hours. The 456/437/475 examples explain implemented conventions, not externally verified billing rules. Any cash unification needs a new versioned contract |

The [layout inventory](../audits/2026-09-20-step4-evidence/layout-inventory.json)
records source locations. The Revenue, Forward and Renewable clipping was
reproduced on the clean #97 base and corrected in #98's loaded-page browser
pass at 1280/1440/390px. Data Trust's loaded metric row showed no clipping and
was not changed. Conditional Revenue branches and arbitrary large values remain
outside that fixture-bound pass.

Other bounded input-review observations from Step 3C remain optional follow-ups:
arbitrary object-typed prices are not normalized by `calculate_average_price`,
and a naive mixed-cadence index can produce a less specific unavailable reason.
Normal ingestion supplies numeric prices and aware timestamps. Do not label
these alternate-input paths as failures observed in normal use.

The two #96 export-copy wording notes and the direct frontier helper's
missing-CapEx-row case were merged in #101. Normal app construction already
supplies the CapEx row; the direct-helper correction makes that provenance
consistent when a nonempty caller omits it.

The user owns external CC review invocation. No automatic reviewer invocation,
merge, ESS change or new model work is authorized by this backlog.
