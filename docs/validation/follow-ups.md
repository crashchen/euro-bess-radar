# Work remaining after the September audit sequence

The [dated verification snapshot](current.md) records the reviewed metric-layout
head and its [#98](https://github.com/crashchen/euro-bess-radar/pull/98) merge
as `3ab98c5`. Step 4 housekeeping and the export/retry follow-ups #94–#97
are also merged. The #94 [snapshot](2026-09-22-frontier.md),
#95 [snapshot](2026-09-22-floor-followup.md) and
#96 [snapshot](2026-09-22-cockpit-export.md) and
#97 [snapshot](2026-09-23-strategy-export.md) and
#98 [candidate snapshot](2026-09-27-metric-layout.md) remain revision-bound.
Historical findings remain in their original handoffs and patches; a merge
does not extend a fixture-bound test to untested inputs.

| Priority | Work | Current evidence and required boundary |
|---|---|---|
| Visual follow-up | Forward benchmark chart year axis | The existing comparison chart supplies integer calendar years but lets Plotly choose numeric ticks, observed as fractional, comma-formatted labels such as `2,027.2` in the #98 browser review. Constrain ticks to integer years and verify the chart at 1280/1440/390px without changing comparison data or revenue calculations. The #98 metric-card pass did not address this chart |
| Operator verification | Manual checks 1–50 | Existing browser evidence covers named synthetic panels and selected interactions. Live imports/fetches, every parameter combination, remaining Revenue branches and Radar→ESS consumer reconciliation have not all been run. Record each tested revision; do not mark the entire checklist passed |
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

Two non-blocking #96 review notes remain for a future export-copy pass: the
multi-day DA+IDA1 dispatch label could say explicitly that the replay uses
realised IDA prices, and the forecast-policy label could distinguish its
perfect-foresight ceiling from its sequential comparison rows. Direct helpers
also differ when the incoming assumptions table lacks a CapEx row: multi-day
adds one, frontier leaves it absent. Normal app construction supplies the row.

The user owns external CC review invocation. No automatic reviewer invocation,
merge, ESS change or new model work is authorized by this backlog.
