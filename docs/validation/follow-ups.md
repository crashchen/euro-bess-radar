# Work remaining after the September audit sequence

The [dated verification snapshot](current.md) records main `68d3486`, the
ordinary merges of manual-acceptance follow-ups #104 (`3bbe9f0`), #106
(`3ba2f40`) and #105 (`68d3486`). Step 4 housekeeping and follow-ups #94–#106
are merged. The [2026-10-02/03 acceptance record](../audits/2026-10-02-manual-acceptance-evidence/README.md)
executed all 52 checklist items; its open findings are listed below. Historical
snapshots remain revision-bound: [#94](2026-09-22-frontier.md),
[#95](2026-09-22-floor-followup.md),
[#96](2026-09-22-cockpit-export.md),
[#97](2026-09-23-strategy-export.md),
[#98](2026-09-27-metric-layout.md),
[#99](2026-09-28-pre-forward-chart.md),
[#100](2026-09-28-forward-chart.md),
[#101](2026-09-28-cockpit-export-copy.md),
[#102](2026-09-28-forward-chart-followup.md) and its
[merged state](2026-09-28-forward-chart-followup-merged.md).
Historical findings remain in their original handoffs and patches; a merge
does not extend a fixture-bound test to untested inputs.

| Priority | Work | Current evidence and required boundary |
|---|---|---|
| Next Radar fix | Non-finite and empty-sample presentation (N3, N4) | With every price non-finite, the PDF row-based median prints `inf` while the workbook cell is empty. With no complete local day, spread cards show €0.00/MWh and negative hours 0.0 h. Both should show the established unavailable state with a reason; never print raw non-finite text or a substituted zero |
| Small layout fix | 390px multi-day replay card (F7) | `Avg Annualized` shows `EUR 131,976/MW…` at 390 px; all other loaded cards fit. Follow the #98 pattern (shorter value plus a visible unit caption) and re-check the width matrix |
| Re-review | Deadband input display (N1) and sparse-capacity row ordering (N2) | N1: typing above the deadband `number_input` maximum leaves the typed text visible while runs use the previous value (Streamlit native). N2: with sparse FCR rows the scalar-mean co-opt row exceeds the per-interval co-opt ceiling, a documented basis difference that reads as paradoxical. Decide whether either needs UI wording before changing behavior |
| ESS-owned | Radar handoff consumer (F5, F6) | ESS must stop double counting DA arbitrage through its grid-tariff fallback once a Radar stream is applied, and reject an asset-size mismatch instead of rescaling. Not part of this repository; Radar's producer contract is unchanged |
| Operator verification | Manual checks 1–52 on later revisions | The dated record certifies `e28ee9c`/`68d3486` with stated scope: synthetic fixtures, isolated caches, provider responses at run time and agent-driven browsers. Re-run affected items for later UI changes. Do not extend it to every parameter combination, arbitrary magnitudes or a desktop save dialog |
| Dependency maintenance | Observed warnings | The dated Step 4 run emitted Streamlit DataFrame-attrs serialization warnings and pandas concatenation FutureWarnings. Address reproducible sources in small behavior-preserving PRs. The historical NumPy scalar-conversion warning did not recur; do not claim it is a current failure or fixed |
| Optional diagnostics | Explicit settlement-version assertions | Existing Project Case real-adapter page/provenance tests fail under simulated registry/profile v2 drift. The screening disclosure assertion lacks a direct binding to those producer version constants; add a focused binding check if tightening that contract. Keep this optional and distinguish PC drift coverage from screening assertion coverage |
| Optional defensive API handling | Null adapter/profile mappings | Calling the raw display helper with explicit `None` mappings raises AttributeError. Validated RunResult paths reject these payloads earlier; a future raw-mapping API hardening change can clarify that interface without widening current model claims |
| Before any model generalization | Sequential/stochastic nonuniform durations | Their current valid routes remain scalar/uniform-grid. Any vector-dt extension must first reconcile reserve average power weighting and add native-duration known answers; do not silently reuse an arithmetic mean on mixed durations |
| Before changing settlement cash | DST convention migration | Project Case's registered DE_LU profile retains nominal blocks; screening retains physical hours. The 456/437/475 examples explain implemented conventions, not externally verified billing rules. Any cash unification needs a new versioned contract |

The [layout inventory](../audits/2026-09-20-step4-evidence/layout-inventory.json)
records source locations. The Revenue, Forward and Renewable clipping was
reproduced on the clean #97 base and corrected in #98's loaded-page browser
pass at 1280/1440/390px. Data Trust's loaded metric row showed no clipping at
those widths. The 2026-10-02 acceptance found it clipping at 960 px with the
sidebar open, and #105 moved it to the container-aware row. The 2026-10-03
matrix covered loaded Revenue branches at 1440/1280/960/390 with the fixture
values it records; arbitrary large values remain outside these passes.

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
