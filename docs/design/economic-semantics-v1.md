# Economic semantics v1 — revenue, shadow wear, and cash NPV

Status: **accepted and implemented**

Decision date: 2026-08-06

## Decision

The platform keeps three economically different layers separate:

1. **Gross dispatch revenue** is the market revenue produced by the dispatch
   model before a battery-wear proxy. It answers what the modeled strategy can
   capture from the selected revenue streams.
2. **Economic margin after shadow wear** subtracts the linear throughput proxy
   `installed_capex / cycle_life` per FEC. It is a non-cash opportunity cost for
   cycle-cap and strategy screening. It may rank operating policies, but it is
   not an accounting payment.
3. **Cash NPV** deducts upfront CapEx and explicit cash flows only. The shadow
   wear proxy must not be deducted again. Augmentation, module replacement,
   residual value, tax, and financing enter cash NPV only when their timing and
   amount are explicitly modeled.

This avoids using the same installed CapEx once as the initial investment and
again as a throughput-proportional cash outflow.

## Current implementation

- Revenue Estimation shows gross annual revenue and simple payback separately.
- The former degradation/net-revenue cards are explicitly labeled **Shadow
  Wear**, **Economic Margin**, and **Economic Payback Proxy**.
- Monte Carlo project NPV and its sensitivity table pass no shadow-wear cash
  deduction. The UI discloses that augmentation/replacement cash flows are not
  yet modeled.
- Exports retain historical machine keys for compatibility, but add explicit
  semantic keys (`annual_shadow_wear_cost_eur`,
  `economic_margin_after_wear_eur`, `cash_npv_includes_shadow_wear=False`) and
  use unambiguous human-readable labels.
- The current charge-plus-discharge denominator is labeled **Two-leg
  Throughput Cost**, not conventional discharge-output LCOS.

## Red lines

- Do not pass `calculate_degradation_cost()` output into project cash NPV.
- Do not describe shadow-wear-adjusted margin as bank-account cash flow.
- Do not add an augmentation reserve without an explicit schedule, cost basis,
  capacity-restoration effect, and residual-value treatment.
- The cycle-cap frontier remains allowed to use the linear wear proxy because
  its purpose is relative operating-policy screening.

## Next financial-model increment

If bankability analysis is added, implement an explicit augmentation/replacement
cash-flow schedule and report both the no-augmentation screening NPV and the
fully specified cash NPV. That increment belongs after the project-case schema
defines ownership of engineering and finance assumptions.
