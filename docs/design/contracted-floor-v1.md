# Contracted-floor versus merchant cash-flow — v1 design contract

Status: **DRAFT - IN REVIEW.** This document locks the first productisation
layer only after review and merge. It is deliberately a cash-flow comparison
overlay, not a dispatch, trading, financing, or legal-contract engine.

## 1. Positioning

The dashboard can now estimate an annualised, degradation-net merchant outcome
from the cycle-cap frontier. That answers, "what can this asset earn if it
remains merchant?" It does not answer the adjacent investment question:

> Is a quoted contracted floor worth accepting instead of bearing all merchant
> cash-flow risk?

This v1 introduces a transparent **floor-protected merchant** comparison. The
counterparty is assumed to top up operating cash flow to an agreed annual
floor during a stated term, while the asset keeps merchant upside above that
floor. It is a screening abstraction for tolling, floor, and revenue-share
discussions; it is not a claim that any particular term sheet settles this
way.

The initial baseline is intentionally narrow: the selected best row from the
existing **DA-only cycle-cap x degradation frontier**. It is the only current
annual merchant figure with a co-temporal window, an explicit operating cap,
and an ex-post wear deduction. v1 therefore does **not** combine DA, IDA,
reserve, activation, or imbalance figures from different optimisation bases.

## 2. Locked economic convention

All annual values below are in EUR unless marked otherwise.

Inputs:

- `merchant_net_eur_per_mw_yr`: selected frontier row's annualised net
  merchant cash flow after the existing linear wear proxy.
- `power_mw`: sidebar power basis.
- `quoted_floor_eur_per_mw_yr`: contractual annual floor quoted per MW-year.
- `contract_availability`: a user-entered scalar in `[0, 1]`; it turns the
  quoted floor into the effective floor. Its default is **100%**. Do not
  silently borrow the reserve-capacity availability haircut: contract
  availability, performance deductions, and liquidated damages are different
  commercial terms.
- `floor_tenor_years`: positive contractual term in years.
- `discount_rate`: non-negative annual discount rate used only for a
  contract-window operating-cash-flow PV comparison.

Definitions:

```text
M = merchant_net_eur_per_mw_yr x power_mw
F = quoted_floor_eur_per_mw_yr x contract_availability x power_mw
P = max(M, F)                         # floor-protected annual cash flow
T = max(P - M, 0)                     # annual counterparty top-up
A(n, r) = annuity PV factor for n years at rate r

merchant_pv_eur          = M x A(floor_tenor_years, discount_rate)
floor_protected_pv_eur   = P x A(floor_tenor_years, discount_rate)
floor_pv_uplift_eur      = T x A(floor_tenor_years, discount_rate)
```

`P` is the primary comparison output. It is **not** `M + F`; the floor is a
minimum cash-flow level, so a top-up is paid only when `M < F`. The formula
remains well-defined for a negative merchant-net year: the top-up then covers
the full gap from that loss to the contractual floor.

`merchant_pv_eur` and `floor_protected_pv_eur` are **contract-window operating
cash-flow PVs**, not project NPVs. v1 deliberately does not subtract initial
CapEx, add debt, model taxes, or assume post-term merchant cash flows. This
prevents the current linear wear proxy from being silently mixed with a
financing model. The PV uplift is nevertheless useful for comparing quotes of
different term and availability. Within the stated tenor, v1 holds the selected
annual merchant-net baseline and effective floor constant in real terms: it
does not forecast escalation, inflation, or merchant-spread decay.

### 2.1 Input validation and units

- Floor rate and power must be non-negative; tenor must be positive; discount
  rate must be non-negative; availability must lie in `[0, 1]`.
- Merchant net remains signed. A bad merchant year is information, not a value
  to clamp before applying the floor.
- The quote basis is always **EUR/MW-year of installed power** in v1. It is
  not EUR/MWh, qualified reserve MW, calendar-year energy, or availability
  payment per settlement interval.
- The UI must display the effective annual floor `F` and both the quoted
  availability and the resulting availability-adjusted rate. A user must not
  have to infer that an availability haircut has been applied.

## 3. Baseline and provenance contract

The calculation consumes a current cycle-frontier result, not a free-form
annual number. The source baseline is the row marked as the frontier's best
cap and is labelled:

`DA-only merchant net after linear wear - cycle-frontier best cap`

The overlay carries this provenance verbatim, together with:

- the selected cap and realised average EFC/day;
- cycle life and CapEx inherited from the same frontier run;
- the frontier's common valid-day count and annualisation convention;
- `frontier_flat` and `n_tiebreak_fallback_days`, when present;
- quoted floor, availability, tenor, and discount rate.

The contracted-floor result is unavailable until a valid frontier result has
been run. Changing any frontier fingerprint input, contract input, zone, or
the loaded date window invalidates the derived result. It must never display a
floor comparison over a stale merchant baseline.

## 4. Computation API (increment CF-A)

New module `src/contracted_floor.py`:

```python
def compute_contracted_floor_overlay(
    *,
    merchant_net_eur_per_mw_yr: float,
    power_mw: float,
    quoted_floor_eur_per_mw_yr: float,
    floor_tenor_years: float,
    contract_availability: float = 1.0,
    discount_rate: float = 0.08,
) -> dict[str, float]:
```

Returned fields include:

- `merchant_net_eur`, `merchant_net_eur_per_mw_yr`;
- `quoted_floor_eur`, `effective_floor_eur`,
  `effective_floor_eur_per_mw_yr`;
- `floor_protected_cashflow_eur`, `annual_top_up_eur`;
- `merchant_pv_eur`, `floor_protected_pv_eur`, `floor_pv_uplift_eur`;
- `floor_tenor_years`, `discount_rate`, `contract_availability`.

The annuity factor must be promoted to a small public helper in `scenario.py`
or another single finance utility. Do not duplicate the fractional-year PV
math currently used by `calculate_npv_distribution`; retaining identical
discounting is part of the contract.

## 5. Cockpit and export placement (increment CF-B)

Add one expander immediately after the cycle-cap frontier section:

**"Contracted floor versus merchant cash flow"**

Inputs are:

- quoted floor (EUR/MW/year);
- contracted availability (%);
- floor term (years);
- discount rate (%).

The panel inherits power, duration, CapEx, cycle-life, date window, and best
cap from the current frontier result. It deliberately has no independent BESS
or merchant-revenue selector.

Outputs are:

- annual merchant net, effective contracted floor, and floor-protected annual
  cash flow;
- annual top-up (zero when the merchant baseline already clears the floor);
- merchant versus floor-protected **contract-window PV**, plus PV uplift;
- one compact merchant/floor/protected grouped bar chart and a table of the
  formula inputs;
- a hard caption: "Screening floor overlay, not a binding contract model; DA
  only; linear wear proxy; no credit, performance, tax, financing, or
  post-term merchant assumption."

Excel export contains a `Contracted floor` table and assumption rows for every
input and source field in section 3. `build_assumptions_table` is unchanged:
the contract's commercial inputs belong to this panel's self-contained export,
not the global model-audit table.

## 6. Red lines and non-goals

- **Not additive.** Never show floor revenue as a new merchant strategy row or
  add it to DA/IDA/reserve/activation/imbalance totals. It changes annual cash
  flow only through `max(M, F)`.
- **No dispatch effect.** The floor cannot alter the cycle cap, daily MILP,
  stochastic policy, reserve commitment, SoC path, or any historical replay.
- **No cross-stream stack.** v1 consumes the DA-only frontier best row only.
  A future unified merchant cash-flow basis must be designed before it can be
  used here.
- **No legal or credit model.** No counterparty default, collateral, indexation,
  inflation, merchant revenue share, minimum-volume condition, availability
  test, liquidated damages, tax, debt sculpting, or termination rights.
- **No post-tenor extrapolation.** The displayed PV covers the floor tenor
  only. Calling it a project NPV is prohibited.
- **No automatic recommendation.** A positive PV uplift is not sufficient to
  accept a contract; it is a screening input for commercial and risk review.

## 7. Pinned identities and tests

Before UI work, CF-A tests must pin:

1. `quoted_floor = 0` and `contract_availability = 0` both reproduce merchant
   cash flow and zero top-up exactly.
2. A floor strictly below merchant net produces `P = M` and zero top-up.
3. A floor strictly above merchant net produces `P = F` and `T = F - M`.
4. A negative merchant-net fixture is not clamped; the top-up reaches the
   effective floor and remains non-negative.
5. Scaling power by `k` scales all EUR outputs by `k`, while per-MW values and
   percentage availability are unchanged.
6. `floor_protected_pv - merchant_pv == floor_pv_uplift` within floating-point
   tolerance, including a fractional tenor and zero discount rate.
7. Invalid floor, availability, tenor, and discount inputs raise early with
   clear messages.
8. The core function does not mutate the merchant baseline or invoke any
   dispatch/solver function.

CF-B AppTest pins must cover: frontier-result gating; exact source caption;
fingerprint invalidation after a contract or frontier input changes; no
contract row in the strategy revenue chart; and Excel assumption provenance.

## 8. Increment plan after lock

- **CF-A:** pure calculation module + public annuity-factor reuse + tests in
  `tests/test_contracted_floor.py`.
- **CF-B:** cockpit expander, state fingerprint, chart, Excel export, and
  AppTest smoke.

Each increment receives the usual implementation review plus an independent
commercial-semantics pass. The design PR itself must be reviewed before either
increment begins.

## 9. Open questions intentionally deferred

These are not v1 implementation decisions. They are reasons to create a v2
contract design rather than silently expanding this one:

1. Pure tolling/fixed rent versus a floor-protected merchant revenue share.
2. Floor settlement against gross revenue, EBITDA, or another defined cash
   waterfall rather than v1's merchant net after linear wear.
3. Availability test mechanics, performance penalties, and qualified-MW
   definitions.
4. Inflation/indexation, degradation-driven augmentation, and post-term
   merchant cash flows in a true project-NPV model.
5. Counterparty credit, collateral, curtailment, taxes, debt, and financing
   covenants.
6. A coherent multi-stream merchant baseline that can include IDA, reserve,
   activation, or imbalance without double counting or violating their
   existing red lines.
