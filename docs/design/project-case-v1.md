# Project Case v1 — unified pre-tax unlevered lifecycle cash NPV

Status: **candidate** (design-contract round — NOT locked. Revised after Codex
artifact review rounds 1–3 and a Gemini adversarial review (all `CHANGES
REQUESTED`); pending a re-review of the round-3 revisions before `Status:
locked`.)

Decision date: 2026-08-09

Extends: [`economic-semantics-v1.md`](./economic-semantics-v1.md) — its "Next
financial-model increment" section defers augmentation/replacement cash flows and
a fuller NPV until "the project-case schema defines ownership of engineering and
finance assumptions". This contract is that schema.

Depends on / must not violate: `economic-semantics-v1.md`,
`cycle-cap-frontier-v1.md`, `contracted-floor-v1.md`,
`contracted-floor-decay-v2.md`, `dispatch-failure-contract-v2.md`,
`spread-decay-v1.md`.

---

## 1. Purpose

The platform computes many revenue/cost signals — DA arbitrage, IDA rebid,
reserve-capacity co-optimisation, activation energy, imbalance settlement, a
contracted floor, a cycle-cap wear frontier, a liquidity cap, a spread-decay
trajectory, an external-trader benchmark. Each is standalone and each is
deliberately guarded by a **non-additivity red-line**. There is no single object
that answers the investor question:

> For *this* asset (power, duration, CapEx, life, degradation, augmentation),
> operating *this* chosen strategy in *this* zone, what is the project-level
> lifetime cash flow and its NPV — and can I audit every number?

`ProjectCase` is that object. Its entire reason to exist is to define the
**legal composition** of streams into one lifecycle cash flow so unifying them
cannot silently break a red-line. The naive failure mode — "sum every stream" —
violates all of: reserve capacity is not additive with DA; activation/imbalance
overlays are non-additive; the floor is `max(M, F)` not `M + F`; and shadow wear
must never enter cash NPV. This contract makes the composition **explicit,
typed, producer-issued, and audited** — not broad.

## 2. Scope

**v1 locked intent:** minimal lifecycle cash-NPV unblock. Define ownership of
engineering/finance assumptions and an explicit **dated** augmentation/
replacement/residual schedule. Report a `No-lifecycle-cost screening NPV` and a `Pre-tax
unlevered lifecycle cash NPV`, **both as distributions** (§3). Do **not** call
the result "bankable" and do **not** compose the full revenue stack.

In scope: the typed case schema (§4); a producer-issued typed
`StrategyRunResult` as the only revenue input (§4.3, §5); a projection-mode
selector that respects `spread-decay-v1` (§4.7); the dated lifecycle schedule
(§4.2); year-by-year cash-flow construction and the two NPV distributions (§6);
an immutable, fingerprinted `RunResult` with full provenance and audit (§4.6).

Out of scope (explicit non-goals): full revenue-stack composition; financing/
tax/DSCR/gearing (v1 is pre-tax unlevered by construction); endogenous physical
fade → augmentation triggering; any new dispatch/MILP formulation (ProjectCase
consumes existing solver outputs, never re-solves).

## 3. Output naming (two NPV **distributions**)

v1 reports two figures, each a Monte-Carlo **distribution** (P10 / P50 / P90 /
`P(NPV>0)`), not a single ambiguous value. Both are built by bootstrapping the
chosen strategy's **daily realised cash series** (§4.3) with
`scenario.bootstrap_annual_revenue`, and **both use the same
`LifecycleCase.project_life_years` horizon and the same per-year projection
multiplier** (§4.7) so the only difference between them is the lifecycle-cost
layer (§6):

1. **`No-lifecycle-cost screening NPV`** — the bootstrapped revenue discounted
   year-by-year with the projection multiplier, minus upfront CapEx over the
   project horizon, with the shadow-wear cash deduction held at zero
   (`cash_npv_includes_shadow_wear=False`, per economic-semantics-v1) and **no**
   dated lifecycle events. Retained for continuity.
2. **`Pre-tax unlevered lifecycle cash NPV`** — the same bootstrapped revenue
   over the same horizon and multiplier, minus upfront CapEx, minus cash fixed
   O&M, minus the **explicit dated** augmentation/replacement schedule, plus
   residual value, all discounted year-by-year. No shadow wear. No tax. No debt.

**No general "exact parity" claim.** The shipped Revenue-tab NPV is built on
DA daily revenue, the sidebar capture haircut, and a *cycle-derived, fractional*
effective life — whereas `project_life_years` is a positive **integer** (§4.2),
so a full ProjectCase run cannot reproduce the tab in general. The only parity
assertion v1 makes is **kernel-level** (Gemini review): the DA-only adapter's
annual-revenue distribution, passed through the **same** underlying
`calculate_npv_distribution` primitive with the tab's own inputs (capture
haircut and the tab's *fractional* effective life), reproduces the shipped
screening NPV. That test exercises the shipped kernel — which internally uses the
aggregated `decaying_annuity_pv_factor` — **not** a full `ProjectCase` (whose
integer-life headline discounts year-by-year, §6). The two coincide for a flat or
geometric projection over an integer life, but ProjectCase never *depends* on the
annuity closed form (it would be wrong for a non-geometric multiplier curve or
lumpy events). Every other configuration is a new figure, not a reproduction.

We deliberately do **not** call figure 2 "bankable NPV." Without tax, debt
service, DSCR, and financing-fee models it would over-promise. economic-
semantics-v1 requires only that, once augmentation is explicit, both a
no-lifecycle-cost screening NPV and a fully-specified cash NPV are reported.

## 4. Schema

Inputs are small typed "cases" grouped by ownership, aggregated by a lightweight
`ProjectCase`; all computed values + provenance live in a separate **immutable,
fingerprinted** `RunResult`.

```
ProjectCase
├── AssetCase          (engineering nameplate + cash fixed O&M)
├── LifecycleCase      (life, dated augmentation/replacement, residual/disposal)
├── MarketCase         (ref to a StrategyRunResult + projection mode)
└── ValuationCase      (discounting; real, base-year EUR, unlevered)
# ContractCase is NOT part of the v1 aggregator — floor is an external
# comparator only (§4.5). It joins the aggregator in v1.1 with a gross basis.

StrategyRunResult      (producer-issued; the ONLY cash-eligible revenue input; §4.3)

RunResult (immutable, fingerprinted, schema-versioned)
├── cashflow table     (year → revenue draw stats, opex, augmentation, residual, net, PV)
├── no_lifecycle_cost_screening_npv  {p10,p50,p90,prob_positive}
├── lifecycle_cash_npv          {p10,p50,p90,prob_positive}
├── floor_comparator?           (external, never summed into the NPVs)
└── provenance         (per-case sources + solver audit + red-line assertions held)
```

### 4.1 `AssetCase` — engineering nameplate + cash O&M

Fields (finite, validated): `power_mw`, `duration_hours` (⇒ derived
`energy_mwh`), `round_trip_efficiency`, `installed_capex_eur` (or
`capex_eur_per_kwh` × capacity — one source of truth), `fixed_om_eur_per_mw_yr`
(the **only** cash OpEx in v1).

**No `variable_om` field.** The dispatch solver already nets VOM internally
(`dispatch.DISPATCH_VOM_COST_EUR_MWH = 0.5` EUR/MWh; `dispatch.py:97-98` prices
charge at `price+VOM` and discharge at `price−VOM`), so an AssetCase VOM knob
would be a no-op that cannot change solver revenue — a misleading input.
The embedded VOM value is instead recorded in the StrategyRunResult provenance
(§4.3) and is **not** re-deducted in §6 (red-line #7).

CapEx must equal the sidebar single source of truth that the frontier and
Revenue tab already inherit — no third CapEx knob.

### 4.2 `LifecycleCase` — life, augmentation, residual/disposal

Fields: `project_life_years` (a **positive integer**); a
`capacity_maintenance_basis` (three-state, below); an **augmentation/replacement
schedule** = an ordered list of dated events, each `{year, cost_eur,
capacity_restored_frac, residual_value_eur}`; an end-of-life `residual_value_eur`;
and `decommissioning_cost_eur`. LifecycleCase is the **sole owner** of residual,
disposal, and decommissioning; no other case models them.

Domains pinned (`validate()`, §4.6): every `cost_eur`, `residual_value_eur`,
`decommissioning_cost_eur`, `fixed_om_eur_per_mw_yr`, and `installed_capex_eur`
is **finite and ≥ 0** (a cost/residual is never negative or NaN); event `year` is
an **integer in `1…project_life_years`**; `capacity_restored_frac ∈ [0, 1]`. A
NaN/Inf/negative in any of these is rejected — it must never reach the §6 sum.

Time/units pinned: cash timing is **end-of-year** (§4.4); all CapEx / O&M /
event / residual / decommissioning amounts are the same **base-year real EUR** as
the ValuationCase. **Residual
timing (Gemini review):** an augmentation/replacement **event's**
`residual_value_eur` is the salvage of the equipment that event *replaces*,
realised as a cash **inflow in the event year** (netted against that event's
`cost_eur`). The distinct LifecycleCase-level **end-of-life**
`residual_value_eur` and `decommissioning_cost_eur` are realised **only in year
`project_life_years`**. The two are never conflated. `capacity_restored_frac` is a fraction of **nameplate
energy** (0–1), never MWh — one unit. In v1 `capacity_restored_frac` is
**descriptive provenance** for the maintenance assertion (below); it does not
drive a revenue multiplier (that is the v1.1 option in §10).

**Augmentation economic effect — explicit three-state (closes the
economic-semantics red-line; applies to ALL projection modes, not just Flat).**
economic-semantics-v1 forbids an augmentation reserve with no capacity-
restoration effect, and a flat/decayed revenue projection is only honest if the
asset is actually kept at the capacity that revenue assumes. So every
LifecycleCase must declare **why** the revenue projection's capacity assumption
holds, via `capacity_maintenance_basis`:

- **`SCHEDULED_NAMEPLATE_MAINTENANCE`** — a non-empty augmentation/replacement
  schedule asserted (with engineering **source + as-of**) to sustain the
  projected capacity over the life. The schedule's costs enter cash NPV (§6).
- **`NO_AUGMENTATION_REQUIRED_ASSERTED`** — the user explicitly asserts (with
  engineering **source + as-of**) that no augmentation is needed for the
  projected capacity over the life. No augmentation cost; the assertion is
  recorded, not a silent default.
- **`UNKNOWN`** — neither asserted. The **lifecycle cash NPV is unavailable**
  (fail-closed, `dispatch-failure-contract-v2` spirit); the screening NPV (§3.1)
  may still render. This is a first-class state, never a warning-and-proceed.

The first two states **require** an engineering source + as-of; a bare flag is
rejected at `validate()`. An augmentation event missing any of year / cost /
restoration / residual is likewise rejected. The richer alternative — an
explicit per-year capacity/revenue multiplier tied to a fade curve — is a v1.1
option (§10), not built in v1.

### 4.3 `MarketCase` and the producer-issued `StrategyRunResult`

`MarketCase` references **one** `StrategyRunResult` plus a projection mode
(§4.7). It does **not** carry a free-floating revenue scalar.

`StrategyRunResult` is a typed object **emitted only by a solver adapter**, never
constructed by the UI or by attaching a label to a number. Eligibility is a
property of *which producer emits it*, so an ineligible quantity (a ceiling, a
delta, an overlay) is **unrepresentable**, not merely rejected. Its identity is a
public **`StrategyKind` enum** (`DA_ONLY`, `DA_ID_FORECAST`, `DA_RESERVE_COOPT`,
`DA_ID_RESERVE_REALISED`) — a stable public API, **not** the private display
labels (`_DA_ONLY`, `_FORECAST`, …) of `strategy_compare.py`, which may change.
It carries:

- `strategy_kind` (the enum) + human label;
- `daily_realised_cash_series` — an **immutable tuple of `(date, value_eur)`**
  pairs (not a mutable pandas `Series`), per local calendar day, in **total EUR
  for the modelled MW** (already MW-bound — never re-scaled downstream; §6),
  **gross** dispatch cash (economic-semantics layer 1: market revenue before
  shadow wear). This series — not a scalar — feeds the bootstrap (§3, §6);
- `cash_basis` — an explicit statement that the series is **post-VOM** and
  whether the **capture haircut** and **liquidity cap** are already applied
  (so PC never re-applies either);
- engineering binding: `power_mw`, `duration_hours`, `round_trip_efficiency`;
- market binding: `zone`, `sample_window`; and kind-specific provenance:
  `forecast_mode` (walk-forward for forecast-driven — §5) + forecast
  bucket/deadband; `reserve_product` + `reserve_source` + `availability`;
  stochastic `seed` + `scenario_count` + `tie_break_fallback` (where applicable);
- a **`CoverageAudit`** (per `dispatch-failure-contract-v2`): observed / valid /
  missing / solver-failed day counts **plus per-day failure details**; an
  all-failure or empty-valid result is *unavailable*, never a €0 series;
- reproducibility: `source_data_content_hash`, `calculator_version`, and a
  content `fingerprint` over all of the above.

`validate()` (§4.6) checks `StrategyRunResult.power_mw == AssetCase.power_mw`
(and duration/RTE), so the revenue and the asset describe the same battery.
The revenue is **gross** — this is what makes subtracting CapEx and augmentation
in cash NPV legal without double-counting: market revenue on the cash line,
capital counted once as capital.

### 4.4 `ValuationCase` — discounting (real, unlevered)

Fields: `discount_rate` interpreted as a **real** rate — validated finite and
`> -1` (typical screening range `[0, 1)`; a rate `≤ -1` breaks the discount
denominator); `base_year` fixing the EUR value basis. **Discounting (Codex review
3):** **both** ProjectCase NPVs discount each year's net cash flow **explicitly
year-by-year** (§6) using the per-year projection multiplier (§4.7) — the
screening NPV simply omits the lifecycle events. This is uniform across all three
projection modes, including a non-geometric `ExplicitAnnualMultiplierCurve` and
the lumpy lifecycle stream, neither of which the aggregated
`scenario.decaying_annuity_pv_factor` scalar can represent. The aggregated factor
is therefore used in **exactly one place** — the shipped-tab **parity kernel**
(§3), which reproduces the shipped screening NPV. The per-year multiplier for
`DAOnlySpreadDecay` is the documented spread-decay weight
`max((1−d)^(t−1), floor)` from a small explicit weight-series helper (the
aggregated factor exposes no per-year weights), never obtained by dividing the
scalar factor.
**Convention lock:** revenue projection (§4.7) and discounting are both in
**real, base-year EUR** — decay and discount must not mix a nominal and a real
convention. Cash timing follows an explicit **end-of-year** convention anchored
at a commercial-operation-date (COD) / valuation date; year 0 =
`−installed_capex_eur`.

Explicit non-fields (asserted absent, recorded as provenance facts): tax rate,
debt fraction, cost of debt, DSCR target, financing fees.

### 4.5 `ContractCase` — floor (v1: external comparator, NOT in the aggregator)

The shipped `contracted_floor` outputs consume a **wear-net** merchant baseline
(`contracted_floor.py`: "Compare wear-net merchant cash flow…"; returns
`floor_protected_cashflow_eur = max(merchant_eur, effective_floor_eur)`), and
`contracted-floor-v1.md` states their PV is explicitly **not** a project NPV.
Feeding that into cash NPV would re-import the linear shadow-wear deduction
through the `max(M, F)` top-up — re-creating the exact CapEx-vs-wear double-count
economic-semantics-v1 removed.

Therefore in v1 `ContractCase` is **not** part of the `ProjectCase` aggregator.
The floor is rendered as a separate `floor_comparator` beside the NPV
distributions and is **never** summed into them. ProjectCase must not read
`floor_protected_cashflow_eur` / `floor_protected_pv_eur`.

v1.1 hook (not built): a `ContractCase.settlement_basis` recomputing
`max(M_t, F_t)` per year from a **gross** cash merchant base, which would then be
cash-NPV-eligible and rejoin the aggregator.

### 4.6 `ProjectCase` aggregator + `RunResult`

`ProjectCase` holds the cases and a `validate()` enforcing: engineering match
(§4.3); `strategy_kind` (§4.3) in the §5 allowlist and its StrategyRunResult
solver-available; projection mode compatible with the strategy (§4.7); the §4.2
lifecycle domains; and CapEx matches the sidebar. **Series/audit invariant (Codex
review 3):** the `daily_realised_cash_series` must be **non-empty**, every value
**finite**, dates **unique**, and the series dates **exactly equal** the audited
valid days — so an "available" result can never be silently thinned into the
prohibited €0 fallback that `bootstrap_annual_revenue` would produce if all
values were non-finite (`scenario.py:128`). **Fail-closed:** if the referenced
strategy result is invalid/unavailable, or the series/audit invariant fails,
`validate()` raises — no silent fallback to a lesser strategy or to €0
(`dispatch-failure-contract-v2`).

`RunResult` is immutable, carries a `schema_version`, is serialisable, and holds
an `input_fingerprint` over every case **plus the bootstrap `seed`, the
`n_simulations`, and the `calculator_version`** (so a stale result is detectable
and cache invalidation is deterministic — the frontier/floor fingerprint
pattern). It is the only object UI and export read from.

### 4.7 Projection modes (respecting `spread-decay-v1`)

`spread-decay-v1.md` (§lines 136–140) locks the decay to "the Revenue tab's
DA-arbitrage-based Monte-Carlo draws … no interaction with the cockpit strategy
rows." Applying one decay to a DA+reserve or DA+IDA+reserve total would violate
that locked contract. v1 therefore exposes three explicit modes:

**Single-stream rule (Gemini review):** any non-flat projection applies **one**
multiplier to the whole annual draw. For a multi-stream total
(`DA_RESERVE_COOPT`, `DA_ID_RESERVE_REALISED`) that would decay DA arbitrage and
reserve/IDA revenue *identically*, contradicting the very reason
`spread-decay-v1` (§9.5) restricts decay to DA (the streams cannibalise on
different physics). Therefore in v1 **any non-flat projection is permitted only
when `strategy_kind == DA_ONLY`**; every multi-stream strategy is
`FlatRealProjection` only. `validate()` enforces this.

- **`FlatRealProjection`** — a constant multiplier `1.0` for every year;
  available to **every** eligible strategy. (Capacity-maintenance rules of §4.2
  apply.) The only mode allowed for multi-stream strategies in v1.
- **`DAOnlySpreadDecay`** — the `spread-decay-v1` decayed **weight** (year 1 =
  `1.0`); `DA_ONLY` only. `validate()` rejects it otherwise.
- **`ExplicitAnnualMultiplierCurve`** — a user-supplied per-year **multiplier**
  vector, `DA_ONLY` only, constrained to **year 1 = `1.0`**, every entry
  **finite and ≥ 0**, and covering **exactly** `project_life_years` entries, and
  carrying a **machine-checkable `source` + `as_of`** (same strictness as the
  §4.2 capacity assertion; a bare float array, a wrong-length vector, or a
  NaN/negative entry is rejected — it would destroy `RunResult` auditability or
  break §6). It scales the annual draw exactly like the other modes (§6). A per-stream decay split, and an
  **absolute per-year EUR** revenue curve (with its own uncertainty), are
  deferred — v1's three modes are all "one multiplier × bootstrap draw".

## 5. Cashflow-eligible strategies (producer allowlist / denylist)

Eligibility is enforced by *which adapter emits a StrategyRunResult*, not by a
consumer-attached label. The strategy-comparison table (`strategy_compare.py`)
mixes totals, ceilings, and deltas in one frame, so it is never a ProjectCase
input.

**Adapters emit `StrategyRunResult`** (with the public `StrategyKind`, §4.3) for
these realised, internally-co-optimised gross totals only:

- `DA_ONLY` — DA-only realised.
- `DA_ID_FORECAST` — DA + IDA1 forecast-driven realised — **walk-forward only**.
- `DA_RESERVE_COOPT` — DA + reserve-capacity co-opt total
  (`solve_joint_capacity_batch` `joint_total_revenue`); the reserve fee is already
  inside this total via the joint MILP, so it is self-consistent.
- `DA_ID_RESERVE_REALISED` — DA + IDA1 + reserve forecast-driven **realised**
  total: the 9.2b reserve-first sequential heuristic
  (`simulate_sequential_da_id_reserve_batch` / `_REALISTIC_DEFAULT`), **not** the
  stochastic batch — **walk-forward only**. (`simulate_stochastic_da_id_batch`
  *does* accept reserve MW/prices, `simulation.py:1875-1876`, so its realised
  totals can include reserve capacity; it is nonetheless ineligible because its
  cockpit-surfaced product is a **policy-value delta**, not a realised total —
  eligibility is by producer/`StrategyKind`, §4.3, not by stream coverage.)

**No adapter exists** (hence unrepresentable as cash revenue) for: any
perfect-foresight ceiling (the `_CEILING` / `_TRIPLE_DEFAULT` / `coopt_ceiling*`
rows); the stochastic policy-value **delta** (`STOCHASTIC_POLICY_VALUE_LABEL` /
`…_RESERVE_LABEL`); activation/imbalance overlays; `gross_additive_total_eur`;
the external-trader benchmark; or any solver-unavailable / no-valid-day result.

**Forecast-mode rule:** forecast-driven strategies are cash-eligible **only in
`walk_forward`** mode. `in_sample` is diagnostics-only; `loo` (leave-one-out) is
an unbiased *skill* estimate that peeks at future days and is therefore too
optimistic for a cash figure — v1 rejects both for the cash NPV (walk-forward
only; resolved §10.6).

## 6. Lifecycle cash-flow construction

The chosen strategy's `daily_realised_cash_series` is **already total EUR for the
modelled MW** (§4.3) — it is **not** re-scaled by `power_mw` here (doing so would
double-count MW). It is bootstrapped to an annual-revenue distribution via
`scenario.bootstrap_annual_revenue`. For draw `d` and year
`t = 1 … project_life_years`:

```
revenue_{d,t} = annual_revenue_draw_d × projection_multiplier_t   # §4.7 (year 1 = 1.0)
opex_t        = fixed_om_eur_per_mw_yr × power_mw                 # VOM already inside gross revenue
augment_t     = Σ (event.cost_eur − event.residual_value_eur) for events in year t   # net capital; event salvage is an inflow in the event year (§4.2)
terminal_t    = (eol_residual_value_eur − decommissioning_cost_eur) if t == project_life_years else 0
net_{d,t}     = revenue_{d,t} − opex_t − augment_t + terminal_t
```

`lifecycle_cash_npv` for draw `d` is the **explicit year-by-year** sum
`−installed_capex_eur + Σ_{t=1..L} net_{d,t} / (1+discount_rate)^t`; over all
draws this yields the P10/P50/P90/`P(NPV>0)` distribution.
`no_lifecycle_cost_screening_npv` is the **same** year-by-year sum with
`augment_t = terminal_t = 0` (no dated events), so the two figures differ only by
the lifecycle-cost layer (§3) **by construction** — same bootstrap draws, same
`projection_multiplier_t`, same horizon, same discounting. Neither uses the
aggregated annuity factor (it cannot express a non-geometric multiplier curve or
lumpy events, §4.4); that factor lives only in the parity kernel (§3). Shadow
wear appears **nowhere**; VOM is not re-deducted; the series is not re-scaled by
MW. All cash amounts are base-year real EUR (§4.4).

## 7. Red lines

1. **No shadow wear in cash NPV** — never `calculate_degradation_cost()` on the
   cash line, and never via the floor back door (§4.5).
2. **Not bankable** — pre-tax unlevered; the label and UI copy say so.
3. **No full-stack composition** — one producer-issued strategy result feeds the
   cash line; overlays/comparators stay non-additive.
4. **Capacity maintenance declared** — `capacity_maintenance_basis` is one of the
   three states (§4.2); the two active states require an engineering source +
   as-of; `UNKNOWN` makes the lifecycle NPV **unavailable**, never a warning-and-
   proceed. Augmentation events need year + cost + restoration (frac of nameplate)
   + residual, or are rejected.
5. **Floor comparator-only** — not in the aggregator; ProjectCase never reads
   `floor_protected_cashflow_eur` / `_pv_eur`.
6. **Gross, producer-issued revenue** — eligibility is a `StrategyKind`/adapter
   property; ceilings, deltas, overlays, gross-additive references, benchmarks,
   and unavailable results are unrepresentable as cash revenue.
7. **VOM counted once** — embedded in the solver (0.5 EUR/MWh); recorded in
   provenance; never re-deducted in §6. No AssetCase VOM knob.
8. **MW counted once** — the strategy series is already total EUR for the modelled
   MW; §6 never re-scales it by `power_mw`.
9. **Decay stays in its lane** — **any** non-flat projection
   (`DAOnlySpreadDecay` or `ExplicitAnnualMultiplierCurve`) is `DA_ONLY` only, per
   `spread-decay-v1`; multi-stream strategies are `FlatRealProjection` only. One
   multiplier on a composite total conflates stream-specific cannibalisation.
10. **Forecast = walk-forward** for any cash-eligible forecast-driven strategy;
    `loo` and `in_sample` are ineligible for the cash NPV.
11. **Fail-closed** — an invalid/unavailable strategy, or a series that is empty /
    non-finite / not exactly the audited valid days, raises; no silent fallback,
    no €0 substitution (never the empty-bootstrap zero, §4.6).
12. **One real, base-year-EUR convention** — all CapEx / O&M / events / residual /
    decommissioning and the discount rate (finite, `> -1`) share the same
    base-year real EUR; the daily series is an immutable `(date, value)` tuple,
    never a mutable Series.
13. **CapEx once** — single sidebar source.
14. **Both NPVs discounted year-by-year** — neither uses the aggregated annuity
    factor (it cannot express a non-geometric multiplier or lumpy events); that
    factor lives only in the parity kernel (§3). The two NPVs differ only by the
    lifecycle-cost layer, by construction (§6). Event salvage is an inflow in the
    event year; terminal residual/decommissioning only in year `L`.
15. **All numeric inputs bounded** — event costs/residuals/decommissioning/O&M/
    CapEx finite ≥ 0, multipliers finite ≥ 0 with year 1 = 1.0 and full-length,
    `capacity_restored_frac ∈ [0,1]`; a NaN/Inf/out-of-range input is rejected
    before §6 (§4.2, §4.7).

## 8. Rejected alternatives

- **Sum the whole revenue stack now.** Breaks non-additivity without a
  per-stream double-count proof. Deferred.
- **A loose `year_1_annual_revenue` scalar as the revenue input.** Cannot
  reproduce the Monte-Carlo NPV and lets any labelled number in — replaced by the
  producer-issued daily-series `StrategyRunResult` (Codex review 1).
- **Consumer-attached eligibility labels.** A ceiling/delta could pose as a
  cash flow — replaced by adapter-typed emission (Codex review 1).
- **An `AssetCase` VOM knob.** A no-op given embedded solver VOM (Codex review 1).
- **One unified decay across all strategies.** Violates `spread-decay-v1`;
  replaced by explicit projection modes (Codex review 1).
- **Consume the shipped contracted-floor protected cash flow.** Wear-net
  contamination (§4.5).
- **Call it "bankable NPV."** Over-promises absent financing/tax.
- **Claim exact parity with the shipped Revenue NPV for any strategy.** Its DA/
  capture/cycle-life basis differs; parity is asserted only for the DA-only
  adapter at matching params (§3) (Codex review 2).
- **Warning-and-proceed on missing augmentation.** Replaced by the three-state
  `capacity_maintenance_basis`; `UNKNOWN` → lifecycle NPV unavailable (Codex
  review 2).
- **Re-scale the strategy series by `power_mw` in §6.** It is already total EUR
  for the modelled MW — re-scaling double-counts MW (Codex review 2).
- **An absolute per-year EUR `ExplicitAnnualCurve`.** Contradicted the multiply-
  the-draw math; replaced by `ExplicitAnnualMultiplierCurve` (year 1 = 1.0);
  absolute-EUR curve deferred (Codex review 2).
- **A single decay/multiplier on a multi-stream total.** Conflates DA and
  reserve/IDA cannibalisation physics; non-flat projections are DA-only
  (Gemini review).
- **An annuity PV factor for either ProjectCase NPV.** It cannot express a
  non-geometric `ExplicitAnnualMultiplierCurve` or lumpy events, and splitting
  the two NPVs across annuity-vs-explicit paths broke the "differ only by
  lifecycle cost" guarantee; **both** NPVs now discount year-by-year and the
  factor is confined to the parity kernel (Codex review 3).
- **Unbounded numeric inputs.** `discount_rate ≤ -1`, `cost_eur = NaN`, or a
  `[1, NaN]` multiplier satisfied the prose but broke §6 — v1 pins finite/sign
  domains and rejects at `validate()` (Codex review 3).
- **Relying on the audit flag alone for the fail-closed guarantee.**
  `bootstrap_annual_revenue` silently drops non-finite values and returns a €0
  distribution, so an "available" audit could become the prohibited €0 fallback;
  v1 requires the series be non-empty/finite/unique and exactly the valid days
  (Codex review 3).
- **Endogenous fade → augmentation / a fade-driven revenue multiplier.**
  Deferred to v1.1; v1 augmentation is a dated schedule with a declared
  maintenance basis.

## 9. Increments (each a dual-reviewed PR)

- **PC-A** — typed schema (`AssetCase`/`LifecycleCase`/`MarketCase`/
  `ValuationCase`/`ProjectCase`/`RunResult`) + public `StrategyKind` + **solver
  adapters** emitting `StrategyRunResult` + `validate()` + fingerprint. Pure, no
  UI. Pins: producer-typed eligibility, walk-forward gate, three-state
  `capacity_maintenance_basis` (source/as-of required; `UNKNOWN`→unavailable),
  augmentation admissibility, fail-closed, engineering match, floor-not-consumed,
  decay-mode gate, immutable tuple series.
- **PC-B** — bootstrap + lifecycle cash-flow + both NPV **distributions** (pure
  calc). Pins: no-shadow-wear, gross basis, VOM-once, no MW re-scale, year-by-year
  discounting for both NPVs (annuity factor only in the parity kernel), projection
  multipliers (year 1 = 1.0, finite ≥ 0, full-length), event-salvage vs
  terminal-year residual/decommissioning, input-domain rejection, and the narrow
  **DA-only kernel** screening-NPV parity (§3).
- **PC-C** — Revenue-tab/cockpit UI + Excel export (RunResult-driven,
  self-documenting assumptions sheet; floor rendered as separate comparator).
- **PC-D (v1.1)** — `ContractCase.settlement_basis` gross floor composition,
  making the floor cash-NPV-eligible without wear contamination.

## 10. Open decisions for the review round (proposed resolutions folded in)

1. **Floor** — *proposed:* comparator-only **and removed from the aggregator** in
   v1 (§4.5); gross settlement basis deferred to PC-D. (Codex round-1 resolution.)
2. **Discount basis** — *proposed:* **real** rate, explicit **base-year EUR**,
   decay + discount both real (§4.4).
3. **Default strategy** — *proposed:* DA-only realised when only DA is loaded;
   **fail-closed** on strategy invalidation, no silent fallback (§4.6).
4. **Augmentation input** — *proposed:* a **CSV template import + validation
   preview** (import-first playbook), not dynamic sidebar rows.
5. **OpEx scope** — *proposed:* v1 adds **fixed O&M only**; dispatch VOM is
   embedded in the strategy result and not re-deducted (§4.1, §6).
6. **Forecast-mode strictness** — *resolved (Codex round 2):* reject **both**
   `in_sample` and `loo` for the cash NPV; walk-forward only (§5, red-line #10).
7. **Augmentation effect model** — *resolved (Codex round 2):* the three-state
   `capacity_maintenance_basis` (§4.2) replaces the ambiguous premise flag;
   an explicit fade-driven multiplier is v1.1.

Resolved in the Gemini adversarial round:

8. **`ExplicitAnnualMultiplierCurve` provenance** — *resolved:* require a
   machine-checkable `source` + `as_of`, and restrict the mode to `DA_ONLY`
   (§4.7); a bare float array is rejected.
9. **Screening-NPV parity vs the shipped tab** — *resolved:* parity is asserted
   at the shared `calculate_npv_distribution` **kernel** with the tab's own
   (fractional) effective life; the integer-life `ProjectCase` headline is a
   distinct figure and makes no cross-parity claim (§3).
10. **Lifecycle discounting method** — *resolved (Codex round 3):* **both** NPVs
    discount year-by-year with the shared multiplier; the annuity factor is
    confined to the parity kernel (§3, §4.4, §6). (Superseded the Gemini-round
    "screening-only annuity" split, which broke the ExplicitAnnualMultiplierCurve
    screening path.)
11. **Event vs terminal residual** — *resolved:* event salvage is an inflow in
    the event year; terminal residual/decommissioning only in year `L` (§4.2, §6).

Resolved in Codex review round 3 (validation/consistency closure):

12. **Input domains** — all cost/residual/O&M/CapEx finite ≥ 0; `discount_rate`
    finite `> -1`; multipliers finite ≥ 0, year 1 = 1.0, full-length (§4.2, §4.4,
    §4.7).
13. **Series/audit invariant** — the daily series must be non-empty, finite,
    unique-dated, and exactly the audited valid days, so the empty-bootstrap €0
    can never masquerade as a result (§4.6).
14. **Identifier ownership** — `project_life_years` on `LifecycleCase`;
    `strategy_kind` (not `strategy_id`) is the validated identity (§3, §4.6).
15. **Stochastic-batch eligibility rationale** — ineligible because it surfaces a
    policy-value **delta**, not because it is DA+IDA-only (it can carry reserve),
    §5.
