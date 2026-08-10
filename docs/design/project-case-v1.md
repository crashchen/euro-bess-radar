# Project Case v1 — unified pre-tax unlevered lifecycle cash NPV

Status: **locked** (design-contract — LOCKED by a dual **APPROVE** on the round-10
candidate `7b8b547`: Codex **APPROVE** and Gemini **APPROVE**, alongside CC's
independent machine-verification of the two encoder golden vectors and the
code-fact claims. Round 10 closed the two round-9 bar-(c) fingerprint blockers
(`reserve_coverage_audit` entry-date domain; `adapter_provenance.capture_rate`
null-matrix), strengthened the `prob_positive` / `solver_failure_details`
fingerprint pins, and reconciled the currency-basis / reserve-audit-shape /
per-draw-bootstrap-read wording — recorded as `R10-01…08` below. Implementation
(PC-A first) begins once this docs-only lock PR merges.)

Locked on: 2026-08-10

Review metadata (lock):

```
reviewed_candidate_commit: 7b8b547a56690848b4f9ac7ad98797ffa9332c64
lock_basis: dual_approve
locked_on: 2026-08-10
```

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
unlevered lifecycle cash NPV` as typed outcomes, with distributions whenever
their required assumptions are available (§3). Do **not** call
the result "bankable" and do **not** compose the full revenue stack.

In scope: the typed case schema (§4); a producer-issued typed
`StrategyRunResult` as the only revenue input (§4.3, §5); a projection-mode
selector that respects `spread-decay-v1` (§4.7); the dated lifecycle schedule
(§4.2); year-by-year cash-flow construction and the two NPV distributions (§6);
an immutable, input-fingerprinted `RunResult` with full provenance and audit (§4.6).

Out of scope (explicit non-goals): full revenue-stack composition; financing/
tax/DSCR/gearing (v1 is pre-tax unlevered by construction); endogenous physical
fade → augmentation triggering; any new dispatch/MILP formulation (ProjectCase
consumes existing solver outputs, never re-solves).

## 3. Output naming (two typed NPV outcomes)

v1 exposes two typed NPV outcome slots. For a fully specified LifecycleCase,
each carries a Monte-Carlo **distribution** (P10 / P50 / P90 / `P(NPV>0)`), not
a single ambiguous value. Both are built by bootstrapping the
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

Each slot has the exact public shape `NpvOutcome = {available, status, message,
distribution}`; `distribution`, when present, is exactly
`{p10, p50, p90, prob_positive}`. A valid case with
`capacity_maintenance_basis=UNKNOWN` still returns the available screening
outcome, but its lifecycle outcome is the typed unavailable state
`{available:false, status:"capacity_maintenance_unknown",
message:"Engineering capacity-maintenance basis is unknown.",
distribution:null}`. An available outcome is exactly
`{available:true, status:"ok", message:null, distribution:<NpvDistribution>}`.
Invalid ProjectCase/StrategyRunResult input raises before producing a RunResult;
it is not represented by a fabricated unavailable NPV.

**No general "exact parity" claim, and the parity test is a pure NPV-kernel test
(Codex review 5).** The shipped Revenue-tab NPV is built on DA **ordered-spread**
daily revenue (`analytics.calculate_daily_spreads` /
`estimate_annual_arbitrage_revenue`), the sidebar capture haircut, and a
*cycle-derived, fractional* effective life — whereas the `DA_ONLY` adapter's daily
series comes from the **MILP replay** (`simulate_replay_batch`, §5) and
`project_life_years` is a positive **integer** (§4.2), so a full ProjectCase run
does not reproduce the tab. The only parity assertion v1 makes is about the
shared **NPV primitive itself**: feeding an **identical annual-revenue array**
into `scenario.calculate_npv_distribution` with the tab's own capture/effective-
life inputs returns the shipped screening NPV. That test pins the kernel — which
internally uses the aggregated `decaying_annuity_pv_factor` — and is **independent
of which adapter produced ProjectCase's daily series** (it does not run the
`DA_ONLY` MILP adapter at all). ProjectCase's own headline discounts year-by-year
(§6) and never depends on the annuity closed form (wrong for a non-geometric
multiplier curve or lumpy events). Every ProjectCase configuration is a new
figure, not a reproduction of the tab.

We deliberately do **not** call figure 2 "bankable NPV." Without tax, debt
service, DSCR, and financing-fee models it would over-promise. economic-
semantics-v1 requires only that, once augmentation is explicit, both a
no-lifecycle-cost screening NPV and a fully-specified cash NPV are reported.

## 4. Schema

Inputs are small typed "cases" grouped by ownership, aggregated by a lightweight
`ProjectCase`; all computed values + provenance live in a separate **immutable,
input-fingerprinted** `RunResult`.

```
ProjectCase
├── AssetCase          (engineering nameplate + cash fixed O&M)
├── LifecycleCase      (life, dated augmentation/replacement, residual/disposal)
├── MarketCase         (ref to a StrategyRunResult + projection mode)
├── ValuationCase      (discounting; real, base-year EUR, unlevered)
└── BootstrapCase      (Monte-Carlo seed / n_simulations / algorithm version; §4.8)
# ContractCase is NOT part of the v1 aggregator — floor is an external
# comparator only (§4.5). It joins the aggregator in v1.1 with a gross basis.

StrategyRunResult      (producer-issued; the ONLY cash-eligible revenue input; §4.3)

RunResult (immutable, input-fingerprinted, schema-versioned)
├── schema_version + input_fingerprint
├── screening_cashflow_table   (always present for a valid ProjectCase)
├── lifecycle_cashflow_table   (present when lifecycle outcome is available; else null)
├── no_lifecycle_cost_screening_npv  NpvOutcome
├── lifecycle_cash_npv               NpvOutcome
└── provenance          (per-case sources + solver audit + red-line assertions held)

FloorComparatorResult (separate; NOT part of RunResult or its fingerprint; §4.5)
└── the wear-net contracted-floor overlay, a presentation-only comparator
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

Fields: `project_life_years` (a **positive integer**, `1…MAX_PROJECT_LIFE_YEARS`,
`MAX_PROJECT_LIFE_YEARS = 100` — bounds the materialised trajectory, cf.
`contracted-floor-decay-v2`); a `capacity_maintenance_basis` (three-state, below);
an **augmentation/replacement schedule** = an ordered list of dated events, each
`{year, cost_eur, capacity_restored_frac, residual_value_eur}` (the event-level
`residual_value_eur` is the salvage of the equipment that event replaces); a
distinct end-of-life `eol_residual_value_eur`; and `decommissioning_cost_eur`.
LifecycleCase is the **sole owner** of residual, disposal, and decommissioning; no
other case models them.

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
`eol_residual_value_eur` and `decommissioning_cost_eur` are realised **only in
year `project_life_years`**. The two are never conflated.
`capacity_restored_frac` is a fraction of **nameplate energy** (0–1), never MWh
— one unit. In v1 `capacity_restored_frac` is
**descriptive provenance** for the maintenance assertion (below); it does not
drive a revenue multiplier (that is the v1.1 option in §10).

**Augmentation economic effect — explicit three-state (closes the
economic-semantics red-line; applies to ALL projection modes, not just Flat).**
economic-semantics-v1 forbids an augmentation reserve with no capacity-
restoration effect, and a flat/decayed revenue projection is only honest if the
asset is actually kept at the capacity that revenue assumes. So every
LifecycleCase must declare **why** the revenue projection's capacity assumption
holds, via `capacity_maintenance_basis`:

- **`SCHEDULED_NAMEPLATE_MAINTENANCE`** — a **non-empty** augmentation/replacement
  schedule with **at least one event carrying a positive `capacity_restored_frac`**,
  asserted (with engineering **source + as-of**) to sustain the projected capacity
  over the life. The schedule's costs enter cash NPV (§6). `validate()` rejects
  this basis with an empty schedule or a schedule of only zero-restoration events.
- **`NO_AUGMENTATION_REQUIRED_ASSERTED`** — the user explicitly asserts (with
  engineering **source + as-of**) that no augmentation is needed for the projected
  capacity over the life. The schedule **must be empty**; `validate()` rejects this
  basis paired with any augmentation event.
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
- `cash_basis` — an immutable, fingerprinted structure stating that the series is
  **post-VOM** (`post_vom` must be `true`) plus the exact haircut basis, never
  booleans without values:
  `capture = {applied, rate, source}` with `rate` finite and in `[0, 1]` (`1.0`
  when not applied), and `liquidity = {applied, assumption_fingerprint?}` (the
  fingerprint is required iff applied). This lets `0.3` and `1.0` remain distinct
  cash bases and prevents PC from re-applying either haircut;
- engineering binding: `power_mw`, `duration_hours`, `round_trip_efficiency`;
- market binding: `zone`; `sample_window = {first_delivery_date,
  last_delivery_date, timezone}` where both endpoints are **inclusive local market
  delivery dates** and `first ≤ last`. `zone` must be a supported code in
  `config.ALL_ZONES.values()`, and `timezone` is **derived, not selectable**:
  `timezone == config.ZONE_TIMEZONES[zone]`; the fallback behaviour of
  `get_zone_timezone()` is not allowed. Its canonical form is two ISO
  `YYYY-MM-DD` strings plus that registry-owned IANA name; it generates every
  local date in the inclusive interval as `evaluation_dates`. A
  **`currency_basis`** — its wire form is the flat `{mode, target_base_year,
  deflator_method, deflator_vintage, deflator_factor}` of §4.8 — with a
  `target_base_year` (which `validate()` requires to equal `ValuationCase.base_year`
  — a mismatch is fail-closed, §4.4) and **either** `mode = DEFLATOR_APPLIED` with a
  recorded `deflator_method` / `deflator_vintage` / `deflator_factor` (`factor`
  finite `> 0`) the adapter applied to convert historical settlement EUR to
  base-year real EUR, **or** `mode = SOURCE_EUR_TREATED_AS_BASE_YEAR_REAL` (the
  explicit screening assumption; the three deflator members null) — never a silent
  mix;
- kind-specific provenance: a **`ForecastAudit`** per forecast leg — DA, IDA, and
  (for `DA_ID_RESERVE_REALISED`) the **reserve-price forecast** leg separately, not
  one `walk_forward` label — carrying `forecast_mode` + `bucket` + `deadband`.
  The exact required/null matrix is below; it is part of validation and the
  fingerprinted tagged union, not display metadata. `reserve_product` +
  `reserve_source` + `availability` (validated
  **finite and ∈ [0, 1]**, default `config.ANCILLARY_CAPACITY_AVAILABILITY = 0.95`
  — it multiplies straight into reserve cash at `dispatch.py:429`, so an
  out-of-range `availability` would inflate revenue and must be rejected, Codex
  review 6); (scenario provenance does not apply — no v1-eligible strategy is the
  stochastic batch, §5);

  | `StrategyKind` | DA audit | IDA audit | reserve audit | reserve fields |
  |---|---|---|---|---|
  | `DA_ONLY` | null | null | null | all null |
  | `DA_ID_FORECAST` | null | `{forecast_mode:"walk_forward", bucket:B, deadband:D}` | null | all null |
  | `DA_RESERVE_COOPT` | null | null | null | all required |
  | `DA_ID_RESERVE_REALISED` | `{forecast_mode:"walk_forward", bucket:B, deadband:null}` | `{forecast_mode:"walk_forward", bucket:B, deadband:null}` | `{forecast_mode:"walk_forward", bucket:"block_of_day_4h", deadband:null}` | all required |

  Here `B` is exactly `"hour_of_day"` or `"hour_of_week"` and the same selected
  value is used for the DA/IDA legs of the triple strategy. `D` is exactly the
  `PC_ADP_DA_ID` input `min_rebid_uplift_eur`, finite and `>= 0`; it changes the
  realised cash and therefore cannot be omitted or relabelled. "Reserve fields"
  means `reserve_product`, `reserve_source`, `availability`, and
  `reserve_coverage_audit`: all are non-null for the two reserve-bearing kinds
  and all are null otherwise. Required text values are non-empty after trimming.
- for reserve-bearing kinds (`DA_RESERVE_COOPT`, `DA_ID_RESERVE_REALISED`) a
  **`ReserveCoverageAudit`**: a **sorted array with exactly one entry per
  `CoverageAudit.observed_dates` member** — entry dates unique and sorted, and the
  entry date-set **equals `observed_dates` exactly**, so a fully-missing day is
  retained with `present_blocks = ∅` and `missing_blocks = required_blocks`, never
  dropped (it is not a date-keyed map — that shape was ambiguous for the
  fingerprint). For non-reserve kinds the whole field is `null`. Each entry carries
  three canonical 4-hour-block sets — `required_blocks`, `present_blocks`,
  `missing_blocks` — plus `settlement_duration_hours_by_block` from the product
  calendar (finite `> 0`; normally 4h, explicit rather than inferred across gaps).
  Its duration-map keys equal `required_blocks` exactly. A block is `present` only when the raw
  input has **exactly one** row for that ID with a finite
  `block_price_eur_mw_h ≥ 0`; duplicates, NaN/Inf, negative prices (unsupported by
  the v1 solver), or a missing/extra duration key fail the data gate and classify
  that date as `missing_dates`. They must never
  reach `_coerce_nonnegative_interval_vector`, which would silently turn malformed
  values into €0. The audit is derived from the **date-qualified raw reserve series
  BEFORE any scalar collapse or zero-fill** (`PC_ADP_RESERVE_COOPT` passes only a
  scalar `capacity_price_eur_mw_h` to `solve_joint_capacity_batch`,
  `dispatch.py:474`, and missing blocks are zero-filled at `simulation.py:~1024`;
  neither can reconstruct coverage after the fact — the adapter must capture it
  upstream). Machine-checkable per day: `present_blocks ∪ missing_blocks ==
  required_blocks` and `present_blocks ∩ missing_blocks == ∅` (so a
  `required=6, present=5, missing=∅` inconsistency cannot pass). A day is
  reserve-cash-valid **only if `missing_blocks == ∅` AND `present_blocks ==
  required_blocks`**; any other raw-data/coverage failure is deterministically
  `missing_dates`. The result becomes unavailable only when the final
  `valid_dates` set is empty; a case-level schema/adapter invariant violation
  instead raises and emits no StrategyRunResult. This is what the §5 no-relabel
  rule refers to;
- a **`CoverageAudit`** (per `dispatch-failure-contract-v2`): the universe
  **`observed_dates`** is the `evaluation_dates` generated from
  `sample_window + timezone` (**not** the dates that happen to survive in the data,
  so a fully-missing day still appears in the universe and cannot be silently
  dropped before classification). It is partitioned into three **immutable,
  pairwise-disjoint canonical date sets** — `valid_dates`, `missing_dates`,
  `solver_failed_dates` — plus a `solver_failure_details` array carrying **exactly
  one** `{date, status, message, stage}` record per `solver_failed_dates` member
  (its date-set equals `solver_failed_dates`; an empty array when there are no
  solver failures). Machine-checkable: `valid_dates ∪ missing_dates ∪
  solver_failed_dates == observed_dates`, pairwise disjoint, and
  `daily_realised_cash_series` dates equal `valid_dates` exactly. **Completeness is
  checked against an EXPECTED grid, not inferred cadence:** for **every consumed
  market leg** (DA, IDA, and reserve as applicable), every required timestamp or
  product block must have **exactly one finite price**. Energy-market prices may
  be negative; reserve-capacity prices are constrained as above. Expected sets are
  **leg-specific**, keyed by `(market_leg, zone, delivery_date)` in the immutable
  versioned market-grid registry: DA and IDA each require their own explicit
  timestamp calendar and may not inherit one another's resolution; reserve uses
  the explicit product-block calendar in `ReserveCoverageAudit`. The current
  DA-only price-transition registry is not, by itself, IDA coverage. If any
  consumed leg lacks an explicit registry entry for a requested zone/date, the
  adapter is unavailable (model-support failure) and emits no StrategyRunResult —
  it never infers cadence from surviving rows or falls back for IE/CH/GB. With
  registry support present, timestamps are exact-matched to that leg's expected
  set. A duplicate, NaN/Inf, missing, or extra row is `missing_dates`, never
  `solver_failed_dates`. Delegating to
  `_is_regular_utc_day` (`simulation.py:848`) is insufficient — it infers cadence
  from surviving rows via `np.diff`, so a 12-row `00:00…22:00` every-2-hours day
  passes as "regular". **Classification order is pinned:** (1) data-completeness
  gate (expected-grid match) and (2) reserve-coverage gate (§ `ReserveCoverageAudit`)
  run first → a failing day is `missing_dates`; only a day that PASSES the data
  gates and then fails the solver is `solver_failed_dates`. An all-failure or
  empty-`valid_dates` result is *unavailable*, never a €0 series;
- reproducibility: `source_data_content_hash`, `calculator_version`, and a content
  **`fingerprint` computed with the identical canonical serialization + SHA-256 as
  the `RunResult.input_fingerprint`** (§4.8) — the nested fingerprint algorithm is not a
  separate, unspecified one.

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
convention. The historical StrategyRunResult cash is reconciled to this basis via
the `StrategyRunResult.currency_basis` (§4.3): either `mode = DEFLATOR_APPLIED`
with the recorded `deflator_method` / `deflator_vintage` / `deflator_factor` (`> 0`),
or `mode = SOURCE_EUR_TREATED_AS_BASE_YEAR_REAL` is stamped — a silent mix is
rejected. **`validate()` requires `currency_basis.target_base_year ==
ValuationCase.base_year`** (fail-closed): a result deflated to a different base
year can never be discounted against this valuation. In v1 the **COD and the
valuation date are the same date** (year 0); a separate pre-COD construction
schedule is out of scope. Cash timing follows an explicit **end-of-year**
convention; year 0 = `−installed_capex_eur`.

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

Therefore in v1 `ContractCase` is **not** part of the `ProjectCase` aggregator,
and — for the same reason — its output is **not** carried inside `RunResult`
either (Codex review 8): putting a `floor_comparator?` field in the only object
UI/export reads would give the floor an implicit home with no ProjectCase owner
and no place in the `input_fingerprint`. Instead the overlay is a **separate
`FloorComparatorResult`** (a presentation sibling, its own inputs and provenance),
rendered beside the NPV distributions and **never** summed into them. ProjectCase
must not read `floor_protected_cashflow_eur` / `floor_protected_pv_eur`, and
`RunResult` neither contains nor fingerprints the floor.

v1.1 hook (not built): a `ContractCase.settlement_basis` recomputing
`max(M_t, F_t)` per year from a **gross** cash merchant base, which would then be
cash-NPV-eligible and rejoin the aggregator.

### 4.6 `ProjectCase` aggregator + `RunResult`

`ProjectCase` holds the cases and a `validate()` enforcing: engineering match
(§4.3); `strategy_kind` (§4.3) in the §5 allowlist and its StrategyRunResult
solver-available; projection mode compatible with the strategy (§4.7); the §4.2
lifecycle domains; and CapEx matches the sidebar. **Series/audit invariant (Codex
reviews 3–5):** `daily_realised_cash_series` must be **non-empty**, every value
**finite**, dates **unique**, and the series dates must **exactly equal**
`CoverageAudit.valid_dates` (§4.3); the audit's **date-set partition** must hold —
`valid_dates ∪ missing_dates ∪ solver_failed_dates == observed_dates` with the
three **pairwise disjoint** (canonical date sets, not counts, so an overlapping
mis-assignment cannot pass). This closes the hole where an "available" result is
silently thinned into the prohibited €0 fallback that `bootstrap_annual_revenue`
produces when all values are non-finite (`scenario.py:128`). A **zero or negative**
cash value is a legitimate market outcome and is **never** treated as failure —
the gate is finiteness and date-set identity, never `value == 0`. **Fail-closed:** if
the referenced strategy result is invalid/unavailable, or the series/audit
invariant fails, `validate()` raises — no silent fallback to a lesser strategy or
to €0 (`dispatch-failure-contract-v2`).

`RunResult` is immutable, carries a `schema_version`, is serialisable, and holds
an `input_fingerprint` over every case — including `BootstrapCase` (§4.8: `seed`,
`n_simulations`, `bootstrap_algorithm_version`) and the `StrategyRunResult`
fingerprint — so a stale result is detectable and cache invalidation is
deterministic (the frontier/floor fingerprint pattern). It is the only object UI
and export read for the **ProjectCase NPV outputs**; the separate
`FloorComparatorResult` (§4.5) is rendered beside it as a presentation sibling.

**Partial availability is typed, not inferred from null arithmetic.** The two NPV
fields use the exact `NpvOutcome` envelope in §3. `screening_cashflow_table` is
always present after successful validation. With either active
capacity-maintenance basis, `lifecycle_cashflow_table` and the lifecycle
distribution are present; with `UNKNOWN`, the lifecycle table and distribution
are null and the fixed unavailable status/message in §3 is returned while the
screening outcome remains available. Distribution numbers must be finite and
`prob_positive ∈ [0,1]`; non-finite calculation output raises and produces no
RunResult. UI/export branch only on `available` + `status`, never on a missing key,
NaN, an empty dict, or an assumed €0.

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

### 4.8 `BootstrapCase` — Monte-Carlo reproducibility (owner for seed/count)

The bootstrap `seed` and `n_simulations` are not free-floating fingerprint
entries; they are a typed, owned case so their domains are validated and their
values are reproducible. Fields (concrete locked constants, Codex review 5 — no
"e.g."): `seed` (a **non-negative int in `[0, 2^64−1]`**; `bool`/`None`/float
rejected — `True` must not slip in as `1` — and the uint64 bound keeps the
fingerprint's `PC-CBOR-F64-v1` integer encoding total, §4.8 serialization);
`n_simulations` (a positive int in
`[MIN_SIMULATIONS, MAX_SIMULATIONS] = [1000, 50000]`, **default 5000** to match
the shipped `scenario.bootstrap_annual_revenue`); and
`bootstrap_algorithm_version`, whose **only v1 literal** is
`"pc-bootstrap-pcg64-choice365-linear-v1"` (unknown versions fail closed). That
literal means: daily values ordered by `valid_dates`; NumPy `Generator(PCG64(seed))`;
one `choice(values, size=(n_simulations, 365), replace=True)`; row-wise float64
sum; and `percentile(..., method="linear")`. PC-A pins a golden annual-output and
percentile vector; any RNG/sampling/percentile change requires a new algorithm
literal. All three fields enter the `RunResult.input_fingerprint` (§4.6).

**Remaining numeric domains (Codex review 4).** `validate()` also bounds every
other live scalar so red-line #15 ("all numeric inputs bounded") is real, not
aspirational: `AssetCase.power_mw > 0`, `duration_hours > 0`,
`0 < round_trip_efficiency ≤ 1`; `DAOnlySpreadDecay` `annual_decay_rate ∈ [0, 1)`
and `decay_floor_share ∈ [0, 1]` (the `spread-decay-v1` domains);
`PC_ADP_DA_ID.min_rebid_uplift_eur` is finite and `>= 0`. No v1-eligible
strategy is the stochastic batch (§5), so **`scenario_count` is not a live domain
in v1** — the earlier "stochastic cap" is moot. A NaN/Inf/out-of-range value in
any live scalar is rejected before §6.

**Deterministic fingerprint (Codex reviews 5–9).** The `RunResult.input_fingerprint`
is **SHA-256** over a **complete canonical binary serialization** so two
independent implementations produce the identical byte stream (and hence hash) for
the same inputs, with no structural collisions. v1 pins a **named local profile
`PC-CBOR-F64-v1`** — it uses CBOR framing (type tags, length-prefixed
strings/arrays/maps) but deliberately **does not claim RFC 8949 §4.2 *core*
deterministic encoding**, because that mandates the shortest float that preserves
value (`§4.2.1`), while this local profile deliberately chooses one uniform f64
wire type for every real-valued schema field. Both choices are lossless for the
numeric value, but they are different deterministic profiles and must not share a
name.
The **schema-normalised envelope** is one CBOR map with exactly four keys:
`{profile: "PC-CBOR-F64-v1", object_type: "ProjectCase" |
"StrategyRunResult", schema_version: "project-case-v1", payload: <map>}`. Unknown
keys are invalid for that schema version. The object's own `fingerprint` /
`input_fingerprint` field is excluded from its payload; ProjectCase includes the
nested StrategyRunResult fingerprint as lowercase 64-character hex text.

`PC-CBOR-F64-v1` then pins every type and normalisation:

- every **real-valued schema field** — money/cash, MW/MWh/hours, rates, ratios,
  availability, multipliers, prices and block durations — is converted to a
  finite IEEE-754 binary64 and emitted in the 64-bit double form (major type 7,
  additional 27); this preserves the normalised binary64 value uniformly,
  `−0.0` normalises to `+0.0`, and int-vs-float caller syntax cannot change its
  bytes;
- the only **integer-valued schema fields** are `seed`, `n_simulations`,
  `project_life_years`, event `year`, `base_year`, and `target_base_year`.
  Booleans are rejected for them. They use the smallest CBOR unsigned-int head;
  `seed ∈ [0, 2^64−1]`, the smaller domains above continue to apply, and both
  base-year fields are integers in `[1900, 9999]` and must match (§4.4);
- booleans use CBOR simple `false`/`true`; text is UTF-8 NFC; dates use
  `YYYY-MM-DD`; enums use their exact member-name text; content hashes use
  lowercase 64-character hexadecimal text;
- every declared optional key is **always present**, with CBOR `null` when absent
  (omitting it is invalid). Map keys are sorted by their encoded bytes. Logical
  sets are arrays sorted by each element's canonical encoded bytes; the daily cash
  tuple is sorted by date; lifecycle events sort by `(year, canonical event
  bytes)`; arrays whose order is semantic (for example annual multipliers) retain
  that declared order.

All remaining scalar wire types are closed: `source`, `method`, `vintage`,
`calculator_version`, adapter/function/field IDs, product, status, stage, and
message are NFC text (required provenance/ID values are non-empty); every `as_of`
is an ISO `YYYY-MM-DD` date. `projection_kind` is exactly one of
`FlatRealProjection`, `DAOnlySpreadDecay`, or
`ExplicitAnnualMultiplierCurve`. Their tagged-union members are respectively:

- Flat: all of `annual_decay_rate`, `decay_floor_share`, `multipliers`, `source`,
  and `as_of` are null;
- spread decay: `annual_decay_rate` and `decay_floor_share` are non-null, with
  `multipliers`/`source`/`as_of` null;
- explicit curve: `multipliers`, `source`, and `as_of` are non-null, with the two
  decay fields null.

The StrategyKind-specific forecast/reserve null matrix is exactly the table in
§4.3. `reserve_price_aggregation` is either null or the sole v1 literal
`duration_weighted_mean_complete_blocks_v1`; it and non-empty
`reserve_pricing_dates` plus `reserve_scalar_price_eur_mw_h` are non-null only for
`PC_ADP_RESERVE_COOPT`. `adapter_provenance.capture_rate` is non-null **only for
`PC_ADP_DA_ONLY`** — there it equals **both** `cash_basis.capture.rate` **and** the
exact value passed to `simulate_replay_batch` — and is **`null` for the other three
adapters**, whose `cash_basis.capture` stays `{applied:false, rate:1.0,
source:"not_applied"}`. So `capture_rate` records the actually-existing
solver-adapter parameter while `cash_basis.capture` remains the single cash-basis
owner (the two never disagree, and the non-DA adapters do not fabricate a
non-existent parameter). `currency_basis.mode` and its branch nullability are
pinned below.

**Production payload registry.** Public typed field names below are the exact
snake_case CBOR keys — aliases such as `asset` for `asset_case` are invalid. The
rule applies recursively; braces below denote a CBOR map, and every listed
optional key is still present-null under the rule above:

- `ProjectCase.payload` has exactly `{asset_case, lifecycle_case, market_case,
  valuation_case, bootstrap_case}`.
- `asset_case` has exactly `{power_mw, duration_hours,
  round_trip_efficiency, installed_capex_eur, fixed_om_eur_per_mw_yr}`.
  Derived `energy_mwh` and the alternative `capex_eur_per_kwh` input spelling are
  excluded after normalisation to `installed_capex_eur`.
- `lifecycle_case` has exactly `{project_life_years,
  capacity_maintenance_basis, capacity_maintenance_source,
  capacity_maintenance_as_of, augmentation_events, eol_residual_value_eur,
  decommissioning_cost_eur}`. Each `augmentation_events` entry has exactly
  `{year, cost_eur, capacity_restored_frac, residual_value_eur}`. The source and
  as-of fields are non-null for the two asserted maintenance bases and null for
  `UNKNOWN`.
- `market_case` has exactly `{strategy_run_fingerprint, projection}` — it embeds
  the lowercase digest, not a second copy of StrategyRunResult. `projection` has
  exactly `{projection_kind, annual_decay_rate, decay_floor_share, multipliers,
  source, as_of}`; unused tagged-union members are null.
- `valuation_case` has exactly `{discount_rate, base_year}`;
  `bootstrap_case` has exactly `{seed, n_simulations,
  bootstrap_algorithm_version}`.
- `StrategyRunResult.payload` has exactly `{strategy_kind,
  daily_realised_cash_series, cash_basis, power_mw, duration_hours,
  round_trip_efficiency, zone, sample_window, currency_basis, forecast_audits,
  reserve_product, reserve_source, availability, reserve_coverage_audit,
  coverage_audit, adapter_provenance, embedded_vom_cost_eur_mwh,
  source_data_content_hash, calculator_version}`. The display-only human label,
  pandas/index objects, derived summary totals, and its own `fingerprint` are
  excluded. Each daily cash item is the two-element array `[date, value_eur]`.
- `cash_basis` has exactly `{post_vom, capture, liquidity}`; `capture` has
  `{applied, rate, source}` and `liquidity` has
  `{applied, assumption_fingerprint}`. `sample_window` has exactly
  `{first_delivery_date, last_delivery_date, timezone}`. `currency_basis` has
  exactly `{mode, target_base_year, deflator_method, deflator_vintage,
  deflator_factor}`. `mode` is exactly `DEFLATOR_APPLIED` or
  `SOURCE_EUR_TREATED_AS_BASE_YEAR_REAL`; the three deflator members are null in
  the latter branch and non-null in the former (`deflator_vintage` is text).
- `forecast_audits` has exactly `{da, ida, reserve}`; each non-null leg has
  exactly `{forecast_mode, bucket, deadband}`. `reserve_coverage_audit` is null for
  non-reserve kinds, otherwise an array sorted by local date with **exactly one
  entry per `coverage_audit.observed_dates` member** (the entry date-set equals
  `observed_dates` exactly), whose entries have
  exactly `{date, required_blocks, present_blocks, missing_blocks,
  settlement_duration_hours_by_block}`. A block ID is its UTC interval start as
  `YYYY-MM-DDTHH:MM:SSZ`; the three block arrays and duration-map keys use those
  exact strings.
- `coverage_audit` has exactly `{observed_dates, valid_dates, missing_dates,
  solver_failed_dates, solver_failure_details}`. `solver_failure_details` carries
  **exactly one entry per `solver_failed_dates` member** (its date-set equals
  `solver_failed_dates` exactly; an empty array when there are no solver failures).
  Each failure detail has exactly `{date, status, message, stage}` and the array
  sorts by `(date, stage, status, message)`.
- `adapter_provenance` has exactly `{producer_adapter_id, source_function,
  per_day_cash_field, excluded_fields, mode, carry_soc, soc_init_frac,
  capture_rate, reserve_price_aggregation, reserve_pricing_dates,
  reserve_scalar_price_eur_mw_h, expected_grid_registry_version,
  expected_grid_profiles}`.
  `expected_grid_registry_version` is the sole v1 literal `pc-market-grid-v1`;
  any calendar-content change requires a new version literal (and calculator
  version), never an in-place mutation. `expected_grid_profiles` has exactly
  `{da, ida, reserve}`: each consumed leg has its non-empty registry profile ID
  and each unconsumed leg is null. Adapter-inapplicable members are null;
  `excluded_fields` and `reserve_pricing_dates` are canonical sorted arrays.

For `expected_grid_profiles`, `da` is required for all four kinds; `ida` is
required only for `DA_ID_FORECAST` and `DA_ID_RESERVE_REALISED`; `reserve` is
required only for `DA_RESERVE_COOPT` and `DA_ID_RESERVE_REALISED`. The referenced
profile must contain every `(leg, zone, delivery_date)` in `sample_window`; a
registry version/profile ID is provenance, not permission to use cadence inference.

Validation requires `cash_basis.post_vom == true`. Every v1 adapter records
`carry_soc=false` and `soc_init_frac=0.5`: the DA-only adapter passes both values
explicitly; the three already-per-day producers must likewise bind the value in
their PC-A adapter rather than inherit an ambient solver default. This makes the
daily-bootstrap SoC basis uniform and fingerprint-visible. Every adapter also
requires `embedded_vom_cost_eur_mwh == dispatch.DISPATCH_VOM_COST_EUR_MWH ==
0.5`; any other value invalidates the StrategyRunResult rather than merely
misstating VOM-once provenance. `producer_adapter_id`, `source_function`,
`per_day_cash_field`, and `excluded_fields` take exactly the values shown in the
§5 allowlist row for that adapter. For `PC_ADP_DA_ONLY`, provenance `mode` is
exactly `DA MILP Replay`; it is null for the other three adapters.

No derived/transient field may enter either digest unless this registry and
`schema_version` are revised together.

The following encoder-conformance vectors are part of the contract. Spaces/newlines
in the diagnostic input are not encoded; the shown hex is the exact envelope CBOR,
and the hash is SHA-256 over those bytes. They are compact **profile probes** that
bypass domain-object validation (so the small diagnostic payloads need not contain
every required ProjectCase/StrategyRunResult field); PC-A adds full valid-object
vectors using the same encoder:

| object | diagnostic payload | canonical CBOR hex | SHA-256 |
|---|---|---|---|
| `StrategyRunResult` | `{available: true, cash_eur: 1.5, optional: null, valid_dates: ["2026-01-01"]}` | `a4677061796c6f6164a468636173685f657572fb3ff8000000000000686f7074696f6e616cf669617661696c61626c65f56b76616c69645f6461746573816a323032362d30312d30316770726f66696c656e50432d43424f522d4636342d76316b6f626a6563745f7479706571537472617465677952756e526573756c746e736368656d615f76657273696f6e6f70726f6a6563742d636173652d7631` | `7822bc55d4814c1f6a19f28e6a6572707d972fc8db965ce00f77c8654e7dc2c5` |
| `ProjectCase` | `{discount_rate: 0.08, seed: 42, strategy_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"}` | `a4677061796c6f6164a36473656564182a6d646973636f756e745f72617465fb3fb47ae147ae147b7473747261746567795f66696e6765727072696e747840303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030306770726f66696c656e50432d43424f522d4636342d76316b6f626a6563745f747970656b50726f6a656374436173656e736368656d615f76657273696f6e6f70726f6a6563742d636173652d7631` | `fba8fe9cf655ed5ce88094bf7a9f576deb8bf81a578355ddb1496dea86f280bd` |

PC-A additionally ships full-object and bootstrap golden vectors. The nested
`StrategyRunResult.fingerprint` (§4.3) uses this **same** profile — exactly one
spec. Nothing outside the canonical payload (dict insertion order, pandas index
identity, object ids, or an ambient library version not explicitly recorded in a
schema field) may influence the hash.

## 5. Cashflow-eligible strategies (producer allowlist / denylist)

Eligibility is enforced by *which adapter emits a StrategyRunResult*, not by a
consumer-attached label. The strategy-comparison table (`strategy_compare.py`)
mixes totals, ceilings, and deltas in one frame, so it is never a ProjectCase
input.

**A `StrategyKind` alone is not enough** (Codex reviews 4–5): one producer
function returns realised, ceiling, and delta figures side by side, so a
mislabelled column could still reach the cash line. Each adapter is a pinned
**5-tuple** `ProducerAdapterId → StrategyKind → source function → per-day cash
field → excluded fields`, and the adapter reads **only** the per-day cash field.
The field names below are the actual per-day DataFrame columns (Codex review 5):

| ProducerAdapterId | StrategyKind | source function | per-day cash field | must NOT read |
|---|---|---|---|---|
| `PC_ADP_DA_ONLY` | `DA_ONLY` | `simulate_replay_batch` (DA-only) | `total_revenue_eur` | `degradation_cost_eur` (shadow wear) |
| `PC_ADP_DA_ID` | `DA_ID_FORECAST` | `simulate_sequential_da_id_batch` | `realised_eur` | `ceiling_eur` |
| `PC_ADP_RESERVE_COOPT` | `DA_RESERVE_COOPT` | `solve_joint_capacity_batch` | `joint_total_revenue` | — |
| `PC_ADP_DA_ID_RESERVE` | `DA_ID_RESERVE_REALISED` | `simulate_sequential_da_id_reserve_batch` | `realised_eur` | `reserve_first_ceiling_eur`, `global_ceiling_eur` |

**Shadow-wear ingress (Codex review 5):** `simulate_replay_batch` also exposes a
per-day `degradation_cost_eur` (`simulation.py:81`); the `DA_ONLY` adapter reads
**only** `total_revenue_eur` (the gross market cash) and never that column — a
named red-line, since the vague "realised per-day cash" wording previously left
this open. All four adapters emit and validate
`embedded_vom_cost_eur_mwh=0.5`, the exact dispatch constant; a producer cannot
claim `post_vom=true` with a different or missing embedded VOM value.

**DA-only adapter settings — concrete pinned values (Codex reviews 5, 8).**
`simulate_replay_batch` defaults `mode="DA MILP Replay"`, `soc_init_frac=0.5`,
`carry_soc=True` (`simulation.py:379-392`). The `DA_ONLY` adapter **pins** (and
records in provenance) the exact values the daily bootstrap requires:

- `mode = "DA MILP Replay"`;
- **`carry_soc = False`** — this is load-bearing, not cosmetic: `carry_soc=True`
  transfers stored energy across days, so day *N*'s cash depends on day *N−1*'s
  ending SoC. Bootstrapping each day's cash **independently** (§6) assumes i.i.d.
  daily draws, which cross-day carry violates. v1's daily bootstrap therefore
  **requires** standalone terminal-neutral days (`carry_soc=False`); a carry-over
  basis would need block/horizon bootstrapping and is out of v1 scope;
- **`soc_init_frac = 0.5`** — a concrete pinned value (matching the
  `simulate_replay_batch` default), because terminal neutrality alone does **not**
  fix a unique initial SoC and different values change feasible dispatch and daily
  cash; the number must be pinned, not left to a caller default;
- adapter input **`capture_rate` is finite and in `[0, 1]`**, with a required
  source. The adapter applies that exact value inside `simulate_replay_batch`, then
  emits `cash_basis.capture = {applied: capture_rate != 1.0, rate:
  capture_rate, source}`; the structure enters the StrategyRunResult fingerprint
  and PC never re-applies it. The other three v1 adapters do not expose a legal
  post-hoc capture haircut: they emit `{applied: false, rate: 1.0,
  source: "not_applied"}`. In particular, multiplying `joint_total_revenue` by a
  DA capture rate is forbidden because it would also haircut reserve-capacity cash
  and embedded VOM. All four v1 adapters emit `liquidity.applied=false`; adding a
  liquidity-aware cash producer requires a new adapter/profile, not a UI toggle.

The adapter does **not** attempt to equal the tab's ordered-spread series (the
parity claim is a pure NPV-kernel test, §3, not "adapter reproduces tab").

**`PC_ADP_RESERVE_COOPT` scalar derivation — pinned (Codex review 9 + bounded
cash-basis audit).**
`solve_joint_capacity_batch` takes a **scalar** `capacity_price_eur_mw_h`
(`dispatch.py:474`) that enters `capacity_revenue` directly (`dispatch.py:429`) and
shapes the optimisation, so the adapter pins and fingerprints the complete
derivation:

1. Before any scalar collapse or solver call, define `pricing_dates =
   DA_expected_grid_complete_dates ∩ reserve_fully_covered_dates`. A
   reserve-incomplete date is `missing_dates`; neither its partial cash nor its
   surviving reserve-price rows may influence another date. Empty `pricing_dates`
   makes the strategy unavailable.
2. Restrict the chosen product's raw rows to `pricing_dates`; require exactly one
   finite non-negative price and one finite positive explicit duration for every
   required canonical block ID, with no duplicates or extra/missing duration keys.
   Compute the scalar as
   `Σ(block_price_eur_mw_h × explicit_settlement_duration_h) /
   Σ(explicit_settlement_duration_h)`, where each duration comes from the canonical
   product block/calendar, never from the gap to the adjacent surviving timestamp.
   This preserves the shipped duration-weighted-mean intent while prohibiting a
   discontiguous slice from treating (for example) a 28-hour cross-day gap as one
   block's duration. Calling the current `_infer_capacity_duration_hours()` on the
   filtered slice is therefore not the normative algorithm.
3. Run `solve_joint_capacity_batch` on exactly `pricing_dates`, with
   **`soc_init_frac = 0.5`** and `capture={applied:false, rate:1.0,
   source:"not_applied"}`. Only data-complete days that subsequently fail the MILP
   become `solver_failed_dates`; the remaining output rows are `valid_dates`.

`DA_ID_FORECAST` / `DA_ID_RESERVE_REALISED` are **walk-forward only** and record a
per-leg `ForecastAudit` (§4.3). Two adapter rules close the last gap:

- **No relabelling — degraded reserve fails, it does not become another kind.**
  A degraded reserve-first run must distinguish three cases: (a) reserve
  data missing/out-of-window (`simulation.py:~1024/1236` aligns missing blocks to
  zero), (b) a valid published zero reserve price, (c) the optimiser choosing
  `0 MW` on valid data. Only (b)/(c) are legitimate `DA_ID_RESERVE_REALISED`
  results, and the **`ReserveCoverageAudit`** (§4.3) proves it: a reserve day is
  cash-valid only when its required 4-hour product blocks are **fully covered**
  (`missing_blocks == ∅`), which distinguishes (a) a missing block (zero-filled
  upstream) from (b) a genuine published zero price. Case (a) deterministically
  puts that day in `missing_dates`; only a final empty `valid_dates` set makes the
  whole strategy unavailable. It is **never** re-emitted as `DA_ID_FORECAST`
  (that kind is bound exclusively to
  `PC_ADP_DA_ID`; relabel is forbidden). Producing a genuine DA+ID result requires
  **re-running** `PC_ADP_DA_ID`, not relabelling.
- **The stochastic batch has no adapter.** `simulate_stochastic_da_id_batch`
  *does* accept reserve MW/prices (`simulation.py:1875-1876`), so its realised
  totals can include reserve capacity — but its cockpit-surfaced product is a
  **policy-value delta**, so no tuple maps to it and it is unrepresentable as
  cash revenue.

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
double-count MW). Its per-day values are bootstrapped by
`scenario.bootstrap_annual_revenue`, and PC reads the **per-draw `["simulations"]`
array** (the raw `n_simulations` annual sums, not the percentile summary) as
`annual_revenue_draw_d`. For draw `d` and year
`t = 1 … project_life_years`:

```
revenue_{d,t} = annual_revenue_draw_d × projection_multiplier_t   # §4.7 (year 1 = 1.0)
opex_t        = fixed_om_eur_per_mw_yr × power_mw                 # VOM already inside gross revenue
augment_t     = Σ (event.cost_eur − event.residual_value_eur) for events in year t   # net capital; event salvage is an inflow in the event year (§4.2)
terminal_t    = (eol_residual_value_eur − decommissioning_cost_eur) if t == project_life_years else 0
net_{d,t}     = revenue_{d,t} − opex_t − augment_t + terminal_t
```

The lifecycle rows and formula below execute only for the two asserted
capacity-maintenance bases. With `UNKNOWN`, ProjectCase computes the screening
rows/outcome, sets `lifecycle_cashflow_table=null`, and returns the fixed typed
lifecycle-unavailable outcome in §3; it does not assume zero augmentation.

`lifecycle_cash_npv` for draw `d` is the **explicit year-by-year** sum
`−installed_capex_eur + Σ_{t=1..L} net_{d,t} / (1+discount_rate)^t`, with
`net_{d,t} = revenue_{d,t} − opex_t − augment_t + terminal_t`; over all
`n_simulations` draws this yields the distribution, where **`p10/p50/p90 =
percentile(npv_draws, {10,50,90}, method="linear")`** and **`prob_positive =
mean(npv_draws > 0.0)`** — the fraction of **all** `n_simulations` draws with a
strictly positive NPV, no draw dropped. `no_lifecycle_cost_screening_npv` uses the
identical summary over its own draws.

`no_lifecycle_cost_screening_npv` is the **same** year-by-year sum with **`opex_t
= augment_t = terminal_t = 0`** (revenue and CapEx only — §3.1 defines it as
revenue minus CapEx; fixed O&M is a lifecycle cash cost and must NOT appear in
the screening figure, Codex review 4). When lifecycle is available, both use the
same bootstrap draws, the same `projection_multiplier_t`, the same horizon, and
the same year-by-year
discounting, so per draw the two differ **only** by the lifecycle-cost layer, by
construction:

```
lifecycle_npv_d − screening_npv_d
  = Σ_{t=1..L} (−opex_t − augment_t + terminal_t) / (1+discount_rate)^t
  = PV( −fixed O&M − augmentation capital + event salvage + terminal residual − decommissioning )
```

Neither NPV uses the aggregated annuity factor (it cannot express a non-geometric
multiplier curve or lumpy events, §4.4); that factor lives only in the parity
kernel (§3). Shadow wear appears **nowhere**; VOM is not re-deducted; the series
is not re-scaled by MW. All cash amounts are base-year real EUR (§4.4). A legal
series may be all-zero or negative in places — those are valid market outcomes,
never a failure signal (§4.6).

## 7. Red lines

1. **No shadow wear in cash NPV** — never `calculate_degradation_cost()` on the
   cash line, and never via the floor back door (§4.5).
2. **Not bankable** — pre-tax unlevered; the label and UI copy say so.
3. **No full-stack composition** — one producer-issued strategy result feeds the
   cash line; overlays/comparators stay non-additive.
4. **Capacity maintenance declared + schedule-consistent** —
   `capacity_maintenance_basis` is one of the three states (§4.2); the two active
   states require an engineering source + as-of; `SCHEDULED_NAMEPLATE_MAINTENANCE`
   requires a non-empty schedule with ≥1 positive-restoration event and
   `NO_AUGMENTATION_REQUIRED_ASSERTED` requires an empty schedule; `UNKNOWN` makes
   the lifecycle NPV the typed unavailable outcome while screening remains
   available, never a warning-and-proceed or a null/€0 distribution. Augmentation
   events need year + cost + restoration (frac of nameplate) + residual, or are
   rejected (§3, §4.6).
5. **Floor comparator-only** — not in the aggregator; ProjectCase never reads
   `floor_protected_cashflow_eur` / `_pv_eur`.
6. **Gross, producer-issued revenue** — eligibility is a `StrategyKind`/adapter
   property; ceilings, deltas, overlays, gross-additive references, benchmarks,
   and unavailable results are unrepresentable as cash revenue. The `DA_ONLY`
   adapter reads only `total_revenue_eur`, never the co-exposed
   `degradation_cost_eur` (shadow-wear ingress, §5).
7. **VOM counted once** — every adapter validates `post_vom=true` and
   `embedded_vom_cost_eur_mwh == dispatch.DISPATCH_VOM_COST_EUR_MWH == 0.5`;
   VOM is never re-deducted in §6. No AssetCase VOM knob.
8. **MW counted once** — the strategy series is already total EUR for the modelled
   MW; §6 never re-scales it by `power_mw`.
9. **Decay stays in its lane** — **any** non-flat projection
   (`DAOnlySpreadDecay` or `ExplicitAnnualMultiplierCurve`) is `DA_ONLY` only, per
   `spread-decay-v1`; multi-stream strategies are `FlatRealProjection` only. One
   multiplier on a composite total conflates stream-specific cannibalisation.
10. **Forecast = walk-forward with a closed tagged union** for any cash-eligible
    forecast-driven strategy; `loo` and `in_sample` are ineligible. The exact
    per-kind DA/IDA/reserve audit null matrix, bucket literals, and DA+ID deadband
    binding are §4.3 public schema.
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
15. **All numeric inputs bounded (concrete constants)** — event costs/residuals/
    decommissioning/O&M/CapEx finite ≥ 0, multipliers finite ≥ 0 with year 1 = 1.0
    and full-length, `capacity_restored_frac ∈ [0,1]`, `power_mw/duration_hours > 0`,
    `0 < RTE ≤ 1`, decay `d ∈ [0,1)` / floor `∈ [0,1]`, reserve `availability ∈
    [0,1]` (default 0.95 — it multiplies into reserve cash, `dispatch.py:429`),
    capture rate `∈ [0,1]`, `min_rebid_uplift_eur` finite `>=0`, `seed ∈
    [0,2^64−1]`, `n_simulations ∈ [1000,
    50000]` (default 5000), base years integer `∈ [1900,9999]`, and
    `project_life_years ∈ [1, MAX_PROJECT_LIFE_YEARS=100]`; a NaN/Inf/out-of-range
    input is rejected before §6 (§4.2, §4.7, §4.8). No "e.g." constants survive in
    the locked text.
16. **Screening excludes O&M** — `no_lifecycle_cost_screening_npv` is revenue −
    CapEx only (`opex = augment = terminal = 0`); fixed O&M is a lifecycle cash
    cost and appears only in `lifecycle_cash_npv` (§3, §6).
17. **Audit partition over an expected universe** — `zone` must be supported and
    `sample_window.timezone == config.ZONE_TIMEZONES[zone]`; `observed_dates` is
    the resulting `evaluation_dates`. `valid`/`missing`/
    `solver_failed` partition it (disjoint + covering) and the series equals
    `valid_dates`. Completeness requires every consumed leg to have exactly one
    finite value at every point in its **leg-specific** versioned registry calendar
    (negative energy prices remain legal); DA/IDA grids are never substituted or
    inferred. Missing registry coverage makes the adapter unavailable.
    Classification order is deterministic data/reserve gates → `missing`, then
    solver → `solver_failed`. A zero or negative cash value is valid, never a
    failure signal (§4.3, §4.6).
18. **One adapter, one field** — each `StrategyKind` binds a fixed
    (`ProducerAdapterId`, source function, per-day cash field, excluded fields)
    tuple with real column names; a degraded reserve→0 day is `missing_dates`,
    and the whole result is unavailable iff final `valid_dates` is empty — never
    relabelled to another kind (§5).
19. **Currency basis matches valuation** — `currency_basis.target_base_year ==
    ValuationCase.base_year` (fail-closed); deflator `factor` finite `> 0` (§4.3,
    §4.4).
20. **Deterministic, lossless fingerprint** — SHA-256 over the named
    `PC-CBOR-F64-v1` schema-normalised envelope (float64 always 64-bit, typed
    integer fields, null-present optionals, sets as canonical sorted arrays,
    explicit object/schema tags, embedded golden vectors); it does **not** claim
    RFC 8949 core deterministic encoding (which mandates shortest-float); one spec
    is shared by `RunResult.input_fingerprint` and the nested
    `StrategyRunResult.fingerprint` (§4.8).
21. **Reserve days must be fully covered** — a `DA_ID_RESERVE_REALISED` /
    `DA_RESERVE_COOPT` day is cash-valid only when its `ReserveCoverageAudit` shows
    `present_blocks == required_blocks` (partition-checked, unique block IDs,
    explicit settlement durations, derived before scalar collapse/fill); a
    zero-filled missing block is deterministically `missing_dates`; whole-result
    unavailable occurs only when no valid date remains (§4.3, §5).
22. **Daily-bootstrap i.i.d. basis** — all four adapters record
    `carry_soc=False` and `soc_init_frac=0.5`; a cross-day carry basis breaks the
    independent-daily-draw assumption of §6 and would need block/horizon
    bootstrapping (out of v1). Capture is a structured,
    fingerprinted `{applied, rate, source}` basis and is applied once by its
    adapter; the three non-DA adapters pin `rate=1.0/not_applied` (§4.3, §5).
23. **Floor stays out of `RunResult`** — the wear-net comparator is a separate
    `FloorComparatorResult`, not a field of the fingerprinted `RunResult` (§4.5).
24. **Reserve scalar excludes incomplete dates** — `PC_ADP_RESERVE_COOPT` builds
    `pricing_dates` from DA-complete ∩ reserve-complete dates before collapse,
    computes the duration-weighted scalar from explicit product-block durations,
    and solves exactly that window. Partial/missing reserve rows never influence
    another day's cash (§5).
25. **Bootstrap algorithm is versioned, not merely seeded** — v1 accepts only
    `pc-bootstrap-pcg64-choice365-linear-v1`; RNG, 365-day sampling, replacement,
    sum and percentile method are pinned, with a golden vector required (§4.8).

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
  capture/cycle-life basis differs; parity is asserted only as a pure
  `calculate_npv_distribution` **kernel** test on identical annual inputs,
  independent of any adapter (§3) (Codex reviews 2, 5).
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
- **Deducting fixed O&M in the screening NPV.** §6 previously kept `−opex_t` in
  both figures; the screening NPV is revenue − CapEx only, O&M is lifecycle-only
  (Codex review 4).
- **A `CoverageAudit` of counts without date identities.** The series==valid-days
  invariant was unimplementable; the audit now carries a canonical `valid_dates`
  set (Codex review 4).
- **`StrategyKind` as the sole eligibility guard.** One producer returns realised
  + ceiling + delta; each kind now binds a fixed `ProducerAdapterId` 5-tuple
  (adapter, kind, function, per-day field, excluded fields) and a degraded
  reserve→0 run cannot be relabelled (Codex reviews 4–5).
- **Free-floating `seed`/`n_simulations` in the fingerprint only.** Promoted to a
  typed `BootstrapCase` with domains; the remaining scalar domains are closed
  (Codex review 4).
- **Silently treating source settlement EUR as base-year real.** The adapter must
  record a `deflator` or stamp the explicit "treat as base-year real" assumption
  (`currency_basis`, Codex review 4).
- **Wrong/vague per-day field names in the producer table.** Round-4 names
  (`realised_total`, "realised per-day cash") did not match code; the table now
  uses the real per-day columns (`total_revenue_eur`, `realised_eur`,
  `joint_total_revenue`) + `ProducerAdapterId` + excluded columns, and pins the
  DA-only call settings (Codex review 5).
- **Relabelling a degraded reserve run as `DA_ID_FORECAST`.** That is a relabel to
  another producer's kind; a degraded reserve day is `missing_dates`, the whole
  result is unavailable only when final `valid_dates` is empty, and a real DA+ID
  figure requires re-running `PC_ADP_DA_ID` (Codex review 5; deterministic rule
  closed in bounded Round-9 review).
- **A `CoverageAudit` with only `valid_dates` + counts.** The disjoint-and-covering
  partition needs `observed_dates`/`missing_dates`/`solver_failed_dates` as
  canonical sets (Codex review 5).
- **`currency_basis` without a `target_base_year`.** A 2024-real result could be
  discounted against a 2026 valuation; v1 pins equality with
  `ValuationCase.base_year` (Codex review 5).
- **"e.g." simulation bounds and an unpinned fingerprint.** Locked to
  `[1000, 50000]` default 5000 and a canonical-serialization SHA-256; a locked
  contract carries no example constants or non-deterministic hash (Codex review 5).
- **A lossy/ambiguous float fingerprint (`repr` or `%.12g`).** `%.12g` collides
  distinct CapEx (`…123.0`/`…124.0` → same string) and `repr`-vs-`%.12g` is two
  encodings; pinned to exact IEEE-754 bytes, one spec for both fingerprints (Codex
  review 6).
- **A "reserve-coverage identity" named but never defined.** §5 referenced it with
  no schema field; added `ReserveCoverageAudit` (required/present/missing 4-hour
  blocks) with a full-coverage rule (Codex review 6).
- **Unbounded reserve `availability`.** It multiplies straight into reserve cash
  (`dispatch.py:429`), so `availability=2` doubled revenue and passed; pinned to
  finite `[0,1]`, default 0.95 (Codex review 6).
- **A `ReserveCoverageAudit` that only checks `missing == ∅`.**
  `required=6, present=5, missing=∅` could pass; v1 requires the per-day partition
  `present ∪ missing == required`, disjoint, derived from raw blocks **before**
  scalar collapse/zero-fill (Codex review 7).
- **Counting an incomplete calendar day as a full observation.**
  `solve_joint_capacity_batch` accepts a clean 12-hour edge day
  (`dispatch.py:490` rejects only NaNs), which would bootstrap as one full day; v1
  requires complete-regular-day classification, else `missing_dates` (Codex review 7).
- **An under-specified canonical serialization.** Listing sort/float rules still
  let two implementations frame bytes differently (or collide `["a","b"]` vs
  `["ab"]`); v1 pins a complete CBOR-framed profile with `schema_version`
  (Codex review 7).
- **Claiming RFC 8949 §4.2 *core* deterministic CBOR while forcing 64-bit
  floats.** RFC core deterministic mandates shortest-float (§4.2.1), which
  differs from the chosen uniform-f64 local profile; and CBOR has no native set, so the dropped
  "sets as sorted arrays" rule reopened set-hash divergence. v1 defines its own
  named `PC-CBOR-F64-v1` profile (not claiming RFC core) + golden vectors, and
  restores sorted-array sets (Codex review 8).
- **Delegating day-completeness to `_is_regular_utc_day`.** It infers cadence via
  `np.diff` (`simulation.py:848`), so a 12-row every-2-hours day passes; v1
  requires an expected-grid exact match from `(zone, date, resolution registry)`,
  `observed_dates == evaluation_dates`, and a pinned classification order
  (Codex review 8).
- **Leaving the DA-only cash basis at `carry_soc=True`.** Cross-day SoC carry
  breaks the independent-daily-draw bootstrap; v1 pins `carry_soc=False` and the
  concrete replay settings (Codex review 8).
- **A `floor_comparator?` field inside `RunResult`.** It gave the excluded floor
  an implicit home with no owner and no fingerprint entry; moved to a separate
  `FloorComparatorResult` (Codex review 8).
- **Leaving a native-CBOR integer input unbounded.** `seed > 2^64−1` had no
  `PC-CBOR-F64-v1` representation; v1 bounds it to uint64 and rejects bool/float
  aliases (Codex review 9).
- **Saying adapter inputs are "pinned" without values.** DA-only now fixes
  `soc_init_frac=0.5`; reserve co-opt fixes the same reset plus its price window
  and aggregation basis (Codex review 9).
- **Collapsing reserve prices before the coverage gate.** A partially missing date
  could be excluded from cash but still skew the scalar applied to good dates; v1
  builds DA-complete ∩ reserve-complete `pricing_dates` first and uses explicit
  block durations rather than adjacent surviving timestamp gaps (bounded Round 9
  cash-basis audit).
- **A boolean-only haircut audit.** "Capture applied" did not distinguish rates
  `0.3` and `1.0`; v1 fingerprints `{applied, rate, source}` and forbids a post-hoc
  haircut on the joint DA+reserve total (bounded Round 9 cash-basis audit).
- **Claiming a complete fingerprint without a total schema mapping or vectors.**
  v1 now pins the envelope, schema-driven float/int/bool/null/text/set/sequence
  normalisation, sample-window form, embedded encoder vectors, and the only
  accepted bootstrap algorithm literal (bounded Round 9 fingerprint audit).
- **Endogenous fade → augmentation / a fade-driven revenue multiplier.**
  Deferred to v1.1; v1 augmentation is a dated schedule with a declared
  maintenance basis.

## 9. Increments (each a dual-reviewed PR)

- **PC-A** — typed schema (`AssetCase`/`LifecycleCase`/`MarketCase`/
  `ValuationCase`/`BootstrapCase`/`ProjectCase`/`RunResult`) + public
  `StrategyKind` + the pinned `ProducerAdapterId` 5-tuple **solver adapters**
  emitting `StrategyRunResult` (with the `CoverageAudit` date-set partition, per-leg
  `ForecastAudit`, `currency_basis` incl. `target_base_year`) + `validate()` +
  a deterministic canonical-serialization fingerprint. Pure, no UI. Pins:
  producer-typed eligibility + real per-day field names + `degradation_cost_eur`
  exclusion + no-relabel/degraded-reserve→`missing_dates`, canonical zone/timezone,
  the per-kind forecast/reserve null matrix + deadband, walk-forward gate, three-state
  `capacity_maintenance_basis` + schedule consistency (source/as-of required;
  `UNKNOWN`→typed lifecycle unavailable), augmentation admissibility, the audit
  date-set partition (disjoint + covering) + finite/unique expected-grid
  classification on every consumed leg, the versioned leg-specific market
  grid registry (DA/IDA mixed-cadence and no-inference coverage), `ReserveCoverageAudit`
  per-day block partition (`present ∪ missing == required`, derived before fill) +
  full-coverage for reserve days, reserve scalar over pre-gated `pricing_dates`
  using explicit block durations, structured capture/liquidity cash basis +
  exact embedded VOM constant,
  `availability ∈ [0,1]`,
  `currency_basis.target_base_year == base_year`, all-input domains (concrete
  constants), fail-closed, engineering match, floor-not-consumed, decay-mode gate,
  immutable tuple series, and the schema-normalised `PC-CBOR-F64-v1` fingerprint
  with embedded + full-object golden vectors (§4.8; one spec for both fingerprints).
- **PC-B** — bootstrap + lifecycle cash-flow + two typed `NpvOutcome` slots (pure
  calc; distributions when available). Pins: `UNKNOWN` screening-only partial
  availability, no-shadow-wear, gross basis, VOM-once, no MW re-scale, **screening
  excludes O&M**, the per-draw `lifecycle − screening` identity, year-by-year
  discounting for both NPVs (annuity factor only in the parity kernel), projection
  multipliers (year 1 = 1.0, finite ≥ 0, full-length), event-salvage-in-event-year
  vs terminal-year residual/decommissioning, all-zero/negative series valid,
  input-domain rejection, and the narrow **DA-only kernel** screening-NPV parity
  (§3), using the single accepted `pc-bootstrap-pcg64-choice365-linear-v1`
  algorithm and its golden vector.
- **PC-C** — Revenue-tab/cockpit UI + Excel export (RunResult-driven,
  self-documenting assumptions sheet; floor rendered as separate comparator).
- **PC-D (v1.1)** — `ContractCase.settlement_basis` gross floor composition,
  making the floor cash-NPV-eligible without wear contamination.

## 10. Selected v1 decisions and review ledger

1. **Floor** — *resolved for v1:* comparator-only **and removed from the aggregator** in
   v1 (§4.5); gross settlement basis deferred to PC-D. (Codex round-1 resolution.)
2. **Discount basis** — *resolved for v1:* **real** rate, explicit **base-year EUR**,
   decay + discount both real (§4.4).
3. **Default strategy** — *resolved for v1:* DA-only realised when only DA is loaded;
   **fail-closed** on strategy invalidation, no silent fallback (§4.6).
4. **Augmentation input** — *resolved for v1:* a **CSV template import + validation
   preview** (import-first playbook), not dynamic sidebar rows.
5. **OpEx scope** — *resolved for v1:* v1 adds **fixed O&M only**; dispatch VOM is
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

Resolved in Codex review round 4 (contract closure):

16. **Screening O&M leak** — the screening NPV zeroes `opex`/`augment`/`terminal`
    (revenue − CapEx only); O&M is lifecycle-only; the per-draw
    `lifecycle − screening` identity is pinned (§3, §6).
17. **CoverageAudit date identities** — `valid_dates` added; the count identity
    and series==valid_dates equality are now implementable (§4.3, §4.6).
18. **Bootstrap ownership + full domains** — new `BootstrapCase` (typed
    `seed`/`n_simulations`/algo version); `power`/`duration`/`RTE`/decay/
    scenario/life domains closed (§4.8, red-line #15).
19. **Producer→field binding** — each `StrategyKind` binds a fixed
    `ProducerAdapterId` 5-tuple (adapter, kind, function, per-day field, excluded
    fields); a degraded reserve→0 run cannot be relabelled `DA_ID_RESERVE_REALISED`;
    per-leg `ForecastAudit` recorded (§5, §4.3).
20. **Lifecycle cross-fields** — `SCHEDULED_NAMEPLATE_MAINTENANCE` needs a
    non-empty positive-restoration schedule, `NO_AUGMENTATION_REQUIRED_ASSERTED`
    an empty one; `eol_residual_value_eur` naming unified with §6; all-zero/
    negative series valid; COD = valuation date in v1 (§4.2, §4.4).
21. **Currency basis** — `StrategyRunResult.currency_basis` records the applied
    deflator or the explicit "source EUR treated as base-year real" assumption;
    no silent mix (§4.3, §4.4).

Resolved in Codex review round 5 (enforceability closure — dual co-review of
`629088d` by both Codex passes; Gemini APPROVE'd but the field-name defects were
real, so they were fixed):

22. **Producer table matches code** — real per-day columns (`total_revenue_eur` /
    `realised_eur` / `realised_eur` / `joint_total_revenue`), `ProducerAdapterId`
    column, explicit `degradation_cost_eur`/ceiling exclusions, pinned DA-only call
    settings; the parity claim is a pure NPV-kernel test independent of the adapter
    (§3, §5).
23. **Reserve degradation is not a relabel** — a missing-reserve day is
    `missing_dates`; only a final empty `valid_dates` set makes the result
    unavailable (with a reserve-coverage identity separating missing data from a
    valid zero price and from an optimiser 0 MW). It is never re-emitted as
    `DA_ID_FORECAST` (§5; deterministic branch closed in bounded Round-9 review).
24. **Audit is a verifiable partition** — `observed`/`valid`/`missing`/
    `solver_failed` are canonical date sets that partition `observed_dates`, not
    counts (§4.3, §4.6).
25. **Currency + valuation base year pinned** — `target_base_year ==
    ValuationCase.base_year`, deflator `factor > 0` (§4.3, §4.4).
26. **Concrete constants + deterministic fingerprint** — `n_simulations ∈
    [1000, 50000]` default 5000, `MAX_PROJECT_LIFE_YEARS = 100`, `scenario_count`
    N/A in v1; SHA-256 over a canonical serialization (§4.8).

Resolved in Codex review round 6 (final enforceability closure — dual co-review
of `7baf26c`: Gemini APPROVE, Codex CHANGES REQUESTED on three real gaps, two of
them defects introduced by the round-5 edits):

27. **Lossless fingerprint** — floats are exact IEEE-754 8-byte encoding, not the
    ambiguous/lossy `repr`/`%.12g` (which collided distinct CapEx); one spec shared
    by `RunResult.input_fingerprint` and the nested
    `StrategyRunResult.fingerprint` (§4.8).
28. **`ReserveCoverageAudit` is a real field** — required/present/missing 4-hour
    block sets; a reserve day is cash-valid only when fully covered, so a
    zero-filled missing block cannot pass as a valid low-reserve day (§4.3, §5).
29. **Reserve `availability` bounded** — finite `[0,1]` (default 0.95); it
    multiplies into reserve cash (`dispatch.py:429`), so an out-of-range value can
    no longer inflate revenue past validation (§4.3, red-line #15).
30. **Wording/citation nits** — the audit is `observed_dates` (universe) + a
    three-set partition; the replay-defaults citation is `simulation.py:379-392`.

Resolved in Codex review round 7 (deep enforceability — dual co-review of
`d39e8cb`: Gemini APPROVE, Codex CHANGES REQUESTED on three deeper gaps):

31. **`ReserveCoverageAudit` is a checkable per-day partition** — `present ∪
    missing == required` + disjoint, derived from raw blocks **before** scalar
    collapse/zero-fill (a `required=6/present=5/missing=∅` inconsistency can no
    longer pass); a day is cash-valid only when `present == required` (§4.3, §5).
32. **Incomplete days are `missing_dates`** — a day is `valid` only if it is a
    complete regular local calendar day; a clean 12-hour edge day that
    `solve_joint_capacity_batch` would accept (`dispatch.py:490`) is not
    bootstrapped as a full observation (§4.3).
33. **Complete canonical serialization** — pinned to a CBOR-framed profile with
    float64 always 64-bit and a `schema_version` wrapper (refined in round 8 to the
    named `PC-CBOR-F64-v1`, §4.8).
34. **Tuple-arity reconciliation** — §8/§10 now say the authoritative §5 5-tuple.

Resolved in Codex review round 8 (narrow contract closure — dual co-review of
`94241a5`: Gemini APPROVE, the user's Codex CHANGES REQUESTED on four items that
still change cash results or audit validity, plus one minor reserve-forecast-leg
addition — five resolutions, 35–39):

35. **Fingerprint self-contradiction removed** — RFC 8949 core deterministic
    mandates shortest-float, conflicting with the forced 64-bit; v1 now names its
    own `PC-CBOR-F64-v1` profile (no RFC-core claim), restores sets-as-sorted-arrays
    (CBOR has no native set), and requires golden vectors (§4.8).
36. **Completeness from an expected grid** — `_is_regular_utc_day` infers cadence
    and passes a 12-row every-2-hours day; v1 requires an expected-grid exact match
    from `(zone, date, resolution registry)`, `observed_dates == evaluation_dates`,
    and a pinned data-gates-before-solver classification order (§4.3).
37. **DA-only `carry_soc=False` pinned** — cross-day SoC carry breaks the
    independent-daily-draw bootstrap; concrete replay settings + capture-haircut
    owner are pinned (§5).
38. **Floor out of `RunResult`** — the wear-net comparator moves to a separate
    `FloorComparatorResult`; `RunResult` neither contains nor fingerprints it (§4.5).
39. **Reserve-price forecast leg audited** — `ForecastAudit` covers the DA, IDA,
    and reserve-price legs separately for `DA_ID_RESERVE_REALISED` (§4.3).

Resolved in review round 9 (bounded co-review of `6b217e7`: Gemini **APPROVE**,
Codex **CHANGES REQUESTED** on four §11 blockers; the bounded takeover audit then
closed the additional cash-basis/fingerprint and final-bar groups before this
candidate commit):

40. **One fingerprint name everywhere** — §9 no longer retained the stale
    `canonical-CBOR (RFC 8949 §4.2)` phrase; every reference names the local
    `PC-CBOR-F64-v1` profile (§4.8, §9).
41. **Native integer domain is total** — `seed ∈ [0,2^64−1]`, bool/float aliases
    rejected, so every legal seed has a native CBOR unsigned-int representation
    (§4.8).
42. **Daily reset SoC is concrete** — every v1 producer records
    `carry_soc=False` and `soc_init_frac=0.5`; the DA-only adapter passes both
    explicitly, and no producer delegates a cash-changing value to an ambient
    solver default (§4.8, §5).
43. **Reserve co-opt basis is concrete and pre-gated** — price aggregation,
    coverage-complete `pricing_dates`, explicit product-block durations, solver
    window, `soc_init_frac=0.5`, and no post-hoc capture haircut are all
    fingerprinted. Missing-date remnants cannot skew retained-date cash (§5).
44. **Haircut values have an owner** — `cash_basis` fingerprints structured
    capture `{applied, rate, source}` and liquidity provenance; v1 non-DA adapters
    explicitly emit capture `1.0/not_applied` (§4.3, §5).
45. **Fingerprint and bootstrap are executable contracts** — the local profile
    now pins its envelope and schema normalisation, embeds two exact CBOR/SHA-256
    encoder vectors, bounds base-year integers, defines the delivery-date window,
    and accepts one versioned PCG64/365-day/linear-percentile bootstrap algorithm
    (§4.3, §4.8).
46. **Round-9 editorial closure** — RunResult's exclusive UI/export wording is
    scoped to ProjectCase NPV outputs, and the round-8 ledger correctly counts the
    separate reserve-forecast audit addition (§4.6, §10).
47. **Production wire schema is closed** — exact recursive payload keys/nesting,
    optional-null rules, nested StrategyRunResult digest boundary, human/derived/
    transient exclusions, cash/currency bases, and adapter/audit submaps are now a
    normative registry; aliases cannot produce implementation-dependent hashes
    (§4.8; bounded Round-9 fingerprint audit).
48. **`R9-FINAL-01` — zone/timezone binding** — disposition:
    **blocking → resolved** (eligible-date basis). Supported zone codes now resolve
    exactly to `config.ZONE_TIMEZONES[zone]`; caller-selected IANA zones are
    invalid. Target: PC-A zone/timezone mismatch + DST date-universe tests (§4.3,
    red-line #17). Owner acceptance: accepted (round-10 dual-approve lock, 2026-08-10).
49. **`R9-FINAL-02` — forecast tagged union/deadband** — disposition:
    **blocking → resolved** (cash output + public fingerprint). The per-kind
    required/null matrix, exact bucket literals, and finite non-negative
    `min_rebid_uplift_eur` binding are normative. Target: PC-A per-adapter matrix,
    invalid-domain, and fingerprint mutation tests (§4.3, §4.8).
    Owner acceptance: accepted (round-10 dual-approve lock, 2026-08-10).
50. **`R9-FINAL-03` — one reserve-failure branch** — disposition:
    **blocking → resolved** (eligible dates + availability). Every per-day raw
    coverage failure is `missing_dates`; whole-result unavailable occurs only when
    final `valid_dates` is empty. Target: PC-A partial-day vs all-day failure tests
    (§4.3, §5). Owner acceptance: accepted (round-10 dual-approve lock, 2026-08-10).
51. **`R9-FINAL-04` — typed partial NPV availability** — disposition:
    **blocking → resolved** (public RunResult schema + availability). Both metrics
    use `NpvOutcome`; `UNKNOWN` keeps screening available and returns the fixed
    lifecycle-unavailable envelope. Target: PC-B schema/UNKNOWN/UI-branch tests
    (§3, §4.6). Owner acceptance: accepted (round-10 dual-approve lock, 2026-08-10).
52. **`R9-FINAL-05` — fingerprint-visible scalar/literal closure** — disposition:
    **blocking → resolved** (public fingerprint). Text/date wire types, projection
    union literals/nulls, currency mode, reserve aggregation literal, and adapter
    row values are pinned. Target: PC-A full-object golden vectors + union mutation
    tests (§4.8). Owner acceptance: accepted (round-10 dual-approve lock, 2026-08-10).
53. **`R9-FINAL-06` — VOM provenance must be true** — disposition:
    **blocking → resolved** (cash-basis red-line). Every adapter validates
    `post_vom=true` and the embedded dispatch constant `0.5`. Target: PC-A wrong/
    missing-VOM rejection tests (§4.8, §5, red-line #7).
    Owner acceptance: accepted (round-10 dual-approve lock, 2026-08-10).
54. **`R9-FINAL-07` — finite unique expected-grid values** — disposition:
    **blocking → resolved** (eligible-date classification). Every consumed market
    leg must have exactly one finite value per required point before a solver can
    fail; duplicate/NaN/Inf is `missing_dates`. Target: PC-A per-leg duplicate/
    NaN/Inf/missing mutation tests (§4.3, red-line #17).
    Owner acceptance: accepted (round-10 dual-approve lock, 2026-08-10).
55. **`R9-FINAL-08` — expected grids are market-leg-specific** — disposition:
    **blocking → resolved** (eligible dates). DA and IDA require separate explicit
    `(leg, zone, delivery_date)` registry calendars; reserve uses its product-block
    calendar. Registry version/profile IDs enter adapter provenance, and any
    unsupported leg/zone/date makes the adapter unavailable rather than invoking
    cadence inference or IE/CH/GB fallback. Target: PC-A mixed DA/IDA cadence,
    missing-registry, and fingerprint-mutation tests (§4.3, §4.8, red-line #17).
    Owner acceptance: accepted (round-10 dual-approve lock, 2026-08-10).

Resolved in review round 10 (final same-hash co-review of `230b8c0`: Codex
**CHANGES REQUESTED** on two bar-(c) fingerprint blockers, Gemini **CHANGES
REQUESTED** on three schema items + one non-blocking; CC independently verified each
against the §11 bar, confirmed the two fingerprint blockers, and folded the
editorial reconciliations into this candidate):

56. **`R10-01` — reserve-coverage entry-date domain** — disposition:
    **blocking → resolved** (public fingerprint). `reserve_coverage_audit` is a
    sorted array with **exactly one entry per `observed_dates` member** (entry
    date-set == `observed_dates`; a fully-missing day retained with `present=∅`,
    `missing=required`); `null` for non-reserve kinds. Closes the Codex
    unpinned-membership fork (array membership could otherwise diverge across
    conformant encoders) (§4.3, §4.8, red-line #21).
57. **`R10-02` — `adapter_provenance.capture_rate` null-matrix** — disposition:
    **blocking → resolved** (public fingerprint). Non-null **only** for
    `PC_ADP_DA_ONLY` (== `cash_basis.capture.rate` == the value passed to
    `simulate_replay_batch`); `null` for the other three, whose `cash_basis.capture`
    stays `{applied:false, rate:1.0, source:"not_applied"}`. `cash_basis.capture`
    remains the sole cash-basis owner. Closes the Gemini null-vs-1.0 ambiguity
    (§4.8, red-line #22).
58. **`R10-03` — `prob_positive` pinned** — disposition:
    **output number → resolved.** `prob_positive = mean(npv_draws > 0.0)` over
    **all** `n_simulations` draws, no draw dropped; `p10/p50/p90 =
    percentile(npv_draws, {10,50,90}, method="linear")` (§3, §6, §4.8).
59. **`R10-04` — `solver_failure_details` fingerprint membership** — disposition:
    **public fingerprint → resolved.** Exactly one `{date, status, message, stage}`
    per `solver_failed_dates` member (date-set equality; empty array when none),
    sorted by `(date, stage, status, message)` (§4.3, §4.8).
60. **`R10-05` — currency-basis wire reconciliation** — disposition:
    **non-blocking (editorial) → applied.** §4.3/§4.4 prose now uses the flat
    `{mode, deflator_method, deflator_vintage, deflator_factor}` wire form of §4.8,
    not the nested `{method, vintage, factor}` notation (Gemini).
61. **`R10-06` — reserve-audit shape reconciliation** — disposition:
    **non-blocking (editorial) → applied.** §4.3 now describes `ReserveCoverageAudit`
    as an explicit sorted array (was "keyed per local date", which read as a map),
    consistent with §4.8 (Gemini; folded with `R10-01`).
62. **`R10-07` — per-draw bootstrap read** — disposition:
    **non-blocking (clarity) → applied.** §6 reads the per-draw
    `bootstrap_annual_revenue(...)["simulations"]` array, not the percentile
    summary, for `annual_revenue_draw_d`.
63. **`R10-08` — cashflow-table schema is PC-C scope** — disposition:
    **non-blocking (PC-C implementation debt).** The NPV *numbers* are pinned by §6
    + the §3 `NpvOutcome`/distribution shape + the §4.8 bootstrap golden vector; the
    `screening_cashflow_table` / `lifecycle_cashflow_table` row/column layout is
    PC-C UI/export presentation and is not input-fingerprinted (Codex).

## 11. Lock exit rule and procedure

To converge (Gemini has APPROVE'd five consecutive hashes while Codex refines
detail), the final same-hash co-review applies an explicit **blocking bar** — a
finding blocks the lock **only** if it would change any of:

- an **output number** or the **cash basis** (revenue/opex/augmentation/residual
  discounting, double-counting, non-additivity);
- the set of **eligible dates** or a result's **available/unavailable** status;
- the **public schema, `StrategyKind`, or the fingerprint format**;
- a **red-line** (§7).

Everything else is recorded as **PC-A/PC-B implementation debt, non-blocking** for
the contract lock (and handled inside those dual-reviewed PRs): helper placement,
library choice (e.g. which CBOR encoder), performance, logging/UI copy, test-fixture
design, and docstring wording.

If a reviewer still returns `CHANGES REQUESTED`, every finding must be recorded
before lock with: `finding_id`, disposition (`blocking` or
`implementation_debt`), rationale against the bar above, target increment/test,
and explicit project-owner acceptance. A generic "only implementation detail"
statement is not an auditable disposition.

**Lock procedure** (only when one candidate commit gets a dual APPROVE, or every
remaining finding has the accepted non-blocking disposition above): add the
review metadata

```
reviewed_candidate_commit: <full 40-character git SHA>
lock_basis: dual_approve | owner_accepted_nonblocking
locked_on: YYYY-MM-DD
```

then make one **metadata-only** lock commit. Its diff from
`reviewed_candidate_commit` may change only `Status: candidate → locked`, `Draft
decision date → Locked on`, the three review pointers above, and the per-finding
review-disposition ledger; no schema, formula, red-line, decision, or acceptance
criterion may change. Open the docs-only PR after verifying that restricted diff.
Implementation (PC-A first) starts only after that PR merges.
