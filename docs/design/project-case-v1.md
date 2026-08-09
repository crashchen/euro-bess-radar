# Project Case v1 — unified pre-tax unlevered lifecycle cash NPV

Status: **candidate** (design-contract round — NOT locked. Revised after Codex
artifact review rounds 1–5 and a Gemini adversarial review. Round 5 closed the
enforceability gaps two independent Codex passes raised on `629088d` (Gemini
APPROVE'd that hash, but the field-name defects were real). Pending a fresh
co-review of THIS commit by both Codex and Gemini — only a dual APPROVE on the
same hash flips `Status: locked`.)

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
fingerprinted** `RunResult`.

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
year `project_life_years`**. The two are never conflated. `capacity_restored_frac` is a fraction of **nameplate
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
- `cash_basis` — an explicit statement that the series is **post-VOM** and
  whether the **capture haircut** and **liquidity cap** are already applied
  (so PC never re-applies either);
- engineering binding: `power_mw`, `duration_hours`, `round_trip_efficiency`;
- market binding: `zone`, `sample_window`; a **`currency_basis`** with a
  `target_base_year` (which `validate()` requires to equal `ValuationCase.base_year`
  — a mismatch is fail-closed, §4.4) and **either** a recorded `deflator`
  (`{method, vintage, factor}` with `factor` finite `> 0`) the adapter applied to
  convert historical settlement EUR to base-year real EUR, **or** the explicit
  screening assumption `source EUR treated as base-year real` — never a silent mix;
- kind-specific provenance: a **`ForecastAudit`** per forecast leg (DA and IDA
  separately, not one `walk_forward` label) carrying `forecast_mode` + bucket +
  deadband; `reserve_product` + `reserve_source` + `availability`; (scenario
  provenance does not apply — no v1-eligible strategy is the stochastic batch, §5);
- a **`CoverageAudit`** (per `dispatch-failure-contract-v2`): three **immutable
  canonical date sets** — `observed_dates`, `valid_dates` (the local dates that
  actually produced a cash value), `missing_dates`, `solver_failed_dates` — plus
  per-day failure details. The audit is machine-checkable:
  `valid_dates ∪ missing_dates ∪ solver_failed_dates == observed_dates`, the three
  are **pairwise disjoint**, and `daily_realised_cash_series` dates equal
  `valid_dates` exactly (counts alone cannot detect an overlapping mis-assignment).
  An all-failure or empty-`valid_dates` result is *unavailable*, never a €0 series;
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
convention. The historical StrategyRunResult cash is reconciled to this basis via
the `StrategyRunResult.currency_basis` (§4.3): either the adapter applied a
recorded `deflator` (`{method, vintage, factor > 0}`), or the explicit screening
assumption "source EUR treated as base-year real" is stamped — a silent mix is
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
and export read from.

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
"e.g."): `seed` (a **non-negative int**; `bool`/`None`/float rejected — `True`
must not slip in as `1`); `n_simulations` (a positive int in
`[MIN_SIMULATIONS, MAX_SIMULATIONS] = [1000, 50000]`, **default 5000** to match
the shipped `scenario.bootstrap_annual_revenue`); `bootstrap_algorithm_version`.
All three enter the `RunResult` fingerprint (§4.6).

**Remaining numeric domains (Codex review 4).** `validate()` also bounds every
other live scalar so red-line #15 ("all numeric inputs bounded") is real, not
aspirational: `AssetCase.power_mw > 0`, `duration_hours > 0`,
`0 < round_trip_efficiency ≤ 1`; `DAOnlySpreadDecay` `annual_decay_rate ∈ [0, 1)`
and `decay_floor_share ∈ [0, 1]` (the `spread-decay-v1` domains). No v1-eligible
strategy is the stochastic batch (§5), so **`scenario_count` is not a live domain
in v1** — the earlier "stochastic cap" is moot. A NaN/Inf/out-of-range value in
any live scalar is rejected before §6.

**Deterministic fingerprint (Codex review 5).** The `RunResult.input_fingerprint`
is a `schema_version`-prefixed **SHA-256** over a **canonical serialization** so
two implementations agree byte-for-byte on the same inputs: keys sorted, dates as
ISO-8601 `YYYY-MM-DD` strings, enums by **name**, floats via a fixed
`repr`/`%.12g` formatting, sets serialized as sorted lists, and every case
included in a fixed order (Asset, Lifecycle, Market+StrategyRunResult fingerprint,
Valuation, Bootstrap). Nothing outside this canonical form (dict insertion order,
pandas index identity, object ids) may influence the hash.

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
this open.

**DA-only parity/reproducibility settings (Codex review 5):** `simulate_replay_batch`
defaults `mode="DA MILP Replay"`, `soc_init_frac=0.5`, `carry_soc=True`
(`simulation.py:23-33`) — which is a *different* daily basis from the shipped
Revenue tab's standalone ordered-spread days. The `DA_ONLY` adapter therefore
**pins its call settings** (recorded in provenance) so its series is deterministic
and reproducible; it does **not** attempt to equal the tab's ordered-spread series
(the parity claim is a pure NPV-kernel test, §3, not "adapter reproduces tab").

`DA_ID_FORECAST` / `DA_ID_RESERVE_REALISED` are **walk-forward only** and record a
per-leg `ForecastAudit` (§4.3). Two adapter rules close the last gap:

- **No relabelling — degraded reserve fails, it does not become another kind.**
  A degraded reserve-first run must distinguish three cases: (a) reserve
  data missing/out-of-window (`simulation.py:~1024/1236` aligns missing blocks to
  zero), (b) a valid published zero reserve price, (c) the optimiser choosing
  `0 MW` on valid data. Only (b)/(c) are legitimate `DA_ID_RESERVE_REALISED`
  results and must carry a **reserve-coverage identity** proving the reserve data
  covered the window. Case (a) makes the day **excluded** (dropped from
  `valid_dates`) or the whole result **unavailable** — it is **never** re-emitted
  as `DA_ID_FORECAST` (that kind is bound exclusively to `PC_ADP_DA_ID`; relabel
  is forbidden). Producing a genuine DA+ID result requires **re-running**
  `PC_ADP_DA_ID`, not relabelling.
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
`−installed_capex_eur + Σ_{t=1..L} net_{d,t} / (1+discount_rate)^t`, with
`net_{d,t} = revenue_{d,t} − opex_t − augment_t + terminal_t`; over all draws this
yields the P10/P50/P90/`P(NPV>0)` distribution.

`no_lifecycle_cost_screening_npv` is the **same** year-by-year sum with **`opex_t
= augment_t = terminal_t = 0`** (revenue and CapEx only — §3.1 defines it as
revenue minus CapEx; fixed O&M is a lifecycle cash cost and must NOT appear in
the screening figure, Codex review 4). Both use the same bootstrap draws, the
same `projection_multiplier_t`, the same horizon, and the same year-by-year
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
   the lifecycle NPV **unavailable**, never a warning-and-proceed. Augmentation
   events need year + cost + restoration (frac of nameplate) + residual, or are
   rejected.
5. **Floor comparator-only** — not in the aggregator; ProjectCase never reads
   `floor_protected_cashflow_eur` / `_pv_eur`.
6. **Gross, producer-issued revenue** — eligibility is a `StrategyKind`/adapter
   property; ceilings, deltas, overlays, gross-additive references, benchmarks,
   and unavailable results are unrepresentable as cash revenue. The `DA_ONLY`
   adapter reads only `total_revenue_eur`, never the co-exposed
   `degradation_cost_eur` (shadow-wear ingress, §5).
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
15. **All numeric inputs bounded (concrete constants)** — event costs/residuals/
    decommissioning/O&M/CapEx finite ≥ 0, multipliers finite ≥ 0 with year 1 = 1.0
    and full-length, `capacity_restored_frac ∈ [0,1]`, `power_mw/duration_hours > 0`,
    `0 < RTE ≤ 1`, decay `d ∈ [0,1)` / floor `∈ [0,1]`, `seed` a non-negative int,
    `n_simulations ∈ [1000, 50000]` (default 5000), `project_life_years ∈
    [1, MAX_PROJECT_LIFE_YEARS=100]`; a NaN/Inf/out-of-range input is rejected
    before §6 (§4.2, §4.7, §4.8). No "e.g." constants survive in the locked text.
16. **Screening excludes O&M** — `no_lifecycle_cost_screening_npv` is revenue −
    CapEx only (`opex = augment = terminal = 0`); fixed O&M is a lifecycle cash
    cost and appears only in `lifecycle_cash_npv` (§3, §6).
17. **Audit carries a date-set partition** — `observed_dates` / `valid_dates` /
    `missing_dates` / `solver_failed_dates` are canonical sets; they partition
    `observed_dates` (disjoint + covering) and the series equals `valid_dates`; a
    zero or negative cash value is valid, never a failure signal (§4.3, §4.6).
18. **One adapter, one field** — each `StrategyKind` binds a fixed
    (`ProducerAdapterId`, source function, per-day cash field, excluded fields)
    tuple with real column names; a degraded reserve→0 run is excluded or
    unavailable, never relabelled to another kind (§5).
19. **Currency basis matches valuation** — `currency_basis.target_base_year ==
    ValuationCase.base_year` (fail-closed); deflator `factor` finite `> 0` (§4.3,
    §4.4).
20. **Deterministic fingerprint** — `schema_version`-prefixed SHA-256 over a
    canonical serialization (sorted keys, ISO dates, enums by name, fixed float
    format, sorted sets, fixed case order); nothing else influences the hash
    (§4.8).

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
  + ceiling + delta; each kind now binds a fixed (adapter, function, field)
  4-tuple and a degraded reserve→0 run cannot be relabelled (Codex review 4).
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
  another producer's kind; a degraded reserve day is excluded or the result is
  unavailable, and a real DA+ID figure requires re-running `PC_ADP_DA_ID`
  (Codex review 5).
- **A `CoverageAudit` with only `valid_dates` + counts.** The disjoint-and-covering
  partition needs `observed_dates`/`missing_dates`/`solver_failed_dates` as
  canonical sets (Codex review 5).
- **`currency_basis` without a `target_base_year`.** A 2024-real result could be
  discounted against a 2026 valuation; v1 pins equality with
  `ValuationCase.base_year` (Codex review 5).
- **"e.g." simulation bounds and an unpinned fingerprint.** Locked to
  `[1000, 50000]` default 5000 and a canonical-serialization SHA-256; a locked
  contract carries no example constants or non-deterministic hash (Codex review 5).
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
  exclusion + no-relabel/degraded-reserve-excluded, walk-forward gate, three-state
  `capacity_maintenance_basis` + schedule consistency (source/as-of required;
  `UNKNOWN`→unavailable), augmentation admissibility, the audit date-set partition
  (disjoint + covering), `currency_basis.target_base_year == base_year`, all-input
  domains (concrete constants), fail-closed, engineering match, floor-not-consumed,
  decay-mode gate, immutable tuple series, fingerprint determinism.
- **PC-B** — bootstrap + lifecycle cash-flow + both NPV **distributions** (pure
  calc). Pins: no-shadow-wear, gross basis, VOM-once, no MW re-scale, **screening
  excludes O&M**, the per-draw `lifecycle − screening` identity, year-by-year
  discounting for both NPVs (annuity factor only in the parity kernel), projection
  multipliers (year 1 = 1.0, finite ≥ 0, full-length), event-salvage-in-event-year
  vs terminal-year residual/decommissioning, all-zero/negative series valid,
  input-domain rejection, and the narrow **DA-only kernel** screening-NPV parity
  (§3).
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

Resolved in Codex review round 4 (contract closure):

16. **Screening O&M leak** — the screening NPV zeroes `opex`/`augment`/`terminal`
    (revenue − CapEx only); O&M is lifecycle-only; the per-draw
    `lifecycle − screening` identity is pinned (§3, §6).
17. **CoverageAudit date identities** — `valid_dates` added; the count identity
    and series==valid_dates equality are now implementable (§4.3, §4.6).
18. **Bootstrap ownership + full domains** — new `BootstrapCase` (typed
    `seed`/`n_simulations`/algo version); `power`/`duration`/`RTE`/decay/
    scenario/life domains closed (§4.8, red-line #15).
19. **Producer→field binding** — each `StrategyKind` binds a fixed (adapter,
    function, field) 4-tuple; a degraded reserve→0 run cannot be relabelled
    `DA_ID_RESERVE_REALISED`; per-leg `ForecastAudit` recorded (§5, §4.3).
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
23. **Reserve degradation is not a relabel** — a missing-reserve day is excluded or
    the result is unavailable (with a reserve-coverage identity separating
    missing-data from a valid zero price and from an optimiser 0 MW); it is never
    re-emitted as `DA_ID_FORECAST` (§5).
24. **Audit is a verifiable partition** — `observed`/`valid`/`missing`/
    `solver_failed` are canonical date sets that partition `observed_dates`, not
    counts (§4.3, §4.6).
25. **Currency + valuation base year pinned** — `target_base_year ==
    ValuationCase.base_year`, deflator `factor > 0` (§4.3, §4.4).
26. **Concrete constants + deterministic fingerprint** — `n_simulations ∈
    [1000, 50000]` default 5000, `MAX_PROJECT_LIFE_YEARS = 100`, `scenario_count`
    N/A in v1; SHA-256 over a canonical serialization (§4.8).
