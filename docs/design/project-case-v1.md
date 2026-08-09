# Project Case v1 — unified pre-tax unlevered lifecycle cash NPV

Status: **candidate** (design-contract round — NOT locked. Revised after Codex
artifact review round 1 (`CHANGES REQUESTED`); pending Gemini adversarial review
then a final Codex diff review before `Status: locked`.)

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
replacement/residual schedule. Report a `Legacy screening NPV` and a `Pre-tax
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
`scenario.bootstrap_annual_revenue`, so the legacy figure reproduces the shipped
Revenue-tab NPV exactly:

1. **`Legacy screening NPV`** — the existing
   `scenario.calculate_npv_distribution(annual_revenue_dist, capex, …)` path with
   the shadow-wear cash deduction held at zero
   (`cash_npv_includes_shadow_wear=False`, per economic-semantics-v1) and no
   dated lifecycle events. Retained for continuity/parity.
2. **`Pre-tax unlevered lifecycle cash NPV`** — the same bootstrapped revenue
   annuity, minus upfront CapEx, minus cash fixed O&M, minus the **explicit
   dated** augmentation/replacement schedule, plus residual value, all
   discounted. No shadow wear. No tax. No debt.

We deliberately do **not** call figure 2 "bankable NPV." Without tax, debt
service, DSCR, and financing-fee models it would over-promise. economic-
semantics-v1 requires only that, once augmentation is explicit, both a
no-augmentation screening NPV and a fully-specified cash NPV are reported.

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
├── legacy_screening_npv        {p10,p50,p90,prob_positive}
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

Fields: `project_life_years`; an **augmentation/replacement schedule** = an
ordered list of dated events, each `{year, cost_eur, capacity_restored_frac,
residual_value_eur}`; an end-of-life `residual_value_eur`; and
`decommissioning_cost_eur` (a dated cash outflow — LifecycleCase is the **sole
owner** of residual, disposal, and decommissioning; no other case models them).

Units are pinned: `capacity_restored_frac` is a fraction of **nameplate energy**
(0–1), never MWh — one unit only.

**Augmentation economic effect (v1, the economic-semantics red-line closes
here).** economic-semantics-v1 forbids an augmentation reserve without a
capacity-restoration effect. v1 makes the effect explicit via a stated
**premise**: the chosen strategy's revenue projection is taken to hold *because*
this augmentation schedule keeps the asset at nameplate over the modelled life.
Consequently:

- If the projection is `FlatRealProjection` (§4.7), the flat revenue is
  admissible **only** with a non-empty augmentation schedule sufficient to
  sustain nameplate; a flat projection with **no** augmentation is flagged
  `optimistic_flat_no_augmentation=True` in provenance (fade would erode it).
- The richer alternative — an explicit per-year capacity/revenue multiplier
  tied to a fade curve — is a v1.1 option (§10.1), not built in v1.

An event missing any of year / cost / restoration / residual is rejected at
`validate()`.

### 4.3 `MarketCase` and the producer-issued `StrategyRunResult`

`MarketCase` references **one** `StrategyRunResult` plus a projection mode
(§4.7). It does **not** carry a free-floating revenue scalar.

`StrategyRunResult` is a typed object **emitted only by a solver adapter**, never
constructed by the UI or by attaching a label to a number. Eligibility is a
property of *which producer emits it*, so an ineligible quantity (a ceiling, a
delta, an overlay) is **unrepresentable**, not merely rejected. It carries:

- `strategy_id` + human label (from the §5 allowlist);
- `daily_realised_cash_series` — per local calendar day, **gross** dispatch
  cash in EUR (economic-semantics layer 1: market revenue before shadow wear).
  This series — not a scalar — is what feeds the bootstrap (§3, §6);
- engineering binding: `power_mw`, `duration_hours`, `round_trip_efficiency`,
  `embedded_vom_eur_mwh`, `capture_rate`;
- market binding: `zone`, `sample_window`, `forecast_mode` (walk-forward for
  forecast-driven strategies — §5), `reserve_source` (where applicable);
- **solver audit** (per `dispatch-failure-contract-v2`): observed / valid /
  missing / solver-failed day counts; an all-failure or empty-valid result is
  *unavailable*, never a €0 series;
- a content `fingerprint` over all of the above.

`validate()` (§4.6) checks `StrategyRunResult.power_mw == AssetCase.power_mw`
(and duration/RTE), so the revenue and the asset describe the same battery.
The revenue is **gross** — this is what makes subtracting CapEx and augmentation
in cash NPV legal without double-counting: market revenue on the cash line,
capital counted once as capital.

### 4.4 `ValuationCase` — discounting (real, unlevered)

Fields: `discount_rate` interpreted as a **real** rate; `base_year` fixing the
EUR value basis; reuse of `scenario.decaying_annuity_pv_factor`. **Convention
lock:** revenue projection (§4.7) and discounting are both in **real, base-year
EUR** — decay and discount must not mix a nominal and a real convention. Cash
timing follows an explicit **end-of-year** convention anchored at a commercial-
operation-date (COD) / valuation date; year 0 = `−installed_capex_eur`.

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
(§4.3); `strategy_id` in the §5 allowlist and its StrategyRunResult
solver-available with a non-empty valid-day audit; projection mode compatible
with the strategy (§4.7); every augmentation `year ≤ project_life_years`; CapEx
matches the sidebar. **Fail-closed:** if the referenced strategy result is
invalid/unavailable, `validate()` raises — no silent fallback to a lesser
strategy or to €0 (`dispatch-failure-contract-v2`).

`RunResult` is immutable, carries a `schema_version`, is serialisable, and holds
an `input_fingerprint` over every case (so a stale result is detectable and
cache invalidation is deterministic — the frontier/floor fingerprint pattern).
It is the only object UI and export read from.

### 4.7 Projection modes (respecting `spread-decay-v1`)

`spread-decay-v1.md` (§lines 136–140) locks the decay to "the Revenue tab's
DA-arbitrage-based Monte-Carlo draws … no interaction with the cockpit strategy
rows." Applying one decay to a DA+reserve or DA+IDA+reserve total would violate
that locked contract. v1 therefore exposes three explicit modes:

- **`FlatRealProjection`** — no decay; available to every eligible strategy.
  (Augmentation-premise rules of §4.2 apply.)
- **`DAOnlySpreadDecay`** — the `spread-decay-v1` decayed factor; permitted
  **only** when `strategy_id == DA-only`. `validate()` rejects it for any other
  strategy.
- **`ExplicitAnnualCurve`** — a user-supplied, sourced per-year total-strategy
  revenue curve (its own provenance); no implicit decay model.

## 5. Cashflow-eligible strategies (producer allowlist / denylist)

Eligibility is enforced by *which adapter emits a StrategyRunResult*, not by a
consumer-attached label. The strategy-comparison table (`strategy_compare.py`)
mixes totals, ceilings, and deltas in one frame, so it is never a ProjectCase
input.

**Adapters emit `StrategyRunResult`** for these realised, internally-co-optimised
gross totals only:

- DA-only realised (`_DA_ONLY`).
- DA + IDA1 forecast-driven realised (`_FORECAST`) — **walk-forward only**.
- DA + reserve-capacity co-opt total (`_RESERVE_DEFAULT` /
  `solve_joint_capacity_batch` `joint_total_revenue`); the reserve fee is already
  inside this total via the joint MILP, so it is self-consistent.
- DA + IDA1 + reserve forecast-driven **realised** total (`_REALISTIC_DEFAULT` /
  stochastic triple realised) — **walk-forward only**.

**No adapter exists** (hence unrepresentable as cash revenue) for: any
perfect-foresight ceiling (`_CEILING`, `_TRIPLE_DEFAULT`, `coopt_ceiling*`); the
stochastic policy-value **delta** (`STOCHASTIC_POLICY_VALUE_LABEL` /
`…_RESERVE_LABEL`); activation/imbalance overlays; `gross_additive_total_eur`;
the external-trader benchmark; or any solver-unavailable / no-valid-day result.

**Forecast-mode rule:** forecast-driven strategies are cash-eligible **only in
`walk_forward`** mode. `in_sample` is diagnostics-only; `loo` (leave-one-out) is
an unbiased *skill* estimate that peeks at future days and is therefore too
optimistic for a cash figure — v1 rejects both for the cash NPV. (This is
slightly stricter than "reject in-sample only"; see §10.6 for the review.)

## 6. Lifecycle cash-flow construction

The chosen strategy's `daily_realised_cash_series` (scaled to `power_mw`) is
bootstrapped to an annual-revenue distribution via
`scenario.bootstrap_annual_revenue`. For draw `d` and year `t = 1 … life`:

```
revenue_{d,t} = annual_revenue_draw_d × projection_weight_t   # §4.7 (flat / DA-only decay / explicit)
opex_t        = fixed_om_eur_per_mw_yr × power_mw              # VOM already inside gross revenue
augment_t     = Σ scheduled augmentation/replacement cost in year t         (cash out, deterministic)
residual_t    = scheduled residual realised in year t − decommissioning in year t   (deterministic)
net_{d,t}     = revenue_{d,t} − opex_t − augment_t + residual_t
```

Year 0 = `−installed_capex_eur`. `lifecycle_cash_npv` discounts each draw's
`net_{d,t}` at the real `discount_rate`, giving a per-draw NPV → the
P10/P50/P90/`P(NPV>0)` distribution. `legacy_screening_npv` runs the existing
`calculate_npv_distribution` on the same annual distribution with no dated
events (parity). Shadow wear appears **nowhere**; VOM is not re-deducted.

## 7. Red lines

1. **No shadow wear in cash NPV** — never `calculate_degradation_cost()` on the
   cash line, and never via the floor back door (§4.5).
2. **Not bankable** — pre-tax unlevered; the label and UI copy say so.
3. **No full-stack composition** — one producer-issued strategy result feeds the
   cash line; overlays/comparators stay non-additive.
4. **Augmentation fully specified** — year + cost + restoration (frac of
   nameplate) + residual, and its economic effect is stated (§4.2), or rejected.
5. **Floor comparator-only** — not in the aggregator; ProjectCase never reads
   `floor_protected_cashflow_eur` / `_pv_eur`.
6. **Gross, producer-issued revenue** — eligibility is a type/adapter property;
   ceilings, deltas, overlays, gross-additive references, benchmarks, and
   unavailable results are unrepresentable as cash revenue.
7. **VOM counted once** — embedded in the solver (0.5 EUR/MWh); recorded in
   provenance; never re-deducted in §6. No AssetCase VOM knob.
8. **Decay stays in its lane** — only `DAOnlySpreadDecay`, only for DA-only, per
   `spread-decay-v1`; no decay on multi-stream totals.
9. **Forecast = walk-forward** for any cash-eligible forecast-driven strategy.
10. **Fail-closed** — an invalid/unavailable strategy raises; no silent fallback,
    no €0 substitution.
11. **CapEx once** — single sidebar source.

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
- **Endogenous fade → augmentation.** Deferred; v1 augmentation is a dated
  schedule with a stated premise.

## 9. Increments (each a dual-reviewed PR)

- **PC-A** — typed schema (`AssetCase`/`LifecycleCase`/`MarketCase`/
  `ValuationCase`/`ProjectCase`/`RunResult`) + **solver adapters** emitting
  `StrategyRunResult` + `validate()` + fingerprint. Pure, no UI. Pins:
  producer-typed eligibility, walk-forward gate, augmentation admissibility,
  fail-closed, engineering match, floor-not-consumed, decay-mode gate.
- **PC-B** — bootstrap + lifecycle cash-flow + both NPV **distributions** (pure
  calc). Pins: no-shadow-wear, gross basis, VOM-once, projection modes, dated
  augmentation/residual/decommissioning timing, legacy-screening parity.
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
6. **Forecast-mode strictness** — *genuinely open:* reject **both** `in_sample`
   and `loo` for cash (walk-forward only, §5), or allow `loo` as a labelled
   optimistic variant? Recommendation: reject both.
7. **Augmentation effect model** — *genuinely open:* v1 "premise" flag (§4.2) vs
   an explicit per-year capacity/revenue multiplier now. Recommendation: premise
   flag in v1, multiplier in v1.1.
