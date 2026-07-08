# Stochastic MILP v2 — endogenous reserve co-optimisation (design contract, scope only)

Status: **r4 — Codex round-4 verdict "LOCK AFTER REVISIONS", the listed
revisions applied. Formal lock = merge of the design PR (Gemini/user review on
the PR).** No solver code lands until this contract is locked, mirroring the
v1 process (`docs/design/stochastic-milp-v1.md`, four review rounds before
Increment A).

Revision log:
- **r0**: initial draft (CC-authored, 2026-07-08).
- **r1** (Codex round 1 — verdict "needs another round", three blockers, all
  probe-verified): (1) the capped-myopic baseline's Stage 0 is now
  **cap-constrained** (`r ≤ min(power, rebid_cap)`) so all three arms share one
  feasible set — the unconstrained 9.2b joint LP can commit `r_myopic >
  rebid_cap` and make the capped executor raise (§3); (2) in v2 reserve mode
  the **deadband is inert on ALL days for ALL arms** — 9.2b reserve-first
  always re-dispatches even on zero-reserve days, so a deadband hold breaks
  the cap=∞ anchor (§3, §6-6); (3) Stage-0 `r*` degeneracy hits the
  **HEADLINE**, not just the split (equal-optimal reserve vectors settle to
  different realised reserve revenue and headroom value), so a **canonical
  Stage-0 selector is IN v2.0 scope**, applied symmetrically to every arm's
  Stage 0, plus a `stage0_tiebreak_stable` flag on the PR #43 pattern (§2.2,
  §3). Also: all Stage-0 inputs INCLUDING the IDA scenario bundle are forced
  walk-forward in reserve mode (§2.3); the Stage-2 headroom constraint is
  written in explicitly linear form (no `(power − r_t)`-big-M bilinearity,
  §2.1); the Stage-0 expected objective is demoted to a diagnostic — realised
  identities anchor strictly post-execution (§2.4); the v1-collapse pin gains
  valid-day + scenario-RNG equality requirements (§6-1); §6-4 is phrased as an
  in-model expectation test; §8 Q1 (per-interval) and Q3 (canonical selector)
  are resolved and closed.
- **r2** (Codex round 2 — all ten round-1 findings verified resolved; two NEW
  blockers introduced by the r1 fixes, both accepted): (1) the §6-1 v1-collapse
  pin contradicted §2.3's forced walk-forward (the v1 batch may run LOO with a
  live deadband) — SPLIT into a **routing pin** (no capacity ⇒ literally the v1
  code path, trivially equal) and a **constrained collapse pin** (comparator
  fixed to walk-forward, deadband 0, same seed/S); (2) §6-10's "same `r*`
  across S=1/S=N at cap=∞" was a false transfer of the v1 #41 pin — reserve has
  NO decoupling analog (§1.1), S=1 and S=N are genuinely different objectives
  whose optimal `r*` may legitimately differ at any cap; the selector only
  stabilises ties WITHIN one problem, so the pin moves to identical-objective
  fixtures. Also: the §2.2 selector is now an implementable three-pass spec
  with per-pass tolerances and an all-or-nothing fallback; §6-5's element-wise
  pin upgraded MAY→MUST; §5 gains an explicit UI disclosure that reserve mode
  ignores the deadband knob; the baseline is NAMED ("cap-feasible myopic
  reserve baseline") and explicitly distinguished from the panel's uncapped
  9.2b row; Stage-0 fallback is escalated to a HEADLINE-level warning with
  zero-fallback required in core fixtures; §8 Q2 (performance budget) is
  committed and closed — no open questions remain.
- **r3** (Codex round 3 — all eight round-2 findings verified resolved; two
  precision blockers in the r2 text): (1) §6-10(e)'s "seed-independent `r*`"
  overreached — different seeds draw different scenario sets, hence different
  Stage-0 objectives the selector cannot reconcile; replaced by
  fixed-scenario repeated-solve determinism; (2) the Q2 "≤25s end-to-end"
  arithmetic ignored that ALL THREE arms run Stage-0 canonicalisation —
  rewritten with per-SOLVE canonical deadlines, an honest pathological
  per-day bound (the sum of caps, completion guaranteed by fallback), and a
  typical-day target. Also: the §5 reserve-mode routing predicate is now
  exact (windowed per-product series non-empty, the same
  `capacity_price_series_for_product` + `_slice_to_local_dates` rule the
  triple rows use, with an out-of-window-only fixture), and the 2a/2b budget
  is one monotonic deadline per solve (2b gets only remaining time).
- **r4** (Codex round 4 — verdict **LOCK AFTER REVISIONS**, three exact
  revisions, all applied): (1) §4/§8 contradiction on the ceiling — the
  `coopt_ceiling_v2` solve is objective-only, canonical selectors are
  objective-preserving, so NO selector or tie-stability flag applies to it
  (and v2 must not enshrine v1's current waste of a canonical pass inside
  `stochastic_coopt_ceiling`); (2) §8 canonical-window count corrected six →
  FIVE (Stage-0 × 3 arms + Stage-1 × 2 stochastic solves), pathological bound
  ~53s → **~45s**; (3) §9 pin ownership split — the §6-1.1 routing spy is
  cockpit wiring (V2-D), the §6-1.2 constrained collapse and §6-9
  adverse-geometry headline pins are batch-level (V2-C). Codex confirms the
  contract now meets the #41 standard: implementable with no further design
  decisions.

## 1. Positioning

v1 upgraded the 9.2b policy's **Stage-1 DA myopia** (the DA commitment now
anticipates the IDA rebid opportunity across a scenario set). v2 upgrades the
remaining **Stage-0 reserve myopia**, which the 9.2b solver documents about
itself (`dispatch.solve_sequential_da_id_reserve_dispatch`):

> Stage 0 sizes reserve on DA arbitrage, ignoring the IDA rebid value left on
> the remaining headroom.

Today, `solve_daily_joint_capacity_lp(da_forecast, reserve_forecast)` commits
per-interval reserve as a trade-off against **DA-forecast arbitrage only**. The
IDA rebid value of the headroom the reserve consumes is invisible to that
decision. v2 makes the reserve commitment **scenario-aware**: reserve is sized
against the reserve-price forecast AND the expected IDA-scenario value of the
physical headroom it would consume.

v2 is NOT a new revenue stream (still DA + IDA1 + reserve **capacity** only),
NOT a replacement for the 9.2a/9.2b triple rows, and inherits every v1
red-line (§7).

### 1.1 Why this has value even at `rebid_cap = ∞` (no decoupling analog)

The v1 decoupling theorem says the *DA financial* position decouples from
Stage 2 at infinite rebid cap, degenerating the extensive form. **There is no
analog for the reserve decision**: committed reserve consumes *physical*
Stage-2 headroom, so `r` changes the expected physical recourse value
`E_s[V_s(power − r)]` at ANY cap — there is no accounting path through the DA
financial leg by which the Stage-0 extensive form collapses to a deterministic
equivalent (Codex round-1 confirmed). This is the mathematical reason v2 is
worth building, and it is pinned as an **in-model expectation test** (§6-4),
not a single-realised-day guarantee.

## 2. Decision structure (the locked fork: layered sequential, scenario-aware stages)

v2 keeps the realistic D-1 auction order (reserve ~08:00 → DA 12:00 → IDA1
~15:00) as a **layered sequential** policy in which each commitment stage is
scenario-aware about IDA but point-forecast about everything not yet realised.
It does NOT build a nested three-stage stochastic program (no DA scenario
generator exists; `S_da × S_ida` explodes the extensive form; screening-grade
budget).

- **Stage 0 (reserve gate, D-1 ~08:00)** — NEW solver
  `stochastic_dispatch.solve_stochastic_reserve_commitment()`: ONE extensive
  form with variables
  - per-interval reserve `r_t`, continuous, bounds `0 ≤ r_t ≤ min(power_mw,
    rebid_cap_mw)` (the endogenous form of the v1 `rebid_cap ≥ max reserve`
    feasibility domain — satisfied by construction);
  - an internal Stage-1′ DA schedule against the **DA point forecast**
    (walk-forward `build_da_price_forecast`, the 9.2b convention — the reserve
    gate must not see realised DA or the target day, independent of the
    panel's LOO toggle);
  - per-scenario Stage-2 schedules (the SAME `ida_scenarios` bundle as
    Stage 1 — same target day, same seed; no new generator; §2.3 pins its
    forecast mode), rebid-cap coupling against the Stage-1′ forecast DA net.

  Objective: `Σ_t reserve_fc_price_t · availability · r_t · dt` + expected
  Stage-1′/Stage-2 energy value (v1 §2 accounting). **Extract ONLY `r*`**; the
  internal DA schedule is discarded — it exists only to price the headroom
  competition, exactly as 9.2b's Stage-0 joint LP discards its DA schedule.

- **Stage 1 (DA gate, 12:00)** — UNCHANGED v1 B1:
  `solve_stochastic_da_commitment(da_realised, scenarios, …, reserve_mw=r*)`.
  Reserve enters exogenously, canonical Stage-1 tie-break and all v1
  identities intact.

- **Stage 2 (execution + settlement)** — UNCHANGED v1 B2 capped execution,
  except settlement uses the **realised** reserve price series on `r*`
  (9.2b convention: commit under forecast, settle at realised). Capacity
  revenue is **no longer a constant across policies** — different policies
  commit different `r*` — so it no longer cancels in the policy-value delta
  (this kills the v1-D "reserve cancels, DA+IDA1 only" simplification, §5).

Zero-price skip (determinism + collapse ordering): if the day's forecast
capacity price is ≤ 0 on every interval, or no reserve forecast exists,
Stage 0 is SKIPPED and `r* ≡ 0` — at zero price any `r > 0` is only weakly
dominated, and we do not want a solver tie deciding whether headroom gets
consumed. The skip decision is taken BEFORE requiring any Stage-0 DA/reserve
walk-forward forecast (so a day v1 can run is never excluded by a Stage-0
input v1 does not need), and Stage 0 never consumes scenario RNG (the bundle
is built once per day, outside Stage 0, and shared) — both load-bearing for
the §6-1 v1 collapse.

Reserve granularity: **per-interval, matching 9.2b Stage 0** (verified:
`solve_daily_joint_capacity_lp` returns per-interval `reserve_mw`; the 4h
block structure enters only through the block-of-day price forecast).
Block-CONSTANT reserve (closer to real FCR/aFRR products) is a deliberate
non-goal in v2.0 — switching only the stochastic arm would change its feasible
set and poison the attribution; if added later it must be a symmetric knob on
ALL arms. UI/export must state the per-interval screening nature (§5).
**RESOLVED (r1, was Q1).**

### 2.1 Linear formulation of the headroom coupling (no bilinearity)

With `r_t` endogenous, the Stage-2 bounds must NOT be written as mode-binary
big-M terms scaled by `(power − r_t)` — that is bilinear (variable × binary).
The constraints stay linear by keeping the v1 mutex rows at FIXED power big-M
and adding one additive headroom row per scenario and interval:

```
stage2_charge_{s,t}    ≤ power · b_{s,t}
stage2_discharge_{s,t} ≤ power · (1 − b_{s,t})
stage2_charge_{s,t} + stage2_discharge_{s,t} + r_t ≤ power        ∀ s, t
```

(the same additive form the 9.2b joint LP already uses for its own reserve
variable, extended per scenario).

### 2.2 Canonical Stage-0 selector (IN scope — headline protection)

Stage-0 `r*` can be degenerate (e.g. flat capacity premium), and unlike the
v1 Stage-1 ties this hits the **headline**: equal-optimal reserve vectors
settle to different realised reserve revenue AND different realised headroom
value, so `policy_value_v2` itself would be solver-tie-sensitive. Therefore
v2.0 includes a canonical Stage-0 selector, applied **symmetrically to every
arm's Stage 0** (the stochastic/S=1 extensive form AND the cap-constrained
myopic joint LP):

- **pass 1**: optimal objective `z*`;
- **pass 2a**: fix the objective at `z*` (v1 #41 tolerance discipline:
  `≤ z* + 1e-9·(1+|z*|)` bound, `> z* + 1e-6` degradation backstop — sign
  convention note: these are written in the SOLVER'S internal minimisation
  form, where `z*` is the scipy `linprog` optimum of the NEGATED revenue
  objective, exactly as `_canonicalize_stage1` implements the v1 #41 pass; in
  the mathematical maximisation formulation of §2 the inequalities invert to
  `≥ z* − tol` / `< z* − 1e-6`) and minimise **total reserve** `Σ_t r_t` →
  optimum `R*` (prefer committing less);
- **pass 2b**: fix the objective at `z*` AND total reserve at
  `Σ_t r_t ≤ R* + 1e-9·(1+R*)` (same backstop discipline on both) and
  minimise **time-weighted reserve** `Σ_t (t+1)·r_t` (earliest placement
  breaks the remaining temporal-permutation ties — the #41 lesson that flat
  weights leave the pattern free). A single weighted objective is NOT
  acceptable: with continuous `r_t` no dominance weight can be proven, so the
  levels must be sequential re-solves;
- **all-or-nothing fallback under ONE monotonic deadline per solve**: passes
  2a and 2b share a single wall-clock deadline (§8 Q2) started when 2a
  begins — 2b receives only the REMAINING time, and an expired deadline at
  any point (or a non-proven-optimal `status != 0` result in either pass)
  means the arm keeps its pass-1 solution (no partially-canonicalised
  vectors; a 2a-only result is NOT accepted) and the day sets
  `stage0_tiebreak_stable = False` on the PR #43 pattern (solver → batch
  per-day column → summary count → cockpit warning). Because Stage-0 ties
  hit the HEADLINE (not just the split), the fallback warning attaches to
  the policy-value metric itself (§5), and the core regression fixtures
  REQUIRE zero Stage-0 fallback days (a fixture that trips the fallback is a
  broken fixture, not a tolerated path).

The v1 Stage-1 canonical selector and its `canonical_tiebreak_applied` /
`tiebreak_stable` plumbing are unchanged; v2 adds the Stage-0 analog beside
them, and the cockpit reports both.

What the selector can and cannot do: it stabilises ties WITHIN one Stage-0
problem (same scenario set, same objective). It CANNOT make different
objectives — e.g. the S=1 co-opt vs the S=N stochastic Stage 0, whose optimal
`r*` legitimately differ at any cap because reserve never decouples (§1.1) —
agree on a reserve vector, and no pin may assume otherwise (§6-10).

### 2.3 Information sets (all Stage-0 inputs walk-forward)

In v2 reserve mode, ALL forecast inputs are forced **walk-forward**: the
Stage-0 DA point forecast, the reserve-price forecast, AND the IDA scenario
bundle's `forecast_mode` (the v1 panel default `loo` may see future days —
acceptable for a pure DA+IDA skill estimate, but the reserve gate is a
real-time commitment; mixing a LOO scenario set into it would leak the target
day into a decision the contract sells as walk-forward). This matches
`simulate_sequential_da_id_reserve_batch`, which already forces
`forecast_mode="walk_forward"` end-to-end. Consequence: the v2 reserve batch
loses the window's first day(s) exactly as 9.2b does; a LOO run of the v2
batch is NOT offered (no half-real information sets).

### 2.4 Stage-0 proxy geometry (diagnostic only, never an identity)

The Stage-0 extensive form prices headroom with the rebid cap anchored to the
Stage-1′ **forecast** DA net; the executed Stage 2 is capped around the
**realised-DA** Stage-1 net. The geometries differ whenever the DA forecast
misses. This is accepted screening forecast error — the Stage-0 expected
objective is a DIAGNOSTIC (exported as such) and is never used in a pinned
identity; all realised identities (§6) anchor strictly after the actual
Stage-1/Stage-2 execution. A signed regression test constructs an adverse
DA-forecast geometry where v2 underperforms the myopic baseline
(`policy_value_v2 < 0`), pinning that the headline is honest rather than
optimistic by construction.

## 3. Comparison basis & headline

All three policies run at a COMMON `rebid_cap_mw`, window, scenario
bundle/seed, forecast series, and realised settlement prices. **In v2 reserve
mode the deadband is INERT on all days for all arms** (`min_rebid_uplift_eur`
stays a v1 DA+IDA1-path knob): 9.2b reserve-first re-dispatches
unconditionally — even on zero-reserve days — so a deadband hold on a
zero-reserve day would break the §6-6 anchor and asymmetrically favour
whichever arm holds.

1. **Cap-feasible myopic reserve baseline** (named; "9.2b-STYLE", NOT the
   panel's 9.2b row): a **cap-constrained** 9.2b Stage 0 — the existing
   `solve_daily_joint_capacity_lp` objective with reserve bounds tightened to
   `r ≤ min(power_mw, rebid_cap_mw)` (the unconstrained joint LP bounds
   reserve only by power, so at a finite common cap it can commit
   `r_myopic > rebid_cap` and the capped executor raises — probe-verified) —
   then `solve_myopic_capped_da_id_dispatch(reserve_mw=r_myopic,
   reserve_price=realised)`. Both stages myopic; all arms share one feasible
   set, so the v2 delta isolates scenario awareness, not machinery.
   **Presentation red-line**: at a finite cap this baseline is a DIFFERENT
   number from the panel's sixth row (the uncapped forecast-driven realistic
   9.2b), and the two coincide only at `rebid_cap = ∞`; the delta-row label
   and the attribution caption must name the baseline distinctly so the panel
   never shows two disagreeing numbers both called "9.2b".
2. **S=1 co-opt**: Stage 0 and Stage 1 both with the single base-forecast
   scenario.
3. **S=N stochastic**: the full policy.

**Headline** (unchanged shape from v1-C1):
`policy_value_v2 = stochastic_realised − myopic_realised`, signed, never
clamped, INCLUDING capacity revenue on both sides (it no longer cancels).
Headline integrity rests on the §2.2 canonical Stage-0 selector: on
`stage0_tiebreak_stable = False` days the HEADLINE itself is non-canonical
(not just the split) — the batch summary counts them
(`n_stage0_fallback_days`), the export carries per-day columns, the cockpit
warning attaches to the policy-value metric, and core regression fixtures
require the count to be zero (§2.2).

**Diagnostic split** (inherited caveats): `commitment_value = coopt_realised −
myopic_realised`, `distribution_value = stochastic_realised − coopt_realised`.
Note "commitment" now bundles Stage-0 AND Stage-1 point-forecast awareness; a
finer four-way split (reserve-decision value vs DA-commitment value) is
explicitly OUT of v2.0 scope.

**Regression anchor**: at `rebid_cap = ∞` (deadband inert per above), the
capped-myopic baseline must tie `simulate_sequential_da_id_reserve_batch`'s
realised total exactly, day by day — and the anchor test must include a
**zero-reserve day** (the case that exposed the deadband asymmetry).

## 4. Ceiling family

`coopt_ceiling_v2` = same-cap perfect-foresight with ENDOGENOUS reserve:
solver reuse of the Stage-0 extensive form with `S=1 ≡ realised IDA`, DA =
realised DA, reserve price = realised — then Stage 1/2 collapse into it (one
solve). Every executed `(r, Stage-1, Stage-2)` triple of any §3 policy is
feasible for this problem (all arms share the §3 cap-constrained feasible
set), so `realised_v2 ≤ coopt_ceiling_v2` by construction. Endogenous-`r`
optimisation dominates any fixed FEASIBLE `r`: `coopt_ceiling_v2 ≥` the v1
co-opt ceiling (`r = 0`) and `≥` the fixed-`r_myopic` ceiling (feasible by the
§3 cap constraint) — both pinned. The 9.2a global triple ceiling remains a
separate UNCAPPED comparator (signed), exactly as the legacy ceiling is in v1.

**Objective-only, selector-disabled**: the ceiling solve reads ONLY the pass-1
optimum objective; canonical selectors are objective-preserving, so neither
the Stage-0 nor the Stage-1 canonical selector runs for `coopt_ceiling_v2`
and no tie-stability flag attaches to it (r4 — v1's current
`stochastic_coopt_ceiling` wastes a canonical pass by calling the
unconditional `solve_stochastic_da_commitment`; v2 uses a selector-disabled
path and must not enshrine that waste).

## 5. Cockpit / UI surface (increment V2-D)

- The existing **Include stochastic policy** checkbox: reserve mode is active
  **iff** the selected `reserve_product`'s per-interval capacity series is
  non-empty AFTER windowing to the valid dates — the EXACT rule the 5th/6th
  rows already use (`capacity_price_series_for_product(window_anc, product)`
  where `window_anc = _slice_to_local_dates(...)`), NOT the full-sample
  `list_capacity_products()` availability check. When active the panel runs
  the v2 batch INSTEAD of the v1 DA+IDA1-only batch; otherwise (including a
  zone whose capacity rows exist only OUTSIDE the window — a pinned fixture)
  the LITERAL v1 path runs and no v2 machinery is invoked (§6-1 routing pin).
  **Mandatory disclosure** (a user with a deadband set must not lose it
  silently): visible text near the checkbox states that reserve-mode
  stochastic IGNORES `min_rebid_uplift_eur` (re-dispatches on all days, §3)
  and forces walk-forward forecasts regardless of the panel's LOO toggle
  (§2.3); the v1 deadband/LOO semantics apply only while the v1 DA+IDA1 path
  is active.
- The strategy-table delta row gets a v2 label
  (`STOCHASTIC_POLICY_VALUE_RESERVE_LABEL`, e.g. "Stochastic policy value
  (vs capped 9.2b reserve-first)"); `_strategy_chart_rows()` must exclude BOTH
  labels from the bar chart (regression-tested, extending the v1 guardrail).
  One delta row, label switches — never two stochastic rows at once.
- Attribution subpanel: caption gains "includes reserve capacity at realised
  prices; committed walk-forward; per-interval reserve (screening, not a 4h
  product commitment)" and reports BOTH tie-stability lines — Stage 1
  (`tiebreak_stable`, PR #43) and Stage 0 (`stage0_tiebreak_stable`, §2.2).
- Export: `_append_stochastic_assumptions` gains the reserve rows (walk-forward
  Stage 0, zero-price skip, per-interval granularity, availability haircut,
  deadband-inert rule, Stage-0 expected objective = diagnostic only); the
  per-day sheet gains `avg_reserve_mw` per policy and both tie-break columns.

## 6. Pinned identities (regression tests before UI)

1. **v1 collapse — TWO pins, not one** (the r1 single pin contradicted §2.3:
   the v1 batch may run LOO with a live deadband, under which element-wise
   equality with a forced-walk-forward, deadband-inert v2 is false):
   1. **Routing pin**: when the §5 predicate is false (no in-window capacity
      series for the selected product — including the capacity-rows-exist-
      only-out-of-window fixture) the panel executes the LITERAL v1 code path
      (`simulate_stochastic_da_id_batch`, user's own LOO/deadband settings) —
      equality is by routing, not by numerics, and the test pins the dispatch
      (v2 entry point not called; assertable via a spy/monkeypatch on the v2
      batch function).
   2. **Constrained collapse pin**: in reserve mode with forecast AND realised
      reserve price ≡ 0, `r* ≡ 0` for every policy and the v2 batch equals the
      v1 batch element-wise WHEN the v1 comparator is run under the reserve-
      mode conventions — `forecast_mode="walk_forward"`, deadband 0, same
      seed/S — including valid-day equality (the Stage-0 skip must not exclude
      days that comparator keeps; §2 skip-ordering) and scenario-RNG equality
      (the bundle is built once, Stage 0 consumes no randomness).
2. **Bound**: `realised_v2 ≤ coopt_ceiling_v2` (no clamp in the check).
3. **Ceiling dominance**: `coopt_ceiling_v2 ≥ v1 coopt_ceiling` (`r = 0`
   feasible) and `≥` the fixed-`r_myopic` ceiling (feasible under the common
   cap by §3).
4. **Anti-decoupling existence (expectation form)**: a constructed case at
   `rebid_cap = ∞` where the scenario-aware Stage 0 commits a DIFFERENT `r*`
   than the cap-constrained myopic joint LP and achieves a strictly higher
   Stage-0 EXPECTED objective — an in-model expectation test (§1.1), not a
   realised-day guarantee.
5. **IDA ≡ DA-forecast collapse**: when every scenario equals the DA forecast
   (no rebid opportunity), Stage 0's optimal objective equals the
   cap-constrained myopic joint LP's — objective equality pinned, AND (because
   the two problems are then IDENTICAL-objective, so the §2.2 selector applies
   to one optimal set) element-wise `r*` equality MUST be pinned once the
   selector lands.
6. **Anchor**: §3's `rebid_cap = ∞`, deadband-inert, day-by-day tie to
   `simulate_sequential_da_id_reserve_batch`, with a zero-reserve day in the
   fixture.
7. **Domain**: `r*` never exceeds `min(power, rebid_cap)` by construction in
   EVERY arm (endogenous bounds + the §3 cap-constrained myopic Stage 0); the
   v1 exogenous-reserve raise stays for external callers.
8. **Rebid gating**: v2 reserve mode re-dispatches on all days for all arms
   (deadband inert, §3); the v1 path's deadband behaviour is untouched
   (regression: v1 tests unaffected).
9. **Adverse-geometry honesty**: the §2.4 signed test — a bad DA-forecast
   geometry makes `policy_value_v2 < 0` (the headline can lose and is not
   clamped).
10. **Stage-0 tie-break — IDENTICAL-OBJECTIVE fixtures only** (the r1 phrasing
    "same-`r*` across S=1/S=N at `rebid_cap = ∞`" was a false transfer of the
    v1 #41 pin: reserve never decouples (§1.1), so S=1 and S=N are different
    objectives whose optimal `r*` may legitimately differ at ANY cap; the
    selector stabilises ties within ONE problem only, §2.2). Pins: (a) an
    S=N run whose scenarios are all IDENTICAL to the base forecast selects
    the same `r*` element-wise as the S=1 run (identical objectives); (b) the
    §6-5 collapse fixture's element-wise equality; (c) the tie-break never
    changes `z*`; (d) fallback sets `stage0_tiebreak_stable = False` without
    changing `z*`; (e) **fixed-scenario repeated-solve determinism**: on a
    degenerate fixture (flat capacity premium with interchangeable
    intervals), solving the SAME Stage-0 problem (same scenario set, same
    objective) twice yields the identical `r*` element-wise. NOT
    seed-independence — different seeds draw different scenario sets, hence
    different objectives the selector cannot and must not reconcile (§2.2).

## 7. Red-lines (inherited + new)

- Reserve **capacity headroom only** — NO activation energy, NO bid-acceptance
  model, NO product-specific SoC duration (v1 §Non-goals, unchanged).
- v2 reserve mode is ALWAYS walk-forward end-to-end — Stage-0 DA forecast,
  reserve-price forecast, AND the IDA scenario bundle (never sees the target
  day); no LOO variant is offered for the reserve batch (§2.3).
- Commit under forecast, settle at realised (both reserve and IDA legs).
- The Stage-0 expected objective is a diagnostic, never a pinned identity or
  a UI revenue number (§2.4).
- Screening-grade, not bankable; the risk block stays a dispersion diagnostic
  (objective risk-neutral); GCC/LFC remain out of scope.

## 8. Resolved questions (all closed — nothing open at lock)

- ~~Q1 — granularity~~ **RESOLVED (r1)**: per-interval locked for v2.0, 9.2b
  parity; block-constant later only as a symmetric all-arms knob (§2).
- ~~Q2 — performance budget~~ **RESOLVED (r3, committed; the r2 "≤25s
  end-to-end" ignored that all three arms canonicalise their Stage 0)**.
  Budget structure, per worst-case 15-min S=10 day:
  - **Per-solve canonical deadline**: every canonical tie-break (Stage-0
    2a+2b under ONE monotonic deadline, §2.2; Stage-1's existing #41 pass)
    carries an **8s** cap. FIVE canonical windows exist per day at most —
    Stage-0 × 3 arms (myopic joint-LP, S=1, S=N) + Stage-1 × 2 stochastic
    solves (S=1, S=N; the myopic arm's Stage 1 is a plain `solve_daily_lp`,
    no canonical pass, and the ceiling solve is objective-only /
    selector-disabled per §4) — but the myopic and S=1 problems are
    LP-sized/small and in practice finish in milliseconds; only the S=N
    Stage-0/Stage-1 passes meaningfully approach the cap.
  - **Pass-1 solves**: Stage-0 extensive S=10 ~2s, Stage-1 B1 S=10 ~2s,
    S=1/myopic/execution/ceiling LPs ~1–2s combined.
  - **Committed targets**: TYPICAL worst-case day ≤ **15s**; PATHOLOGICAL
    hard bound = pass-1 total + the sum of canonical caps (~5s + 5×8s ≈
    **45s**), reachable only if every canonical window times out — and every
    timeout degrades ONLY tie-stability via the all-or-nothing fallback
    (§2.2), never correctness or completion, so the day always finishes.
  - If pass-1 itself blows the budget on real data, cut S before touching
    the binaries (v1 §6 precedent).
- ~~Q3 — Stage-0 tie-break~~ **RESOLVED (r1)**: canonical Stage-0 selector is
  IN v2.0 scope (§2.2) — Codex round 1 showed the tie hits the headline, not
  just the split, which was the v1 rationale for deferring; that rationale
  does not transfer.

## 9. Increment plan (after lock)

- **V2-A**: `solve_stochastic_reserve_commitment()` (linear form §2.1,
  zero-price skip, canonical Stage-0 selector §2.2) + pins §6-4/5/7/10.
- **V2-B**: day wrapper (`solve_stochastic_triple_dispatch()`: Stage 0 → B1 →
  B2 + realised reserve settlement + `coopt_ceiling_v2` selector-disabled per
  §4) + pins §6-2/3.
- **V2-C**: batch (3 policies, common cap, cap-constrained myopic Stage 0,
  deadband-inert) + pins §6-1.2 (constrained collapse) / §6-6 (anchor) /
  §6-8 / §6-9 (adverse-geometry headline — a batch-level `policy_value_v2`
  quantity) + summary/risk block + both tie-break flags.
- **V2-D**: cockpit/export wiring per §5 + pin §6-1.1 (routing spy) + the
  label guardrail test.

Each increment is its own PR with dual review; solver increments get the
pre-merge math audit, per the v1 lane.
