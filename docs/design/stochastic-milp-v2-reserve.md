# Stochastic MILP v2 — endogenous reserve co-optimisation (design contract, scope only)

Status: **DRAFT r0 — under joint scope review (Gemini + Codex), not yet locked.**
No solver code lands until this contract is locked, mirroring the v1 process
(`docs/design/stochastic-milp-v1.md`, four review rounds before Increment A).

Revision log:
- **r0**: initial draft (CC-authored, 2026-07-08).

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
Stage-2 headroom (`stage2_charge + stage2_discharge + reserve ≤ power`), so
the IDA scenario value interacts with `reserve_mw` at ANY cap. Scenario
awareness can change the optimal reserve even at `rebid_cap = ∞`. This is the
mathematical reason v2 is worth building — and it must be pinned by an
existence test (§6-4), not just asserted.

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
    Stage 1 — same target day, same seed; no new generator), headroom
    `power − r_t`, rebid-cap coupling against the Stage-1′ forecast DA net.

  Objective: `Σ_t reserve_fc_price_t · availability · r_t · dt` + expected
  Stage-1′/Stage-2 energy value (v1 §2 accounting). **Extract ONLY `r*`**; the
  internal DA schedule is discarded — it exists only to price the headroom
  competition, exactly as 9.2b's Stage-0 joint LP discards its DA schedule.

- **Stage 1 (DA gate, 12:00)** — UNCHANGED v1 B1:
  `solve_stochastic_da_commitment(da_realised, scenarios, …, reserve_mw=r*)`.
  Reserve enters exogenously, canonical tie-break and all v1 identities intact.

- **Stage 2 (execution + settlement)** — UNCHANGED v1 B2 capped execution,
  except settlement uses the **realised** reserve price series on `r*`
  (9.2b convention: commit under forecast, settle at realised). Capacity
  revenue is **no longer a constant across policies** — different policies
  commit different `r*` — so it no longer cancels in the policy-value delta
  (this kills the v1-D "reserve cancels, DA+IDA1 only" simplification, §5).

Zero-price skip (determinism): if the day's forecast capacity price is ≤ 0 on
every interval, Stage 0 is SKIPPED and `r* ≡ 0` — at zero price any `r > 0` is
only weakly dominated, and we do not want a solver tie deciding whether
headroom gets consumed. Days with no reserve forecast likewise commit
`r* ≡ 0` (9.2b safe-degrade, not silent optimism).

Reserve granularity: **per-interval, matching 9.2b Stage 0** (verified:
`solve_daily_joint_capacity_lp` returns per-interval `reserve_mw`; the 4h
block structure enters only through the block-of-day price forecast).
Block-CONSTANT reserve (closer to real FCR/aFRR products) is a deliberate
non-goal in v2.0 — it would change the baseline's feasible set and break the
attribution (§8 open question Q1).

## 3. Comparison basis & headline

All three policies run at a COMMON `rebid_cap_mw`, window, scenario
bundle/seed, forecast series, and realised settlement prices:

1. **Capped-myopic baseline**: 9.2b Stage 0 (`solve_daily_joint_capacity_lp`
   on DA forecast + reserve forecast, IDA-blind) → `r_myopic` → existing
   `solve_myopic_capped_da_id_dispatch(reserve_mw=r_myopic,
   reserve_price=realised)`. Both stages myopic; the v2 delta therefore
   isolates scenario awareness, not machinery differences.
2. **S=1 co-opt**: Stage 0 and Stage 1 both with the single base-forecast
   scenario.
3. **S=N stochastic**: the full policy.

**Headline** (unchanged shape from v1-C1):
`policy_value_v2 = stochastic_realised − myopic_realised`, signed, never
clamped, INCLUDING capacity revenue on both sides (it no longer cancels).

**Diagnostic split** (inherited caveats): `commitment_value = coopt_realised −
myopic_realised`, `distribution_value = stochastic_realised − coopt_realised`.
Note "commitment" now bundles Stage-0 AND Stage-1 point-forecast awareness; a
finer four-way split (reserve-decision value vs DA-commitment value) is
explicitly OUT of v2.0 scope. Stage-0 `r*` can itself be degenerate (flat
capacity premium); no canonical selector for `r` in v2.0 — the split keeps a
tie-sensitivity caveat on reserve days even though Stage 1 is canonicalised,
and the per-day `tiebreak_stable` flag (PR #43) covers Stage 1 only.

**Regression anchor**: at `rebid_cap = ∞`, the capped-myopic baseline must tie
`simulate_sequential_da_id_reserve_batch`'s realised total exactly, day by day
(the v2 analog of the C1 anchor; both then share Stage 0, full-power Stage 2
vs IDA forecast, reserve-day always-rebid, and realised settlement).

## 4. Ceiling family

`coopt_ceiling_v2` = same-cap perfect-foresight with ENDOGENOUS reserve:
solver reuse of the Stage-0 extensive form with `S=1 ≡ realised IDA`, DA =
realised DA, reserve price = realised — then Stage 1/2 collapse into it (one
solve). Every executed `(r, Stage-1, Stage-2)` triple of any §3 policy is
feasible for this problem, so `realised_v2 ≤ coopt_ceiling_v2` by
construction. Because endogenous-`r` optimisation dominates any fixed `r`,
`coopt_ceiling_v2 ≥` the v1 co-opt ceiling (r = 0) and `≥` the v1 ceiling at
`r = r_9.2b` — both pinned. The 9.2a global triple ceiling remains a separate
UNCAPPED comparator (signed), exactly as the legacy ceiling is in v1.

## 5. Cockpit / UI surface (increment V2-D)

- The existing **Include stochastic policy** checkbox: when in-window capacity
  prices exist (same `_reserve_triple_totals` window rule as the 5th/6th
  rows), the panel runs the v2 batch INSTEAD of the v1 DA+IDA1-only batch;
  with no in-window capacity prices the v1 path runs unchanged.
- The strategy-table delta row gets a v2 label
  (`STOCHASTIC_POLICY_VALUE_RESERVE_LABEL`, e.g. "Stochastic policy value
  (vs capped 9.2b reserve-first)"); `_strategy_chart_rows()` must exclude BOTH
  labels from the bar chart (regression-tested, extending the v1 guardrail).
  One delta row, label switches — never two stochastic rows at once.
- Attribution subpanel: caption gains "includes reserve capacity at realised
  prices; committed walk-forward" and keeps the PR #43 tie-stability line
  (Stage 1 only) + a Stage-0 degeneracy caveat.
- Export: `_append_stochastic_assumptions` gains the reserve rows (walk-forward
  Stage 0, zero-price skip, per-interval granularity, availability haircut);
  the per-day sheet gains `avg_reserve_mw` per policy.

## 6. Pinned identities (regression tests before UI)

1. **v1 collapse**: no capacity rows in window, or forecast AND realised
   reserve price ≡ 0 ⇒ `r* ≡ 0` for every policy and all v2 numbers equal the
   v1 batch element-wise (same seed).
2. **Bound**: `realised_v2 ≤ coopt_ceiling_v2` (no clamp in the check).
3. **Ceiling dominance**: `coopt_ceiling_v2 ≥ v1 coopt_ceiling` (r = 0 is
   feasible) and `≥` the fixed-`r_9.2b` ceiling.
4. **Anti-decoupling existence**: a constructed case at `rebid_cap = ∞` where
   scenario-aware Stage 0 commits a DIFFERENT `r*` than the 9.2b joint LP and
   realises strictly more in expectation (the §1.1 claim, made falsifiable).
5. **IDA ≡ DA-forecast collapse**: when every scenario equals the DA forecast
   (no rebid opportunity), Stage 0's objective/solution set equals the 9.2b
   joint LP's (the scenario term adds nothing) — objective equality pinned,
   NOT schedule equality (multi-optima).
6. **Anchor**: §3's `rebid_cap = ∞` day-by-day tie to
   `simulate_sequential_da_id_reserve_batch`.
7. **Domain**: `r*` never exceeds `min(power, rebid_cap)` by construction;
   the v1 exogenous-reserve raise stays for external callers.
8. **Deadband**: endogenous `r* > 0` days always re-dispatch (v1 §5 gating
   applies to the committed `r*`).

## 7. Red-lines (inherited + new)

- Reserve **capacity headroom only** — NO activation energy, NO bid-acceptance
  model, NO product-specific SoC duration (v1 §Non-goals, unchanged).
- Stage 0 is ALWAYS walk-forward (never sees the target day), independent of
  the panel's LOO/walk-forward toggle — same rule as 9.2b.
- Commit under forecast, settle at realised (both reserve and IDA legs).
- Screening-grade, not bankable; the risk block stays a dispersion diagnostic
  (objective risk-neutral); GCC/LFC remain out of scope.

## 8. Open questions for the review round

- **Q1 — granularity**: per-interval `r_t` (9.2b parity, proposed) vs
  block-constant 4h `r_b` (product realism). Proposal: per-interval in v2.0;
  block-constant later as an opt-in knob applied to BOTH arms symmetrically.
- **Q2 — performance budget**: Stage 0 adds one B1-sized extensive form (+n
  continuous vars, no new integers) per day per policy arm; worst-case 15-min
  S=10 day ≈ 2× v1 ≈ 4–5s before canonical passes. Proposed budget ≤ 20s/day
  worst case; if exceeded, reuse the v1 trick (fix scenario Stage-2 binaries
  from a relaxation) before cutting S.
- **Q3 — Stage-0 tie-break**: is a canonical `r*` selector needed in v2.0, or
  is the §3 split caveat + zero-price skip enough? Proposal: caveat only;
  selector deferred (mirrors v1→#41 history).

## 9. Increment plan (after lock)

- **V2-A**: `solve_stochastic_reserve_commitment()` + zero-price skip +
  pins §6-4/5/7.
- **V2-B**: day wrapper (`solve_stochastic_triple_dispatch()`: Stage 0 → B1 →
  B2 + realised reserve settlement + `coopt_ceiling_v2`) + pins §6-1/2/3/8.
- **V2-C**: batch (3 policies, common cap) + anchor §6-6 + summary/risk block.
- **V2-D**: cockpit/export wiring per §5 + label guardrail test.

Each increment is its own PR with dual review; solver increments get the
pre-merge math audit, per the v1 lane.
