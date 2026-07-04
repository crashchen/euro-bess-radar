# Scenario-based stochastic MILP — v1 design contract (scope only)

Status: **REVISED after joint scope review (Gemini + Codex), awaiting
re-lock.** No solver code ships with this document; per the agreed ordering
the solver core starts only after this contract is locked.

Revision log:
- **r1**: the first draft inherited the ceiling solver's pure-financial DA
  accounting, under which Stage 1 and Stage 2 decouple mathematically and
  the extensive form degenerates to a deterministic equivalent (Gemini
  catch, §2.1). r1 made the **rebid volume cap** the load-bearing coupling,
  fixed the ceiling/window/deadband references to the reserve-aware 9.2b
  semantics (Codex catches), and pinned scenario-resolution handling.
- **r2** (Codex re-lock round): the legacy 9.2a ceiling is NOT an upper
  bound for the new co-opt/stochastic policy family (its Stage-1 is the
  myopic DA-only schedule) — v1 gains a **same-cap perfect-foresight co-opt
  ceiling** and demotes the 9.2a ceiling to a signed legacy comparator
  (§2.4); the cap=∞ identity gains its missing premise via **mean-centred
  error paths** (§3); the deadband HOLD is **gated to no-reserve days**
  (holding a full-power Stage-1 schedule can violate reserve headroom, §5).
- **r3** (Codex re-lock round 3): the capped MYOPIC baseline's Stage-1 is
  solved without the Stage-2 constraints, so reserve + a sub-reserve cap can
  empty its recourse set — v1 pins the feasibility domain
  `rebid_cap_mw ≥ max reserve` (sufficient via a proportional-scaling
  witness; default cap always satisfies it; below it the batch raises, §5).

## 1. Positioning

The stochastic MILP is an **upgrade of the 9.2b reserve-first sequential
policy's Stage-1 myopia** (`dispatch.solve_sequential_da_id_dispatch` inside
`simulation.simulate_sequential_da_id_reserve_batch`); with no reserve loaded
it degrades to upgrading the plain Phase-7 DA+ID sequential policy. It is NOT
a replacement for the 9.2a perfect-foresight ceiling and NOT a new revenue
stream. Today Stage 1 commits the DA schedule **myopically** (DA-only MILP
that ignores the IDA opportunity entirely); the stochastic upgrade makes the
Stage-1 DA commitment anticipate the IDA rebid opportunity across a
**scenario set** of IDA price paths, under a finite rebid volume cap.

The headline is a **two-way decomposition** at a common rebid cap, common
reserve series, and common valid-day window:

- `co_opt_realised − sequential_realised` = value of a *non-myopic* Stage-1
  commitment (the S=1 point-forecast co-optimisation — cheap, deterministic);
- `stochastic_realised − co_opt_realised` = value of *distribution awareness*
  on top (the scenario set proper).

Both deltas are signed and MAY be negative on bad windows — report them
signed, never clamped. Everything stays screening-grade historical replay —
not trading advice, not live dispatch.

### Non-goals (v1 red-lines)

- **No activation-energy or reBAP/imbalance terms in the objective.** Those
  remain separate, non-additive overlays with their own red-lines. v1 covers
  DA + IDA1 + reserve capacity only.
- **No stochastic reserve decision.** Reserve stays committed by the existing
  9.2b reserve-first walk-forward rule (§2.3); a three-stage stochastic
  program (reserve → DA → IDA) is explicitly v2+.
- **No continuous-horizon SoC carry.** v1 solves standalone terminal-neutral
  days (like the sequential batch), isolating scenario value from carry
  value.
- **No new market data requirement.** Scenarios are built from the SAME
  loaded IDA history the climatology forecast already uses.

## 2. Decision structure

Two stages, mirroring the existing sequential policy's information timeline:

| Stage | Decision | Information available | Variables |
|-------|----------|----------------------|-----------|
| 1 (D-1 12:00, DA gate) | Full-power DA schedule (financial commitment) | Actual DA prices (limit-order clearing, as today); IDA **distribution** (scenario set), NOT the print | One non-anticipative schedule: `da_charge/da_discharge/da_soc` + binary mode per interval |
| 2 (IDA, recourse) | Physical re-dispatch per scenario, **within the rebid cap** | That scenario's IDA path | Per-scenario `id_charge/id_discharge/id_soc` + binary mode per interval |

Extensive-form MILP: Stage-1 variables are shared across scenarios
(non-anticipativity by construction), Stage-2 variables are indexed by
scenario. Settlement accounting per scenario reuses the existing implicit-MtM
identity (`total_s = da_gross − implicit_mtm_s + stage2_total_s` + reserve
capacity revenue, §2.3), and the objective is `max Σ_s w_s · total_s`.

### 2.1 The rebid cap is load-bearing (decoupling theorem)

Under the inherited accounting alone, Stage 2 has no constraint referencing
Stage-1 variables, so the objective separates:
`E_s[total_s] = Σ_t da_net_t · (DA_t − E_s[IDA_{s,t}]) · dt + E_s[stage2_total_s]`
— Stage 1 sees only the **scenario mean**, and with a mean-centred scenario
set the "stochastic" commitment is identical to a deterministic
co-optimisation against the point forecast (Gemini review, PR #32). The
scenario distribution only shapes Stage 1 through a coupling, which v1
introduces as a **per-interval rebid volume cap**:

`|stage2_net_{t,s} − da_net_t| ≤ rebid_cap_mw` for every interval t and
scenario s.

Economics: the implicit-MtM settlement assumes the whole DA-vs-physical gap
can be unwound at the IDA1 print — infinite auction liquidity, an optimistic
assumption. A finite cap makes large unwinds infeasible, so the per-scenario
recourse value becomes a piecewise, scenario-dependent function of the
Stage-1 position (concave in the LP relaxation / for a fixed binary pattern;
the binary mode variables break global concavity) and the distribution (not
just its mean) genuinely shapes the commitment.

Pinned consequence (regression test, §8-1): at `rebid_cap_mw = ∞` the
extensive form reduces exactly to the deterministic co-optimisation against
the **scenario-mean path**; because v1 mean-centres the sampled error paths
(§3), that path IS the base forecast, so `stochastic − co_opt ≡ 0` holds
exactly. That is the decoupling theorem embraced as a test, and the reason
v1 without a finite cap would be pointless.

`rebid_cap_mw` is a caller parameter, default `power_mw`, surfaced in the
assumptions audit table. It is a liquidity proxy, not a market rule — the
audit row must say so.

### 2.2 Integrality

**Binary mutual exclusion stays in BOTH stages, per scenario.** The
LP-relaxation degeneracy at negative prices (simultaneous charge+discharge
burning round-trip losses) applies to every scenario path; do not weaken this
to recover solve time — cap the scenario count instead (§6).

### 2.3 Reserve: exogenous, but the full 9.2b contract

When the day has committed reserve blocks, v1 replicates the 9.2b treatment
end to end — not just the power cap:

- `reserve_mw` per interval comes from the existing 9.2b reserve-first
  walk-forward rule, committed BEFORE the DA gate; it is an input, never a
  recourse variable.
- Physical headroom: each scenario's Stage-2 schedule satisfies
  `stage2_charge_s + stage2_discharge_s + reserve ≤ power_mw` per interval;
  the DA financial commitment does NOT occupy physical headroom (9.2a/9.2b
  fused-solver semantics).
- Capacity revenue: settled off the SAME per-interval reserve price series
  and `ANCILLARY_CAPACITY_AVAILABILITY` haircut as the 9.2b rows
  (`ancillary.capacity_price_series_for_product()` windowed to the valid
  days), added to `total_s` identically across scenarios.
With no reserve loaded, all of the above degrades to the plain DA+ID
semantics.

### 2.4 Ceiling family: the legacy 9.2a ceiling is NOT the new policies' bound

The existing perfect-foresight ceilings (`solve_daily_da_id_dispatch`,
reserve-aware `solve_daily_da_id_reserve_dispatch`) fix Stage 1 at the
**myopic DA-only schedule** and grant perfect foresight only to Stage 2 —
they bound the *myopic-commitment* policy family. The new co-opt/stochastic
Stage-1 optimises the financial leg `da_net·(DA − IDA)` directly, which can
legitimately exceed that bound (Codex re-lock catch). v1 therefore defines
TWO ceiling roles:

- **`coopt_ceiling_eur` (the binding upper bound)**: same-cap
  perfect-foresight co-optimisation — Stage 1 AND Stage 2 optimised with the
  realised IDA path known, under the SAME `rebid_cap_mw`, reserve headroom,
  and capacity settlement as the policy being bounded. Implementation reuses
  the stochastic solver itself with a single scenario ≡ the realised IDA path
  and optimal (forecast-free, gate-free) execution. Every executed
  (Stage-1, Stage-2) pair of the capped policies is feasible for this
  problem, so `realised ≤ coopt_ceiling` is guaranteed by construction.
- **Legacy 9.2a ceiling (signed comparator only)**: stays in the strategy
  table unchanged, as the ceiling of the myopic family. The gap
  `coopt_ceiling − legacy_ceiling` and any `realised − legacy_ceiling` are
  **signed** — a co-opt policy beating the legacy ceiling is information
  (the value of non-myopic commitment), not a violation.

## 3. Scenario generation contract (`ida_scenarios.build_ida_scenarios`)

**LOCKED (Codex ruling): mode = `error_resample`.** scenario_s = climatology
point forecast + a resampled historical *daily error path* (realised −
forecast, whole days). `day_resample` is dropped from v1.

Implementation pins:

- **History discipline**: the error pool obeys the TARGET day's
  `forecast_mode` allowed history exactly as `ida_forecast.build_ida_forecast`
  does (`loo` default / `walk_forward` / `in_sample`, same caveats — LOO is a
  skill estimate, not what a desk knew). Under LOO the target day itself is
  NEVER in the pool (no leakage).
- **Resolution partitioning (Gemini catch)**: error paths are sampled ONLY
  from days whose market time unit matches the target day's interval grid
  (60-min history cannot be added to a 15-min day — DE_LU switched Oct 2025).
  The pool is partitioned by resolution first; no interpolation or upsampling
  in v1 (it would fabricate flat sub-hour structure). When the same-resolution
  pool has fewer than S distinct days, the generator samples with replacement
  and reports degraded support — it does not silently widen the pool.
- **Mean-centring (Codex re-lock catch — the cap=∞ identity's missing
  premise)**: after sampling, the S error paths are re-centred by subtracting
  their weighted sample-mean path, so the weighted scenario mean is EXACTLY
  the base forecast per interval. Without this, a finite resample has a
  nonzero mean-error path and the cap=∞ decoupled solution would equal the
  *scenario-mean* co-opt, not the *base-forecast* co-opt (§8-1/§8-2 would
  not line up). The shift is a per-interval constant across scenarios, so
  cross-hour error correlation within each path is preserved; the centred
  paths are no longer raw historical errors, which the metadata records
  (`mean_centered=True`, default on).
- **Determinism**: the generator accepts `seed`/`random_state` and reports it
  in metadata.
- **Bundle contract**: the returned bundle carries the scenario paths, equal
  weights `w_s = 1/S` (weighted scenarios are a v2 knob), AND the base point
  forecast (`base_forecast`) — the solver needs it for execution (§5), so it
  travels with the scenarios rather than being re-derived.
- **Metadata**: `n_scenarios`, `mode`, `forecast_mode`, `seed`, same-resolution
  pool size, and coverage/fallback reporting in the `build_ida_forecast`
  style, so the cockpit can label scenario support honestly.

## 4. Solver contract (`dispatch.solve_stochastic_da_id_dispatch`)

Inputs: day frames (DA prices, realised IDA prices), the scenario bundle
(paths + weights + `base_forecast`), BESS params
(power/duration/efficiency/VOM), `rebid_cap_mw`, optional per-interval
`reserve_mw` + reserve price series (§2.3), optional `min_rebid_uplift_eur`
(§5). Outputs — superset of the sequential solver's decomposition so
`build_strategy_comparison` can consume it:

- `da_only_revenue_eur` (baseline, unchanged definition),
- `realised_total_eur` (stochastic Stage-1 commitment, executed per §5,
  settled at the REALISED IDA print, including capacity revenue when reserve
  is present),
- `stochastic_hold_eur` (settlement when the deadband HOLDS the stochastic
  Stage-1 schedule — see §5; NOT equal to `da_only_revenue_eur` in general),
- `forecast_uplift_eur` (the predicted uplift of the point-forecast rebid
  over holding the stochastic commitment — the deadband gate input),
- `coopt_ceiling_eur` (the binding same-cap perfect-foresight co-opt bound,
  §2.4 — computed by reusing this solver with a single scenario ≡ the
  realised IDA path and optimal execution),
- `legacy_ceiling_eur` (the existing 9.2a/DA+ID ceiling, signed comparator
  only, §2.4),
- `expected_total_eur` (the in-solver objective value over scenarios —
  diagnostic only, never presented as realised revenue),
- `captured_uplift_eur = realised − da_only` (MAY be negative; do not clamp),
- `forecast_error_cost_eur = coopt_ceiling − realised ≥ 0`,
- per-scenario totals for the risk report (§7).

Identity pinned by test: `captured + forecast_error == coopt_ceiling −
da_only` (decomposition as in the sequential solver, but against the new
binding ceiling — the legacy ceiling appears in no ≥0 identity).

## 5. Execution / settlement semantics

At replay time the desk still cannot see the print, so execution after the
stochastic Stage-1 commitment follows the sequential policy's structure, with
one semantic rewrite forced by the new Stage 1:

- Stage 2 re-dispatches against the **base point forecast** (from the
  scenario bundle — not the scenario set, not the print), subject to the
  SAME `rebid_cap_mw` as the solver and the reserve headroom.
- **Deadband baseline changes (Codex catch)**: the risk gate compares the
  forecast-predicted rebid value against **holding the stochastic Stage-1
  schedule**, not against the DA-only schedule. On HOLD, the physical
  schedule equals the Stage-1 commitment, the MtM term vanishes, and the day
  settles to `stochastic_hold_eur` — which can sit BELOW
  `da_only_revenue_eur`, because a scenario-aware commitment may sacrifice DA
  revenue for expected rebid value that the gate then declines to chase.
  Both `stochastic_hold_eur` and `forecast_uplift_eur` are therefore output
  and audited (§4); `captured_uplift_eur` stays measured against the
  unchanged `da_only` baseline.
- **The deadband gate applies on NO-RESERVE days only (Codex re-lock
  catch)**: the Stage-1 DA schedule is full-power and may occupy intervals
  where reserve headroom is committed, so "hold the Stage-1 schedule" can be
  physically infeasible on reserve days. v1 resolves this by ALWAYS
  re-dispatching Stage 2 when `reserve_mw > 0` anywhere in the day (the
  Stage-2 problem is headroom-feasible by construction);
  `min_rebid_uplift_eur` then has no effect on such days, which the batch
  summary and audit rows must state. A reserve-feasible hold *projection* is
  a possible v2 refinement, not v1 scope.
- Settlement at the realised IDA uses the implicit-MtM identity, plus
  capacity revenue when reserve is present.
- `min_rebid_uplift_eur = 0` reproduces the always-rebid variant, as today.

**Comparison baselines are computed at the SAME cap.** The batch runs three
policies internally under identical `rebid_cap_mw`, reserve series, and
window: (i) myopic sequential (Stage-1 = DA-only MILP, capped Stage-2), (ii)
deterministic co-opt (S=1, zero error path), (iii) stochastic (S=N). At
`rebid_cap_mw = ∞`, policy (i) ties the existing 9.2b sequential row exactly
(regression pin) — the existing uncapped strategy-table rows are NOT
modified by this feature.

**Myopic-baseline feasibility domain (Codex re-lock r3 catch)**: policies
(ii)/(iii) embed the Stage-2 constraints in their Stage-1 problem, so their
executed pairs are coopt-ceiling-feasible by construction. Policy (i)'s
Stage-1 is solved WITHOUT them, and with committed reserve a small cap can
make its capped, headroom-feasible Stage-2 set EMPTY (full-power DA interval
with `reserve = R` needs a deviation ≥ R to reach the headroom box). v1
therefore requires `rebid_cap_mw ≥ max_t reserve_mw_t` on reserve days —
**sufficient** by a proportional-scaling witness: scaling the myopic DA
schedule by `(power − max_reserve) / power` gives a Stage-2 candidate with
per-interval deviation `≤ |da_net|·max_reserve/power ≤ max_reserve ≤ cap`,
headroom satisfied, and SoC/terminal-neutral feasibility preserved (SoC is
linear in the flows, and a convex combination of the feasible original path
with the flat-at-initial path stays in bounds). The default
`rebid_cap_mw = power_mw` always satisfies the domain (reserve ≤ power by
construction). Below the domain the batch RAISES a clear validation error —
never a silent infeasibility; a recourse-feasible myopic Stage-1 (DA-only
objective + Stage-2 feasibility constraints) is the v2 refinement if
sub-reserve caps are ever needed.

## 6. Performance budget

Extensive form multiplies Stage-2 binaries by S (~1,056 binaries for a
15-min day at S=10: 96 Stage-1 + 960 Stage-2). Budget, enforced by parameter
caps + a batch-summary timing report (measured, not assumed — Increment B
must ship timings):

- Defaults: `S = 10`; hard cap `S ≤ 20` for hourly days, `S ≤ 10` for 15-min
  days.
- Target: single-day solve ≤ ~10 s worst case on the reference laptop; batch
  stays per-day independent (embarrassingly sequential, no continuous run).
- If HiGHS cannot hold the target at the default S, reduce the default S —
  never relax integrality (§2.2) and never silently sub-sample intervals.

## 7. Risk reporting (objective stays risk-neutral in v1)

Objective = expected value. Per-scenario totals feed a REPORTED risk block:
P10/P50/P90 of the scenario totals and **downside CVaR@90 — the mean of the
worst 10% of scenario revenues** (never the upper tail), labelled as in-model
scenario dispersion (NOT a market risk measure). A CVaR term in the objective
(`max (1−λ)·E + λ·CVaR`) is a deliberate v2 knob — adding it later must not
change v1 results at `λ=0`, which is the natural regression pin.

## 8. Comparison basis & pinned identities

Strategy integration adds rows via the batch of §5 (working titles: *DA+IDA1
co-optimised commitment (deterministic)* and *DA+IDA1 stochastic commitment
(S=…)*), scored — when reserve is present — over the 9.2b walk-forward
window (`triple_valid_days`, which may span one fewer day than the plain
DA+ID rows), sharing the 9.2b rows' reserve price series; with no reserve,
over the DA+ID valid days. Pinned identities / regression guards, in the
house cross-validation discipline (naive-reference + random price paths
BEFORE any merge of aggregation/attribution code):

1. **Decoupling theorem**: at `rebid_cap_mw = ∞` the stochastic Stage-1
   commitment equals the deterministic co-optimisation against the
   scenario-mean path, and — **because the generator mean-centres the error
   paths (§3)** — that path is the base forecast, so `stochastic − co_opt ≡
   0` for any S. (Embraces the Gemini review's mathematical observation as a
   test; the mean-centring premise is what makes the identity exact.)
2. **S=1 degeneracy**: S=1 with a zero error path ⇒ the deterministic
   point-forecast co-optimisation at the configured cap. This is
   **deliberately NOT equal to the myopic sequential row** — the co-opt
   Stage-1 sees the forecast, the sequential Stage-1 does not; their signed
   difference is the first headline delta (§1).
3. **IDA ≡ DA collapse (weakened form, no-reserve statement)**: with IDA
   scenarios and realised prices all ≡ DA prices on a NO-RESERVE day, the
   *realised stochastic row equals the DA-only row under default execution*.
   (Stage-1 asserted only through settlement: MtM cancellation makes the
   Stage-1 schedule degenerate/multi-optimal, so schedule-level equality
   would need a lexicographic tie-breaker — out of scope for v1; the
   assertion is on money, not schedules.) On a reserve day the collapse
   target is the **capped myopic reserve baseline** (identical headroom and
   capacity settlement), NOT the plain DA-only row.
4. `stochastic_realised ≤ coopt_ceiling` on every day (§2.4, guaranteed by
   construction at matched cap/reserve). Gaps to the LEGACY 9.2a ceiling are
   **signed** and carry information — a co-opt policy may beat it.
5. `expected_total_eur ≥` the myopic commitment's expected value under the
   SAME scenario set and cap (the stochastic solution cannot be worse
   in-sample than the commitment it generalises).
6. Decomposition identity of §4; `min_rebid_uplift_eur=0` reproduces the
   always-rebid variant; at `rebid_cap_mw = ∞` the capped myopic policy (i)
   ties the existing 9.2b sequential row.

## 9. Increment plan (after scope lock)

1. **Increment A — scenario generator** (`ida_scenarios.py` + tests; no
   solver). Ships the bundle contract of §3 including `base_forecast`,
   resolution partitioning, **mean-centring (with a scenario-mean ≡
   base-forecast test)**, seed/metadata.
2. **Increment B — solver core** (`solve_stochastic_da_id_dispatch` +
   synthetic tiny-scenario tests). Pins FIRST: the **coopt ceiling** (§2.4,
   solver-reuse implementation + `realised ≤ coopt_ceiling` and a case where
   the legacy ceiling is legitimately exceeded), deadband/hold semantics of
   §5 **including the no-reserve-only gating**, decoupling identity (§8-1),
   IDA≡DA collapse (§8-3, both reserve variants), and the myopic-baseline
   feasibility domain (§5: `cap ≥ max reserve` validation raises below the
   domain — finite-cap reserve counterexample test). Ships measured solve
   timings (§6).
3. **Increment C — batch + strategy rows**
   (`simulate_stochastic_da_id_batch`, the three-policy comparison at a
   common cap, risk block, timing report, 9.2b `triple_valid_days` window
   alignment).
4. **Increment D — cockpit panel + export/audit rows** (assumptions table
   gains scenario mode/S/`forecast_mode`/seed/`rebid_cap_mw`; export parity).
5. Real-data windows (imported IDA1 via the CSV path) only after C; cockpit
   copy states the LOO-vs-walk-forward caveat exactly as the sequential panel
   does.

Each increment is a separate PR in the established review lane.

## 10. Relationship to existing docs

- CLAUDE.md "Intraday uplift Phase 2" and "sequential DA+ID policy" bullets
  describe this as the "heavier future upgrade" — when Increment B lands they
  should point here. The original Phase-2 sketch's "explicit rebid-direction
  feasibility against the existing DA position" is realised here as the
  rebid volume cap (§2.1).
- The reserve red-line (headroom, not activation energy) and the
  system-vs-asset red-line are inherited unchanged from
  `docs/import-templates.md` and the cockpit captions.
