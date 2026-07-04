# Scenario-based stochastic MILP — v1 design contract (scope only)

Status: **DRAFT — awaiting scope lock (Codex review).** No solver code ships
with this document; per the agreed ordering the solver core starts only after
this contract is locked.

## 1. Positioning

The stochastic MILP is an **upgrade of the Phase-7 sequential DA+ID policy**
(`dispatch.solve_sequential_da_id_dispatch`), NOT a replacement for the
perfect-foresight ceiling (`solve_daily_da_id_dispatch`) and NOT a new revenue
stream. Today Stage 1 commits the DA schedule **myopically** (DA-only MILP
that ignores the IDA opportunity entirely); the rebid stage then reacts to an
IDA point forecast. The stochastic upgrade makes the *Stage-1 DA commitment*
anticipate the IDA rebid opportunity across a **scenario set** of IDA price
paths instead of ignoring it — e.g. holding back cycles or positioning SoC
where the IDA distribution says rebids are likely to pay.

The value question it answers, in the strategy-comparison frame: *how much of
the gap between the 9.2b forecast-driven row and the perfect-foresight
ceiling is recoverable by committing DA against a distribution instead of
ignoring IDA?* Everything stays screening-grade historical replay — not
trading advice, not live dispatch.

### Non-goals (v1 red-lines)

- **No activation-energy or reBAP/imbalance terms in the objective.** Those
  remain separate, non-additive overlays with their own red-lines. v1 covers
  DA + IDA1 + reserve capacity only.
- **No stochastic reserve decision.** Reserve stays committed by the existing
  9.2b reserve-first walk-forward rule; it enters the stochastic solver as an
  **exogenous per-block power cap** (same `reserve_mw` semantics as
  `solve_daily_da_id_reserve_dispatch`), never as a recourse variable. A
  three-stage stochastic program (reserve → DA → IDA) is explicitly v2+.
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
| 2 (IDA, recourse) | Physical re-dispatch per scenario | That scenario's IDA path | Per-scenario `id_charge/id_discharge/id_soc` + binary mode per interval |

Extensive-form MILP: Stage-1 variables are shared across scenarios
(non-anticipativity by construction), Stage-2 variables are indexed by
scenario. Settlement accounting per scenario reuses the existing implicit-MtM
identity (`total_s = da_gross − implicit_mtm_s + stage2_total_s`), so the
objective is `max Σ_s w_s · total_s` — the DA leg is financial, physical
execution is the Stage-2 schedule, exactly as in the ex-post and sequential
solvers today.

**Binary mutual exclusion stays in BOTH stages, per scenario.** The
LP-relaxation degeneracy at negative prices (simultaneous charge+discharge
burning round-trip losses) applies to every scenario path; do not weaken this
to recover solve time — cap the scenario count instead (§6).

Reserve (when the day has committed reserve blocks): identical treatment to
the 9.2a/9.2b fused solver — the DA financial commitment does NOT occupy
physical headroom; each scenario's Stage-2 physical schedule satisfies
`stage2_charge_s + stage2_discharge_s + reserve ≤ power_mw` per interval.

## 3. Scenario generation contract (`ida_scenarios.build_ida_scenarios`)

New module `src/ida_scenarios.py`, sharing the forecast-history discipline of
`ida_forecast.build_ida_forecast` (`forecast_mode`: `loo` default /
`walk_forward` / `in_sample`, same caveats — LOO is a skill estimate, not
what a desk knew).

Two candidate modes; **v1 recommendation: (a) error resampling**, Codex to
confirm or override at scope lock:

- **(a) `error_resample` (recommended):** scenario_s = climatology point
  forecast + a resampled historical *daily error path* (realised − forecast,
  whole days, from the allowed history under `forecast_mode`). Preserves
  cross-hour error correlation; centres the set on the existing forecast so
  S=1 with a zero error path degenerates EXACTLY to today's sequential
  Stage-2 input; composes with the existing `compute_forecast_skill` report.
- **(b) `day_resample`:** bootstrap whole historical IDA days directly.
  Simpler, non-parametric, but the set is no longer centred on the
  climatology and the S=1 degeneracy to the sequential policy is lost.

Contract regardless of mode: equal weights `w_s = 1/S` in v1 (weighted
scenarios are a knob, not a v1 feature); every scenario is a full-day path on
the day's actual interval grid; scenario count `S` is a caller parameter with
defaults from §6; the generator reports `n_scenarios`, `mode`,
`forecast_mode`, and source-history coverage the same way `build_ida_forecast`
reports `coverage`/`fallback_points`, so the cockpit can label scenario
support honestly.

## 4. Solver contract (`dispatch.solve_stochastic_da_id_dispatch`)

Inputs: day frames (DA prices, realised IDA prices), the scenario set, BESS
params (power/duration/efficiency/VOM), optional per-interval `reserve_mw`,
optional `min_rebid_uplift_eur` (see §5). Outputs — superset of the
sequential solver's decomposition so `build_strategy_comparison` can consume
it unchanged:

- `da_only_revenue_eur` (baseline, unchanged definition),
- `realised_total_eur` (stochastic Stage-1 commitment, executed per §5,
  settled at the REALISED IDA print),
- `ceiling_total_eur` (= existing perfect-foresight `solve_daily_da_id_dispatch`),
- `expected_total_eur` (the in-solver objective value over scenarios —
  diagnostic only, never presented as realised revenue),
- `captured_uplift_eur = realised − da_only` (MAY be negative; do not clamp),
- `forecast_error_cost_eur = ceiling − realised ≥ 0`,
- per-scenario totals for the risk report (§7).

Identity pinned by test: `captured + forecast_error == ceiling − da_only`
(same as sequential).

## 5. Execution / settlement semantics (unchanged from 9.2b)

At replay time the desk still cannot see the print, so execution after the
stochastic Stage-1 commitment is IDENTICAL to the sequential policy: Stage 2
re-dispatches against the climatology point forecast (not the scenario set,
not the print), the deadband risk gate `min_rebid_uplift_eur` applies
unchanged, and settlement is at the realised IDA with the implicit-MtM
identity. **The ONLY behavioural delta vs 9.2b is the Stage-1 DA schedule.**
Therefore `stochastic_realised − sequential_realised` isolates the value of
scenario-aware commitment — that difference is the headline number of the
whole exercise, and it MAY be negative on bad windows (report it signed).

## 6. Performance budget

Extensive form multiplies Stage-2 binaries by S. Budget, enforced by
parameter caps + a batch-summary timing report (measured, not assumed):

- Defaults: `S = 10`; hard cap `S ≤ 20` hourly days, `S ≤ 10` for 15-min days
  (96 intervals ⇒ ~1k Stage-2 binaries at S=10, plus 96 Stage-1).
- Target: single-day solve ≤ ~10 s worst case on the reference laptop; batch
  stays per-day independent (embarrassingly sequential, no continuous run).
- If HiGHS cannot hold the target at the default S, reduce the default rather
  than relaxing binaries or silently sub-sampling intervals.

## 7. Risk reporting (objective stays risk-neutral in v1)

Objective = expected value. Per-scenario totals feed a REPORTED risk block:
P10/P50/P90 of the scenario totals and CVaR@90 as diagnostics, labelled as
in-model scenario dispersion (NOT a market risk measure). A CVaR term in the
objective (`max (1−λ)·E + λ·CVaR`) is a deliberate v2 knob — adding it later
must not change v1 results at `λ=0`, which is the natural regression pin.

## 8. Comparison basis & pinned identities

Strategy-comparison integration adds ONE row (working title: *DA + IDA1
stochastic commitment (S=…)*), scored over the same valid-day window as the
DA+ID rows. Pinned identities / regression guards, in the house
cross-validation discipline (naive-reference + random price paths BEFORE any
merge of aggregation/attribution code):

1. `S=1` with a zero error path ⇒ Stage-1 problem equals the deterministic
   DA+forecast co-optimisation; with IDA scenario ≡ DA prices the rebid adds
   nothing and the solution collapses to the DA-only MILP (ties the 9.2b
   sequential row with no rebid).
2. `stochastic_realised ≤ ceiling` on every day (ceiling is perfect
   foresight).
3. `expected_total_eur ≥` the Stage-1-myopic expected value under the SAME
   scenario set (the stochastic solution cannot be worse in-sample than the
   myopic commitment it generalises — solver-level sanity).
4. Decomposition identity of §4; deadband `min_rebid_uplift_eur=0` reproduces
   the always-rebid variant.
5. Random-path cross-validation: on synthetic days where IDA ≡ DA, the
   stochastic row equals the DA-only row exactly.

## 9. Increment plan (after scope lock)

1. **Increment A — scenario generator** (`ida_scenarios.py` + tests; no
   solver). Small, independently reviewable.
2. **Increment B — solver core** (`solve_stochastic_da_id_dispatch` +
   synthetic tiny-scenario tests + identity pins of §8).
3. **Increment C — batch + strategy row**
   (`simulate_stochastic_da_id_batch`, comparison row, risk block, timing
   report).
4. **Increment D — cockpit panel + export/audit rows** (assumptions table
   gains scenario mode/S/forecast_mode; export parity).
5. Real-data windows (imported IDA1 via the CSV path) only after C; cockpit
   copy states the LOO-vs-walk-forward caveat exactly as the sequential panel
   does.

Each increment is a separate PR in the established review lane.

## 10. Relationship to existing docs

- CLAUDE.md "Intraday uplift Phase 2" and "sequential DA+ID policy" bullets
  describe this as the "heavier future upgrade" — when Increment B lands they
  should point here.
- The reserve red-line (headroom, not activation energy) and the
  system-vs-asset red-line are inherited unchanged from
  `docs/import-templates.md` and the cockpit captions.
