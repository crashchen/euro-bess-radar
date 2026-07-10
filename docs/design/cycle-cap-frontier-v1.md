# Cycle-cap × degradation net-revenue frontier — v1 design contract

Status: **r3 — Gemini PR review (2 medium, both accepted, Codex concurred):
the degradation backstop now SCALES with the bound cushion
(`max(1e-6, 10·tol_z)` — fixes the known v1/v2 tolerance-crossing edge in
this new contract rather than inheriting it) and the uplift-vs-uncapped
denominator is ABSOLUTE (sign stays meaningful at negative uncapped net).
r2 — Codex round-2 verified all round-1 revisions RESOLVED; four local
consistency edits applied (z* in solver-objective space; one fallback field
name; cap-compliance tolerance in MWh; solver-failure observability +
exclusion wording). Formal lock = merge of the design PR.** Increment progress: **F-A landed** (`solve_daily_lp` cap + min-FEC tie-break + `success` key + pins §6-1/2/3/9); **F-B landed** (`cycle_frontier.compute_cycle_cap_frontier` pure calc + pins §6-4..8/10); **F-C landed** (cockpit expander + Excel export + AppTest smoke — the frontier line is complete). No code lands until locked, per the house playbook
(`stochastic-milp-v1.md` / `stochastic-milp-v2-reserve.md`: contract → review
rounds → lock → small PRs).

Origin: r2b "How BESS Earn Money" workshop deck (2026-06) + the Paternò Sicily
45MW/270MWh cycle debate. The deck's third-party trader table prices
{duration} × {cycle cap} → gross revenue; radar can reproduce it AND take the
step the trader did not: subtract the wear cost of each cap's incremental
throughput and read off the NET-revenue-optimal cycle cap. The question this
answers is an investment question ("is 1.2 cycles/day or 2 cycles/day worth
more, net of battery wear?"), not a dispatch-strategy question.

## 1. Positioning & red-lines

- **Screening-grade investment table, not an electrochemical model.** The wear
  cost is a LINEAR capex-amortisation proxy — EUR per full-equivalent cycle
  (FEC) = `capex_eur_kwh · capacity_kwh / cycle_life` (the existing
  `degradation.calculate_degradation_cost`), with NO DoD / C-rate /
  temperature / calendar-interaction dependence. Every surface (caption,
  assumptions table, export) states this.
- **DA-only merchant basis in v1.** The frontier sweeps the DA-only daily
  MILP (`solve_daily_lp`), per-day standalone terminal-neutral — the same
  basis as the trader table and the cheapest correct thing. IDA rebid /
  reserve co-opt / multi-day SoC carry interactions with a cycle cap are
  explicitly OUT of v1 (see §7).
- **The cap constrains the OPTIMISER, not the accounting.** A capped run's
  reported EFC is the realised optimum under the constraint (≤ cap), never
  clamped after the fact.
- **The optimiser is WEAR-BLIND; only the table nets wear.** `gross_eur` is
  raw `solve_daily_lp` revenue — already net of the in-objective
  `DISPATCH_VOM_COST_EUR_MWH = 0.5` EUR/MWh on both legs — while `wear_eur`
  is applied EX-POST in the frontier table only. No capture haircut is
  applied anywhere in v1. VOM and wear are DIFFERENT costs (operational vs
  capex amortisation) and are never merged into one coefficient; the
  assumptions rows state both.
- **Canonical min-FEC tie-break on frontier solves.** Equal-gross optima can
  differ in FEC (degenerate/flat days), which would make `wear_eur` and
  `net_eur` solver-tie-sensitive — the same class of problem the stochastic
  canonical selectors exist for. Frontier solves therefore run a two-pass
  lexicographic per day: pass 1 optimal revenue `z*`; pass 2 fixes revenue at
  `z*` (bound cushion `tol_z = 1e-9·(1+|z*|)` in the solver's minimisation
  form; degradation backstop `> z* + max(1e-6, 10·tol_z)` → keep pass 1 — the
  backstop SCALES with the bound cushion instead of being a fixed absolute,
  so the two can never cross at large `|z*|`. This deliberately FIXES the
  known v1-#41/v2-§2.2 edge where the fixed `1e-6` backstop undercuts the
  relative cushion once `|z*| ≳ 1e3` and trips benign fallbacks; those
  contracts are locked with the old constants — this scaled pair is the
  erratum candidate to backport as v2.1 if their real-data batches trip it)
  and
  minimises discharged energy. Sign convention: `z*` is the scipy `linprog`
  optimum of the NEGATED-revenue objective (`z* = result.fun`; the reported
  revenue is `-result.fun`), so the inequalities above are written in the
  SOLVER'S minimisation space — exactly the `_canonicalize_stage1` discipline.
  Exposed as
  `solve_daily_lp(..., min_throughput_tiebreak: bool = False)` — default off,
  so no existing caller changes behaviour; the frontier always passes True.
  A pass-2 failure falls back to pass 1; the frontier summary counts such
  days as `n_tiebreak_fallback_days` (ONE contract field, #43 pattern).
- **No cycle-life curve fitting.** `cycle_life` is a user knob (default
  `degradation.DEFAULT_CYCLE_LIFE = 6000`), not something estimated from the
  cap (a real cycle-life vs DoD curve is a non-goal; the linear proxy is the
  contract).

## 2. Solver change (increment F-A)

`dispatch.solve_daily_lp` gains one optional parameter:

```python
def solve_daily_lp(
    ..., max_efc_per_day: float | None = None,
    min_throughput_tiebreak: bool = False,
) -> dict:
```

Both new parameters are appended AFTER the existing ones (positional API
compatibility for every current caller).

- Semantics: **discharged energy per solved day** is capped —
  `Σ_t p_discharge_t · dt ≤ max_efc_per_day · capacity_mwh`. ONE extra A_ub
  row; FEC counted on the discharge leg, identical to the existing
  `n_cycles = total_discharged / capacity_mwh` convention (charge leg follows
  via efficiency and the SoC window; no separate charge cap).
- `None` (default) ⇒ no row appended ⇒ **bit-identical to today** for every
  existing caller (regression pin).
- `max_efc_per_day < 0` ⇒ `ValueError`. `= 0` ⇒ a feasible forced-idle day
  (zero dispatch, zero revenue — the constraint admits the zero vector).
- A cap large enough to be slack (e.g. `24/duration_hours`) reproduces the
  uncapped optimum exactly (pin).
- **FEC source is RAW**: the frontier's wear and cap-compliance accounting use
  `p_discharge.sum() · dt / capacity_mwh` recomputed from the returned
  schedule — never the ROUNDED `n_cycles` convenience field (4-decimal
  rounding would leak into money).
- The knob is NOT added to `solve_daily_da_id_dispatch`,
  `solve_daily_joint_capacity_lp`, the continuous-horizon replay, or any
  stochastic solver in v1 (§7 non-goals). The continuous multi-day MILP would
  need one row per contained day — mechanically easy but a different
  cap-vs-carry semantics discussion; deferred deliberately.

## 3. Frontier computation (increment F-B, pure calc — no UI)

New module `src/cycle_frontier.py`:

```python
def compute_cycle_cap_frontier(
    da_prices: pd.DataFrame,
    *,
    dates: list[date] | None = None,
    tz: str | None = None,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    capex_eur_kwh: float,
    cycle_life: float = DEFAULT_CYCLE_LIFE,
    cycle_caps: Sequence[float] = DEFAULT_CYCLE_CAPS,
) -> tuple[pd.DataFrame, dict]
```

- `DEFAULT_CYCLE_CAPS = (0.5, 1.0, 1.2, 1.5, 2.0, 3.0, None)` — `None` =
  uncapped reference row (labelled "uncapped"); the values bracket the r2b /
  Sicily debate points and 3.0 covers 1h systems (Q2 closed). User-overridable.
- Cap-set normalisation: finite caps must be non-negative floats (negative or
  non-finite values raise); duplicates deduped; rows ordered ascending finite
  caps then uncapped last; at most one `None`.
- Per cap: loop the window's clean local days (same day-selection rules as the
  existing replay batches: `_select_local_day`, regular-UTC-day check),
  `solve_daily_lp(max_efc_per_day=cap)` per day, sum revenue and FEC.
  **Identical valid-day set across caps** — a day is excluded for DATA or
  SOLVER reasons only, never for cap reasons, and any exclusion applies to
  ALL caps, so every row is co-temporal (the strategy-comparison lesson).
- Wear: `fec_total × cost_per_cycle_eur` with
  `cost_per_cycle_eur = capex_eur_kwh · capacity_kwh / cycle_life` (via the
  existing `calculate_degradation_cost` — no new formula). Also expose the
  equivalent **EUR per MWh discharged** (`cost_per_cycle / capacity_mwh`) as
  an audit convenience.
- Per-row outputs (one row per cap):
  `cycle_cap` (float or NaN for uncapped), `label`,
  `gross_eur` (window), `avg_efc_per_day` (realised),
  `wear_eur` (window), `net_eur = gross − wear`,
  `gross_eur_per_mw_yr`, `wear_eur_per_mw_yr`, `net_eur_per_mw_yr`
  (365.25 i.i.d. annualisation over the common valid days — the house
  convention), `net_delta_vs_uncapped_eur`, and `net_uplift_vs_uncapped_pct`
  = `(net_eur − uncapped_net_eur) / abs(uncapped_net_eur)` — the ABSOLUTE
  denominator keeps the sign meaningful when the uncapped net is negative
  (wear can exceed gross: −5 vs −10 is a +50% improvement, not −50%); NaN
  when |uncapped net| is within the window tolerance of zero (no division
  blow-ups).
- Summary dict: `valid_days`, `excluded_days`, `cost_per_cycle_eur`,
  `wear_eur_per_mwh_discharged`, `cycle_life`, `capex_eur_kwh`,
  `best_cap_label`, `frontier_flat`, `n_tiebreak_fallback_days`.
- **One tolerance policy** (all comparisons in this module):
  `NET_TOL_EUR_PER_MW_YR = 1.0` on annualised per-MW figures, converted to a
  window-EUR tolerance via `× power_mw × valid_days / 365.25` where a window
  quantity is compared. Cap compliance uses solver tolerance `1e-6` MWh.
- **Best-cap rule (r1, replaces the conflicting r0 wording): the LOWEST
  FINITE cap whose `net_eur` is within tolerance of the maximum wins;
  "uncapped" wins only when it is STRICTLY better than every finite cap by
  more than the tolerance.** (Prefer committing less wear for equal money —
  the Stage-0 selector philosophy — while an uncapped strict winner stays
  visible.) `frontier_flat` = True when max(net) − min(net) across rows is
  within tolerance ("the cap barely matters here" — honest flag).
- Extra column per row (Q3 closed): `cycle_limited_life_years` =
  `cycle_life / (avg_efc_per_day × 365.25)` via the existing
  `estimate_battery_lifetime` (cycle-limited figure ONLY — calendar/effective
  life is NOT surfaced unless the export also states that assumption).

- Degenerate handling: `dates=None` ⇒ all available local dates (house
  convention); zero valid days ⇒ a TYPED empty frame + a summary with
  `valid_days=0` and NaN-free scalars (no division by zero); a solver failure
  on any day excludes that day FOR ALL caps (the co-temporal rule) and counts
  it in `excluded_days`. **Failure observability (F-A scope)**: today
  `solve_daily_lp` returns a zero schedule on solver failure with NO flag —
  indistinguishable from a genuine zero-revenue day — so F-A adds a
  `success: bool` key to its return dict (additive; existing callers
  unaffected) and the frontier excludes days where ANY cap's solve reports
  `success=False`. Day selection reuses the replay helpers
  (`_select_local_day` + regular-UTC-day check), so DST days follow the
  existing convention.

Identities to pin (§6): gross is non-decreasing in the cap; realised
`avg_efc_per_day ≤ cap + tol` per row; `net = gross − wear` exactly;
the uncapped row equals the slack-cap row; cap=0 row has zero gross and zero
wear; at `capex = 0` the net frontier equals the gross frontier and the
best-cap rule picks the LOWEST cap whose net ties the maximum within
tolerance (uncapped only on a strict win); the min-FEC tie-break never
changes `z*` and reduces (never raises) reported FEC on a degenerate fixture.

## 4. Cockpit placement (increment F-C)

- New expander in the Simulation Cockpit, immediately after the multi-day
  replay section (it consumes the same loaded DA window):
  **"Cycle-cap × degradation net-revenue frontier"**.
- Inputs: cycle-cap set (multiselect over the defaults + free entry),
  `cycle_life` number input (default 6000, help text names the linear-proxy
  red-line), capex read from the SIDEBAR (single source of truth — no second
  capex knob; the panel shows the value it inherited).
- Output: the table (net column highlighted, best row flagged) + one grouped
  bar chart (gross vs net EUR/MW/yr per cap — NOT two separate charts; the
  gap IS the wear). Caption: linear wear proxy, DA-only basis, per-day
  standalone solves, screening not bankable.
- Excel export via the existing `cockpit_tables_to_excel` pattern: the
  frontier table + assumptions rows (cost per cycle, EUR/MWh-discharged
  equivalent, cycle life, capex source, DA-only basis, i.i.d. annualisation,
  linear-proxy red-line).
- `build_assumptions_table` is NOT extended in v1 (the panel's export carries
  its own rows; the global audit table only carries sidebar/config state).

## 5. Non-goals (v1)

- No IDA / reserve / stochastic interaction with the cap (the frontier is
  DA-only; a capped DA+ID frontier is v2 IF the v1 table proves useful).
- No duration sweep in v1 — the table runs at the sidebar duration. The r2b
  deck's 2D {duration}×{cap} matrix needs re-solving the window per duration;
  v1.1 candidate once runtimes are known.
- No DoD/C-rate/temperature wear model, no cycle-life curve, no warranty
  contract modelling (throughput warranties differ from wear economics).
- No continuous-horizon (multi-day carry) capped dispatch.
- No automatic cap recommendation into other panels — the frontier is a
  table, not a global constraint switch.

## 6. Pinned identities (regression tests before UI)

1. `max_efc_per_day=None` leaves `solve_daily_lp` bit-identical (existing
   tests untouched + an explicit equality pin).
2. Slack cap ≡ uncapped; `cap=0` ⇒ zero dispatch, feasible; negative raises.
3. Realised per-day EFC ≤ cap + solver tolerance on every solved day.
4. Gross monotone non-decreasing in the cap across the sweep.
5. `net = gross − FEC·cost_per_cycle` exactly; wear consistent with
   `calculate_degradation_cost` (no second formula).
6. Common valid-day set across all rows (excluding a day excludes it for
   every cap).
7. `capex=0` ⇒ net ≡ gross; best-cap = lowest cap tying the max net within
   tolerance (uncapped only on a strict win).
8. `frontier_flat` flag behaviour on a flat-price fixture.
9. Min-FEC tie-break: objective preserved (backstop), FEC never higher than
   pass-1 on a degenerate fixture, `min_throughput_tiebreak=False` default
   bit-identical, fallback counted (#43 pattern).
10. Raw-FEC accounting: wear computed from the schedule sum, not the rounded
    `n_cycles` field (a fixture where the rounding would differ).

## 7. Increment plan (after lock)

- **F-A**: `solve_daily_lp(max_efc_per_day=..., min_throughput_tiebreak=...)`
  + the `success` return key + pins §6-1/2/3/9.
- **F-B**: `cycle_frontier.compute_cycle_cap_frontier` + pins §6-4..8.
- **F-C**: cockpit expander + export + AppTest smoke.

Each increment its own PR with dual review (Codex + agy/Gemini), per the
house lane.

### Reporting addendum: realised DA VWAPs

The completed frontier also reports `charge_vwap_eur_mwh` and
`discharge_vwap_eur_mwh` for each cap. These are energy-weighted prices of the
actual DA-only physical schedule across the common valid-day window:
`sum(price x charged/discharged MWh) / sum(charged/discharged MWh)`. They are
reporting diagnostics only: negative prices remain signed, a missing physical
leg is `NaN` rather than a made-up zero, and neither VOM nor a capture haircut
is included. The fields do not enter the solver, gross revenue, wear, net
revenue, best-cap rule, or any red line above.

## 8. Resolved questions (all closed at r1 — nothing open)

- ~~Q1~~ **CLOSED: discharge-leg EFC** (industry convention + matches the
  existing `n_cycles` accounting; Codex r1 confirmed).
- ~~Q2~~ **CLOSED: defaults `(0.5, 1.0, 1.2, 1.5, 2.0, 3.0, None)`** (3.0
  added for 1h systems; Codex r1).
- ~~Q3~~ **CLOSED: YES — `cycle_limited_life_years` column**, cycle-limited
  figure only; calendar/effective life stays out unless the export states
  that extra assumption (Codex r1).
