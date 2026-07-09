# Cycle-cap × degradation net-revenue frontier — v1 design contract

Status: **r0 draft (CC-authored, 2026-07-09). Not locked — no code lands until
this contract is locked**, per the house playbook (`stochastic-milp-v1.md` /
`stochastic-milp-v2-reserve.md`: contract → review rounds → lock → small PRs).

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
- **No cycle-life curve fitting.** `cycle_life` is a user knob (default
  `degradation.DEFAULT_CYCLE_LIFE = 6000`), not something estimated from the
  cap (a real cycle-life vs DoD curve is a non-goal; the linear proxy is the
  contract).

## 2. Solver change (increment F-A)

`dispatch.solve_daily_lp` gains one optional parameter:

```python
def solve_daily_lp(..., max_efc_per_day: float | None = None) -> dict:
```

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

- `DEFAULT_CYCLE_CAPS = (0.5, 1.0, 1.2, 1.5, 2.0, None)` — `None` = uncapped
  reference row (labelled "uncapped"), the values chosen to bracket the r2b /
  Sicily debate points. User-overridable.
- Per cap: loop the window's clean local days (same day-selection rules as the
  existing replay batches: `_select_local_day`, regular-UTC-day check),
  `solve_daily_lp(max_efc_per_day=cap)` per day, sum revenue and FEC.
  **Identical valid-day set across caps** — a day is excluded for data
  reasons only, never for cap reasons, so every row is co-temporal (the
  strategy-comparison lesson).
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
  convention), `net_uplift_vs_uncapped_pct`.
- Summary dict: `valid_days`, `excluded_days`, `cost_per_cycle_eur`,
  `wear_eur_per_mwh_discharged`, `cycle_life`, `capex_eur_kwh`,
  `best_cap_label` = argmax over `net_eur` (ties → the LOWEST cap: prefer
  committing less wear for equal money, echoing the Stage-0 selector
  philosophy), and `frontier_flat` = True when max(net) − min(net) is within
  a small tolerance (an honest "the cap barely matters here" flag).

Identities to pin (§6): gross is non-decreasing in the cap; realised
`avg_efc_per_day ≤ cap + tol` per row; `net = gross − wear` exactly;
the uncapped row equals the slack-cap row; cap=0 row has zero gross and zero
wear; at `capex = 0` the net frontier equals the gross frontier and
`best_cap` = the uncapped row (ties resolved to the lowest equal-net cap).

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
7. `capex=0` ⇒ net ≡ gross and `best_cap` = lowest-tie resolution pin.
8. `frontier_flat` flag behaviour on a flat-price fixture.

## 7. Increment plan (after lock)

- **F-A**: `solve_daily_lp(max_efc_per_day=...)` + pins §6-1/2/3.
- **F-B**: `cycle_frontier.compute_cycle_cap_frontier` + pins §6-4..8.
- **F-C**: cockpit expander + export + AppTest smoke.

Each increment its own PR with dual review (Codex + agy/Gemini), per the
house lane.

## 8. Open questions (to resolve before lock)

- Q1 — should the FEC cap count the discharge leg only (proposed, matches
  `n_cycles`) or (charge+discharge)/2? Discharge-only is the industry EFC
  convention and the existing accounting; flag for reviewer confirmation.
- Q2 — `DEFAULT_CYCLE_CAPS` set: is (0.5, 1.0, 1.2, 1.5, 2.0, uncapped) the
  right bracket, and should 3.0 be included for 1h systems?
- Q3 — should the frontier ALSO report the cycle-limited lifetime per cap
  (`estimate_battery_lifetime`) as an extra column (years to 6000 FEC at the
  realised EFC rate)? Cheap and investment-relevant; proposed YES.
