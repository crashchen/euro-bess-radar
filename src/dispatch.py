"""MILP-based multi-cycle BESS dispatch optimizer."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.analytics import _infer_interval_hours, _to_local

logger = logging.getLogger(__name__)
DISPATCH_VOM_COST_EUR_MWH = 0.5


def solve_daily_lp(
    prices: np.ndarray,
    dt: float,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    soc_init_frac: float = 0.5,
) -> dict:
    """Solve optimal charge/discharge schedule for one day via MILP.

    Args:
        prices: 1-D array of prices (EUR/MWh) for each interval.
        dt: Interval duration in hours (e.g. 1.0, 0.5, 0.25).
        power_mw: BESS power rating in MW.
        duration_hours: BESS energy duration in hours.
        efficiency: Round-trip efficiency (0-1).
        soc_init_frac: Initial SoC as fraction of capacity (0-1).

    Returns:
        Dict with keys: revenue_eur, p_charge, p_discharge, soc, n_cycles.
    """
    n = len(prices)
    if n == 0 or np.isnan(prices).any():
        return {
            "revenue_eur": 0.0,
            "p_charge": np.zeros(max(n, 0)),
            "p_discharge": np.zeros(max(n, 0)),
            "soc": np.zeros(max(n, 0) + 1),
            "n_cycles": 0.0,
        }

    capacity_mwh = power_mw * duration_hours
    soc_init = soc_init_frac * capacity_mwh
    sqrt_eff = math.sqrt(efficiency)

    # Decision variables: x = [p_charge_0..N-1, p_discharge_0..N-1, b_0..N-1] (3N)
    # b[t] is binary: 1 = charging mode, 0 = discharging mode. The LP relaxation
    # of pure mutual exclusion (sum constraint) lets the solver split power
    # between charging and discharging in the same interval, which is physically
    # impossible and lets it "burn" grid energy through round-trip losses to
    # earn revenue at negative prices. The binary b enforces strict exclusion.
    c = np.zeros(3 * n)
    c[:n] = (prices + DISPATCH_VOM_COST_EUR_MWH) * dt
    c[n:2 * n] = -(prices - DISPATCH_VOM_COST_EUR_MWH) * dt

    bounds = [(0.0, power_mw)] * (2 * n) + [(0.0, 1.0)] * n
    integrality = np.zeros(3 * n)
    integrality[2 * n:] = 1  # binary b[t]

    a_ub_rows = []
    b_ub_rows = []

    # Strict mutual exclusion via binary mode variable:
    #   p_charge[t]    - power_mw * b[t] <= 0       (charge only when b=1)
    #   p_discharge[t] + power_mw * b[t] <= power_mw (discharge only when b=0)
    for t in range(n):
        row_ch = np.zeros(3 * n)
        row_ch[t] = 1.0
        row_ch[2 * n + t] = -power_mw
        a_ub_rows.append(row_ch)
        b_ub_rows.append(0.0)

        row_dis = np.zeros(3 * n)
        row_dis[n + t] = 1.0
        row_dis[2 * n + t] = power_mw
        a_ub_rows.append(row_dis)
        b_ub_rows.append(power_mw)

    for t in range(1, n + 1):
        row_upper = np.zeros(3 * n)
        row_upper[:t] = sqrt_eff * dt
        row_upper[n:n + t] = -dt / sqrt_eff
        a_ub_rows.append(row_upper)
        b_ub_rows.append(capacity_mwh - soc_init)

        row_lower = np.zeros(3 * n)
        row_lower[:t] = -sqrt_eff * dt
        row_lower[n:n + t] = dt / sqrt_eff
        a_ub_rows.append(row_lower)
        b_ub_rows.append(soc_init)

    row_terminal = np.zeros(3 * n)
    row_terminal[:n] = sqrt_eff * dt
    row_terminal[n:2 * n] = -dt / sqrt_eff
    a_eq = row_terminal.reshape(1, -1)
    b_eq = np.array([0.0])

    a_ub = np.array(a_ub_rows)
    b_ub = np.array(b_ub_rows)

    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                     bounds=bounds, integrality=integrality, method="highs")

    if not result.success:
        logger.warning("MILP dispatch failed: %s", result.message)
        return {
            "revenue_eur": 0.0,
            "p_charge": np.zeros(n),
            "p_discharge": np.zeros(n),
            "soc": np.full(n + 1, soc_init),
            "n_cycles": 0.0,
        }

    p_charge = result.x[:n]
    p_discharge = result.x[n:2 * n]
    revenue = -result.fun  # negate because linprog minimizes

    # Reconstruct SoC trajectory
    soc = np.zeros(n + 1)
    soc[0] = soc_init
    for t in range(n):
        soc[t + 1] = soc[t] + (p_charge[t] * sqrt_eff - p_discharge[t] / sqrt_eff) * dt

    # Cycle count: total energy discharged / capacity
    total_discharged = np.sum(p_discharge) * dt
    n_cycles = total_discharged / capacity_mwh if capacity_mwh > 0 else 0.0

    return {
        "revenue_eur": round(float(revenue), 6),
        "p_charge": p_charge,
        "p_discharge": p_discharge,
        "soc": soc,
        "n_cycles": round(float(n_cycles), 4),
    }


def solve_daily_joint_capacity_lp(
    prices: np.ndarray,
    dt: float,
    capacity_price_eur_mw_h: float,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    soc_init_frac: float = 0.5,
    availability: float = 0.95,
) -> dict:
    """Jointly optimize DA dispatch and reserve capacity headroom via MILP.

    The reserve variable consumes power headroom in each interval but does not
    model activation energy, bid acceptance, or product-specific SoC duration.
    This keeps the estimate screening-grade while improving on a pure time-split
    heuristic.
    """
    n = len(prices)
    if n == 0 or np.isnan(prices).any():
        zeros = np.zeros(max(n, 0))
        return {
            "total_revenue_eur": 0.0,
            "da_revenue_eur": 0.0,
            "capacity_revenue_eur": 0.0,
            "p_charge": zeros,
            "p_discharge": zeros,
            "reserve_mw": zeros,
            "soc": np.zeros(max(n, 0) + 1),
            "n_cycles": 0.0,
            "avg_reserve_mw": 0.0,
        }

    capacity_mwh = power_mw * duration_hours
    soc_init = soc_init_frac * capacity_mwh
    sqrt_eff = math.sqrt(efficiency)
    capacity_price = max(float(capacity_price_eur_mw_h), 0.0)

    # Decision variables: [charge, discharge, reserve, b]  (4N total)
    # b[t] is binary: 1 = charging mode, 0 = discharging mode. Same MILP-based
    # mutual exclusion as solve_daily_lp; reserve is independent of b and only
    # competes for power headroom via the existing power-balance constraint.
    nv = 4 * n
    c = np.zeros(nv)
    c[:n] = (prices + DISPATCH_VOM_COST_EUR_MWH) * dt
    c[n:2 * n] = -(prices - DISPATCH_VOM_COST_EUR_MWH) * dt
    c[2 * n:3 * n] = -capacity_price * availability * dt

    bounds = [(0.0, power_mw)] * (3 * n) + [(0.0, 1.0)] * n
    integrality = np.zeros(nv)
    integrality[3 * n:] = 1

    a_ub_rows = []
    b_ub_rows = []

    for t in range(n):
        # Power-balance: charge + discharge + reserve <= power_mw
        row_power = np.zeros(nv)
        row_power[t] = 1.0
        row_power[n + t] = 1.0
        row_power[2 * n + t] = 1.0
        a_ub_rows.append(row_power)
        b_ub_rows.append(power_mw)

        # Strict mutual exclusion via binary mode b[t]
        row_ch = np.zeros(nv)
        row_ch[t] = 1.0
        row_ch[3 * n + t] = -power_mw
        a_ub_rows.append(row_ch)
        b_ub_rows.append(0.0)

        row_dis = np.zeros(nv)
        row_dis[n + t] = 1.0
        row_dis[3 * n + t] = power_mw
        a_ub_rows.append(row_dis)
        b_ub_rows.append(power_mw)

    for t in range(1, n + 1):
        row_upper = np.zeros(nv)
        row_upper[:t] = sqrt_eff * dt
        row_upper[n:n + t] = -dt / sqrt_eff
        a_ub_rows.append(row_upper)
        b_ub_rows.append(capacity_mwh - soc_init)

        row_lower = np.zeros(nv)
        row_lower[:t] = -sqrt_eff * dt
        row_lower[n:n + t] = dt / sqrt_eff
        a_ub_rows.append(row_lower)
        b_ub_rows.append(soc_init)

    row_terminal = np.zeros(nv)
    row_terminal[:n] = sqrt_eff * dt
    row_terminal[n:2 * n] = -dt / sqrt_eff

    result = linprog(
        c,
        A_ub=np.array(a_ub_rows),
        b_ub=np.array(b_ub_rows),
        A_eq=row_terminal.reshape(1, -1),
        b_eq=np.array([0.0]),
        bounds=bounds,
        integrality=integrality,
        method="highs",
    )

    if not result.success:
        logger.warning("Joint capacity MILP failed: %s", result.message)
        zeros = np.zeros(n)
        return {
            "total_revenue_eur": 0.0,
            "da_revenue_eur": 0.0,
            "capacity_revenue_eur": 0.0,
            "p_charge": zeros,
            "p_discharge": zeros,
            "reserve_mw": zeros,
            "soc": np.full(n + 1, soc_init),
            "n_cycles": 0.0,
            "avg_reserve_mw": 0.0,
        }

    p_charge = result.x[:n]
    p_discharge = result.x[n:2 * n]
    reserve_mw = result.x[2 * n:3 * n]
    da_revenue = float(
        ((prices - DISPATCH_VOM_COST_EUR_MWH) * p_discharge * dt).sum()
        - ((prices + DISPATCH_VOM_COST_EUR_MWH) * p_charge * dt).sum()
    )
    capacity_revenue = float((capacity_price * availability * reserve_mw * dt).sum())

    soc = np.zeros(n + 1)
    soc[0] = soc_init
    for t in range(n):
        soc[t + 1] = soc[t] + (p_charge[t] * sqrt_eff - p_discharge[t] / sqrt_eff) * dt

    total_discharged = np.sum(p_discharge) * dt
    n_cycles = total_discharged / capacity_mwh if capacity_mwh > 0 else 0.0

    return {
        "total_revenue_eur": round(da_revenue + capacity_revenue, 6),
        "da_revenue_eur": round(da_revenue, 6),
        "capacity_revenue_eur": round(capacity_revenue, 6),
        "p_charge": p_charge,
        "p_discharge": p_discharge,
        "reserve_mw": reserve_mw,
        "soc": soc,
        "n_cycles": round(float(n_cycles), 4),
        "avg_reserve_mw": round(float(reserve_mw.mean()), 6),
    }


def solve_joint_capacity_batch(
    price_df: pd.DataFrame,
    capacity_price_eur_mw_h: float,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    tz: str | None = None,
    soc_init_frac: float = 0.5,
    availability: float = 0.95,
) -> pd.DataFrame:
    """Run joint DA + reserve-capacity MILP for each local calendar day."""
    local = _to_local(price_df, tz)
    prices = local["price_eur_mwh"]

    records = []
    excluded_days = 0
    for date, group in prices.groupby(prices.index.date):
        sorted_group = group.sort_index()
        if sorted_group.isna().any():
            excluded_days += 1
            continue
        # Infer dt from this day's own index so mixed-resolution windows
        # (e.g. DE_LU crossing the 2025-10 60min→15min boundary) solve
        # each side at its native cadence instead of the frame mode.
        dt = _infer_interval_hours(sorted_group.index)
        result = solve_daily_joint_capacity_lp(
            sorted_group.values,
            dt=dt,
            capacity_price_eur_mw_h=capacity_price_eur_mw_h,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            soc_init_frac=soc_init_frac,
            availability=availability,
        )
        records.append({
            "date": date,
            "joint_total_revenue": result["total_revenue_eur"],
            "joint_da_revenue": result["da_revenue_eur"],
            "joint_capacity_revenue": result["capacity_revenue_eur"],
            "avg_reserve_mw": result["avg_reserve_mw"],
            "reserve_fraction": (
                round(result["avg_reserve_mw"] / power_mw, 6) if power_mw > 0 else 0.0
            ),
            "n_cycles": result["n_cycles"],
        })

    if not records:
        result = pd.DataFrame(
            columns=[
                "date", "joint_total_revenue", "joint_da_revenue",
                "joint_capacity_revenue", "avg_reserve_mw", "reserve_fraction",
                "n_cycles",
            ],
        )
        result.attrs["excluded_days_due_to_missing"] = excluded_days
        return result

    result = pd.DataFrame.from_records(records)
    result.attrs["excluded_days_due_to_missing"] = excluded_days
    return result


def solve_daily_da_id_dispatch(
    da_prices: np.ndarray,
    ida_prices: np.ndarray,
    dt: float,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    soc_init_frac: float = 0.5,
) -> dict:
    """Two-stage DA + ID dispatch with ex-post-perfect IDA knowledge.

    Stage 1 solves the DA-only MILP using ``da_prices``; this is the
    position committed before IDA1 clears. Stage 2 assumes the operator
    can observe the IDA1 print and re-dispatch physically at IDA prices,
    settling the DA-vs-IDA difference on the originally committed
    volumes.

    With sunk DA settlement and no rebid transaction costs, the
    optimal Stage 2 reduces to a standalone MILP at IDA prices: the
    DA position contributes only a constant mark-to-market term to
    total cash, not a constraint on physical flows. This is the
    "ex-post perfect-foresight" upper bound on rebid value, useful as
    a screening benchmark and the natural extension of the Phase-1
    average-spread heuristic.

    Cash accounting (per interval, then summed):
        da_value      = (da_discharge - da_charge) * DA * dt
        ida_lp_value  = (final_discharge - final_charge) * IDA * dt
                        (maximised by re-solving solve_daily_lp at IDA)
        implicit_mtm  = (da_discharge - da_charge) * IDA * dt
                        (= what holding DA to maturity is worth at IDA)
        rebid_uplift  = ida_lp_value - implicit_mtm
                        (extra cash from rebid; >=0 by construction
                         because the Stage-1 DA schedule is itself a
                         feasible Stage-2 solution at any IDA prices)
        total_cash    = da_value + rebid_uplift

    Returns:
        Dict with da_revenue_eur, ida_lp_value_eur, implicit_mtm_eur,
        rebid_uplift_eur, total_cash_eur, plus DA-stage and IDA-stage
        dispatch details (p_charge, p_discharge, soc, n_cycles).
    """
    n = len(da_prices)
    if n == 0 or n != len(ida_prices) or np.isnan(da_prices).any() or np.isnan(ida_prices).any():
        zeros = np.zeros(max(n, 0))
        return {
            "da_revenue_eur": 0.0,
            "ida_lp_value_eur": 0.0,
            "implicit_mtm_eur": 0.0,
            "rebid_uplift_eur": 0.0,
            "total_cash_eur": 0.0,
            "da_p_charge": zeros, "da_p_discharge": zeros,
            "ida_p_charge": zeros, "ida_p_discharge": zeros,
            "da_soc": np.zeros(max(n, 0) + 1),
            "ida_soc": np.zeros(max(n, 0) + 1),
            "da_n_cycles": 0.0, "ida_n_cycles": 0.0,
        }

    stage_1 = solve_daily_lp(
        da_prices, dt=dt, power_mw=power_mw, duration_hours=duration_hours,
        efficiency=efficiency, soc_init_frac=soc_init_frac,
    )
    stage_2 = solve_daily_lp(
        ida_prices, dt=dt, power_mw=power_mw, duration_hours=duration_hours,
        efficiency=efficiency, soc_init_frac=soc_init_frac,
    )

    da_net = stage_1["p_discharge"] - stage_1["p_charge"]
    # da_gross is the pre-VOM DA settlement value. solve_daily_lp returns
    # the post-VOM revenue; the original formula
    #     total = stage_1["revenue_eur"] + (ida_lp_value - implicit_mtm)
    # mixed a post-VOM term with a gross MtM term, double-counting VOM
    # whenever Stage 2 cancelled DA physical flow (no DA delivery -> no
    # real DA VOM cost). The correct accounting is:
    #     total_cash = da_gross - implicit_mtm + ida_lp_value
    # where implicit_mtm and ida_lp_value are both at IDA prices (gross
    # MtM cancels out, leaving the Stage-2 physical VOM exactly once).
    da_gross = float((da_net * da_prices * dt).sum())
    implicit_mtm = float((da_net * ida_prices * dt).sum())
    ida_lp_value = stage_2["revenue_eur"]  # post-VOM at Stage-2 dispatch
    total_cash = da_gross - implicit_mtm + ida_lp_value
    rebid_uplift = total_cash - stage_1["revenue_eur"]
    # Numerical floor: Stage-1 dispatch is always a feasible Stage-2
    # solution, so the optimal uplift is non-negative in theory. Clamp
    # any small solver-tolerance negative back to zero (and renormalise
    # total_cash so the decomposition still holds).
    if rebid_uplift < 0:
        rebid_uplift = 0.0
        total_cash = stage_1["revenue_eur"]

    return {
        "da_revenue_eur": round(stage_1["revenue_eur"], 6),
        "ida_lp_value_eur": round(ida_lp_value, 6),
        "implicit_mtm_eur": round(implicit_mtm, 6),
        "rebid_uplift_eur": round(rebid_uplift, 6),
        "total_cash_eur": round(total_cash, 6),
        "da_p_charge": stage_1["p_charge"],
        "da_p_discharge": stage_1["p_discharge"],
        "ida_p_charge": stage_2["p_charge"],
        "ida_p_discharge": stage_2["p_discharge"],
        "da_soc": stage_1["soc"],
        "ida_soc": stage_2["soc"],
        "da_n_cycles": stage_1["n_cycles"],
        "ida_n_cycles": stage_2["n_cycles"],
    }


def _schedule_value_at_prices(
    p_charge: np.ndarray,
    p_discharge: np.ndarray,
    prices: np.ndarray,
    dt: float,
) -> float:
    """Post-VOM cash of a fixed physical schedule settled at `prices`.

    Matches the objective sign convention of ``solve_daily_lp``:
    discharge earns ``(price - VOM)`` and charge costs ``(price + VOM)``
    per MWh. Used to re-settle a schedule chosen at forecast prices
    against the realised IDA print.
    """
    discharge_cash = ((prices - DISPATCH_VOM_COST_EUR_MWH) * p_discharge * dt).sum()
    charge_cash = ((prices + DISPATCH_VOM_COST_EUR_MWH) * p_charge * dt).sum()
    return float(discharge_cash - charge_cash)


def solve_sequential_da_id_dispatch(
    da_prices: np.ndarray,
    ida_forecast: np.ndarray,
    ida_realised: np.ndarray,
    dt: float,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    soc_init_frac: float = 0.5,
) -> dict:
    """Sequential DA + IDA1 policy under an imperfect IDA forecast.

    Unlike ``solve_daily_da_id_dispatch`` (which re-dispatches with
    ex-post-perfect knowledge of the realised IDA print and is therefore
    an upper bound), this models a screening forecast-following policy:

      Stage 1 — commit a DA position by solving the DA-only MILP on
                ``da_prices`` (the position locked in before IDA clears).
      Stage 2 — choose the physical re-dispatch that is optimal against
                the IDA *forecast* (``ida_forecast``), NOT the realised
                price. The desk only knows the forecast when it rebids.
      Settlement — the executed (forecast-optimal) schedule is settled at
                the *realised* IDA price; the DA position is marked to
                realised IDA via the same implicit-MtM accounting as the
                ex-post solver.

    When ``ida_forecast == ida_realised`` the Stage-2 decision coincides
    with the perfect-foresight solve and ``realised_total_eur`` equals the
    ceiling, so the ceiling is a strict upper bound and
    ``forecast_error_cost_eur >= 0`` by construction (the realised-optimal
    schedule cannot be beaten at realised prices).

    Returns keys:
        da_only_revenue_eur     — Stage-1 DA-only comparison baseline.
        realised_total_eur      — forecast-driven policy settled at realised.
        ceiling_total_eur       — ex-post perfect-foresight upper bound.
        forecast_error_cost_eur — ceiling - realised (>= 0).
        captured_uplift_eur     — realised - da_only (may be negative if
                                  the forecast misleads the rebid).
        ida_p_charge/discharge  — executed (forecast-optimal) schedule.
        ida_soc                 — SoC trajectory of the executed schedule.
        da_p_charge/discharge, da_soc — Stage-1 committed DA schedule.
    """
    n = len(da_prices)
    lengths_ok = n > 0 and len(ida_forecast) == n and len(ida_realised) == n
    has_nan = (
        np.isnan(da_prices).any()
        or np.isnan(ida_forecast).any()
        or np.isnan(ida_realised).any()
    )
    if not lengths_ok or has_nan:
        zeros = np.zeros(max(n, 0))
        return {
            "da_only_revenue_eur": 0.0,
            "realised_total_eur": 0.0,
            "ceiling_total_eur": 0.0,
            "forecast_error_cost_eur": 0.0,
            "captured_uplift_eur": 0.0,
            "da_p_charge": zeros, "da_p_discharge": zeros,
            "ida_p_charge": zeros, "ida_p_discharge": zeros,
            "da_soc": np.zeros(max(n, 0) + 1),
            "ida_soc": np.zeros(max(n, 0) + 1),
        }

    stage_1 = solve_daily_lp(
        da_prices, dt=dt, power_mw=power_mw, duration_hours=duration_hours,
        efficiency=efficiency, soc_init_frac=soc_init_frac,
    )
    # Stage 2: physical re-dispatch chosen against the FORECAST.
    stage_2_fc = solve_daily_lp(
        ida_forecast, dt=dt, power_mw=power_mw, duration_hours=duration_hours,
        efficiency=efficiency, soc_init_frac=soc_init_frac,
    )

    da_net = stage_1["p_discharge"] - stage_1["p_charge"]
    da_gross = float((da_net * da_prices * dt).sum())
    implicit_mtm = float((da_net * ida_realised * dt).sum())
    # Executed schedule settled at the realised IDA print.
    ida_value_realised = _schedule_value_at_prices(
        stage_2_fc["p_charge"], stage_2_fc["p_discharge"], ida_realised, dt,
    )
    realised_total = da_gross - implicit_mtm + ida_value_realised

    ceiling = solve_daily_da_id_dispatch(
        da_prices, ida_realised, dt=dt, power_mw=power_mw,
        duration_hours=duration_hours, efficiency=efficiency,
        soc_init_frac=soc_init_frac,
    )
    ceiling_total = ceiling["total_cash_eur"]
    # Clamp tiny solver-tolerance negatives; the realised-optimal schedule
    # is feasible for the forecast policy's settlement, so the ceiling is a
    # true upper bound.
    forecast_error_cost = max(ceiling_total - realised_total, 0.0)

    return {
        "da_only_revenue_eur": round(stage_1["revenue_eur"], 6),
        "realised_total_eur": round(realised_total, 6),
        "ceiling_total_eur": round(ceiling_total, 6),
        "forecast_error_cost_eur": round(forecast_error_cost, 6),
        "captured_uplift_eur": round(realised_total - stage_1["revenue_eur"], 6),
        "da_p_charge": stage_1["p_charge"],
        "da_p_discharge": stage_1["p_discharge"],
        "ida_p_charge": stage_2_fc["p_charge"],
        "ida_p_discharge": stage_2_fc["p_discharge"],
        "da_soc": stage_1["soc"],
        "ida_soc": stage_2_fc["soc"],
    }


def solve_dispatch_batch(
    price_df: pd.DataFrame,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    tz: str | None = None,
    soc_init_frac: float = 0.5,
) -> pd.DataFrame:
    """Run MILP dispatch for each calendar day in the price series.

    Args:
        price_df: DataFrame with DatetimeIndex and 'price_eur_mwh' column.
        power_mw: BESS power rating in MW.
        duration_hours: BESS energy duration in hours.
        efficiency: Round-trip efficiency (0-1).
        tz: IANA timezone for local-day grouping. None = use index as-is.
        soc_init_frac: Initial SoC fraction for each day (0-1).

    Returns:
        DataFrame with columns: [date, lp_revenue, n_cycles, lp_spread_eur_mwh].
    """
    local = _to_local(price_df, tz)
    prices = local["price_eur_mwh"]

    records = []
    excluded_days = 0
    for date, group in prices.groupby(prices.index.date):
        sorted_group = group.sort_index()
        if sorted_group.isna().any():
            excluded_days += 1
            continue
        # Per-day dt — see solve_joint_capacity_batch for the mixed-
        # resolution rationale.
        dt = _infer_interval_hours(sorted_group.index)
        result = solve_daily_lp(
            sorted_group.values,
            dt=dt,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            soc_init_frac=soc_init_frac,
        )
        energy_mwh = power_mw * duration_hours
        lp_spread = result["revenue_eur"] / energy_mwh if energy_mwh > 0 else 0.0
        records.append({
            "date": date,
            "lp_revenue": result["revenue_eur"],
            "n_cycles": result["n_cycles"],
            "lp_spread_eur_mwh": round(lp_spread, 6),
        })

    if not records:
        result = pd.DataFrame(
            columns=["date", "lp_revenue", "n_cycles", "lp_spread_eur_mwh"]
        )
        result.attrs["excluded_days_due_to_missing"] = excluded_days
        return result

    result = pd.DataFrame.from_records(records)
    result.attrs["excluded_days_due_to_missing"] = excluded_days
    return result
