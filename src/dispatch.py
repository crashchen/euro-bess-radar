"""LP-based multi-cycle BESS dispatch optimizer."""

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
    """Solve optimal charge/discharge schedule for one day via LP.

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

    # Decision variables: x = [p_charge_0..N-1, p_discharge_0..N-1]  (2N)
    # Objective: minimize c^T x  where charging costs money, discharging earns.
    # A small cycling cost (VOM, EUR/MWh) suppresses unrealistic micro-cycling.
    c = np.zeros(2 * n)
    c[:n] = (prices + DISPATCH_VOM_COST_EUR_MWH) * dt
    c[n:] = -(prices - DISPATCH_VOM_COST_EUR_MWH) * dt

    # Variable bounds: 0 <= p_charge, p_discharge <= power_mw
    bounds = [(0.0, power_mw)] * (2 * n)

    # SoC constraints via cumulative sums:
    # soc(t) = soc_init + sum_{k=0}^{t-1} (p_charge[k]*sqrt_eff - p_discharge[k]/sqrt_eff) * dt
    # We need: 0 <= soc(t) <= capacity_mwh  for t = 1..N

    a_ub_rows = []
    b_ub_rows = []

    # Mutual exclusion: p_charge[t] + p_discharge[t] <= power_mw for each t.
    # This prevents physically impossible simultaneous charge+discharge.
    for t in range(n):
        row_excl = np.zeros(2 * n)
        row_excl[t] = 1.0
        row_excl[n + t] = 1.0
        a_ub_rows.append(row_excl)
        b_ub_rows.append(power_mw)

    for t in range(1, n + 1):
        # Upper SoC bound: cumulative charge effect <= capacity - soc_init
        row_upper = np.zeros(2 * n)
        row_upper[:t] = sqrt_eff * dt
        row_upper[n:n + t] = -dt / sqrt_eff
        a_ub_rows.append(row_upper)
        b_ub_rows.append(capacity_mwh - soc_init)

        # Lower SoC bound: -(cumulative) <= soc_init
        row_lower = np.zeros(2 * n)
        row_lower[:t] = -sqrt_eff * dt
        row_lower[n:n + t] = dt / sqrt_eff
        a_ub_rows.append(row_lower)
        b_ub_rows.append(soc_init)

    # Terminal SoC equality: soc(N) = soc_init (energy-neutral over the day).
    # Equality prevents the LP from exploiting negative prices by ending with
    # surplus SoC that vanishes between daily solves.
    row_terminal = np.zeros(2 * n)
    row_terminal[:n] = sqrt_eff * dt    # charge adds to SoC
    row_terminal[n:] = -dt / sqrt_eff   # discharge removes from SoC
    a_eq = row_terminal.reshape(1, -1)
    b_eq = np.array([0.0])  # net SoC change = 0

    a_ub = np.array(a_ub_rows)
    b_ub = np.array(b_ub_rows)

    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    if not result.success:
        logger.warning("LP dispatch failed: %s", result.message)
        return {
            "revenue_eur": 0.0,
            "p_charge": np.zeros(n),
            "p_discharge": np.zeros(n),
            "soc": np.full(n + 1, soc_init),
            "n_cycles": 0.0,
        }

    p_charge = result.x[:n]
    p_discharge = result.x[n:]
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
    """Jointly optimize DA dispatch and reserve capacity headroom for one day.

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

    # Decision variables: [charge_0..N, discharge_0..N, reserve_0..N]
    c = np.zeros(3 * n)
    c[:n] = (prices + DISPATCH_VOM_COST_EUR_MWH) * dt
    c[n:2 * n] = -(prices - DISPATCH_VOM_COST_EUR_MWH) * dt
    c[2 * n:] = -capacity_price * availability * dt

    bounds = [(0.0, power_mw)] * (3 * n)
    a_ub_rows = []
    b_ub_rows = []

    for t in range(n):
        row_power = np.zeros(3 * n)
        row_power[t] = 1.0
        row_power[n + t] = 1.0
        row_power[2 * n + t] = 1.0
        a_ub_rows.append(row_power)
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

    result = linprog(
        c,
        A_ub=np.array(a_ub_rows),
        b_ub=np.array(b_ub_rows),
        A_eq=row_terminal.reshape(1, -1),
        b_eq=np.array([0.0]),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        logger.warning("Joint capacity LP failed: %s", result.message)
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
    reserve_mw = result.x[2 * n:]
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
    """Run joint DA + reserve-capacity LP for each local calendar day."""
    local = _to_local(price_df, tz)
    prices = local["price_eur_mwh"]
    dt = _infer_interval_hours(local.index)

    records = []
    for date, group in prices.groupby(prices.index.date):
        result = solve_daily_joint_capacity_lp(
            group.sort_index().values,
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
        return pd.DataFrame(
            columns=[
                "date", "joint_total_revenue", "joint_da_revenue",
                "joint_capacity_revenue", "avg_reserve_mw", "reserve_fraction",
                "n_cycles",
            ],
        )

    return pd.DataFrame.from_records(records)


def solve_dispatch_batch(
    price_df: pd.DataFrame,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    tz: str | None = None,
    soc_init_frac: float = 0.5,
) -> pd.DataFrame:
    """Run LP dispatch for each calendar day in the price series.

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
    dt = _infer_interval_hours(local.index)

    records = []
    for date, group in prices.groupby(prices.index.date):
        sorted_group = group.sort_index()
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
        return pd.DataFrame(columns=["date", "lp_revenue", "n_cycles", "lp_spread_eur_mwh"])

    return pd.DataFrame.from_records(records)
