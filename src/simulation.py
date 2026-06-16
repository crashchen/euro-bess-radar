"""Interval-level BESS dispatch replay for the Simulation Cockpit."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.analytics import _infer_interval_hours, _to_local
from src.degradation import (
    DEFAULT_CYCLE_LIFE,
    DEFAULT_EOL_CAPACITY_PCT,
    calculate_degradation_cost,
)
from src.dispatch import DISPATCH_VOM_COST_EUR_MWH, solve_daily_da_id_dispatch, solve_daily_lp

DAYS_PER_YEAR = 365.25
_SIM_COLUMNS = [
    "timestamp",
    "local_time",
    "price_eur_mwh",
    "p_charge_mw",
    "p_discharge_mw",
    "net_dispatch_mw",
    "soc_mwh",
    "soc_pct",
    "available_charge_mw",
    "available_discharge_mw",
    "interval_revenue_eur",
    "cumulative_revenue_eur",
    "interval_throughput_mwh",
    "cumulative_throughput_mwh",
]


def empty_simulation_result(reason: str = "") -> dict[str, Any]:
    """Return a safe empty cockpit result with stable keys."""
    summary = {
        "total_revenue_eur": 0.0,
        "annualized_eur_per_mw": 0.0,
        "number_of_trades": 0,
        "physical_throughput_mwh": 0.0,
        "traded_volume_mwh": 0.0,
        "rebalancing_factor": 0.0,
        "daily_fce": 0.0,
        "avg_c_rate": 0.0,
        "max_depth_of_discharge_pct": 0.0,
        "degradation_cost_eur": 0.0,
        "soh_delta_pct": 0.0,
        "reason": reason,
    }
    return {"summary": summary, "timeseries": pd.DataFrame(columns=_SIM_COLUMNS)}


def available_local_dates(price_df: pd.DataFrame, tz: str | None = None) -> list[date]:
    """Return sorted local calendar dates present in a price DataFrame."""
    if price_df is None or price_df.empty:
        return []
    local = _to_local(price_df, tz)
    return sorted(set(pd.DatetimeIndex(local.index).date))


def simulate_da_milp_replay(
    price_df: pd.DataFrame,
    *,
    simulation_date: date,
    tz: str | None = None,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    capture_rate: float = 0.70,
    capex_eur_kwh: float = 0.0,
    soc_init_frac: float = 0.5,
    include_reserve_headroom: bool = False,
) -> dict[str, Any]:
    """Replay one local day with the existing DA MILP dispatch solver."""
    day = _select_local_day(price_df, simulation_date, tz)
    if day.empty:
        return empty_simulation_result("No price data for the selected local day.")
    if day["price_eur_mwh"].isna().any():
        return empty_simulation_result("Selected day contains missing prices.")

    dt = _infer_interval_hours(pd.DatetimeIndex(day.index))
    result = solve_daily_lp(
        day["price_eur_mwh"].to_numpy(dtype=float),
        dt=dt,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        soc_init_frac=soc_init_frac,
    )
    revenue = _da_interval_revenue(
        day["price_eur_mwh"].to_numpy(dtype=float),
        result["p_charge"],
        result["p_discharge"],
        dt=dt,
        capture_rate=capture_rate,
    )
    daily_fce = _full_equivalent_cycles(
        result["p_discharge"],
        dt=dt,
        capacity_mwh=power_mw * duration_hours,
    )
    return _build_result(
        day,
        p_charge=result["p_charge"],
        p_discharge=result["p_discharge"],
        soc=result["soc"],
        interval_revenue=revenue,
        dt=dt,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        capex_eur_kwh=capex_eur_kwh,
        daily_fce=daily_fce,
        include_reserve_headroom=include_reserve_headroom,
    )


def simulate_da_id_replay(
    da_prices: pd.DataFrame,
    ida_prices: pd.DataFrame,
    *,
    simulation_date: date,
    tz: str | None = None,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    capture_rate: float = 0.70,
    capex_eur_kwh: float = 0.0,
    soc_init_frac: float = 0.5,
    include_reserve_headroom: bool = False,
) -> dict[str, Any]:
    """Replay one local day with the ex-post DA + IDA1 two-stage solver."""
    da_day = _select_local_day(da_prices, simulation_date, tz)
    ida_day = _select_local_day(ida_prices, simulation_date, tz)
    if da_day.empty:
        return empty_simulation_result("No DA data for the selected local day.")
    if ida_day.empty or "intraday_price_eur_mwh" not in ida_day.columns:
        return empty_simulation_result("IDA1 data is not loaded for the selected day.")

    merged = da_day[["price_eur_mwh"]].join(
        ida_day[["intraday_price_eur_mwh"]],
        how="inner",
    ).dropna()
    if merged.empty:
        return empty_simulation_result("DA and IDA1 data have no overlapping intervals.")

    dt = _infer_interval_hours(pd.DatetimeIndex(merged.index))
    result = solve_daily_da_id_dispatch(
        merged["price_eur_mwh"].to_numpy(dtype=float),
        merged["intraday_price_eur_mwh"].to_numpy(dtype=float),
        dt=dt,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        soc_init_frac=soc_init_frac,
    )
    revenue = _da_id_interval_revenue(
        da_prices=merged["price_eur_mwh"].to_numpy(dtype=float),
        ida_prices=merged["intraday_price_eur_mwh"].to_numpy(dtype=float),
        result=result,
        dt=dt,
        capture_rate=capture_rate,
    )
    daily_fce = _full_equivalent_cycles(
        result["ida_p_discharge"],
        dt=dt,
        capacity_mwh=power_mw * duration_hours,
    )
    out = _build_result(
        merged.rename(columns={"price_eur_mwh": "price_eur_mwh"}),
        p_charge=result["ida_p_charge"],
        p_discharge=result["ida_p_discharge"],
        soc=result["ida_soc"],
        interval_revenue=revenue,
        dt=dt,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        capex_eur_kwh=capex_eur_kwh,
        daily_fce=daily_fce,
        include_reserve_headroom=include_reserve_headroom,
    )
    out["timeseries"]["intraday_price_eur_mwh"] = merged["intraday_price_eur_mwh"].to_numpy()
    out["summary"]["rebid_uplift_eur"] = round(float(result["rebid_uplift_eur"]) * capture_rate, 2)
    return out


def _select_local_day(
    price_df: pd.DataFrame,
    simulation_date: date,
    tz: str | None,
) -> pd.DataFrame:
    """Select one complete local date from a timestamp-indexed frame."""
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    local = _to_local(price_df, tz).sort_index()
    selected = local[pd.DatetimeIndex(local.index).date == simulation_date].copy()
    selected.index.name = "timestamp"
    return selected


def _da_interval_revenue(
    prices: np.ndarray,
    p_charge: np.ndarray,
    p_discharge: np.ndarray,
    *,
    dt: float,
    capture_rate: float,
) -> np.ndarray:
    """Post-VOM DA interval revenue, haircut by capture rate."""
    raw = (
        (prices - DISPATCH_VOM_COST_EUR_MWH) * p_discharge * dt
        - (prices + DISPATCH_VOM_COST_EUR_MWH) * p_charge * dt
    )
    return raw * capture_rate


def _da_id_interval_revenue(
    *,
    da_prices: np.ndarray,
    ida_prices: np.ndarray,
    result: dict,
    dt: float,
    capture_rate: float,
) -> np.ndarray:
    """Interval cash decomposition matching solve_daily_da_id_dispatch."""
    da_net = result["da_p_discharge"] - result["da_p_charge"]
    ida_value = (
        (ida_prices - DISPATCH_VOM_COST_EUR_MWH) * result["ida_p_discharge"] * dt
        - (ida_prices + DISPATCH_VOM_COST_EUR_MWH) * result["ida_p_charge"] * dt
    )
    raw = da_net * da_prices * dt - da_net * ida_prices * dt + ida_value
    return raw * capture_rate


def _build_result(
    day: pd.DataFrame,
    *,
    p_charge: np.ndarray,
    p_discharge: np.ndarray,
    soc: np.ndarray,
    interval_revenue: np.ndarray,
    dt: float,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    capex_eur_kwh: float,
    daily_fce: float,
    include_reserve_headroom: bool,
) -> dict[str, Any]:
    """Build the cockpit summary and interval DataFrame from solver arrays."""
    capacity_mwh = power_mw * duration_hours
    capacity_kwh = capacity_mwh * 1000.0
    soc_start = np.asarray(soc[:-1], dtype=float)
    soc_end = np.asarray(soc[1:], dtype=float)
    net_dispatch = np.asarray(p_discharge, dtype=float) - np.asarray(p_charge, dtype=float)
    interval_throughput = (np.asarray(p_charge) + np.asarray(p_discharge)) * dt
    physical_throughput = float(interval_throughput.sum())
    traded_volume = physical_throughput
    total_revenue = float(np.asarray(interval_revenue).sum())
    active_power = np.abs(net_dispatch)
    active = active_power > 1e-6
    avg_c_rate = (
        float(active_power[active].mean()) / capacity_mwh
        if active.any() and capacity_mwh > 0 else 0.0
    )
    deg = calculate_degradation_cost(
        n_cycles=max(daily_fce, 0.0),
        capex_eur_kwh=max(capex_eur_kwh, 0.0),
        capacity_kwh=capacity_kwh,
    )
    soh_delta_pct = -(
        max(daily_fce, 0.0)
        * (1.0 - DEFAULT_EOL_CAPACITY_PCT)
        * 100.0
        / DEFAULT_CYCLE_LIFE
    )

    ts = pd.DataFrame({
        "timestamp": pd.DatetimeIndex(day.index).tz_convert("UTC"),
        "local_time": pd.DatetimeIndex(day.index),
        "price_eur_mwh": day["price_eur_mwh"].to_numpy(dtype=float),
        "p_charge_mw": p_charge,
        "p_discharge_mw": p_discharge,
        "net_dispatch_mw": net_dispatch,
        "soc_mwh": soc_end,
        "soc_pct": _safe_pct(soc_end, capacity_mwh),
        "available_charge_mw": _available_charge_power(
            soc_start, capacity_mwh, power_mw, dt, efficiency,
        ),
        "available_discharge_mw": _available_discharge_power(
            soc_start, power_mw, dt, efficiency,
        ),
        "interval_revenue_eur": interval_revenue,
        "cumulative_revenue_eur": np.cumsum(interval_revenue),
        "interval_throughput_mwh": interval_throughput,
        "cumulative_throughput_mwh": np.cumsum(interval_throughput),
    })
    if include_reserve_headroom:
        ts["reserve_headroom_mw"] = np.maximum(
            power_mw - np.asarray(p_charge) - np.asarray(p_discharge),
            0.0,
        )

    summary = {
        "total_revenue_eur": total_revenue,
        "annualized_eur_per_mw": total_revenue * DAYS_PER_YEAR / power_mw
        if power_mw > 0 else 0.0,
        "number_of_trades": _count_dispatch_blocks(net_dispatch),
        "physical_throughput_mwh": physical_throughput,
        "traded_volume_mwh": traded_volume,
        "rebalancing_factor": traded_volume / physical_throughput
        if physical_throughput > 0 else 0.0,
        "daily_fce": max(daily_fce, 0.0),
        "avg_c_rate": avg_c_rate,
        "max_depth_of_discharge_pct": _max_dod_pct(soc, capacity_mwh),
        "degradation_cost_eur": float(deg["total_degradation_eur"]),
        "soh_delta_pct": soh_delta_pct,
        "reason": "",
    }
    return {"summary": summary, "timeseries": ts}


def _full_equivalent_cycles(
    p_discharge: np.ndarray,
    *,
    dt: float,
    capacity_mwh: float,
) -> float:
    if capacity_mwh <= 0:
        return 0.0
    return float(np.asarray(p_discharge).sum() * dt / capacity_mwh)


def _available_charge_power(
    soc_start: np.ndarray,
    capacity_mwh: float,
    power_mw: float,
    dt: float,
    efficiency: float,
) -> np.ndarray:
    if capacity_mwh <= 0 or dt <= 0:
        return np.zeros_like(soc_start)
    sqrt_eff = math.sqrt(max(efficiency, 1e-12))
    return np.minimum(
        power_mw,
        np.maximum((capacity_mwh - soc_start) / (sqrt_eff * dt), 0.0),
    )


def _available_discharge_power(
    soc_start: np.ndarray,
    power_mw: float,
    dt: float,
    efficiency: float,
) -> np.ndarray:
    if dt <= 0:
        return np.zeros_like(soc_start)
    sqrt_eff = math.sqrt(max(efficiency, 1e-12))
    return np.minimum(power_mw, np.maximum(soc_start * sqrt_eff / dt, 0.0))


def _safe_pct(values: np.ndarray, denominator: float) -> np.ndarray:
    if denominator <= 0:
        return np.zeros_like(values)
    return values / denominator * 100.0


def _max_dod_pct(soc: np.ndarray, capacity_mwh: float) -> float:
    if capacity_mwh <= 0 or len(soc) == 0:
        return 0.0
    return float((np.nanmax(soc) - np.nanmin(soc)) / capacity_mwh * 100.0)


def _count_dispatch_blocks(net_dispatch: np.ndarray, threshold: float = 1e-6) -> int:
    """Count contiguous non-zero dispatch blocks, splitting on sign changes."""
    signs = np.where(np.abs(net_dispatch) > threshold, np.sign(net_dispatch), 0.0)
    count = 0
    prev = 0.0
    for sign in signs:
        if sign != 0.0 and sign != prev:
            count += 1
        prev = sign
    return count
