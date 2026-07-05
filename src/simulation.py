"""Interval-level BESS dispatch replay for the Simulation Cockpit."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.analytics import _infer_interval_hours, _to_local
from src.config import ANCILLARY_CAPACITY_AVAILABILITY, MAX_CONTINUOUS_REPLAY_INTERVALS
from src.da_forecast import DA_FORECAST_COL, build_da_price_forecast
from src.degradation import (
    DEFAULT_CYCLE_LIFE,
    DEFAULT_EOL_CAPACITY_PCT,
    calculate_degradation_cost,
)
from src.dispatch import (
    DISPATCH_VOM_COST_EUR_MWH,
    solve_daily_da_id_dispatch,
    solve_daily_da_id_reserve_dispatch,
    solve_daily_lp,
    solve_sequential_da_id_dispatch,
    solve_sequential_da_id_reserve_dispatch,
)
from src.ida_forecast import (
    FORECAST_COL,
    IDA_VALUE_COL,
    build_ida_forecast,
    compute_forecast_skill,
)
from src.ida_scenarios import build_ida_scenarios
from src.reserve_forecast import (
    RESERVE_FORECAST_COL,
    RESERVE_VALUE_COL,
    build_reserve_price_forecast,
)
from src.stochastic_dispatch import (
    solve_myopic_capped_da_id_dispatch,
    solve_stochastic_da_id_dispatch,
)

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
_BATCH_COLUMNS = [
    "date",
    "mode",
    "total_revenue_eur",
    "annualized_eur_per_mw",
    "number_of_trades",
    "physical_throughput_mwh",
    "traded_volume_mwh",
    "rebalancing_factor",
    "daily_fce",
    "avg_c_rate",
    "max_depth_of_discharge_pct",
    "degradation_cost_eur",
    "soh_delta_pct",
    "soc_start_pct",
    "soc_end_pct",
    "n_intervals",
]
_EVENT_COLUMNS = [
    "event_id",
    "event_type",
    "start_time",
    "end_time",
    "duration_h",
    "avg_power_mw",
    "energy_mwh",
    "avg_price_eur_mwh",
    "revenue_eur",
    "soc_start_pct",
    "soc_end_pct",
    "avg_rebid_delta_mw",
    "max_abs_rebid_delta_mw",
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
    capture_rate: float = 1.0,
    capex_eur_kwh: float = 0.0,
    soc_init_frac: float = 0.5,
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
    capture_rate: float = 1.0,
    capex_eur_kwh: float = 0.0,
    soc_init_frac: float = 0.5,
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
    # ida_p_* is the FINAL physical schedule after ex-post rebid, so FCE
    # based on it equals real battery wear.
    daily_fce = _full_equivalent_cycles(
        result["ida_p_discharge"],
        dt=dt,
        capacity_mwh=power_mw * duration_hours,
    )
    # Traded volume = financial transaction volume across the two market
    # sessions. DA gross is the volume committed at the DA auction; the
    # |ID rebid delta| is the additional volume traded on ID to move the
    # physical schedule away from the DA commitment. When DA == IDA (no
    # rebid) the delta is zero and traded_volume == physical_throughput.
    # The abs is applied per leg (charge and discharge) so a sign-flipped
    # rebid (e.g. DA buys, IDA sells) counts both legs as new trades.
    # MILP mutual exclusion ensures p_charge and p_discharge are never
    # simultaneously non-zero, so the per-leg abs cannot overcount within
    # a single interval.
    da_charge = np.asarray(result["da_p_charge"], dtype=float)
    da_discharge = np.asarray(result["da_p_discharge"], dtype=float)
    ida_charge = np.asarray(result["ida_p_charge"], dtype=float)
    ida_discharge = np.asarray(result["ida_p_discharge"], dtype=float)
    da_gross_volume = float((da_charge + da_discharge).sum() * dt)
    id_delta_volume = float((
        np.abs(ida_charge - da_charge) + np.abs(ida_discharge - da_discharge)
    ).sum() * dt)
    traded_volume = da_gross_volume + id_delta_volume

    extra_columns = {
        "intraday_price_eur_mwh": merged["intraday_price_eur_mwh"].to_numpy(),
        "da_position_mw": da_discharge - da_charge,
        "ida_position_mw": ida_discharge - ida_charge,
        "rebid_delta_mw": (ida_discharge - ida_charge) - (da_discharge - da_charge),
    }

    out = _build_result(
        merged,
        p_charge=ida_charge,
        p_discharge=ida_discharge,
        soc=result["ida_soc"],
        interval_revenue=revenue,
        dt=dt,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        capex_eur_kwh=capex_eur_kwh,
        daily_fce=daily_fce,
        traded_volume_mwh=traded_volume,
        extra_columns=extra_columns,
    )
    out["summary"]["rebid_uplift_eur"] = round(
        float(result["rebid_uplift_eur"]) * capture_rate, 2
    )
    return out


def simulate_replay_batch(
    price_df: pd.DataFrame,
    *,
    mode: str = "DA MILP Replay",
    intraday_df: pd.DataFrame | None = None,
    tz: str | None = None,
    dates: list[date] | None = None,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    capture_rate: float = 1.0,
    capex_eur_kwh: float = 0.0,
    soc_init_frac: float = 0.5,
    carry_soc: bool = True,
) -> pd.DataFrame:
    """Run replay summaries for many local days.

    With `carry_soc=True` (default) and `mode="DA MILP Replay"`, the
    requested window is solved as ONE continuous MILP per contiguous
    no-gap segment (terminal-neutral applied only at the END of each
    segment), so day N+1 can monetise SoC stored at the end of day N.
    This fixes the per-day terminal-neutral bias that previously capped
    multi-day aggregates at the in-day spread.

    With `carry_soc=False`, every day re-uses the legacy per-day solve
    starting from `soc_init_frac` (50% by default).

    Continuous-horizon carry-over is supported for BOTH `"DA MILP
    Replay"` and `"DA + IDA1 Replay"`: each contiguous clean run is
    solved once over the whole multi-day horizon (the two-stage DA+ID
    solver is itself horizon-agnostic, so terminal-neutral applies only
    at the run end for both the DA and the IDA stage).
    `batch.attrs["da_id_carry_soc_supported"]` stays `True` for parity
    with the DA-only path.
    """
    selected_dates = dates or available_local_dates(price_df, tz=tz)
    has_intraday = intraday_df is not None and not intraday_df.empty
    is_da_id = mode == "DA + IDA1 Replay"
    can_continue = carry_soc and len(selected_dates) >= 2

    if can_continue and not is_da_id:
        rows, excluded_days = _simulate_continuous_da_replay(
            price_df,
            dates=selected_dates,
            tz=tz,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            capture_rate=capture_rate,
            capex_eur_kwh=capex_eur_kwh,
            soc_init_frac=soc_init_frac,
        )
        use_continuous = True
    elif can_continue and is_da_id and has_intraday:
        rows, excluded_days = _simulate_continuous_da_id_replay(
            price_df,
            intraday_df,  # type: ignore[arg-type]
            dates=selected_dates,
            tz=tz,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            capture_rate=capture_rate,
            capex_eur_kwh=capex_eur_kwh,
            soc_init_frac=soc_init_frac,
        )
        use_continuous = True
    else:
        rows, excluded_days = _simulate_per_day_replay(
            price_df,
            mode=mode,
            intraday_df=intraday_df,
            dates=selected_dates,
            tz=tz,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            capture_rate=capture_rate,
            capex_eur_kwh=capex_eur_kwh,
            soc_init_frac=soc_init_frac,
        )
        use_continuous = False

    if not rows:
        out = pd.DataFrame(columns=_BATCH_COLUMNS)
    else:
        out = (
            pd.DataFrame(rows, columns=_BATCH_COLUMNS)
            .sort_values("date")
            .reset_index(drop=True)
        )
    out.attrs["excluded_days"] = excluded_days
    out.attrs["carry_soc"] = carry_soc
    out.attrs["carry_mode"] = (
        "continuous_horizon" if use_continuous else "per_day_reset"
    )
    # Continuous DA+ID carry-over is now implemented; kept for UI parity.
    out.attrs["da_id_carry_soc_supported"] = True
    return out


def _simulate_per_day_replay(
    price_df: pd.DataFrame,
    *,
    mode: str,
    intraday_df: pd.DataFrame | None,
    dates: list[date],
    tz: str | None,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    capture_rate: float,
    capex_eur_kwh: float,
    soc_init_frac: float,
) -> tuple[list[dict[str, Any]], int]:
    """Legacy per-day loop. Every day starts at `soc_init_frac`."""
    rows: list[dict[str, Any]] = []
    excluded = 0
    for local_date in dates:
        if mode == "DA + IDA1 Replay":
            if intraday_df is None or intraday_df.empty:
                excluded += 1
                continue
            result = simulate_da_id_replay(
                price_df, intraday_df,
                simulation_date=local_date, tz=tz,
                power_mw=power_mw, duration_hours=duration_hours,
                efficiency=efficiency, capture_rate=capture_rate,
                capex_eur_kwh=capex_eur_kwh, soc_init_frac=soc_init_frac,
            )
        else:
            result = simulate_da_milp_replay(
                price_df,
                simulation_date=local_date, tz=tz,
                power_mw=power_mw, duration_hours=duration_hours,
                efficiency=efficiency, capture_rate=capture_rate,
                capex_eur_kwh=capex_eur_kwh, soc_init_frac=soc_init_frac,
            )
        if result["timeseries"].empty:
            excluded += 1
            continue
        rows.append(_summary_row(local_date, mode, result, soc_init_frac))
    return rows, excluded


def _simulate_continuous_da_replay(
    price_df: pd.DataFrame,
    *,
    dates: list[date],
    tz: str | None,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    capture_rate: float,
    capex_eur_kwh: float,
    soc_init_frac: float,
) -> tuple[list[dict[str, Any]], int]:
    """Solve the requested window as one MILP per contiguous clean run.

    Splits `dates` into maximal contiguous sequences that yield a complete
    NaN-free price slice (any NaN or missing interval breaks the run at
    that day). Each run gets one `solve_daily_lp` call with
    `soc_init_frac` = end-of-previous-run SoC, with terminal-neutral
    equality applied at the END of the run. Days inside the run keep
    full multi-day SoC freedom.
    """
    rows: list[dict[str, Any]] = []
    excluded = 0
    capacity_mwh = max(power_mw * duration_hours, 1e-9)
    current_soc_frac = soc_init_frac

    def build_da_day(local_date: date) -> pd.DataFrame:
        day = _select_local_day(price_df, local_date, tz)
        return day[["price_eur_mwh"]] if not day.empty else pd.DataFrame()

    runs = _group_clean_runs(dates=dates, build_day=build_da_day)
    handled_dates: set[date] = set()

    for run_dates, slice_df, day_breaks in runs:
        handled_dates.update(run_dates)
        prices = slice_df["price_eur_mwh"].to_numpy(dtype=float)
        dt = _infer_interval_hours(pd.DatetimeIndex(slice_df.index))
        run_result = solve_daily_lp(
            prices,
            dt=dt,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            soc_init_frac=current_soc_frac,
        )
        full_revenue = _da_interval_revenue(
            prices,
            run_result["p_charge"],
            run_result["p_discharge"],
            dt=dt,
            capture_rate=capture_rate,
        )

        soc_start_frac = current_soc_frac
        for local_date, (start, end) in zip(run_dates, day_breaks, strict=True):
            day_df = slice_df.iloc[start:end]
            day_soc_slice = run_result["soc"][start:end + 1]
            day_p_charge = run_result["p_charge"][start:end]
            day_p_discharge = run_result["p_discharge"][start:end]
            day_interval_rev = full_revenue[start:end]
            daily_fce = _full_equivalent_cycles(
                day_p_discharge, dt=dt, capacity_mwh=capacity_mwh,
            )
            day_result = _build_result(
                day_df,
                p_charge=day_p_charge,
                p_discharge=day_p_discharge,
                soc=day_soc_slice,
                interval_revenue=day_interval_rev,
                dt=dt,
                power_mw=power_mw,
                duration_hours=duration_hours,
                efficiency=efficiency,
                capex_eur_kwh=capex_eur_kwh,
                daily_fce=daily_fce,
            )
            rows.append(
                _summary_row(local_date, "DA MILP Replay", day_result, soc_start_frac)
            )
            soc_start_frac = float(day_soc_slice[-1] / capacity_mwh)

        current_soc_frac = min(max(soc_start_frac, 0.0), 1.0)

    excluded = sum(1 for d in dates if d not in handled_dates)
    return rows, excluded


def _da_id_traded_volume(
    *,
    da_charge: np.ndarray,
    da_discharge: np.ndarray,
    ida_charge: np.ndarray,
    ida_discharge: np.ndarray,
    dt: float,
) -> float:
    """DA gross + |ID rebid delta| traded volume (see simulate_da_id_replay)."""
    da_gross = float((da_charge + da_discharge).sum() * dt)
    id_delta = float((
        np.abs(ida_charge - da_charge) + np.abs(ida_discharge - da_discharge)
    ).sum() * dt)
    return da_gross + id_delta


def _simulate_continuous_da_id_replay(
    da_prices: pd.DataFrame,
    ida_prices: pd.DataFrame,
    *,
    dates: list[date],
    tz: str | None,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    capture_rate: float,
    capex_eur_kwh: float,
    soc_init_frac: float,
) -> tuple[list[dict[str, Any]], int]:
    """Continuous-horizon DA + IDA1 replay.

    Mirrors `_simulate_continuous_da_replay` but feeds each contiguous
    clean run's merged DA/IDA prices to `solve_daily_da_id_dispatch`
    once. That two-stage solver runs `solve_daily_lp` on the DA prices
    and again on the IDA prices, both terminal-neutral only at the run
    end — so SoC carries across days in BOTH the committed DA stage and
    the ex-post IDA rebid stage. Per-day KPI rows slice the run's IDA
    (final physical) arrays, exactly like the single-day path.
    """
    rows: list[dict[str, Any]] = []
    capacity_mwh = max(power_mw * duration_hours, 1e-9)
    current_soc_frac = soc_init_frac

    def build_da_id_day(local_date: date) -> pd.DataFrame:
        da_day = _select_local_day(da_prices, local_date, tz)
        ida_day = _select_local_day(ida_prices, local_date, tz)
        if da_day.empty or ida_day.empty or "intraday_price_eur_mwh" not in ida_day.columns:
            return pd.DataFrame()
        merged = da_day[["price_eur_mwh"]].join(
            ida_day[["intraday_price_eur_mwh"]], how="inner",
        ).dropna()
        return merged if not merged.empty else pd.DataFrame()

    runs = _group_clean_runs(dates=dates, build_day=build_da_id_day)
    handled_dates: set[date] = set()

    for run_dates, slice_df, day_breaks in runs:
        handled_dates.update(run_dates)
        da_arr = slice_df["price_eur_mwh"].to_numpy(dtype=float)
        ida_arr = slice_df["intraday_price_eur_mwh"].to_numpy(dtype=float)
        dt = _infer_interval_hours(pd.DatetimeIndex(slice_df.index))
        run = solve_daily_da_id_dispatch(
            da_arr, ida_arr, dt=dt,
            power_mw=power_mw, duration_hours=duration_hours,
            efficiency=efficiency, soc_init_frac=current_soc_frac,
        )
        full_revenue = _da_id_interval_revenue(
            da_prices=da_arr, ida_prices=ida_arr, result=run,
            dt=dt, capture_rate=capture_rate,
        )
        da_charge = np.asarray(run["da_p_charge"], dtype=float)
        da_discharge = np.asarray(run["da_p_discharge"], dtype=float)
        ida_charge = np.asarray(run["ida_p_charge"], dtype=float)
        ida_discharge = np.asarray(run["ida_p_discharge"], dtype=float)
        da_position = da_discharge - da_charge
        ida_position = ida_discharge - ida_charge

        soc_start_frac = current_soc_frac
        for local_date, (start, end) in zip(run_dates, day_breaks, strict=True):
            day_df = slice_df.iloc[start:end]
            day_ida_charge = ida_charge[start:end]
            day_ida_discharge = ida_discharge[start:end]
            day_da_charge = da_charge[start:end]
            day_da_discharge = da_discharge[start:end]
            daily_fce = _full_equivalent_cycles(
                day_ida_discharge, dt=dt, capacity_mwh=capacity_mwh,
            )
            traded_volume = _da_id_traded_volume(
                da_charge=day_da_charge, da_discharge=day_da_discharge,
                ida_charge=day_ida_charge, ida_discharge=day_ida_discharge,
                dt=dt,
            )
            extra_columns = {
                "intraday_price_eur_mwh": ida_arr[start:end],
                "da_position_mw": da_position[start:end],
                "ida_position_mw": ida_position[start:end],
                "rebid_delta_mw": ida_position[start:end] - da_position[start:end],
            }
            day_result = _build_result(
                day_df,
                p_charge=day_ida_charge,
                p_discharge=day_ida_discharge,
                soc=run["ida_soc"][start:end + 1],
                interval_revenue=full_revenue[start:end],
                dt=dt,
                power_mw=power_mw,
                duration_hours=duration_hours,
                efficiency=efficiency,
                capex_eur_kwh=capex_eur_kwh,
                daily_fce=daily_fce,
                traded_volume_mwh=traded_volume,
                extra_columns=extra_columns,
            )
            rows.append(
                _summary_row(local_date, "DA + IDA1 Replay", day_result, soc_start_frac)
            )
            soc_start_frac = float(run["ida_soc"][end] / capacity_mwh)

        current_soc_frac = min(max(soc_start_frac, 0.0), 1.0)

    excluded = sum(1 for d in dates if d not in handled_dates)
    return rows, excluded


def _group_clean_runs(
    *,
    dates: list[date],
    build_day: Callable[[date], pd.DataFrame],
) -> list[tuple[list[date], pd.DataFrame, list[tuple[int, int]]]]:
    """Group `dates` into contiguous segments with clean per-day frames.

    `build_day(local_date)` returns the day's frame (DA-only or merged
    DA+IDA) or an empty frame when the day is unusable. Returns a list of
    `(run_dates, slice_df, day_breaks)` where `day_breaks` is a list of
    `(start_idx, end_idx)` pairs into `slice_df`'s rows for each local
    date in `run_dates`. A run breaks whenever:
      - the next requested date is more than one calendar day later,
      - the day's frame is empty,
      - the day's frame has any NaN value,
      - the day's frame is sparse (non-uniform UTC index),
      - adding the day would push the run past
        `MAX_CONTINUOUS_REPLAY_INTERVALS` (a performance cap — a single
        large MILP for 90 days of 15-min data otherwise hangs the
        dashboard). This split is a *soft* reset: the caller seeds the
        next run with the end-of-previous-run SoC, so SoC still carries
        across the boundary, but terminal-neutral equality is re-applied
        at each chunk edge. The split is contiguous (no day is dropped).
    """
    if not dates:
        return []
    sorted_dates = sorted(dates)
    runs: list[tuple[list[date], pd.DataFrame, list[tuple[int, int]]]] = []
    current_dates: list[date] = []
    current_frames: list[pd.DataFrame] = []
    current_breaks: list[tuple[int, int]] = []
    cursor = 0
    previous_date: date | None = None

    def flush() -> None:
        nonlocal cursor, current_dates, current_frames, current_breaks
        if current_dates:
            slice_df = pd.concat(current_frames)
            runs.append((current_dates, slice_df, current_breaks))
        current_dates = []
        current_frames = []
        current_breaks = []
        cursor = 0

    for local_date in sorted_dates:
        if previous_date is not None and (local_date - previous_date).days != 1:
            flush()
        day_df = build_day(local_date)
        if (
            day_df.empty
            or bool(day_df.isna().to_numpy().any())
            or not _is_regular_utc_day(day_df)
        ):
            # Empty / NaN / sparse days break the continuous horizon. A
            # sparse day (e.g. a missing 02:00 interval that upstream
            # cleaning did not reindex to NaN) would otherwise compress
            # the MILP time axis. DST spring-forward (23 local hours) and
            # fall-back (25 local hours) days are NOT sparse — their UTC
            # index is still uniform — so they pass this guard.
            flush()
            previous_date = None
            continue
        n = len(day_df)
        if current_dates and cursor + n > MAX_CONTINUOUS_REPLAY_INTERVALS:
            # Soft size-cap reset: flush the accumulated chunk but keep
            # `previous_date` so the next chunk stays contiguous. The
            # continuous solvers seed it with the prior chunk's end SoC.
            flush()
        current_dates.append(local_date)
        current_frames.append(day_df)
        current_breaks.append((cursor, cursor + n))
        cursor += n
        previous_date = local_date

    flush()
    return runs


def _is_regular_utc_day(day_df: pd.DataFrame) -> bool:
    """True if the day's UTC index has a single uniform interval delta.

    DST-safe sparsity check: a missing interval leaves a gap in the UTC
    index (non-uniform delta), while DST transition days stay uniform in
    UTC even though their LOCAL hour count is 23 or 25. Used to keep the
    continuous-horizon MILP from silently compressing a sparse day.
    """
    if len(day_df) < 2:
        return True
    utc_index = pd.DatetimeIndex(day_df.index).tz_convert("UTC").sort_values()
    deltas = np.diff(utc_index.asi8)
    return bool(np.all(deltas == deltas[0]))


_SEQ_COLUMNS = [
    "date",
    "da_only_eur",
    "realised_eur",
    "ceiling_eur",
    "forecast_error_eur",
    "captured_eur",
    "rebid",
    "forecast_coverage",
]


def simulate_sequential_da_id_batch(
    da_prices: pd.DataFrame,
    ida_prices: pd.DataFrame,
    *,
    dates: list[date] | None = None,
    tz: str | None = None,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    bucket: str = "hour_of_day",
    forecast_mode: str = "loo",
    min_rebid_uplift_eur: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-day three-way DA+ID comparison under an imperfect IDA forecast.

    For each local day this builds an IDA forecast (see
    `ida_forecast.build_ida_forecast`; `forecast_mode` selects "loo"
    cross-validation, "walk_forward", or "in_sample") and runs
    `solve_sequential_da_id_dispatch` to get three benchmarks:
      - DA-only baseline (no IDA participation),
      - forecast-driven realised (rebid at forecast, settle at realised),
      - perfect-foresight ceiling (ex-post optimal rebid).

    `min_rebid_uplift_eur` is the per-day risk gate / deadband passed to the
    solver: a day only rebids when its forecast-predicted uplift clears the
    hurdle, else it holds the DA schedule (`rebid=False`, captured 0). The
    summary reports `n_rebid_days` / `n_hold_days`.

    Each day is solved standalone (per-day terminal-neutral) so the
    comparison isolates *forecast quality* from the multi-day SoC-carry
    effect that `simulate_replay_batch` models separately. All values are
    raw solver outputs — the cockpit's DA-slippage capture haircut is NOT
    applied here, so this panel stays a clean forecast-quality comparison.

    Returns `(per_day_df, summary)`. `summary` carries absolute EUR totals
    (`total_da_only_eur`, `total_realised_eur`, `total_ceiling_eur`,
    `total_forecast_error_eur`, `total_ceiling_uplift_eur`,
    `total_captured_eur`), the forecast metadata, and a guarded
    `capture_rate` (= total captured / total achievable uplift; None when
    no rebid opportunity exists in the window).
    """
    selected = dates or available_local_dates(da_prices, tz=tz)
    forecast_df, fc_meta = build_ida_forecast(
        ida_prices, target_dates=selected, tz=tz,
        bucket=bucket, forecast_mode=forecast_mode,
    )

    rows: list[dict[str, Any]] = []
    excluded = 0
    if not forecast_df.empty:
        coverage_by_point = forecast_df["n_samples"] > 0
        for local_date in selected:
            row = _sequential_day_row(
                da_prices, ida_prices, forecast_df, coverage_by_point,
                local_date=local_date, tz=tz, power_mw=power_mw,
                duration_hours=duration_hours, efficiency=efficiency,
                min_rebid_uplift_eur=min_rebid_uplift_eur,
            )
            if row is None:
                excluded += 1
                continue
            rows.append(row)
    else:
        excluded = len(selected)

    if not rows:
        per_day = pd.DataFrame(columns=_SEQ_COLUMNS)
    else:
        per_day = (
            pd.DataFrame(rows, columns=_SEQ_COLUMNS)
            .sort_values("date")
            .reset_index(drop=True)
        )

    if per_day.empty:
        total_da_only = total_realised = total_ceiling = total_error = 0.0
        total_ceiling_uplift = total_captured = 0.0
    else:
        total_da_only = float(per_day["da_only_eur"].sum())
        total_realised = float(per_day["realised_eur"].sum())
        total_ceiling = float(per_day["ceiling_eur"].sum())
        total_error = float(per_day["forecast_error_eur"].sum())
        total_ceiling_uplift = total_ceiling - total_da_only
        total_captured = total_realised - total_da_only
    n_rebid_days = 0 if per_day.empty else int(per_day["rebid"].sum())
    n_hold_days = len(per_day) - n_rebid_days
    # capture_rate = captured / achievable uplift. Left None when there is
    # effectively no rebid opportunity in the window (achievable ~ 0), where
    # the ratio is meaningless; it can be negative when the forecast misleads
    # the rebid into churn losses below the DA-only baseline.
    capture_rate = (
        total_captured / total_ceiling_uplift
        if total_ceiling_uplift > 1e-6 else None
    )
    summary = {
        "valid_days": len(per_day),
        "excluded_days": excluded,
        "total_da_only_eur": total_da_only,
        "total_realised_eur": total_realised,
        "total_ceiling_eur": total_ceiling,
        "total_forecast_error_eur": total_error,
        "total_ceiling_uplift_eur": total_ceiling_uplift,
        "total_captured_eur": total_captured,
        "capture_rate": capture_rate,
        "min_rebid_uplift_eur": min_rebid_uplift_eur,
        "n_rebid_days": n_rebid_days,
        "n_hold_days": n_hold_days,
        "forecast_meta": fc_meta,
        # Price-space forecast accuracy vs realised IDA (+ skill vs the
        # DA-as-IDA naive baseline), so the user can judge how trustworthy
        # the forecast — and any deadband set on it — actually is.
        "forecast_skill": compute_forecast_skill(
            forecast_df, ida_prices, da_prices=da_prices, tz=tz,
        ),
    }
    per_day.attrs["summary"] = summary
    return per_day, summary


def align_reserve_price_to_index(
    reserve_series: pd.Series | None,
    target_index: pd.DatetimeIndex,
    tz: str | None,
) -> np.ndarray:
    """Per-interval reserve price for target intervals via (local date, 4h block).

    Maps a timestamp-indexed reserve capacity-price series (typically
    block-granular FCR/aFRR data) onto the target interval timestamps by
    matching local date + 4h block, averaging any multiple source rows that
    fall in the same (date, block). Target intervals with no matching reserve
    price get 0.0 (no reserve incentive). Both the 9.2a ceiling batch and the
    9.2b sequential batch use this so they price reserve identically per
    interval.
    """
    target = pd.DatetimeIndex(target_index)
    if reserve_series is None or len(reserve_series) == 0:
        return np.zeros(len(target))
    src = reserve_series.dropna()
    if len(src) == 0:
        return np.zeros(len(target))
    src_idx = pd.DatetimeIndex(src.index)
    tgt_idx = target
    if tz:
        # Project-internal timestamps are UTC. Treat naive cache/manual series
        # as UTC before converting, otherwise local block matching can shift.
        if src_idx.tz is None:
            src_idx = src_idx.tz_localize("UTC")
        src_idx = src_idx.tz_convert(tz)
        if tgt_idx.tz is None:
            tgt_idx = tgt_idx.tz_localize("UTC")
        tgt_idx = tgt_idx.tz_convert(tz)
    src_keys = pd.MultiIndex.from_arrays(
        [src_idx.date, np.asarray(src_idx.hour) // 4],  # 4h FCR/aFRR product block
    )
    lookup = (
        pd.Series(np.asarray(src.values, dtype=float), index=src_keys)
        .groupby(level=[0, 1])
        .mean()
        .to_dict()
    )
    tgt_keys = zip(tgt_idx.date, np.asarray(tgt_idx.hour) // 4, strict=True)
    return np.array([lookup.get(key, 0.0) for key in tgt_keys], dtype=float)


def simulate_da_id_reserve_ceiling_batch(
    da_prices: pd.DataFrame,
    ida_prices: pd.DataFrame,
    capacity_price_eur_mw_h: float | pd.Series,
    *,
    dates: list[date],
    tz: str | None = None,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    availability: float = ANCILLARY_CAPACITY_AVAILABILITY,
) -> dict[str, Any]:
    """Sum the DA + IDA + reserve perfect-foresight ceiling over local days.

    Each day's DA and IDA prices are inner-joined on the delivery timestamp
    (the same merge as the sequential DA+ID batch) and solved with
    ``dispatch.solve_daily_da_id_reserve_dispatch``. Days without overlapping
    DA/IDA intervals are skipped. Returns ``{"total_eur", "solved_days"}``.

    ``capacity_price_eur_mw_h`` is a scalar cleared price OR a timestamp-indexed
    reserve-price ``pd.Series`` (block-granular). The series form is aligned to
    each day's intervals via :func:`align_reserve_price_to_index`, so the 9.2a
    ceiling shown in the cockpit prices reserve identically (per interval) to
    the 9.2b sequential solver's ``global_ceiling_eur``.

    This is a perfect-foresight UPPER BOUND, reserve = capacity headroom only
    (no activation energy), and NOT a forecast-driven policy. Each day is solved
    standalone terminal-neutral (no multi-day SoC carry), matching the
    standalone basis of ``simulate_sequential_da_id_batch``.
    """
    reserve_series = None
    scalar_price = None
    if isinstance(capacity_price_eur_mw_h, pd.Series):
        reserve_series = capacity_price_eur_mw_h
    else:
        try:
            scalar_price = float(capacity_price_eur_mw_h)
        except (TypeError, ValueError):
            return {"total_eur": 0.0, "solved_days": 0}
        if not math.isfinite(scalar_price):
            return {"total_eur": 0.0, "solved_days": 0}

    total = 0.0
    solved_days = 0
    for local_date in dates:
        da_day = _select_local_day(da_prices, local_date, tz)
        ida_day = _select_local_day(ida_prices, local_date, tz)
        if da_day.empty or ida_day.empty or "intraday_price_eur_mwh" not in ida_day.columns:
            continue
        merged = da_day[["price_eur_mwh"]].join(
            ida_day[["intraday_price_eur_mwh"]], how="inner",
        ).dropna()
        if merged.empty or not _is_regular_utc_day(merged):
            continue
        dt = _infer_interval_hours(pd.DatetimeIndex(merged.index))
        if reserve_series is not None:
            day_reserve_price = align_reserve_price_to_index(
                reserve_series, pd.DatetimeIndex(merged.index), tz,
            )
        else:
            day_reserve_price = scalar_price
        result = solve_daily_da_id_reserve_dispatch(
            merged["price_eur_mwh"].to_numpy(dtype=float),
            merged["intraday_price_eur_mwh"].to_numpy(dtype=float),
            dt=dt,
            capacity_price_eur_mw_h=day_reserve_price,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            availability=availability,
        )
        total += result["total_cash_eur"]
        solved_days += 1
    return {"total_eur": round(total, 6), "solved_days": solved_days}


_SEQ_RESERVE_COLUMNS = [
    "date", "da_only_eur", "realised_eur", "reserve_first_ceiling_eur",
    "global_ceiling_eur", "forecast_effect_eur", "timing_cost_eur",
    "full_gap_eur", "avg_reserve_mw",
]


def _empty_seq_reserve_summary(forecast_mode: str, bucket: str) -> dict[str, Any]:
    return {
        "total_da_only_eur": 0.0,
        "total_realised_eur": 0.0,
        "total_reserve_first_ceiling_eur": 0.0,
        "total_global_ceiling_eur": 0.0,
        "total_forecast_effect_eur": 0.0,
        "total_timing_cost_eur": 0.0,
        "total_full_gap_eur": 0.0,
        "valid_days": 0,
        "excluded_days": 0,
        "forecast_mode": forecast_mode,
        "bucket": bucket,
    }


def simulate_sequential_da_id_reserve_batch(
    da_prices: pd.DataFrame,
    ida_prices: pd.DataFrame,
    reserve_price_series: pd.Series | None,
    *,
    dates: list[date],
    tz: str | None = None,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    availability: float = ANCILLARY_CAPACITY_AVAILABILITY,
    bucket: str = "hour_of_day",
    forecast_mode: str = "walk_forward",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Forecast-driven reserve-first DA+IDA+reserve policy over a window.

    Builds DA / IDA / reserve-price forecasts once (``forecast_mode`` defaults
    to ``"walk_forward"`` — Stage-0 reserve is a real-time commitment that must
    not see the target/future day), aligns them and the realised reserve price
    to each day's merged DA/IDA intervals (per-interval, via
    :func:`align_reserve_price_to_index`), and solves
    ``dispatch.solve_sequential_da_id_reserve_dispatch`` per day. Aggregates the
    gap attribution. Returns ``(per_day_df, summary)``.

    Days missing a usable DA/IDA forecast (e.g. walk-forward's first day) or
    without DA/IDA overlap are excluded; a day with no reserve forecast simply
    commits ``reserve_mw = 0`` (safe degrade, not silent optimism). The summary
    keeps ``forecast_effect`` signed (negative = the forecast helped) and the
    identity ``full_gap == forecast_effect + timing_cost`` holds per day.
    """
    if not dates:
        return pd.DataFrame(columns=_SEQ_RESERVE_COLUMNS), _empty_seq_reserve_summary(
            forecast_mode, bucket,
        )

    da_fc_df, _ = build_da_price_forecast(
        da_prices, target_dates=dates, tz=tz, bucket=bucket, forecast_mode=forecast_mode,
    )
    ida_fc_df, _ = build_ida_forecast(
        ida_prices, target_dates=dates, tz=tz, bucket=bucket, forecast_mode=forecast_mode,
    )
    if reserve_price_series is not None and len(reserve_price_series) > 0:
        reserve_hist = pd.DataFrame({RESERVE_VALUE_COL: reserve_price_series})
        reserve_fc_df, _ = build_reserve_price_forecast(
            reserve_hist, target_dates=dates, tz=tz, forecast_mode=forecast_mode,
        )
        reserve_fc_series = (
            reserve_fc_df[RESERVE_FORECAST_COL] if not reserve_fc_df.empty else None
        )
    else:
        reserve_fc_series = None

    rows: list[dict[str, Any]] = []
    excluded = 0
    for local_date in dates:
        da_day = _select_local_day(da_prices, local_date, tz)
        ida_day = _select_local_day(ida_prices, local_date, tz)
        if da_day.empty or ida_day.empty or "intraday_price_eur_mwh" not in ida_day.columns:
            excluded += 1
            continue
        merged = da_day[["price_eur_mwh"]].join(
            ida_day[["intraday_price_eur_mwh"]], how="inner",
        ).dropna()
        if merged.empty or not _is_regular_utc_day(merged):
            excluded += 1
            continue
        idx = pd.DatetimeIndex(merged.index)
        da_fc = (
            da_fc_df[DA_FORECAST_COL].reindex(idx)
            if not da_fc_df.empty else pd.Series(np.nan, index=idx)
        )
        ida_fc = (
            ida_fc_df[FORECAST_COL].reindex(idx)
            if not ida_fc_df.empty else pd.Series(np.nan, index=idx)
        )
        if da_fc.isna().any() or ida_fc.isna().any():
            # No usable DA/IDA forecast for this day (e.g. walk-forward first day).
            excluded += 1
            continue
        reserve_realised = align_reserve_price_to_index(reserve_price_series, idx, tz)
        reserve_forecast = align_reserve_price_to_index(reserve_fc_series, idx, tz)
        result = solve_sequential_da_id_reserve_dispatch(
            da_fc.to_numpy(dtype=float),
            merged["price_eur_mwh"].to_numpy(dtype=float),
            ida_fc.to_numpy(dtype=float),
            merged["intraday_price_eur_mwh"].to_numpy(dtype=float),
            reserve_forecast,
            reserve_realised,
            dt=_infer_interval_hours(idx),
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            availability=availability,
        )
        rows.append({
            "date": local_date,
            "da_only_eur": result["da_only_revenue_eur"],
            "realised_eur": result["realised_total_eur"],
            "reserve_first_ceiling_eur": result["reserve_first_ceiling_eur"],
            "global_ceiling_eur": result["global_ceiling_eur"],
            "forecast_effect_eur": result["forecast_cost_eur"],
            "timing_cost_eur": result["timing_cost_eur"],
            "full_gap_eur": result["full_gap_eur"],
            "avg_reserve_mw": float(np.mean(result["reserve_mw"])),
        })

    per_day = pd.DataFrame(rows, columns=_SEQ_RESERVE_COLUMNS)
    if per_day.empty:
        summary = _empty_seq_reserve_summary(forecast_mode, bucket)
        summary["excluded_days"] = excluded
        return per_day, summary

    per_day = per_day.sort_values("date").reset_index(drop=True)
    summary = {
        "total_da_only_eur": float(per_day["da_only_eur"].sum()),
        "total_realised_eur": float(per_day["realised_eur"].sum()),
        "total_reserve_first_ceiling_eur": float(per_day["reserve_first_ceiling_eur"].sum()),
        "total_global_ceiling_eur": float(per_day["global_ceiling_eur"].sum()),
        "total_forecast_effect_eur": float(per_day["forecast_effect_eur"].sum()),
        "total_timing_cost_eur": float(per_day["timing_cost_eur"].sum()),
        "total_full_gap_eur": float(per_day["full_gap_eur"].sum()),
        "valid_days": len(per_day),
        "excluded_days": excluded,
        "forecast_mode": forecast_mode,
        "bucket": bucket,
    }
    return per_day, summary


def _sequential_day_row(
    da_prices: pd.DataFrame,
    ida_prices: pd.DataFrame,
    forecast_df: pd.DataFrame,
    coverage_by_point: pd.Series,
    *,
    local_date: date,
    tz: str | None,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    min_rebid_uplift_eur: float = 0.0,
) -> dict[str, Any] | None:
    """Solve one day's three-way sequential comparison, or None if unusable."""
    da_day = _select_local_day(da_prices, local_date, tz)
    ida_day = _select_local_day(ida_prices, local_date, tz)
    if da_day.empty or ida_day.empty or "intraday_price_eur_mwh" not in ida_day.columns:
        return None
    merged = (
        da_day[["price_eur_mwh"]]
        .join(ida_day[["intraday_price_eur_mwh"]], how="inner")
        .join(forecast_df[[FORECAST_COL]], how="inner")
        .dropna()
    )
    if merged.empty or not _is_regular_utc_day(merged):
        return None

    dt = _infer_interval_hours(pd.DatetimeIndex(merged.index))
    result = solve_sequential_da_id_dispatch(
        merged["price_eur_mwh"].to_numpy(dtype=float),
        merged[FORECAST_COL].to_numpy(dtype=float),
        merged["intraday_price_eur_mwh"].to_numpy(dtype=float),
        dt=dt, power_mw=power_mw, duration_hours=duration_hours,
        efficiency=efficiency, min_rebid_uplift_eur=min_rebid_uplift_eur,
    )
    day_coverage = float(coverage_by_point.reindex(merged.index).fillna(False).mean())
    return {
        "date": local_date,
        "da_only_eur": result["da_only_revenue_eur"],
        "realised_eur": result["realised_total_eur"],
        "ceiling_eur": result["ceiling_total_eur"],
        "forecast_error_eur": result["forecast_error_cost_eur"],
        "captured_eur": result["captured_uplift_eur"],
        "rebid": bool(result["rebid"]),
        "forecast_coverage": day_coverage,
    }


def build_dispatch_event_table(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Collapse interval dispatch into contiguous charge/discharge events."""
    if timeseries is None or timeseries.empty or "net_dispatch_mw" not in timeseries.columns:
        return pd.DataFrame(columns=_EVENT_COLUMNS)

    events: list[dict[str, Any]] = []
    signs = np.where(
        timeseries["net_dispatch_mw"].abs().to_numpy() > 1e-6,
        np.sign(timeseries["net_dispatch_mw"].to_numpy()),
        0.0,
    )
    start: int | None = None
    current_sign = 0.0
    for pos, sign in enumerate(signs):
        if sign == 0.0:
            if start is not None:
                events.append(_event_row(timeseries, start, pos, current_sign, len(events) + 1))
                start = None
                current_sign = 0.0
            continue
        if start is None:
            start = pos
            current_sign = sign
        elif sign != current_sign:
            events.append(_event_row(timeseries, start, pos, current_sign, len(events) + 1))
            start = pos
            current_sign = sign

    if start is not None:
        events.append(_event_row(timeseries, start, len(timeseries), current_sign, len(events) + 1))

    if not events:
        return pd.DataFrame(columns=_EVENT_COLUMNS)
    return pd.DataFrame(events, columns=_EVENT_COLUMNS)


def _summary_row(
    local_date: date,
    mode: str,
    result: dict[str, Any],
    soc_start_frac: float,
) -> dict[str, Any]:
    """Flatten a daily replay result for multi-day cockpit summaries."""
    summary = result["summary"]
    ts = result["timeseries"]
    return {
        "date": local_date,
        "mode": mode,
        "total_revenue_eur": float(summary["total_revenue_eur"]),
        "annualized_eur_per_mw": float(summary["annualized_eur_per_mw"]),
        "number_of_trades": int(summary["number_of_trades"]),
        "physical_throughput_mwh": float(summary["physical_throughput_mwh"]),
        "traded_volume_mwh": float(summary["traded_volume_mwh"]),
        "rebalancing_factor": float(summary["rebalancing_factor"]),
        "daily_fce": float(summary["daily_fce"]),
        "avg_c_rate": float(summary["avg_c_rate"]),
        "max_depth_of_discharge_pct": float(summary["max_depth_of_discharge_pct"]),
        "degradation_cost_eur": float(summary["degradation_cost_eur"]),
        "soh_delta_pct": float(summary["soh_delta_pct"]),
        "soc_start_pct": float(soc_start_frac * 100.0),
        "soc_end_pct": float(ts["soc_pct"].iloc[-1]) if not ts.empty else 0.0,
        "n_intervals": len(ts),
    }


def _event_row(
    timeseries: pd.DataFrame,
    start: int,
    end: int,
    sign: float,
    event_id: int,
) -> dict[str, Any]:
    """Summarise one contiguous non-zero physical dispatch event."""
    window = timeseries.iloc[start:end]
    index = pd.DatetimeIndex(timeseries["local_time"])
    dt = _infer_interval_hours(index)
    event_type = "Discharge" if sign > 0 else "Charge"
    power_col = "p_discharge_mw" if sign > 0 else "p_charge_mw"
    power = window[power_col].abs()
    rebid = (
        window["rebid_delta_mw"].astype(float)
        if "rebid_delta_mw" in window.columns
        else pd.Series([0.0] * len(window), index=window.index)
    )
    soc_start = (
        float(timeseries["soc_pct"].iloc[start - 1])
        if start > 0 else float(window["soc_pct"].iloc[0])
    )
    return {
        "event_id": event_id,
        "event_type": event_type,
        "start_time": window["local_time"].iloc[0],
        "end_time": window["local_time"].iloc[-1] + pd.Timedelta(hours=dt),
        "duration_h": float(len(window) * dt),
        "avg_power_mw": float(power.mean()) if not power.empty else 0.0,
        "energy_mwh": float(power.sum() * dt),
        "avg_price_eur_mwh": float(window["price_eur_mwh"].mean()),
        "revenue_eur": float(window["interval_revenue_eur"].sum()),
        "soc_start_pct": soc_start,
        "soc_end_pct": float(window["soc_pct"].iloc[-1]),
        "avg_rebid_delta_mw": float(rebid.mean()) if not rebid.empty else 0.0,
        "max_abs_rebid_delta_mw": float(rebid.abs().max()) if not rebid.empty else 0.0,
    }


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
    traded_volume_mwh: float | None = None,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Build the cockpit summary and interval DataFrame from solver arrays."""
    capacity_mwh = power_mw * duration_hours
    capacity_kwh = capacity_mwh * 1000.0
    soc_start = np.asarray(soc[:-1], dtype=float)
    soc_end = np.asarray(soc[1:], dtype=float)
    net_dispatch = np.asarray(p_discharge, dtype=float) - np.asarray(p_charge, dtype=float)
    interval_throughput = (np.asarray(p_charge) + np.asarray(p_discharge)) * dt
    physical_throughput = float(interval_throughput.sum())
    traded_volume = (
        float(traded_volume_mwh)
        if traded_volume_mwh is not None
        else physical_throughput
    )
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
    if extra_columns:
        for col_name, values in extra_columns.items():
            ts[col_name] = np.asarray(values)

    # Rebalancing factor: traded / physical. Edge cases:
    #   physical == 0 and traded == 0 → 1.0 (no activity, no rebalancing)
    #   physical == 0 and traded  > 0 → inf (DA was fully unwound on ID,
    #     so all volume was for rebalancing — UI renders this as "Inf")
    #   otherwise → traded / physical
    if physical_throughput <= 0:
        rebalancing_factor = 1.0 if traded_volume <= 0 else float("inf")
    else:
        rebalancing_factor = traded_volume / physical_throughput
    summary = {
        "total_revenue_eur": total_revenue,
        "annualized_eur_per_mw": total_revenue * DAYS_PER_YEAR / power_mw
        if power_mw > 0 else 0.0,
        "number_of_trades": _count_dispatch_blocks(net_dispatch),
        "physical_throughput_mwh": physical_throughput,
        "traded_volume_mwh": traded_volume,
        "rebalancing_factor": rebalancing_factor,
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


# ── Stochastic DA+ID batch (stochastic MILP Increment C1) ─────────────────────

_STOCH_COLUMNS = [
    "date", "da_only_eur", "myopic_realised_eur", "coopt_realised_eur",
    "stochastic_realised_eur", "coopt_ceiling_eur", "policy_value_eur",
    "commitment_value_eur", "distribution_value_eur", "rebid", "n_scenarios",
    "coverage",
]


def _risk_block(pooled: list[float]) -> dict[str, Any]:
    """P10/P50/P90 + downside CVaR@90 of the pooled per-day scenario totals.

    A dispersion DIAGNOSTIC (the objective stays risk-neutral, scope §7):
    ``cvar90`` is the mean of the worst 10% of the pooled per-(day, scenario)
    energy totals — the downside tail, never the upper tail. Pooling mixes
    day-level and scenario-level variation, so read it as a spread indicator,
    not a portfolio VaR.
    """
    if not pooled:
        return {"p10": None, "p50": None, "p90": None, "cvar90": None, "n": 0}
    arr = np.sort(np.asarray(pooled, dtype=float))
    k = max(1, int(np.ceil(0.10 * arr.size)))
    return {
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "cvar90": float(arr[:k].mean()),  # mean of the worst 10%
        "n": int(arr.size),
    }


def _align_day_scenarios(bundle: dict, index: pd.DatetimeIndex):
    """Align a scenario bundle to a day's merged UTC index, or None if it can't.

    Returns ``(base_forecast_arr, scenarios_arr)`` positionally matched to
    ``index``. None when any merged timestamp is missing from the bundle (the
    scenario grid must cover the day being priced).
    """
    # The merged day index is in the zone's LOCAL tz (from ``_select_local_day``),
    # but the bundle timestamps are UTC — align to UTC before the lookup or every
    # day is silently dropped on any real (tz-aware) zone.
    index = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    ts = bundle["timestamps"]
    pos = ts.get_indexer(index)
    if (pos < 0).any():
        return None
    base = bundle["base_forecast"].to_numpy()[pos]
    scen = bundle["scenarios"][:, pos]
    return base, scen


def _stochastic_day(
    da_prices: pd.DataFrame, ida_prices: pd.DataFrame, bundle: dict | None,
    *, local_date: date, tz: str | None, power_mw: float, duration_hours: float,
    efficiency: float, rebid_cap_mw, reserve_mw, reserve_price_eur_mw_h,
    availability: float, min_rebid_uplift_eur: float,
) -> tuple[dict[str, Any], np.ndarray] | None:
    """Solve one day's three-policy stochastic comparison, or None if unusable."""
    if bundle is None:
        return None
    da_day = _select_local_day(da_prices, local_date, tz)
    ida_day = _select_local_day(ida_prices, local_date, tz)
    if da_day.empty or ida_day.empty or IDA_VALUE_COL not in ida_day.columns:
        return None
    merged = (
        da_day[["price_eur_mwh"]]
        .join(ida_day[[IDA_VALUE_COL]], how="inner")
        .dropna()
    )
    if merged.empty or not _is_regular_utc_day(merged):
        return None
    aligned = _align_day_scenarios(bundle, pd.DatetimeIndex(merged.index))
    if aligned is None:
        return None
    base, scen = aligned
    da = merged["price_eur_mwh"].to_numpy(dtype=float)
    realised = merged[IDA_VALUE_COL].to_numpy(dtype=float)
    idx = pd.DatetimeIndex(merged.index)
    dt = _infer_interval_hours(idx)
    weights = np.full(scen.shape[0], 1.0 / scen.shape[0])
    # A window-indexed reserve Series (the loaded-capacity standard) is aligned
    # to THIS day's intervals by (local date, 4h block) — the same helper the
    # 9.2b batches use — so the batch's internal day loop can price real
    # block-of-day reserve; a scalar / per-interval array passes through.
    day_reserve_mw = (
        align_reserve_price_to_index(reserve_mw, idx, tz)
        if isinstance(reserve_mw, pd.Series) else reserve_mw
    )
    day_reserve_price = (
        align_reserve_price_to_index(reserve_price_eur_mw_h, idx, tz)
        if isinstance(reserve_price_eur_mw_h, pd.Series) else reserve_price_eur_mw_h
    )
    common = dict(
        dt=dt, power_mw=power_mw, duration_hours=duration_hours,
        efficiency=efficiency, rebid_cap_mw=rebid_cap_mw, reserve_mw=day_reserve_mw,
        reserve_price_eur_mw_h=day_reserve_price, availability=availability,
        min_rebid_uplift_eur=min_rebid_uplift_eur,
    )
    myopic = solve_myopic_capped_da_id_dispatch(da, base, realised, **common)
    coopt = solve_stochastic_da_id_dispatch(
        da, base[None, :], np.array([1.0]), base, realised, **common,
    )
    stoch = solve_stochastic_da_id_dispatch(da, scen, weights, base, realised, **common)
    if not (myopic["success"] and coopt["success"] and stoch["success"]):
        return None
    row = {
        "date": local_date,
        "da_only_eur": stoch["da_only_revenue_eur"],
        "myopic_realised_eur": myopic["realised_total_eur"],
        "coopt_realised_eur": coopt["realised_total_eur"],
        "stochastic_realised_eur": stoch["realised_total_eur"],
        "coopt_ceiling_eur": stoch["coopt_ceiling_eur"],
        # Headline: robust policy value of the stochastic commitment over the
        # deterministic myopic baseline. The commitment/distribution split below
        # is an arithmetic decomposition of it (commitment + distribution ==
        # policy_value). As of the v2 canonical Stage-1 selector
        # (stochastic_dispatch._canonicalize_stage1) the co-opt and stochastic
        # commitments break Stage-1 degeneracy identically (earliest-activity
        # min-throughput tie-break), so the split is now tie-STABLE — it vanishes
        # at rebid_cap = inf (the decoupling regime) and otherwise reflects the
        # real value of the coupling, not multi-optimum noise (scope §1/§8-1).
        "policy_value_eur": stoch["realised_total_eur"] - myopic["realised_total_eur"],
        "commitment_value_eur": coopt["realised_total_eur"] - myopic["realised_total_eur"],
        "distribution_value_eur": stoch["realised_total_eur"] - coopt["realised_total_eur"],
        "rebid": bool(stoch["rebid"]),
        "n_scenarios": int(scen.shape[0]),
        "coverage": float(bundle["base_coverage"]),
    }
    return row, np.asarray(stoch["scenario_total_eur"], dtype=float)


def simulate_stochastic_da_id_batch(
    da_prices: pd.DataFrame,
    ida_prices: pd.DataFrame,
    *,
    dates: list[date] | None = None,
    tz: str | None = None,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    n_scenarios: int = 10,
    bucket: str = "hour_of_day",
    forecast_mode: str = "loo",
    seed: int | None = None,
    rebid_cap_mw: float | None = None,
    reserve_mw: float | np.ndarray | None = None,
    reserve_price_eur_mw_h: float | np.ndarray | None = None,
    availability: float = ANCILLARY_CAPACITY_AVAILABILITY,
    min_rebid_uplift_eur: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-day three-policy stochastic DA+ID comparison at a COMMON rebid cap.

    For each local day this builds a scenario bundle
    (:func:`ida_scenarios.build_ida_scenarios`) and runs three policies under the
    IDENTICAL ``rebid_cap_mw`` / reserve / window (scope §5):

    - **capped myopic** — DA-only Stage-1 (ignores IDA) through the capped
      execution (``solve_myopic_capped_da_id_dispatch``);
    - **deterministic co-opt** — S=1 stochastic commit against the base forecast;
    - **stochastic** — S=N scenario-aware commit.

    The **headline** is the robust ``policy_value = stochastic_realised -
    myopic_realised`` — the realised lift of the scenario-aware policy over the
    deterministic myopic baseline. It also carries a DIAGNOSTIC arithmetic split
    ``commitment_value = co_opt_realised - myopic_realised`` (worth of a
    non-myopic DA commitment) + ``distribution_value = stochastic_realised -
    co_opt_realised`` (worth of anticipating the IDA distribution); these sum to
    ``policy_value``. As of the v2 canonical Stage-1 selector
    (``stochastic_dispatch._canonicalize_stage1``) the co-opt and stochastic
    commitments break Stage-1 degeneracy identically (earliest-activity
    min-throughput tie-break), so the split is **tie-stable**: it vanishes at
    ``rebid_cap = inf`` (the decoupling regime) and otherwise reflects the real
    value of the coupling rather than multi-optimum noise (scope §8-1). The only
    residual tie-noise is on the rare day where the time-limited tie-break falls
    back to the pass-1 schedule; the robust ``policy_value`` headline never
    depends on the split. Each day is solved standalone (per-day terminal-neutral),
    isolating commitment quality from multi-day SoC carry.

    Returns ``(per_day_df, summary)``. ``summary`` carries the window totals for
    each policy, the headline ``total_policy_value_eur`` (+ the diagnostic
    split), the risk block (:func:`_risk_block`, pooled per-day scenario
    dispersion — a diagnostic), and the scenario/forecast metadata. At
    ``rebid_cap_mw = inf`` with no reserve the myopic policy ties the 9.2b
    sequential row exactly (a robust regression anchor).
    """
    selected = dates or available_local_dates(da_prices, tz=tz)
    scenarios_by_date, scen_meta = build_ida_scenarios(
        ida_prices, target_dates=selected, n_scenarios=n_scenarios, tz=tz,
        bucket=bucket, forecast_mode=forecast_mode, seed=seed,
    )

    rows: list[dict[str, Any]] = []
    pooled: list[float] = []
    excluded = 0
    for local_date in selected:
        result = _stochastic_day(
            da_prices, ida_prices, scenarios_by_date.get(local_date),
            local_date=local_date, tz=tz, power_mw=power_mw,
            duration_hours=duration_hours, efficiency=efficiency,
            rebid_cap_mw=rebid_cap_mw, reserve_mw=reserve_mw,
            reserve_price_eur_mw_h=reserve_price_eur_mw_h,
            availability=availability, min_rebid_uplift_eur=min_rebid_uplift_eur,
        )
        if result is None:
            excluded += 1
            continue
        row, scen_totals = result
        rows.append(row)
        pooled.extend(scen_totals.tolist())

    if not rows:
        per_day = pd.DataFrame(columns=_STOCH_COLUMNS)
    else:
        per_day = (
            pd.DataFrame(rows, columns=_STOCH_COLUMNS)
            .sort_values("date").reset_index(drop=True)
        )
    summary = _stochastic_summary(
        per_day, pooled, excluded, scen_meta, rebid_cap_mw, n_scenarios,
        min_rebid_uplift_eur,
    )
    per_day.attrs["summary"] = summary
    return per_day, summary


def _stochastic_summary(
    per_day: pd.DataFrame, pooled: list[float], excluded: int, scen_meta: dict,
    rebid_cap_mw, n_scenarios: int, min_rebid_uplift_eur: float,
) -> dict[str, Any]:
    """Aggregate the per-day three-policy rows into window totals + risk block."""
    def _tot(col: str) -> float:
        return 0.0 if per_day.empty else float(per_day[col].sum())

    return {
        "valid_days": len(per_day),
        "excluded_days": excluded,
        "total_da_only_eur": _tot("da_only_eur"),
        "total_myopic_realised_eur": _tot("myopic_realised_eur"),
        "total_coopt_realised_eur": _tot("coopt_realised_eur"),
        "total_stochastic_realised_eur": _tot("stochastic_realised_eur"),
        "total_coopt_ceiling_eur": _tot("coopt_ceiling_eur"),
        # Headline robust metric; the two below are its diagnostic split (they
        # sum to it). The v2 canonical Stage-1 selector makes the split tie-stable
        # (zero distribution at rebid_cap=inf); only a tie-break fallback day can
        # reintroduce noise — the headline never depends on the split.
        "total_policy_value_eur": _tot("policy_value_eur"),
        "total_commitment_value_eur": _tot("commitment_value_eur"),
        "total_distribution_value_eur": _tot("distribution_value_eur"),
        "risk_block": _risk_block(pooled),
        "rebid_cap_mw": rebid_cap_mw,
        "n_scenarios": n_scenarios,
        "min_rebid_uplift_eur": min_rebid_uplift_eur,
        "n_rebid_days": 0 if per_day.empty else int(per_day["rebid"].sum()),
        "scenario_meta": scen_meta,
    }
