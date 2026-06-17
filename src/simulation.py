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

    `mode="DA + IDA1 Replay"` always uses the per-day two-stage solver
    today; a continuous-horizon DA+ID solver is on the roadmap but not
    implemented. `carry_soc=True` is silently downgraded to per-day for
    this mode and `batch.attrs["da_id_carry_soc_supported"]` is set
    `False` so the UI can warn.
    """
    selected_dates = dates or available_local_dates(price_df, tz=tz)
    use_continuous = (
        carry_soc and mode == "DA MILP Replay" and len(selected_dates) >= 2
    )
    da_id_carry_supported = mode != "DA + IDA1 Replay"

    if use_continuous:
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
    out.attrs["da_id_carry_soc_supported"] = da_id_carry_supported
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

    runs = _group_clean_runs(price_df, dates=dates, tz=tz)
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


def _group_clean_runs(
    price_df: pd.DataFrame,
    *,
    dates: list[date],
    tz: str | None,
) -> list[tuple[list[date], pd.DataFrame, list[tuple[int, int]]]]:
    """Group `dates` into contiguous segments with NaN-free price slices.

    Returns a list of `(run_dates, slice_df, day_breaks)` where
    `day_breaks` is a list of `(start_idx, end_idx)` pairs into
    `slice_df`'s rows for each local date in `run_dates`. A run breaks
    whenever:
      - the next requested date is more than one calendar day later,
      - the day's slice is empty,
      - the day's slice has any NaN price.
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
        day_df = _select_local_day(price_df, local_date, tz)
        if (
            day_df.empty
            or day_df["price_eur_mwh"].isna().any()
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
