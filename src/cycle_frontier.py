"""Cycle-cap x degradation net-revenue frontier (pure computation, no UI).

Implements increment F-B of ``docs/design/cycle-cap-frontier-v1.md``: sweep
the DA-only daily MILP over a set of per-day cycle caps, subtract a linear
capex-amortisation wear cost EX-POST, and report the net-revenue frontier.

Red-lines (contract section 1): screening-grade LINEAR wear proxy (EUR per
full-equivalent cycle = capex * capacity / cycle_life; no DoD / C-rate /
temperature / calendar dependence), DA-only per-day standalone
terminal-neutral basis, the cap constrains the OPTIMISER never the
accounting (reported EFC is the realised optimum under the constraint),
the optimiser stays WEAR-BLIND (wear is netted in the table only, never in
the objective), and every frontier solve runs the canonical min-FEC
tie-break so ``wear_eur``/``net_eur`` are not solver-tie-sensitive.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.analytics import _infer_interval_hours
from src.degradation import (
    DAYS_PER_YEAR,
    DEFAULT_CYCLE_LIFE,
    calculate_degradation_cost,
    estimate_battery_lifetime,
)
from src.dispatch import solve_daily_lp
from src.simulation import _is_regular_utc_day, _select_local_day, available_local_dates

logger = logging.getLogger(__name__)

DEFAULT_CYCLE_CAPS: tuple[float | None, ...] = (0.5, 1.0, 1.2, 1.5, 2.0, 3.0, None)
UNCAPPED_LABEL = "uncapped"
# One tolerance policy (contract section 3): comparisons on annualised
# per-MW figures use this, converted to a window-EUR tolerance via
# `* power_mw * valid_days / DAYS_PER_YEAR` where a window quantity is
# compared. Cap compliance uses solver tolerance 1e-6 MWh (pinned in tests).
NET_TOL_EUR_PER_MW_YR = 1.0

FRONTIER_COLUMNS = [
    "cycle_cap",
    "label",
    "gross_eur",
    "avg_efc_per_day",
    "wear_eur",
    "net_eur",
    "gross_eur_per_mw_yr",
    "wear_eur_per_mw_yr",
    "net_eur_per_mw_yr",
    "net_delta_vs_uncapped_eur",
    "net_uplift_vs_uncapped_pct",
    "cycle_limited_life_years",
]


def _normalize_cycle_caps(cycle_caps: Sequence[float | None]) -> list[float | None]:
    """Validate and order the cap set: ascending finite caps, uncapped last."""
    finite: set[float] = set()
    has_uncapped = False
    for cap in cycle_caps:
        if cap is None:
            has_uncapped = True
            continue
        value = float(cap)
        if not math.isfinite(value):
            raise ValueError(f"Finite cycle caps must be finite numbers, got {cap!r}.")
        if value < 0:
            raise ValueError(f"Cycle caps must be >= 0, got {cap!r}.")
        finite.add(value)
    ordered: list[float | None] = sorted(finite)
    if has_uncapped:
        ordered.append(None)
    if not ordered:
        raise ValueError("cycle_caps must contain at least one cap.")
    return ordered


def _cap_label(cap: float | None) -> str:
    """Human label for one frontier row."""
    return UNCAPPED_LABEL if cap is None else f"{cap:g} EFC/day"


def _sweep_window(
    da_prices: pd.DataFrame,
    *,
    dates: list[date],
    tz: str | None,
    caps: list[float | None],
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    capacity_mwh: float,
) -> dict[str, Any]:
    """Solve every (day, cap) pair under the co-temporal valid-day rule.

    A day is excluded for DATA reasons (empty / NaN / sparse-irregular UTC
    grid) or SOLVER reasons (`success=False` on ANY cap) only — never for
    cap reasons — and any exclusion applies to ALL caps, so every frontier
    row aggregates the identical valid-day set.
    """
    gross = {cap: 0.0 for cap in caps}
    fec = {cap: 0.0 for cap in caps}
    valid_days = 0
    excluded_days = 0
    tiebreak_fallback_days = 0

    for local_date in dates:
        day = _select_local_day(da_prices, local_date, tz)
        if (
            day.empty
            or bool(day["price_eur_mwh"].isna().any())
            or not _is_regular_utc_day(day)
        ):
            excluded_days += 1
            continue
        prices = day["price_eur_mwh"].to_numpy(dtype=float)
        dt = _infer_interval_hours(pd.DatetimeIndex(day.index))

        day_results: dict[float | None, dict[str, Any]] = {}
        for cap in caps:
            result = solve_daily_lp(
                prices,
                dt=dt,
                power_mw=power_mw,
                duration_hours=duration_hours,
                efficiency=efficiency,
                max_efc_per_day=cap,
                min_throughput_tiebreak=True,
            )
            if not result.get("success", False):
                logger.warning(
                    "Frontier solve failed on %s (cap=%s); excluding the day "
                    "for ALL caps (co-temporal rule).",
                    local_date,
                    _cap_label(cap),
                )
                day_results = {}
                break
            day_results[cap] = result
        if not day_results:
            excluded_days += 1
            continue

        valid_days += 1
        if any(r.get("tiebreak_applied") is False for r in day_results.values()):
            tiebreak_fallback_days += 1
        for cap, result in day_results.items():
            gross[cap] += float(result["revenue_eur"])
            # RAW FEC from the returned schedule — never the ROUNDED
            # `n_cycles` convenience field (4-decimal rounding would leak
            # into money).
            discharged_mwh = float(
                np.asarray(result["p_discharge"], dtype=float).sum() * dt
            )
            fec[cap] += discharged_mwh / capacity_mwh

    return {
        "gross": gross,
        "fec": fec,
        "valid_days": valid_days,
        "excluded_days": excluded_days,
        "n_tiebreak_fallback_days": tiebreak_fallback_days,
    }


def _build_rows(
    caps: list[float | None],
    sweep: dict[str, Any],
    *,
    power_mw: float,
    capacity_kwh: float,
    capex_eur_kwh: float,
    cycle_life: float,
) -> pd.DataFrame:
    """Aggregate the sweep into one frontier row per cap (wear netted here)."""
    valid_days = sweep["valid_days"]
    annualize = DAYS_PER_YEAR / (power_mw * valid_days)
    rows: list[dict[str, Any]] = []
    for cap in caps:
        gross_eur = sweep["gross"][cap]
        fec_total = sweep["fec"][cap]
        wear = calculate_degradation_cost(
            fec_total, capex_eur_kwh, capacity_kwh, cycle_life
        )
        wear_eur = wear["total_degradation_eur"]
        avg_efc = fec_total / valid_days
        lifetime = estimate_battery_lifetime(avg_efc, cycle_life)
        rows.append(
            {
                "cycle_cap": math.nan if cap is None else cap,
                "label": _cap_label(cap),
                "gross_eur": gross_eur,
                "avg_efc_per_day": avg_efc,
                "wear_eur": wear_eur,
                "net_eur": gross_eur - wear_eur,
                "gross_eur_per_mw_yr": gross_eur * annualize,
                "wear_eur_per_mw_yr": wear_eur * annualize,
                "net_eur_per_mw_yr": (gross_eur - wear_eur) * annualize,
                "cycle_limited_life_years": lifetime["cycle_limited_years"],
            }
        )
    return pd.DataFrame(rows)


def _append_uncapped_comparison(
    frame: pd.DataFrame, has_uncapped: bool, tol_window_eur: float
) -> pd.DataFrame:
    """Add the vs-uncapped delta/uplift columns (NaN without an uncapped row).

    The uplift denominator is ABSOLUTE (contract r3): the sign stays
    meaningful when the uncapped net is negative (wear can exceed gross —
    -5 vs -10 is a +50% improvement, not -50%). NaN when |uncapped net| is
    within the window tolerance of zero.
    """
    if not has_uncapped:
        frame["net_delta_vs_uncapped_eur"] = math.nan
        frame["net_uplift_vs_uncapped_pct"] = math.nan
        return frame
    uncapped_net = float(frame.loc[frame["label"] == UNCAPPED_LABEL, "net_eur"].iloc[0])
    frame["net_delta_vs_uncapped_eur"] = frame["net_eur"] - uncapped_net
    if abs(uncapped_net) <= tol_window_eur:
        frame["net_uplift_vs_uncapped_pct"] = math.nan
    else:
        frame["net_uplift_vs_uncapped_pct"] = (
            (frame["net_eur"] - uncapped_net) / abs(uncapped_net) * 100.0
        )
    return frame


def _best_cap_label(frame: pd.DataFrame, tol_window_eur: float) -> str:
    """Best-cap rule (contract r1): lowest finite cap within tolerance of
    the max net wins; uncapped wins only on a STRICT win over every finite
    cap by more than the tolerance."""
    max_net = float(frame["net_eur"].max())
    finite = frame[frame["label"] != UNCAPPED_LABEL]
    candidates = finite[finite["net_eur"] >= max_net - tol_window_eur]
    if not candidates.empty:
        best = candidates.sort_values("cycle_cap").iloc[0]
        return str(best["label"])
    return UNCAPPED_LABEL


def _empty_result(
    excluded_days: int,
    *,
    cost_per_cycle_eur: float,
    wear_eur_per_mwh_discharged: float,
    cycle_life: float,
    capex_eur_kwh: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Typed empty frame + NaN-free summary for a zero-valid-day window."""
    frame = pd.DataFrame(columns=FRONTIER_COLUMNS)
    summary = {
        "valid_days": 0,
        "excluded_days": excluded_days,
        "cost_per_cycle_eur": cost_per_cycle_eur,
        "wear_eur_per_mwh_discharged": wear_eur_per_mwh_discharged,
        "cycle_life": cycle_life,
        "capex_eur_kwh": capex_eur_kwh,
        "best_cap_label": None,
        "frontier_flat": False,
        "n_tiebreak_fallback_days": 0,
    }
    return frame, summary


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
    cycle_caps: Sequence[float | None] = DEFAULT_CYCLE_CAPS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sweep per-day cycle caps and net a linear wear cost off DA revenue.

    Per cap, every clean local day in the window is solved standalone
    (terminal-neutral) with `solve_daily_lp(max_efc_per_day=cap,
    min_throughput_tiebreak=True)`; revenue and RAW discharge-leg FEC are
    summed over an IDENTICAL valid-day set across caps (a day excluded for
    data or solver reasons is excluded for ALL caps), then
    `wear = FEC_total * capex_eur_kwh * capacity_kwh / cycle_life` is
    subtracted EX-POST. This is a screening-grade investment table, not an
    electrochemical model or a dispatch strategy.

    Args:
        da_prices: UTC timestamp-indexed frame with `price_eur_mwh`.
        dates: Local dates to include; `None` = all available local dates.
        tz: Zone-local timezone for day grouping.
        power_mw: BESS power rating in MW (must be positive).
        duration_hours: BESS energy duration in hours (must be positive).
        efficiency: Round-trip efficiency (0-1).
        capex_eur_kwh: Installed CapEx basis in EUR/kWh.
        cycle_life: Full-equivalent cycle life (linear-proxy denominator).
        cycle_caps: Cap sweep; `None` entry = uncapped reference row.

    Returns:
        `(frontier_df, summary)` — one row per cap ordered ascending finite
        caps then uncapped; summary with `valid_days`, `excluded_days`,
        `cost_per_cycle_eur`, `wear_eur_per_mwh_discharged`, `cycle_life`,
        `capex_eur_kwh`, `best_cap_label`, `frontier_flat`,
        `n_tiebreak_fallback_days`.
    """
    if power_mw <= 0:
        raise ValueError(f"power_mw must be positive, got {power_mw}")
    if duration_hours <= 0:
        raise ValueError(f"duration_hours must be positive, got {duration_hours}")
    caps = _normalize_cycle_caps(cycle_caps)
    capacity_mwh = power_mw * duration_hours
    capacity_kwh = capacity_mwh * 1000.0
    cost_per_cycle_eur = calculate_degradation_cost(
        0.0, capex_eur_kwh, capacity_kwh, cycle_life
    )["cost_per_cycle_eur"]
    wear_eur_per_mwh_discharged = cost_per_cycle_eur / capacity_mwh

    selected_dates = list(dates) if dates is not None else available_local_dates(
        da_prices, tz=tz
    )
    sweep = _sweep_window(
        da_prices,
        dates=selected_dates,
        tz=tz,
        caps=caps,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        capacity_mwh=capacity_mwh,
    )
    if sweep["valid_days"] == 0:
        return _empty_result(
            sweep["excluded_days"],
            cost_per_cycle_eur=cost_per_cycle_eur,
            wear_eur_per_mwh_discharged=wear_eur_per_mwh_discharged,
            cycle_life=cycle_life,
            capex_eur_kwh=capex_eur_kwh,
        )

    frame = _build_rows(
        caps,
        sweep,
        power_mw=power_mw,
        capacity_kwh=capacity_kwh,
        capex_eur_kwh=capex_eur_kwh,
        cycle_life=cycle_life,
    )
    tol_window_eur = (
        NET_TOL_EUR_PER_MW_YR * power_mw * sweep["valid_days"] / DAYS_PER_YEAR
    )
    frame = _append_uncapped_comparison(frame, None in caps, tol_window_eur)
    frame = frame[FRONTIER_COLUMNS]

    net = frame["net_eur"].astype(float)
    summary = {
        "valid_days": sweep["valid_days"],
        "excluded_days": sweep["excluded_days"],
        "cost_per_cycle_eur": cost_per_cycle_eur,
        "wear_eur_per_mwh_discharged": wear_eur_per_mwh_discharged,
        "cycle_life": cycle_life,
        "capex_eur_kwh": capex_eur_kwh,
        "best_cap_label": _best_cap_label(frame, tol_window_eur),
        "frontier_flat": bool(net.max() - net.min() <= tol_window_eur),
        "n_tiebreak_fallback_days": sweep["n_tiebreak_fallback_days"],
    }
    return frame, summary
