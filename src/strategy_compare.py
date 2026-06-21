"""Side-by-side dispatch-strategy comparison for the screening dashboard.

The sequential DA+ID batch already computes three benchmarks for one window
(DA-only / forecast-driven realised / perfect-foresight ceiling). This module
reframes those totals as an *investment* comparison — annualised EUR/MW/yr and
uplift over the DA-only baseline — so a user can answer "is participating in
IDA worth it, and how much is left on the table?" at a glance.

An optional fourth row reframes a DA + reserve-capacity co-optimisation total
(from ``dispatch.solve_joint_capacity_batch``) the same way. That row is a
*different revenue stream* (reserve capacity headroom, not intraday rebidding)
and is a headroom-aware capacity estimate — it does NOT model activation
energy and is NOT additive with DA, because the joint MILP makes reserve
headroom compete with DA charge/discharge for the same power budget. A true
cumulative DA + IDA + reserve triple-joint solve is a heavier future upgrade.
"""

from __future__ import annotations

import math

import pandas as pd

from src.simulation import DAYS_PER_YEAR

STRATEGY_COMPARE_COLUMNS = [
    "strategy", "window_revenue_eur", "annualized_eur_per_mw", "uplift_vs_da_pct",
]

_DA_ONLY = "DA-only"
_FORECAST = "DA + IDA1 (forecast-driven)"
_CEILING = "DA + IDA1 (perfect-foresight ceiling)"
_RESERVE_DEFAULT = "DA + reserve co-opt (headroom)"


def build_strategy_comparison(
    summary: dict, *, power_mw: float,
    reserve_coopt_total: float | None = None,
    reserve_label: str | None = None,
) -> pd.DataFrame:
    """Reframe a sequential DA+ID batch summary as a strategy comparison.

    Args:
        summary: Output of ``simulation.simulate_sequential_da_id_batch``
            (uses ``total_da_only_eur`` / ``total_realised_eur`` /
            ``total_ceiling_eur`` / ``valid_days``).
        power_mw: BESS power rating for the per-MW annualisation.
        reserve_coopt_total: Optional window total (EUR) for a DA + reserve
            capacity co-optimisation, summed from
            ``dispatch.solve_joint_capacity_batch``. When provided and finite,
            adds a fourth row annualised/uplifted on the same window basis.
            This is a headroom-aware capacity estimate (no activation energy,
            not additive with DA — the joint MILP already trades reserve
            headroom against DA dispatch), and a *different stream* from the
            DA+IDA rows, not a cumulative ladder.
        reserve_label: Display label for the reserve row.

    Returns:
        One row per strategy with columns ``[strategy, window_revenue_eur,
        annualized_eur_per_mw, uplift_vs_da_pct]``. Annualisation uses the
        same 365.25-day i.i.d. convention as the rest of the dashboard.
        Empty when the window has no valid days.
    """
    valid_days = int(summary.get("valid_days", 0) or 0)
    if valid_days <= 0:
        return pd.DataFrame(columns=STRATEGY_COMPARE_COLUMNS)

    da = float(summary.get("total_da_only_eur", 0.0))
    realised = float(summary.get("total_realised_eur", 0.0))
    ceiling = float(summary.get("total_ceiling_eur", 0.0))

    def annualized_per_mw(total: float) -> float:
        if power_mw <= 0:
            return float("nan")
        return total * DAYS_PER_YEAR / valid_days / power_mw

    def uplift_pct(total: float) -> float:
        # Undefined when the DA-only baseline is ~0 (no meaningful denominator).
        if abs(da) < 1e-9:
            return float("nan")
        return (total - da) / da * 100.0

    rows = [
        (_DA_ONLY, da, annualized_per_mw(da), 0.0),
        (_FORECAST, realised, annualized_per_mw(realised), uplift_pct(realised)),
        (_CEILING, ceiling, annualized_per_mw(ceiling), uplift_pct(ceiling)),
    ]
    if reserve_coopt_total is not None and math.isfinite(float(reserve_coopt_total)):
        rc = float(reserve_coopt_total)
        rows.append(
            (reserve_label or _RESERVE_DEFAULT, rc,
             annualized_per_mw(rc), uplift_pct(rc)),
        )
    return pd.DataFrame(rows, columns=STRATEGY_COMPARE_COLUMNS)
