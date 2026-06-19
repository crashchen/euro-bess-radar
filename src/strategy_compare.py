"""Side-by-side dispatch-strategy comparison for the screening dashboard.

The sequential DA+ID batch already computes three benchmarks for one window
(DA-only / forecast-driven realised / perfect-foresight ceiling). This module
reframes those totals as an *investment* comparison — annualised EUR/MW/yr and
uplift over the DA-only baseline — so a user can answer "is participating in
IDA worth it, and how much is left on the table?" at a glance. A reserve
co-optimisation strategy is a planned fourth row.
"""

from __future__ import annotations

import pandas as pd

from src.simulation import DAYS_PER_YEAR

STRATEGY_COMPARE_COLUMNS = [
    "strategy", "window_revenue_eur", "annualized_eur_per_mw", "uplift_vs_da_pct",
]

_DA_ONLY = "DA-only"
_FORECAST = "DA + IDA1 (forecast-driven)"
_CEILING = "DA + IDA1 (perfect-foresight ceiling)"


def build_strategy_comparison(
    summary: dict, *, power_mw: float,
) -> pd.DataFrame:
    """Reframe a sequential DA+ID batch summary as a strategy comparison.

    Args:
        summary: Output of ``simulation.simulate_sequential_da_id_batch``
            (uses ``total_da_only_eur`` / ``total_realised_eur`` /
            ``total_ceiling_eur`` / ``valid_days``).
        power_mw: BESS power rating for the per-MW annualisation.

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
    return pd.DataFrame(rows, columns=STRATEGY_COMPARE_COLUMNS)
