"""Cross-zone portfolio analysis for BESS merchant revenue screening.

Builds a per-zone daily revenue matrix, derives correlation / Sharpe stats,
and solves the long-only Markowitz frontier so users can see whether a
multi-zone BESS deployment diversifies away DA revenue risk.

Math is intentionally simple — annualisation = mean(daily) * 365.25 — to
stay consistent with the rest of the dashboard (estimate_annual_arbitrage_revenue
in analytics.py) and the screening-grade intent.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.analytics import calculate_daily_dispatch, calculate_daily_spreads

DAYS_PER_YEAR = 365.25


def build_daily_revenue_matrix(
    zone_data: dict[str, pd.DataFrame],
    *,
    zone_timezones: dict[str, str] | None = None,
    duration_hours: float = 1.0,
    power_mw: float = 1.0,
    efficiency: float = 0.88,
    capture_rate: float = 0.70,
    use_lp_dispatch: bool = False,
) -> pd.DataFrame:
    """Build a wide [date x zone] frame of daily revenue per MW (EUR).

    Aligns on the intersection of valid dates across zones — the same
    "complete local day" filter that calculate_daily_spreads applies, so a
    single missing hour in one zone drops that date from the portfolio
    view (not just from the zone with the gap). Without intersection, the
    correlation matrix is computed on shifting subsets and becomes
    inconsistent across pairs.
    """
    if not zone_data:
        return pd.DataFrame()

    zone_timezones = zone_timezones or {}
    series: dict[str, pd.Series] = {}

    for zone, df in zone_data.items():
        if df is None or df.empty:
            continue
        tz = zone_timezones.get(zone)
        if use_lp_dispatch:
            daily = calculate_daily_dispatch(
                df, tz=tz, duration_hours=duration_hours,
                power_mw=power_mw, efficiency=efficiency,
            )
            revenue_col = "lp_revenue" if "lp_revenue" in daily.columns else None
        else:
            daily = calculate_daily_spreads(
                df, tz=tz, duration_hours=duration_hours,
            )
            revenue_col = None

        if daily.empty:
            continue

        if revenue_col == "lp_revenue":
            rev = daily.set_index("date")["lp_revenue"] * capture_rate
        else:
            # Greedy fallback: spread * energy * sqrt(eff) * capture_rate per cycle/day.
            energy_mwh = power_mw * duration_hours
            sqrt_eff = math.sqrt(efficiency)
            rev = daily.set_index("date")["spread"] * energy_mwh * sqrt_eff * capture_rate

        # Normalize to per-MW for portfolio comparison.
        series[zone] = rev / power_mw

    if not series:
        return pd.DataFrame()

    # Outer-join first to see the union, then keep only fully-aligned rows.
    wide = pd.concat(series, axis=1).sort_index()
    wide.columns.name = "zone"
    wide.index.name = "date"
    return wide.dropna(how="any")


def compute_correlation_matrix(rev_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation of daily revenue per MW.

    Zones with constant revenue (zero variance) produce NaN correlations
    in pandas; that's the correct behaviour — a flat series has no
    meaningful linear relationship — but we preserve diagonal=1.0 by
    convention so a heatmap renders cleanly.
    """
    if rev_df.empty or rev_df.shape[1] < 2:
        return pd.DataFrame(index=rev_df.columns, columns=rev_df.columns, dtype=float)
    corr = rev_df.corr()
    # Pandas leaves NaN on flat columns; keep that so the user notices, but
    # always set the self-correlation diagonal to 1 for plotting hygiene.
    for col in corr.columns:
        corr.loc[col, col] = 1.0
    return corr


def compute_zone_stats(rev_df: pd.DataFrame) -> pd.DataFrame:
    """Per-zone mean / std / Sharpe-like ratio, both daily and annualised.

    Sharpe here is the dimensionless mean/std on daily revenue (no
    risk-free rate subtracted — appropriate for a relative screening
    metric, not a financial-product comparison).
    """
    if rev_df.empty:
        return pd.DataFrame(
            columns=[
                "mean_daily", "std_daily", "sharpe_daily",
                "mean_annual", "std_annual",
            ],
        )

    mean_d = rev_df.mean()
    std_d = rev_df.std()
    sharpe = pd.Series(
        np.where(std_d > 0, mean_d / std_d, np.nan),
        index=mean_d.index,
    )
    # Annualise mean linearly; std under iid assumption scales as sqrt(N).
    mean_a = mean_d * DAYS_PER_YEAR
    std_a = std_d * math.sqrt(DAYS_PER_YEAR)
    out = pd.DataFrame({
        "mean_daily": mean_d,
        "std_daily": std_d,
        "sharpe_daily": sharpe,
        "mean_annual": mean_a,
        "std_annual": std_a,
    })
    out.index.name = "zone"
    return out


def _portfolio_stats(
    weights: np.ndarray, mean_d: np.ndarray, cov_d: np.ndarray,
) -> tuple[float, float]:
    """Return (annual return, annual risk) for a weight vector."""
    daily_ret = float(weights @ mean_d)
    daily_var = float(weights @ cov_d @ weights)
    daily_risk = math.sqrt(max(daily_var, 0.0))
    annual_ret = daily_ret * DAYS_PER_YEAR
    annual_risk = daily_risk * math.sqrt(DAYS_PER_YEAR)
    return annual_ret, annual_risk


def _solve_min_variance(
    mean_d: np.ndarray,
    cov_d: np.ndarray,
    target_return_daily: float | None = None,
) -> np.ndarray | None:
    """Long-only QP via SLSQP: min w'Σw  s.t. sum(w)=1, w>=0, optional w'μ=target."""
    n = len(mean_d)
    if n == 0:
        return None
    if n == 1:
        return np.array([1.0])

    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    constraints: list[dict] = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if target_return_daily is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w, t=target_return_daily: float(w @ mean_d - t),
        })

    result = minimize(
        lambda w: float(w @ cov_d @ w),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 200},
    )
    if not result.success:
        return None
    w = np.clip(result.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else None


def compute_min_variance_portfolio(rev_df: pd.DataFrame) -> dict:
    """Long-only weights that minimise daily revenue variance."""
    if rev_df.empty:
        return {"weights": pd.Series(dtype=float), "annual_return": 0.0, "annual_risk": 0.0}
    mean_d = rev_df.mean().to_numpy()
    # Single-row inputs: cov is NaN under the default ddof=1; we still want a
    # usable answer (equal weights, zero risk) instead of propagating NaN.
    if len(rev_df) < 2:
        w = np.full(rev_df.shape[1], 1.0 / rev_df.shape[1])
        annual_ret = float(w @ mean_d) * DAYS_PER_YEAR
        return {
            "weights": pd.Series(w, index=rev_df.columns, name="weight"),
            "annual_return": annual_ret,
            "annual_risk": 0.0,
        }
    cov_d = rev_df.cov().to_numpy()
    w = _solve_min_variance(mean_d, cov_d)
    if w is None:
        # Fallback: equal weight.
        w = np.full(len(mean_d), 1.0 / len(mean_d))
    annual_ret, annual_risk = _portfolio_stats(w, mean_d, cov_d)
    return {
        "weights": pd.Series(w, index=rev_df.columns, name="weight"),
        "annual_return": annual_ret,
        "annual_risk": annual_risk,
    }


def compute_max_sharpe_portfolio(rev_df: pd.DataFrame) -> dict:
    """Long-only weights that maximise Sharpe-like ratio (mean/std, daily)."""
    if rev_df.empty:
        return {"weights": pd.Series(dtype=float), "annual_return": 0.0, "annual_risk": 0.0, "sharpe": 0.0}
    mean_d = rev_df.mean().to_numpy()
    # Single-row inputs: cov is undefined; treat as zero-risk equal-weight so
    # the dashboard doesn't surface NaN risk + NaN Sharpe.
    if len(rev_df) < 2:
        w = np.full(rev_df.shape[1], 1.0 / rev_df.shape[1])
        annual_ret = float(w @ mean_d) * DAYS_PER_YEAR
        return {
            "weights": pd.Series(w, index=rev_df.columns, name="weight"),
            "annual_return": annual_ret,
            "annual_risk": 0.0,
            "sharpe": float("inf") if annual_ret > 0 else 0.0,
        }
    cov_d = rev_df.cov().to_numpy()
    n = len(mean_d)

    if n == 1:
        w = np.array([1.0])
    else:
        def neg_sharpe(w: np.ndarray) -> float:
            ret = float(w @ mean_d)
            var = float(w @ cov_d @ w)
            return -ret / math.sqrt(var) if var > 0 else 0.0

        x0 = np.full(n, 1.0 / n)
        result = minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"ftol": 1e-10, "maxiter": 200},
        )
        if not result.success:
            w = np.full(n, 1.0 / n)
        else:
            w = np.clip(result.x, 0.0, 1.0)
            s = w.sum()
            w = w / s if s > 0 else np.full(n, 1.0 / n)

    annual_ret, annual_risk = _portfolio_stats(w, mean_d, cov_d)
    sharpe = annual_ret / annual_risk if annual_risk > 0 else 0.0
    return {
        "weights": pd.Series(w, index=rev_df.columns, name="weight"),
        "annual_return": annual_ret,
        "annual_risk": annual_risk,
        "sharpe": sharpe,
    }


def compute_efficient_frontier(
    rev_df: pd.DataFrame,
    n_points: int = 30,
) -> pd.DataFrame:
    """Long-only Markowitz frontier sampled at n_points target returns.

    Returns rows of (annual_return, annual_risk, weight_<zone>...) for
    plotting and for downstream weight comparison. With <2 zones the
    frontier collapses to a single point and we return that one row.
    """
    if rev_df.empty:
        return pd.DataFrame(columns=["annual_return", "annual_risk"])

    mean_d = rev_df.mean().to_numpy()
    n = len(mean_d)

    # Single-row inputs: cov is undefined under ddof=1. Return one equal-weight
    # row with zero risk so the UI has something stable to plot.
    if len(rev_df) < 2:
        w = np.full(n, 1.0 / n)
        ret_a = float(w @ mean_d) * DAYS_PER_YEAR
        row = {"annual_return": ret_a, "annual_risk": 0.0}
        for zone, weight in zip(rev_df.columns, w, strict=True):
            row[f"weight_{zone}"] = float(weight)
        return pd.DataFrame([row])

    cov_d = rev_df.cov().to_numpy()

    if n == 1:
        ret_a, risk_a = _portfolio_stats(np.array([1.0]), mean_d, cov_d)
        row = {"annual_return": ret_a, "annual_risk": risk_a, f"weight_{rev_df.columns[0]}": 1.0}
        return pd.DataFrame([row])

    # Sample target daily returns between the worst-single-zone and best-
    # single-zone means; for each, solve min-variance subject to that return.
    target_lo = float(mean_d.min())
    target_hi = float(mean_d.max())
    if target_hi - target_lo < 1e-9:
        # All zones share the same mean — frontier collapses to one point.
        w = np.full(n, 1.0 / n)
        ret_a, risk_a = _portfolio_stats(w, mean_d, cov_d)
        row = {"annual_return": ret_a, "annual_risk": risk_a}
        for zone, weight in zip(rev_df.columns, w, strict=True):
            row[f"weight_{zone}"] = float(weight)
        return pd.DataFrame([row])
    else:
        target_grid = np.linspace(target_lo, target_hi, n_points)

    rows: list[dict] = []
    for t in target_grid:
        w = _solve_min_variance(mean_d, cov_d, target_return_daily=float(t))
        if w is None:
            continue
        ret_a, risk_a = _portfolio_stats(w, mean_d, cov_d)
        row = {"annual_return": ret_a, "annual_risk": risk_a}
        for zone, weight in zip(rev_df.columns, w, strict=True):
            row[f"weight_{zone}"] = float(weight)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["annual_return", "annual_risk"])

    # Filter to the upper branch: for a given risk level, only keep the
    # portfolio with the highest return. Sampling target returns from min
    # to max single-zone mean otherwise leaves dominated lower-branch
    # points on the chart, which is not what "efficient" means.
    df = pd.DataFrame(rows).sort_values("annual_risk").reset_index(drop=True)
    cummax_return = df["annual_return"].cummax()
    df = df[df["annual_return"] >= cummax_return - 1e-9].reset_index(drop=True)
    return df
