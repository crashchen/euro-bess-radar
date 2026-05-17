"""Tests for cross-zone portfolio analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    DAYS_PER_YEAR,
    build_daily_revenue_matrix,
    compute_correlation_matrix,
    compute_efficient_frontier,
    compute_max_sharpe_portfolio,
    compute_min_variance_portfolio,
    compute_zone_stats,
)


def _make_zone_prices(start: str, n_days: int, base: float, daily_swing: float) -> pd.DataFrame:
    """Build a 24h price series with a deterministic daily peak-trough pattern."""
    idx = pd.date_range(start, periods=24 * n_days, freq="h", tz="UTC")
    # Trough at hour 5, peak at hour 17 — a clean spread for ordered-dispatch math.
    hours = np.arange(len(idx)) % 24
    prices = base + daily_swing * np.sin((hours - 5) * np.pi / 12)
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def synthetic_rev_df() -> pd.DataFrame:
    """Three-zone daily-revenue matrix with known correlation structure."""
    rng = np.random.default_rng(0)
    n = 60
    a = rng.normal(100, 20, n)
    b = 0.6 * a + rng.normal(80, 10, n)
    c = -0.4 * a + rng.normal(70, 25, n) + 100
    dates = pd.date_range("2025-01-01", periods=n, freq="D").date
    return pd.DataFrame(
        {"A": a, "B": b, "C": c}, index=pd.Index(dates, name="date"),
    )


class TestBuildDailyRevenueMatrix:
    def test_empty_input_returns_empty(self) -> None:
        out = build_daily_revenue_matrix({})
        assert out.empty

    def test_single_zone_produces_one_column(self) -> None:
        zd = {"DE_LU": _make_zone_prices("2025-01-01", 5, base=50.0, daily_swing=30.0)}
        out = build_daily_revenue_matrix(zd)
        assert list(out.columns) == ["DE_LU"]
        assert len(out) == 5
        assert (out["DE_LU"] > 0).all()  # Revenue per MW > 0 with non-zero spread

    def test_inner_join_on_common_dates(self) -> None:
        """Zones with different date ranges must intersect: the frame only
        contains dates where every zone has a valid daily-spread row.
        """
        zd = {
            "A": _make_zone_prices("2025-01-01", 7, 50.0, 30.0),
            # B starts 2 days later
            "B": _make_zone_prices("2025-01-03", 7, 60.0, 25.0),
        }
        out = build_daily_revenue_matrix(zd)
        assert set(out.columns) == {"A", "B"}
        # Intersection: 2025-01-03 .. 2025-01-07 = 5 days
        assert len(out) == 5
        # No NaN — that would defeat correlation math.
        assert not out.isna().any().any()

    def test_per_mw_normalisation(self) -> None:
        zd = {"A": _make_zone_prices("2025-01-01", 5, 50.0, 30.0)}
        out_1mw = build_daily_revenue_matrix(zd, power_mw=1.0)
        out_5mw = build_daily_revenue_matrix(zd, power_mw=5.0)
        # Per-MW normalisation makes the two outputs equal up to ordering of
        # operations; sqrt(eff) and capture cancel out at the same power.
        pd.testing.assert_series_equal(out_1mw["A"], out_5mw["A"], check_exact=False)


class TestCorrelationMatrix:
    def test_symmetric_and_unit_diagonal(self, synthetic_rev_df: pd.DataFrame) -> None:
        corr = compute_correlation_matrix(synthetic_rev_df)
        # Symmetric
        np.testing.assert_allclose(corr.values, corr.values.T)
        # Diagonal == 1
        for col in corr.columns:
            assert corr.loc[col, col] == pytest.approx(1.0)

    def test_known_sign_structure(self, synthetic_rev_df: pd.DataFrame) -> None:
        corr = compute_correlation_matrix(synthetic_rev_df)
        # A and B positively correlated; A and C anti-correlated by design.
        assert corr.loc["A", "B"] > 0
        assert corr.loc["A", "C"] < 0

    def test_empty_input_returns_empty(self) -> None:
        out = compute_correlation_matrix(pd.DataFrame())
        assert out.empty


class TestZoneStats:
    def test_mean_std_sharpe_columns(self, synthetic_rev_df: pd.DataFrame) -> None:
        stats = compute_zone_stats(synthetic_rev_df)
        assert {"mean_daily", "std_daily", "sharpe_daily", "mean_annual", "std_annual"} <= set(stats.columns)
        # Annual mean = daily mean * 365.25
        for zone in stats.index:
            assert stats.loc[zone, "mean_annual"] == pytest.approx(
                stats.loc[zone, "mean_daily"] * DAYS_PER_YEAR,
            )

    def test_annual_std_scales_by_sqrt_n(self, synthetic_rev_df: pd.DataFrame) -> None:
        stats = compute_zone_stats(synthetic_rev_df)
        for zone in stats.index:
            assert stats.loc[zone, "std_annual"] == pytest.approx(
                stats.loc[zone, "std_daily"] * math.sqrt(DAYS_PER_YEAR),
            )

    def test_flat_zone_yields_nan_sharpe(self) -> None:
        flat = pd.DataFrame({"X": [100.0] * 30})
        stats = compute_zone_stats(flat)
        assert pd.isna(stats.loc["X", "sharpe_daily"])


class TestMinVariancePortfolio:
    def test_weights_sum_to_one_and_long_only(self, synthetic_rev_df: pd.DataFrame) -> None:
        out = compute_min_variance_portfolio(synthetic_rev_df)
        w = out["weights"]
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-9).all()

    def test_min_var_risk_below_any_single_zone(self, synthetic_rev_df: pd.DataFrame) -> None:
        """Diversification: portfolio risk must be at most the worst single
        zone's risk; with anti-correlated zones it should be strictly lower
        than the best single zone too.
        """
        out = compute_min_variance_portfolio(synthetic_rev_df)
        zone_std_annual = synthetic_rev_df.std() * math.sqrt(DAYS_PER_YEAR)
        assert out["annual_risk"] < zone_std_annual.min() + 1e-6

    def test_single_zone_collapses(self) -> None:
        df = pd.DataFrame({"only": [10.0, 12.0, 9.0, 11.0]})
        out = compute_min_variance_portfolio(df)
        assert out["weights"].iloc[0] == pytest.approx(1.0)


class TestMaxSharpePortfolio:
    def test_weights_sum_to_one(self, synthetic_rev_df: pd.DataFrame) -> None:
        out = compute_max_sharpe_portfolio(synthetic_rev_df)
        assert out["weights"].sum() == pytest.approx(1.0, abs=1e-6)
        assert (out["weights"] >= -1e-9).all()

    def test_sharpe_at_least_best_single_zone(self, synthetic_rev_df: pd.DataFrame) -> None:
        """The optimised portfolio Sharpe must be >= the best single-zone
        Sharpe (the single zone is a feasible portfolio).
        """
        out = compute_max_sharpe_portfolio(synthetic_rev_df)
        stats = compute_zone_stats(synthetic_rev_df)
        best_single = (stats["mean_annual"] / stats["std_annual"]).max()
        assert out["sharpe"] >= best_single - 1e-6


class TestEfficientFrontier:
    def test_frontier_returns_dataframe_with_expected_columns(
        self, synthetic_rev_df: pd.DataFrame,
    ) -> None:
        ef = compute_efficient_frontier(synthetic_rev_df, n_points=20)
        assert "annual_return" in ef.columns
        assert "annual_risk" in ef.columns
        for zone in synthetic_rev_df.columns:
            assert f"weight_{zone}" in ef.columns

    def test_frontier_weights_sum_to_one(
        self, synthetic_rev_df: pd.DataFrame,
    ) -> None:
        ef = compute_efficient_frontier(synthetic_rev_df, n_points=15)
        weight_cols = [c for c in ef.columns if c.startswith("weight_")]
        sums = ef[weight_cols].sum(axis=1)
        assert (sums - 1.0).abs().max() < 1e-6

    def test_frontier_risk_is_sorted_ascending(
        self, synthetic_rev_df: pd.DataFrame,
    ) -> None:
        ef = compute_efficient_frontier(synthetic_rev_df, n_points=15)
        # Sort key in implementation; assert it stays monotone.
        assert (ef["annual_risk"].diff().dropna() >= -1e-9).all()

    def test_single_zone_returns_one_row(self) -> None:
        df = pd.DataFrame({"only": [10.0, 12.0, 9.0, 11.0]})
        ef = compute_efficient_frontier(df)
        assert len(ef) == 1
        assert ef["weight_only"].iloc[0] == pytest.approx(1.0)

    def test_empty_input_returns_empty(self) -> None:
        ef = compute_efficient_frontier(pd.DataFrame())
        assert ef.empty

    def test_dominated_lower_branch_filtered(self) -> None:
        """An 'efficient' frontier must not contain dominated portfolios:
        for a given risk level, only the highest-return point should
        survive. Sampling target returns from worst-to-best mean otherwise
        leaves lower-branch points on the chart that are strictly worse
        than the upper branch.
        """
        # high_ret_low_risk dominates: 100% weight there is the only
        # point on the true efficient frontier.
        df = pd.DataFrame({
            "low_ret_high_risk": [0.0, 20.0, 0.0, 20.0],
            "high_ret_low_risk": [19.0, 21.0, 19.0, 21.0],
        })
        ef = compute_efficient_frontier(df, n_points=10)
        # Either the frontier is a single point (the dominant zone) OR all
        # returns are strictly non-decreasing as risk increases.
        assert (ef["annual_return"].diff().dropna() >= -1e-6).all()

    def test_single_aligned_day_returns_zero_risk(self) -> None:
        """With only one common date, daily covariance is undefined; the
        helpers must still produce a usable equal-weight result rather
        than NaN risk that leaks into the UI.
        """
        df = pd.DataFrame({"A": [10.0], "B": [12.0]})
        mv = compute_min_variance_portfolio(df)
        ms = compute_max_sharpe_portfolio(df)
        ef = compute_efficient_frontier(df)
        assert mv["annual_risk"] == 0.0
        assert ms["annual_risk"] == 0.0
        assert len(ef) == 1
        assert ef["annual_risk"].iloc[0] == 0.0

    def test_all_zones_same_mean_collapses_to_one_point(self) -> None:
        """When every zone has the same mean revenue (degenerate frontier),
        return a single equal-weight point instead of an empty frame.
        """
        df = pd.DataFrame({"A": [1.0] * 4, "B": [1.0] * 4})
        ef = compute_efficient_frontier(df)
        assert len(ef) == 1
        assert ef["annual_risk"].iloc[0] == pytest.approx(0.0)
