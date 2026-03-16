"""Tests for analytics module — all with synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    analyze_price_renewable_correlation,
    build_price_heatmap,
    build_renewable_price_scatter,
    build_spread_heatmap,
    calculate_daily_spreads,
    calculate_monthly_spreads,
    calculate_negative_price_hours,
    calculate_spread_percentiles,
    compare_zones,
    estimate_annual_arbitrage_revenue,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def seven_day_prices() -> pd.DataFrame:
    """7 days of hourly prices with a sine-wave pattern.

    Price = 60 + 40*sin(2*pi*hour/24), giving:
    - daily max = 100 (at hour 6)
    - daily min = 20  (at hour 18)
    - daily spread = 80
    """
    idx = pd.date_range("2025-01-01", periods=7 * 24, freq="h", tz="UTC")
    hours = np.arange(7 * 24) % 24
    prices = 60.0 + 40.0 * np.sin(2 * np.pi * hours / 24)
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def single_day_prices() -> pd.DataFrame:
    """1 day of hourly prices — linear ramp 0..23."""
    idx = pd.date_range("2025-06-15", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"price_eur_mwh": list(range(24))}, index=idx)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def negative_prices() -> pd.DataFrame:
    """24h with half negative, half positive prices."""
    idx = pd.date_range("2025-03-01", periods=24, freq="h", tz="UTC")
    prices = [-20.0] * 12 + [40.0] * 12
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    return df


# ── Daily spreads ────────────────────────────────────────────────────────────

class TestDailySpreads:
    def test_correct_spread(self, seven_day_prices: pd.DataFrame) -> None:
        result = calculate_daily_spreads(seven_day_prices)
        assert len(result) == 7
        assert "spread" in result.columns
        # Sine wave: max ≈ 100, min ≈ 20, spread ≈ 80
        assert all(result["spread"] > 70)

    def test_single_day(self, single_day_prices: pd.DataFrame) -> None:
        result = calculate_daily_spreads(single_day_prices)
        assert len(result) == 1
        assert result["spread"].iloc[0] == 23.0  # max=23, min=0
        assert result["max_hour"].iloc[0] == 23
        assert result["min_hour"].iloc[0] == 0

    def test_columns(self, seven_day_prices: pd.DataFrame) -> None:
        result = calculate_daily_spreads(seven_day_prices)
        expected_cols = {"date", "daily_min", "daily_max", "spread", "max_hour", "min_hour"}
        assert set(result.columns) == expected_cols


# ── Monthly spreads ──────────────────────────────────────────────────────────

class TestMonthlySpreads:
    def test_aggregation(self, seven_day_prices: pd.DataFrame) -> None:
        result = calculate_monthly_spreads(seven_day_prices)
        assert len(result) >= 1
        assert "avg_spread" in result.columns
        assert result["avg_spread"].iloc[0] > 70

    def test_columns(self, seven_day_prices: pd.DataFrame) -> None:
        result = calculate_monthly_spreads(seven_day_prices)
        expected = {"year_month", "avg_spread", "median_spread", "max_spread",
                    "min_spread", "avg_daily_max", "avg_daily_min"}
        assert set(result.columns) == expected


# ── Percentiles ──────────────────────────────────────────────────────────────

class TestPercentiles:
    def test_keys(self, seven_day_prices: pd.DataFrame) -> None:
        daily = calculate_daily_spreads(seven_day_prices)
        result = calculate_spread_percentiles(daily)
        assert set(result.keys()) == {"p50", "p75", "p90", "mean", "std"}

    def test_p50_equals_median(self, seven_day_prices: pd.DataFrame) -> None:
        daily = calculate_daily_spreads(seven_day_prices)
        result = calculate_spread_percentiles(daily)
        assert abs(result["p50"] - daily["spread"].median()) < 0.01


# ── Heatmaps ─────────────────────────────────────────────────────────────────

class TestHeatmaps:
    def test_price_heatmap_shape(self, seven_day_prices: pd.DataFrame) -> None:
        result = build_price_heatmap(seven_day_prices)
        assert result.shape[0] == 24  # 24 hours
        assert result.shape[1] >= 1   # at least 1 month column

    def test_spread_heatmap_shape(self, seven_day_prices: pd.DataFrame) -> None:
        result = build_spread_heatmap(seven_day_prices)
        assert result.shape[0] == 24

    def test_spread_heatmap_sums_near_zero(self, seven_day_prices: pd.DataFrame) -> None:
        """Deviations from daily mean should roughly sum to zero per column."""
        result = build_spread_heatmap(seven_day_prices)
        for col in result.columns:
            assert abs(result[col].sum()) < 1.0


# ── Revenue estimation ───────────────────────────────────────────────────────

class TestRevenue:
    def test_hand_calculated(self) -> None:
        """Verify math with a known spread."""
        daily = pd.DataFrame({"spread": [100.0] * 365})  # constant 100 EUR spread

        result = estimate_annual_arbitrage_revenue(
            daily, power_mw=1.0, duration_hours=1.0,
            roundtrip_efficiency=1.0, cycles_per_day=1.0,
        )
        # revenue = 100 * 1 * 1 * 1.0 * 0.70 * 1.0 * 365.25
        expected = 100 * 0.70 * 365.25
        assert abs(result["annual_revenue_eur"] - expected) < 1.0
        assert result["capture_rate_assumption"] == 0.70

    def test_keys(self, seven_day_prices: pd.DataFrame) -> None:
        daily = calculate_daily_spreads(seven_day_prices)
        result = estimate_annual_arbitrage_revenue(daily)
        expected_keys = {
            "annual_revenue_eur", "annual_revenue_eur_per_mw",
            "avg_daily_revenue", "capture_rate_assumption",
        }
        assert set(result.keys()) == expected_keys


# ── Negative price hours ─────────────────────────────────────────────────────

class TestNegativePrices:
    def test_all_positive(self, seven_day_prices: pd.DataFrame) -> None:
        result = calculate_negative_price_hours(seven_day_prices)
        # Sine: min is 20, no negatives
        assert result["total_negative_hours"] == 0
        assert result["pct_negative"] == 0.0

    def test_half_negative(self, negative_prices: pd.DataFrame) -> None:
        result = calculate_negative_price_hours(negative_prices)
        assert result["total_negative_hours"] == 12
        assert result["pct_negative"] == 50.0
        assert result["avg_negative_price"] == -20.0
        assert result["most_negative_price"] == -20.0

    def test_all_zero_prices(self) -> None:
        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"price_eur_mwh": [0.0] * 24}, index=idx)
        df.index.name = "timestamp"
        result = calculate_negative_price_hours(df)
        assert result["total_negative_hours"] == 0


# ── Zone comparison ──────────────────────────────────────────────────────────

class TestCompareZones:
    def test_multi_zone(self, seven_day_prices: pd.DataFrame, single_day_prices: pd.DataFrame) -> None:
        result = compare_zones({"DE_LU": seven_day_prices, "FR": single_day_prices})
        assert len(result) == 2
        assert set(result["zone"]) == {"DE_LU", "FR"}

    def test_empty_zone_skipped(self, seven_day_prices: pd.DataFrame) -> None:
        empty_df = pd.DataFrame(columns=["price_eur_mwh"])
        result = compare_zones({"DE_LU": seven_day_prices, "GB": empty_df})
        assert len(result) == 1
        assert result["zone"].iloc[0] == "DE_LU"

    def test_columns(self, seven_day_prices: pd.DataFrame) -> None:
        result = compare_zones({"DE_LU": seven_day_prices})
        expected = {"zone", "avg_price", "std_price", "avg_spread", "p50_spread",
                    "p90_spread", "negative_pct", "estimated_annual_revenue_per_mw"}
        assert set(result.columns) == expected


# ── Renewable correlation ────────────────────────────────────────────────────

@pytest.fixture
def generation_df() -> pd.DataFrame:
    """Generation data inversely correlated with prices (BESS-friendly)."""
    idx = pd.date_range("2025-01-01", periods=7 * 24, freq="h", tz="UTC")
    hours = np.arange(7 * 24) % 24
    # High renewables when prices are low (inverse of sine pattern)
    renewable_pct = 50.0 - 30.0 * np.sin(2 * np.pi * hours / 24)
    total_gen = 40000.0 + np.random.default_rng(42).normal(0, 500, len(idx))
    solar = total_gen * (renewable_pct / 100) * 0.4
    wind = total_gen * (renewable_pct / 100) * 0.6

    df = pd.DataFrame({
        "solar_mw": solar,
        "wind_onshore_mw": wind,
        "wind_offshore_mw": np.zeros(len(idx)),
        "total_renewable_mw": solar + wind,
        "total_generation_mw": total_gen,
        "renewable_pct": renewable_pct,
    }, index=idx)
    df.index.name = "timestamp"
    return df


class TestRenewableCorrelation:
    def test_negative_correlation(
        self, seven_day_prices: pd.DataFrame, generation_df: pd.DataFrame,
    ) -> None:
        """BESS-friendly market should have negative wind/solar-price correlation."""
        result = analyze_price_renewable_correlation(seven_day_prices, generation_df)
        assert result["correlation_renewable_price"] < 0

    def test_high_low_re_prices(
        self, seven_day_prices: pd.DataFrame, generation_df: pd.DataFrame,
    ) -> None:
        result = analyze_price_renewable_correlation(seven_day_prices, generation_df)
        assert result["avg_price_high_renewable"] < result["avg_price_low_renewable"]

    def test_result_keys(
        self, seven_day_prices: pd.DataFrame, generation_df: pd.DataFrame,
    ) -> None:
        result = analyze_price_renewable_correlation(seven_day_prices, generation_df)
        expected_keys = {
            "correlation_wind_price", "correlation_solar_price",
            "correlation_renewable_price", "avg_price_high_renewable",
            "avg_price_low_renewable", "price_spread_by_renewable_quartile",
            "negative_price_renewable_pct",
        }
        assert set(result.keys()) == expected_keys

    def test_empty_generation(self, seven_day_prices: pd.DataFrame) -> None:
        empty_gen = pd.DataFrame(columns=[
            "solar_mw", "wind_onshore_mw", "wind_offshore_mw",
            "total_renewable_mw", "total_generation_mw", "renewable_pct",
        ])
        result = analyze_price_renewable_correlation(seven_day_prices, empty_gen)
        assert result["correlation_renewable_price"] == 0.0

    def test_scatter_df(
        self, seven_day_prices: pd.DataFrame, generation_df: pd.DataFrame,
    ) -> None:
        result = build_renewable_price_scatter(seven_day_prices, generation_df)
        assert "price_eur_mwh" in result.columns
        assert "renewable_pct" in result.columns
        assert "hour" in result.columns
        assert len(result) > 0
