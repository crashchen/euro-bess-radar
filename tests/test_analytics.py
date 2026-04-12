"""Tests for analytics module — all with synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    _find_daily_ordered_trade,
    analyze_renewable_bess_signal,
    build_price_heatmap,
    build_daily_renewable_spread_view,
    build_renewable_price_scatter,
    build_spread_heatmap,
    calculate_daily_spreads,
    calculate_monthly_spreads,
    calculate_negative_price_hours,
    calculate_spread_percentiles,
    compare_zones,
    estimate_annual_arbitrage_revenue,
)
from src.config import get_zone_timezone


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def seven_day_prices() -> pd.DataFrame:
    """7 days of hourly prices with a sine-wave pattern.

    Price = 60 + 40*sin(2*pi*hour/24), giving:
    - daily max = 100 (at hour 6)
    - daily min = 20  (at hour 18)
    - best ordered daily spread = 40 (buy at hour 0, sell at hour 6)
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


@pytest.fixture
def half_hour_negative_prices() -> pd.DataFrame:
    """24h of 30-min prices with half the intervals negative."""
    idx = pd.date_range("2025-03-01", periods=48, freq="30min", tz="UTC")
    prices = [-20.0] * 24 + [40.0] * 24
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    return df


# ── Daily spreads ────────────────────────────────────────────────────────────

class TestDailySpreads:
    def test_correct_spread(self, seven_day_prices: pd.DataFrame) -> None:
        result = calculate_daily_spreads(seven_day_prices)
        assert len(result) == 7
        assert "spread" in result.columns
        assert result["spread"].between(39.9, 40.1).all()

    def test_single_day(self, single_day_prices: pd.DataFrame) -> None:
        result = calculate_daily_spreads(single_day_prices)
        assert len(result) == 1
        assert result["spread"].iloc[0] == 23.0  # max=23, min=0
        assert result["max_hour"].iloc[0] == 23
        assert result["min_hour"].iloc[0] == 0

    def test_descending_day_has_zero_ordered_spread(self) -> None:
        idx = pd.date_range("2025-06-15", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"price_eur_mwh": list(range(23, -1, -1))}, index=idx)
        df.index.name = "timestamp"

        result = calculate_daily_spreads(df)
        assert result["spread"].iloc[0] == 0.0

    def test_two_hour_window_uses_rolling_average(self, single_day_prices: pd.DataFrame) -> None:
        result = calculate_daily_spreads(single_day_prices, duration_hours=2.0)
        assert result["spread"].iloc[0] == 22.0
        assert result["max_hour"].iloc[0] == 22
        assert result["min_hour"].iloc[0] == 0

    def test_two_hour_window_indices_still_point_to_window_start(self) -> None:
        idx = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
        prices = pd.Series([50.0, 0.0, 0.0, 10.0, 100.0, 100.0], index=idx)

        trade = _find_daily_ordered_trade(prices, duration_hours=2.0)

        assert trade["buy_start_idx"] == 1
        assert trade["sell_start_idx"] == 4

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
        assert result["avg_spread"].iloc[0] > 39

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

    def test_spread_heatmap_marks_selected_charge_and_discharge_windows(
        self, seven_day_prices: pd.DataFrame,
    ) -> None:
        result = build_spread_heatmap(seven_day_prices)
        january = result.columns[0]
        assert result.loc[0, january] == pytest.approx(-40.0)
        assert result.loc[6, january] == pytest.approx(40.0)
        assert abs(result.loc[12, january]) < 1e-6

    def test_spread_heatmap_is_signed_signal_not_hourly_revenue(
        self, seven_day_prices: pd.DataFrame,
    ) -> None:
        """Selected-window signals should net to zero within each month column."""
        result = build_spread_heatmap(seven_day_prices)
        for col in result.columns:
            assert abs(result[col].sum()) < 1.0


# ── Revenue estimation ───────────────────────────────────────────────────────

class TestRevenue:
    def test_hand_calculated(self) -> None:
        """Verify math with a known spread and the default 1 modeled cycle/day."""
        daily = pd.DataFrame({"spread": [100.0] * 365})  # constant 100 EUR spread

        result = estimate_annual_arbitrage_revenue(
            daily, power_mw=1.0, duration_hours=1.0, roundtrip_efficiency=1.0,
        )
        # revenue = 100 * 1 * 1 * 1.0 * 0.70 * 1.0 * 365.25
        expected = 100 * 0.70 * 365.25
        assert abs(result["annual_revenue_eur"] - expected) < 1.0
        assert result["capture_rate_assumption"] == 0.70
        assert result["cycles_per_day_assumption"] == 1.0

    def test_capture_rate_is_adjustable(self) -> None:
        daily = pd.DataFrame({"spread": [100.0] * 365})
        result = estimate_annual_arbitrage_revenue(
            daily,
            power_mw=1.0,
            duration_hours=1.0,
            roundtrip_efficiency=1.0,
            capture_rate=0.55,
        )
        expected = 100 * 0.55 * 365.25
        assert abs(result["annual_revenue_eur"] - expected) < 1.0
        assert result["capture_rate_assumption"] == 0.55

    def test_keys(self, seven_day_prices: pd.DataFrame) -> None:
        daily = calculate_daily_spreads(seven_day_prices)
        result = estimate_annual_arbitrage_revenue(daily)
        expected_keys = {
            "annual_revenue_eur", "annual_revenue_eur_per_mw",
            "avg_daily_revenue", "capture_rate_assumption",
            "cycles_per_day_assumption",
        }
        assert set(result.keys()) == expected_keys


# ── Negative price hours ─────────────────────────────────────────────────────

class TestNegativePrices:
    def test_all_positive(self, seven_day_prices: pd.DataFrame) -> None:
        result = calculate_negative_price_hours(seven_day_prices)
        # Sine: min is 20, no negatives
        assert result["negative_hours"] == 0
        assert result["pct_negative"] == 0.0

    def test_half_negative(self, negative_prices: pd.DataFrame) -> None:
        result = calculate_negative_price_hours(negative_prices)
        assert result["negative_hours"] == 12.0
        assert result["negative_intervals"] == 12
        assert result["pct_negative"] == 50.0
        assert result["avg_negative_price"] == -20.0
        assert result["most_negative_price"] == -20.0

    def test_half_negative_subhourly(self, half_hour_negative_prices: pd.DataFrame) -> None:
        result = calculate_negative_price_hours(half_hour_negative_prices)
        assert result["negative_hours"] == 12.0
        assert result["negative_intervals"] == 24
        assert result["pct_negative"] == 50.0

    def test_all_zero_prices(self) -> None:
        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"price_eur_mwh": [0.0] * 24}, index=idx)
        df.index.name = "timestamp"
        result = calculate_negative_price_hours(df)
        assert result["negative_hours"] == 0


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

    def test_capture_rate_flows_into_revenue(
        self, seven_day_prices: pd.DataFrame,
    ) -> None:
        low_capture = compare_zones(
            {"DE_LU": seven_day_prices},
            capture_rate=0.50,
        )
        high_capture = compare_zones(
            {"DE_LU": seven_day_prices},
            capture_rate=0.80,
        )
        assert (
            high_capture["estimated_annual_revenue_per_mw"].iloc[0]
            > low_capture["estimated_annual_revenue_per_mw"].iloc[0]
        )


# ── Renewable correlation ────────────────────────────────────────────────────

@pytest.fixture
def renewable_signal_prices() -> pd.DataFrame:
    """Hourly prices with day-level spread variation for BESS-signal tests."""
    idx = pd.date_range("2025-01-01", periods=8 * 24, freq="h", tz="UTC")
    hours = np.arange(8 * 24) % 24
    day_idx = np.arange(8 * 24) // 24
    amplitude = 20.0 + day_idx * 5.0
    prices = 60.0 + amplitude * np.sin(2 * np.pi * hours / 24)
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def generation_df() -> pd.DataFrame:
    """Generation data with hourly inverse price signal and daily RE variation."""
    idx = pd.date_range("2025-01-01", periods=8 * 24, freq="h", tz="UTC")
    hours = np.arange(8 * 24) % 24
    day_idx = np.arange(8 * 24) // 24
    renewable_base = 30.0 + day_idx * 5.0
    renewable_pct = renewable_base - 15.0 * np.sin(2 * np.pi * hours / 24)
    renewable_pct = np.clip(renewable_pct, 5.0, 95.0)
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
        self, renewable_signal_prices: pd.DataFrame, generation_df: pd.DataFrame,
    ) -> None:
        """BESS-friendly market should have negative renewable-price correlation."""
        result = analyze_renewable_bess_signal(renewable_signal_prices, generation_df)
        assert result["correlation_renewable_price"] < 0

    def test_high_low_re_prices(
        self, renewable_signal_prices: pd.DataFrame, generation_df: pd.DataFrame,
    ) -> None:
        result = analyze_renewable_bess_signal(renewable_signal_prices, generation_df)
        assert result["avg_price_high_renewable"] < result["avg_price_low_renewable"]

    def test_positive_spread_uplift(
        self, renewable_signal_prices: pd.DataFrame, generation_df: pd.DataFrame,
    ) -> None:
        result = analyze_renewable_bess_signal(renewable_signal_prices, generation_df)
        assert result["spread_uplift_high_vs_low_renewable"] > 0
        assert (
            result["spread_by_renewable_quartile"]["Q4"]
            > result["spread_by_renewable_quartile"]["Q1"]
        )

    def test_result_keys(
        self, renewable_signal_prices: pd.DataFrame, generation_df: pd.DataFrame,
    ) -> None:
        result = analyze_renewable_bess_signal(renewable_signal_prices, generation_df)
        expected_keys = {
            "correlation_renewable_price", "avg_price_high_renewable",
            "avg_price_low_renewable", "avg_spread_high_renewable_day",
            "avg_spread_low_renewable_day", "spread_uplift_high_vs_low_renewable",
            "price_by_renewable_quartile", "spread_by_renewable_quartile",
            "hourly_points", "daily_points",
        }
        assert set(result.keys()) == expected_keys

    def test_quartile_outputs_complete_and_stable(
        self, renewable_signal_prices: pd.DataFrame, generation_df: pd.DataFrame,
    ) -> None:
        result = analyze_renewable_bess_signal(renewable_signal_prices, generation_df)
        assert list(result["price_by_renewable_quartile"].keys()) == ["Q1", "Q2", "Q3", "Q4"]
        assert list(result["spread_by_renewable_quartile"].keys()) == ["Q1", "Q2", "Q3", "Q4"]

    def test_empty_generation(self, renewable_signal_prices: pd.DataFrame) -> None:
        empty_gen = pd.DataFrame(columns=[
            "solar_mw", "wind_onshore_mw", "wind_offshore_mw",
            "total_renewable_mw", "total_generation_mw", "renewable_pct",
        ])
        result = analyze_renewable_bess_signal(renewable_signal_prices, empty_gen)
        assert result["correlation_renewable_price"] is None
        assert result["price_by_renewable_quartile"] == {}
        assert result["spread_by_renewable_quartile"] == {}

    def test_insufficient_daily_overlap(
        self, renewable_signal_prices: pd.DataFrame, generation_df: pd.DataFrame,
    ) -> None:
        result = analyze_renewable_bess_signal(
            renewable_signal_prices,
            generation_df.iloc[:48],
        )
        assert result["daily_points"] == 2
        assert result["spread_uplift_high_vs_low_renewable"] is None
        assert result["spread_by_renewable_quartile"] == {}

    def test_scatter_df(
        self, renewable_signal_prices: pd.DataFrame, generation_df: pd.DataFrame,
    ) -> None:
        result = build_renewable_price_scatter(renewable_signal_prices, generation_df)
        assert "price_eur_mwh" in result.columns
        assert "renewable_pct" in result.columns
        assert "hour" in result.columns
        assert "month" in result.columns
        assert len(result) > 0

    def test_daily_view_respects_timezone(self, cet_boundary_prices: pd.DataFrame) -> None:
        idx = cet_boundary_prices.index
        gen_df = pd.DataFrame({
            "solar_mw": [0.0] * len(idx),
            "wind_onshore_mw": [10.0] * len(idx),
            "wind_offshore_mw": [0.0] * len(idx),
            "total_renewable_mw": [10.0] * len(idx),
            "total_generation_mw": [100.0] * len(idx),
            "renewable_pct": [10.0] * len(idx),
        }, index=idx)
        gen_df.iloc[23, gen_df.columns.get_loc("renewable_pct")] = 90.0

        utc_view = build_daily_renewable_spread_view(cet_boundary_prices, gen_df)
        cet_view = build_daily_renewable_spread_view(
            cet_boundary_prices,
            gen_df,
            tz="Europe/Berlin",
        )

        assert utc_view["daily_avg_renewable_pct"].tolist() != cet_view["daily_avg_renewable_pct"].tolist()

    def test_daily_view_respects_duration_hours(
        self, renewable_signal_prices: pd.DataFrame, generation_df: pd.DataFrame,
    ) -> None:
        view_1h = build_daily_renewable_spread_view(
            renewable_signal_prices,
            generation_df,
            duration_hours=1.0,
        )
        view_2h = build_daily_renewable_spread_view(
            renewable_signal_prices,
            generation_df,
            duration_hours=2.0,
        )

        assert view_1h["spread"].mean() != view_2h["spread"].mean()


# ── Local timezone analytics ────────────────────────────────────────────────

@pytest.fixture
def cet_boundary_prices() -> pd.DataFrame:
    """Prices that straddle a UTC/CET day boundary.

    UTC 2025-01-01 22:00 = CET 2025-01-01 23:00  (still Jan 1 in CET)
    UTC 2025-01-01 23:00 = CET 2025-01-02 00:00  (Jan 2 in CET, Jan 1 in UTC)
    UTC 2025-01-02 00:00 = CET 2025-01-02 01:00  (Jan 2 in both)

    We create 48h of data (Jan 1 00:00 UTC to Jan 2 23:00 UTC).
    In UTC: two days (Jan 1, Jan 2).
    In CET: the last hour of UTC Jan 1 (23:00) belongs to CET Jan 2.
    """
    idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    # Make Jan 1 UTC 23:00 an extreme spike (200). In UTC grouping this
    # lands on Jan 1; in CET grouping it lands on Jan 2.
    prices = [50.0] * 48
    prices[23] = 200.0  # UTC 23:00 Jan 1 = CET 00:00 Jan 2
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    return df


class TestLocalTimezoneAnalytics:
    def test_daily_spread_utc_vs_cet(self, cet_boundary_prices: pd.DataFrame) -> None:
        """The Jan 1 UTC spike is not capturable once it shifts to the next CET day."""
        utc_result = calculate_daily_spreads(cet_boundary_prices)
        cet_result = calculate_daily_spreads(cet_boundary_prices, tz="Europe/Berlin")

        # In UTC: Jan 1 includes the spike → spread=150
        utc_jan1 = utc_result[utc_result["date"].astype(str) == "2025-01-01"]
        assert utc_jan1["spread"].iloc[0] == 150.0

        # In CET: Jan 1 does NOT include the spike (it moved to Jan 2)
        cet_jan1 = cet_result[cet_result["date"].astype(str) == "2025-01-01"]
        assert cet_jan1["spread"].iloc[0] == 0.0  # all 50.0 in CET Jan 1

        # In CET: Jan 2 includes the spike at local midnight, so there is still
        # no profitable charge-before-discharge pair within that day.
        cet_jan2 = cet_result[cet_result["date"].astype(str) == "2025-01-02"]
        assert cet_jan2["spread"].iloc[0] == 0.0

    def test_heatmap_hour_shifted(self, cet_boundary_prices: pd.DataFrame) -> None:
        """Heatmap hour should reflect local time, not UTC."""
        utc_hm = build_price_heatmap(cet_boundary_prices)
        cet_hm = build_price_heatmap(cet_boundary_prices, tz="Europe/Berlin")

        # The spike is at UTC hour 23. In CET it's hour 0 (next day).
        # UTC heatmap: hour 23 should have the spike
        assert utc_hm.loc[23].max() > 100
        # CET heatmap: hour 0 should have the spike
        assert cet_hm.loc[0].max() > 100

    def test_spread_heatmap_uses_local_time(self, cet_boundary_prices: pd.DataFrame) -> None:
        """Spread heatmap deviation should use local-time day mean."""
        utc_shm = build_spread_heatmap(cet_boundary_prices)
        cet_shm = build_spread_heatmap(cet_boundary_prices, tz="Europe/Berlin")
        # Both should produce 24 rows
        assert utc_shm.shape[0] == 24
        assert cet_shm.shape[0] == 24

    def test_backward_compat_no_tz(self, seven_day_prices: pd.DataFrame) -> None:
        """Passing tz=None should behave identically to the old code."""
        result_none = calculate_daily_spreads(seven_day_prices, tz=None)
        result_default = calculate_daily_spreads(seven_day_prices)
        pd.testing.assert_frame_equal(result_none, result_default)

    def test_get_zone_timezone_known(self) -> None:
        assert get_zone_timezone("DE_LU") == "Europe/Berlin"
        assert get_zone_timezone("GB") == "Europe/London"
        assert get_zone_timezone("FI") == "Europe/Helsinki"

    def test_get_zone_timezone_unknown_falls_back(self) -> None:
        assert get_zone_timezone("UNKNOWN") == "UTC"

    def test_compare_zones_with_timezones(
        self, seven_day_prices: pd.DataFrame,
    ) -> None:
        """compare_zones should accept zone_timezones dict."""
        result = compare_zones(
            {"DE_LU": seven_day_prices},
            zone_timezones={"DE_LU": "Europe/Berlin"},
        )
        assert len(result) == 1
