"""Tests for LP-based multi-cycle BESS dispatch optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dispatch import solve_daily_lp, solve_dispatch_batch


class TestSolveDailyLp:
    def test_flat_prices_yield_zero_revenue(self) -> None:
        """Flat prices offer no arbitrage; revenue should be ~0."""
        prices = np.array([50.0] * 24)
        r = solve_daily_lp(prices, dt=1.0, power_mw=1.0, duration_hours=2.0, efficiency=0.88)
        assert abs(r["revenue_eur"]) < 0.01
        assert r["n_cycles"] == pytest.approx(0.0, abs=0.01)

    def test_simple_spread(self) -> None:
        """Known 2-block price pattern should produce positive revenue."""
        prices = np.array([20.0] * 12 + [80.0] * 12)
        r = solve_daily_lp(prices, dt=1.0, power_mw=1.0, duration_hours=2.0, efficiency=0.88)
        assert r["revenue_eur"] > 0
        # Revenue should be less than theoretical max (spread * capacity * cycles)
        # because of efficiency losses
        assert r["revenue_eur"] < 60.0 * 2.0  # spread=60, cap=2MWh

    def test_perfect_efficiency(self) -> None:
        """With eff=1.0 and soc_init=0, one full cycle captures ~spread * capacity minus VOM."""
        prices = np.array([20.0] * 12 + [80.0] * 12)
        r = solve_daily_lp(
            prices, dt=1.0, power_mw=1.0, duration_hours=2.0,
            efficiency=1.0, soc_init_frac=0.0,
        )
        # charge 2MWh @ 20+0.5 VOM, discharge 2MWh @ 80-0.5 VOM
        # revenue = (79.5-20.5)*2 = 118
        assert r["revenue_eur"] == pytest.approx(118.0, rel=1e-4)

    def test_soc_never_exceeds_capacity(self) -> None:
        """SoC must stay within [0, capacity_mwh] at all times."""
        prices = np.array([10.0] * 6 + [100.0] * 6 + [10.0] * 6 + [100.0] * 6)
        r = solve_daily_lp(prices, dt=1.0, power_mw=1.0, duration_hours=2.0, efficiency=0.90)
        capacity = 1.0 * 2.0
        assert r["soc"].min() >= -1e-6
        assert r["soc"].max() <= capacity + 1e-6

    def test_soc_terminal_constraint(self) -> None:
        """Final SoC must be >= initial SoC (energy-neutral)."""
        prices = np.array([20.0] * 12 + [80.0] * 12)
        r = solve_daily_lp(prices, dt=1.0, power_mw=1.0, duration_hours=2.0,
                           efficiency=0.88, soc_init_frac=0.5)
        soc_init = 0.5 * 1.0 * 2.0
        assert r["soc"][-1] >= soc_init - 1e-6

    def test_efficiency_reduces_revenue(self) -> None:
        """Lower efficiency should produce lower revenue for the same price pattern."""
        prices = np.array([20.0] * 12 + [80.0] * 12)
        r_high = solve_daily_lp(prices, dt=1.0, efficiency=0.95, soc_init_frac=0.0)
        r_low = solve_daily_lp(prices, dt=1.0, efficiency=0.80, soc_init_frac=0.0)
        assert r_high["revenue_eur"] > r_low["revenue_eur"]

    def test_multi_cycle_pattern(self) -> None:
        """Alternating cheap/expensive blocks should produce multiple cycles."""
        prices = np.array([20.0, 20.0, 80.0, 80.0] * 6)  # 24h
        r = solve_daily_lp(prices, dt=1.0, power_mw=1.0, duration_hours=2.0, efficiency=0.90)
        assert r["n_cycles"] > 1.5  # should find multiple cycles
        assert r["revenue_eur"] > 0

    def test_negative_prices_bounded_revenue(self) -> None:
        """With all negative prices, BESS earns by absorbing excess energy.

        Revenue is bounded by SoC capacity and terminal equality — not infinite.
        Total power per interval never exceeds rated power.
        """
        prices = np.array([-50.0] * 24)
        r = solve_daily_lp(prices, dt=1.0, power_mw=1.0, duration_hours=2.0,
                           efficiency=0.88, soc_init_frac=0.0)
        assert r["revenue_eur"] > 0  # BESS earns from negative-price absorption
        # Total power per interval respects mutual exclusion
        total_power = r["p_charge"] + r["p_discharge"]
        assert total_power.max() <= 1.0 + 1e-6
        # Terminal SoC equals initial
        assert r["soc"][-1] == pytest.approx(r["soc"][0], abs=1e-4)

    def test_negative_then_positive_prices(self) -> None:
        """Negative→positive spread should produce positive revenue."""
        prices = np.array([-50.0] * 12 + [50.0] * 12)
        r = solve_daily_lp(prices, dt=1.0, power_mw=1.0, duration_hours=2.0,
                           efficiency=0.88, soc_init_frac=0.0)
        assert r["revenue_eur"] > 0
        # Should earn more than a purely positive spread of same magnitude
        r_pos = solve_daily_lp(np.array([0.0] * 12 + [100.0] * 12), dt=1.0,
                               power_mw=1.0, duration_hours=2.0, efficiency=0.88,
                               soc_init_frac=0.0)
        assert r["revenue_eur"] > r_pos["revenue_eur"] * 0.5  # substantial revenue

    def test_nan_prices_return_zero(self) -> None:
        """NaN in prices should return zero result, not crash."""
        prices = np.array([20.0, np.nan, 80.0])
        r = solve_daily_lp(prices, dt=1.0)
        assert r["revenue_eur"] == 0.0
        assert r["n_cycles"] == 0.0

    def test_empty_prices(self) -> None:
        """Empty price array should return zero result."""
        r = solve_daily_lp(np.array([]), dt=1.0)
        assert r["revenue_eur"] == 0.0
        assert r["n_cycles"] == 0.0
        assert len(r["p_charge"]) == 0

    def test_sub_hourly_resolution(self) -> None:
        """15-min intervals (dt=0.25) should work correctly."""
        prices = np.array([20.0] * 48 + [80.0] * 48)  # 96 intervals = 24h at 15min
        r = solve_daily_lp(prices, dt=0.25, power_mw=1.0, duration_hours=2.0, efficiency=0.88)
        assert r["revenue_eur"] > 0
        assert len(r["soc"]) == 97  # N+1 SoC points


class TestSolveDispatchBatch:
    @pytest.fixture()
    def multi_day_prices(self) -> pd.DataFrame:
        idx = pd.date_range("2025-01-01", periods=72, freq="h", tz="UTC")
        # 3 days: each day has cheap morning + expensive afternoon
        prices = []
        for _ in range(3):
            prices.extend([20.0] * 12 + [80.0] * 12)
        df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
        df.index.name = "timestamp"
        return df

    def test_returns_dataframe_with_expected_columns(self, multi_day_prices) -> None:
        result = solve_dispatch_batch(multi_day_prices, power_mw=1.0, duration_hours=2.0)
        assert isinstance(result, pd.DataFrame)
        assert "date" in result.columns
        assert "lp_revenue" in result.columns
        assert "n_cycles" in result.columns
        assert "lp_spread_eur_mwh" in result.columns

    def test_one_row_per_day(self, multi_day_prices) -> None:
        result = solve_dispatch_batch(multi_day_prices)
        assert len(result) == 3

    def test_positive_revenue_per_day(self, multi_day_prices) -> None:
        result = solve_dispatch_batch(multi_day_prices, efficiency=0.88)
        assert (result["lp_revenue"] > 0).all()

    def test_timezone_grouping(self) -> None:
        """Local timezone should shift day boundaries."""
        # UTC 23:00 Jan 1 = Berlin 00:00 Jan 2
        idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
        prices = [20.0] * 48
        prices[23] = 200.0  # UTC 23:00
        df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
        df.index.name = "timestamp"

        result_utc = solve_dispatch_batch(df, tz=None)
        result_berlin = solve_dispatch_batch(df, tz="Europe/Berlin")

        # Different day grouping should produce different per-day revenues
        assert len(result_utc) == 2
        assert len(result_berlin) == 3  # Berlin sees 3 local days

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(
            {"price_eur_mwh": pd.Series(dtype=float)},
            index=pd.DatetimeIndex([], name="timestamp", tz="UTC"),
        )
        result = solve_dispatch_batch(df)
        assert len(result) == 0
        assert list(result.columns) == ["date", "lp_revenue", "n_cycles", "lp_spread_eur_mwh"]

    def test_lp_spread_equals_revenue_over_capacity(self, multi_day_prices) -> None:
        """lp_spread_eur_mwh should equal lp_revenue / (power * duration)."""
        result = solve_dispatch_batch(
            multi_day_prices, power_mw=2.0, duration_hours=4.0, efficiency=0.90,
        )
        expected_spread = result["lp_revenue"] / (2.0 * 4.0)
        pd.testing.assert_series_equal(
            result["lp_spread_eur_mwh"],
            expected_spread.round(6),
            check_names=False,
        )
