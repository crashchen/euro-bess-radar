"""Tests for MILP-based multi-cycle BESS dispatch optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dispatch import (
    solve_daily_da_id_dispatch,
    solve_daily_joint_capacity_lp,
    solve_daily_lp,
    solve_dispatch_batch,
    solve_joint_capacity_batch,
)


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
        assert r["revenue_eur"] >= 0
        # Total power per interval respects mutual exclusion
        total_power = r["p_charge"] + r["p_discharge"]
        assert total_power.max() <= 1.0 + 1e-6
        # Terminal SoC equals initial
        assert r["soc"][-1] == pytest.approx(r["soc"][0], abs=1e-4)
        # Cycles are bounded by physical SoC capacity, not by within-interval wash energy.
        # 1MW/2MWh battery, 24h, terminal-neutral → at most ~6 partial cycles even
        # in pathological cases. With proper mutual exclusion it's usually 0.
        assert r["n_cycles"] <= 6.0

    def test_no_simultaneous_charge_and_discharge_negative_prices(self) -> None:
        """Charging and discharging in the same interval is physically impossible
        and energy-destructive. Under all-negative prices the LP relaxation can
        exploit this to dump grid energy through the round-trip loss; enforce
        strict mutual exclusion via MILP.
        """
        prices = np.array([-50.0] * 24)
        r = solve_daily_lp(prices, dt=1.0, power_mw=1.0, duration_hours=2.0,
                           efficiency=0.88, soc_init_frac=0.0)
        both = (r["p_charge"] > 1e-6) & (r["p_discharge"] > 1e-6)
        assert int(both.sum()) == 0, (
            f"LP allowed {int(both.sum())} intervals with simultaneous "
            f"charge+discharge — should be 0"
        )

    def test_no_simultaneous_charge_and_discharge_normal_prices(self) -> None:
        """Even on normal arbitrage patterns LP degeneracy can pick simultaneous
        flow as one of multiple optima. Mutual exclusion must hold strictly.
        """
        prices = np.array([20.0] * 6 + [80.0] * 6 + [20.0] * 6 + [80.0] * 6)
        r = solve_daily_lp(prices, dt=1.0, power_mw=1.0, duration_hours=2.0,
                           efficiency=0.88, soc_init_frac=0.5)
        both = (r["p_charge"] > 1e-6) & (r["p_discharge"] > 1e-6)
        assert int(both.sum()) == 0

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

    def test_nan_days_are_excluded_not_zeroed(self, multi_day_prices) -> None:
        multi_day_prices = multi_day_prices.copy()
        multi_day_prices.loc[multi_day_prices.index[30:34], "price_eur_mwh"] = np.nan

        result = solve_dispatch_batch(multi_day_prices)

        assert len(result) == 2
        assert result.attrs["excluded_days_due_to_missing"] == 1
        assert (result["lp_revenue"] > 0).all()

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


class TestSolveDailyJointCapacityLp:
    def test_flat_prices_commit_capacity_when_capacity_price_positive(self) -> None:
        prices = np.array([50.0] * 24)
        result = solve_daily_joint_capacity_lp(
            prices,
            dt=1.0,
            capacity_price_eur_mw_h=10.0,
            power_mw=2.0,
            duration_hours=2.0,
            availability=0.95,
        )
        assert result["da_revenue_eur"] == pytest.approx(0.0, abs=1e-6)
        assert result["capacity_revenue_eur"] == pytest.approx(10.0 * 2.0 * 24 * 0.95)
        assert result["avg_reserve_mw"] == pytest.approx(2.0)

    def test_capacity_competes_with_dispatch_power_headroom(self) -> None:
        prices = np.array([20.0] * 12 + [120.0] * 12)
        da_only = solve_daily_lp(
            prices,
            dt=1.0,
            power_mw=1.0,
            duration_hours=2.0,
            efficiency=0.88,
            soc_init_frac=0.0,
        )
        joint = solve_daily_joint_capacity_lp(
            prices,
            dt=1.0,
            capacity_price_eur_mw_h=1.0,
            power_mw=1.0,
            duration_hours=2.0,
            efficiency=0.88,
            soc_init_frac=0.0,
        )
        assert joint["total_revenue_eur"] >= da_only["revenue_eur"]
        assert joint["avg_reserve_mw"] < 1.0

    def test_high_capacity_price_suppresses_cycling(self) -> None:
        prices = np.array([20.0] * 12 + [120.0] * 12)
        result = solve_daily_joint_capacity_lp(
            prices,
            dt=1.0,
            capacity_price_eur_mw_h=200.0,
            power_mw=1.0,
            duration_hours=2.0,
            efficiency=0.88,
            soc_init_frac=0.0,
        )
        assert result["avg_reserve_mw"] == pytest.approx(1.0)
        assert result["n_cycles"] == pytest.approx(0.0)


    def test_no_simultaneous_charge_and_discharge_under_capacity(self) -> None:
        """Joint MILP must also enforce strict charge/discharge mutual exclusion."""
        prices = np.array([-30.0] * 12 + [40.0] * 12)
        r = solve_daily_joint_capacity_lp(
            prices, dt=1.0, capacity_price_eur_mw_h=5.0,
            power_mw=1.0, duration_hours=2.0, efficiency=0.88, soc_init_frac=0.0,
        )
        both = (r["p_charge"] > 1e-6) & (r["p_discharge"] > 1e-6)
        assert int(both.sum()) == 0


class TestSolveJointCapacityBatch:
    def test_batch_returns_expected_columns(self) -> None:
        idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame({"price_eur_mwh": [50.0] * 48}, index=idx)
        df.index.name = "timestamp"
        result = solve_joint_capacity_batch(
            df,
            capacity_price_eur_mw_h=5.0,
            power_mw=1.0,
            duration_hours=2.0,
        )
        assert len(result) == 2
        expected = {
            "date", "joint_total_revenue", "joint_da_revenue",
            "joint_capacity_revenue", "avg_reserve_mw", "reserve_fraction",
            "n_cycles",
        }
        assert set(result.columns) == expected
        assert (result["joint_capacity_revenue"] > 0).all()

    def test_nan_days_are_excluded(self) -> None:
        idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame({"price_eur_mwh": [50.0] * 48}, index=idx)
        df.loc[idx[4:8], "price_eur_mwh"] = np.nan
        df.index.name = "timestamp"

        result = solve_joint_capacity_batch(
            df,
            capacity_price_eur_mw_h=5.0,
            power_mw=1.0,
            duration_hours=2.0,
        )

        assert len(result) == 1
        assert result.attrs["excluded_days_due_to_missing"] == 1


class TestSolveDailyDaIdDispatch:
    def test_ida_equal_to_da_yields_zero_uplift(self) -> None:
        """When IDA prints the same as DA, the optimal Stage-2 solution is
        the Stage-1 schedule itself, so rebid uplift = 0.
        """
        prices = np.array([20.0] * 12 + [80.0] * 12)
        r = solve_daily_da_id_dispatch(
            prices, prices.copy(), dt=1.0, power_mw=1.0,
            duration_hours=2.0, efficiency=0.88, soc_init_frac=0.0,
        )
        assert r["rebid_uplift_eur"] == pytest.approx(0.0, abs=1.0)
        # Total cash equals DA revenue when nothing changes.
        assert r["total_cash_eur"] == pytest.approx(r["da_revenue_eur"], abs=1.0)

    def test_ida_flipped_unlocks_cancellation_arb(self) -> None:
        """If IDA prices flip relative to DA, the DA position becomes
        unfavourable; cancelling at IDA captures cancellation arbitrage.
        Total cash should significantly exceed DA-only.
        """
        da_prices = np.array([20.0] * 12 + [80.0] * 12)
        ida_prices = np.array([80.0] * 12 + [20.0] * 12)
        r = solve_daily_da_id_dispatch(
            da_prices, ida_prices, dt=1.0, power_mw=1.0,
            duration_hours=2.0, efficiency=0.88, soc_init_frac=0.0,
        )
        assert r["rebid_uplift_eur"] > 50.0  # at least ~50 EUR captured
        assert r["total_cash_eur"] > r["da_revenue_eur"] + 50.0

    def test_uplift_is_non_negative(self) -> None:
        """Stage-1 dispatch is always a feasible Stage-2 solution at any
        IDA prices, so the optimal uplift cannot drop below zero —
        guaranteed by both the math and our explicit max(.,0) clamp.
        """
        da_prices = np.array([20.0] * 12 + [80.0] * 12)
        for ida_offset in [-30.0, -10.0, 0.0, 10.0, 30.0]:
            ida = da_prices + ida_offset
            r = solve_daily_da_id_dispatch(
                da_prices, ida, dt=1.0, power_mw=1.0,
                duration_hours=2.0, efficiency=0.88, soc_init_frac=0.0,
            )
            assert r["rebid_uplift_eur"] >= -1e-6

    def test_mismatched_lengths_returns_zero(self) -> None:
        r = solve_daily_da_id_dispatch(
            np.array([50.0] * 24),
            np.array([60.0] * 23),
            dt=1.0,
        )
        assert r["da_revenue_eur"] == 0.0
        assert r["rebid_uplift_eur"] == 0.0

    def test_nan_input_returns_zero(self) -> None:
        prices = np.array([20.0, np.nan, 80.0])
        r = solve_daily_da_id_dispatch(prices, prices.copy(), dt=1.0)
        assert r["rebid_uplift_eur"] == 0.0

    def test_no_vom_double_counting_on_full_cancellation(self) -> None:
        """If IDA exactly inverts DA prices and the battery starts empty so
        Stage 2 can't physically dispatch, the optimal strategy is to cancel
        all DA flow via IDA rebid (no physical dispatch -> no VOM).
        Codex repro: 1MW/1h, eff=1, DA [10,10,80,80], IDA [80,80,10,10]
        -> total_cash should be 140 (gross DA + IDA cancellation arb, no VOM).
        """
        da = np.array([10.0, 10.0, 80.0, 80.0])
        ida = np.array([80.0, 80.0, 10.0, 10.0])
        r = solve_daily_da_id_dispatch(
            da, ida, dt=1.0, power_mw=1.0, duration_hours=1.0,
            efficiency=1.0, soc_init_frac=0.0,
        )
        # Expect exactly 140 (within solver/rounding tolerance).
        assert r["total_cash_eur"] == pytest.approx(140.0, abs=0.5)

    def test_equal_prices_total_cash_equals_da_revenue(self) -> None:
        """When IDA = DA, the optimal Stage 2 is the Stage 1 dispatch
        unchanged; rebid_uplift = 0 and total_cash = da_revenue. Used to
        verify VOM is not double-counted on equal-price inputs.
        """
        da = np.array([10.0, 10.0, 80.0, 80.0])
        r = solve_daily_da_id_dispatch(
            da, da.copy(), dt=1.0, power_mw=1.0, duration_hours=1.0,
            efficiency=1.0, soc_init_frac=0.0,
        )
        assert r["rebid_uplift_eur"] == pytest.approx(0.0, abs=0.01)
        assert r["total_cash_eur"] == pytest.approx(r["da_revenue_eur"], abs=0.01)

    def test_total_cash_equals_da_plus_uplift(self) -> None:
        da_prices = np.array([10.0] * 8 + [60.0] * 8 + [10.0] * 8)
        ida_prices = np.array([15.0] * 8 + [55.0] * 8 + [12.0] * 8)
        r = solve_daily_da_id_dispatch(
            da_prices, ida_prices, dt=1.0, power_mw=1.0,
            duration_hours=2.0, efficiency=0.88, soc_init_frac=0.0,
        )
        assert r["total_cash_eur"] == pytest.approx(
            r["da_revenue_eur"] + r["rebid_uplift_eur"], abs=0.01,
        )
