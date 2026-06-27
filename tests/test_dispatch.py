"""Tests for MILP-based multi-cycle BESS dispatch optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dispatch import (
    solve_daily_da_id_dispatch,
    solve_daily_da_id_reserve_dispatch,
    solve_daily_joint_capacity_lp,
    solve_daily_lp,
    solve_dispatch_batch,
    solve_joint_capacity_batch,
    solve_sequential_da_id_dispatch,
    solve_sequential_da_id_reserve_dispatch,
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

    def test_power_cap_none_and_full_scalar_preserve_old_result(self) -> None:
        prices = np.array([20.0] * 12 + [80.0] * 12)
        kw = dict(dt=1.0, power_mw=1.0, duration_hours=2.0, efficiency=0.88)

        base = solve_daily_lp(prices, **kw)
        full_cap = solve_daily_lp(prices, power_cap_mw=1.0, **kw)

        assert full_cap["revenue_eur"] == pytest.approx(base["revenue_eur"], abs=1e-6)
        assert np.allclose(full_cap["p_charge"], base["p_charge"])
        assert np.allclose(full_cap["p_discharge"], base["p_discharge"])

    def test_power_cap_vector_limits_physical_dispatch(self) -> None:
        prices = np.array([20.0] * 12 + [90.0] * 12)
        cap = np.array([0.25] * 12 + [1.0] * 12)

        r = solve_daily_lp(
            prices, dt=1.0, power_mw=1.0, duration_hours=2.0,
            efficiency=0.88, power_cap_mw=cap,
        )

        assert r["p_charge"][:12].max() <= 0.25 + 1e-6
        assert r["p_discharge"].max() <= 1.0 + 1e-6

    def test_power_cap_length_mismatch_raises(self) -> None:
        prices = np.array([50.0] * 24)
        with pytest.raises(ValueError, match="power_cap_mw"):
            solve_daily_lp(prices, dt=1.0, power_cap_mw=np.ones(10))

    def test_non_finite_or_negative_power_cap_entries_clip_to_zero(self) -> None:
        prices = np.array([20.0] * 12 + [90.0] * 12)
        cap = np.ones(24)
        cap[0] = np.nan
        cap[1] = -5.0

        r = solve_daily_lp(
            prices, dt=1.0, power_mw=1.0, duration_hours=2.0,
            efficiency=0.88, power_cap_mw=cap,
        )

        assert r["p_charge"][0] == pytest.approx(0.0, abs=1e-9)
        assert r["p_discharge"][0] == pytest.approx(0.0, abs=1e-9)
        assert r["p_charge"][1] == pytest.approx(0.0, abs=1e-9)
        assert r["p_discharge"][1] == pytest.approx(0.0, abs=1e-9)


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

    def test_uniform_vector_capacity_price_equals_scalar(self) -> None:
        # Backward compat for the 9.2b per-interval extension: a constant vector
        # must reproduce the scalar result exactly.
        prices = np.array([20.0] * 12 + [120.0] * 12)
        kw = dict(dt=1.0, power_mw=1.0, duration_hours=2.0, efficiency=0.88)
        scalar = solve_daily_joint_capacity_lp(
            prices, capacity_price_eur_mw_h=8.0, **kw,
        )["total_revenue_eur"]
        vector = solve_daily_joint_capacity_lp(
            prices, capacity_price_eur_mw_h=np.full(24, 8.0), **kw,
        )["total_revenue_eur"]
        assert scalar == pytest.approx(vector, abs=1e-6)

    def test_per_interval_capacity_price_concentrates_reserve(self) -> None:
        # Reserve headroom follows the per-interval capacity price.
        prices = np.array([10.0] * 8 + [60.0] * 8 + [10.0] * 8)
        cap = np.array([20.0] * 12 + [0.0] * 12)  # paid only in the first half
        r = solve_daily_joint_capacity_lp(
            prices, dt=1.0, capacity_price_eur_mw_h=cap,
            power_mw=1.0, duration_hours=2.0, efficiency=0.88,
        )
        assert r["reserve_mw"][:12].mean() > r["reserve_mw"][12:].mean()
        assert r["reserve_mw"][12:].mean() == pytest.approx(0.0, abs=1e-6)

    def test_capacity_price_length_mismatch_raises(self) -> None:
        prices = np.array([50.0] * 24)
        with pytest.raises(ValueError, match="scalar or length"):
            solve_daily_joint_capacity_lp(
                prices, dt=1.0, capacity_price_eur_mw_h=np.ones(10),
            )

    @pytest.mark.parametrize("bad_price", [None, object()])
    def test_invalid_capacity_price_type_raises_value_error(self, bad_price) -> None:
        prices = np.array([50.0] * 24)
        with pytest.raises(ValueError, match="capacity_price_eur_mw_h"):
            solve_daily_joint_capacity_lp(
                prices, dt=1.0, capacity_price_eur_mw_h=bad_price,
            )

    def test_non_finite_capacity_price_element_treated_as_zero(self) -> None:
        prices = np.array([10.0] * 8 + [60.0] * 8 + [10.0] * 8)
        cap = np.array([20.0] * 12 + [0.0] * 12, dtype=float)
        cap[0] = np.nan
        r = solve_daily_joint_capacity_lp(
            prices, dt=1.0, capacity_price_eur_mw_h=cap,
            power_mw=1.0, duration_hours=2.0, efficiency=0.88,
        )
        # No crash; the NaN interval simply carries no reserve incentive.
        assert np.isfinite(r["total_revenue_eur"])


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


_RESERVE_KW = {"dt": 1.0, "power_mw": 1.0, "duration_hours": 2.0, "efficiency": 0.88}


def _da_id_reserve_prices() -> tuple[np.ndarray, np.ndarray]:
    da = np.array([10.0] * 8 + [60.0] * 8 + [10.0] * 8)
    ida = np.array([15.0] * 8 + [55.0] * 8 + [12.0] * 8)
    return da, ida


class TestSolveDailyDaIdReserveDispatch:
    """Perfect-foresight DA + IDA + reserve co-optimisation ceiling."""

    def test_collapses_to_da_id_ceiling_at_zero_capacity(self) -> None:
        # cap_price=0 removes the reserve incentive -> exactly the DA+ID ceiling.
        da, ida = _da_id_reserve_prices()
        triple = solve_daily_da_id_reserve_dispatch(
            da, ida, capacity_price_eur_mw_h=0.0, **_RESERVE_KW,
        )["total_cash_eur"]
        ceiling = solve_daily_da_id_dispatch(da, ida, **_RESERVE_KW)["total_cash_eur"]
        assert triple == pytest.approx(ceiling, abs=1e-6)

    def test_collapses_to_da_reserve_when_ida_equals_da(self) -> None:
        # IDA==DA cancels the MtM legs -> exactly the DA+reserve joint at DA.
        da, _ = _da_id_reserve_prices()
        triple = solve_daily_da_id_reserve_dispatch(
            da, da.copy(), capacity_price_eur_mw_h=8.0, **_RESERVE_KW,
        )["total_cash_eur"]
        joint = solve_daily_joint_capacity_lp(
            da, capacity_price_eur_mw_h=8.0, **_RESERVE_KW,
        )["total_revenue_eur"]
        assert triple == pytest.approx(joint, abs=1e-6)

    def test_dominates_da_id_ceiling_and_da_only(self) -> None:
        # Cumulative ladder: triple >= DA+IDA ceiling >= DA-only.
        da, ida = _da_id_reserve_prices()
        triple = solve_daily_da_id_reserve_dispatch(
            da, ida, capacity_price_eur_mw_h=8.0, **_RESERVE_KW,
        )["total_cash_eur"]
        ceiling = solve_daily_da_id_dispatch(da, ida, **_RESERVE_KW)["total_cash_eur"]
        da_only = solve_daily_lp(da, **_RESERVE_KW)["revenue_eur"]
        assert triple >= ceiling - 1e-6
        assert ceiling >= da_only - 1e-6

    def test_decomposition_identity(self) -> None:
        da, ida = _da_id_reserve_prices()
        r = solve_daily_da_id_reserve_dispatch(
            da, ida, capacity_price_eur_mw_h=8.0, **_RESERVE_KW,
        )
        # total = da_gross - implicit_mtm + stage2_total
        assert r["total_cash_eur"] == pytest.approx(
            r["da_gross_eur"] - r["implicit_mtm_eur"] + r["stage2_total_eur"],
            abs=1e-6,
        )
        # stage2_total = ida energy revenue + reserve capacity payment
        assert r["stage2_total_eur"] == pytest.approx(
            r["ida_energy_revenue_eur"] + r["capacity_revenue_eur"], abs=1e-6,
        )

    def test_nan_and_length_mismatch_return_zero(self) -> None:
        da, ida = _da_id_reserve_prices()
        bad = da.copy()
        bad[0] = np.nan
        assert solve_daily_da_id_reserve_dispatch(
            bad, ida, capacity_price_eur_mw_h=8.0, **_RESERVE_KW,
        )["total_cash_eur"] == 0.0
        assert solve_daily_da_id_reserve_dispatch(
            da, ida[:-1], capacity_price_eur_mw_h=8.0, **_RESERVE_KW,
        )["total_cash_eur"] == 0.0
        for bad_capacity_price in (float("nan"), float("inf")):
            assert solve_daily_da_id_reserve_dispatch(
                da, ida, capacity_price_eur_mw_h=bad_capacity_price, **_RESERVE_KW,
            )["total_cash_eur"] == 0.0


class TestSolveSequentialDaIdReserveDispatch:
    """Forecast-driven reserve-first DA + IDA + reserve policy."""

    _DA = np.array([20.0] * 8 + [90.0] * 8 + [25.0] * 8)
    _IDA = np.array([95.0] * 8 + [20.0] * 8 + [88.0] * 8)
    _RESERVE = np.array([4.0] * 8 + [20.0] * 8 + [8.0] * 8)

    def test_stage0_reserve_lock_does_not_peek_at_realised_prices(self) -> None:
        da_forecast = self._DA + 3.0
        ida_forecast = self._IDA - 5.0
        reserve_forecast = self._RESERVE + 2.0

        base = solve_sequential_da_id_reserve_dispatch(
            da_forecast, self._DA, ida_forecast, self._IDA,
            reserve_forecast, self._RESERVE, **_RESERVE_KW,
        )
        perturbed = solve_sequential_da_id_reserve_dispatch(
            da_forecast,
            self._DA[::-1] + 500.0,
            ida_forecast,
            self._IDA[::-1] - 300.0,
            reserve_forecast,
            self._RESERVE[::-1] * 10.0,
            **_RESERVE_KW,
        )

        assert np.allclose(base["reserve_mw"], perturbed["reserve_mw"])

    def test_perfect_forecasts_match_reserve_first_ceiling(self) -> None:
        r = solve_sequential_da_id_reserve_dispatch(
            self._DA, self._DA, self._IDA, self._IDA,
            self._RESERVE, self._RESERVE, **_RESERVE_KW,
        )

        assert r["realised_total_eur"] == pytest.approx(
            r["reserve_first_ceiling_eur"], abs=1e-6,
        )
        assert r["forecast_cost_eur"] == pytest.approx(0.0, abs=1e-6)

    def test_realistic_is_bounded_by_global_ceiling(self) -> None:
        r = solve_sequential_da_id_reserve_dispatch(
            self._DA + 5.0,
            self._DA,
            self._DA,  # deliberately weak IDA forecast
            self._IDA,
            np.array([30.0] * 8 + [1.0] * 16),
            self._RESERVE,
            **_RESERVE_KW,
        )

        assert r["realised_total_eur"] <= r["global_ceiling_eur"] + 1e-6
        assert r["full_gap_eur"] == pytest.approx(
            r["global_ceiling_eur"] - r["realised_total_eur"], abs=1e-6,
        )
        assert r["timing_cost_eur"] >= -1e-6

    def test_zero_reserve_price_collapses_to_phase7_sequential(self) -> None:
        zero = np.zeros_like(self._DA)
        triple = solve_sequential_da_id_reserve_dispatch(
            self._DA, self._DA, self._IDA, self._IDA,
            zero, zero, **_RESERVE_KW,
        )
        phase7 = solve_sequential_da_id_dispatch(
            self._DA, self._IDA, self._IDA, **_RESERVE_KW,
        )

        assert np.allclose(triple["reserve_mw"], 0.0)
        assert triple["capacity_revenue_eur"] == pytest.approx(0.0, abs=1e-6)
        assert triple["realised_total_eur"] == pytest.approx(
            phase7["realised_total_eur"], abs=1e-6,
        )
        assert triple["reserve_first_ceiling_eur"] == pytest.approx(
            phase7["ceiling_total_eur"], abs=1e-6,
        )

    def test_missing_reserve_forecast_safely_skips_reserve_commitment(self) -> None:
        missing = np.full_like(self._DA, np.nan)
        triple = solve_sequential_da_id_reserve_dispatch(
            self._DA, self._DA, self._IDA, self._IDA,
            missing, self._RESERVE, **_RESERVE_KW,
        )
        phase7 = solve_sequential_da_id_dispatch(
            self._DA, self._IDA, self._IDA, **_RESERVE_KW,
        )

        assert np.allclose(triple["reserve_mw"], 0.0)
        assert triple["realised_total_eur"] == pytest.approx(
            phase7["realised_total_eur"], abs=1e-6,
        )

    def test_nan_or_length_mismatch_returns_zero(self) -> None:
        bad = self._DA.copy()
        bad[0] = np.nan
        assert solve_sequential_da_id_reserve_dispatch(
            bad, self._DA, self._IDA, self._IDA,
            self._RESERVE, self._RESERVE, **_RESERVE_KW,
        )["realised_total_eur"] == 0.0
        assert solve_sequential_da_id_reserve_dispatch(
            self._DA, self._DA[:-1], self._IDA, self._IDA,
            self._RESERVE, self._RESERVE, **_RESERVE_KW,
        )["realised_total_eur"] == 0.0
        assert solve_sequential_da_id_reserve_dispatch(
            self._DA, self._DA, self._IDA, self._IDA,
            np.ones(5), self._RESERVE, **_RESERVE_KW,
        )["realised_total_eur"] == 0.0

    def test_none_da_realised_returns_zero_without_len_crash(self) -> None:
        r = solve_sequential_da_id_reserve_dispatch(
            self._DA, None, self._IDA, self._IDA,
            self._RESERVE, self._RESERVE, **_RESERVE_KW,
        )
        assert r["realised_total_eur"] == 0.0
        assert len(r["reserve_mw"]) == 0

    def test_gap_attribution_identity_and_signed_forecast_cost(self) -> None:
        # full_gap == forecast_cost + timing_cost EXACTLY, across many random
        # paths. forecast_cost is signed: the reserve-first sequential policy is
        # not globally optimal even with perfect inputs, so a lucky imperfect
        # forecast can beat its perfect-input benchmark (realised >
        # reserve_first). The previous max(.,0) clamp broke this identity (it
        # could print timing_cost > full_gap).
        rng = np.random.default_rng(0)
        t = np.arange(24)
        saw_negative_forecast_cost = False
        for _ in range(30):
            da_r = 40 + 30 * np.sin(t / 24 * 2 * np.pi) + rng.normal(0, 8, 24)
            ida_r = 40 + 30 * np.sin(t / 24 * 2 * np.pi + 0.6) + rng.normal(0, 12, 24)
            res_r = np.clip(np.tile(rng.uniform(0, 25, 6), 4)[:24], 0, None)
            da_f = da_r + rng.normal(0, 6, 24)
            ida_f = ida_r + rng.normal(0, 10, 24)
            res_f = np.clip(res_r + rng.normal(0, 4, 24), 0, None)
            r = solve_sequential_da_id_reserve_dispatch(
                da_f, da_r, ida_f, ida_r, res_f, res_r, **_RESERVE_KW,
            )
            assert r["full_gap_eur"] == pytest.approx(
                r["forecast_cost_eur"] + r["timing_cost_eur"], abs=1e-4,
            )
            assert r["timing_cost_eur"] >= -1e-6
            assert r["realised_total_eur"] <= r["global_ceiling_eur"] + 1e-6
            if r["forecast_cost_eur"] < -1e-6:
                saw_negative_forecast_cost = True
        assert saw_negative_forecast_cost


class TestSolveSequentialDaIdDispatch:
    """Sequential DA + IDA1 policy under an imperfect IDA forecast."""

    # DA cheap AM / expensive PM; realised IDA is SHAPE-INVERTED so the
    # DA-only schedule is wrong for IDA and a rebid genuinely matters.
    _DA = np.array([20.0] * 8 + [90.0] * 8 + [25.0] * 8)
    _REALISED = np.array([95.0] * 8 + [20.0] * 8 + [88.0] * 8)

    def test_perfect_forecast_matches_ceiling(self) -> None:
        r = solve_sequential_da_id_dispatch(
            self._DA, self._REALISED, self._REALISED,
            dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        assert r["realised_total_eur"] == pytest.approx(
            r["ceiling_total_eur"], abs=1e-6,
        )
        assert r["forecast_error_cost_eur"] == pytest.approx(0.0, abs=1e-6)

    def test_naive_forecast_loses_full_rebid_value(self) -> None:
        # Forecasting IDA == DA gives no rebid signal: the desk just
        # delivers DA, so realised collapses to the DA-only baseline and
        # the entire ceiling uplift becomes forecast error.
        r = solve_sequential_da_id_dispatch(
            self._DA, self._DA, self._REALISED,
            dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        assert r["realised_total_eur"] == pytest.approx(
            r["da_only_revenue_eur"], abs=1e-6,
        )
        assert r["forecast_error_cost_eur"] == pytest.approx(
            r["ceiling_total_eur"] - r["da_only_revenue_eur"], abs=1e-6,
        )

    def test_ordering_da_only_le_realised_le_ceiling(self) -> None:
        forecast = 0.5 * self._DA + 0.5 * self._REALISED
        r = solve_sequential_da_id_dispatch(
            self._DA, forecast, self._REALISED,
            dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        assert r["da_only_revenue_eur"] <= r["realised_total_eur"] + 1e-6
        assert r["realised_total_eur"] <= r["ceiling_total_eur"] + 1e-6
        assert r["forecast_error_cost_eur"] >= -1e-6

    def test_captured_plus_error_equals_full_uplift(self) -> None:
        forecast = 0.5 * self._DA + 0.5 * self._REALISED
        r = solve_sequential_da_id_dispatch(
            self._DA, forecast, self._REALISED,
            dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        full_uplift = r["ceiling_total_eur"] - r["da_only_revenue_eur"]
        assert r["captured_uplift_eur"] + r["forecast_error_cost_eur"] == pytest.approx(
            full_uplift, abs=1e-6,
        )

    def test_wrong_direction_forecast_drives_negative_uplift(self) -> None:
        # A forecast that inverts the DA shape makes the desk rebid the
        # wrong way; settled against a realised print that prints like DA,
        # the rebid destroys value and realised drops BELOW the DA-only
        # baseline. captured_uplift must be allowed to go negative (the UI
        # must not clamp it) — naive ID participation can genuinely lose.
        forecast = np.array([90.0] * 8 + [20.0] * 8 + [88.0] * 8)
        realised = self._DA.copy()
        r = solve_sequential_da_id_dispatch(
            self._DA, forecast, realised,
            dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        assert r["captured_uplift_eur"] < 0.0
        assert r["realised_total_eur"] < r["da_only_revenue_eur"]
        # Ceiling is still a valid upper bound and the decomposition holds.
        assert r["forecast_error_cost_eur"] >= -1e-6
        full_uplift = r["ceiling_total_eur"] - r["da_only_revenue_eur"]
        assert r["captured_uplift_eur"] + r["forecast_error_cost_eur"] == pytest.approx(
            full_uplift, abs=1e-6,
        )

    def test_default_threshold_is_backward_compatible_always_rebid(self) -> None:
        # forecast_uplift is non-negative by construction, so the default 0.0
        # gate always rebids — same policy as before the deadband existed.
        forecast = 0.5 * self._DA + 0.5 * self._REALISED
        r = solve_sequential_da_id_dispatch(
            self._DA, forecast, self._REALISED,
            dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        # Floored at 0.0 — the >= 0 contract is strict, not within tolerance.
        assert r["forecast_uplift_eur"] >= 0.0
        assert r["rebid"] is True

    def test_deadband_holds_da_schedule_and_neutralises_churn_loss(self) -> None:
        # A wrong-direction forecast loses money when followed unconditionally.
        forecast = np.array([90.0] * 8 + [20.0] * 8 + [88.0] * 8)
        realised = self._DA.copy()
        ungated = solve_sequential_da_id_dispatch(
            self._DA, forecast, realised,
            dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        assert ungated["rebid"] is True
        assert ungated["captured_uplift_eur"] < 0.0  # churn loss

        # A deadband just above the forecast-predicted uplift makes the desk
        # hold its committed DA schedule, turning the realised loss into
        # break-even (captured == 0, realised == da_only).
        threshold = ungated["forecast_uplift_eur"] + 1.0
        gated = solve_sequential_da_id_dispatch(
            self._DA, forecast, realised,
            dt=1.0, power_mw=1.0, duration_hours=2.0,
            min_rebid_uplift_eur=threshold,
        )
        assert gated["rebid"] is False
        assert gated["captured_uplift_eur"] == pytest.approx(0.0, abs=1e-6)
        assert gated["realised_total_eur"] == pytest.approx(
            gated["da_only_revenue_eur"], abs=1e-6,
        )
        assert gated["realised_total_eur"] > ungated["realised_total_eur"]

    def test_nan_or_length_mismatch_returns_zero(self) -> None:
        bad = np.array([20.0, np.nan, 80.0])
        r = solve_sequential_da_id_dispatch(bad, bad.copy(), bad.copy(), dt=1.0)
        assert r["realised_total_eur"] == 0.0
        short = solve_sequential_da_id_dispatch(
            self._DA, self._DA, self._REALISED[:-1], dt=1.0,
        )
        assert short["ceiling_total_eur"] == 0.0


class TestMixedResolutionDtInference:
    """Mixed-resolution windows (DE_LU 2025-10 60min→15min switch).

    Before the per-day fix, frame-global dt inference produced mode=0.25h
    and the hourly side was solved as if each calendar day were 6 physical
    hours, inflating revenue ~4x. These tests pin the per-day behaviour.
    """

    def _build_mixed_frame(self) -> pd.DataFrame:
        """Day 1 (hourly) + Day 2 (15-min). Same physical shape: 12h cheap,
        12h expensive. With per-day dt both days must produce comparable
        revenue scaled by their identical 24-hour wall-clock duration.
        """
        day1_idx = pd.date_range("2025-10-04", periods=24, freq="h", tz="UTC")
        day1_prices = [20.0] * 12 + [80.0] * 12

        day2_idx = pd.date_range("2025-10-05", periods=96, freq="15min", tz="UTC")
        day2_prices = [20.0] * 48 + [80.0] * 48

        idx = day1_idx.append(day2_idx)
        prices = day1_prices + day2_prices
        df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
        df.index.name = "timestamp"
        return df

    def test_dispatch_batch_solves_each_day_at_native_cadence(self) -> None:
        df = self._build_mixed_frame()
        result = solve_dispatch_batch(
            df, power_mw=1.0, duration_hours=2.0, efficiency=1.0,
        )

        assert len(result) == 2
        rev_day1 = float(result.iloc[0]["lp_revenue"])
        rev_day2 = float(result.iloc[1]["lp_revenue"])

        # Same physical shape and wall clock → both days within 10% of each
        # other. The old global-dt code reported ~4x divergence.
        assert rev_day1 > 0 and rev_day2 > 0
        ratio = max(rev_day1, rev_day2) / min(rev_day1, rev_day2)
        assert ratio < 1.10, (
            f"per-day dt regression: day1={rev_day1:.2f} day2={rev_day2:.2f}"
        )

    def test_joint_capacity_batch_per_day_dt(self) -> None:
        df = self._build_mixed_frame()
        result = solve_joint_capacity_batch(
            df, capacity_price_eur_mw_h=5.0,
            power_mw=1.0, duration_hours=2.0,
        )

        assert len(result) == 2
        # Capacity revenue is dt-linear: 24 hours * 5 EUR/MW/h * availability
        # * commitment. With per-day dt, both days should report nearly
        # identical capacity revenue (within a small MILP tolerance).
        cap_day1 = float(result.iloc[0]["joint_capacity_revenue"])
        cap_day2 = float(result.iloc[1]["joint_capacity_revenue"])
        assert cap_day1 > 0 and cap_day2 > 0
        ratio = max(cap_day1, cap_day2) / min(cap_day1, cap_day2)
        assert ratio < 1.05
