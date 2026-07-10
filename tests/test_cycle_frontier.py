"""Tests for the cycle-cap x degradation net-revenue frontier (F-B).

Pins the contract identities of docs/design/cycle-cap-frontier-v1.md
sections 6-4..8 and 6-10, plus cap-set normalisation and degenerate
handling.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

import src.cycle_frontier as cf
from src.analytics import calculate_dispatch_price_vwaps
from src.cycle_frontier import (
    DEFAULT_CYCLE_CAPS,
    FRONTIER_COLUMNS,
    UNCAPPED_LABEL,
    _normalize_cycle_caps,
    compute_cycle_cap_frontier,
)
from src.degradation import calculate_degradation_cost, estimate_battery_lifetime
from src.dispatch import solve_daily_lp


def _frame(prices: list[float], start: str = "2026-03-02") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq="h", tz="UTC")
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    return df


def _double_cycle_day() -> list[float]:
    """Two valley->peak pairs: uncapped profitably runs ~2 EFC."""
    return [10.0] * 6 + [100.0] * 3 + [10.0] * 6 + [100.0] * 3 + [50.0] * 6


def _single_cycle_day() -> list[float]:
    """One valley->peak pair: extra cycle headroom is worthless."""
    return [10.0] * 8 + [100.0] * 8 + [55.0] * 8


def _thin_second_cycle_day() -> list[float]:
    """Big first spread plus a thin-but-positive second spread."""
    return [10.0] * 6 + [100.0] * 3 + [10.0] * 6 + [20.0] * 3 + [15.0] * 6


_KW = dict(power_mw=1.0, duration_hours=1.0, efficiency=0.9)


class TestCapNormalisation:
    def test_orders_ascending_finite_then_uncapped_last(self) -> None:
        assert _normalize_cycle_caps((None, 2.0, 0.5, 1.0)) == [0.5, 1.0, 2.0, None]

    def test_dedupes_finite_and_uncapped(self) -> None:
        assert _normalize_cycle_caps((1.0, 1.0, None, None)) == [1.0, None]

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            _normalize_cycle_caps((-0.5,))

    @pytest.mark.parametrize("bad", [math.inf, math.nan])
    def test_non_finite_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            _normalize_cycle_caps((bad,))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _normalize_cycle_caps(())

    def test_default_caps_bracket_the_debate_points(self) -> None:
        assert DEFAULT_CYCLE_CAPS == (0.5, 1.0, 1.2, 1.5, 2.0, 3.0, None)


class TestFrontierIdentities:
    def test_gross_monotone_non_decreasing_in_cap(self) -> None:
        """Pin section 6-4: gross never decreases as the cap loosens."""
        frame, _summary = compute_cycle_cap_frontier(
            _frame(_double_cycle_day()),
            capex_eur_kwh=200.0,
            cycle_caps=(0.5, 1.0, 1.5, 2.0, None),
            **_KW,
        )
        gross = frame["gross_eur"].to_numpy()
        assert np.all(np.diff(gross) >= -1e-9)
        # The cap binds on this fixture: uncapped strictly beats cap=0.5.
        assert gross[-1] > gross[0] + 1.0

    def test_net_equals_gross_minus_wear_via_degradation_formula(self) -> None:
        """Pin section 6-5: one wear formula, netted exactly."""
        frame, summary = compute_cycle_cap_frontier(
            _frame(_double_cycle_day()),
            capex_eur_kwh=250.0,
            cycle_life=5000.0,
            cycle_caps=(1.0, None),
            **_KW,
        )
        for _, row in frame.iterrows():
            fec_total = row["avg_efc_per_day"] * summary["valid_days"]
            expected = calculate_degradation_cost(
                fec_total, 250.0, 1000.0, 5000.0
            )["total_degradation_eur"]
            assert row["wear_eur"] == pytest.approx(expected, abs=1e-9)
            assert row["net_eur"] == pytest.approx(
                row["gross_eur"] - row["wear_eur"], abs=1e-12
            )
        assert summary["cost_per_cycle_eur"] == pytest.approx(250.0 * 1000.0 / 5000.0)
        assert summary["wear_eur_per_mwh_discharged"] == pytest.approx(
            summary["cost_per_cycle_eur"] / 1.0
        )

    def test_realised_efc_within_cap_per_row(self) -> None:
        """Pin section 6-3 at row level: avg EFC <= cap + solver tolerance."""
        frame, _ = compute_cycle_cap_frontier(
            _frame(_double_cycle_day()),
            capex_eur_kwh=0.0,
            cycle_caps=(0.5, 1.0, 1.5, None),
            **_KW,
        )
        finite = frame[frame["label"] != UNCAPPED_LABEL]
        assert np.all(
            finite["avg_efc_per_day"].to_numpy()
            <= finite["cycle_cap"].to_numpy() + 1e-6
        )

    def test_uncapped_row_equals_slack_cap_row(self) -> None:
        frame, _ = compute_cycle_cap_frontier(
            _frame(_double_cycle_day()),
            capex_eur_kwh=100.0,
            cycle_caps=(24.0, None),
            **_KW,
        )
        slack, uncapped = frame.iloc[0], frame.iloc[1]
        assert slack["gross_eur"] == pytest.approx(uncapped["gross_eur"], abs=1e-9)
        assert slack["avg_efc_per_day"] == pytest.approx(
            uncapped["avg_efc_per_day"], abs=1e-9
        )
        assert slack["net_eur"] == pytest.approx(uncapped["net_eur"], abs=1e-9)

    def test_cap_zero_row_zero_gross_zero_wear(self) -> None:
        frame, _ = compute_cycle_cap_frontier(
            _frame(_double_cycle_day()),
            capex_eur_kwh=100.0,
            cycle_caps=(0.0, None),
            **_KW,
        )
        zero_row = frame.iloc[0]
        assert zero_row["gross_eur"] == pytest.approx(0.0, abs=1e-9)
        assert zero_row["wear_eur"] == pytest.approx(0.0, abs=1e-9)
        assert zero_row["avg_efc_per_day"] == pytest.approx(0.0, abs=1e-9)
        assert math.isinf(zero_row["cycle_limited_life_years"])

    def test_capex_zero_net_equals_gross_and_lowest_tying_cap_wins(self) -> None:
        """Pin section 6-7: with no wear, best cap = lowest cap tying max net."""
        frame, summary = compute_cycle_cap_frontier(
            _frame(_single_cycle_day()),
            capex_eur_kwh=0.0,
            cycle_caps=(0.5, 1.0, 2.0, None),
            **_KW,
        )
        pd.testing.assert_series_equal(
            frame["net_eur"], frame["gross_eur"], check_names=False
        )
        # Extra headroom is worthless on a single-spread day, so cap=1.0
        # ties 2.0 and uncapped; the LOWEST tying cap wins.
        assert summary["best_cap_label"] == "1 EFC/day"

    def test_uncapped_wins_only_on_a_strict_win(self) -> None:
        """Pin section 6-7 second half: strict uncapped win over every finite cap."""
        frame, summary = compute_cycle_cap_frontier(
            _frame(_double_cycle_day()),
            capex_eur_kwh=0.0,
            cycle_caps=(0.5, 1.0, None),
            **_KW,
        )
        uncapped_net = frame.loc[frame["label"] == UNCAPPED_LABEL, "net_eur"].iloc[0]
        assert (uncapped_net > frame["net_eur"].iloc[:-1] + 1.0).all()
        assert summary["best_cap_label"] == UNCAPPED_LABEL

    def test_frontier_flat_on_flat_prices(self) -> None:
        """Pin section 6-8: no spread => every cap nets ~0 => flat flag."""
        frame, summary = compute_cycle_cap_frontier(
            _frame([50.0] * 24),
            capex_eur_kwh=300.0,
            cycle_caps=(0.5, 1.0, None),
            **_KW,
        )
        assert summary["frontier_flat"] is True
        assert frame["gross_eur"].abs().max() <= 1e-6
        # A flat frontier still resolves best-cap to the lowest finite cap
        # (prefer committing less wear headroom for equal money).
        assert summary["best_cap_label"] == "0.5 EFC/day"
        # Uplift vs an ~zero uncapped net is undefined, not a blow-up.
        assert frame["net_uplift_vs_uncapped_pct"].isna().all()
        assert frame["net_delta_vs_uncapped_eur"].abs().max() <= 1e-6

    def test_frontier_not_flat_when_cap_binds(self) -> None:
        _, summary = compute_cycle_cap_frontier(
            _frame(_double_cycle_day()),
            capex_eur_kwh=0.0,
            cycle_caps=(0.5, None),
            **_KW,
        )
        assert summary["frontier_flat"] is False

    def test_uplift_denominator_is_absolute_at_negative_uncapped_net(self) -> None:
        """Contract r3: capping away a thin cycle IMPROVES a negative net,
        and the absolute denominator keeps that uplift positive."""
        frame, summary = compute_cycle_cap_frontier(
            _frame(_thin_second_cycle_day()),
            power_mw=1.0,
            duration_hours=1.0,
            efficiency=1.0,
            capex_eur_kwh=600.0,  # cost_per_cycle = 100 EUR > thin-cycle margin
            cycle_caps=(1.0, None),
        )
        capped, uncapped = frame.iloc[0], frame.iloc[1]
        # The wear-blind optimiser still runs the thin second cycle...
        assert uncapped["avg_efc_per_day"] > capped["avg_efc_per_day"] + 0.5
        # ...so at this capex both nets are negative and the cap helps.
        assert uncapped["net_eur"] < 0
        assert capped["net_eur"] > uncapped["net_eur"]
        assert capped["net_delta_vs_uncapped_eur"] > 0
        assert capped["net_uplift_vs_uncapped_pct"] > 0
        assert summary["best_cap_label"] == "1 EFC/day"

    def test_cycle_limited_life_years_matches_lifetime_helper(self) -> None:
        frame, _summary = compute_cycle_cap_frontier(
            _frame(_double_cycle_day()),
            capex_eur_kwh=100.0,
            cycle_life=4000.0,
            cycle_caps=(1.0,),
            **_KW,
        )
        row = frame.iloc[0]
        expected = estimate_battery_lifetime(row["avg_efc_per_day"], 4000.0)
        assert row["cycle_limited_life_years"] == pytest.approx(
            expected["cycle_limited_years"]
        )

    def test_reports_realised_charge_and_discharge_vwaps(self) -> None:
        prices = _double_cycle_day()
        frame, _summary = compute_cycle_cap_frontier(
            _frame(prices),
            capex_eur_kwh=0.0,
            cycle_caps=(1.0,),
            **_KW,
        )
        result = solve_daily_lp(
            np.asarray(prices), dt=1.0, max_efc_per_day=1.0,
            min_throughput_tiebreak=True, **_KW,
        )
        expected = calculate_dispatch_price_vwaps(
            np.asarray(prices), result["p_charge"], result["p_discharge"],
            dt_hours=1.0,
        )
        row = frame.iloc[0]
        assert row["charge_vwap_eur_mwh"] == pytest.approx(
            expected["charge_vwap_eur_mwh"], abs=1e-9
        )
        assert row["discharge_vwap_eur_mwh"] == pytest.approx(
            expected["discharge_vwap_eur_mwh"], abs=1e-9
        )

    def test_vwaps_are_energy_weighted_across_days(self) -> None:
        """Frontier VWAP is total value / total energy, not a mean of days."""
        day_one = _double_cycle_day()  # Two profitable cycles at 10 -> 100.
        day_two = [20.0] * 8 + [80.0] * 8 + [55.0] * 8  # One at 20 -> 80.
        frame, _summary = compute_cycle_cap_frontier(
            _frame(day_one + day_two),
            capex_eur_kwh=0.0,
            cycle_caps=(2.0,),
            **_KW,
        )

        expected_charge_energy = 0.0
        expected_charge_value = 0.0
        expected_discharge_energy = 0.0
        expected_discharge_value = 0.0
        for prices in (day_one, day_two):
            result = solve_daily_lp(
                np.asarray(prices),
                dt=1.0,
                max_efc_per_day=2.0,
                min_throughput_tiebreak=True,
                **_KW,
            )
            vwap = calculate_dispatch_price_vwaps(
                np.asarray(prices),
                result["p_charge"],
                result["p_discharge"],
                dt_hours=1.0,
            )
            expected_charge_energy += vwap["charge_energy_mwh"]
            expected_charge_value += (
                vwap["charge_vwap_eur_mwh"] * vwap["charge_energy_mwh"]
            )
            expected_discharge_energy += vwap["discharge_energy_mwh"]
            expected_discharge_value += (
                vwap["discharge_vwap_eur_mwh"] * vwap["discharge_energy_mwh"]
            )

        row = frame.iloc[0]
        assert row["charge_vwap_eur_mwh"] == pytest.approx(
            expected_charge_value / expected_charge_energy, abs=1e-9
        )
        assert row["discharge_vwap_eur_mwh"] == pytest.approx(
            expected_discharge_value / expected_discharge_energy, abs=1e-9
        )

    def test_idle_cap_reports_missing_vwaps(self) -> None:
        frame, _summary = compute_cycle_cap_frontier(
            _frame(_double_cycle_day()),
            capex_eur_kwh=0.0,
            cycle_caps=(0.0,),
            **_KW,
        )
        assert math.isnan(frame["charge_vwap_eur_mwh"].iloc[0])
        assert math.isnan(frame["discharge_vwap_eur_mwh"].iloc[0])


class TestCoTemporalValidDays:
    def test_nan_day_excluded_for_all_caps(self) -> None:
        """Pin section 6-6 (data reason): one dirty day drops everywhere."""
        good = _double_cycle_day()
        prices = good + good + good
        df = _frame(prices)
        df.loc[df.index[30], "price_eur_mwh"] = np.nan  # dirties day 2 only
        frame, summary = compute_cycle_cap_frontier(
            df, capex_eur_kwh=100.0, cycle_caps=(1.0, None), **_KW
        )
        assert summary["valid_days"] == 2
        assert summary["excluded_days"] == 1
        # Both rows aggregate the same two days: uncapped gross == 2x one day.
        one_day, _ = compute_cycle_cap_frontier(
            _frame(good), capex_eur_kwh=100.0, cycle_caps=(1.0, None), **_KW
        )
        assert frame["gross_eur"].iloc[1] == pytest.approx(
            2.0 * one_day["gross_eur"].iloc[1], rel=1e-9
        )

    def test_solver_failure_on_one_cap_excludes_day_for_all_caps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pin section 6-6 (solver reason): success=False on ANY cap drops
        the day from EVERY row — the failed day's revenue/FEC must not leak
        into rows whose own solve succeeded (Codex hardening: fail a MIDDLE
        cap on a HIGH-revenue day, assert every row equals the day-1-only
        aggregate, not just the summary counts)."""
        # Day 2 doubles the peaks: a leak of its cap=0.5/uncapped solves
        # would visibly inflate those rows.
        day2 = [p * 2.0 if p >= 100.0 else p for p in _double_cycle_day()]
        day2[0] = 11.0  # marker price
        df = _frame(_double_cycle_day() + day2)

        def fake_solve(prices, dt, **kwargs):
            result = solve_daily_lp(prices, dt, **kwargs)
            # Fail only day 2's MIDDLE cap solve (marker price identifies it).
            if kwargs.get("max_efc_per_day") == 1.0 and prices[0] == 11.0:
                return {**result, "success": False}
            return result

        monkeypatch.setattr(cf, "solve_daily_lp", fake_solve)
        caps = (0.5, 1.0, None)
        frame_out, summary = compute_cycle_cap_frontier(
            df, capex_eur_kwh=100.0, cycle_caps=caps, **_KW
        )
        assert summary["valid_days"] == 1
        assert summary["excluded_days"] == 1
        # Row-level equality with a day-1-only run (the wrapper delegates to
        # the real solver on day 1, so this is the true reference).
        day1_only, _ = compute_cycle_cap_frontier(
            _frame(_double_cycle_day()), capex_eur_kwh=100.0, cycle_caps=caps, **_KW
        )
        for col in ("gross_eur", "avg_efc_per_day", "wear_eur", "net_eur"):
            np.testing.assert_allclose(
                frame_out[col].to_numpy(), day1_only[col].to_numpy(), rtol=1e-9
            )

    def test_dates_none_uses_all_available_local_days(self) -> None:
        df = _frame(_double_cycle_day() + _double_cycle_day())
        _, summary = compute_cycle_cap_frontier(
            df, capex_eur_kwh=100.0, cycle_caps=(1.0,), **_KW
        )
        assert summary["valid_days"] == 2

    def test_explicit_dates_subset_window(self) -> None:
        df = _frame(_double_cycle_day() + _double_cycle_day())
        _, summary = compute_cycle_cap_frontier(
            df,
            dates=[date(2026, 3, 2)],
            capex_eur_kwh=100.0,
            cycle_caps=(1.0,),
            **_KW,
        )
        assert summary["valid_days"] == 1
        assert summary["excluded_days"] == 0


class TestSolverWiring:
    def test_frontier_always_requests_min_throughput_tiebreak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[bool] = []

        def spy(prices, dt, **kwargs):
            seen.append(kwargs.get("min_throughput_tiebreak"))
            return solve_daily_lp(prices, dt, **kwargs)

        monkeypatch.setattr(cf, "solve_daily_lp", spy)
        compute_cycle_cap_frontier(
            _frame(_double_cycle_day()),
            capex_eur_kwh=100.0,
            cycle_caps=(1.0, None),
            **_KW,
        )
        assert seen and all(flag is True for flag in seen)

    def test_tiebreak_fallback_days_counted_once_per_day(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#43 pattern: a pass-2 fallback is visible, counted per DAY."""

        def fake_solve(prices, dt, **kwargs):
            result = solve_daily_lp(prices, dt, **kwargs)
            # Report fallback on every solve of day 1 (both caps): still 1 day.
            if prices[0] == 10.0:
                return {**result, "tiebreak_applied": False}
            return result

        day2 = list(_double_cycle_day())
        day2[0] = 12.0
        df = _frame(_double_cycle_day() + day2)
        monkeypatch.setattr(cf, "solve_daily_lp", fake_solve)
        _, summary = compute_cycle_cap_frontier(
            df, capex_eur_kwh=100.0, cycle_caps=(1.0, None), **_KW
        )
        assert summary["valid_days"] == 2
        assert summary["n_tiebreak_fallback_days"] == 1

    def test_wear_uses_raw_schedule_fec_not_rounded_n_cycles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pin section 6-10: wear = sum(p_discharge)*dt/capacity * cost.

        Non-unit dt (30-min data) and non-unit capacity (2 MW x 2 h) so a
        mutation dropping the dt factor or the capacity normalisation moves
        the number (Codex hardening), and the raw FEC (4e-5) rounds to 0.0
        in the `n_cycles` convenience field, so using the rounded field
        would zero the wear entirely.
        """
        n = 48  # 30-min intervals -> dt = 0.5 h
        p_discharge = np.zeros(n)
        p_discharge[10] = 3.2e-4  # MW; discharged = 1.6e-4 MWh, FEC = 4e-5

        def fake_solve(prices, dt, **kwargs):
            return {
                "revenue_eur": 1.0,
                "p_charge": np.zeros(n),
                "p_discharge": p_discharge,
                "soc": np.zeros(n),
                "n_cycles": round(float(p_discharge.sum() * dt / 4.0), 4),  # 0.0
                "success": True,
                "tiebreak_applied": True,
            }

        monkeypatch.setattr(cf, "solve_daily_lp", fake_solve)
        idx = pd.date_range("2026-03-02", periods=n, freq="30min", tz="UTC")
        df = pd.DataFrame({"price_eur_mwh": [50.0] * n}, index=idx)
        df.index.name = "timestamp"
        frame, summary = compute_cycle_cap_frontier(
            df,
            power_mw=2.0,
            duration_hours=2.0,
            efficiency=0.9,
            capex_eur_kwh=600.0,
            cycle_life=6000.0,
            cycle_caps=(1.0,),
        )
        # cost_per_cycle = 600 * 4000 / 6000 = 400 EUR; FEC = 3.2e-4*0.5/4.
        assert summary["cost_per_cycle_eur"] == pytest.approx(400.0)
        assert frame["wear_eur"].iloc[0] == pytest.approx(4e-5 * 400.0)
        assert frame["wear_eur"].iloc[0] > 0.0


def _revenue_by_cap_solver(revenue_by_cap: dict):
    """Mock solve returning a fixed per-day revenue per cap, zero schedule."""

    def fake_solve(prices, dt, **kwargs):
        n = len(prices)
        return {
            "revenue_eur": float(revenue_by_cap[kwargs.get("max_efc_per_day")]),
            "p_charge": np.zeros(n),
            "p_discharge": np.zeros(n),
            "soc": np.zeros(n),
            "n_cycles": 0.0,
            "success": True,
            "tiebreak_applied": True,
        }

    return fake_solve


class TestToleranceBoundary:
    """Codex hardening: pin the CONVERTED window tolerance
    (1 EUR/MW/yr x power_mw x valid_days / 365.25), not a flat 1 EUR.

    power_mw=10 and valid_days=3 give tol_window = 30/365.25 ~ 0.0821 EUR.
    A net gap of 0.05 (inside) discriminates a power-less mutation
    (3/365.25 ~ 0.0082 would flip it to outside), and a gap of 0.5
    (outside) discriminates a flat-1-EUR mutation (would flip it to
    inside).
    """

    def _run(self, monkeypatch: pytest.MonkeyPatch, revenue_by_cap: dict):
        monkeypatch.setattr(cf, "solve_daily_lp", _revenue_by_cap_solver(revenue_by_cap))
        return compute_cycle_cap_frontier(
            _frame([50.0] * 72),  # 3 flat local days
            power_mw=10.0,
            duration_hours=1.0,
            efficiency=0.9,
            capex_eur_kwh=0.0,
            cycle_caps=(1.0, None),
        )

    def test_uncapped_gap_outside_tolerance_is_a_strict_win(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Window net gap = 0.5 EUR > tol (~0.0821): uncapped strict win.
        _, summary = self._run(monkeypatch, {1.0: 100.0, None: 100.0 + 0.5 / 3.0})
        assert summary["best_cap_label"] == UNCAPPED_LABEL
        assert summary["frontier_flat"] is False

    def test_uncapped_gap_inside_tolerance_ties_to_lowest_finite_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Window net gap = 0.05 EUR < tol: the finite cap ties and wins.
        _, summary = self._run(monkeypatch, {1.0: 100.0, None: 100.0 + 0.05 / 3.0})
        assert summary["best_cap_label"] == "1 EFC/day"
        assert summary["frontier_flat"] is True

    def test_uplift_nan_gate_uses_converted_tolerance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # |uncapped net| = 0.05 EUR (inside tol): uplift undefined -> NaN.
        frame_in, _ = self._run(monkeypatch, {1.0: 0.0, None: 0.05 / 3.0})
        assert frame_in["net_uplift_vs_uncapped_pct"].isna().all()
        assert frame_in["net_delta_vs_uncapped_eur"].notna().all()
        # |uncapped net| = 0.5 EUR (outside tol): uplift defined and signed.
        frame_out, _ = self._run(monkeypatch, {1.0: 0.0, None: 0.5 / 3.0})
        assert frame_out["net_uplift_vs_uncapped_pct"].notna().all()
        assert frame_out["net_uplift_vs_uncapped_pct"].iloc[0] == pytest.approx(-100.0)


class TestDegenerateHandling:
    def test_zero_valid_days_returns_typed_empty_frame(self) -> None:
        df = _frame([float("nan")] * 24)
        frame, summary = compute_cycle_cap_frontier(
            df, capex_eur_kwh=100.0, cycle_caps=(1.0, None), **_KW
        )
        assert frame.empty
        assert list(frame.columns) == FRONTIER_COLUMNS
        assert summary["valid_days"] == 0
        assert summary["excluded_days"] == 1
        assert summary["best_cap_label"] is None
        assert summary["frontier_flat"] is False
        assert summary["n_tiebreak_fallback_days"] == 0
        assert summary["cost_per_cycle_eur"] == pytest.approx(100.0 * 1000.0 / 6000.0)

    def test_no_uncapped_row_gives_nan_comparison_columns(self) -> None:
        frame, summary = compute_cycle_cap_frontier(
            _frame(_single_cycle_day()),
            capex_eur_kwh=100.0,
            cycle_caps=(0.5, 1.0),
            **_KW,
        )
        assert frame["net_delta_vs_uncapped_eur"].isna().all()
        assert frame["net_uplift_vs_uncapped_pct"].isna().all()
        assert summary["best_cap_label"] in {"0.5 EFC/day", "1 EFC/day"}

    def test_row_order_and_columns(self) -> None:
        frame, _ = compute_cycle_cap_frontier(
            _frame(_single_cycle_day()),
            capex_eur_kwh=100.0,
            cycle_caps=(2.0, None, 0.5),
            **_KW,
        )
        assert list(frame.columns) == FRONTIER_COLUMNS
        assert list(frame["label"]) == ["0.5 EFC/day", "2 EFC/day", UNCAPPED_LABEL]
        assert math.isnan(frame["cycle_cap"].iloc[-1])

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(power_mw=0.0, duration_hours=1.0), "power_mw"),
            (dict(power_mw=1.0, duration_hours=0.0), "duration_hours"),
        ],
    )
    def test_non_positive_sizing_raises(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            compute_cycle_cap_frontier(
                _frame(_single_cycle_day()),
                efficiency=0.9,
                capex_eur_kwh=100.0,
                **kwargs,
            )

    def test_annualisation_uses_common_valid_days(self) -> None:
        frame, summary = compute_cycle_cap_frontier(
            _frame(_double_cycle_day() + _double_cycle_day()),
            capex_eur_kwh=100.0,
            cycle_caps=(1.0,),
            **_KW,
        )
        row = frame.iloc[0]
        factor = 365.25 / (1.0 * summary["valid_days"])
        assert row["net_eur_per_mw_yr"] == pytest.approx(row["net_eur"] * factor)
        assert row["gross_eur_per_mw_yr"] == pytest.approx(row["gross_eur"] * factor)
        assert row["wear_eur_per_mw_yr"] == pytest.approx(row["wear_eur"] * factor)
