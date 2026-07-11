"""Contract pins for the liquidity-haircut LH-A calculation layer."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

import src.cycle_frontier as cycle_frontier
import src.liquidity as liquidity
from src.cycle_frontier import UNCAPPED_LABEL, compute_cycle_cap_frontier
from src.liquidity import compute_liquidity_cap


def _frame(prices: list[float], start: str = "2026-03-02") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(prices), freq="h", tz="UTC")
    frame = pd.DataFrame({"price_eur_mwh": prices}, index=index)
    frame.index.name = "timestamp"
    return frame


def _arbitrage_day() -> list[float]:
    return [10.0] * 8 + [100.0] * 8 + [50.0] * 8


def _two_spread_day() -> list[float]:
    return [10.0] * 6 + [100.0] * 3 + [10.0] * 6 + [80.0] * 3 + [50.0] * 6


def _peaky_day() -> list[float]:
    """One narrow valley and peak, so executable MW changes captured energy."""
    return [10.0] + [50.0] * 7 + [100.0] + [50.0] * 15


_FRONTIER_KW = dict(
    power_mw=2.0,
    duration_hours=2.0,
    efficiency=0.9,
    capex_eur_kwh=100.0,
    cycle_life=6000.0,
)


class TestComputeLiquidityCap:
    def test_binding_known_answer_and_echoes(self) -> None:
        result = compute_liquidity_cap(
            power_mw=100.0,
            zone_da_volume_mw=400.0,
            max_participation_share=0.10,
        )
        assert result == {
            "executable_power_mw": 40.0,
            "binding": True,
            "participation_at_full_power": 0.25,
            "max_participation_share": 0.10,
            "zone_da_volume_mw": 400.0,
            "power_mw": 100.0,
        }

    def test_non_binding_known_answer(self) -> None:
        result = compute_liquidity_cap(
            power_mw=20.0,
            zone_da_volume_mw=400.0,
            max_participation_share=0.10,
        )
        assert result["executable_power_mw"] == 20.0
        assert result["binding"] is False
        assert result["participation_at_full_power"] == 0.05

    def test_share_one_is_valid_and_non_binding_when_volume_covers_power(self) -> None:
        result = compute_liquidity_cap(
            power_mw=10.0,
            zone_da_volume_mw=15.0,
            max_participation_share=1.0,
        )
        assert result["executable_power_mw"] == 10.0
        assert result["binding"] is False
        assert result["max_participation_share"] == 1.0

    def test_tiny_positive_volume_never_rounds_to_zero(self) -> None:
        result = compute_liquidity_cap(
            power_mw=10.0,
            zone_da_volume_mw=1e-8,
        )
        assert result["executable_power_mw"] == pytest.approx(1e-9)
        assert result["executable_power_mw"] > 0
        assert result["binding"] is True

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("power_mw", 0.0),
            ("power_mw", math.nan),
            ("zone_da_volume_mw", 0.0),
            ("zone_da_volume_mw", -1.0),
            ("zone_da_volume_mw", math.nan),
            ("zone_da_volume_mw", math.inf),
            ("max_participation_share", 0.0),
            ("max_participation_share", -0.1),
            ("max_participation_share", 1.01),
            ("max_participation_share", math.nan),
            ("max_participation_share", math.inf),
        ],
    )
    def test_invalid_inputs_raise_with_field_name(
        self, field: str, value: float,
    ) -> None:
        kwargs = dict(
            power_mw=10.0,
            zone_da_volume_mw=100.0,
            max_participation_share=0.1,
        )
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            compute_liquidity_cap(**kwargs)

    def test_pure_module_and_no_input_mutation(self) -> None:
        kwargs = dict(
            power_mw=10.0,
            zone_da_volume_mw=100.0,
            max_participation_share=0.1,
        )
        before = kwargs.copy()
        compute_liquidity_cap(**kwargs)
        assert kwargs == before
        source = inspect.getsource(liquidity)
        assert "src.dispatch" not in source
        assert "solve_" not in source
        assert "import pandas" not in source


class TestLiquidityFrontierIntegration:
    def test_feature_off_is_bit_identical(self) -> None:
        kwargs = dict(
            da_prices=_frame(_two_spread_day()),
            cycle_caps=(0.5, 1.0, None),
            **_FRONTIER_KW,
        )
        historical_frame, historical_summary = compute_cycle_cap_frontier(**kwargs)
        explicit_frame, explicit_summary = compute_cycle_cap_frontier(
            **kwargs, executable_power_mw=None,
        )
        pd.testing.assert_frame_equal(historical_frame, explicit_frame)
        assert historical_summary == explicit_summary
        assert historical_summary["executable_power_mw"] is None

    def test_non_binding_cap_reproduces_financial_outputs(self) -> None:
        kwargs = dict(
            da_prices=_frame(_two_spread_day()),
            cycle_caps=(0.5, 1.0, None),
            **_FRONTIER_KW,
        )
        off_frame, off_summary = compute_cycle_cap_frontier(**kwargs)
        on_frame, on_summary = compute_cycle_cap_frontier(
            **kwargs, executable_power_mw=_FRONTIER_KW["power_mw"],
        )
        pd.testing.assert_frame_equal(off_frame, on_frame)
        assert {
            key: value for key, value in off_summary.items()
            if key != "executable_power_mw"
        } == {
            key: value for key, value in on_summary.items()
            if key != "executable_power_mw"
        }
        assert on_summary["executable_power_mw"] == 2.0

    def test_binding_cap_matches_same_capacity_resized_asset_window_fields(
        self,
    ) -> None:
        prices = _frame(_two_spread_day())
        capped, capped_summary = compute_cycle_cap_frontier(
            prices,
            cycle_caps=(0.5, 1.0, None),
            executable_power_mw=1.0,
            **_FRONTIER_KW,
        )
        resized, resized_summary = compute_cycle_cap_frontier(
            prices,
            power_mw=1.0,
            duration_hours=4.0,
            efficiency=0.9,
            capex_eur_kwh=100.0,
            cycle_life=6000.0,
            cycle_caps=(0.5, 1.0, None),
        )
        assert capped_summary["valid_days"] == resized_summary["valid_days"]
        for column in (
            "gross_eur",
            "wear_eur",
            "net_eur",
            "avg_efc_per_day",
            "charge_vwap_eur_mwh",
            "discharge_vwap_eur_mwh",
        ):
            np.testing.assert_allclose(
                capped[column], resized[column], rtol=1e-7, atol=1e-7,
                equal_nan=True,
            )

    def test_selector_tolerance_remains_installed_power_based(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_solve(
            prices, dt, power_mw, duration_hours, efficiency,
            power_cap_mw, max_efc_per_day, min_throughput_tiebreak,
        ) -> dict:
            del dt, duration_hours, efficiency, min_throughput_tiebreak
            assert power_mw == 10.0
            assert power_cap_mw == 1.0
            daily_revenue = (
                100.0 if max_efc_per_day == 1.0
                else 100.0 + (0.05 / 3.0)
            )
            n = len(prices)
            return {
                "revenue_eur": daily_revenue,
                "p_charge": np.zeros(n),
                "p_discharge": np.zeros(n),
                "soc": np.zeros(n + 1),
                "n_cycles": 0.0,
                "success": True,
                "tiebreak_applied": True,
            }

        monkeypatch.setattr(cycle_frontier, "solve_daily_lp", fake_solve)
        frame, summary = compute_cycle_cap_frontier(
            _frame([50.0] * 72),
            power_mw=10.0,
            duration_hours=1.0,
            efficiency=0.9,
            capex_eur_kwh=0.0,
            cycle_caps=(1.0, None),
            executable_power_mw=1.0,
        )
        # Gap = 0.05 EUR. Installed-power tolerance is ~0.0821 EUR, so the
        # lowest finite cap wins. An executable-power denominator would use
        # ~0.0082 EUR and incorrectly choose uncapped as a strict winner.
        assert frame["net_eur"].iloc[-1] - frame["net_eur"].iloc[0] == pytest.approx(0.05)
        assert summary["best_cap_label"] == "1 EFC/day"
        assert summary["frontier_flat"] is True

    def test_binding_cap_reduces_gross_with_installed_power_denominator(self) -> None:
        prices = _frame(_peaky_day())
        off, _ = compute_cycle_cap_frontier(
            prices, cycle_caps=(None,), **_FRONTIER_KW,
        )
        capped, summary = compute_cycle_cap_frontier(
            prices, cycle_caps=(None,), executable_power_mw=0.5,
            **_FRONTIER_KW,
        )
        assert capped["gross_eur"].iloc[0] < off["gross_eur"].iloc[0]
        expected = (
            capped["gross_eur"].iloc[0]
            * 365.25
            / (_FRONTIER_KW["power_mw"] * summary["valid_days"])
        )
        assert capped["gross_eur_per_mw_yr"].iloc[0] == pytest.approx(expected)
        executable_denominator = (
            capped["gross_eur"].iloc[0]
            * 365.25
            / (0.5 * summary["valid_days"])
        )
        assert capped["gross_eur_per_mw_yr"].iloc[0] != pytest.approx(
            executable_denominator
        )

    def test_cycle_uncapped_reference_is_still_liquidity_capped(self) -> None:
        prices = _frame(_two_spread_day())
        off, _ = compute_cycle_cap_frontier(
            prices, cycle_caps=(0.5, None), **_FRONTIER_KW,
        )
        capped, _ = compute_cycle_cap_frontier(
            prices, cycle_caps=(0.5, None), executable_power_mw=0.5,
            **_FRONTIER_KW,
        )
        off_uncapped = off[off["label"] == UNCAPPED_LABEL].iloc[0]
        capped_uncapped = capped[capped["label"] == UNCAPPED_LABEL].iloc[0]
        assert capped_uncapped["gross_eur"] < off_uncapped["gross_eur"]
        finite = capped[capped["label"] != UNCAPPED_LABEL].iloc[0]
        assert finite["net_delta_vs_uncapped_eur"] == pytest.approx(
            finite["net_eur"] - capped_uncapped["net_eur"]
        )

    def test_power_cap_clips_charge_and_discharge_legs(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = cycle_frontier.solve_daily_lp
        schedules: list[dict] = []

        def recording_solve(*args, **kwargs) -> dict:
            result = original(*args, **kwargs)
            schedules.append(result)
            return result

        monkeypatch.setattr(cycle_frontier, "solve_daily_lp", recording_solve)
        compute_cycle_cap_frontier(
            _frame([-100.0] * 8 + [100.0] * 8 + [0.0] * 8),
            cycle_caps=(None,), executable_power_mw=0.4,
            **_FRONTIER_KW,
        )
        assert len(schedules) == 1
        assert schedules[0]["p_charge"].max() == pytest.approx(0.4)
        assert schedules[0]["p_discharge"].max() == pytest.approx(0.4)

    def test_liquidity_and_cycle_caps_bind_together(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = cycle_frontier.solve_daily_lp
        schedules: list[dict] = []

        def recording_solve(*args, **kwargs) -> dict:
            result = original(*args, **kwargs)
            schedules.append(result)
            return result

        monkeypatch.setattr(cycle_frontier, "solve_daily_lp", recording_solve)
        compute_cycle_cap_frontier(
            _frame(_arbitrage_day()),
            power_mw=1.0,
            duration_hours=2.0,
            efficiency=0.9,
            capex_eur_kwh=0.0,
            cycle_caps=(0.25,),
            executable_power_mw=0.5,
        )
        assert len(schedules) == 1
        discharged_mwh = float(schedules[0]["p_discharge"].sum())
        assert discharged_mwh == pytest.approx(0.5, abs=1e-7)
        assert schedules[0]["p_discharge"].max() <= 0.5 + 1e-9

    @pytest.mark.parametrize(
        "bad",
        [0.0, -1.0, 2.01, math.nan, math.inf],
    )
    def test_frontier_rejects_invalid_executable_power(self, bad: float) -> None:
        with pytest.raises(ValueError, match="executable_power_mw"):
            compute_cycle_cap_frontier(
                _frame(_arbitrage_day()),
                cycle_caps=(None,),
                executable_power_mw=bad,
                **_FRONTIER_KW,
            )

    def test_summary_echoes_cap_on_and_off_including_empty_windows(self) -> None:
        _, off = compute_cycle_cap_frontier(
            _frame(_arbitrage_day()), cycle_caps=(None,), **_FRONTIER_KW,
        )
        _, on = compute_cycle_cap_frontier(
            _frame(_arbitrage_day()), cycle_caps=(None,),
            executable_power_mw=0.5, **_FRONTIER_KW,
        )
        _, empty = compute_cycle_cap_frontier(
            _frame([math.nan] * 24), cycle_caps=(None,),
            executable_power_mw=0.5, **_FRONTIER_KW,
        )
        assert off["executable_power_mw"] is None
        assert on["executable_power_mw"] == 0.5
        assert empty["executable_power_mw"] == 0.5
