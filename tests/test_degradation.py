"""Tests for battery degradation screening calculations."""

from __future__ import annotations

import math

import pytest

from src.degradation import (
    DAYS_PER_YEAR,
    calculate_annual_throughput_mwh,
    calculate_degradation_cost,
    calculate_levelized_cost_of_storage,
    calculate_net_revenue,
    estimate_battery_lifetime,
)


class TestDegradationCost:
    def test_degradation_cost_basic(self) -> None:
        result = calculate_degradation_cost(
            n_cycles=1.0,
            capex_eur_kwh=300.0,
            capacity_kwh=1000.0,
            cycle_life=6000,
        )

        assert result["cost_per_cycle_eur"] == pytest.approx(50.0)
        assert result["total_degradation_eur"] == pytest.approx(50.0)
        assert result["cycle_life"] == 6000

    def test_degradation_cost_zero_capex(self) -> None:
        result = calculate_degradation_cost(
            n_cycles=365.25,
            capex_eur_kwh=0.0,
            capacity_kwh=1000.0,
        )

        assert result["cost_per_cycle_eur"] == 0.0
        assert result["total_degradation_eur"] == 0.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_cycles": -1.0}, "n_cycles"),
            ({"capex_eur_kwh": -1.0}, "capex_eur_kwh"),
            ({"capacity_kwh": -1.0}, "capacity_kwh"),
            ({"cycle_life": 0.0}, "cycle_life"),
        ],
    )
    def test_degradation_cost_rejects_invalid_inputs(
        self, kwargs: dict[str, float], match: str,
    ) -> None:
        params = {
            "n_cycles": 1.0,
            "capex_eur_kwh": 300.0,
            "capacity_kwh": 1000.0,
            "cycle_life": 6000.0,
        }
        params.update(kwargs)

        with pytest.raises(ValueError, match=match):
            calculate_degradation_cost(**params)


class TestBatteryLifetime:
    def test_lifetime_cycle_limited(self) -> None:
        result = estimate_battery_lifetime(
            avg_cycles_per_day=2.0,
            cycle_life=6000,
            calendar_life_years=20,
        )

        assert result["cycle_limited_years"] == pytest.approx(6000 / (2 * DAYS_PER_YEAR))
        assert result["effective_life_years"] == pytest.approx(result["cycle_limited_years"])
        assert result["limiting_factor"] == "cycling"

    def test_lifetime_calendar_limited(self) -> None:
        result = estimate_battery_lifetime(
            avg_cycles_per_day=0.2,
            cycle_life=6000,
            calendar_life_years=20,
        )

        assert result["effective_life_years"] == 20
        assert result["limiting_factor"] == "calendar"

    def test_zero_cycles_is_calendar_limited(self) -> None:
        result = estimate_battery_lifetime(avg_cycles_per_day=0.0)

        assert math.isinf(result["cycle_limited_years"])
        assert result["effective_life_years"] == 20
        assert result["limiting_factor"] == "calendar"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"avg_cycles_per_day": -0.1}, "avg_cycles_per_day"),
            ({"cycle_life": 0.0}, "cycle_life"),
            ({"calendar_life_years": 0.0}, "calendar_life_years"),
        ],
    )
    def test_lifetime_rejects_invalid_inputs(
        self, kwargs: dict[str, float], match: str,
    ) -> None:
        params = {
            "avg_cycles_per_day": 1.0,
            "cycle_life": 6000.0,
            "calendar_life_years": 20.0,
        }
        params.update(kwargs)

        with pytest.raises(ValueError, match=match):
            estimate_battery_lifetime(**params)


class TestNetRevenue:
    def test_net_revenue_calculation(self) -> None:
        result = calculate_net_revenue(
            annual_gross_revenue=100000.0,
            annual_degradation_cost=15000.0,
        )

        assert result["gross_revenue_eur"] == 100000.0
        assert result["degradation_cost_eur"] == 15000.0
        assert result["net_revenue_eur"] == 85000.0

    def test_net_revenue_degradation_pct(self) -> None:
        result = calculate_net_revenue(
            annual_gross_revenue=100000.0,
            annual_degradation_cost=20000.0,
        )

        assert result["degradation_pct"] == 20.0

    def test_net_revenue_rejects_negative_inputs(self) -> None:
        with pytest.raises(ValueError, match="annual_gross_revenue"):
            calculate_net_revenue(-1.0, 0.0)

        with pytest.raises(ValueError, match="annual_degradation_cost"):
            calculate_net_revenue(1.0, -1.0)


class TestThroughputAndLcos:
    def test_annual_throughput(self) -> None:
        result = calculate_annual_throughput_mwh(
            avg_cycles_per_day=1.5,
            capacity_kwh=4000.0,
        )

        assert result == pytest.approx(1.5 * 4.0 * 2.0 * DAYS_PER_YEAR)

    def test_lcos_basic(self) -> None:
        result = calculate_levelized_cost_of_storage(
            capex_eur_kwh=300.0,
            capacity_kwh=1000.0,
            effective_life_years=10.0,
            annual_throughput_mwh=1000.0,
            opex_eur_kwh_yr=5.0,
        )

        assert result == pytest.approx((300000.0 + 50000.0) / 10000.0)

    def test_lcos_rejects_zero_throughput(self) -> None:
        with pytest.raises(ValueError, match="annual_throughput_mwh"):
            calculate_levelized_cost_of_storage(
                capex_eur_kwh=300.0,
                capacity_kwh=1000.0,
                effective_life_years=10.0,
                annual_throughput_mwh=0.0,
            )
