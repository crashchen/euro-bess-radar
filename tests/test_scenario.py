"""Tests for Monte Carlo scenario analysis."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.scenario import (
    annuity_pv_factor,
    bootstrap_annual_revenue,
    calculate_npv_distribution,
    decaying_annuity_pv_factor,
    sensitivity_table,
)


def _legacy_sensitivity_frame(
    *,
    base_revenue: float,
    total_capex: float,
    effective_life_years: float,
    annual_degradation_cost: float,
    discount_rate: float,
) -> pd.DataFrame:
    vary = {
        "revenue": [0.7, 1.0, 1.3],
        "capex": [0.8, 1.0, 1.2],
        "discount_rate": [0.06, 0.08, 0.10],
        "lifetime": [15, 20, 25],
    }
    rows = []
    for param, values in vary.items():
        for value in values:
            revenue = (
                base_revenue * value if param == "revenue" else base_revenue
            )
            capex = total_capex * value if param == "capex" else total_capex
            rate = value if param == "discount_rate" else discount_rate
            life = value if param == "lifetime" else effective_life_years
            factor = annuity_pv_factor(float(life), rate)
            npv = (revenue - annual_degradation_cost) * factor - capex
            rows.append({"param": param, "value": value, "npv": round(npv, 2)})
    return pd.DataFrame(rows)


class TestDecayingAnnuityPvFactor:
    @pytest.mark.parametrize(
        ("life", "rate", "decay", "floor"),
        [
            (20.0, 0.08, 0.0, 0.0),
            (2.5, 0.0, 0.0, 0.4),
            (14.9, 0.08, 0.0, 1.0),
            (20.0, 0.08, 0.3, 1.0),
            (0.0, 0.08, 0.4, 1.0),
        ],
    )
    def test_inactive_is_bit_identical_to_flat_factor(
        self,
        life: float,
        rate: float,
        decay: float,
        floor: float,
    ) -> None:
        actual = decaying_annuity_pv_factor(life, rate, decay, floor)
        assert actual == annuity_pv_factor(life, rate)

    @pytest.mark.parametrize(("decay", "floor"), [(0.0, 0.2), (0.4, 1.0)])
    def test_inactive_delegates_to_flat_helper(
        self,
        monkeypatch: pytest.MonkeyPatch,
        decay: float,
        floor: float,
    ) -> None:
        calls: list[tuple[float, float]] = []

        def fake_flat(life: float, rate: float) -> float:
            calls.append((life, rate))
            return 123.456

        monkeypatch.setattr("src.scenario.annuity_pv_factor", fake_flat)
        actual = decaying_annuity_pv_factor(7.5, 0.09, decay, floor)
        assert actual == 123.456
        assert calls == [(7.5, 0.09)]

    @pytest.mark.parametrize(
        ("life", "rate", "decay", "floor", "expected"),
        [
            (3.0, 0.0, 0.1, 0.0, 2.71),
            (2.5, 0.0, 0.1, 0.0, 2.305),
            (4.0, 0.0, 0.5, 0.4, 2.3),
            (
                3.0,
                0.07,
                0.2,
                0.1,
                1 / 1.07 + 0.8 / 1.07**2 + 0.64 / 1.07**3,
            ),
        ],
    )
    def test_known_answers(
        self,
        life: float,
        rate: float,
        decay: float,
        floor: float,
        expected: float,
    ) -> None:
        actual = decaying_annuity_pv_factor(life, rate, decay, floor)
        assert actual == pytest.approx(expected)

    def test_zero_floor_is_strictly_decreasing_in_decay(self) -> None:
        factors = [
            decaying_annuity_pv_factor(8.0, 0.08, decay, 0.0)
            for decay in (0.05, 0.15, 0.35)
        ]
        assert factors[0] > factors[1] > factors[2]

    def test_floor_is_nonincreasing_and_has_exact_plateau(self) -> None:
        before_floor = decaying_annuity_pv_factor(4.0, 0.0, 0.2, 0.4)
        plateau_a = decaying_annuity_pv_factor(4.0, 0.0, 0.7, 0.4)
        plateau_b = decaying_annuity_pv_factor(4.0, 0.0, 0.8, 0.4)
        assert before_floor > plateau_a
        assert plateau_a == plateau_b
        assert plateau_a == pytest.approx(2.2)

    @pytest.mark.parametrize(
        ("life", "rate", "decay", "floor"),
        [
            (2.5, 0.0, 0.1, 0.0),
            (10.0, 0.08, 0.2, 0.25),
            (20.0, 0.12, 0.7, 0.4),
            (0.5, 0.08, 0.3, 0.6),
        ],
    )
    def test_factor_is_bounded_by_floor_and_flat_annuity(
        self,
        life: float,
        rate: float,
        decay: float,
        floor: float,
    ) -> None:
        flat = annuity_pv_factor(life, rate)
        decayed = decaying_annuity_pv_factor(life, rate, decay, floor)
        assert floor * flat <= decayed <= flat

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("annual_decay_rate", -0.01),
            ("annual_decay_rate", 1.0),
            ("annual_decay_rate", np.nan),
            ("annual_decay_rate", np.inf),
            ("annual_decay_rate", None),
            ("decay_floor_share", -0.01),
            ("decay_floor_share", 1.01),
            ("decay_floor_share", np.nan),
            ("decay_floor_share", np.inf),
            ("decay_floor_share", None),
        ],
    )
    def test_invalid_decay_inputs_raise_with_field_name(
        self,
        field: str,
        value: float | None,
    ) -> None:
        kwargs = {field: value}
        with pytest.raises(ValueError, match=field):
            decaying_annuity_pv_factor(10.0, 0.08, **kwargs)

    def test_validation_precedes_inactive_delegation(self) -> None:
        with pytest.raises(ValueError, match="decay_floor_share"):
            decaying_annuity_pv_factor(
                0.0,
                0.08,
                annual_decay_rate=0.0,
                decay_floor_share=1.1,
            )


class TestBootstrapAnnualRevenue:
    def test_api_remains_year_one_only_without_decay_parameters(self) -> None:
        parameters = inspect.signature(bootstrap_annual_revenue).parameters
        assert "annual_decay_rate" not in parameters
        assert "decay_floor_share" not in parameters

    def test_output_shape(self) -> None:
        daily = pd.Series([100.0] * 30)
        result = bootstrap_annual_revenue(daily, n_simulations=1000)
        assert result["simulations"].shape == (1000,)

    def test_percentile_ordering(self) -> None:
        rng = np.random.default_rng(0)
        daily = pd.Series(rng.normal(200, 50, 365))
        result = bootstrap_annual_revenue(daily)
        assert result["p10"] <= result["p25"] <= result["p50"]
        assert result["p50"] <= result["p75"] <= result["p90"]

    def test_constant_daily_revenue(self) -> None:
        daily = pd.Series([100.0] * 365)
        result = bootstrap_annual_revenue(daily, n_simulations=100)
        assert result["p50"] == pytest.approx(365 * 100, rel=1e-6)
        assert result["std"] == pytest.approx(0.0, abs=1e-6)

    def test_empty_input(self) -> None:
        result = bootstrap_annual_revenue(pd.Series([], dtype=float))
        assert result["mean"] == 0.0
        assert result["p50"] == 0.0

    def test_nan_values_ignored(self) -> None:
        daily = pd.Series([100.0, np.nan, 100.0] * 50)
        result = bootstrap_annual_revenue(daily, n_simulations=100)
        assert result["p50"] == pytest.approx(365 * 100, rel=0.05)

    def test_reproducible_with_seed(self) -> None:
        daily = pd.Series(np.random.default_rng(0).normal(200, 50, 365))
        r1 = bootstrap_annual_revenue(daily, seed=42)
        r2 = bootstrap_annual_revenue(daily, seed=42)
        assert r1["p50"] == r2["p50"]


class TestNpvDistribution:
    @pytest.mark.parametrize(
        "decay_kwargs",
        [
            {},
            {"annual_decay_rate": 0.0, "decay_floor_share": 0.0},
            {"annual_decay_rate": 0.0, "decay_floor_share": 0.4},
            {"annual_decay_rate": 0.2, "decay_floor_share": 1.0},
        ],
    )
    def test_inactive_decay_is_bit_identical_to_legacy_expression(
        self,
        decay_kwargs: dict[str, float],
    ) -> None:
        annual_revenue = np.array([100_000.25, -4_000.5, 37_777.75])
        flat_factor = annuity_pv_factor(14.9, 0.08)
        expected = (annual_revenue - 12_345.67) * flat_factor - 456_789.01

        result = calculate_npv_distribution(
            annual_revenue,
            total_capex=456_789.01,
            annual_degradation_cost=12_345.67,
            effective_life_years=14.9,
            discount_rate=0.08,
            **decay_kwargs,
        )

        assert np.array_equal(result["npv_array"], expected)

    def test_active_decay_keeps_wear_cost_flat(self) -> None:
        annual_revenue = np.array([100.0, 140.0])
        decayed_factor = decaying_annuity_pv_factor(3.0, 0.0, 0.5, 0.0)
        flat_factor = annuity_pv_factor(3.0, 0.0)
        expected = annual_revenue * decayed_factor - 40.0 * flat_factor - 25.0
        prohibited_cost_decay = (annual_revenue - 40.0) * decayed_factor - 25.0

        result = calculate_npv_distribution(
            annual_revenue,
            total_capex=25.0,
            annual_degradation_cost=40.0,
            effective_life_years=3.0,
            discount_rate=0.0,
            annual_decay_rate=0.5,
        )

        np.testing.assert_allclose(result["npv_array"], expected)
        assert not np.allclose(result["npv_array"], prohibited_cost_decay)

    def test_active_decay_does_not_mutate_revenue_draws(self) -> None:
        annual_revenue = np.array([10.0, 20.0, 30.0])
        original = annual_revenue.copy()
        calculate_npv_distribution(
            annual_revenue,
            total_capex=10.0,
            annual_decay_rate=0.2,
            decay_floor_share=0.1,
        )
        assert np.array_equal(annual_revenue, original)

    def test_positive_npv_with_good_revenue(self) -> None:
        # Revenue = 100k/yr, capex = 500k, life = 20yr, rate = 8%
        # PV factor ~9.82 → NPV ~9.82*100k - 500k > 0
        annual_rev = np.full(1000, 100000.0)
        result = calculate_npv_distribution(
            annual_rev, total_capex=500000.0,
            effective_life_years=20, discount_rate=0.08,
        )
        assert result["npv_p50"] > 0
        assert result["prob_positive_npv"] == 1.0

    def test_negative_npv_with_low_revenue(self) -> None:
        annual_rev = np.full(1000, 10000.0)
        result = calculate_npv_distribution(
            annual_rev, total_capex=500000.0,
            effective_life_years=10, discount_rate=0.10,
        )
        assert result["npv_p50"] < 0
        assert result["prob_positive_npv"] == 0.0

    def test_degradation_reduces_npv(self) -> None:
        annual_rev = np.full(100, 100000.0)
        r_no_deg = calculate_npv_distribution(annual_rev, total_capex=500000.0)
        r_deg = calculate_npv_distribution(
            annual_rev, total_capex=500000.0, annual_degradation_cost=20000.0,
        )
        assert r_deg["npv_p50"] < r_no_deg["npv_p50"]

    def test_zero_capex_equals_pv_of_revenue(self) -> None:
        annual_rev = np.full(100, 50000.0)
        result = calculate_npv_distribution(
            annual_rev, total_capex=0.0,
            effective_life_years=10, discount_rate=0.08,
        )
        assert result["npv_p50"] > 0
        assert result["prob_positive_npv"] == 1.0

    def test_fractional_lifetime_smoothly_varies_npv(self) -> None:
        """Fractional life (e.g. 14.9 from degradation model) must not be
        floored to 14 — that drops nearly a full year of discounted revenue.
        """
        annual_rev = np.full(100, 100000.0)
        kwargs = dict(total_capex=500000.0, discount_rate=0.08)
        r_14 = calculate_npv_distribution(annual_rev, effective_life_years=14.0, **kwargs)
        r_14_9 = calculate_npv_distribution(annual_rev, effective_life_years=14.9, **kwargs)
        r_15 = calculate_npv_distribution(annual_rev, effective_life_years=15.0, **kwargs)
        # 14.9 must be strictly between 14 and 15, not equal to 14
        assert r_14["npv_p50"] < r_14_9["npv_p50"] < r_15["npv_p50"]
        # And much closer to 15 than to 14
        gap_to_14 = r_14_9["npv_p50"] - r_14["npv_p50"]
        gap_to_15 = r_15["npv_p50"] - r_14_9["npv_p50"]
        assert gap_to_14 > gap_to_15


class TestSensitivityTable:
    @pytest.mark.parametrize(
        "decay_kwargs",
        [
            {},
            {"annual_decay_rate": 0.0, "decay_floor_share": 0.0},
            {"annual_decay_rate": 0.0, "decay_floor_share": 0.4},
            {"annual_decay_rate": 0.2, "decay_floor_share": 1.0},
        ],
    )
    def test_inactive_decay_is_exact_legacy_frame(
        self,
        decay_kwargs: dict[str, float],
    ) -> None:
        expected = _legacy_sensitivity_frame(
            base_revenue=100_000.25,
            total_capex=456_789.01,
            effective_life_years=14.9,
            annual_degradation_cost=12_345.67,
            discount_rate=0.08,
        )
        actual = sensitivity_table(
            base_revenue=100_000.25,
            total_capex=456_789.01,
            effective_life_years=14.9,
            annual_degradation_cost=12_345.67,
            discount_rate=0.08,
            **decay_kwargs,
        )
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    @pytest.mark.parametrize(
        ("decay", "expected_high"),
        [(0.2, 0.4), (0.97, 0.985)],
    )
    def test_default_active_table_adds_absolute_decay_axis(
        self,
        decay: float,
        expected_high: float,
    ) -> None:
        result = sensitivity_table(
            base_revenue=100_000.0,
            total_capex=500_000.0,
            annual_decay_rate=decay,
            decay_floor_share=0.1,
        )
        decay_values = result.loc[result["param"] == "decay", "value"].tolist()
        assert len(result) == 15
        assert decay_values == pytest.approx([0.0, decay, expected_high])

    def test_custom_vary_is_not_mutated_or_auto_extended(self) -> None:
        vary = {"revenue": [0.8, 1.0, 1.2]}
        original = {key: values.copy() for key, values in vary.items()}
        result = sensitivity_table(
            base_revenue=100_000.0,
            total_capex=500_000.0,
            annual_decay_rate=0.2,
            vary=vary,
        )
        assert vary == original
        assert list(result["param"].unique()) == ["revenue"]

    def test_custom_decay_axis_values_are_absolute_rates(self) -> None:
        result = sensitivity_table(
            base_revenue=100.0,
            total_capex=25.0,
            effective_life_years=3.0,
            annual_degradation_cost=10.0,
            discount_rate=0.0,
            annual_decay_rate=0.3,
            decay_floor_share=0.2,
            vary={"decay": [0.0, 0.4]},
        )
        flat = annuity_pv_factor(3.0, 0.0)
        expected_flat = round((100.0 - 10.0) * flat - 25.0, 2)
        expected_decayed = round(
            100.0 * decaying_annuity_pv_factor(3.0, 0.0, 0.4, 0.2)
            - 10.0 * flat
            - 25.0,
            2,
        )
        assert result["value"].tolist() == [0.0, 0.4]
        assert result["npv"].tolist() == [expected_flat, expected_decayed]

    def test_non_decay_axis_uses_base_decay_and_flat_wear(self) -> None:
        result = sensitivity_table(
            base_revenue=100.0,
            total_capex=25.0,
            effective_life_years=2.5,
            annual_degradation_cost=10.0,
            discount_rate=0.0,
            annual_decay_rate=0.2,
            decay_floor_share=0.1,
            vary={"revenue": [1.3]},
        )
        expected = round(
            130.0 * decaying_annuity_pv_factor(2.5, 0.0, 0.2, 0.1)
            - 10.0 * annuity_pv_factor(2.5, 0.0)
            - 25.0,
            2,
        )
        assert result.iloc[0].to_dict() == {
            "param": "revenue",
            "value": 1.3,
            "npv": expected,
        }

    @pytest.mark.parametrize(
        ("decay", "floor"),
        [(0.0, 0.4), (0.2, 1.0)],
    )
    def test_inactive_decay_keeps_default_table_shape(
        self,
        decay: float,
        floor: float,
    ) -> None:
        result = sensitivity_table(
            base_revenue=100_000.0,
            total_capex=500_000.0,
            annual_decay_rate=decay,
            decay_floor_share=floor,
        )
        assert len(result) == 12
        assert "decay" not in set(result["param"])

    def test_validation_runs_even_for_empty_custom_vary(self) -> None:
        with pytest.raises(ValueError, match="decay_floor_share"):
            sensitivity_table(
                base_revenue=100_000.0,
                total_capex=500_000.0,
                annual_decay_rate=0.0,
                decay_floor_share=1.1,
                vary={},
            )

    def test_expected_shape(self) -> None:
        result = sensitivity_table(
            base_revenue=100000.0, total_capex=500000.0,
        )
        # 4 params x 3 values each = 12 rows
        assert len(result) == 12
        assert set(result.columns) == {"param", "value", "npv"}

    def test_higher_revenue_higher_npv(self) -> None:
        result = sensitivity_table(
            base_revenue=100000.0, total_capex=500000.0,
        )
        rev_rows = result[result["param"] == "revenue"].sort_values("value")
        npvs = rev_rows["npv"].tolist()
        assert npvs[0] < npvs[1] < npvs[2]

    def test_higher_capex_lower_npv(self) -> None:
        result = sensitivity_table(
            base_revenue=100000.0, total_capex=500000.0,
        )
        capex_rows = result[result["param"] == "capex"].sort_values("value")
        npvs = capex_rows["npv"].tolist()
        assert npvs[0] > npvs[1] > npvs[2]

    def test_custom_vary(self) -> None:
        result = sensitivity_table(
            base_revenue=100000.0,
            total_capex=500000.0,
            vary={"revenue": [0.8, 1.0, 1.2]},
        )
        assert len(result) == 3
        assert list(result["param"].unique()) == ["revenue"]
