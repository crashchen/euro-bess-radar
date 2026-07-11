"""Contract pins for the contracted-floor CF-A calculation layer."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import src.contracted_floor as contracted_floor
import src.scenario as scenario
from src.contracted_floor import compute_contracted_floor_overlay

_BASE = {
    "merchant_net_eur_per_mw_yr": 100.0,
    "power_mw": 2.0,
    "quoted_floor_eur_per_mw_yr": 150.0,
    "floor_tenor_years": 5.0,
    "contract_availability": 1.0,
    "discount_rate": 0.08,
}


class TestAnnuityHelperReuse:
    def test_public_helper_preserves_fractional_year_formula(self) -> None:
        assert scenario.annuity_pv_factor(2.5, 0.0) == 2.5
        expected = 1 / 1.1 + 1 / 1.1**2 + 0.5 / 1.1**3
        assert scenario.annuity_pv_factor(2.5, 0.1) == pytest.approx(expected)

    def test_existing_npv_callers_use_public_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(scenario, "annuity_pv_factor", lambda *_: 2.0)
        npv = scenario.calculate_npv_distribution(
            np.array([100.0]), total_capex=0.0
        )
        sensitivity = scenario.sensitivity_table(
            base_revenue=100.0,
            total_capex=0.0,
            vary={"revenue": [1.0]},
        )
        assert npv["npv_p50"] == 200.0
        assert sensitivity["npv"].iloc[0] == 200.0

    def test_overlay_uses_public_annuity_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(contracted_floor, "annuity_pv_factor", lambda *_: 3.0)
        result = compute_contracted_floor_overlay(**_BASE)
        assert result["merchant_pv_eur"] == 3.0 * result["merchant_net_eur"]
        assert result["floor_protected_pv_eur"] == (
            3.0 * result["floor_protected_cashflow_eur"]
        )


class TestContractedFloorIdentities:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"quoted_floor_eur_per_mw_yr": 0.0},
            {"contract_availability": 0.0},
        ],
    )
    def test_zero_effective_floor_reproduces_nonnegative_merchant(
        self, overrides: dict[str, float]
    ) -> None:
        kwargs = {**_BASE, **overrides}
        result = compute_contracted_floor_overlay(**kwargs)
        assert result["merchant_net_eur"] == 200.0
        assert result["floor_protected_cashflow_eur"] == 200.0
        assert result["annual_top_up_eur"] == 0.0

    def test_zero_floor_protects_negative_merchant_to_zero(self) -> None:
        result = compute_contracted_floor_overlay(
            **{
                **_BASE,
                "merchant_net_eur_per_mw_yr": -50.0,
                "quoted_floor_eur_per_mw_yr": 0.0,
            }
        )
        assert result["merchant_net_eur"] == -100.0
        assert result["effective_floor_eur"] == 0.0
        assert result["floor_protected_cashflow_eur"] == 0.0
        assert result["annual_top_up_eur"] == 100.0

    def test_floor_below_merchant_has_no_top_up(self) -> None:
        result = compute_contracted_floor_overlay(
            **{**_BASE, "quoted_floor_eur_per_mw_yr": 50.0}
        )
        assert result["effective_floor_eur"] == 100.0
        assert result["floor_protected_cashflow_eur"] == 200.0
        assert result["annual_top_up_eur"] == 0.0

    def test_floor_above_merchant_pays_only_the_gap(self) -> None:
        result = compute_contracted_floor_overlay(
            **{
                **_BASE,
                "quoted_floor_eur_per_mw_yr": 200.0,
                "contract_availability": 0.75,
            }
        )
        assert result["quoted_floor_eur"] == 400.0
        assert result["effective_floor_eur_per_mw_yr"] == 150.0
        assert result["effective_floor_eur"] == 300.0
        assert result["floor_protected_cashflow_eur"] == 300.0
        assert result["annual_top_up_eur"] == 100.0

    def test_positive_floor_covers_negative_merchant_without_clamping_it(self) -> None:
        result = compute_contracted_floor_overlay(
            **{
                **_BASE,
                "merchant_net_eur_per_mw_yr": -50.0,
                "quoted_floor_eur_per_mw_yr": 40.0,
                "contract_availability": 0.5,
            }
        )
        assert result["merchant_net_eur"] == -100.0
        assert result["effective_floor_eur"] == 40.0
        assert result["floor_protected_cashflow_eur"] == 40.0
        assert result["annual_top_up_eur"] == 140.0

    def test_power_scaling_changes_only_total_eur_outputs(self) -> None:
        base = compute_contracted_floor_overlay(**_BASE)
        scaled = compute_contracted_floor_overlay(**{**_BASE, "power_mw": 8.0})
        total_keys = (
            "merchant_net_eur",
            "quoted_floor_eur",
            "effective_floor_eur",
            "floor_protected_cashflow_eur",
            "annual_top_up_eur",
            "merchant_pv_eur",
            "floor_protected_pv_eur",
            "floor_pv_uplift_eur",
        )
        for key in total_keys:
            assert scaled[key] == pytest.approx(4.0 * base[key])
        for key in (
            "merchant_net_eur_per_mw_yr",
            "effective_floor_eur_per_mw_yr",
            "contract_availability",
        ):
            assert scaled[key] == base[key]

    @pytest.mark.parametrize("discount_rate", [0.0, 0.08])
    def test_pv_uplift_identity_with_fractional_tenor(
        self, discount_rate: float
    ) -> None:
        result = compute_contracted_floor_overlay(
            **{
                **_BASE,
                "floor_tenor_years": 2.5,
                "discount_rate": discount_rate,
            }
        )
        assert (
            result["floor_protected_pv_eur"] - result["merchant_pv_eur"]
        ) == pytest.approx(result["floor_pv_uplift_eur"])


class TestContractedFloorValidation:
    @pytest.mark.parametrize(
        ("field", "bad"),
        [
            ("merchant_net_eur_per_mw_yr", np.nan),
            ("power_mw", np.inf),
            ("quoted_floor_eur_per_mw_yr", -np.inf),
            ("floor_tenor_years", np.nan),
            ("contract_availability", np.inf),
            ("discount_rate", np.nan),
        ],
    )
    def test_nonfinite_input_raises(self, field: str, bad: float) -> None:
        with pytest.raises(ValueError, match=field):
            compute_contracted_floor_overlay(**{**_BASE, field: bad})

    @pytest.mark.parametrize(
        ("field", "bad"),
        [
            ("power_mw", -0.1),
            ("quoted_floor_eur_per_mw_yr", -0.1),
            ("floor_tenor_years", 0.0),
            ("floor_tenor_years", -1.0),
            ("contract_availability", -0.01),
            ("contract_availability", 1.01),
            ("discount_rate", -0.01),
        ],
    )
    def test_invalid_domain_raises(self, field: str, bad: float) -> None:
        with pytest.raises(ValueError, match=field):
            compute_contracted_floor_overlay(**{**_BASE, field: bad})

    def test_core_is_pure_and_has_no_solver_dependency(self) -> None:
        kwargs = dict(_BASE)
        before = dict(kwargs)
        compute_contracted_floor_overlay(**kwargs)
        assert kwargs == before
        source = inspect.getsource(contracted_floor)
        assert "src.dispatch" not in source
        assert "solve_" not in source
