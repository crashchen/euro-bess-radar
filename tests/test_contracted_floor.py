"""Contract pins for the contracted-floor CF-A calculation layer."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import src.contracted_floor as contracted_floor
import src.scenario as scenario
from src.contracted_floor import (
    MAX_FLOOR_TRAJECTORY_YEARS,
    compute_contracted_floor_overlay,
    compute_decaying_contracted_floor_overlay,
)

_BASE = {
    "merchant_net_eur_per_mw_yr": 100.0,
    "power_mw": 2.0,
    "quoted_floor_eur_per_mw_yr": 150.0,
    "floor_tenor_years": 5.0,
    "contract_availability": 1.0,
    "discount_rate": 0.08,
}

_DECAY_BASE = {
    "merchant_net_eur_per_mw_yr": 80.0,
    "merchant_gross_eur_per_mw_yr": 100.0,
    "power_mw": 1.0,
    "quoted_floor_eur_per_mw_yr": 60.0,
    "floor_tenor_years": 3.0,
    "contract_availability": 1.0,
    "discount_rate": 0.0,
}

_V1_KEYS = (
    "merchant_net_eur",
    "merchant_net_eur_per_mw_yr",
    "quoted_floor_eur",
    "effective_floor_eur",
    "effective_floor_eur_per_mw_yr",
    "floor_protected_cashflow_eur",
    "annual_top_up_eur",
    "merchant_pv_eur",
    "floor_protected_pv_eur",
    "floor_pv_uplift_eur",
    "floor_tenor_years",
    "discount_rate",
    "contract_availability",
)


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


class TestDecayingContractedFloorInactive:
    @pytest.mark.parametrize(
        ("decay", "floor", "escalation"),
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.4, 0.0),
            (0.3, 1.0, 0.0),
        ],
    )
    def test_inactive_outputs_are_bit_identical_to_v1(
        self,
        decay: float,
        floor: float,
        escalation: float,
    ) -> None:
        v1_kwargs = {
            key: value
            for key, value in _DECAY_BASE.items()
            if key != "merchant_gross_eur_per_mw_yr"
        }
        expected = compute_contracted_floor_overlay(**v1_kwargs)
        result = compute_decaying_contracted_floor_overlay(
            **_DECAY_BASE,
            annual_decay_rate=decay,
            decay_floor_share=floor,
            floor_escalation_rate=escalation,
        )
        for key in _V1_KEYS:
            assert result[key] == expected[key]
        assert result["composition_active"] is False
        assert result["per_year"] == []
        assert result["crossover_year"] is None
        assert result["n_binding_years"] is None

    def test_omitted_options_delegate_even_when_flat_floor_binds(self) -> None:
        kwargs = {**_DECAY_BASE, "quoted_floor_eur_per_mw_yr": 120.0}
        result = compute_decaying_contracted_floor_overlay(**kwargs)
        expected = compute_contracted_floor_overlay(
            **{
                key: value
                for key, value in kwargs.items()
                if key != "merchant_gross_eur_per_mw_yr"
            }
        )
        assert result["annual_top_up_eur"] > 0.0
        for key in _V1_KEYS:
            assert result[key] == expected[key]
        assert result["n_binding_years"] is None

    def test_inactive_path_calls_v1_delegate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel = compute_contracted_floor_overlay(
            **{
                key: value
                for key, value in _DECAY_BASE.items()
                if key != "merchant_gross_eur_per_mw_yr"
            }
        )
        sentinel["merchant_pv_eur"] = object()
        monkeypatch.setattr(
            contracted_floor,
            "compute_contracted_floor_overlay",
            lambda **_: sentinel,
        )
        result = compute_decaying_contracted_floor_overlay(**_DECAY_BASE)
        assert result["merchant_pv_eur"] is sentinel["merchant_pv_eur"]


class TestDecayingContractedFloorKnownAnswers:
    def test_decay_known_answer(self) -> None:
        result = compute_decaying_contracted_floor_overlay(
            **_DECAY_BASE,
            annual_decay_rate=0.5,
        )
        assert [row["merchant_eur"] for row in result["per_year"]] == [
            80.0,
            30.0,
            5.0,
        ]
        assert [row["protected_eur"] for row in result["per_year"]] == [
            80.0,
            60.0,
            60.0,
        ]
        assert [row["top_up_eur"] for row in result["per_year"]] == [
            0.0,
            30.0,
            55.0,
        ]
        assert result["merchant_pv_eur"] == 115.0
        assert result["floor_protected_pv_eur"] == 200.0
        assert result["floor_pv_uplift_eur"] == 85.0
        assert result["crossover_year"] == 2
        assert result["n_binding_years"] == 2

    def test_escalation_only_known_answer(self) -> None:
        result = compute_decaying_contracted_floor_overlay(
            **{
                **_DECAY_BASE,
                "quoted_floor_eur_per_mw_yr": 75.0,
            },
            floor_escalation_rate=0.1,
        )
        assert result["composition_active"] is True
        assert [row["merchant_eur"] for row in result["per_year"]] == [
            80.0,
            80.0,
            80.0,
        ]
        assert [row["floor_eur"] for row in result["per_year"]] == pytest.approx(
            [75.0, 82.5, 90.75]
        )
        assert [row["protected_eur"] for row in result["per_year"]] == (
            pytest.approx([80.0, 82.5, 90.75])
        )
        assert result["crossover_year"] == 2

    @pytest.mark.parametrize("discount_rate", [0.0, 0.07])
    def test_fractional_tenor_uses_residual_year_values_and_discount(
        self, discount_rate: float
    ) -> None:
        result = compute_decaying_contracted_floor_overlay(
            **{
                **_DECAY_BASE,
                "quoted_floor_eur_per_mw_yr": 75.0,
                "floor_tenor_years": 2.5,
                "discount_rate": discount_rate,
            },
            annual_decay_rate=0.1,
        )
        rows = result["per_year"]
        assert [row["merchant_eur"] for row in rows] == pytest.approx(
            [80.0, 70.0, 61.0]
        )
        assert [row["year_fraction"] for row in rows] == [1.0, 1.0, 0.5]
        if discount_rate == 0.0:
            expected_merchant = 180.5
            expected_protected = 192.5
            expected_top_up = 12.0
        else:
            q = 1.0 + discount_rate
            expected_merchant = 80.0 / q + 70.0 / q**2 + 0.5 * 61.0 / q**3
            expected_protected = 80.0 / q + 75.0 / q**2 + 0.5 * 75.0 / q**3
            expected_top_up = 5.0 / q**2 + 0.5 * 14.0 / q**3
        assert result["merchant_pv_eur"] == pytest.approx(expected_merchant)
        assert result["floor_protected_pv_eur"] == pytest.approx(
            expected_protected
        )
        assert result["floor_pv_uplift_eur"] == pytest.approx(expected_top_up)

    def test_flat_wear_not_net_decay(self) -> None:
        result = compute_decaying_contracted_floor_overlay(
            **_DECAY_BASE,
            annual_decay_rate=0.5,
        )
        assert result["per_year"][1]["merchant_eur"] == 30.0
        assert result["per_year"][1]["merchant_eur"] != 40.0

    def test_active_year_one_anchors_exactly_to_v1_annual_values(self) -> None:
        active = compute_decaying_contracted_floor_overlay(
            **_DECAY_BASE,
            annual_decay_rate=0.5,
        )
        inactive = compute_decaying_contracted_floor_overlay(**_DECAY_BASE)
        annual_keys = (
            "merchant_net_eur",
            "effective_floor_eur",
            "floor_protected_cashflow_eur",
            "annual_top_up_eur",
        )
        for key in annual_keys:
            assert active[key] == inactive[key]
        first = active["per_year"][0]
        assert first["merchant_eur"] == active["merchant_net_eur"]
        assert first["floor_eur"] == active["effective_floor_eur"]
        assert first["protected_eur"] == active["floor_protected_cashflow_eur"]
        assert first["top_up_eur"] == active["annual_top_up_eur"]

    def test_decay_floor_plateau_is_exact(self) -> None:
        lower = compute_decaying_contracted_floor_overlay(
            **{**_DECAY_BASE, "floor_tenor_years": 4.0},
            annual_decay_rate=0.7,
            decay_floor_share=0.4,
        )
        higher = compute_decaying_contracted_floor_overlay(
            **{**_DECAY_BASE, "floor_tenor_years": 4.0},
            annual_decay_rate=0.8,
            decay_floor_share=0.4,
        )
        assert lower["per_year"] == higher["per_year"]
        for key in (
            "merchant_pv_eur",
            "floor_protected_pv_eur",
            "floor_pv_uplift_eur",
        ):
            assert lower[key] == higher[key]


class TestDecayingContractedFloorBinding:
    def test_exact_tie_is_not_binding_and_binding_flags_are_suffix(self) -> None:
        result = compute_decaying_contracted_floor_overlay(
            **{**_DECAY_BASE, "quoted_floor_eur_per_mw_yr": 30.0},
            annual_decay_rate=0.5,
        )
        rows = result["per_year"]
        assert [row["merchant_eur"] for row in rows] == [80.0, 30.0, 5.0]
        assert [row["top_up_eur"] for row in rows] == [0.0, 0.0, 25.0]
        assert [row["binding"] for row in rows] == [False, False, True]
        assert result["crossover_year"] == 3
        assert result["n_binding_years"] == 1

    def test_negative_merchant_years_are_not_clamped(self) -> None:
        result = compute_decaying_contracted_floor_overlay(
            **{
                **_DECAY_BASE,
                "merchant_net_eur_per_mw_yr": 10.0,
                "quoted_floor_eur_per_mw_yr": 20.0,
                "floor_tenor_years": 4.0,
            },
            annual_decay_rate=0.5,
            decay_floor_share=0.4,
        )
        rows = result["per_year"]
        assert [row["merchant_eur"] for row in rows] == [10.0, -40.0, -50.0, -50.0]
        assert [row["top_up_eur"] for row in rows] == [10.0, 60.0, 70.0, 70.0]
        for row in rows:
            assert row["protected_eur"] == row["merchant_eur"] + row["top_up_eur"]
        assert all(
            row["top_up_eur"] > row["floor_eur"]
            for row in rows
            if row["merchant_eur"] < 0.0
        )
        assert result["merchant_pv_eur"] == -130.0
        assert result["floor_protected_pv_eur"] == 80.0
        assert result["floor_pv_uplift_eur"] == 210.0
        assert result["crossover_year"] == 1
        assert result["n_binding_years"] == 4

    @pytest.mark.parametrize(
        ("decay", "escalation"),
        [(0.2, 0.0), (0.2, 0.1), (0.0, 0.1)],
    )
    def test_binding_flags_are_non_decreasing(
        self, decay: float, escalation: float
    ) -> None:
        result = compute_decaying_contracted_floor_overlay(
            **{**_DECAY_BASE, "floor_tenor_years": 6.0},
            annual_decay_rate=decay,
            floor_escalation_rate=escalation,
        )
        flags = [bool(row["binding"]) for row in result["per_year"]]
        assert flags == sorted(flags)
        assert result["n_binding_years"] == sum(flags)
        expected_crossover = next(
            (index for index, flag in enumerate(flags, start=1) if flag),
            None,
        )
        assert result["crossover_year"] == expected_crossover


class TestDecayingContractedFloorPV:
    @pytest.mark.parametrize("discount_rate", [0.0, 0.08])
    def test_pv_identities_match_public_decay_factor(
        self, discount_rate: float
    ) -> None:
        tenor = 2.5
        decay = 0.2
        floor = 0.1
        result = compute_decaying_contracted_floor_overlay(
            **{
                **_DECAY_BASE,
                "floor_tenor_years": tenor,
                "discount_rate": discount_rate,
            },
            annual_decay_rate=decay,
            decay_floor_share=floor,
        )
        flat_factor = scenario.annuity_pv_factor(tenor, discount_rate)
        decay_factor = scenario.decaying_annuity_pv_factor(
            tenor,
            discount_rate,
            decay,
            floor,
        )
        expected_merchant = 80.0 * flat_factor + 100.0 * (
            decay_factor - flat_factor
        )
        assert result["merchant_pv_eur"] == pytest.approx(expected_merchant)
        assert (
            result["floor_protected_pv_eur"] - result["merchant_pv_eur"]
        ) == pytest.approx(result["floor_pv_uplift_eur"])


class TestDecayingContractedFloorValidation:
    @pytest.mark.parametrize(
        ("field", "bad"),
        [
            ("annual_decay_rate", -0.01),
            ("annual_decay_rate", 1.0),
            ("annual_decay_rate", np.nan),
            ("annual_decay_rate", np.inf),
            ("decay_floor_share", -0.01),
            ("decay_floor_share", 1.01),
            ("decay_floor_share", np.nan),
            ("decay_floor_share", np.inf),
            ("floor_escalation_rate", -0.01),
            ("floor_escalation_rate", 1.01),
            ("floor_escalation_rate", np.nan),
            ("floor_escalation_rate", np.inf),
            ("merchant_gross_eur_per_mw_yr", -0.01),
            ("merchant_gross_eur_per_mw_yr", np.nan),
            ("merchant_gross_eur_per_mw_yr", np.inf),
        ],
    )
    def test_new_domains_raise_before_inactive_delegation(
        self, field: str, bad: float
    ) -> None:
        with pytest.raises(ValueError, match=field):
            compute_decaying_contracted_floor_overlay(
                **{**_DECAY_BASE, field: bad}
            )

    def test_gross_below_net_raises_with_field_name(self) -> None:
        with pytest.raises(ValueError, match="merchant_gross_eur_per_mw_yr"):
            compute_decaying_contracted_floor_overlay(
                **{
                    **_DECAY_BASE,
                    "merchant_gross_eur_per_mw_yr": 79.99,
                }
            )

    def test_active_trajectory_budget_accepts_exact_limit(self) -> None:
        result = compute_decaying_contracted_floor_overlay(
            **{
                **_DECAY_BASE,
                "floor_tenor_years": MAX_FLOOR_TRAJECTORY_YEARS,
            },
            annual_decay_rate=0.01,
        )
        assert len(result["per_year"]) == 100

    def test_active_trajectory_budget_rejects_before_delegate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            contracted_floor,
            "compute_contracted_floor_overlay",
            lambda **_: pytest.fail("active over-budget tenor reached v1 delegate"),
        )
        with pytest.raises(ValueError, match="floor_tenor_years"):
            compute_decaying_contracted_floor_overlay(
                **{
                    **_DECAY_BASE,
                    "floor_tenor_years": MAX_FLOOR_TRAJECTORY_YEARS + 0.5,
                },
                annual_decay_rate=0.01,
            )

    def test_inactive_tenor_above_trajectory_budget_still_delegates(self) -> None:
        tenor = MAX_FLOOR_TRAJECTORY_YEARS + 0.5
        result = compute_decaying_contracted_floor_overlay(
            **{**_DECAY_BASE, "floor_tenor_years": tenor}
        )
        expected = compute_contracted_floor_overlay(
            **{
                key: value
                for key, value in {
                    **_DECAY_BASE,
                    "floor_tenor_years": tenor,
                }.items()
                if key != "merchant_gross_eur_per_mw_yr"
            }
        )
        for key in _V1_KEYS:
            assert result[key] == expected[key]
        assert result["per_year"] == []

    def test_power_scaling_and_input_purity(self) -> None:
        kwargs = {**_DECAY_BASE, "annual_decay_rate": 0.2}
        before = dict(kwargs)
        base = compute_decaying_contracted_floor_overlay(**kwargs)
        scaled = compute_decaying_contracted_floor_overlay(
            **{**kwargs, "power_mw": 4.0}
        )
        assert kwargs == before
        top_level_eur = (
            "merchant_net_eur",
            "quoted_floor_eur",
            "effective_floor_eur",
            "floor_protected_cashflow_eur",
            "annual_top_up_eur",
            "merchant_pv_eur",
            "floor_protected_pv_eur",
            "floor_pv_uplift_eur",
        )
        for key in top_level_eur:
            assert scaled[key] == pytest.approx(4.0 * base[key])
        for base_row, scaled_row in zip(
            base["per_year"], scaled["per_year"], strict=True
        ):
            for key in ("merchant_eur", "floor_eur", "protected_eur", "top_up_eur"):
                assert scaled_row[key] == pytest.approx(4.0 * base_row[key])

    def test_composition_core_remains_pandas_and_solver_free(self) -> None:
        source = inspect.getsource(contracted_floor)
        assert "import pandas" not in source
        assert "import numpy" not in source
        assert "src.dispatch" not in source
        assert "solve_" not in source
