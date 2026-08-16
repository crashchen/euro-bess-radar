"""PC-D2 annual strategy-cash floor settlement and valuation contract tests."""

from __future__ import annotations

import dataclasses as dc

import numpy as np
import pytest

from src.project_case.enums import (
    CONTRACT_ASSET_SCOPE_V1,
    CONTRACT_QUOTE_BASIS_V1,
    CONTRACT_SETTLEMENT_FREQUENCY_V1,
    CapacityMaintenanceBasis,
    ContractCurrencyBasisMode,
    ContractQuoteStatus,
    ContractSettlementBasis,
)
from src.project_case.schema import (
    AnnualPreLifecycleStrategyCashFloor,
    ContractCase,
    ContractCurrencyBasis,
    ProjectCase,
    ProjectCaseValidationError,
)
from src.project_case.valuation import (
    _discount_factors,
    _lifecycle_adjustment,
    _projection_multipliers,
    _rank_interpolated_representative,
    _screening_npv_draws,
    _screening_npv_draws_from_settled,
    _settled_revenue_matrix,
    _summarise_npv_draws,
    compute_project_case,
)
from tests.test_project_case_valuation import _case, _distribution


def _with_floor(
    case: ProjectCase,
    *,
    start_year: int = 1,
    rates: tuple[float, ...] = (30.0,),
    factors: tuple[float, ...] = (1.0,),
) -> ProjectCase:
    terms = AnnualPreLifecycleStrategyCashFloor(
        contract_start_project_year=start_year,
        floor_rate_real_eur_per_modeled_mw_year_by_contract_year=rates,
        floor_entitlement_factor_by_contract_year=factors,
        quote_basis=CONTRACT_QUOTE_BASIS_V1,
        settlement_frequency=CONTRACT_SETTLEMENT_FREQUENCY_V1,
        asset_scope=CONTRACT_ASSET_SCOPE_V1,
        currency_basis=ContractCurrencyBasis(
            ContractCurrencyBasisMode.USER_ASSERTED_REAL_BASE_YEAR_EUR_CURVE,
            case.valuation_case.base_year,
        ),
        quote_status=ContractQuoteStatus.USER_SCENARIO,
        source="PC-D2 test scenario",
        source_as_of_date="2026-08-16",
        source_document_sha256=None,
    )
    return dc.replace(
        case,
        contract_case=ContractCase(
            ContractSettlementBasis.ANNUAL_PRE_LIFECYCLE_STRATEGY_CASH_FLOOR_V1,
            terms,
        ),
    )


def _fake_annual_draws(monkeypatch, draws: np.ndarray) -> None:
    values = np.asarray(draws, dtype=np.float64)

    def fake_bootstrap(_daily, *, seed, n_simulations):
        assert n_simulations == values.size
        return values.copy()

    monkeypatch.setattr(
        "src.project_case.valuation.bootstrap_annual_sums",
        fake_bootstrap,
    )
    # RunResult independently reconstructs the locked bootstrap at its public
    # validation boundary. Keep the test oracle identical on both sides.
    monkeypatch.setattr(
        "src.project_case.schema.bootstrap_annual_sums",
        fake_bootstrap,
    )


def test_golden_vector_a_settles_each_draw_before_linear_percentiles():
    annual = np.asarray([0.0, 100.0])
    covered = np.asarray([True])
    floors = np.asarray([60.0])
    merchant, settled, top_up = _settled_revenue_matrix(
        annual, np.asarray([1.0]), covered, floors
    )
    np.testing.assert_array_equal(merchant[:, 0], [0.0, 100.0])
    np.testing.assert_array_equal(settled[:, 0], [60.0, 100.0])
    np.testing.assert_array_equal(top_up[:, 0], [60.0, 0.0])
    npv = _screening_npv_draws_from_settled(settled, np.ones(1), 70.0)
    np.testing.assert_array_equal(npv, [-10.0, 30.0])
    dist = _summarise_npv_draws(npv)
    assert (dist.p10, dist.p50, dist.p90, dist.prob_positive) == pytest.approx(
        (-6.0, 10.0, 26.0, 0.5)
    )
    merchant_star, settled_star, top_up_star, interpolation = (
        _rank_interpolated_representative(annual, merchant, settled)
    )
    np.testing.assert_array_equal(merchant_star, [50.0])
    np.testing.assert_array_equal(settled_star, [80.0])
    np.testing.assert_array_equal(top_up_star, [30.0])
    assert interpolation == {
        "lower_sorted_rank": 0,
        "upper_sorted_rank": 1,
        "lower_original_draw_index": 0,
        "upper_original_draw_index": 1,
        "interpolation_weight": 0.5,
    }


def test_golden_vector_b_term_projection_and_lifecycle_identity():
    annual = np.asarray([40.0, 80.0, 120.0])
    multipliers = np.asarray([1.0, 0.5, 0.25])
    covered = np.asarray([True, True, False])
    floors = np.asarray([60.0, 50.0, 0.0])
    _merchant, settled, _top_up = _settled_revenue_matrix(
        annual, multipliers, covered, floors
    )
    np.testing.assert_array_equal(
        settled,
        [[60.0, 50.0, 10.0], [80.0, 50.0, 20.0], [120.0, 60.0, 30.0]],
    )
    screening = _screening_npv_draws_from_settled(settled, np.ones(3), 10.0)
    np.testing.assert_array_equal(screening, [110.0, 140.0, 200.0])
    screen_dist = _summarise_npv_draws(screening)
    assert (screen_dist.p10, screen_dist.p50, screen_dist.p90) == pytest.approx(
        (116.0, 140.0, 188.0)
    )
    adjustment = _lifecycle_adjustment(
        fixed_om_eur=5.0,
        augmentation_eur=np.asarray([0.0, 20.0, 0.0]),
        terminal_eur=np.asarray([0.0, 0.0, 7.0]),
        discount_factors=np.ones(3),
    )
    assert adjustment == -28.0
    lifecycle = screening + adjustment
    np.testing.assert_array_equal(lifecycle, [82.0, 112.0, 172.0])
    life_dist = _summarise_npv_draws(lifecycle)
    assert (life_dist.p10, life_dist.p50, life_dist.p90) == pytest.approx(
        (88.0, 112.0, 160.0)
    )
    np.testing.assert_array_equal(lifecycle - screening, [-28.0, -28.0, -28.0])


def test_golden_vector_c_absent_floor_is_not_zero_floor():
    merchant, settled, top_up = _settled_revenue_matrix(
        np.asarray([-20.0]),
        np.ones(2),
        np.asarray([True, False]),
        np.asarray([0.0, 0.0]),
    )
    np.testing.assert_array_equal(merchant, [[-20.0, -20.0]])
    np.testing.assert_array_equal(settled, [[0.0, -20.0]])
    np.testing.assert_array_equal(top_up, [[20.0, 0.0]])
    assert _screening_npv_draws_from_settled(settled, np.ones(2), 0.0)[0] == -20.0


def test_golden_vector_d_delayed_start_mw_factor_and_discounting(monkeypatch):
    annual = np.full(1000, 5.0)
    _fake_annual_draws(monkeypatch, annual)
    case = _with_floor(
        _case(power_mw=2.0, life_years=4, discount_rate=0.10),
        start_year=2,
        rates=(10.0, 20.0),
        factors=(0.5, 1.0),
    )
    result = compute_project_case(case)
    assert _distribution(result).p50 == pytest.approx(46.27757666826035)
    rows = result.screening_cashflow_table.rows
    assert [row.effective_contract_floor_eur for row in rows] == [
        None,
        10.0,
        40.0,
        None,
    ]
    assert [row.merchant_revenue_eur for row in rows] == [5.0] * 4
    assert [row.revenue_eur for row in rows] == [5.0, 10.0, 40.0, 5.0]
    assert [row.contract_top_up_eur for row in rows] == [0.0, 5.0, 35.0, 0.0]


def test_public_nonnull_path_rank_interpolates_and_reconciles(monkeypatch):
    annual = np.concatenate((np.zeros(500), np.full(500, 100.0)))
    _fake_annual_draws(monkeypatch, annual)
    case = _with_floor(_case(capex_eur=70.0, life_years=1))
    result = compute_project_case(case)
    dist = _distribution(result)
    assert dist.p50 == 10.0
    assert dist.prob_positive == 0.5
    row = result.screening_cashflow_table.rows[0]
    assert row.merchant_revenue_eur == 50.0
    assert row.effective_contract_floor_eur == 60.0
    assert row.contract_top_up_eur == 30.0
    assert row.revenue_eur == 80.0
    assert -case.asset_case.installed_capex_eur + row.discounted_net_eur == dist.p50


def test_null_contract_keeps_pc_b_numeric_path_bit_identical(monkeypatch):
    annual = np.linspace(-123.4, 987.6, 1000, dtype=np.float64)
    _fake_annual_draws(monkeypatch, annual)
    case = _case(
        capex_eur=1234.5,
        life_years=3,
        discount_rate=0.08,
    )
    result = compute_project_case(case)
    multipliers = _projection_multipliers(case)
    discounts = _discount_factors(case)
    expected_draws = _screening_npv_draws(
        annual,
        multipliers,
        discounts,
        case.asset_case.installed_capex_eur,
    )
    expected = _summarise_npv_draws(expected_draws)
    actual = _distribution(result)
    assert actual.to_payload() == expected.to_payload()

    annual_p50 = float(np.percentile(annual, 50.0, method="linear"))
    for index, row in enumerate(result.screening_cashflow_table.rows):
        revenue = annual_p50 * float(multipliers[index])
        discounted = revenue * float(discounts[index])
        assert row.merchant_revenue_eur.hex() == revenue.hex()
        assert row.revenue_eur.hex() == revenue.hex()
        assert row.net_eur.hex() == revenue.hex()
        assert row.discounted_net_eur.hex() == discounted.hex()
        assert row.effective_contract_floor_eur is None
        assert row.contract_top_up_eur == 0.0


def test_nonnull_unknown_keeps_screening_and_exact_lifecycle_unavailable(monkeypatch):
    _fake_annual_draws(monkeypatch, np.full(1000, -20.0))
    case = _with_floor(
        _case(
            life_years=2,
            basis=CapacityMaintenanceBasis.UNKNOWN,
        ),
        rates=(0.0,),
    )
    result = compute_project_case(case)
    assert _distribution(result).p50 == -20.0
    assert result.no_lifecycle_cost_screening_npv.available
    assert not result.lifecycle_cash_npv.available
    assert result.lifecycle_cash_npv.status == "capacity_maintenance_unknown"
    assert result.lifecycle_cash_npv.message == (
        "Engineering capacity-maintenance basis is unknown."
    )
    assert result.lifecycle_cash_npv.distribution is None
    assert result.lifecycle_cashflow_table is None


def test_contract_provenance_is_exact_and_floor_comparator_independent(monkeypatch):
    _fake_annual_draws(monkeypatch, np.concatenate((np.zeros(500), np.ones(500))))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("wear-net comparator must not enter Project Case")

    monkeypatch.setattr(
        "src.contracted_floor.compute_contracted_floor_overlay", forbidden
    )
    monkeypatch.setattr(
        "src.contracted_floor.compute_decaying_contracted_floor_overlay", forbidden
    )
    case = _with_floor(_case(life_years=1), rates=(0.25,))
    result = compute_project_case(case)
    provenance = result.to_payload()["provenance"]
    assert set(provenance) == {
        "calculator_version",
        "project_case_input_fingerprint",
        "strategy_run_fingerprint",
        "project_case",
        "strategy_run_result",
        "bootstrap",
        "projection",
        "valuation",
        "capacity_maintenance_basis",
        "contract_settlement",
        "cashflow_table_statistic",
        "cashflow_reconciliation",
        "red_line_assertions",
    }
    assert provenance["calculator_version"] == "pc-d2-v1.1"
    assert provenance["cashflow_table_statistic"] == (
        "p50_npv_rank_interpolated_cashflow_linear_v1"
    )
    assert provenance["cashflow_reconciliation"] == {
        "version": "pc-cashflow-p50-reconciliation-v1",
        "relative_tolerance": 1e-10,
        "absolute_tolerance_eur": 1e-6,
    }
    settlement = provenance["contract_settlement"]
    assert settlement["basis"] == (
        "ANNUAL_PRE_LIFECYCLE_STRATEGY_CASH_FLOOR_V1"
    )
    assert settlement["algorithm_version"] == (
        "pc-annual-pre-lifecycle-strategy-cash-floor-v1"
    )
    assert settlement["resolved_floor_by_project_year"] == [
        {"year": 1, "effective_floor_eur": 0.5}
    ]
    assert settlement["representative_interpolation"] == {
        "lower_sorted_rank": 499,
        "upper_sorted_rank": 500,
        "lower_original_draw_index": 499,
        "upper_original_draw_index": 500,
        "interpolation_weight": 0.5,
    }
    assert provenance["red_line_assertions"] == {
        "cash_npv_includes_shadow_wear": False,
        "vom_rededucted": False,
        "mw_rescaled": False,
        "wear_net_floor_comparator_included": False,
        "contract_settlement_included": True,
        "contract_settlement_basis": (
            "ANNUAL_PRE_LIFECYCLE_STRATEGY_CASH_FLOOR_V1"
        ),
        "pre_tax_unlevered": True,
        "tax_included": False,
        "debt_included": False,
        "financing_fees_included": False,
    }


def test_effective_floor_overflow_fails_before_runresult(monkeypatch):
    _fake_annual_draws(monkeypatch, np.zeros(1000))
    case = _with_floor(
        _case(power_mw=2.0, life_years=1),
        rates=(1e308,),
    )
    with pytest.raises(ProjectCaseValidationError, match="effective contract floors"):
        compute_project_case(case)
