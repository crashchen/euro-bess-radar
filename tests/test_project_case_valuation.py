"""PC-B lifecycle cash-flow and NPV contract tests (§3, §6, §9)."""

from __future__ import annotations

import dataclasses as dc
import datetime as dt

import numpy as np
import pytest

from src import scenario
from src.project_case import (
    AssetCase,
    AugmentationEvent,
    BootstrapCase,
    CapacityMaintenanceBasis,
    CoverageAudit,
    LifecycleCase,
    MarketCase,
    ProjectCase,
    ProjectCaseValidationError,
    Projection,
    ProjectionKind,
    SampleWindow,
    ValuationCase,
    compute_project_case,
)
from src.project_case.enums import (
    BOOTSTRAP_ALGORITHM_V1,
    LIFECYCLE_UNKNOWN_MESSAGE,
    LIFECYCLE_UNKNOWN_STATUS,
)
from src.project_case.schema import _issue_strategy_run_result
from src.project_case.valuation import (
    _screening_npv_draws,
    _summarise_npv_draws,
)
from tests import pc_case_fixtures as fx


def _strategy_with_daily(
    values: tuple[float, ...],
    *,
    power_mw: float = 2.0,
    duration_hours: float = 2.0,
    efficiency: float = 0.88,
):
    dates = tuple(fx.D1 + dt.timedelta(days=i) for i in range(len(values)))
    with _issue_strategy_run_result():
        return dc.replace(
            fx.da_only_srr(),
            daily_realised_cash_series=tuple(zip(dates, values, strict=True)),
            power_mw=power_mw,
            duration_hours=duration_hours,
            round_trip_efficiency=efficiency,
            sample_window=SampleWindow(dates[0], dates[-1], fx.TZ),
            coverage_audit=CoverageAudit(dates, dates, (), (), ()),
        )


def _case(
    values: tuple[float, ...] = (1.0, 1.0),
    *,
    power_mw: float = 2.0,
    capex_eur: float = 0.0,
    fixed_om_eur_per_mw_yr: float = 0.0,
    life_years: int = 1,
    discount_rate: float = 0.0,
    projection: Projection | None = None,
    basis: CapacityMaintenanceBasis = (
        CapacityMaintenanceBasis.NO_AUGMENTATION_REQUIRED_ASSERTED
    ),
    events: tuple[AugmentationEvent, ...] = (),
    eol_residual_eur: float = 0.0,
    decommissioning_eur: float = 0.0,
    seed: int = 12345,
    n_simulations: int = 1000,
) -> ProjectCase:
    strategy = _strategy_with_daily(values, power_mw=power_mw)
    source = None if basis is CapacityMaintenanceBasis.UNKNOWN else "engineering_source"
    as_of = None if basis is CapacityMaintenanceBasis.UNKNOWN else "2026-01-01"
    return ProjectCase(
        asset_case=AssetCase(
            power_mw=power_mw,
            duration_hours=2.0,
            round_trip_efficiency=0.88,
            installed_capex_eur=capex_eur,
            fixed_om_eur_per_mw_yr=fixed_om_eur_per_mw_yr,
        ),
        lifecycle_case=LifecycleCase(
            project_life_years=life_years,
            capacity_maintenance_basis=basis,
            capacity_maintenance_source=source,
            capacity_maintenance_as_of=as_of,
            augmentation_events=events,
            eol_residual_value_eur=eol_residual_eur,
            decommissioning_cost_eur=decommissioning_eur,
        ),
        market_case=MarketCase(
            strategy,
            projection or Projection(ProjectionKind.FlatRealProjection),
        ),
        valuation_case=ValuationCase(discount_rate, 2026),
        bootstrap_case=BootstrapCase(
            seed,
            n_simulations,
            BOOTSTRAP_ALGORITHM_V1,
        ),
    )


def _distribution(result, slot: str = "screening"):
    outcome = (
        result.no_lifecycle_cost_screening_npv
        if slot == "screening"
        else result.lifecycle_cash_npv
    )
    assert outcome.distribution is not None
    return outcome.distribution


def test_pc_b_bootstrap_golden_vector_reaches_public_result():
    result = compute_project_case(
        _case((100.0, 50.0, 200.0, -30.0, 75.0), life_years=1)
    )
    dist = _distribution(result)
    assert (dist.p10, dist.p50, dist.p90) == pytest.approx(
        (26_969.5, 28_840.0, 30_745.5)
    )
    assert dist.prob_positive == 1.0


def test_non_geometric_lifecycle_known_answer_and_p50_table_reconciliation():
    event = AugmentationEvent(2, 120.0, 0.25, 5.0)
    result = compute_project_case(
        _case(
            (1.0, 1.0),
            power_mw=2.0,
            capex_eur=1_000.0,
            fixed_om_eur_per_mw_yr=10.0,
            life_years=3,
            discount_rate=0.10,
            projection=Projection(
                ProjectionKind.ExplicitAnnualMultiplierCurve,
                multipliers=(1.0, 0.5, 0.0),
                source="explicit_curve",
                as_of="2026-01-01",
            ),
            basis=CapacityMaintenanceBasis.SCHEDULED_NAMEPLATE_MAINTENANCE,
            events=(event,),
            eol_residual_eur=100.0,
            decommissioning_eur=40.0,
        )
    )

    screening = _distribution(result)
    lifecycle = _distribution(result, "lifecycle")
    assert screening.p50 == pytest.approx(-517.3553719008265)
    assert lifecycle.p50 == pytest.approx(-617.0548459804659)
    assert lifecycle.p50 - screening.p50 == pytest.approx(-99.69947407963934)

    assert result.screening_cashflow_table is not None
    assert result.lifecycle_cashflow_table is not None
    screen_rows = result.screening_cashflow_table.rows
    life_rows = result.lifecycle_cashflow_table.rows
    assert [row.net_eur for row in screen_rows] == pytest.approx([365.0, 182.5, 0.0])
    assert [row.net_eur for row in life_rows] == pytest.approx([345.0, 47.5, 40.0])
    assert all(
        row.opex_eur == row.augmentation_eur == row.terminal_eur == 0.0
        for row in screen_rows
    )
    assert -1_000.0 + sum(row.discounted_net_eur for row in screen_rows) == pytest.approx(
        screening.p50
    )
    assert -1_000.0 + sum(row.discounted_net_eur for row in life_rows) == pytest.approx(
        lifecycle.p50
    )


def test_spread_decay_resolves_year_weights_with_floor():
    result = compute_project_case(
        _case(
            life_years=4,
            projection=Projection(ProjectionKind.DAOnlySpreadDecay, 0.5, 0.4),
        )
    )
    assert result.provenance["projection"]["resolved_annual_multipliers"] == (
        1.0,
        0.5,
        0.4,
        0.4,
    )
    assert _distribution(result).p50 == pytest.approx(839.5)


def test_unknown_is_screening_only_and_does_not_derive_lifecycle_cash():
    # The individually finite O&M inputs overflow if multiplied. UNKNOWN must not
    # derive that unavailable lifecycle layer; screening remains computable.
    result = compute_project_case(
        _case(
            power_mw=1e308,
            fixed_om_eur_per_mw_yr=1e308,
            basis=CapacityMaintenanceBasis.UNKNOWN,
        )
    )
    assert result.no_lifecycle_cost_screening_npv.available
    assert _distribution(result).p50 == pytest.approx(365.0)
    assert not result.lifecycle_cash_npv.available
    assert result.lifecycle_cash_npv.status == LIFECYCLE_UNKNOWN_STATUS
    assert result.lifecycle_cash_npv.message == LIFECYCLE_UNKNOWN_MESSAGE
    assert result.lifecycle_cash_npv.distribution is None
    assert result.lifecycle_cashflow_table is None


def test_unknown_does_not_underwrite_an_existing_unasserted_schedule():
    result = compute_project_case(
        _case(
            capex_eur=100.0,
            fixed_om_eur_per_mw_yr=50.0,
            life_years=2,
            basis=CapacityMaintenanceBasis.UNKNOWN,
            events=(AugmentationEvent(1, 1_000.0, 0.5, 100.0),),
            eol_residual_eur=500.0,
            decommissioning_eur=200.0,
        )
    )
    assert _distribution(result).p50 == pytest.approx(630.0)
    assert not result.lifecycle_cash_npv.available
    assert result.lifecycle_cashflow_table is None
    assert all(
        row.opex_eur == row.augmentation_eur == row.terminal_eur == 0.0
        for row in result.screening_cashflow_table.rows
    )


def test_screening_excludes_om_and_daily_cash_is_not_rescaled_by_mw():
    no_om = compute_project_case(_case(power_mw=10.0, life_years=2))
    with_om = compute_project_case(
        _case(
            power_mw=10.0,
            fixed_om_eur_per_mw_yr=5.0,
            life_years=2,
        )
    )
    assert _distribution(no_om).p50 == pytest.approx(730.0)
    assert _distribution(with_om).p50 == _distribution(no_om).p50
    assert _distribution(with_om, "lifecycle").p50 == pytest.approx(630.0)


def test_event_salvage_can_exceed_cost_and_terminal_is_last_year_only():
    result = compute_project_case(
        _case(
            life_years=2,
            basis=CapacityMaintenanceBasis.SCHEDULED_NAMEPLATE_MAINTENANCE,
            events=(AugmentationEvent(2, 20.0, 0.1, 50.0),),
            eol_residual_eur=25.0,
            decommissioning_eur=5.0,
        )
    )
    rows = result.lifecycle_cashflow_table.rows
    assert [row.augmentation_eur for row in rows] == [0.0, -30.0]
    assert [row.terminal_eur for row in rows] == [0.0, 20.0]
    assert [row.net_eur for row in rows] == pytest.approx([365.0, 415.0])


def test_one_bootstrap_draw_is_reused_across_all_years(monkeypatch):
    annual_draws = np.linspace(0.0, 999.0, 1000)
    calls = []

    def fake_bootstrap(values, *, seed, n_simulations):
        calls.append((values.copy(), seed, n_simulations))
        return annual_draws.copy()

    monkeypatch.setattr(
        "src.project_case.valuation.bootstrap_annual_sums",
        fake_bootstrap,
    )
    monkeypatch.setattr(
        "src.project_case.schema.bootstrap_annual_sums",
        lambda _values, *, seed, n_simulations: annual_draws.copy(),
    )
    result = compute_project_case(
        _case(
            life_years=2,
            projection=Projection(
                ProjectionKind.ExplicitAnnualMultiplierCurve,
                multipliers=(1.0, 0.5),
                source="curve",
                as_of="2026-01-01",
            ),
        )
    )
    assert len(calls) == 1
    assert _distribution(result).p50 == pytest.approx(499.5 * 1.5)
    assert _distribution(result).p10 == pytest.approx(99.9 * 1.5)
    assert _distribution(result).p90 == pytest.approx(899.1 * 1.5)


def test_cashflow_table_uses_p50_not_mean_for_a_skewed_bootstrap(monkeypatch):
    annual_draws = np.zeros(1000)
    annual_draws[-1] = 1_000_000.0  # P50=0, mean=1,000.

    def fake_bootstrap(_values, *, seed, n_simulations):
        assert n_simulations == len(annual_draws)
        return annual_draws.copy()

    monkeypatch.setattr(
        "src.project_case.valuation.bootstrap_annual_sums",
        fake_bootstrap,
    )
    monkeypatch.setattr(
        "src.project_case.schema.bootstrap_annual_sums",
        fake_bootstrap,
    )
    result = compute_project_case(_case(life_years=2))
    assert all(row.revenue_eur == 0.0 for row in result.screening_cashflow_table.rows)
    assert _distribution(result).p50 == 0.0
    assert sum(
        row.discounted_net_eur for row in result.screening_cashflow_table.rows
    ) == _distribution(result).p50


def test_lifecycle_distribution_is_the_same_per_draw_adjustment(monkeypatch):
    annual_draws = np.linspace(-10.0, 10.0, 1000)

    def fake_bootstrap(_values, *, seed, n_simulations):
        return annual_draws.copy()

    monkeypatch.setattr(
        "src.project_case.valuation.bootstrap_annual_sums",
        fake_bootstrap,
    )
    monkeypatch.setattr(
        "src.project_case.schema.bootstrap_annual_sums",
        fake_bootstrap,
    )
    result = compute_project_case(
        _case(power_mw=2.0, fixed_om_eur_per_mw_yr=1.0, life_years=1)
    )
    screening = _distribution(result)
    lifecycle = _distribution(result, "lifecycle")
    for name in ("p10", "p50", "p90"):
        assert getattr(lifecycle, name) - getattr(screening, name) == pytest.approx(-2.0)
    # Screening has half the draws above zero; after the €2 lifecycle cost only
    # draws with annual revenue > 2 remain positive.
    assert screening.prob_positive == pytest.approx(0.5)
    assert lifecycle.prob_positive == pytest.approx(float(np.mean(annual_draws > 2.0)))


def test_probability_is_strictly_positive_and_linear_percentiles_use_all_draws():
    dist = _summarise_npv_draws(np.asarray([-1.0, 0.0, 1.0]))
    assert dist.prob_positive == pytest.approx(1.0 / 3.0)
    assert (dist.p10, dist.p50, dist.p90) == pytest.approx((-0.8, 0.0, 0.8))


def test_screening_kernel_matches_legacy_primitive_on_identical_annual_draws():
    annual = np.asarray([-100.0, 0.0, 250.0, 500.0])
    multipliers = np.asarray([1.0, 0.8, 0.64, 0.512])
    discounts = np.asarray([1.08**-year for year in range(1, 5)])
    actual = _screening_npv_draws(annual, multipliers, discounts, 50.0)
    legacy = scenario.calculate_npv_distribution(
        annual,
        total_capex=50.0,
        annual_degradation_cost=0.0,
        effective_life_years=4,
        discount_rate=0.08,
        annual_decay_rate=0.2,
        decay_floor_share=0.5,
    )["npv_array"]
    assert actual == pytest.approx(legacy)


def test_all_zero_and_negative_series_are_valid_market_outcomes():
    zero = compute_project_case(_case((0.0, 0.0), capex_eur=0.0))
    negative = compute_project_case(_case((-1.0, -2.0), capex_eur=0.0))
    assert _distribution(zero).p50 == 0.0
    assert _distribution(zero).prob_positive == 0.0
    assert _distribution(negative).p90 < 0.0
    assert _distribution(negative).prob_positive == 0.0


def test_nonfinite_derived_bootstrap_cash_fails_closed():
    with pytest.raises(ProjectCaseValidationError, match="annual bootstrap draws"):
        compute_project_case(_case((1e308, 1e308)))


def test_nonfinite_active_lifecycle_cash_fails_closed():
    with pytest.raises(ProjectCaseValidationError, match="fixed O&M"):
        compute_project_case(
            _case(
                power_mw=1e308,
                fixed_om_eur_per_mw_yr=1e308,
                basis=CapacityMaintenanceBasis.NO_AUGMENTATION_REQUIRED_ASSERTED,
            )
        )


def test_result_fingerprint_provenance_and_red_line_assertions():
    case = _case(life_years=2)
    result = compute_project_case(case)
    assert result.input_fingerprint == case.input_fingerprint()
    assert result.schema_version == "project-case-v1.1"
    assert result.provenance["calculator_version"] == "pc-d2-v1.1"
    assert result.provenance["project_case_input_fingerprint"] == case.input_fingerprint()
    assert result.provenance["strategy_run_fingerprint"] == (
        case.market_case.strategy_run_result.fingerprint()
    )
    payload = result.to_payload()["provenance"]
    assert payload["project_case"] == case.to_payload()
    assert payload["strategy_run_result"] == case.market_case.strategy_run_result.to_payload()
    assert result.provenance["cashflow_table_statistic"] == (
        "p50_annual_bootstrap_draw_linear"
    )
    assert result.provenance["contract_settlement"] == {
        "basis": None,
        "algorithm_version": None,
        "resolved_floor_by_project_year": (),
        "representative_interpolation": None,
    }
    assert result.provenance["cashflow_reconciliation"] == {
        "version": "pc-cashflow-p50-reconciliation-v1",
        "relative_tolerance": 1e-10,
        "absolute_tolerance_eur": 1e-6,
    }
    assertions = result.provenance["red_line_assertions"]
    assert assertions == {
        "cash_npv_includes_shadow_wear": False,
        "vom_rededucted": False,
        "mw_rescaled": False,
        "wear_net_floor_comparator_included": False,
        "contract_settlement_included": False,
        "contract_settlement_basis": None,
        "pre_tax_unlevered": True,
        "tax_included": False,
        "debt_included": False,
        "financing_fees_included": False,
    }
    assert payload["bootstrap"]["seed"] == 12345
    bootstrap = result.provenance["bootstrap"]
    assert bootstrap == {
        "algorithm_version": BOOTSTRAP_ALGORITHM_V1,
        "seed": 12345,
        "n_simulations": 1000,
        "days_per_annual_draw": 365,
        "percentile_method": "linear",
        "prob_positive_rule": "strict_gt_zero_all_draws",
    }
    assert result.provenance["valuation"]["cash_timing"] == "end_of_year"


def test_cashflow_rows_pin_each_years_arithmetic_and_discount_factor():
    result = compute_project_case(
        _case(
            capex_eur=100.0,
            fixed_om_eur_per_mw_yr=3.0,
            life_years=3,
            discount_rate=0.10,
            basis=CapacityMaintenanceBasis.SCHEDULED_NAMEPLATE_MAINTENANCE,
            events=(AugmentationEvent(2, 20.0, 0.2, 5.0),),
            eol_residual_eur=10.0,
            decommissioning_eur=2.0,
        )
    )
    for table in (result.screening_cashflow_table, result.lifecycle_cashflow_table):
        assert [row.year for row in table.rows] == [1, 2, 3]
        for row in table.rows:
            assert row.discount_factor == pytest.approx(1.10**-row.year)
            assert row.net_eur == pytest.approx(
                row.revenue_eur
                - row.opex_eur
                - row.augmentation_eur
                + row.terminal_eur
            )
            assert row.discounted_net_eur == pytest.approx(
                row.net_eur * row.discount_factor
            )


def test_same_year_events_sum_and_restoration_fraction_is_not_a_multiplier():
    events_a = (
        AugmentationEvent(2, 20.0, 0.1, 5.0),
        AugmentationEvent(2, 30.0, 0.2, 10.0),
    )
    events_b = (
        AugmentationEvent(2, 30.0, 0.9, 10.0),
        AugmentationEvent(2, 20.0, 0.8, 5.0),
    )
    case_a = _case(
        life_years=3,
        basis=CapacityMaintenanceBasis.SCHEDULED_NAMEPLATE_MAINTENANCE,
        events=events_a,
    )
    case_b = _case(
        life_years=3,
        basis=CapacityMaintenanceBasis.SCHEDULED_NAMEPLATE_MAINTENANCE,
        events=events_b,
    )
    result_a = compute_project_case(case_a)
    result_b = compute_project_case(case_b)
    assert result_a.lifecycle_cashflow_table.rows[1].augmentation_eur == 35.0
    assert result_b.lifecycle_cashflow_table.rows[1].augmentation_eur == 35.0
    assert result_a.lifecycle_cash_npv.to_payload() == result_b.lifecycle_cash_npv.to_payload()
    assert result_a.input_fingerprint != result_b.input_fingerprint


def test_negative_discount_rate_is_applied_year_by_year():
    result = compute_project_case(_case(life_years=2, discount_rate=-0.5))
    assert [row.discount_factor for row in result.screening_cashflow_table.rows] == [
        2.0,
        4.0,
    ]
    assert _distribution(result).p50 == pytest.approx(365.0 * 6.0)
