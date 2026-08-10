"""Schema domain + cross-invariant validation (contract §4, §7 red-lines)."""

from __future__ import annotations

import dataclasses as dc

import pytest

from src.project_case import (
    AssetCase,
    AugmentationEvent,
    BootstrapCase,
    CapacityMaintenanceBasis,
    CaptureBasis,
    CashBasis,
    CoverageAudit,
    CurrencyBasis,
    CurrencyBasisMode,
    ForecastAudit,
    ForecastAudits,
    LifecycleCase,
    LiquidityBasis,
    MarketCase,
    ProjectCase,
    ProjectCaseValidationError,
    Projection,
    ProjectionKind,
    SampleWindow,
    SolverFailureDetail,
    ValuationCase,
)
from tests import pc_case_fixtures as fx

D1, D2 = fx.D1, fx.D2


# --- AssetCase / numeric domains (red-line #15) ------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"power_mw": 0.0},
        {"power_mw": -1.0},
        {"duration_hours": 0.0},
        {"round_trip_efficiency": 0.0},
        {"round_trip_efficiency": 1.5},
        {"installed_capex_eur": -1.0},
        {"fixed_om_eur_per_mw_yr": float("nan")},
    ],
)
def test_asset_case_rejects_bad_domains(kwargs):
    base = dict(
        power_mw=10.0,
        duration_hours=2.0,
        round_trip_efficiency=0.88,
        installed_capex_eur=1.0,
        fixed_om_eur_per_mw_yr=1.0,
    )
    base.update(kwargs)
    with pytest.raises(ProjectCaseValidationError):
        AssetCase(**base)


def test_asset_case_from_capex_per_kwh_normalises():
    a = AssetCase.from_capex_per_kwh(
        power_mw=10.0, duration_hours=2.0, round_trip_efficiency=0.9,
        capex_eur_per_kwh=300.0, fixed_om_eur_per_mw_yr=1.0,
    )
    assert a.installed_capex_eur == 300.0 * 20_000.0  # 20 MWh * 1000 kWh/MWh
    assert "capex_eur_per_kwh" not in a.to_payload()
    assert "energy_mwh" not in a.to_payload()


# --- ValuationCase / BootstrapCase -------------------------------------------
def test_discount_rate_must_exceed_minus_one():
    with pytest.raises(ProjectCaseValidationError):
        ValuationCase(-1.0, 2026)
    ValuationCase(-0.5, 2026)  # allowed


@pytest.mark.parametrize("bad", [True, -1, 2**64, 1.0])
def test_seed_domain_rejects_bool_float_and_out_of_range(bad):
    with pytest.raises(ProjectCaseValidationError):
        BootstrapCase(bad, 5000, "pc-bootstrap-pcg64-choice365-linear-v1")


@pytest.mark.parametrize("bad", [999, 50001, 0])
def test_n_simulations_range(bad):
    with pytest.raises(ProjectCaseValidationError):
        BootstrapCase(0, bad, "pc-bootstrap-pcg64-choice365-linear-v1")


def test_bootstrap_algorithm_literal_locked():
    with pytest.raises(ProjectCaseValidationError):
        BootstrapCase(0, 5000, "some-other-algo")


# --- Capacity-maintenance three-state (§4.2, red-line #4) --------------------
def test_scheduled_needs_positive_restoration_event():
    with pytest.raises(ProjectCaseValidationError):
        LifecycleCase(
            10, CapacityMaintenanceBasis.SCHEDULED_NAMEPLATE_MAINTENANCE, "src", "2026-01-01",
            (AugmentationEvent(3, 100.0, 0.0, 0.0),), 0.0, 0.0,
        )
    LifecycleCase(
        10, CapacityMaintenanceBasis.SCHEDULED_NAMEPLATE_MAINTENANCE, "src", "2026-01-01",
        (AugmentationEvent(3, 100.0, 0.3, 0.0),), 0.0, 0.0,
    )


def test_no_augmentation_requires_empty_schedule():
    with pytest.raises(ProjectCaseValidationError):
        LifecycleCase(
            10, CapacityMaintenanceBasis.NO_AUGMENTATION_REQUIRED_ASSERTED, "src", "2026-01-01",
            (AugmentationEvent(3, 100.0, 0.3, 0.0),), 0.0, 0.0,
        )
    LifecycleCase(
        10, CapacityMaintenanceBasis.NO_AUGMENTATION_REQUIRED_ASSERTED, "src", "2026-01-01",
        (), 0.0, 0.0,
    )


def test_asserted_basis_requires_source_and_as_of():
    with pytest.raises(ProjectCaseValidationError):
        LifecycleCase(
            10, CapacityMaintenanceBasis.NO_AUGMENTATION_REQUIRED_ASSERTED, None, "2026-01-01",
            (), 0.0, 0.0,
        )
    with pytest.raises(ProjectCaseValidationError):
        LifecycleCase(
            10, CapacityMaintenanceBasis.NO_AUGMENTATION_REQUIRED_ASSERTED, "src", None, (), 0.0, 0.0,
        )


def test_unknown_basis_requires_null_source_and_as_of():
    LifecycleCase(10, CapacityMaintenanceBasis.UNKNOWN, None, None, (), 0.0, 0.0)
    with pytest.raises(ProjectCaseValidationError):
        LifecycleCase(10, CapacityMaintenanceBasis.UNKNOWN, "src", None, (), 0.0, 0.0)


def test_event_year_within_project_life():
    with pytest.raises(ProjectCaseValidationError):
        LifecycleCase(
            5, CapacityMaintenanceBasis.SCHEDULED_NAMEPLATE_MAINTENANCE, "s", "2026-01-01",
            (AugmentationEvent(6, 1.0, 0.3, 0.0),), 0.0, 0.0,
        )


# --- Projection tagged union (§4.7, red-line #9) -----------------------------
def test_flat_projection_members_all_null():
    Projection(ProjectionKind.FlatRealProjection)
    with pytest.raises(ProjectCaseValidationError):
        Projection(ProjectionKind.FlatRealProjection, annual_decay_rate=0.1)


def test_spread_decay_domains():
    Projection(ProjectionKind.DAOnlySpreadDecay, 0.1, 0.5)
    with pytest.raises(ProjectCaseValidationError):
        Projection(ProjectionKind.DAOnlySpreadDecay, 1.0, 0.5)  # decay must be < 1
    with pytest.raises(ProjectCaseValidationError):
        Projection(ProjectionKind.DAOnlySpreadDecay, 0.1, 1.5)  # floor in [0,1]


def test_explicit_curve_year1_must_be_one_and_source_required():
    Projection(ProjectionKind.ExplicitAnnualMultiplierCurve, multipliers=(1.0, 0.9), source="s", as_of="2026-01-01")
    with pytest.raises(ProjectCaseValidationError):
        Projection(ProjectionKind.ExplicitAnnualMultiplierCurve, multipliers=(0.9, 0.9), source="s", as_of="2026-01-01")
    with pytest.raises(ProjectCaseValidationError):
        Projection(ProjectionKind.ExplicitAnnualMultiplierCurve, multipliers=(1.0, -0.1), source="s", as_of="2026-01-01")
    with pytest.raises(ProjectCaseValidationError):
        Projection(ProjectionKind.ExplicitAnnualMultiplierCurve, multipliers=(1.0, 0.9), source=None, as_of="2026-01-01")


# --- CurrencyBasis (§4.3/§4.4, red-line #19) ---------------------------------
def test_currency_deflator_needs_members():
    with pytest.raises(ProjectCaseValidationError):
        CurrencyBasis(CurrencyBasisMode.DEFLATOR_APPLIED, 2026)  # missing method/vintage/factor
    with pytest.raises(ProjectCaseValidationError):
        CurrencyBasis(CurrencyBasisMode.DEFLATOR_APPLIED, 2026, "cpi", "2026", 0.0)  # factor > 0


def test_currency_source_mode_members_must_be_null():
    CurrencyBasis(CurrencyBasisMode.SOURCE_EUR_TREATED_AS_BASE_YEAR_REAL, 2026)
    with pytest.raises(ProjectCaseValidationError):
        CurrencyBasis(CurrencyBasisMode.SOURCE_EUR_TREATED_AS_BASE_YEAR_REAL, 2026, deflator_method="cpi")


# --- CoverageAudit partition (§4.3/§4.6, red-line #17) -----------------------
def test_coverage_partition_must_cover_and_be_disjoint():
    with pytest.raises(ProjectCaseValidationError):
        CoverageAudit((D1, D2), (D1,), (), (), ())  # D2 unassigned
    with pytest.raises(ProjectCaseValidationError):
        CoverageAudit((D1, D2), (D1, D2), (D1,), (), ())  # D1 in valid and missing


def test_solver_failure_details_one_per_failed_date():
    det = SolverFailureDetail(D1, "s", "m", "stage")
    CoverageAudit((D1, D2), (D2,), (), (D1,), (det,))
    with pytest.raises(ProjectCaseValidationError):
        CoverageAudit((D1, D2), (D2,), (), (D1,), ())  # missing detail


# --- StrategyRunResult invariants --------------------------------------------
def test_series_must_equal_valid_dates():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        dc.replace(srr, daily_realised_cash_series=((D1, 1.0),))  # valid_dates=(D1,D2)


def test_empty_series_rejected():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        dc.replace(
            srr,
            daily_realised_cash_series=(),
            coverage_audit=CoverageAudit((D1, D2), (), (D1, D2), (), ()),
        )


def test_vom_must_equal_dispatch_constant():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        dc.replace(srr, embedded_vom_cost_eur_mwh=0.6)


def test_post_vom_must_be_true():
    with pytest.raises(ProjectCaseValidationError):
        CashBasis(False, CaptureBasis(False, 1.0, "not_applied"), LiquidityBasis(False))


def test_timezone_must_be_registry_derived():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        dc.replace(srr, sample_window=SampleWindow(D1, D2, "Europe/Paris"))  # DE_LU tz is Berlin


def test_unsupported_zone_rejected():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        dc.replace(srr, zone="XX")


# --- Forecast null matrix per kind (§4.3, red-line #10) ----------------------
def test_da_only_forecast_audits_must_be_null():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        dc.replace(srr, forecast_audits=ForecastAudits(ida=ForecastAudit("walk_forward", "hour_of_day", 0.0)))


def test_da_id_reserve_requires_three_legs_with_reserve_block_bucket():
    srr = fx.da_id_reserve_srr()
    with pytest.raises(ProjectCaseValidationError):
        dc.replace(srr, forecast_audits=ForecastAudits(
            da=ForecastAudit("walk_forward", "hour_of_day", None),
            ida=ForecastAudit("walk_forward", "hour_of_day", None),
            reserve=ForecastAudit("walk_forward", "hour_of_day", None),  # wrong bucket
        ))


def test_forecast_mode_must_be_walk_forward():
    with pytest.raises(ProjectCaseValidationError):
        ForecastAudit("loo", "hour_of_day", 0.0)


# --- Reserve fields null matrix + capture_rate matrix (R10-01/R10-02) --------
def test_non_reserve_kind_reserve_fields_null():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        dc.replace(srr, reserve_product="FCR")


def test_capture_rate_null_only_for_da_only():
    srr = fx.da_id_reserve_srr()
    prov = dc.replace(srr.adapter_provenance, capture_rate=1.0)
    with pytest.raises(ProjectCaseValidationError):
        dc.replace(srr, adapter_provenance=prov)


def test_da_only_capture_rate_must_equal_cash_rate():
    srr = fx.da_only_srr()
    prov = dc.replace(srr.adapter_provenance, capture_rate=0.5)  # cash rate is 0.9
    with pytest.raises(ProjectCaseValidationError):
        dc.replace(srr, adapter_provenance=prov)


# --- ProjectCase cross-invariants (§4.6) -------------------------------------
def test_engineering_match_enforced():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        ProjectCase(
            AssetCase(11.0, 2.0, 0.88, 1.0, 1.0),  # power mismatch
            LifecycleCase(15, CapacityMaintenanceBasis.UNKNOWN, None, None, (), 0.0, 0.0),
            MarketCase(srr, Projection(ProjectionKind.FlatRealProjection)),
            ValuationCase(0.08, 2026),
            BootstrapCase(0, 5000, "pc-bootstrap-pcg64-choice365-linear-v1"),
        )


def test_currency_base_year_must_match_valuation():
    srr = fx.da_only_srr()  # currency target_base_year 2026
    with pytest.raises(ProjectCaseValidationError):
        ProjectCase(
            AssetCase(10.0, 2.0, 0.88, 1.0, 1.0),
            LifecycleCase(15, CapacityMaintenanceBasis.UNKNOWN, None, None, (), 0.0, 0.0),
            MarketCase(srr, Projection(ProjectionKind.FlatRealProjection)),
            ValuationCase(0.08, 2025),  # mismatch
            BootstrapCase(0, 5000, "pc-bootstrap-pcg64-choice365-linear-v1"),
        )


def test_non_flat_projection_only_for_da_only():
    srr = fx.da_id_reserve_srr()
    with pytest.raises(ProjectCaseValidationError):
        ProjectCase(
            AssetCase(10.0, 2.0, 0.88, 1.0, 1.0),
            LifecycleCase(15, CapacityMaintenanceBasis.UNKNOWN, None, None, (), 0.0, 0.0),
            MarketCase(srr, Projection(ProjectionKind.DAOnlySpreadDecay, 0.02, 0.5)),
            ValuationCase(0.08, 2026),
            BootstrapCase(0, 5000, "pc-bootstrap-pcg64-choice365-linear-v1"),
        )


def test_explicit_curve_length_must_equal_project_life():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        ProjectCase(
            AssetCase(10.0, 2.0, 0.88, 1.0, 1.0),
            LifecycleCase(15, CapacityMaintenanceBasis.UNKNOWN, None, None, (), 0.0, 0.0),
            MarketCase(
                srr,
                Projection(ProjectionKind.ExplicitAnnualMultiplierCurve, multipliers=(1.0, 0.9), source="s", as_of="2026-01-01"),
            ),
            ValuationCase(0.08, 2026),
            BootstrapCase(0, 5000, "pc-bootstrap-pcg64-choice365-linear-v1"),
        )


def test_lifecycle_available_flag():
    assert fx.project_case().lifecycle_case.lifecycle_available is True
    unknown = LifecycleCase(15, CapacityMaintenanceBasis.UNKNOWN, None, None, (), 0.0, 0.0)
    assert unknown.lifecycle_available is False
