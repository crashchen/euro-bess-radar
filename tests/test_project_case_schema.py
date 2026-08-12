"""Schema domain + cross-invariant validation (contract §4, §7 red-lines)."""

from __future__ import annotations

import dataclasses as dc
import datetime as dt

import pytest

from src.project_case import (
    AdapterProvenance,
    AssetCase,
    AugmentationEvent,
    BootstrapCase,
    CapacityMaintenanceBasis,
    CaptureBasis,
    CashBasis,
    CashflowRow,
    CashflowTable,
    CoverageAudit,
    CurrencyBasis,
    CurrencyBasisMode,
    ForecastAudit,
    ForecastAudits,
    LifecycleCase,
    LiquidityBasis,
    MarketCase,
    NpvDistribution,
    NpvOutcome,
    ProducerAdapterId,
    ProjectCase,
    ProjectCaseValidationError,
    Projection,
    ProjectionKind,
    ReserveCoverageAudit,
    ReserveCoverageEntry,
    RunResult,
    SampleWindow,
    SolverFailureDetail,
    ValuationCase,
    grid,
)
from src.project_case.enums import LIFECYCLE_UNKNOWN_MESSAGE, LIFECYCLE_UNKNOWN_STATUS
from src.project_case.schema import _issue_strategy_run_result
from tests import pc_case_fixtures as fx

D1, D2 = fx.D1, fx.D2
D3 = dt.date(2026, 3, 12)


def _unknown_lifecycle() -> NpvOutcome:
    """The one locked lifecycle-unavailable envelope (§3, §4.6)."""
    return NpvOutcome.unavailable(LIFECYCLE_UNKNOWN_STATUS, LIFECYCLE_UNKNOWN_MESSAGE)


def _reissue(srr, **changes):
    """Re-emit a StrategyRunResult variant through the producer-issuance context.

    Field-validation tests build a "producer emits this (bad) field" case by cloning
    a valid fixture with one field changed. Construction is producer-issued only
    (§4.3, red-line #6/#18), so the clone must go through the issuance context —
    otherwise it trips the issuance guard rather than the field invariant under
    test. A bare ``dc.replace`` (the forge path) is covered separately.
    """
    with _issue_strategy_run_result():
        return dc.replace(srr, **changes)


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
        _reissue(srr, daily_realised_cash_series=((D1, 1.0),))  # valid_dates=(D1,D2)


def test_empty_series_rejected():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        _reissue(
            srr,
            daily_realised_cash_series=(),
            coverage_audit=CoverageAudit((D1, D2), (), (D1, D2), (), ()),
        )


def test_vom_must_equal_dispatch_constant():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        _reissue(srr, embedded_vom_cost_eur_mwh=0.6)


def test_post_vom_must_be_true():
    with pytest.raises(ProjectCaseValidationError):
        CashBasis(False, CaptureBasis(False, 1.0, "not_applied"), LiquidityBasis(False))


def test_timezone_must_be_registry_derived():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        _reissue(srr, sample_window=SampleWindow(D1, D2, "Europe/Paris"))  # DE_LU tz is Berlin


def test_unsupported_zone_rejected():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        _reissue(srr, zone="XX")


# --- Forecast null matrix per kind (§4.3, red-line #10) ----------------------
def test_da_only_forecast_audits_must_be_null():
    srr = fx.da_only_srr()
    with pytest.raises(ProjectCaseValidationError):
        _reissue(srr, forecast_audits=ForecastAudits(ida=ForecastAudit("walk_forward", "hour_of_day", 0.0)))


def test_da_id_reserve_requires_three_legs_with_reserve_block_bucket():
    srr = fx.da_id_reserve_srr()
    with pytest.raises(ProjectCaseValidationError):
        _reissue(srr, forecast_audits=ForecastAudits(
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
        _reissue(srr, reserve_product="FCR")


def test_capture_rate_null_only_for_da_only():
    srr = fx.da_id_reserve_srr()
    prov = dc.replace(srr.adapter_provenance, capture_rate=1.0)
    with pytest.raises(ProjectCaseValidationError):
        _reissue(srr, adapter_provenance=prov)


def test_da_only_capture_rate_must_equal_cash_rate():
    srr = fx.da_only_srr()
    prov = dc.replace(srr.adapter_provenance, capture_rate=0.5)  # cash rate is 0.9
    with pytest.raises(ProjectCaseValidationError):
        _reissue(srr, adapter_provenance=prov)


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


# --- Producer 5-tuple lock (§5, red-line #6/#18) -----------------------------
def test_producer_source_function_locked():
    srr = fx.da_only_srr()
    bad = dc.replace(srr.adapter_provenance, source_function="totally_wrong")
    with pytest.raises(ProjectCaseValidationError):
        _reissue(srr, adapter_provenance=bad)


def test_producer_excluded_fields_locked():
    srr = fx.da_only_srr()
    # DA-only must exclude degradation_cost_eur (shadow wear), not ceiling_eur.
    bad = dc.replace(srr.adapter_provenance, excluded_fields=("ceiling_eur",))
    with pytest.raises(ProjectCaseValidationError):
        _reissue(srr, adapter_provenance=bad)


def test_producer_kind_must_match_adapter_id():
    srr = fx.da_only_srr()
    bad = dc.replace(srr.adapter_provenance, producer_adapter_id=ProducerAdapterId.PC_ADP_DA_ID)
    with pytest.raises(ProjectCaseValidationError):
        _reissue(srr, adapter_provenance=bad)


def test_grid_profile_must_be_bound_to_zone():
    """A profile must be the exact registry id for THIS (leg, zone): a DE_LU result
    carrying the CH DA profile (both are 'da' profiles) is rejected (review r3 #3)."""
    srr = fx.da_only_srr()  # DE_LU
    profiles = dict(srr.adapter_provenance.expected_grid_profiles)
    profiles["da"] = grid.DA_PROFILE_CH  # CH's DA profile, wrong for DE_LU
    bad = dc.replace(srr.adapter_provenance, expected_grid_profiles=profiles)
    with pytest.raises(ProjectCaseValidationError, match="registry profile for zone"):
        _reissue(srr, adapter_provenance=bad)


def test_strategy_run_result_is_producer_issued_only():
    """§4.3 red-line #6/#18: a StrategyRunResult cannot be forged by direct
    construction or by a ``dataclasses.replace`` that swaps in an arbitrary cash
    series while keeping canonical provenance and a valid fingerprint."""
    srr = fx.da_only_srr()
    # The forge: clone a legitimate result and inject an absurd cash series.
    with pytest.raises(ProjectCaseValidationError, match="producer-issued only"):
        dc.replace(
            srr,
            daily_realised_cash_series=((D1, 9.99e99), (D2, 9.99e99)),
        )
    # Direct construction outside an adapter is equally rejected.
    fields = {f.name: getattr(srr, f.name) for f in dc.fields(srr)}
    with pytest.raises(ProjectCaseValidationError, match="producer-issued only"):
        type(srr)(**fields)
    # Re-issuing the identical result through the producer context succeeds and is
    # byte-identical (the guard adds no state to the fingerprint).
    assert _reissue(srr).fingerprint() == srr.fingerprint()


# --- Deep immutability of fingerprint-bearing objects ------------------------
def test_reserve_entry_duration_map_is_immutable():
    entry = fx.da_id_reserve_srr().reserve_coverage_audit.entries[0]
    with pytest.raises(TypeError):
        entry.settlement_duration_hours_by_block[entry.required_blocks[0]] = 9.0


def test_provenance_grid_profiles_immutable_and_fingerprint_stable():
    srr = fx.da_only_srr()
    fp = srr.fingerprint()
    with pytest.raises(TypeError):
        srr.adapter_provenance.expected_grid_profiles["da"] = "hacked"
    assert srr.fingerprint() == fp


# --- Reserve: uncovered day cannot be solver_failed (§5, no-relabel) ----------
def _covered_entry(d: dt.date) -> ReserveCoverageEntry:
    blocks = grid.reserve_blocks("DE_LU", d)
    ids = tuple(b for b, _ in blocks)
    return ReserveCoverageEntry(d, ids, ids, (), {b: 4.0 for b, _ in blocks})


def _uncovered_entry(d: dt.date) -> ReserveCoverageEntry:
    blocks = grid.reserve_blocks("DE_LU", d)
    ids = tuple(b for b, _ in blocks)
    return ReserveCoverageEntry(d, ids, ids[:5], (ids[5],), {b: 4.0 for b, _ in blocks})


def test_reserve_solver_failed_date_must_be_covered():
    with pytest.raises(ProjectCaseValidationError):
        _reissue(
            fx.da_id_reserve_srr(),
            sample_window=SampleWindow(D1, D3, "Europe/Berlin"),
            daily_realised_cash_series=((D1, 300.0),),
            coverage_audit=CoverageAudit(
                (D1, D2, D3), (D1,), (D3,), (D2,),
                (SolverFailureDetail(D2, "s", "m", "st"),),
            ),
            reserve_coverage_audit=ReserveCoverageAudit(
                (_covered_entry(D1), _uncovered_entry(D2), _uncovered_entry(D3))
            ),
        )


# --- RunResult / NpvOutcome scaffold (§3, §4.6) ------------------------------
def _cashflow_table(basis: str) -> CashflowTable:
    return CashflowTable(
        basis,
        (
            CashflowRow(1, 100.0, 10.0, 0.0, 0.0, 90.0, 0.9259, 83.3),
            CashflowRow(2, 100.0, 10.0, 0.0, 0.0, 90.0, 0.8573, 77.2),
        ),
    )


def test_npv_outcome_shapes():
    dist = NpvDistribution(1.0, 2.0, 3.0, 0.5)
    ok = NpvOutcome.ok(dist)
    assert ok.available and ok.status == "ok" and ok.message is None
    un = NpvOutcome.unavailable("capacity_maintenance_unknown", "unknown basis")
    assert not un.available and un.distribution is None
    with pytest.raises(ProjectCaseValidationError):
        NpvOutcome(True, "ok", "msg", dist)  # available cannot carry a message
    with pytest.raises(ProjectCaseValidationError):
        NpvOutcome(False, "x", "m", dist)  # unavailable cannot carry a distribution
    with pytest.raises(ProjectCaseValidationError):
        NpvDistribution(1.0, 2.0, 3.0, 1.5)  # prob_positive out of [0,1]


def test_run_result_state_matrix_and_serialisation():
    ok = NpvOutcome.ok(NpvDistribution(1.0, 2.0, 3.0, 0.5))
    un = _unknown_lifecycle()
    # Valid: screening available + table present; lifecycle unavailable + null table.
    rr = RunResult(
        input_fingerprint="ab" * 32,
        no_lifecycle_cost_screening_npv=ok,
        lifecycle_cash_npv=un,
        provenance={"note": "pc-b fills this", "nested": {"a": [1, 2]}},
        screening_cashflow_table=_cashflow_table("screening"),
    )
    assert rr.schema_version == "project-case-v1"
    payload = rr.to_payload()
    assert payload["lifecycle_cashflow_table"] is None
    assert payload["screening_cashflow_table"]["rows"][0]["year"] == 1
    assert payload["provenance"]["nested"]["a"] == [1, 2]  # round-trips to plain list
    # Both bases available: both tables present, each bound to its own basis.
    both = RunResult(
        "ab" * 32, ok, ok, provenance={},
        screening_cashflow_table=_cashflow_table("screening"),
        lifecycle_cashflow_table=_cashflow_table("lifecycle"),
    )
    assert both.to_payload()["lifecycle_cashflow_table"]["basis"] == "lifecycle"


def test_run_result_screening_must_be_available():
    """§4.6: the screening NPV is always available; a screening-unavailable /
    lifecycle-available result is unrepresentable (review r3 #4)."""
    ok = NpvOutcome.ok(NpvDistribution(1.0, 2.0, 3.0, 0.5))
    un = _unknown_lifecycle()
    with pytest.raises(ProjectCaseValidationError, match="screening"):
        RunResult(
            "ab" * 32, un, ok, provenance={},
            lifecycle_cashflow_table=_cashflow_table("lifecycle"),
        )
    # Both unavailable is also rejected (screening must be available).
    with pytest.raises(ProjectCaseValidationError, match="screening"):
        RunResult("ab" * 32, un, un, provenance={})


def test_run_result_lifecycle_unavailable_is_the_locked_envelope():
    """An unavailable lifecycle NPV must be EXACTLY the §3 UNKNOWN status+message."""
    ok = NpvOutcome.ok(NpvDistribution(1.0, 2.0, 3.0, 0.5))
    for status, message in (
        ("capacity_maintenance_unknown", "some other message"),
        ("something_else", LIFECYCLE_UNKNOWN_MESSAGE),
    ):
        with pytest.raises(ProjectCaseValidationError, match="unavailable lifecycle"):
            RunResult(
                "ab" * 32, ok, NpvOutcome.unavailable(status, message), provenance={},
                screening_cashflow_table=_cashflow_table("screening"),
            )


def test_run_result_rejects_contradictory_states():
    ok = NpvOutcome.ok(NpvDistribution(1.0, 2.0, 3.0, 0.5))
    un = _unknown_lifecycle()
    # Available NPV (screening) but a null table is a contradiction.
    with pytest.raises(ProjectCaseValidationError):
        RunResult("ab" * 32, ok, un, provenance={})
    # Unavailable NPV (lifecycle) but a present table is a contradiction.
    with pytest.raises(ProjectCaseValidationError):
        RunResult(
            "ab" * 32, ok, un, provenance={},
            screening_cashflow_table=_cashflow_table("screening"),
            lifecycle_cashflow_table=_cashflow_table("lifecycle"),
        )
    with pytest.raises(ProjectCaseValidationError):
        RunResult("nothex", ok, un, provenance={}, screening_cashflow_table=_cashflow_table("screening"))


def test_run_result_table_basis_must_match_slot():
    """A lifecycle table filed under the screening slot (or vice versa) is rejected."""
    ok = NpvOutcome.ok(NpvDistribution(1.0, 2.0, 3.0, 0.5))
    # Both NPVs available but the tables are swapped between slots.
    with pytest.raises(ProjectCaseValidationError, match="basis"):
        RunResult(
            "ab" * 32, ok, ok, provenance={},
            screening_cashflow_table=_cashflow_table("lifecycle"),
            lifecycle_cashflow_table=_cashflow_table("screening"),
        )


def test_run_result_provenance_is_deep_frozen():
    ok = NpvOutcome.ok(NpvDistribution(1.0, 2.0, 3.0, 0.5))
    un = _unknown_lifecycle()
    src = {"solver": {"failures": [1, 2]}}
    rr = RunResult(
        "ab" * 32, ok, un, provenance=src,
        screening_cashflow_table=_cashflow_table("screening"),
    )
    src["solver"]["failures"].append(3)  # mutating the caller's dict must not leak in
    assert rr.provenance["solver"]["failures"] == (1, 2)
    with pytest.raises(TypeError):
        rr.provenance["solver"] = None  # deep-frozen


def test_run_result_provenance_rejects_unserialisable_domain():
    """Provenance must be a closed JSON/CBOR tree — a set / bytes / object that
    ``_deep_freeze`` used to pass through untouched is rejected (review r3 #4)."""
    ok = NpvOutcome.ok(NpvDistribution(1.0, 2.0, 3.0, 0.5))
    un = _unknown_lifecycle()
    for bad in ({"k": {1, 2, 3}}, {"k": b"bytes"}, {"k": object()}, {1: "int-key"},
                {"k": float("nan")}):
        with pytest.raises(ProjectCaseValidationError):
            RunResult(
                "ab" * 32, ok, un, provenance=bad,
                screening_cashflow_table=_cashflow_table("screening"),
            )


def test_cashflow_table_rejects_bad_rows():
    with pytest.raises(ProjectCaseValidationError):
        CashflowTable("bogus", (CashflowRow(1, 0, 0, 0, 0, 0, 1.0, 0),))  # bad basis
    with pytest.raises(ProjectCaseValidationError):
        CashflowTable("screening", ())  # empty
    with pytest.raises(ProjectCaseValidationError):  # duplicate/unsorted years
        CashflowTable(
            "screening",
            (CashflowRow(2, 0, 0, 0, 0, 0, 1.0, 0), CashflowRow(2, 0, 0, 0, 0, 0, 1.0, 0)),
        )
    with pytest.raises(ProjectCaseValidationError):
        CashflowRow(1, float("nan"), 0, 0, 0, 0, 1.0, 0)  # non-finite cash


# --- Capture basis coupling (review r2 #6) -----------------------------------
def test_capture_basis_applied_flag_and_source_are_coupled():
    CaptureBasis(True, 0.9, "da_slippage")          # ok
    CaptureBasis(False, 1.0, "not_applied")         # ok
    with pytest.raises(ProjectCaseValidationError):
        CaptureBasis(True, 0.9, "not_applied")      # applied haircut needs a source
    with pytest.raises(ProjectCaseValidationError):
        CaptureBasis(True, 1.0, "da_slippage")      # applied must mean rate != 1.0
    with pytest.raises(ProjectCaseValidationError):
        CaptureBasis(False, 0.9, "da_slippage")     # rate != 1.0 must be applied


# --- AdapterProvenance immutability + profile validation (review r2 #2/#3) ----
def _provenance(excluded, profiles) -> AdapterProvenance:
    return AdapterProvenance(
        ProducerAdapterId.PC_ADP_DA_ONLY, "simulate_replay_batch", "total_revenue_eur",
        excluded, "DA MILP Replay", False, 0.5, 0.9, None, None, None,
        "pc-market-grid-v1", profiles,
    )


def test_adapter_provenance_excluded_fields_is_owned_tuple():
    lst = ["degradation_cost_eur"]
    prov = _provenance(lst, {"da": grid.da_profile_id("DE_LU"), "ida": None, "reserve": None})
    lst.append("SNUCK_IN")  # mutating the caller's list must not change the object
    assert prov.excluded_fields == ("degradation_cost_eur",)


def test_adapter_provenance_rejects_bogus_profile_id():
    with pytest.raises(ProjectCaseValidationError):
        _provenance(("degradation_cost_eur",), {"da": "bogus-profile", "ida": None, "reserve": None})
