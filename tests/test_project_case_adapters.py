"""Producer adapters + bootstrap (contract §5, §4.8, §9 PC-A)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.project_case import (
    SPECS,
    AdapterUnavailableError,
    ProducerAdapterId,
    StrategyKind,
    bootstrap_annual_sums,
    emit_da_id,
    emit_da_id_reserve,
    emit_da_only,
    emit_reserve_coopt,
    grid,
)
from src.project_case.adapters import PC_A_CALCULATOR_VERSION
from tests import pc_case_fixtures as fx

ZONE = "DE_LU"
DAYS = [dt.date(2025, 6, 5), dt.date(2025, 6, 6), dt.date(2025, 6, 7)]
CB = fx.CURRENCY_SOURCE


# --- Pinned 5-tuple registry (§5) --------------------------------------------
def test_specs_match_contract_table():
    da = SPECS[ProducerAdapterId.PC_ADP_DA_ONLY]
    assert (da.strategy_kind, da.source_function, da.per_day_cash_field, da.excluded_fields) == (
        StrategyKind.DA_ONLY, "simulate_replay_batch", "total_revenue_eur", ("degradation_cost_eur",),
    )
    di = SPECS[ProducerAdapterId.PC_ADP_DA_ID]
    assert (di.source_function, di.per_day_cash_field, di.excluded_fields) == (
        "simulate_sequential_da_id_batch", "realised_eur", ("ceiling_eur",),
    )
    rc = SPECS[ProducerAdapterId.PC_ADP_RESERVE_COOPT]
    assert (rc.source_function, rc.per_day_cash_field) == ("solve_joint_capacity_batch", "joint_total_revenue")
    tr = SPECS[ProducerAdapterId.PC_ADP_DA_ID_RESERVE]
    assert (tr.source_function, tr.per_day_cash_field, tr.excluded_fields) == (
        "simulate_sequential_da_id_reserve_batch", "realised_eur",
        ("reserve_first_ceiling_eur", "global_ceiling_eur"),
    )


# --- DA_ONLY -----------------------------------------------------------------
def test_da_only_emit_and_provenance():
    srr = emit_da_only(
        fx.da_frame(ZONE, DAYS), zone=ZONE, first_delivery_date=DAYS[0], last_delivery_date=DAYS[-1],
        power_mw=10.0, duration_hours=2.0, efficiency=0.88, currency_basis=CB,
        capture_rate=0.9, capture_source="da_slippage",
        runner=fx.fake_runner("total_revenue_eur", [(DAYS[0], 100.0), (DAYS[1], 110.0), (DAYS[2], 90.0)]),
    )
    assert srr.strategy_kind is StrategyKind.DA_ONLY
    assert srr.coverage_audit.valid_dates == tuple(DAYS)
    assert srr.adapter_provenance.capture_rate == 0.9
    assert srr.cash_basis.capture.applied is True
    assert srr.adapter_provenance.mode == "DA MILP Replay"
    assert srr.adapter_provenance.expected_grid_profiles == {
        "da": grid.da_profile_id(ZONE), "ida": None, "reserve": None,
    }
    assert srr.calculator_version == PC_A_CALCULATOR_VERSION


def test_da_only_capture_not_applied_at_rate_one():
    srr = emit_da_only(
        fx.da_frame(ZONE, DAYS), zone=ZONE, first_delivery_date=DAYS[0], last_delivery_date=DAYS[-1],
        power_mw=10.0, duration_hours=2.0, efficiency=0.88, currency_basis=CB,
        runner=fx.fake_runner("total_revenue_eur", [(d, 10.0) for d in DAYS]),
    )
    assert srr.cash_basis.capture.applied is False
    assert srr.cash_basis.capture.rate == 1.0


def test_da_only_missing_data_day_is_missing_dates():
    # Drop the last stamp of DAYS[1] -> that day is data-incomplete -> missing.
    frame = fx.da_frame(ZONE, DAYS)
    last_stamp = grid.expected_da_timestamps(ZONE, DAYS[1])[-1]
    frame = frame.drop(index=last_stamp)
    srr = emit_da_only(
        frame, zone=ZONE, first_delivery_date=DAYS[0], last_delivery_date=DAYS[-1],
        power_mw=10.0, duration_hours=2.0, efficiency=0.88, currency_basis=CB,
        runner=fx.fake_runner("total_revenue_eur", [(DAYS[0], 1.0), (DAYS[2], 1.0)]),
    )
    assert DAYS[1] in srr.coverage_audit.missing_dates
    assert DAYS[1] not in srr.coverage_audit.valid_dates


def test_da_only_empty_valid_is_unavailable():
    with pytest.raises(AdapterUnavailableError):
        emit_da_only(
            fx.da_frame(ZONE, DAYS), zone=ZONE, first_delivery_date=DAYS[0], last_delivery_date=DAYS[-1],
            power_mw=10.0, duration_hours=2.0, efficiency=0.88, currency_basis=CB,
            runner=fx.fake_runner("total_revenue_eur", []),
        )


def test_unsupported_zone_unavailable():
    with pytest.raises(AdapterUnavailableError):
        emit_da_only(
            fx.da_frame(ZONE, DAYS), zone="GB", first_delivery_date=DAYS[0], last_delivery_date=DAYS[-1],
            power_mw=10.0, duration_hours=2.0, efficiency=0.88, currency_basis=CB,
            runner=fx.fake_runner("total_revenue_eur", [(d, 1.0) for d in DAYS]),
        )


# --- DA_ID -------------------------------------------------------------------
def test_da_id_walk_forward_drop_is_solver_failed():
    srr = emit_da_id(
        fx.da_frame(ZONE, DAYS), fx.ida_frame(ZONE, DAYS), zone=ZONE,
        first_delivery_date=DAYS[0], last_delivery_date=DAYS[-1], power_mw=10.0,
        duration_hours=2.0, efficiency=0.88, currency_basis=CB, bucket="hour_of_day",
        min_rebid_uplift_eur=5.0,
        runner=fx.fake_runner("realised_eur", [(DAYS[1], 55.0), (DAYS[2], 47.0)]),
    )
    assert srr.coverage_audit.valid_dates == (DAYS[1], DAYS[2])
    assert srr.coverage_audit.solver_failed_dates == (DAYS[0],)
    assert srr.forecast_audits.ida.deadband == 5.0
    assert srr.forecast_audits.da is None and srr.forecast_audits.reserve is None
    assert srr.cash_basis.capture.rate == 1.0 and srr.cash_basis.capture.source == "not_applied"


def test_da_id_invalid_bucket_rejected():
    from src.project_case import ProjectCaseValidationError

    with pytest.raises(ProjectCaseValidationError):
        emit_da_id(
            fx.da_frame(ZONE, DAYS), fx.ida_frame(ZONE, DAYS), zone=ZONE,
            first_delivery_date=DAYS[0], last_delivery_date=DAYS[-1], power_mw=10.0,
            duration_hours=2.0, efficiency=0.88, currency_basis=CB, bucket="bogus",
            min_rebid_uplift_eur=0.0,
        )


# --- RESERVE_COOPT -----------------------------------------------------------
def test_reserve_coopt_degraded_day_is_missing_not_relabelled():
    rs = fx.reserve_series(ZONE, [(DAYS[0], 6), (DAYS[1], 6), (DAYS[2], 5)])  # day 3 missing a block
    srr = emit_reserve_coopt(
        fx.da_frame(ZONE, DAYS), rs, zone=ZONE, first_delivery_date=DAYS[0],
        last_delivery_date=DAYS[-1], power_mw=10.0, duration_hours=2.0, efficiency=0.88,
        currency_basis=CB, reserve_product="FCR", reserve_source="regelleistung",
        runner=fx.fake_coopt_runner([(DAYS[0], 200.0), (DAYS[1], 210.0)]),
    )
    assert srr.strategy_kind is StrategyKind.DA_RESERVE_COOPT
    assert srr.coverage_audit.valid_dates == (DAYS[0], DAYS[1])
    assert srr.coverage_audit.missing_dates == (DAYS[2],)
    assert srr.adapter_provenance.reserve_scalar_price_eur_mw_h == 12.0
    assert srr.adapter_provenance.reserve_pricing_dates == (DAYS[0], DAYS[1])
    assert srr.adapter_provenance.reserve_price_aggregation == "duration_weighted_mean_complete_blocks_v1"
    assert srr.reserve_coverage_audit.date_set() == frozenset(DAYS)  # entry per observed date


def test_reserve_coopt_all_days_uncovered_unavailable():
    rs = fx.reserve_series(ZONE, [(d, 3) for d in DAYS])  # no day fully covered
    with pytest.raises(AdapterUnavailableError):
        emit_reserve_coopt(
            fx.da_frame(ZONE, DAYS), rs, zone=ZONE, first_delivery_date=DAYS[0],
            last_delivery_date=DAYS[-1], power_mw=10.0, duration_hours=2.0, efficiency=0.88,
            currency_basis=CB, reserve_product="FCR", reserve_source="regelleistung",
            runner=fx.fake_coopt_runner([]),
        )


# --- DA_ID_RESERVE -----------------------------------------------------------
def test_da_id_reserve_null_reserve_scalar_members():
    rs = fx.reserve_series(ZONE, [(DAYS[0], 6), (DAYS[1], 6), (DAYS[2], 5)])
    srr = emit_da_id_reserve(
        fx.da_frame(ZONE, DAYS), fx.ida_frame(ZONE, DAYS), rs, zone=ZONE,
        first_delivery_date=DAYS[0], last_delivery_date=DAYS[-1], power_mw=10.0,
        duration_hours=2.0, efficiency=0.88, currency_basis=CB, reserve_product="aFRR",
        reserve_source="regelleistung", bucket="hour_of_day",
        runner=fx.fake_runner("realised_eur", [(DAYS[1], 77.0)]),
    )
    assert srr.strategy_kind is StrategyKind.DA_ID_RESERVE_REALISED
    assert srr.coverage_audit.valid_dates == (DAYS[1],)
    assert srr.coverage_audit.solver_failed_dates == (DAYS[0],)  # walk-forward drop
    assert srr.coverage_audit.missing_dates == (DAYS[2],)  # degraded reserve
    assert srr.adapter_provenance.reserve_scalar_price_eur_mw_h is None
    assert srr.adapter_provenance.reserve_price_aggregation is None
    assert srr.forecast_audits.reserve.bucket == "block_of_day_4h"


# --- Bootstrap algorithm literal + golden (§4.8, red-line #25) ---------------
def test_bootstrap_golden_vector():
    vals = np.array([100.0, 50.0, 200.0, -30.0, 75.0], dtype=float)
    sums = bootstrap_annual_sums(vals, seed=12345, n_simulations=1000)
    assert sums.shape == (1000,)
    p10, p50, p90 = np.percentile(sums, [10, 50, 90], method="linear")
    assert (float(p10), float(p50), float(p90)) == (26969.5, 28840.0, 30745.5)
    assert float(sums[0]) == 27075.0


def test_bootstrap_rejects_non_finite_and_bool_seed():
    with pytest.raises(ValueError):
        bootstrap_annual_sums(np.array([1.0, float("nan")]), seed=1, n_simulations=10)
    with pytest.raises(ValueError):
        bootstrap_annual_sums(np.array([1.0]), seed=True, n_simulations=10)


# --- Slow real-solver integration -------------------------------------------
def _tradeable_da_frame(days):
    """A DA frame with a deep intraday spread so the MILP actually cycles."""
    idx, vals = [], []
    for d in days:
        for ts in grid.expected_da_timestamps(ZONE, d):
            h = ts.tz_convert("Europe/Berlin").hour
            vals.append(10.0 if 2 <= h <= 5 else (120.0 if 18 <= h <= 21 else 55.0))
            idx.append(ts)
    return pd.DataFrame({"price_eur_mwh": vals}, index=pd.DatetimeIndex(idx))


@pytest.mark.slow
def test_da_only_real_solver_integration():
    srr = emit_da_only(
        _tradeable_da_frame(DAYS), zone=ZONE, first_delivery_date=DAYS[0],
        last_delivery_date=DAYS[-1], power_mw=10.0, duration_hours=2.0, efficiency=0.88,
        currency_basis=CB,
    )
    assert srr.coverage_audit.valid_dates == tuple(DAYS)
    assert len(srr.daily_realised_cash_series) == 3
    assert all(np.isfinite(v) for _, v in srr.daily_realised_cash_series)
    assert srr.fingerprint()  # fingerprints without error


@pytest.mark.slow
def test_reserve_coopt_real_solver_integration():
    rs = fx.reserve_series(ZONE, [(d, 6) for d in DAYS])
    srr = emit_reserve_coopt(
        _tradeable_da_frame(DAYS), rs, zone=ZONE, first_delivery_date=DAYS[0],
        last_delivery_date=DAYS[-1], power_mw=10.0, duration_hours=2.0, efficiency=0.88,
        currency_basis=CB, reserve_product="FCR", reserve_source="regelleistung",
    )
    assert srr.coverage_audit.valid_dates == tuple(DAYS)
    assert srr.adapter_provenance.reserve_scalar_price_eur_mw_h == 12.0
    assert srr.reserve_coverage_audit.covered_dates == frozenset(DAYS)
