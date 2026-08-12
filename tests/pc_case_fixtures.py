"""Shared Project Case v1 (PC-A) fixture builders for the tests.

Not a test module (no ``test_`` prefix, not collected). Canonical objects here are
hand-built (solver-independent) so the fingerprint golden vectors are stable.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.project_case import (
    AdapterProvenance,
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
    ProducerAdapterId,
    ProjectCase,
    Projection,
    ProjectionKind,
    ReserveCoverageAudit,
    ReserveCoverageEntry,
    SampleWindow,
    StrategyKind,
    StrategyRunResult,
    ValuationCase,
    grid,
)
from src.project_case.schema import _issue_strategy_run_result

ZONE = "DE_LU"
TZ = "Europe/Berlin"
D1 = dt.date(2026, 3, 10)
D2 = dt.date(2026, 3, 11)
CURRENCY_DEFLATOR = CurrencyBasis(CurrencyBasisMode.DEFLATOR_APPLIED, 2026, "cpi", "2026Q1", 1.05)
CURRENCY_SOURCE = CurrencyBasis(CurrencyBasisMode.SOURCE_EUR_TREATED_AS_BASE_YEAR_REAL, 2026)


def da_only_srr() -> StrategyRunResult:
    """Canonical DA_ONLY StrategyRunResult (golden-vector fixture).

    Constructed inside the producer-issuance context: the fixture stands in for an
    ``emit_*`` adapter (the only legal issuer, §4.3), so it must issue through the
    same guarded path a real adapter uses (review r3 #5).
    """
    with _issue_strategy_run_result():
        return StrategyRunResult(
            strategy_kind=StrategyKind.DA_ONLY,
            daily_realised_cash_series=((D1, 123.5), (D2, -12.0)),
            cash_basis=CashBasis(True, CaptureBasis(True, 0.9, "da_slippage"), LiquidityBasis(False)),
            power_mw=10.0,
            duration_hours=2.0,
            round_trip_efficiency=0.88,
            zone=ZONE,
            sample_window=SampleWindow(D1, D2, TZ),
            currency_basis=CURRENCY_DEFLATOR,
            forecast_audits=ForecastAudits(),
            reserve_product=None,
            reserve_source=None,
            availability=None,
            reserve_coverage_audit=None,
            coverage_audit=CoverageAudit((D1, D2), (D1, D2), (), (), ()),
            adapter_provenance=AdapterProvenance(
                ProducerAdapterId.PC_ADP_DA_ONLY,
                "simulate_replay_batch",
                "total_revenue_eur",
                ("degradation_cost_eur",),
                "DA MILP Replay",
                False,
                0.5,
                0.9,
                None,
                None,
                None,
                "pc-market-grid-v1",
                {"da": grid.da_profile_id(ZONE), "ida": None, "reserve": None},
            ),
            embedded_vom_cost_eur_mwh=0.5,
            source_data_content_hash="ab" * 32,
            calculator_version="pc-a-v1",
        )


def _reserve_entry(d: dt.date) -> ReserveCoverageEntry:
    blocks = grid.reserve_blocks(ZONE, d)
    ids = tuple(b for b, _ in blocks)
    return ReserveCoverageEntry(d, ids, ids, (), {b: dur for b, dur in blocks})


def da_id_reserve_srr() -> StrategyRunResult:
    """Canonical DA_ID_RESERVE_REALISED StrategyRunResult (golden-vector fixture).

    Issued through the producer-issuance context, like ``da_only_srr`` (§4.3).
    """
    with _issue_strategy_run_result():
        return StrategyRunResult(
            strategy_kind=StrategyKind.DA_ID_RESERVE_REALISED,
            daily_realised_cash_series=((D1, 300.0), (D2, 250.0)),
            cash_basis=CashBasis(True, CaptureBasis(False, 1.0, "not_applied"), LiquidityBasis(False)),
            power_mw=10.0,
            duration_hours=2.0,
            round_trip_efficiency=0.88,
            zone=ZONE,
            sample_window=SampleWindow(D1, D2, TZ),
            currency_basis=CURRENCY_DEFLATOR,
            forecast_audits=ForecastAudits(
                da=ForecastAudit("walk_forward", "hour_of_day", None),
                ida=ForecastAudit("walk_forward", "hour_of_day", None),
                reserve=ForecastAudit("walk_forward", "block_of_day_4h", None),
            ),
            reserve_product="aFRR",
            reserve_source="regelleistung",
            availability=0.95,
            reserve_coverage_audit=ReserveCoverageAudit((_reserve_entry(D1), _reserve_entry(D2))),
            coverage_audit=CoverageAudit((D1, D2), (D1, D2), (), (), ()),
            adapter_provenance=AdapterProvenance(
                ProducerAdapterId.PC_ADP_DA_ID_RESERVE,
                "simulate_sequential_da_id_reserve_batch",
                "realised_eur",
                ("global_ceiling_eur", "reserve_first_ceiling_eur"),
                None,
                False,
                0.5,
                None,
                None,
                None,
                None,
                "pc-market-grid-v1",
                {
                    "da": grid.da_profile_id(ZONE),
                    "ida": grid.ida_profile_id(ZONE),
                    "reserve": grid.reserve_profile_id(ZONE),
                },
            ),
            embedded_vom_cost_eur_mwh=0.5,
            source_data_content_hash="cd" * 32,
            calculator_version="pc-a-v1",
        )


def project_case(srr: StrategyRunResult | None = None) -> ProjectCase:
    """Canonical ProjectCase over the DA_ONLY strategy (golden-vector fixture)."""
    srr = srr or da_only_srr()
    return ProjectCase(
        AssetCase(10.0, 2.0, 0.88, 5_000_000.0, 12_000.0),
        LifecycleCase(
            15,
            CapacityMaintenanceBasis.SCHEDULED_NAMEPLATE_MAINTENANCE,
            "engineering_memo_v3",
            "2026-02-01",
            (AugmentationEvent(8, 800_000.0, 0.3, 20_000.0),),
            100_000.0,
            50_000.0,
        ),
        MarketCase(srr, Projection(ProjectionKind.DAOnlySpreadDecay, 0.02, 0.5)),
        ValuationCase(0.08, 2026),
        BootstrapCase(20260810, 5000, "pc-bootstrap-pcg64-choice365-linear-v1"),
    )


# --- Raw price data builders for adapter tests -------------------------------
def da_frame(zone: str, days: list[dt.date]) -> pd.DataFrame:
    idx, vals = [], []
    for d in days:
        for ts in grid.expected_da_timestamps(zone, d):
            idx.append(ts)
            vals.append(40.0 + (ts.hour % 12) * 3.0)
    return pd.DataFrame({"price_eur_mwh": vals}, index=pd.DatetimeIndex(idx))


def ida_frame(zone: str, days: list[dt.date]) -> pd.DataFrame:
    idx, vals = [], []
    for d in days:
        for ts in grid.expected_ida_timestamps(zone, d):
            idx.append(ts)
            vals.append(42.0 + (ts.hour % 12) * 3.0)
    return pd.DataFrame({"intraday_price_eur_mwh": vals}, index=pd.DatetimeIndex(idx))


def reserve_series(zone: str, days_blocks: list[tuple[dt.date, int]]) -> pd.Series:
    idx, vals = [], []
    for d, n in days_blocks:
        for bid, _ in grid.reserve_blocks(zone, d)[:n]:
            idx.append(pd.Timestamp(bid))
            vals.append(12.0)
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def fake_runner(
    cash_field: str,
    dates_values: list[tuple[dt.date, float]],
    *,
    failures: list[dict] | None = None,
    nested_failures: list[dict] | None = None,
):
    """A per-day-frame seam replacement for adapter unit tests (no real MILP).

    ``failures`` attaches a batch-style ``attrs['solver_failure_details']`` (the
    DA-only / joint-capacity convention); ``nested_failures`` attaches
    ``attrs['summary']['solver_failure_details']`` (the sequential-batch
    convention) so both audit-read paths can be exercised. Signature-agnostic so it
    can replace any ``adapters._run_*`` seam via monkeypatch.
    """

    def _run(*_args, **_kwargs):
        df = pd.DataFrame(
            {"date": [d for d, _ in dates_values], cash_field: [v for _, v in dates_values]}
        )
        if failures is not None:
            df.attrs["solver_failure_details"] = list(failures)
        if nested_failures is not None:
            df.attrs["summary"] = {"solver_failure_details": list(nested_failures)}
        return df

    return _run


def fake_coopt_runner(
    dates_values: list[tuple[dt.date, float]],
    *,
    failures: list[dict] | None = None,
):
    def _run(*_args, **_kwargs):
        df = pd.DataFrame(
            {"date": [d for d, _ in dates_values], "joint_total_revenue": [v for _, v in dates_values]}
        )
        if failures is not None:
            df.attrs["solver_failure_details"] = list(failures)
        return df

    return _run
