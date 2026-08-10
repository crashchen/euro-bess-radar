"""The four producer-issued solver adapters (contract §5, §9 PC-A).

Eligibility is a property of **which adapter emits** a StrategyRunResult, so an
ineligible quantity (a ceiling, a delta, an overlay) is unrepresentable. Each
adapter is a pinned 5-tuple ``ProducerAdapterId → StrategyKind → source function →
per-day cash field → excluded fields`` and reads **only** the per-day cash field
(red-lines #6/#18). A degraded reserve day is ``missing_dates`` — never relabelled
to another kind (§5); an empty ``valid_dates`` set is *unavailable*
(``AdapterUnavailableError``), never a €0 series (red-line #11).

The real solver is injectable (``runner=``) so the PC-A wiring can be unit-tested
without running a MILP; provenance always records the pinned canonical
``source_function`` regardless of injection.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from collections.abc import Callable

import pandas as pd

from src import config
from src.dispatch import DISPATCH_VOM_COST_EUR_MWH
from src.project_case import grid
from src.project_case.audit import (
    AdapterUnavailableError,
    build_reserve_coverage_audit,
    classify_leg_complete_dates,
    reserve_scalar_price,
)
from src.project_case.enums import (
    BUCKET_BLOCK_OF_DAY_4H,
    DA_ID_BUCKETS,
    RESERVE_PRICE_AGGREGATION_V1,
    WALK_FORWARD,
    CurrencyBasisMode,
    ProducerAdapterId,
)
from src.project_case.fingerprint import encode_value
from src.project_case.producer_specs import SPECS, AdapterSpec
from src.project_case.schema import (
    AdapterProvenance,
    CaptureBasis,
    CashBasis,
    CoverageAudit,
    CurrencyBasis,
    ForecastAudit,
    ForecastAudits,
    LiquidityBasis,
    ProjectCaseValidationError,
    ReserveCoverageAudit,
    SampleWindow,
    SolverFailureDetail,
    StrategyRunResult,
)

PC_A_CALCULATOR_VERSION = "pc-a-v1"

_DA_COL = "price_eur_mwh"
_IDA_COL = "intraday_price_eur_mwh"

Runner = Callable[[list[_dt.date]], pd.DataFrame]

# SPECS / AdapterSpec are the canonical immutable producer 5-tuples (§5), imported
# from ``producer_specs`` so schema validation and the adapters share one source.
__all__ = [
    "PC_A_CALCULATOR_VERSION",
    "SPECS",
    "AdapterSpec",
    "emit_da_id",
    "emit_da_id_reserve",
    "emit_da_only",
    "emit_reserve_coopt",
]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _require_supported_zone(zone: str) -> str:
    if zone not in set(config.ALL_ZONES.values()):
        raise AdapterUnavailableError(f"zone {zone!r} is not a supported code")
    return config.ZONE_TIMEZONES[zone]


def _column_series(df: pd.DataFrame, column: str, label: str) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or column not in df.columns:
        raise AdapterUnavailableError(f"{label} frame missing {column!r} column")
    return df[column]


def _restrict_frame(df: pd.DataFrame, tz: str, dates: frozenset[_dt.date]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    local_dates = pd.DatetimeIndex(idx.tz_convert(tz)).date
    mask = [d in dates for d in local_dates]
    return df.loc[mask]


def _content_hash(series: dict[str, pd.Series]) -> str:
    """Deterministic SHA-256 over the consumed raw data (sorted UTC pairs).

    Values are stringified with ``repr`` so a raw NaN/Inf on a (correctly
    classified) missing day yields a stable token and never trips the encoder's
    non-finite-float rejection — the source hash is opaque provenance, and a bad
    input day must not sink the whole result (a valid day still produces cash).
    """
    payload: dict[str, list[list[str]]] = {}
    for leg, s in sorted(series.items()):
        idx = pd.DatetimeIndex(s.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        pairs = sorted(
            [ts.strftime("%Y-%m-%dT%H:%M:%SZ"), repr(float(v))]
            for ts, v in zip(idx, s.to_numpy(dtype=float), strict=True)
        )
        payload[leg] = pairs
    return hashlib.sha256(encode_value(payload)).hexdigest()


def _partition(
    *,
    observed: tuple[_dt.date, ...],
    gate: frozenset[_dt.date],
    per_day: pd.DataFrame,
    cash_field: str,
    stage: str,
) -> tuple[
    tuple[tuple[_dt.date, float], ...],
    tuple[_dt.date, ...],
    tuple[_dt.date, ...],
    tuple[_dt.date, ...],
    tuple[SolverFailureDetail, ...],
]:
    """Turn a per-day solver frame into the audited date-set partition."""
    outputs: dict[_dt.date, float] = {}
    if per_day is not None and not per_day.empty:
        if cash_field not in per_day.columns:
            raise ProjectCaseValidationError(
                f"solver output missing pinned per-day cash field {cash_field!r}"
            )
        for _, row in per_day.iterrows():
            d = pd.Timestamp(row["date"]).date()
            value = float(row[cash_field])
            if d in gate and pd.notna(value) and d not in outputs:
                outputs[d] = value
    # Preserve the batch's real per-day solver-failure audit (dispatch-failure
    # contract) where the batch exposes it in attrs["solver_failure_details"];
    # fall back to an honest generic detail otherwise (never fabricate status).
    real_details: dict[_dt.date, dict] = {}
    for rec in (getattr(per_day, "attrs", {}) or {}).get("solver_failure_details", []) or []:
        try:
            rec_date = pd.Timestamp(rec.get("date")).date()
        except (ValueError, TypeError):
            continue
        real_details[rec_date] = rec
    valid = tuple(sorted(d for d in gate if d in outputs))
    series = tuple((d, outputs[d]) for d in valid)
    solver_failed = tuple(sorted(d for d in gate if d not in outputs))
    missing = tuple(sorted(d for d in observed if d not in gate))
    details = tuple(_failure_detail(d, real_details.get(d), stage) for d in solver_failed)
    return series, valid, missing, solver_failed, details


def _failure_detail(d: _dt.date, rec: dict | None, stage: str) -> SolverFailureDetail:
    """Honest ``SolverFailureDetail`` — real batch record when present, else generic."""
    if rec is not None:
        status = (str(rec.get("status") or "").strip()) or "solver_failed"
        message = (str(rec.get("message") or "").strip()) or "solver failure (no message)"
        return SolverFailureDetail(d, status, message, stage)
    return SolverFailureDetail(
        d,
        "no_output_for_data_complete_day",
        "data-complete day produced no per-day cash row: a solver failure, or a "
        "forecast-driven policy could not score it (e.g. walk-forward's first day "
        "has no prior-day history)",
        stage,
    )


def _build_result(
    *,
    spec: AdapterSpec,
    zone: str,
    sample_window: SampleWindow,
    series: tuple[tuple[_dt.date, float], ...],
    observed: tuple[_dt.date, ...],
    valid: tuple[_dt.date, ...],
    missing: tuple[_dt.date, ...],
    solver_failed: tuple[_dt.date, ...],
    details: tuple[SolverFailureDetail, ...],
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    cash_basis: CashBasis,
    currency_basis: CurrencyBasis,
    forecast_audits: ForecastAudits,
    reserve_product: str | None,
    reserve_source: str | None,
    availability: float | None,
    reserve_coverage_audit: ReserveCoverageAudit | None,
    capture_rate: float | None,
    reserve_price_aggregation: str | None,
    reserve_pricing_dates: tuple[_dt.date, ...] | None,
    reserve_scalar_price_eur_mw_h: float | None,
    mode: str | None,
    source_data_content_hash: str,
) -> StrategyRunResult:
    if not valid:
        raise AdapterUnavailableError(
            f"{spec.producer_adapter_id.value}: no valid dates (result unavailable)"
        )
    # Currency basis actually converts the cash: DEFLATOR_APPLIED means the adapter
    # scales historical settlement EUR to base-year real EUR by the recorded factor
    # (a stamped-but-unapplied deflator would be a lie; §4.3, red-line #19).
    if currency_basis.mode is CurrencyBasisMode.DEFLATOR_APPLIED:
        factor = float(currency_basis.deflator_factor)
        series = tuple((d, v * factor) for d, v in series)
    coverage = CoverageAudit(observed, valid, missing, solver_failed, details)
    profiles = {
        "da": grid.da_profile_id(zone),
        "ida": grid.ida_profile_id(zone) if spec.consumes_ida else None,
        "reserve": grid.reserve_profile_id(zone) if spec.consumes_reserve else None,
    }
    provenance = AdapterProvenance(
        producer_adapter_id=spec.producer_adapter_id,
        source_function=spec.source_function,
        per_day_cash_field=spec.per_day_cash_field,
        excluded_fields=spec.excluded_fields,
        mode=mode,
        carry_soc=False,
        soc_init_frac=0.5,
        capture_rate=capture_rate,
        reserve_price_aggregation=reserve_price_aggregation,
        reserve_pricing_dates=reserve_pricing_dates,
        reserve_scalar_price_eur_mw_h=reserve_scalar_price_eur_mw_h,
        expected_grid_registry_version=grid.REGISTRY_VERSION,
        expected_grid_profiles=profiles,
    )
    return StrategyRunResult(
        strategy_kind=spec.strategy_kind,
        daily_realised_cash_series=series,
        cash_basis=cash_basis,
        power_mw=power_mw,
        duration_hours=duration_hours,
        round_trip_efficiency=efficiency,
        zone=zone,
        sample_window=sample_window,
        currency_basis=currency_basis,
        forecast_audits=forecast_audits,
        reserve_product=reserve_product,
        reserve_source=reserve_source,
        availability=availability,
        reserve_coverage_audit=reserve_coverage_audit,
        coverage_audit=coverage,
        adapter_provenance=provenance,
        embedded_vom_cost_eur_mwh=float(DISPATCH_VOM_COST_EUR_MWH),
        source_data_content_hash=source_data_content_hash,
        calculator_version=PC_A_CALCULATOR_VERSION,
    )


# --------------------------------------------------------------------------- #
# PC_ADP_DA_ONLY                                                              #
# --------------------------------------------------------------------------- #
def emit_da_only(
    da_prices: pd.DataFrame,
    *,
    zone: str,
    first_delivery_date: _dt.date,
    last_delivery_date: _dt.date,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    currency_basis: CurrencyBasis,
    capture_rate: float = 1.0,
    capture_source: str = "not_applied",
    runner: Runner | None = None,
) -> StrategyRunResult:
    """Emit a ``DA_ONLY`` StrategyRunResult (reads only ``total_revenue_eur``)."""
    spec = SPECS[ProducerAdapterId.PC_ADP_DA_ONLY]
    tz = _require_supported_zone(zone)
    sample_window = SampleWindow(first_delivery_date, last_delivery_date, tz)
    observed = sample_window.evaluation_dates()
    da_series = _column_series(da_prices, _DA_COL, "DA")
    gate = classify_leg_complete_dates(
        da_series, zone=zone, leg="da", evaluation_dates=observed
    )

    def _default(gate_dates: list[_dt.date]) -> pd.DataFrame:
        from src.simulation import simulate_replay_batch

        return simulate_replay_batch(
            da_prices,
            mode="DA MILP Replay",
            tz=tz,
            dates=list(gate_dates),
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            capture_rate=capture_rate,
            soc_init_frac=0.5,
            carry_soc=False,
        )

    per_day = (runner or _default)(sorted(gate))
    series, valid, missing, solver_failed, details = _partition(
        observed=observed, gate=gate, per_day=per_day,
        cash_field=spec.per_day_cash_field, stage=spec.source_function,
    )
    capture = CaptureBasis(applied=(float(capture_rate) != 1.0), rate=capture_rate, source=capture_source)
    return _build_result(
        spec=spec, zone=zone, sample_window=sample_window, series=series,
        observed=observed, valid=valid, missing=missing, solver_failed=solver_failed,
        details=details, power_mw=power_mw, duration_hours=duration_hours,
        efficiency=efficiency,
        cash_basis=CashBasis(True, capture, LiquidityBasis(False)),
        currency_basis=currency_basis, forecast_audits=ForecastAudits(),
        reserve_product=None, reserve_source=None, availability=None,
        reserve_coverage_audit=None, capture_rate=float(capture_rate),
        reserve_price_aggregation=None, reserve_pricing_dates=None,
        reserve_scalar_price_eur_mw_h=None, mode="DA MILP Replay",
        source_data_content_hash=_content_hash({"da": da_series}),
    )


# --------------------------------------------------------------------------- #
# PC_ADP_DA_ID                                                                #
# --------------------------------------------------------------------------- #
def emit_da_id(
    da_prices: pd.DataFrame,
    ida_prices: pd.DataFrame,
    *,
    zone: str,
    first_delivery_date: _dt.date,
    last_delivery_date: _dt.date,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    currency_basis: CurrencyBasis,
    bucket: str,
    min_rebid_uplift_eur: float,
    runner: Runner | None = None,
) -> StrategyRunResult:
    """Emit a ``DA_ID_FORECAST`` StrategyRunResult (walk-forward; reads ``realised_eur``)."""
    spec = SPECS[ProducerAdapterId.PC_ADP_DA_ID]
    if bucket not in DA_ID_BUCKETS:
        raise ProjectCaseValidationError("bucket must be hour_of_day|hour_of_week")
    tz = _require_supported_zone(zone)
    sample_window = SampleWindow(first_delivery_date, last_delivery_date, tz)
    observed = sample_window.evaluation_dates()
    da_series = _column_series(da_prices, _DA_COL, "DA")
    ida_series = _column_series(ida_prices, _IDA_COL, "IDA")
    gate = classify_leg_complete_dates(da_series, zone=zone, leg="da", evaluation_dates=observed) & \
        classify_leg_complete_dates(ida_series, zone=zone, leg="ida", evaluation_dates=observed)

    def _default(gate_dates: list[_dt.date]) -> pd.DataFrame:
        from src.simulation import simulate_sequential_da_id_batch

        per_day, _ = simulate_sequential_da_id_batch(
            da_prices, ida_prices, dates=list(gate_dates), tz=tz,
            power_mw=power_mw, duration_hours=duration_hours, efficiency=efficiency,
            bucket=bucket, forecast_mode=WALK_FORWARD,
            min_rebid_uplift_eur=min_rebid_uplift_eur,
            soc_init_frac=0.5,  # bound explicitly (red-line #22), not inherited
        )
        return per_day

    per_day = (runner or _default)(sorted(gate))
    series, valid, missing, solver_failed, details = _partition(
        observed=observed, gate=gate, per_day=per_day,
        cash_field=spec.per_day_cash_field, stage=spec.source_function,
    )
    return _build_result(
        spec=spec, zone=zone, sample_window=sample_window, series=series,
        observed=observed, valid=valid, missing=missing, solver_failed=solver_failed,
        details=details, power_mw=power_mw, duration_hours=duration_hours,
        efficiency=efficiency,
        cash_basis=CashBasis(True, CaptureBasis(False, 1.0, "not_applied"), LiquidityBasis(False)),
        currency_basis=currency_basis,
        forecast_audits=ForecastAudits(
            ida=ForecastAudit(WALK_FORWARD, bucket, float(min_rebid_uplift_eur))
        ),
        reserve_product=None, reserve_source=None, availability=None,
        reserve_coverage_audit=None, capture_rate=None,
        reserve_price_aggregation=None, reserve_pricing_dates=None,
        reserve_scalar_price_eur_mw_h=None, mode=None,
        source_data_content_hash=_content_hash({"da": da_series, "ida": ida_series}),
    )


# --------------------------------------------------------------------------- #
# PC_ADP_RESERVE_COOPT                                                        #
# --------------------------------------------------------------------------- #
def emit_reserve_coopt(
    da_prices: pd.DataFrame,
    reserve_block_prices: pd.Series,
    *,
    zone: str,
    first_delivery_date: _dt.date,
    last_delivery_date: _dt.date,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    currency_basis: CurrencyBasis,
    reserve_product: str,
    reserve_source: str,
    availability: float = config.ANCILLARY_CAPACITY_AVAILABILITY,
    runner: Callable[[list[_dt.date], float], pd.DataFrame] | None = None,
) -> StrategyRunResult:
    """Emit a ``DA_RESERVE_COOPT`` StrategyRunResult (reads ``joint_total_revenue``)."""
    spec = SPECS[ProducerAdapterId.PC_ADP_RESERVE_COOPT]
    tz = _require_supported_zone(zone)
    sample_window = SampleWindow(first_delivery_date, last_delivery_date, tz)
    observed = sample_window.evaluation_dates()
    da_series = _column_series(da_prices, _DA_COL, "DA")
    da_complete = classify_leg_complete_dates(da_series, zone=zone, leg="da", evaluation_dates=observed)
    reserve_audit = build_reserve_coverage_audit(
        reserve_block_prices, zone=zone, evaluation_dates=observed
    )
    covered = reserve_audit.covered_dates
    # pricing_dates = DA-complete ∩ reserve-fully-covered (pre-gate before collapse).
    pricing_dates = frozenset(d for d in da_complete if d in covered)
    if not pricing_dates:
        raise AdapterUnavailableError("PC_ADP_RESERVE_COOPT: no DA-complete reserve-covered dates")
    pricing_tuple = tuple(sorted(pricing_dates))
    scalar = reserve_scalar_price(reserve_block_prices, zone=zone, pricing_dates=pricing_tuple)

    def _default(gate_dates: list[_dt.date], reserve_scalar: float) -> pd.DataFrame:
        from src.dispatch import solve_joint_capacity_batch

        restricted = _restrict_frame(da_prices, tz, frozenset(gate_dates))
        return solve_joint_capacity_batch(
            restricted, capacity_price_eur_mw_h=reserve_scalar,
            power_mw=power_mw, duration_hours=duration_hours, efficiency=efficiency,
            tz=tz, soc_init_frac=0.5, availability=availability,
        )

    run = runner or _default
    per_day = run(list(pricing_tuple), scalar)
    series, valid, missing, solver_failed, details = _partition(
        observed=observed, gate=pricing_dates, per_day=per_day,
        cash_field=spec.per_day_cash_field, stage=spec.source_function,
    )
    return _build_result(
        spec=spec, zone=zone, sample_window=sample_window, series=series,
        observed=observed, valid=valid, missing=missing, solver_failed=solver_failed,
        details=details, power_mw=power_mw, duration_hours=duration_hours,
        efficiency=efficiency,
        cash_basis=CashBasis(True, CaptureBasis(False, 1.0, "not_applied"), LiquidityBasis(False)),
        currency_basis=currency_basis, forecast_audits=ForecastAudits(),
        reserve_product=reserve_product, reserve_source=reserve_source,
        availability=availability, reserve_coverage_audit=reserve_audit,
        capture_rate=None, reserve_price_aggregation=RESERVE_PRICE_AGGREGATION_V1,
        reserve_pricing_dates=pricing_tuple, reserve_scalar_price_eur_mw_h=scalar,
        mode=None,
        source_data_content_hash=_content_hash(
            {"da": da_series, "reserve": _block_series(reserve_block_prices)}
        ),
    )


# --------------------------------------------------------------------------- #
# PC_ADP_DA_ID_RESERVE                                                        #
# --------------------------------------------------------------------------- #
def emit_da_id_reserve(
    da_prices: pd.DataFrame,
    ida_prices: pd.DataFrame,
    reserve_block_prices: pd.Series,
    *,
    zone: str,
    first_delivery_date: _dt.date,
    last_delivery_date: _dt.date,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    currency_basis: CurrencyBasis,
    reserve_product: str,
    reserve_source: str,
    bucket: str,
    availability: float = config.ANCILLARY_CAPACITY_AVAILABILITY,
    runner: Runner | None = None,
) -> StrategyRunResult:
    """Emit a ``DA_ID_RESERVE_REALISED`` StrategyRunResult (walk-forward; ``realised_eur``)."""
    spec = SPECS[ProducerAdapterId.PC_ADP_DA_ID_RESERVE]
    if bucket not in DA_ID_BUCKETS:
        raise ProjectCaseValidationError("bucket must be hour_of_day|hour_of_week")
    tz = _require_supported_zone(zone)
    sample_window = SampleWindow(first_delivery_date, last_delivery_date, tz)
    observed = sample_window.evaluation_dates()
    da_series = _column_series(da_prices, _DA_COL, "DA")
    ida_series = _column_series(ida_prices, _IDA_COL, "IDA")
    reserve_audit = build_reserve_coverage_audit(
        reserve_block_prices, zone=zone, evaluation_dates=observed
    )
    covered = reserve_audit.covered_dates
    da_complete = classify_leg_complete_dates(da_series, zone=zone, leg="da", evaluation_dates=observed)
    ida_complete = classify_leg_complete_dates(ida_series, zone=zone, leg="ida", evaluation_dates=observed)
    gate = frozenset(d for d in (da_complete & ida_complete) if d in covered)

    def _default(gate_dates: list[_dt.date]) -> pd.DataFrame:
        from src.simulation import simulate_sequential_da_id_reserve_batch

        per_day, _ = simulate_sequential_da_id_reserve_batch(
            da_prices, ida_prices, _block_series(reserve_block_prices),
            dates=list(gate_dates), tz=tz, power_mw=power_mw,
            duration_hours=duration_hours, efficiency=efficiency,
            availability=availability, bucket=bucket, forecast_mode=WALK_FORWARD,
            soc_init_frac=0.5,  # bound explicitly (red-line #22), not inherited
        )
        return per_day

    per_day = (runner or _default)(sorted(gate))
    series, valid, missing, solver_failed, details = _partition(
        observed=observed, gate=gate, per_day=per_day,
        cash_field=spec.per_day_cash_field, stage=spec.source_function,
    )
    return _build_result(
        spec=spec, zone=zone, sample_window=sample_window, series=series,
        observed=observed, valid=valid, missing=missing, solver_failed=solver_failed,
        details=details, power_mw=power_mw, duration_hours=duration_hours,
        efficiency=efficiency,
        cash_basis=CashBasis(True, CaptureBasis(False, 1.0, "not_applied"), LiquidityBasis(False)),
        currency_basis=currency_basis,
        forecast_audits=ForecastAudits(
            da=ForecastAudit(WALK_FORWARD, bucket, None),
            ida=ForecastAudit(WALK_FORWARD, bucket, None),
            reserve=ForecastAudit(WALK_FORWARD, BUCKET_BLOCK_OF_DAY_4H, None),
        ),
        reserve_product=reserve_product, reserve_source=reserve_source,
        availability=availability, reserve_coverage_audit=reserve_audit,
        capture_rate=None, reserve_price_aggregation=None,
        reserve_pricing_dates=None, reserve_scalar_price_eur_mw_h=None, mode=None,
        source_data_content_hash=_content_hash(
            {"da": da_series, "ida": ida_series, "reserve": _block_series(reserve_block_prices)}
        ),
    )


def _block_series(block_prices: pd.Series) -> pd.Series:
    if not isinstance(block_prices, pd.Series):
        raise AdapterUnavailableError("reserve_block_prices must be a pandas Series")
    return block_prices
