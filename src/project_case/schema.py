"""Typed Project Case v1 schema (contract §4) with domain validation + payloads.

Every ``to_payload()`` returns the exact recursive CBOR key map of §4.8 (the
production payload registry) with correctly-typed Python values, so the
``fingerprint`` module can encode it deterministically. Domain validation runs at
construction (fail-closed, §4.6, red-line #11/#15); cross-object invariants run in
``ProjectCase.validate()`` and ``StrategyRunResult.validate()``.

PC-A is pure schema + adapters + fingerprint; it computes no NPV (PC-B).
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from src import config
from src.dispatch import DISPATCH_VOM_COST_EUR_MWH
from src.project_case.enums import (
    BOOTSTRAP_ALGORITHM_V1,
    BUCKET_BLOCK_OF_DAY_4H,
    DA_ID_BUCKETS,
    EXPECTED_GRID_REGISTRY_VERSION,
    MAINTENANCE_PROVENANCE_MODE,
    MAX_BASE_YEAR,
    MAX_PROJECT_LIFE_YEARS,
    MAX_SEED,
    MAX_SIMULATIONS,
    MIN_BASE_YEAR,
    MIN_SIMULATIONS,
    RESERVE_PRICE_AGGREGATION_V1,
    WALK_FORWARD,
    CapacityMaintenanceBasis,
    CurrencyBasisMode,
    ProducerAdapterId,
    ProjectionKind,
    StrategyKind,
)
from src.project_case.fingerprint import (
    SCHEMA_VERSION,
    encode_value,
    fingerprint_hex,
    sorted_by_encoding,
)
from src.project_case.producer_specs import canonical_spec

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_BLOCK_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_RESERVE_KINDS = frozenset(
    {StrategyKind.DA_RESERVE_COOPT, StrategyKind.DA_ID_RESERVE_REALISED}
)


class ProjectCaseValidationError(ValueError):
    """Raised for any schema/domain/cross-invariant violation (fail-closed)."""


# --------------------------------------------------------------------------- #
# Primitive validators                                                        #
# --------------------------------------------------------------------------- #
def _finite_float(x: Any, name: str) -> float:
    if isinstance(x, bool):
        raise ProjectCaseValidationError(f"{name} must be a real number, not bool")
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise ProjectCaseValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(v):
        raise ProjectCaseValidationError(f"{name} must be finite (got {x!r})")
    return v


def _nonneg_float(x: Any, name: str) -> float:
    v = _finite_float(x, name)
    if v < 0.0:
        raise ProjectCaseValidationError(f"{name} must be >= 0 (got {v})")
    return v


def _pos_float(x: Any, name: str) -> float:
    v = _finite_float(x, name)
    if v <= 0.0:
        raise ProjectCaseValidationError(f"{name} must be > 0 (got {v})")
    return v


def _ratio_float(x: Any, name: str) -> float:
    v = _finite_float(x, name)
    if not 0.0 <= v <= 1.0:
        raise ProjectCaseValidationError(f"{name} must be in [0, 1] (got {v})")
    return v


def _int_in(x: Any, lo: int, hi: int, name: str) -> int:
    if isinstance(x, bool):
        raise ProjectCaseValidationError(f"{name} must be an int, not bool")
    if not isinstance(x, int):
        raise ProjectCaseValidationError(f"{name} must be an int (got {type(x).__name__})")
    if not lo <= x <= hi:
        raise ProjectCaseValidationError(f"{name} must be in [{lo}, {hi}] (got {x})")
    return x


def _text(x: Any, name: str) -> str:
    if not isinstance(x, str):
        raise ProjectCaseValidationError(f"{name} must be text (got {type(x).__name__})")
    if not x.strip():
        raise ProjectCaseValidationError(f"{name} must be non-empty after trimming")
    return x


def _hex64(x: Any, name: str) -> str:
    if not isinstance(x, str) or not _HEX64_RE.match(x):
        raise ProjectCaseValidationError(
            f"{name} must be a lowercase 64-character hex digest"
        )
    return x


def _as_date(x: Any, name: str) -> _dt.date:
    if isinstance(x, _dt.datetime):
        return x.date()
    if isinstance(x, _dt.date):
        return x
    if isinstance(x, str) and _ISO_DATE_RE.match(x):
        try:
            return _dt.date.fromisoformat(x)
        except ValueError as exc:
            raise ProjectCaseValidationError(f"{name} is not a valid date") from exc
    raise ProjectCaseValidationError(f"{name} must be a date or YYYY-MM-DD string")


def _iso(d: _dt.date) -> str:
    return d.isoformat()


def _block_id(x: Any, name: str) -> str:
    if not isinstance(x, str) or not _BLOCK_ID_RE.match(x):
        raise ProjectCaseValidationError(
            f"{name} must be a UTC block id YYYY-MM-DDTHH:MM:SSZ"
        )
    return x


def _date_tuple(values: Any, name: str) -> tuple[_dt.date, ...]:
    """Coerce an iterable of dates into a sorted, unique tuple of ``date``."""
    out = [_as_date(v, name) for v in values]
    uniq = sorted(set(out))
    if len(uniq) != len(out):
        raise ProjectCaseValidationError(f"{name} must contain unique dates")
    return tuple(uniq)


def _iso_date_array(dates: tuple[_dt.date, ...]) -> list[str]:
    return sorted_by_encoding([_iso(d) for d in dates])


# --------------------------------------------------------------------------- #
# Cash basis                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CaptureBasis:
    """``{applied, rate, source}`` capture haircut basis (§4.3)."""

    applied: bool
    rate: float
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.applied, bool):
            raise ProjectCaseValidationError("capture.applied must be bool")
        object.__setattr__(self, "rate", _ratio_float(self.rate, "capture.rate"))
        _text(self.source, "capture.source")

    def to_payload(self) -> dict[str, Any]:
        return {"applied": self.applied, "rate": float(self.rate), "source": self.source}


@dataclass(frozen=True)
class LiquidityBasis:
    """``{applied, assumption_fingerprint}`` (§4.3). Fingerprint required iff applied."""

    applied: bool
    assumption_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.applied, bool):
            raise ProjectCaseValidationError("liquidity.applied must be bool")
        if self.applied:
            _hex64(self.assumption_fingerprint, "liquidity.assumption_fingerprint")
        elif self.assumption_fingerprint is not None:
            raise ProjectCaseValidationError(
                "liquidity.assumption_fingerprint must be null when not applied"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "assumption_fingerprint": self.assumption_fingerprint,
        }


@dataclass(frozen=True)
class CashBasis:
    """``{post_vom, capture, liquidity}`` (§4.3). ``post_vom`` must be True."""

    post_vom: bool
    capture: CaptureBasis
    liquidity: LiquidityBasis

    def __post_init__(self) -> None:
        if self.post_vom is not True:
            raise ProjectCaseValidationError("cash_basis.post_vom must be True")
        if not isinstance(self.capture, CaptureBasis):
            raise ProjectCaseValidationError("cash_basis.capture must be a CaptureBasis")
        if not isinstance(self.liquidity, LiquidityBasis):
            raise ProjectCaseValidationError(
                "cash_basis.liquidity must be a LiquidityBasis"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "post_vom": self.post_vom,
            "capture": self.capture.to_payload(),
            "liquidity": self.liquidity.to_payload(),
        }


@dataclass(frozen=True)
class CurrencyBasis:
    """``{mode, target_base_year, deflator_method, deflator_vintage, deflator_factor}``."""

    mode: CurrencyBasisMode
    target_base_year: int
    deflator_method: str | None = None
    deflator_vintage: str | None = None
    deflator_factor: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CurrencyBasisMode):
            raise ProjectCaseValidationError("currency_basis.mode invalid")
        _int_in(
            self.target_base_year,
            MIN_BASE_YEAR,
            MAX_BASE_YEAR,
            "currency_basis.target_base_year",
        )
        if self.mode is CurrencyBasisMode.DEFLATOR_APPLIED:
            _text(self.deflator_method, "currency_basis.deflator_method")
            _text(self.deflator_vintage, "currency_basis.deflator_vintage")
            object.__setattr__(
                self,
                "deflator_factor",
                _pos_float(self.deflator_factor, "currency_basis.deflator_factor"),
            )
        else:
            if (
                self.deflator_method is not None
                or self.deflator_vintage is not None
                or self.deflator_factor is not None
            ):
                raise ProjectCaseValidationError(
                    "currency_basis deflator members must be null unless DEFLATOR_APPLIED"
                )

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "target_base_year": int(self.target_base_year),
            "deflator_method": self.deflator_method,
            "deflator_vintage": self.deflator_vintage,
            "deflator_factor": (
                None if self.deflator_factor is None else float(self.deflator_factor)
            ),
        }


@dataclass(frozen=True)
class SampleWindow:
    """``{first_delivery_date, last_delivery_date, timezone}`` (§4.3)."""

    first_delivery_date: _dt.date
    last_delivery_date: _dt.date
    timezone: str

    def __post_init__(self) -> None:
        first = _as_date(self.first_delivery_date, "sample_window.first_delivery_date")
        last = _as_date(self.last_delivery_date, "sample_window.last_delivery_date")
        if first > last:
            raise ProjectCaseValidationError(
                "sample_window.first_delivery_date must be <= last_delivery_date"
            )
        object.__setattr__(self, "first_delivery_date", first)
        object.__setattr__(self, "last_delivery_date", last)
        _text(self.timezone, "sample_window.timezone")

    def evaluation_dates(self) -> tuple[_dt.date, ...]:
        """Every inclusive local delivery date in the window (§4.3 universe)."""
        out: list[_dt.date] = []
        cur = self.first_delivery_date
        while cur <= self.last_delivery_date:
            out.append(cur)
            cur += _dt.timedelta(days=1)
        return tuple(out)

    def to_payload(self) -> dict[str, Any]:
        return {
            "first_delivery_date": _iso(self.first_delivery_date),
            "last_delivery_date": _iso(self.last_delivery_date),
            "timezone": self.timezone,
        }


# --------------------------------------------------------------------------- #
# Forecast audits                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ForecastAudit:
    """Per-leg forecast provenance ``{forecast_mode, bucket, deadband}`` (§4.3)."""

    forecast_mode: str
    bucket: str
    deadband: float | None

    def __post_init__(self) -> None:
        if self.forecast_mode != WALK_FORWARD:
            raise ProjectCaseValidationError(
                "forecast_mode must be 'walk_forward' for a cash-eligible strategy"
            )
        _text(self.bucket, "forecast_audit.bucket")
        if self.deadband is not None:
            object.__setattr__(
                self, "deadband", _nonneg_float(self.deadband, "forecast_audit.deadband")
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "forecast_mode": self.forecast_mode,
            "bucket": self.bucket,
            "deadband": (None if self.deadband is None else float(self.deadband)),
        }


@dataclass(frozen=True)
class ForecastAudits:
    """``{da, ida, reserve}`` per-leg audits; each leg is a ForecastAudit or null."""

    da: ForecastAudit | None = None
    ida: ForecastAudit | None = None
    reserve: ForecastAudit | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "da": None if self.da is None else self.da.to_payload(),
            "ida": None if self.ida is None else self.ida.to_payload(),
            "reserve": None if self.reserve is None else self.reserve.to_payload(),
        }


# --------------------------------------------------------------------------- #
# Reserve coverage audit                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReserveCoverageEntry:
    """One per-day reserve-block coverage entry (§4.3, red-line #21)."""

    date: _dt.date
    required_blocks: tuple[str, ...]
    present_blocks: tuple[str, ...]
    missing_blocks: tuple[str, ...]
    settlement_duration_hours_by_block: dict[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "date", _as_date(self.date, "reserve entry date"))
        req = frozenset(_block_id(b, "required_blocks") for b in self.required_blocks)
        pres = frozenset(_block_id(b, "present_blocks") for b in self.present_blocks)
        miss = frozenset(_block_id(b, "missing_blocks") for b in self.missing_blocks)
        if len(req) != len(self.required_blocks):
            raise ProjectCaseValidationError("required_blocks must be unique")
        if pres | miss != req:
            raise ProjectCaseValidationError(
                "present_blocks | missing_blocks must equal required_blocks"
            )
        if pres & miss:
            raise ProjectCaseValidationError(
                "present_blocks and missing_blocks must be disjoint"
            )
        durations = {
            _block_id(k, "duration key"): _pos_float(v, "settlement_duration_hours")
            for k, v in self.settlement_duration_hours_by_block.items()
        }
        if frozenset(durations) != req:
            raise ProjectCaseValidationError(
                "settlement_duration_hours_by_block keys must equal required_blocks"
            )
        object.__setattr__(self, "required_blocks", tuple(sorted(req)))
        object.__setattr__(self, "present_blocks", tuple(sorted(pres)))
        object.__setattr__(self, "missing_blocks", tuple(sorted(miss)))
        # Read-only view so a fingerprint-bearing object stays deeply immutable.
        object.__setattr__(
            self, "settlement_duration_hours_by_block", MappingProxyType(durations)
        )

    @property
    def fully_covered(self) -> bool:
        return not self.missing_blocks and (
            frozenset(self.present_blocks) == frozenset(self.required_blocks)
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "date": _iso(self.date),
            "required_blocks": sorted_by_encoding(self.required_blocks),
            "present_blocks": sorted_by_encoding(self.present_blocks),
            "missing_blocks": sorted_by_encoding(self.missing_blocks),
            "settlement_duration_hours_by_block": {
                k: float(v)
                for k, v in self.settlement_duration_hours_by_block.items()
            },
        }


@dataclass(frozen=True)
class ReserveCoverageAudit:
    """Sorted array of one ReserveCoverageEntry per observed date (§4.3)."""

    entries: tuple[ReserveCoverageEntry, ...]

    def __post_init__(self) -> None:
        entries = tuple(sorted(self.entries, key=lambda e: e.date))
        dates = [e.date for e in entries]
        if len(set(dates)) != len(dates):
            raise ProjectCaseValidationError("reserve_coverage_audit dates must be unique")
        object.__setattr__(self, "entries", entries)

    @property
    def covered_dates(self) -> frozenset[_dt.date]:
        return frozenset(e.date for e in self.entries if e.fully_covered)

    def date_set(self) -> frozenset[_dt.date]:
        return frozenset(e.date for e in self.entries)

    def to_payload(self) -> list[dict[str, Any]]:
        return [e.to_payload() for e in self.entries]


# --------------------------------------------------------------------------- #
# Coverage audit                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SolverFailureDetail:
    """``{date, status, message, stage}`` (§4.3)."""

    date: _dt.date
    status: str
    message: str
    stage: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "date", _as_date(self.date, "solver failure date"))
        _text(self.status, "solver_failure.status")
        _text(self.message, "solver_failure.message")
        _text(self.stage, "solver_failure.stage")

    def _sort_key(self) -> tuple[str, str, str, str]:
        return (_iso(self.date), self.stage, self.status, self.message)

    def to_payload(self) -> dict[str, Any]:
        return {
            "date": _iso(self.date),
            "status": self.status,
            "message": self.message,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class CoverageAudit:
    """Date-set partition of the evaluation universe (§4.3, red-line #17)."""

    observed_dates: tuple[_dt.date, ...]
    valid_dates: tuple[_dt.date, ...]
    missing_dates: tuple[_dt.date, ...]
    solver_failed_dates: tuple[_dt.date, ...]
    solver_failure_details: tuple[SolverFailureDetail, ...]

    def __post_init__(self) -> None:
        observed = _date_tuple(self.observed_dates, "observed_dates")
        valid = _date_tuple(self.valid_dates, "valid_dates")
        missing = _date_tuple(self.missing_dates, "missing_dates")
        failed = _date_tuple(self.solver_failed_dates, "solver_failed_dates")
        s_obs, s_val, s_mis, s_fail = (
            frozenset(observed),
            frozenset(valid),
            frozenset(missing),
            frozenset(failed),
        )
        if s_val | s_mis | s_fail != s_obs:
            raise ProjectCaseValidationError(
                "valid|missing|solver_failed must cover observed_dates"
            )
        if s_val & s_mis or s_val & s_fail or s_mis & s_fail:
            raise ProjectCaseValidationError(
                "valid/missing/solver_failed date sets must be pairwise disjoint"
            )
        details = tuple(self.solver_failure_details)
        if frozenset(d.date for d in details) != s_fail or len(details) != len(failed):
            raise ProjectCaseValidationError(
                "solver_failure_details must have exactly one entry per solver_failed date"
            )
        object.__setattr__(self, "observed_dates", observed)
        object.__setattr__(self, "valid_dates", valid)
        object.__setattr__(self, "missing_dates", missing)
        object.__setattr__(self, "solver_failed_dates", failed)
        object.__setattr__(
            self,
            "solver_failure_details",
            tuple(sorted(details, key=lambda d: d._sort_key())),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "observed_dates": _iso_date_array(self.observed_dates),
            "valid_dates": _iso_date_array(self.valid_dates),
            "missing_dates": _iso_date_array(self.missing_dates),
            "solver_failed_dates": _iso_date_array(self.solver_failed_dates),
            "solver_failure_details": [d.to_payload() for d in self.solver_failure_details],
        }


# --------------------------------------------------------------------------- #
# Adapter provenance                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdapterProvenance:
    """Fingerprinted adapter provenance submap (§4.8)."""

    producer_adapter_id: ProducerAdapterId
    source_function: str
    per_day_cash_field: str
    excluded_fields: tuple[str, ...]
    mode: str | None
    carry_soc: bool
    soc_init_frac: float
    capture_rate: float | None
    reserve_price_aggregation: str | None
    reserve_pricing_dates: tuple[_dt.date, ...] | None
    reserve_scalar_price_eur_mw_h: float | None
    expected_grid_registry_version: str
    expected_grid_profiles: dict[str, str | None]

    def __post_init__(self) -> None:
        if not isinstance(self.producer_adapter_id, ProducerAdapterId):
            raise ProjectCaseValidationError("producer_adapter_id invalid")
        _text(self.source_function, "source_function")
        _text(self.per_day_cash_field, "per_day_cash_field")
        for f in self.excluded_fields:
            _text(f, "excluded_fields[]")
        if self.mode is not None:
            _text(self.mode, "adapter_provenance.mode")
        if not isinstance(self.carry_soc, bool):
            raise ProjectCaseValidationError("carry_soc must be bool")
        object.__setattr__(self, "soc_init_frac", _ratio_float(self.soc_init_frac, "soc_init_frac"))
        if self.capture_rate is not None:
            object.__setattr__(self, "capture_rate", _ratio_float(self.capture_rate, "capture_rate"))
        if self.reserve_price_aggregation is not None:
            _text(self.reserve_price_aggregation, "reserve_price_aggregation")
        if self.reserve_pricing_dates is not None:
            object.__setattr__(
                self,
                "reserve_pricing_dates",
                _date_tuple(self.reserve_pricing_dates, "reserve_pricing_dates"),
            )
        if self.reserve_scalar_price_eur_mw_h is not None:
            object.__setattr__(
                self,
                "reserve_scalar_price_eur_mw_h",
                _nonneg_float(
                    self.reserve_scalar_price_eur_mw_h, "reserve_scalar_price_eur_mw_h"
                ),
            )
        if self.expected_grid_registry_version != EXPECTED_GRID_REGISTRY_VERSION:
            raise ProjectCaseValidationError(
                f"expected_grid_registry_version must be {EXPECTED_GRID_REGISTRY_VERSION!r}"
            )
        if set(self.expected_grid_profiles) != {"da", "ida", "reserve"}:
            raise ProjectCaseValidationError(
                "expected_grid_profiles must have exactly {da, ida, reserve}"
            )
        for leg, prof in self.expected_grid_profiles.items():
            if prof is not None:
                _text(prof, f"expected_grid_profiles.{leg}")
        # Read-only view so a fingerprint-bearing object stays deeply immutable.
        object.__setattr__(
            self, "expected_grid_profiles", MappingProxyType(dict(self.expected_grid_profiles))
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "producer_adapter_id": self.producer_adapter_id.value,
            "source_function": self.source_function,
            "per_day_cash_field": self.per_day_cash_field,
            "excluded_fields": sorted_by_encoding(list(self.excluded_fields)),
            "mode": self.mode,
            "carry_soc": self.carry_soc,
            "soc_init_frac": float(self.soc_init_frac),
            "capture_rate": (None if self.capture_rate is None else float(self.capture_rate)),
            "reserve_price_aggregation": self.reserve_price_aggregation,
            "reserve_pricing_dates": (
                None
                if self.reserve_pricing_dates is None
                else _iso_date_array(self.reserve_pricing_dates)
            ),
            "reserve_scalar_price_eur_mw_h": (
                None
                if self.reserve_scalar_price_eur_mw_h is None
                else float(self.reserve_scalar_price_eur_mw_h)
            ),
            "expected_grid_registry_version": self.expected_grid_registry_version,
            "expected_grid_profiles": dict(self.expected_grid_profiles),
        }


# --------------------------------------------------------------------------- #
# StrategyRunResult                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StrategyRunResult:
    """Producer-issued typed revenue input (§4.3). The only cash-eligible source."""

    strategy_kind: StrategyKind
    daily_realised_cash_series: tuple[tuple[_dt.date, float], ...]
    cash_basis: CashBasis
    power_mw: float
    duration_hours: float
    round_trip_efficiency: float
    zone: str
    sample_window: SampleWindow
    currency_basis: CurrencyBasis
    forecast_audits: ForecastAudits
    reserve_product: str | None
    reserve_source: str | None
    availability: float | None
    reserve_coverage_audit: ReserveCoverageAudit | None
    coverage_audit: CoverageAudit
    adapter_provenance: AdapterProvenance
    embedded_vom_cost_eur_mwh: float
    source_data_content_hash: str
    calculator_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_kind, StrategyKind):
            raise ProjectCaseValidationError("strategy_kind invalid")
        object.__setattr__(self, "power_mw", _pos_float(self.power_mw, "power_mw"))
        object.__setattr__(self, "duration_hours", _pos_float(self.duration_hours, "duration_hours"))
        object.__setattr__(
            self,
            "round_trip_efficiency",
            self._rte(self.round_trip_efficiency),
        )
        # Zone + derived timezone (red-line #17).
        if self.zone not in set(config.ALL_ZONES.values()):
            raise ProjectCaseValidationError(f"zone {self.zone!r} is not a supported code")
        expected_tz = config.ZONE_TIMEZONES[self.zone]
        if self.sample_window.timezone != expected_tz:
            raise ProjectCaseValidationError(
                f"sample_window.timezone must equal config.ZONE_TIMEZONES[{self.zone}]"
                f" ({expected_tz!r}), got {self.sample_window.timezone!r}"
            )
        # Series: immutable tuple of (date, finite value), unique dates == valid_dates.
        series = tuple((_as_date(d, "series date"), _finite_float(v, "series value"))
                       for d, v in self.daily_realised_cash_series)
        series_dates = [d for d, _ in series]
        if not series:
            raise ProjectCaseValidationError("daily_realised_cash_series must be non-empty")
        if len(set(series_dates)) != len(series_dates):
            raise ProjectCaseValidationError("daily_realised_cash_series dates must be unique")
        if frozenset(series_dates) != frozenset(self.coverage_audit.valid_dates):
            raise ProjectCaseValidationError(
                "daily_realised_cash_series dates must equal coverage_audit.valid_dates"
            )
        object.__setattr__(
            self,
            "daily_realised_cash_series",
            tuple(sorted(series, key=lambda item: item[0])),
        )
        # Coverage universe must equal the sample-window evaluation dates.
        if frozenset(self.coverage_audit.observed_dates) != frozenset(
            self.sample_window.evaluation_dates()
        ):
            raise ProjectCaseValidationError(
                "coverage_audit.observed_dates must equal sample_window.evaluation_dates()"
            )
        # VOM once (red-line #7).
        object.__setattr__(
            self,
            "embedded_vom_cost_eur_mwh",
            _finite_float(self.embedded_vom_cost_eur_mwh, "embedded_vom_cost_eur_mwh"),
        )
        if self.embedded_vom_cost_eur_mwh != float(DISPATCH_VOM_COST_EUR_MWH):
            raise ProjectCaseValidationError(
                "embedded_vom_cost_eur_mwh must equal dispatch.DISPATCH_VOM_COST_EUR_MWH"
            )
        _hex64(self.source_data_content_hash, "source_data_content_hash")
        _text(self.calculator_version, "calculator_version")
        self._validate_reserve_fields()
        self._validate_forecast_matrix()
        self._validate_provenance_consistency()

    @staticmethod
    def _rte(x: Any) -> float:
        v = _finite_float(x, "round_trip_efficiency")
        if not 0.0 < v <= 1.0:
            raise ProjectCaseValidationError("round_trip_efficiency must be in (0, 1]")
        return v

    def _is_reserve_kind(self) -> bool:
        return self.strategy_kind in _RESERVE_KINDS

    def _validate_reserve_fields(self) -> None:
        if self._is_reserve_kind():
            _text(self.reserve_product, "reserve_product")
            _text(self.reserve_source, "reserve_source")
            object.__setattr__(
                self, "availability", _ratio_float(self.availability, "availability")
            )
            if self.reserve_coverage_audit is None:
                raise ProjectCaseValidationError(
                    "reserve_coverage_audit is required for a reserve-bearing kind"
                )
            if self.reserve_coverage_audit.date_set() != frozenset(
                self.coverage_audit.observed_dates
            ):
                raise ProjectCaseValidationError(
                    "reserve_coverage_audit must have exactly one entry per observed date"
                )
            # A reserve day is cash-valid only when fully covered (red-line #21).
            # BOTH valid AND solver-failed days must have passed the reserve gate:
            # an uncovered reserve day is deterministically missing_dates, never
            # solver_failed (§4.3/§5, no-relabel).
            covered = self.reserve_coverage_audit.covered_dates
            for d in self.coverage_audit.valid_dates:
                if d not in covered:
                    raise ProjectCaseValidationError(
                        f"reserve valid date {d} is not fully covered (red-line #21)"
                    )
            for d in self.coverage_audit.solver_failed_dates:
                if d not in covered:
                    raise ProjectCaseValidationError(
                        f"reserve solver-failed date {d} is not fully covered; an "
                        "uncovered reserve day must be missing_dates, not solver_failed (§5)"
                    )
        else:
            for name, val in (
                ("reserve_product", self.reserve_product),
                ("reserve_source", self.reserve_source),
                ("availability", self.availability),
                ("reserve_coverage_audit", self.reserve_coverage_audit),
            ):
                if val is not None:
                    raise ProjectCaseValidationError(
                        f"{name} must be null for a non-reserve strategy kind"
                    )

    def _validate_forecast_matrix(self) -> None:
        fa = self.forecast_audits
        kind = self.strategy_kind
        if kind is StrategyKind.DA_ONLY or kind is StrategyKind.DA_RESERVE_COOPT:
            if fa.da or fa.ida or fa.reserve:
                raise ProjectCaseValidationError(
                    f"{kind.value} must have all forecast audits null"
                )
        elif kind is StrategyKind.DA_ID_FORECAST:
            if fa.da is not None or fa.reserve is not None or fa.ida is None:
                raise ProjectCaseValidationError(
                    "DA_ID_FORECAST requires only the IDA forecast audit"
                )
            if fa.ida.bucket not in DA_ID_BUCKETS:
                raise ProjectCaseValidationError("IDA bucket must be hour_of_day|hour_of_week")
            if fa.ida.deadband is None:
                raise ProjectCaseValidationError("DA_ID_FORECAST IDA audit deadband is required")
        elif kind is StrategyKind.DA_ID_RESERVE_REALISED:
            if fa.da is None or fa.ida is None or fa.reserve is None:
                raise ProjectCaseValidationError(
                    "DA_ID_RESERVE_REALISED requires DA, IDA and reserve forecast audits"
                )
            if fa.da.bucket not in DA_ID_BUCKETS or fa.ida.bucket not in DA_ID_BUCKETS:
                raise ProjectCaseValidationError("DA/IDA buckets must be hour_of_day|hour_of_week")
            if fa.da.bucket != fa.ida.bucket:
                raise ProjectCaseValidationError("DA and IDA legs must share the same bucket")
            if fa.reserve.bucket != BUCKET_BLOCK_OF_DAY_4H:
                raise ProjectCaseValidationError("reserve bucket must be block_of_day_4h")
            if (
                fa.da.deadband is not None
                or fa.ida.deadband is not None
                or fa.reserve.deadband is not None
            ):
                raise ProjectCaseValidationError(
                    "DA_ID_RESERVE_REALISED forecast deadbands must be null"
                )

    def _validate_provenance_consistency(self) -> None:
        prov = self.adapter_provenance
        # Producer eligibility is the pinned 5-tuple (§5, red-line #6/#18): a
        # StrategyRunResult whose provenance disagrees with the canonical
        # (kind, source function, per-day cash field, excluded fields) for its
        # adapter id is unconstructible — a mislabelled column can never reach cash.
        spec = canonical_spec(prov.producer_adapter_id)
        if self.strategy_kind is not spec.strategy_kind:
            raise ProjectCaseValidationError(
                f"strategy_kind {self.strategy_kind.value} is not bound to adapter "
                f"{prov.producer_adapter_id.value} (expected {spec.strategy_kind.value})"
            )
        if prov.source_function != spec.source_function:
            raise ProjectCaseValidationError(
                f"source_function must be {spec.source_function!r} for {prov.producer_adapter_id.value}"
            )
        if prov.per_day_cash_field != spec.per_day_cash_field:
            raise ProjectCaseValidationError(
                f"per_day_cash_field must be {spec.per_day_cash_field!r} for "
                f"{prov.producer_adapter_id.value}"
            )
        if tuple(sorted(prov.excluded_fields)) != tuple(sorted(spec.excluded_fields)):
            raise ProjectCaseValidationError(
                f"excluded_fields must be {spec.excluded_fields!r} for {prov.producer_adapter_id.value}"
            )
        # capture_rate null-matrix (R10-02): non-null only for PC_ADP_DA_ONLY, and
        # equal to cash_basis.capture.rate there.
        if prov.producer_adapter_id is ProducerAdapterId.PC_ADP_DA_ONLY:
            if prov.capture_rate is None:
                raise ProjectCaseValidationError(
                    "PC_ADP_DA_ONLY must record adapter_provenance.capture_rate"
                )
            if float(prov.capture_rate) != float(self.cash_basis.capture.rate):
                raise ProjectCaseValidationError(
                    "adapter_provenance.capture_rate must equal cash_basis.capture.rate"
                )
        else:
            if prov.capture_rate is not None:
                raise ProjectCaseValidationError(
                    "adapter_provenance.capture_rate must be null for non-DA-only adapters"
                )
            cap = self.cash_basis.capture
            if cap.applied or float(cap.rate) != 1.0 or cap.source != "not_applied":
                raise ProjectCaseValidationError(
                    "non-DA-only adapters must emit capture {applied:false, rate:1.0, "
                    "source:'not_applied'}"
                )
        if self.cash_basis.liquidity.applied:
            raise ProjectCaseValidationError("all v1 adapters emit liquidity.applied=false")
        # Reserve scalar members are non-null only for PC_ADP_RESERVE_COOPT.
        is_coopt = prov.producer_adapter_id is ProducerAdapterId.PC_ADP_RESERVE_COOPT
        coopt_members = (
            prov.reserve_price_aggregation,
            prov.reserve_pricing_dates,
            prov.reserve_scalar_price_eur_mw_h,
        )
        if is_coopt:
            if any(m is None for m in coopt_members):
                raise ProjectCaseValidationError(
                    "PC_ADP_RESERVE_COOPT must record reserve price aggregation/dates/scalar"
                )
            if prov.reserve_price_aggregation != RESERVE_PRICE_AGGREGATION_V1:
                raise ProjectCaseValidationError(
                    f"reserve_price_aggregation must be {RESERVE_PRICE_AGGREGATION_V1!r}"
                )
            if not prov.reserve_pricing_dates:
                raise ProjectCaseValidationError("reserve_pricing_dates must be non-empty")
        elif any(m is not None for m in coopt_members):
            raise ProjectCaseValidationError(
                "reserve price aggregation/dates/scalar are non-null only for PC_ADP_RESERVE_COOPT"
            )
        # Provenance mode string (§4.8).
        if prov.producer_adapter_id is ProducerAdapterId.PC_ADP_DA_ONLY:
            if prov.mode != MAINTENANCE_PROVENANCE_MODE:
                raise ProjectCaseValidationError(
                    f"PC_ADP_DA_ONLY provenance mode must be {MAINTENANCE_PROVENANCE_MODE!r}"
                )
        elif prov.mode is not None:
            raise ProjectCaseValidationError("provenance mode must be null for non-DA-only adapters")
        # Daily bootstrap i.i.d. basis (red-line #22).
        if prov.carry_soc is not False:
            raise ProjectCaseValidationError("adapter_provenance.carry_soc must be False")
        if float(prov.soc_init_frac) != 0.5:
            raise ProjectCaseValidationError("adapter_provenance.soc_init_frac must be 0.5")
        # expected_grid_profiles leg presence per kind (§4.8).
        self._validate_grid_profiles(prov.expected_grid_profiles)

    def _validate_grid_profiles(self, profiles: dict[str, str | None]) -> None:
        kind = self.strategy_kind
        needs_ida = kind in (StrategyKind.DA_ID_FORECAST, StrategyKind.DA_ID_RESERVE_REALISED)
        needs_reserve = self._is_reserve_kind()
        required = {"da": True, "ida": needs_ida, "reserve": needs_reserve}
        for leg, is_required in required.items():
            prof = profiles[leg]
            if is_required and not prof:
                raise ProjectCaseValidationError(
                    f"expected_grid_profiles.{leg} is required for {kind.value}"
                )
            if not is_required and prof is not None:
                raise ProjectCaseValidationError(
                    f"expected_grid_profiles.{leg} must be null for {kind.value}"
                )

    def validate(self) -> None:
        """Re-run construction-time invariants (already enforced in __post_init__)."""
        # Construction already fails closed; this is a no-op affirmation hook.
        return None

    def to_payload(self) -> dict[str, Any]:
        return {
            "strategy_kind": self.strategy_kind.value,
            "daily_realised_cash_series": [
                [_iso(d), float(v)] for d, v in self.daily_realised_cash_series
            ],
            "cash_basis": self.cash_basis.to_payload(),
            "power_mw": float(self.power_mw),
            "duration_hours": float(self.duration_hours),
            "round_trip_efficiency": float(self.round_trip_efficiency),
            "zone": self.zone,
            "sample_window": self.sample_window.to_payload(),
            "currency_basis": self.currency_basis.to_payload(),
            "forecast_audits": self.forecast_audits.to_payload(),
            "reserve_product": self.reserve_product,
            "reserve_source": self.reserve_source,
            "availability": (None if self.availability is None else float(self.availability)),
            "reserve_coverage_audit": (
                None
                if self.reserve_coverage_audit is None
                else self.reserve_coverage_audit.to_payload()
            ),
            "coverage_audit": self.coverage_audit.to_payload(),
            "adapter_provenance": self.adapter_provenance.to_payload(),
            "embedded_vom_cost_eur_mwh": float(self.embedded_vom_cost_eur_mwh),
            "source_data_content_hash": self.source_data_content_hash,
            "calculator_version": self.calculator_version,
        }

    def fingerprint(self) -> str:
        return fingerprint_hex("StrategyRunResult", self.to_payload())


# --------------------------------------------------------------------------- #
# AssetCase / LifecycleCase / ValuationCase / BootstrapCase / MarketCase      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AssetCase:
    """Engineering nameplate + cash fixed O&M (§4.1)."""

    power_mw: float
    duration_hours: float
    round_trip_efficiency: float
    installed_capex_eur: float
    fixed_om_eur_per_mw_yr: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "power_mw", _pos_float(self.power_mw, "power_mw"))
        object.__setattr__(self, "duration_hours", _pos_float(self.duration_hours, "duration_hours"))
        v = _finite_float(self.round_trip_efficiency, "round_trip_efficiency")
        if not 0.0 < v <= 1.0:
            raise ProjectCaseValidationError("round_trip_efficiency must be in (0, 1]")
        object.__setattr__(self, "round_trip_efficiency", v)
        object.__setattr__(
            self, "installed_capex_eur", _nonneg_float(self.installed_capex_eur, "installed_capex_eur")
        )
        object.__setattr__(
            self,
            "fixed_om_eur_per_mw_yr",
            _nonneg_float(self.fixed_om_eur_per_mw_yr, "fixed_om_eur_per_mw_yr"),
        )

    @classmethod
    def from_capex_per_kwh(
        cls,
        *,
        power_mw: float,
        duration_hours: float,
        round_trip_efficiency: float,
        capex_eur_per_kwh: float,
        fixed_om_eur_per_mw_yr: float,
    ) -> AssetCase:
        energy_mwh = _pos_float(power_mw, "power_mw") * _pos_float(duration_hours, "duration_hours")
        capex = _nonneg_float(capex_eur_per_kwh, "capex_eur_per_kwh") * energy_mwh * 1000.0
        return cls(
            power_mw=power_mw,
            duration_hours=duration_hours,
            round_trip_efficiency=round_trip_efficiency,
            installed_capex_eur=capex,
            fixed_om_eur_per_mw_yr=fixed_om_eur_per_mw_yr,
        )

    @property
    def energy_mwh(self) -> float:
        return float(self.power_mw) * float(self.duration_hours)

    def to_payload(self) -> dict[str, Any]:
        return {
            "power_mw": float(self.power_mw),
            "duration_hours": float(self.duration_hours),
            "round_trip_efficiency": float(self.round_trip_efficiency),
            "installed_capex_eur": float(self.installed_capex_eur),
            "fixed_om_eur_per_mw_yr": float(self.fixed_om_eur_per_mw_yr),
        }


@dataclass(frozen=True)
class AugmentationEvent:
    """Dated augmentation/replacement event (§4.2)."""

    year: int
    cost_eur: float
    capacity_restored_frac: float
    residual_value_eur: float

    def __post_init__(self) -> None:
        # Upper bound against project life is enforced by LifecycleCase.
        _int_in(self.year, 1, MAX_PROJECT_LIFE_YEARS, "augmentation event year")
        object.__setattr__(self, "cost_eur", _nonneg_float(self.cost_eur, "event.cost_eur"))
        object.__setattr__(
            self,
            "capacity_restored_frac",
            _ratio_float(self.capacity_restored_frac, "event.capacity_restored_frac"),
        )
        object.__setattr__(
            self,
            "residual_value_eur",
            _nonneg_float(self.residual_value_eur, "event.residual_value_eur"),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "year": int(self.year),
            "cost_eur": float(self.cost_eur),
            "capacity_restored_frac": float(self.capacity_restored_frac),
            "residual_value_eur": float(self.residual_value_eur),
        }


@dataclass(frozen=True)
class LifecycleCase:
    """Life, dated augmentation schedule, residual/disposal (§4.2)."""

    project_life_years: int
    capacity_maintenance_basis: CapacityMaintenanceBasis
    capacity_maintenance_source: str | None
    capacity_maintenance_as_of: str | None
    augmentation_events: tuple[AugmentationEvent, ...]
    eol_residual_value_eur: float
    decommissioning_cost_eur: float

    def __post_init__(self) -> None:
        _int_in(self.project_life_years, 1, MAX_PROJECT_LIFE_YEARS, "project_life_years")
        if not isinstance(self.capacity_maintenance_basis, CapacityMaintenanceBasis):
            raise ProjectCaseValidationError("capacity_maintenance_basis invalid")
        events = tuple(self.augmentation_events)
        for ev in events:
            if not isinstance(ev, AugmentationEvent):
                raise ProjectCaseValidationError("augmentation_events must be AugmentationEvent")
            if ev.year > self.project_life_years:
                raise ProjectCaseValidationError(
                    "augmentation event year exceeds project_life_years"
                )
        events = tuple(
            sorted(events, key=lambda e: (e.year, encode_value(e.to_payload())))
        )
        object.__setattr__(self, "augmentation_events", events)
        object.__setattr__(
            self,
            "eol_residual_value_eur",
            _nonneg_float(self.eol_residual_value_eur, "eol_residual_value_eur"),
        )
        object.__setattr__(
            self,
            "decommissioning_cost_eur",
            _nonneg_float(self.decommissioning_cost_eur, "decommissioning_cost_eur"),
        )
        self._validate_maintenance_basis()

    def _validate_maintenance_basis(self) -> None:
        basis = self.capacity_maintenance_basis
        events = self.augmentation_events
        if basis is CapacityMaintenanceBasis.UNKNOWN:
            if self.capacity_maintenance_source is not None or self.capacity_maintenance_as_of is not None:
                raise ProjectCaseValidationError(
                    "UNKNOWN maintenance basis must have null source and as-of"
                )
            return
        # The two asserted bases require an engineering source + as-of.
        _text(self.capacity_maintenance_source, "capacity_maintenance_source")
        if self.capacity_maintenance_as_of is None or not _ISO_DATE_RE.match(
            str(self.capacity_maintenance_as_of)
        ):
            raise ProjectCaseValidationError(
                "capacity_maintenance_as_of must be an ISO YYYY-MM-DD date"
            )
        if basis is CapacityMaintenanceBasis.SCHEDULED_NAMEPLATE_MAINTENANCE:
            if not events or not any(e.capacity_restored_frac > 0.0 for e in events):
                raise ProjectCaseValidationError(
                    "SCHEDULED_NAMEPLATE_MAINTENANCE requires a non-empty schedule with "
                    "at least one positive-restoration event"
                )
        elif basis is CapacityMaintenanceBasis.NO_AUGMENTATION_REQUIRED_ASSERTED and events:
            raise ProjectCaseValidationError(
                "NO_AUGMENTATION_REQUIRED_ASSERTED requires an empty schedule"
            )

    @property
    def lifecycle_available(self) -> bool:
        return self.capacity_maintenance_basis is not CapacityMaintenanceBasis.UNKNOWN

    def to_payload(self) -> dict[str, Any]:
        return {
            "project_life_years": int(self.project_life_years),
            "capacity_maintenance_basis": self.capacity_maintenance_basis.value,
            "capacity_maintenance_source": self.capacity_maintenance_source,
            "capacity_maintenance_as_of": self.capacity_maintenance_as_of,
            "augmentation_events": [e.to_payload() for e in self.augmentation_events],
            "eol_residual_value_eur": float(self.eol_residual_value_eur),
            "decommissioning_cost_eur": float(self.decommissioning_cost_eur),
        }


@dataclass(frozen=True)
class Projection:
    """Projection-mode tagged union (§4.7). Year-1 multiplier is always 1.0."""

    projection_kind: ProjectionKind
    annual_decay_rate: float | None = None
    decay_floor_share: float | None = None
    multipliers: tuple[float, ...] | None = None
    source: str | None = None
    as_of: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.projection_kind, ProjectionKind):
            raise ProjectCaseValidationError("projection_kind invalid")
        kind = self.projection_kind
        if kind is ProjectionKind.FlatRealProjection:
            if any(
                v is not None
                for v in (
                    self.annual_decay_rate,
                    self.decay_floor_share,
                    self.multipliers,
                    self.source,
                    self.as_of,
                )
            ):
                raise ProjectCaseValidationError("FlatRealProjection members must all be null")
        elif kind is ProjectionKind.DAOnlySpreadDecay:
            d = _finite_float(self.annual_decay_rate, "annual_decay_rate")
            if not 0.0 <= d < 1.0:
                raise ProjectCaseValidationError("annual_decay_rate must be in [0, 1)")
            f = _ratio_float(self.decay_floor_share, "decay_floor_share")
            object.__setattr__(self, "annual_decay_rate", d)
            object.__setattr__(self, "decay_floor_share", f)
            if any(v is not None for v in (self.multipliers, self.source, self.as_of)):
                raise ProjectCaseValidationError(
                    "DAOnlySpreadDecay must have null multipliers/source/as_of"
                )
        elif kind is ProjectionKind.ExplicitAnnualMultiplierCurve:
            if self.multipliers is None or len(self.multipliers) == 0:
                raise ProjectCaseValidationError("ExplicitAnnualMultiplierCurve needs multipliers")
            mult = tuple(_nonneg_float(m, "multiplier") for m in self.multipliers)
            if mult[0] != 1.0:
                raise ProjectCaseValidationError("ExplicitAnnualMultiplierCurve year 1 must be 1.0")
            object.__setattr__(self, "multipliers", mult)
            _text(self.source, "projection.source")
            if self.as_of is None or not _ISO_DATE_RE.match(str(self.as_of)):
                raise ProjectCaseValidationError("projection.as_of must be an ISO date")
            if self.annual_decay_rate is not None or self.decay_floor_share is not None:
                raise ProjectCaseValidationError(
                    "ExplicitAnnualMultiplierCurve must have null decay members"
                )

    @property
    def is_flat(self) -> bool:
        return self.projection_kind is ProjectionKind.FlatRealProjection

    def to_payload(self) -> dict[str, Any]:
        return {
            "projection_kind": self.projection_kind.value,
            "annual_decay_rate": (
                None if self.annual_decay_rate is None else float(self.annual_decay_rate)
            ),
            "decay_floor_share": (
                None if self.decay_floor_share is None else float(self.decay_floor_share)
            ),
            "multipliers": (
                None if self.multipliers is None else [float(m) for m in self.multipliers]
            ),
            "source": self.source,
            "as_of": self.as_of,
        }


@dataclass(frozen=True)
class MarketCase:
    """Reference to one StrategyRunResult + a projection mode (§4.3)."""

    strategy_run_result: StrategyRunResult
    projection: Projection

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_run_result, StrategyRunResult):
            raise ProjectCaseValidationError("market_case needs a StrategyRunResult")
        if not isinstance(self.projection, Projection):
            raise ProjectCaseValidationError("market_case needs a Projection")

    @property
    def strategy_kind(self) -> StrategyKind:
        return self.strategy_run_result.strategy_kind

    def to_payload(self) -> dict[str, Any]:
        return {
            "strategy_run_fingerprint": self.strategy_run_result.fingerprint(),
            "projection": self.projection.to_payload(),
        }


@dataclass(frozen=True)
class ValuationCase:
    """Discounting: real rate, base-year EUR, unlevered (§4.4)."""

    discount_rate: float
    base_year: int

    def __post_init__(self) -> None:
        r = _finite_float(self.discount_rate, "discount_rate")
        if r <= -1.0:
            raise ProjectCaseValidationError("discount_rate must be > -1")
        object.__setattr__(self, "discount_rate", r)
        _int_in(self.base_year, MIN_BASE_YEAR, MAX_BASE_YEAR, "base_year")

    def to_payload(self) -> dict[str, Any]:
        return {"discount_rate": float(self.discount_rate), "base_year": int(self.base_year)}


@dataclass(frozen=True)
class BootstrapCase:
    """Monte-Carlo reproducibility owner (§4.8)."""

    seed: int
    n_simulations: int
    bootstrap_algorithm_version: str

    def __post_init__(self) -> None:
        _int_in(self.seed, 0, MAX_SEED, "seed")
        _int_in(self.n_simulations, MIN_SIMULATIONS, MAX_SIMULATIONS, "n_simulations")
        if self.bootstrap_algorithm_version != BOOTSTRAP_ALGORITHM_V1:
            raise ProjectCaseValidationError(
                f"bootstrap_algorithm_version must be {BOOTSTRAP_ALGORITHM_V1!r}"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "n_simulations": int(self.n_simulations),
            "bootstrap_algorithm_version": self.bootstrap_algorithm_version,
        }


@dataclass(frozen=True)
class ProjectCase:
    """The v1 aggregator (§4.6). ``validate()`` enforces cross-object invariants."""

    asset_case: AssetCase
    lifecycle_case: LifecycleCase
    market_case: MarketCase
    valuation_case: ValuationCase
    bootstrap_case: BootstrapCase

    def __post_init__(self) -> None:
        for name, cls in (
            ("asset_case", AssetCase),
            ("lifecycle_case", LifecycleCase),
            ("market_case", MarketCase),
            ("valuation_case", ValuationCase),
            ("bootstrap_case", BootstrapCase),
        ):
            if not isinstance(getattr(self, name), cls):
                raise ProjectCaseValidationError(f"{name} must be a {cls.__name__}")
        self.validate()

    def validate(self) -> None:
        """Fail-closed cross-object invariants (§4.6). Raises on any violation."""
        srr = self.market_case.strategy_run_result
        srr.validate()
        asset = self.asset_case
        # Engineering match (§4.3).
        if float(srr.power_mw) != float(asset.power_mw):
            raise ProjectCaseValidationError("StrategyRunResult.power_mw != AssetCase.power_mw")
        if float(srr.duration_hours) != float(asset.duration_hours):
            raise ProjectCaseValidationError("duration_hours mismatch between asset and strategy")
        if float(srr.round_trip_efficiency) != float(asset.round_trip_efficiency):
            raise ProjectCaseValidationError("round_trip_efficiency mismatch")
        # Currency basis matches valuation (red-line #19).
        if int(srr.currency_basis.target_base_year) != int(self.valuation_case.base_year):
            raise ProjectCaseValidationError(
                "currency_basis.target_base_year must equal ValuationCase.base_year"
            )
        # Projection-mode gate (red-line #9).
        proj = self.market_case.projection
        if not proj.is_flat and srr.strategy_kind is not StrategyKind.DA_ONLY:
            raise ProjectCaseValidationError(
                "non-flat projection is permitted only for DA_ONLY (red-line #9)"
            )
        if proj.projection_kind is ProjectionKind.ExplicitAnnualMultiplierCurve and len(
            proj.multipliers
        ) != int(self.lifecycle_case.project_life_years):
            raise ProjectCaseValidationError(
                "ExplicitAnnualMultiplierCurve must cover exactly project_life_years entries"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "asset_case": self.asset_case.to_payload(),
            "lifecycle_case": self.lifecycle_case.to_payload(),
            "market_case": self.market_case.to_payload(),
            "valuation_case": self.valuation_case.to_payload(),
            "bootstrap_case": self.bootstrap_case.to_payload(),
        }

    def input_fingerprint(self) -> str:
        return fingerprint_hex("ProjectCase", self.to_payload())


# --------------------------------------------------------------------------- #
# RunResult envelope (§3, §4.6) — typed scaffold; PC-B populates the NPVs      #
# --------------------------------------------------------------------------- #
# PC-A ships the immutable output-envelope TYPES so downstream (PC-B compute,
# PC-C UI/export) has one schema to fill; PC-A itself computes NO NPV. Partial
# availability is typed, never inferred from null arithmetic (§4.6).
@dataclass(frozen=True)
class NpvDistribution:
    """``{p10, p50, p90, prob_positive}`` (§3). All finite; prob_positive ∈ [0,1]."""

    p10: float
    p50: float
    p90: float
    prob_positive: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "p10", _finite_float(self.p10, "p10"))
        object.__setattr__(self, "p50", _finite_float(self.p50, "p50"))
        object.__setattr__(self, "p90", _finite_float(self.p90, "p90"))
        object.__setattr__(self, "prob_positive", _ratio_float(self.prob_positive, "prob_positive"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "p10": float(self.p10),
            "p50": float(self.p50),
            "p90": float(self.p90),
            "prob_positive": float(self.prob_positive),
        }


@dataclass(frozen=True)
class NpvOutcome:
    """Typed NPV outcome envelope ``{available, status, message, distribution}`` (§3)."""

    available: bool
    status: str
    message: str | None
    distribution: NpvDistribution | None

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise ProjectCaseValidationError("NpvOutcome.available must be bool")
        _text(self.status, "NpvOutcome.status")
        if self.available:
            if self.status != "ok" or self.message is not None or self.distribution is None:
                raise ProjectCaseValidationError(
                    "an available NpvOutcome is {available:true, status:'ok', message:null, "
                    "distribution:<NpvDistribution>}"
                )
            if not isinstance(self.distribution, NpvDistribution):
                raise ProjectCaseValidationError("NpvOutcome.distribution must be an NpvDistribution")
        else:
            if self.distribution is not None:
                raise ProjectCaseValidationError("unavailable NpvOutcome must have null distribution")
            _text(self.message, "NpvOutcome.message")

    @classmethod
    def ok(cls, distribution: NpvDistribution) -> NpvOutcome:
        return cls(True, "ok", None, distribution)

    @classmethod
    def unavailable(cls, status: str, message: str) -> NpvOutcome:
        return cls(False, status, message, None)

    def to_payload(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "status": self.status,
            "message": self.message,
            "distribution": None if self.distribution is None else self.distribution.to_payload(),
        }


@dataclass(frozen=True)
class RunResult:
    """Immutable, input-fingerprinted result envelope (§4.6).

    PC-A ships this typed container; PC-B computes/populates the two ``NpvOutcome``
    slots and the cashflow tables. ``input_fingerprint`` is the ProjectCase digest
    the result carries; the RunResult itself is not separately fingerprinted. The
    floor comparator is deliberately NOT a field here (red-line #23, §4.5).
    """

    input_fingerprint: str
    no_lifecycle_cost_screening_npv: NpvOutcome
    lifecycle_cash_npv: NpvOutcome
    provenance: MappingProxyType
    schema_version: str = SCHEMA_VERSION
    screening_cashflow_table: Any = None
    lifecycle_cashflow_table: Any = None

    def __post_init__(self) -> None:
        _hex64(self.input_fingerprint, "input_fingerprint")
        if self.schema_version != SCHEMA_VERSION:
            raise ProjectCaseValidationError(f"schema_version must be {SCHEMA_VERSION!r}")
        for name in ("no_lifecycle_cost_screening_npv", "lifecycle_cash_npv"):
            if not isinstance(getattr(self, name), NpvOutcome):
                raise ProjectCaseValidationError(f"{name} must be an NpvOutcome")
        if not isinstance(self.provenance, MappingProxyType):
            object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
