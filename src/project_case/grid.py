"""Leg-specific versioned market-grid registry ``pc-market-grid-v1`` (contract §4.3).

Completeness is checked against an **expected grid**, never inferred from surviving
rows (red-line #17, review 36/55). Each consumed leg (DA, IDA, reserve) has its own
explicit ``(leg, zone, delivery_date)`` calendar; DA and IDA may not inherit one
another's resolution, and reserve uses an explicit 4-hour product-block calendar. A
requested ``(leg, zone, date)`` with no registry entry makes the adapter unavailable
(no cadence inference, no IE/CH/GB fallback).

Any calendar-content change requires a **new** ``expected_grid_registry_version``
literal (and calculator version), never an in-place mutation (§4.8).
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

from src import config
from src.project_case.enums import EXPECTED_GRID_REGISTRY_VERSION

REGISTRY_VERSION = EXPECTED_GRID_REGISTRY_VERSION

# Per-leg profile identifiers (stamped into adapter_provenance.expected_grid_profiles).
DA_PROFILE_SDAC = "pc-da-sdac-60to15-v1"
DA_PROFILE_IE_SEM = "pc-da-ie-sem-30min-v1"
DA_PROFILE_CH = "pc-da-ch-60min-v1"
IDA_PROFILE = "pc-ida-sidc-15min-v1"
RESERVE_PROFILE = "pc-reserve-block-of-day-4h-v1"

# v1 leg support (registry policy; every code is validated against config below).
_DA_ZONES = frozenset(config.ENTSOE_ZONES.values())  # EUR ENTSO-E zones (excludes GB)
_IDA_ZONES = frozenset({"DE_LU", "NL", "BE", "FR", "AT", "IT_NORD"})
_RESERVE_ZONES = frozenset({"DE_LU", "FI"})

# 4-hour product blocks per local delivery day: canonical settlement duration 4.0h
# each ("normally 4h, explicit rather than inferred across gaps", §4.3).
_RESERVE_BLOCK_START_HOURS = (0, 4, 8, 12, 16, 20)
_RESERVE_BLOCK_DURATION_H = 4.0

# Fail fast if the local support lists drift from the supported-zone registry.
_all_codes = frozenset(config.ALL_ZONES.values())
assert _all_codes >= _IDA_ZONES, "IDA registry references an unsupported zone code"
assert _all_codes >= _RESERVE_ZONES, "Reserve registry references an unsupported zone code"


def _zone_tz(zone: str) -> str:
    """Registry-owned IANA name (never ``get_zone_timezone`` fallback, §4.3)."""
    return config.ZONE_TIMEZONES[zone]


def _local_day_bounds_utc(zone: str, delivery_date: _dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC ``[start, end)`` of one local delivery day (DST-correct)."""
    tz = _zone_tz(zone)
    start = pd.Timestamp(delivery_date).tz_localize(tz).tz_convert("UTC")
    end = (pd.Timestamp(delivery_date) + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC")
    return start, end


def _grid(start: pd.Timestamp, end: pd.Timestamp, minutes: int) -> tuple[pd.Timestamp, ...]:
    step = pd.Timedelta(minutes=minutes)
    out: list[pd.Timestamp] = []
    cur = start
    while cur < end:
        out.append(cur)
        cur = cur + step
    return tuple(out)


# SDAC switched the DA market time unit 60->15min at ONE shared market instant —
# the CET/CEST market midnight of delivery day 2025-10-01 = 2025-09-30T22:00:00Z —
# NOT at each zone's civil midnight (price-resolution-transition-v1.md). A local
# delivery day that straddles that instant is genuinely mixed-resolution: e.g.
# FI 2025-10-01 (local start 21:00Z) is 1 hourly point + 92 quarter-hours = 93, and
# PT 2025-09-30 (local end 23:00Z) is 23 hourly + 4 quarter-hours = 27. Resolving
# the cutover through each civil date would mis-exclude these real days.
_SDAC_CUTOVER_UTC = (
    pd.Timestamp(config.SDAC_15MIN_DELIVERY_DATE)
    .tz_localize(config.SDAC_MARKET_TIMEZONE)
    .tz_convert("UTC")
)


def _segmented_da_grid(
    zone: str, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.Timestamp, ...]:
    """Expected DA timestamps for one UTC ``[start, end)`` window, cutover-aware."""
    if zone == "IE_SEM":
        return _grid(start, end, 30)  # 30-min throughout (outside SDAC rollout)
    if zone == "CH":
        return _grid(start, end, 60)  # 60-min throughout (outside SDAC rollout)
    cut = _SDAC_CUTOVER_UTC
    if end <= cut:
        return _grid(start, end, 60)
    if start >= cut:
        return _grid(start, end, 15)
    return _grid(start, cut, 60) + _grid(cut, end, 15)


def da_profile_id(zone: str) -> str | None:
    if zone not in _DA_ZONES:
        return None
    if zone == "IE_SEM":
        return DA_PROFILE_IE_SEM
    if zone == "CH":
        return DA_PROFILE_CH
    return DA_PROFILE_SDAC


def ida_profile_id(zone: str) -> str | None:
    return IDA_PROFILE if zone in _IDA_ZONES else None


def reserve_profile_id(zone: str) -> str | None:
    return RESERVE_PROFILE if zone in _RESERVE_ZONES else None


def profile_id(leg: str, zone: str) -> str | None:
    if leg == "da":
        return da_profile_id(zone)
    if leg == "ida":
        return ida_profile_id(zone)
    if leg == "reserve":
        return reserve_profile_id(zone)
    raise ValueError(f"unknown leg {leg!r}")


def expected_da_timestamps(zone: str, delivery_date: _dt.date) -> tuple[pd.Timestamp, ...] | None:
    if zone not in _DA_ZONES:
        return None
    start, end = _local_day_bounds_utc(zone, delivery_date)
    return _segmented_da_grid(zone, start, end)


def expected_ida_timestamps(zone: str, delivery_date: _dt.date) -> tuple[pd.Timestamp, ...] | None:
    if zone not in _IDA_ZONES:
        return None
    start, end = _local_day_bounds_utc(zone, delivery_date)
    return _grid(start, end, 15)


def expected_timestamps(leg: str, zone: str, delivery_date: _dt.date) -> tuple[pd.Timestamp, ...] | None:
    if leg == "da":
        return expected_da_timestamps(zone, delivery_date)
    if leg == "ida":
        return expected_ida_timestamps(zone, delivery_date)
    raise ValueError(f"leg {leg!r} has no per-timestamp grid; use reserve_blocks for reserve")


def reserve_blocks(
    zone: str, delivery_date: _dt.date
) -> tuple[tuple[str, float], ...] | None:
    """Return ``((block_id, settlement_duration_hours), ...)`` for the day, or None.

    A block id is its UTC interval start as ``YYYY-MM-DDTHH:MM:SSZ`` (§4.8).
    """
    if zone not in _RESERVE_ZONES:
        return None
    tz = _zone_tz(zone)
    base = pd.Timestamp(delivery_date)
    out: list[tuple[str, float]] = []
    for hour in _RESERVE_BLOCK_START_HOURS:
        start_utc = (base + pd.Timedelta(hours=hour)).tz_localize(tz).tz_convert("UTC")
        block_id = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append((block_id, _RESERVE_BLOCK_DURATION_H))
    return tuple(out)
