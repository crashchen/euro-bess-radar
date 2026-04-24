"""Shared timestamp helpers for market-specific settlement conventions."""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime, time as time_type
import re
from typing import Iterable

import pandas as pd


def _local_midnight(value, timezone: str) -> pd.Timestamp:
    """Interpret a date-like value as local midnight in the requested timezone."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.normalize().tz_localize(timezone)
    return ts.tz_convert(timezone).normalize()


def gb_settlement_period_to_utc(
    settlement_dates: Iterable,
    settlement_periods: Iterable,
) -> pd.DatetimeIndex:
    """Convert GB settlement date/period pairs to UTC interval-start timestamps.

    Period 1 is local midnight in Europe/London. Subsequent periods advance in
    30-minute settlement intervals, so BST and fall-back days are represented by
    absolute elapsed settlement time rather than naive UTC wall-clock labels.
    """
    dates = pd.Series(settlement_dates).reset_index(drop=True)
    periods = pd.to_numeric(pd.Series(settlement_periods).reset_index(drop=True), errors="raise")
    if len(dates) != len(periods):
        raise ValueError("settlement_dates and settlement_periods must have equal length")

    timestamps = []
    for raw_date, raw_period in zip(dates, periods):
        period = int(raw_period)
        if period < 1:
            raise ValueError("GB settlement periods are 1-indexed")
        local_start = _local_midnight(raw_date, "Europe/London")
        timestamps.append(local_start + pd.Timedelta(minutes=(period - 1) * 30))

    return pd.DatetimeIndex(timestamps, name="timestamp").tz_convert("UTC")


def _time_from_numeric(value: int | float) -> tuple[int, int] | None:
    """Interpret Excel time fractions or serial date/time values."""
    numeric = float(value)
    if pd.isna(numeric):
        return None
    if 0 <= numeric < 1:
        minutes = int(round(numeric * 24 * 60))
    elif numeric >= 1:
        ts = pd.Timestamp("1899-12-30") + pd.Timedelta(days=numeric)
        minutes = ts.hour * 60 + ts.minute
    else:
        return None
    return divmod(minutes, 60)


def _time_from_text(value: str) -> tuple[int, int] | None:
    """Extract the start time from common Regelleistung block labels."""
    text = value.strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None

    # A block like "00:00-04:00" or "00:00 – 04:00" starts at the first time.
    text = text.replace("Uhr", "").replace("uhr", "").strip()
    text = re.split(r"\s*[-–—]\s*", text, maxsplit=1)[0].strip()

    match = re.search(r"(?P<hour>\d{1,2})[:.](?P<minute>\d{2})", text)
    if match:
        return int(match.group("hour")), int(match.group("minute"))

    if re.fullmatch(r"\d{1,2}", text):
        return int(text), 0

    try:
        parsed = pd.Timestamp(text)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return int(parsed.hour), int(parsed.minute)


def parse_regelleistung_time_block_start(
    target_date: str | date_type | pd.Timestamp,
    time_block,
    *,
    timezone: str = "Europe/Berlin",
) -> pd.Timestamp:
    """Parse a Regelleistung delivery block start into a UTC timestamp."""
    local_start = _local_midnight(target_date, timezone)

    if time_block is None or pd.isna(time_block):
        return local_start.tz_convert("UTC")

    parsed_time: tuple[int, int] | None
    if isinstance(time_block, pd.Timestamp):
        parsed_time = (int(time_block.hour), int(time_block.minute))
    elif isinstance(time_block, datetime):
        parsed_time = (time_block.hour, time_block.minute)
    elif isinstance(time_block, time_type):
        parsed_time = (time_block.hour, time_block.minute)
    elif isinstance(time_block, (int, float)):
        parsed_time = _time_from_numeric(time_block)
    else:
        parsed_time = _time_from_text(str(time_block))

    if parsed_time is None:
        raise ValueError(f"Could not parse Regelleistung time block: {time_block!r}")

    hour, minute = parsed_time
    if hour == 24 and minute == 0:
        return (local_start + pd.Timedelta(days=1)).tz_convert("UTC")
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid Regelleistung time block: {time_block!r}")

    return (local_start + pd.Timedelta(hours=hour, minutes=minute)).tz_convert("UTC")
