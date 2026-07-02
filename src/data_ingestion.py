"""Unified data ingestion for eu-bess-pulse: ENTSO-E + Elexon APIs."""

from __future__ import annotations

import base64
import functools
import io
import itertools
import json
import logging
import re
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
import requests
from entsoe import EntsoePandasClient

from src.config import (
    ALL_ZONES,
    CACHE_DIR,
    DB_PATH,
    DEFAULT_QUERY_TIMEZONE,
    ELEXON_BASE_URL,
    ELEXON_MARKET_INDEX_ENDPOINT,
    ESIOS_BASE_URL,
    FINGRID_BASE_URL,
    GBP_EUR_YEARLY,
    MAX_SHORT_GAP_HOURS,
    NETZTRANSPARENZ_BASE_URL,
    PRICE_CACHE_TTL_HOURS,
    REGELLEISTUNG_API_URL,
    get_api_key,
    get_esios_api_key,
    get_fingrid_api_key,
    get_zone_timezone,
    is_elexon_zone,
)
from src.time_utils import (
    gb_settlement_period_to_utc,
    parse_regelleistung_time_block_start,
)

logger = logging.getLogger(__name__)

_warned_fx_years: set[int] = set()
ELEXON_MAX_DAYS_PER_REQUEST = 7
_ZONE_RESOLUTION_TRANSITIONS: dict[str, tuple[tuple[pd.Timestamp, pd.Timedelta, pd.Timedelta], ...]] = {
    "DE_LU": (
        (
            pd.Timestamp("2025-10-01T00:00:00Z"),
            pd.Timedelta(hours=1),
            pd.Timedelta(minutes=15),
        ),
    ),
}


# ── Exceptions ───────────────────────────────────────────────────────────────

class DataSourceError(RuntimeError):
    """Base error for user-relevant data source failures."""


class DataSourceAuthError(DataSourceError):
    """Raised when authentication or local configuration is invalid."""


class DataSourceNetworkError(DataSourceError):
    """Raised when a remote data source request fails."""


class DataSourceParseError(DataSourceError):
    """Raised when a remote data source response cannot be parsed safely."""


def _raise_if_auth_failed(
    resp: requests.Response, source: str, hint: str | None = None,
) -> None:
    """Convert HTTP 401/403 into DataSourceAuthError so callers and retry
    decorators do not treat unfixable auth failures as transient network
    errors.
    """
    if resp.status_code in (401, 403):
        suffix = f" {hint}" if hint else ""
        raise DataSourceAuthError(
            f"{source} auth failed (HTTP {resp.status_code}).{suffix}"
        )


# ── Secret scrubbing ─────────────────────────────────────────────────────────

# entsoe-py and other upstream clients embed credentials directly in request
# URLs (e.g. ``...?securityToken=abcdef...&periodStart=...``). When a network
# error fires and the exception stringifies the URL, those credentials end
# up in logs and Streamlit error panels. Strip them at the boundary.
_SECRET_QS_KEYS = (
    "securityToken", "security_token", "api_key", "apiKey", "x-api-key",
    "token", "key",
)
_SECRET_QS_RE = re.compile(
    r"(?i)\b(" + "|".join(re.escape(k) for k in _SECRET_QS_KEYS) + r")=[^&\s\"']+",
)


def _scrub_secrets(text: str) -> str:
    """Redact credential-bearing query-string params in arbitrary text.

    Preserves the surrounding URL so the operator can still identify the
    failing endpoint.
    """
    return _SECRET_QS_RE.sub(r"\1=***", text)


# ── Retry decorator ──────────────────────────────────────────────────────────

def retry(
    max_retries: int = 3,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    auth_check: Callable[[Exception], bool] | None = None,
) -> Callable[..., Any]:
    """Decorator: retry with exponential backoff on exception.

    Args:
        max_retries, backoff, exceptions: standard retry knobs.
        auth_check: optional predicate. When set, any caught exception that
            matches it is re-raised immediately without retry or sleep. Used
            to ensure invalid-token errors propagate to the UI as auth
            failures instead of stalling the user with 3 backoff cycles.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    # Auth failures never recover by retrying. Re-raise now
                    # so the caller can surface a ``set API key`` prompt
                    # instead of a stale ``request failed`` message after
                    # several seconds of backoff.
                    if auth_check is not None and auth_check(exc):
                        raise
                    last_exc = exc
                    if attempt < max_retries:
                        wait = backoff ** attempt
                        logger.warning(
                            "%s failed (attempt %d/%d), retrying in %.1fs: %s",
                            func.__name__, attempt + 1, max_retries, wait,
                            _scrub_secrets(str(exc)),
                        )
                        time.sleep(wait)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_zone(zone: str) -> None:
    """Raise ValueError if zone code is not recognised."""
    valid = set(ALL_ZONES.values())
    if zone not in valid:
        raise ValueError(
            f"Unknown bidding zone '{zone}'. Supported zones: {sorted(valid)}"
        )


def _to_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Normalise a timestamp to UTC."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(DEFAULT_QUERY_TIMEZONE)
    return ts.tz_convert("UTC")


def _to_local_midnight(
    value: str | date | pd.Timestamp,
    timezone: str,
) -> pd.Timestamp:
    """Interpret a calendar date as local midnight in the given timezone."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.normalize().tz_localize(timezone)
    return ts.tz_convert(timezone).normalize()


def build_zone_query_window(
    zone: str,
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert inclusive local calendar dates into a UTC [start, end) window."""
    timezone = get_zone_timezone(zone)
    start_local = _to_local_midnight(start_date, timezone)
    end_local = _to_local_midnight(pd.Timestamp(end_date) + pd.DateOffset(days=1), timezone)

    if end_local <= start_local:
        raise ValueError("end_date must be on or after start_date.")

    return start_local.tz_convert("UTC"), end_local.tz_convert("UTC")


def _expected_cache_interval(
    zone: str,
    index: pd.DatetimeIndex,
) -> pd.Timedelta:
    """Infer the finest credible cache cadence for the zone."""
    fallback = pd.Timedelta(minutes=30) if is_elexon_zone(zone) else pd.Timedelta(hours=1)
    if len(index) < 2:
        return fallback

    deltas = pd.Series(index).sort_values().diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        return fallback

    observed = positive.mode().iloc[0]
    return min(observed, fallback)


def _get_gbp_eur_rate_for_year(year: int) -> float:
    """Return the configured GBP/EUR rate for a year, falling back to nearest."""
    if year in GBP_EUR_YEARLY:
        return GBP_EUR_YEARLY[year]

    nearest_year = min(GBP_EUR_YEARLY, key=lambda known: (abs(known - year), known))
    if year not in _warned_fx_years:
        logger.warning(
            "No GBP/EUR rate configured for %s; using nearest known year %s",
            year, nearest_year,
        )
        _warned_fx_years.add(year)
    return GBP_EUR_YEARLY[nearest_year]


def _looks_like_auth_error(exc: Exception) -> bool:
    """Best-effort detection of authentication-related upstream failures."""
    text = str(exc).lower()
    auth_terms = [
        "401", "403", "unauthor", "forbidden", "security token",
        "api key", "access denied", "invalid token",
    ]
    return any(term in text for term in auth_terms)


def _cache_updated_at(
    conn: sqlite3.Connection,
    zone: str,
) -> pd.Timestamp | None:
    """Return the last write timestamp for a zone cache, or None if unknown."""
    try:
        row = conn.execute(
            "SELECT updated_at FROM cache_metadata WHERE zone = ?",
            (zone,),
        ).fetchone()
    except sqlite3.DatabaseError:
        return None

    if row is None or row[0] is None:
        return None

    ts = pd.Timestamp(row[0])
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _gb_settlement_period_to_utc(
    settlement_dates,
    settlement_periods,
) -> pd.DatetimeIndex:
    """Internal wrapper for GB settlement-date/period UTC conversion."""
    return gb_settlement_period_to_utc(settlement_dates, settlement_periods)


def _parse_regelleistung_time_block_start(
    target_date: str | date | pd.Timestamp,
    time_block,
) -> pd.Timestamp:
    """Internal wrapper for German reserve block start parsing."""
    return parse_regelleistung_time_block_start(target_date, time_block)


def _convert_gbp_series_to_eur(
    gbp_values: pd.Series,
    timestamps,
) -> pd.Series:
    """Convert GBP prices to EUR using year-specific FX rates."""
    ts_index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    rates_by_year = {
        int(year): _get_gbp_eur_rate_for_year(int(year))
        for year in sorted(set(ts_index.year))
    }
    rates = pd.Series(ts_index.year, index=gbp_values.index).map(rates_by_year)
    return gbp_values * rates


# ── Cleaning ──────────────────────────────────────────────────────────────────

def _format_distinct_deltas(positive: pd.Series) -> str:
    """Render distinct positive deltas for logging."""
    return ", ".join(
        sorted(
            {
                str(delta)
                for delta in positive.drop_duplicates().tolist()
            }
        )
    )


def _infer_segment_freq(
    index: pd.DatetimeIndex,
    zone: str | None = None,
) -> pd.Timedelta | None:
    """Infer a usable frequency for one contiguous resolution segment."""
    if len(index) < 2:
        return None

    deltas = pd.Series(index).diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        return None

    mode = positive.mode()
    if not mode.empty:
        return mode.iloc[0]

    inferred = pd.infer_freq(index)
    if inferred is not None:
        offset = pd.tseries.frequencies.to_offset(inferred)
        return pd.Timedelta(offset.nanos, unit="ns")

    logger.warning(
        "Skipping reindex for %s: could not infer frequency from mixed deltas [%s]",
        zone or "unknown zone",
        _format_distinct_deltas(positive),
    )
    return None


def _segment_and_reindex_prices(
    df: pd.DataFrame,
    zone: str | None = None,
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reindex one or more contiguous resolution segments without crashing."""
    if df.empty:
        df.index.name = "timestamp"
        return df

    index = pd.DatetimeIndex(df.index)
    if expected_start is not None and expected_end is not None:
        start = _to_utc_timestamp(expected_start)
        end = _to_utc_timestamp(expected_end)
        if end <= start:
            df.index.name = "timestamp"
            return df

        inferred = _infer_segment_freq(index, zone=zone)
        fallback = _expected_cache_interval(zone or "", index)

        if zone and _ZONE_RESOLUTION_TRANSITIONS.get(zone):
            reindexed_segments: list[pd.DataFrame] = []
            split_points = [start, *_zone_resolution_boundaries(zone, start, end), end]
            for segment_start, segment_end in itertools.pairwise(split_points):
                interval = _expected_interval_for_segment(
                    zone,
                    segment_start,
                    inferred or fallback,
                )
                expected_points = max(round((segment_end - segment_start) / interval), 1)
                expected_idx = pd.date_range(
                    start=segment_start,
                    periods=expected_points,
                    freq=interval,
                )
                segment = df[(df.index >= segment_start) & (df.index < segment_end)]
                reindexed_segments.append(segment.reindex(expected_idx))
            out = pd.concat(reindexed_segments).sort_index()
            out.index.name = "timestamp"
            return out

        interval = min(inferred, fallback) if inferred is not None else fallback
        expected_points = max(round((end - start) / interval), 1)
        full_idx = pd.date_range(start=start, periods=expected_points, freq=interval)
        out = df.reindex(full_idx)
        out.index.name = "timestamp"
        return out

    if len(df) < 2:
        df.index.name = "timestamp"
        return df

    deltas = pd.Series(index).diff()
    positive = deltas.dropna()[deltas.dropna() > pd.Timedelta(0)]

    if positive.empty:
        df.index.name = "timestamp"
        return df

    distinct_deltas = positive.drop_duplicates().tolist()
    if len(distinct_deltas) == 1:
        freq = _infer_segment_freq(index, zone=zone)
        if freq is None:
            df.index.name = "timestamp"
            return df
        full_idx = pd.date_range(start=index.min(), end=index.max(), freq=freq)
        out = df.reindex(full_idx)
        out.index.name = "timestamp"
        return out

    # Mixed deltas usually mean "regular cadence with a gap" rather than
    # "actually varying resolution". If the modal delta is also the smallest
    # one and dominates (>= 50% of deltas), treat it as the true cadence and
    # reindex the whole range — that lets sparse gaps surface as NaN rows
    # downstream instead of being silently swallowed by segment splitting.
    # When the index crosses a known zone resolution boundary, run the same
    # mode-delta heuristic on each side independently so a sparse gap inside
    # the post-transition segment still surfaces, without upsampling the
    # pre-transition rows to the post-transition cadence.
    transition_pts = (
        _zone_resolution_boundaries(zone, index.min(), index.max())
        if zone else []
    )
    if transition_pts:
        boundaries = [index.min(), *transition_pts, index.max()]
        sub_outs: list[pd.DataFrame] = []
        last_idx = len(boundaries) - 2
        for i, (seg_start, seg_end) in enumerate(itertools.pairwise(boundaries)):
            is_last = i == last_idx
            inclusive = "both" if is_last else "left"
            if is_last:
                sub_df = df[(df.index >= seg_start) & (df.index <= seg_end)]
            else:
                sub_df = df[(df.index >= seg_start) & (df.index < seg_end)]
            if sub_df.empty:
                continue
            # Use zone-aware expected interval per side so a 60-min pre-DE_LU
            # segment isn't reindexed at 15-min, and so a row missing exactly
            # at the boundary (or right before it) is still surfaced. The
            # boundary itself anchors each side instead of the observed min/max.
            sub_idx = pd.DatetimeIndex(sub_df.index)
            inferred = _infer_segment_freq(sub_idx, zone=zone)
            fallback = _expected_cache_interval(zone or "", sub_idx)
            interval = _expected_interval_for_segment(
                zone, seg_start, inferred or fallback,
            )
            full_idx = pd.date_range(
                start=seg_start, end=seg_end,
                freq=interval, inclusive=inclusive,
            )
            sub_outs.append(sub_df.reindex(full_idx))
        if sub_outs:
            out = pd.concat(sub_outs).sort_index()
            out = out[~out.index.duplicated(keep="last")]
            out.index.name = "timestamp"
            return out

    mode_delta = positive.mode().iloc[0]
    if (
        mode_delta == min(distinct_deltas)
        and (positive == mode_delta).mean() >= 0.5
    ):
        full_idx = pd.date_range(start=index.min(), end=index.max(), freq=mode_delta)
        out = df.reindex(full_idx)
        out.index.name = "timestamp"
        return out

    segments: list[pd.DataFrame] = []
    run_start = 1
    delta_values = deltas.tolist()

    for pos in range(2, len(index)):
        if delta_values[pos] != delta_values[pos - 1]:
            segments.append(df.iloc[run_start - 1:pos])
            run_start = pos
    segments.append(df.iloc[run_start - 1:])

    reindexed_segments: list[pd.DataFrame] = []
    for segment in segments:
        freq = _infer_segment_freq(pd.DatetimeIndex(segment.index), zone=zone)
        if freq is None:
            segment = segment.copy()
            segment.index.name = "timestamp"
            reindexed_segments.append(segment)
            continue

        full_idx = pd.date_range(
            start=segment.index.min(),
            end=segment.index.max(),
            freq=freq,
        )
        reindexed = segment.reindex(full_idx)
        reindexed.index.name = "timestamp"
        reindexed_segments.append(reindexed)

    out = pd.concat(reindexed_segments).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "timestamp"
    return out


def _dominant_interval_hours(index: pd.DatetimeIndex) -> float:
    """Return the dominant interval length in hours for data-quality metrics."""
    freq = _infer_segment_freq(index)
    if freq is None:
        return 1.0
    return max(freq.total_seconds() / 3600.0, 1.0 / 60.0)


def _max_consecutive_true(mask: pd.Series) -> int:
    """Return the longest consecutive True run in a boolean Series."""
    max_run = 0
    current = 0
    for value in mask.astype(bool).to_numpy():
        if value:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _short_internal_gap_mask(
    missing: pd.Series,
    max_gap_intervals: int,
) -> pd.Series:
    """Mark internal missing runs short enough for safe interpolation."""
    values = missing.astype(bool).to_numpy()
    impute = np.zeros(len(values), dtype=bool)
    pos = 0
    while pos < len(values):
        if not values[pos]:
            pos += 1
            continue
        start = pos
        while pos < len(values) and values[pos]:
            pos += 1
        end = pos
        is_internal = start > 0 and end < len(values)
        if is_internal and end - start <= max_gap_intervals:
            impute[start:end] = True
    return pd.Series(impute, index=missing.index)


def summarize_price_data_quality(df: pd.DataFrame) -> dict[str, float | int]:
    """Summarise source-backed, imputed, and unresolved price intervals."""
    total = len(df)
    if total == 0 or "price_eur_mwh" not in df.columns:
        return {
            "total_intervals": total,
            "source_gap_intervals": 0,
            "imputed_intervals": 0,
            "missing_intervals": 0,
            "valid_intervals": 0,
            "source_gap_ratio": 0.0,
            "imputed_ratio": 0.0,
            "missing_ratio": 0.0,
            "max_source_gap_hours": 0.0,
        }

    missing = df["price_eur_mwh"].isna()
    filled = (
        df["filled"].astype(bool)
        if "filled" in df.columns
        else missing
    )
    imputed = (
        df["imputed"].astype(bool)
        if "imputed" in df.columns
        else pd.Series(False, index=df.index)
    )
    interval_hours = _dominant_interval_hours(pd.DatetimeIndex(df.index))
    source_gap_intervals = int(filled.sum())
    imputed_intervals = int(imputed.sum())
    missing_intervals = int(missing.sum())
    valid_intervals = int(df["price_eur_mwh"].notna().sum())
    max_gap_hours = _max_consecutive_true(filled) * interval_hours

    return {
        "total_intervals": total,
        "source_gap_intervals": source_gap_intervals,
        "imputed_intervals": imputed_intervals,
        "missing_intervals": missing_intervals,
        "valid_intervals": valid_intervals,
        "source_gap_ratio": source_gap_intervals / total if total else 0.0,
        "imputed_ratio": imputed_intervals / total if total else 0.0,
        "missing_ratio": missing_intervals / total if total else 0.0,
        "max_source_gap_hours": round(max_gap_hours, 2),
    }


def _zone_resolution_boundaries(
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Return known resolution transitions that fall inside the requested window."""
    transitions = _ZONE_RESOLUTION_TRANSITIONS.get(zone, ())
    return [
        boundary
        for boundary, _before, _after in transitions
        if start < boundary < end
    ]


def _index_spans_known_resolution_boundary(zone: str, index: pd.DatetimeIndex) -> bool:
    """Return True when the cached index crosses a known zone resolution change."""
    if len(index) < 2:
        return False

    start = index.min()
    end = index.max()
    return any(
        start < boundary <= end
        for boundary, _before, _after in _ZONE_RESOLUTION_TRANSITIONS.get(zone, ())
    )


def _expected_interval_for_segment(
    zone: str,
    segment_start: pd.Timestamp,
    fallback: pd.Timedelta,
) -> pd.Timedelta:
    """Return the expected cadence for a segment start, honoring known transitions."""
    interval = fallback
    for boundary, before, after in _ZONE_RESOLUTION_TRANSITIONS.get(zone, ()):
        if segment_start < boundary:
            return before
        interval = after
    return interval


def _cache_segment_has_gaps(
    df: pd.DataFrame,
    zone: str,
    segment_start: pd.Timestamp,
    segment_end: pd.Timestamp,
    fallback_interval: pd.Timedelta,
) -> bool:
    """Check whether one cache segment is complete at its own cadence."""
    segment = df[(df.index >= segment_start) & (df.index < segment_end)]
    inferred = _infer_segment_freq(pd.DatetimeIndex(segment.index), zone=zone)
    expected_interval = _expected_interval_for_segment(zone, segment_start, fallback_interval)
    interval = min(inferred, expected_interval) if inferred is not None else expected_interval
    expected_points = max(round((segment_end - segment_start) / interval), 1)
    expected_index = pd.date_range(start=segment_start, periods=expected_points, freq=interval)
    missing_points = int(segment.reindex(expected_index)["price_eur_mwh"].isna().sum())

    if missing_points:
        logger.info(
            "Cache incomplete for %s: %d missing points in [%s, %s) at %s cadence",
            zone,
            missing_points,
            segment_start,
            segment_end,
            interval,
        )
        return True

    return False


def clean_prices(
    df: pd.DataFrame,
    zone: str | None = None,
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Clean raw price data: handle gaps, outliers, timezone normalisation.

    Args:
        df: DataFrame with DatetimeIndex named 'timestamp' and column 'price_eur_mwh'.
        zone: Optional zone code for more actionable logging.
        expected_start: Optional requested UTC start boundary for gap detection.
        expected_end: Optional requested UTC exclusive end boundary for gap detection.

    Returns:
        Cleaned DataFrame with short internal gaps interpolated, long or edge
        gaps left as NaN, plus flags marking rows that were missing in source
        data (`filled`) and actually imputed (`imputed`).
    """
    if df.empty:
        return df

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = _segment_and_reindex_prices(
        df,
        zone=zone,
        expected_start=expected_start,
        expected_end=expected_end,
    )
    missing = df["price_eur_mwh"].isna()
    interval_hours = _dominant_interval_hours(pd.DatetimeIndex(df.index))
    max_gap_intervals = max(round(MAX_SHORT_GAP_HOURS / interval_hours), 1)
    impute_mask = _short_internal_gap_mask(missing, max_gap_intervals)

    interpolated = df["price_eur_mwh"].interpolate(method="time", limit_area="inside")
    df["filled"] = missing
    df["imputed"] = impute_mask
    df.loc[impute_mask, "price_eur_mwh"] = interpolated.loc[impute_mask]
    df["filled"] = df["filled"].astype(bool)
    df["imputed"] = df["imputed"].astype(bool)
    df.attrs["data_quality"] = summarize_price_data_quality(df)
    return df


# ── ENTSO-E fetcher ──────────────────────────────────────────────────────────

@retry(
    max_retries=3, backoff=2.0,
    auth_check=lambda exc: _looks_like_auth_error(exc),
)
def _call_entsoe_api(
    client: EntsoePandasClient, zone: str,
    start: pd.Timestamp, end: pd.Timestamp,
) -> pd.Series:
    """Call entsoe-py with retry logic. Auth failures bypass retry."""
    return client.query_day_ahead_prices(zone, start=start, end=end)


def fetch_entsoe_prices(
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: EntsoePandasClient | None = None,
) -> pd.DataFrame:
    """Fetch day-ahead prices from ENTSO-E for a given zone.

    Returns:
        DataFrame with columns: [timestamp (index), price_eur_mwh].
        Timestamps are UTC, timezone-aware.
    """
    if client is None:
        try:
            client = EntsoePandasClient(api_key=get_api_key())
        except OSError as exc:
            raise DataSourceAuthError(
                "ENTSO-E API key is missing or invalid. Check your .env configuration."
            ) from exc

    start_q = start.tz_convert(DEFAULT_QUERY_TIMEZONE)
    end_q = end.tz_convert(DEFAULT_QUERY_TIMEZONE)

    logger.info("Fetching ENTSO-E data for %s [%s, %s)", zone, start_q, end_q)
    try:
        raw = _call_entsoe_api(client, zone, start_q, end_q)
    except requests.RequestException as exc:
        # entsoe-py exception strings often embed the full request URL
        # including ``securityToken=...``; scrub before logging.
        logger.error(
            "ENTSO-E request failed for %s: %s", zone, _scrub_secrets(str(exc)),
        )
        raise DataSourceNetworkError(
            f"ENTSO-E request failed for {zone}. Please retry."
        ) from None
    except Exception as exc:
        logger.error(
            "ENTSO-E fetch failed for %s: %s", zone, _scrub_secrets(str(exc)),
        )
        if _looks_like_auth_error(exc):
            raise DataSourceAuthError(
                "ENTSO-E API authentication failed. Check your token and permissions."
            ) from None
        raise DataSourceParseError(
            f"ENTSO-E returned data for {zone} in an unexpected format."
        ) from None

    if isinstance(raw, pd.DataFrame):
        series = raw.iloc[:, 0]
    else:
        series = raw

    series = pd.to_numeric(series, errors="coerce")

    if series.empty:
        logger.warning("ENTSO-E returned empty data for zone %s", zone)
        return pd.DataFrame(columns=["price_eur_mwh"])

    df = pd.DataFrame({"price_eur_mwh": series})
    df.index.name = "timestamp"
    df.index = df.index.tz_convert("UTC")
    return df


# ── Elexon fetcher ────────────────────────────────────────────────────────────

def _elexon_date_bounds(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[str, str]:
    """Return date bounds that cover the requested [start, end) window."""
    range_start = start.normalize()
    range_end = end.normalize() + pd.Timedelta(days=1)
    return range_start.strftime("%Y-%m-%d"), range_end.strftime("%Y-%m-%d")


def _extract_elexon_records(payload: Any, label: str) -> list[dict[str, Any]]:
    """Normalize Elexon payloads into a list of records."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise DataSourceParseError(
        f"Elexon payload for {label} did not contain a list of records."
    )


@retry(max_retries=3, backoff=2.0, exceptions=(requests.RequestException,))
def _call_elexon_api(from_date_str: str, to_date_str: str) -> list[dict[str, Any]]:
    """Fetch Elexon market index data for a date range."""
    resp = requests.get(
        ELEXON_MARKET_INDEX_ENDPOINT,
        params={"from": from_date_str, "to": to_date_str, "format": "json"},
        timeout=30,
    )
    _raise_if_auth_failed(resp, "Elexon market-index")
    resp.raise_for_status()
    return resp.json()


@retry(max_retries=3, backoff=2.0, exceptions=(requests.RequestException,))
def _call_elexon_system_prices_api(
    from_date_str: str,
    to_date_str: str,
) -> list[dict[str, Any]]:
    """Fetch Elexon system prices for a date range."""
    resp = requests.get(
        ELEXON_SYSTEM_PRICES_ENDPOINT,
        params={"from": from_date_str, "to": to_date_str, "format": "json"},
        timeout=30,
    )
    _raise_if_auth_failed(resp, "Elexon system-prices")
    resp.raise_for_status()
    return resp.json()


def _drop_elexon_zero_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate zero placeholder rows without removing legitimate £0 prices."""
    if df.empty or "timestamp" not in df.columns or "price_gbp" not in df.columns:
        return df

    duplicate_period = df.groupby("timestamp")["price_gbp"].transform("size") > 1
    has_nonzero_price = df.groupby("timestamp")["price_gbp"].transform(
        lambda values: values.ne(0).any()
    )
    zero_candidate = df["price_gbp"].eq(0) & duplicate_period & has_nonzero_price

    volume_col = next((c for c in df.columns if "volume" in c.lower()), None)
    if volume_col is not None:
        zero_candidate &= pd.to_numeric(df[volume_col], errors="coerce").fillna(0).eq(0)
    else:
        provider_col = next((c for c in df.columns if "provider" in c.lower()), None)
        if provider_col is None:
            # A real MID trade can clear at £0/MWh. Without a provider/volume
            # hint, keep it and average duplicates rather than silently deleting it.
            return df
        provider = df[provider_col].fillna("").astype(str).str.lower()
        zero_candidate &= provider.str.contains("placeholder|zero|fallback|dummy")

    dropped = int(zero_candidate.sum())
    if dropped:
        logger.info("Dropped %d duplicate Elexon zero placeholder rows", dropped)
    return df.loc[~zero_candidate].copy()


def _elexon_date_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[str, str]]:
    """Split a [start, end) window into small Elexon date-bound pairs.

    Elexon Market Index rejects longer windows with 400 responses, so keep
    chunks short even for one-month dashboard requests.
    """
    from_date, _ = _elexon_date_bounds(start, end)
    _, to_date = _elexon_date_bounds(start, end)
    cursor = pd.Timestamp(from_date)
    final = pd.Timestamp(to_date)
    chunks: list[tuple[str, str]] = []
    while cursor < final:
        chunk_end = min(
            cursor + pd.Timedelta(days=ELEXON_MAX_DAYS_PER_REQUEST),
            final,
        )
        chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end
    return chunks


def fetch_elexon_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch Market Index Data from Elexon for GB.

    Args:
        start: Start timestamp (UTC, timezone-aware).
        end: End timestamp (UTC, timezone-aware).

    Returns:
        DataFrame with columns: [timestamp (index), price_eur_mwh].
        Original GBP/MWh prices converted to EUR using year-specific GBP/EUR rates.
        30-min settlement periods, UTC timestamps.
    """
    all_records: list[dict[str, Any]] = []
    request_errors: list[DataSourceError] = []
    chunks = _elexon_date_chunks(start, end)

    for from_date_str, to_date_str in chunks:
        range_label = f"[{from_date_str}, {to_date_str})"
        logger.info("Fetching Elexon data for %s", range_label)
        try:
            data = _call_elexon_api(from_date_str, to_date_str)
            all_records.extend(_extract_elexon_records(data, range_label))
        except requests.RequestException as exc:
            logger.warning("Failed to fetch Elexon data for %s: %s", range_label, exc)
            request_errors.append(
                DataSourceNetworkError(
                    f"Elexon request failed for {range_label}. Please retry."
                )
            )
        except (TypeError, ValueError, KeyError, DataSourceParseError) as exc:
            logger.warning("Failed to parse Elexon data for %s: %s", range_label, exc)
            request_errors.append(
                DataSourceParseError(
                    f"Elexon response for {range_label} could not be parsed."
                )
            )

    if request_errors:
        raise request_errors[0]

    if not all_records:
        logger.warning("Elexon returned no data for [%s, %s)", start, end)
        return pd.DataFrame(columns=["price_eur_mwh"])

    df = pd.DataFrame(all_records)
    try:
        df["timestamp"] = pd.to_datetime(df["startTime"], utc=True)
        df["price_gbp"] = pd.to_numeric(df["price"], errors="coerce")
    except KeyError as exc:
        logger.exception("Elexon response missing expected columns")
        raise DataSourceParseError(
            "Elexon response is missing expected pricing fields."
        ) from exc
    except (TypeError, ValueError) as exc:
        logger.exception("Elexon response could not be parsed")
        raise DataSourceParseError(
            "Elexon response could not be parsed into timestamps and prices."
        ) from exc

    # Some Elexon provider feeds contain duplicate £0 placeholders next to a
    # real price. A legitimate market price can also be exactly £0/MWh, so only
    # remove rows that carry a duplicate/provider hint instead of blanket
    # filtering all zero prices.
    df = _drop_elexon_zero_placeholders(df)
    df = df.groupby("timestamp", as_index=True)["price_gbp"].mean().to_frame()
    df["price_eur_mwh"] = _convert_gbp_series_to_eur(df["price_gbp"], df.index)
    df = df[["price_eur_mwh"]].sort_index()

    if end > start:
        df = df[(df.index >= start) & (df.index < end)]

    return df


# ── Cache (SQLite + CSV) ─────────────────────────────────────────────────────

def _ensure_price_cache_schema(conn: sqlite3.Connection, table_name: str) -> None:
    """Create or upgrade a price-cache table with row-level freshness metadata."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            timestamp TEXT PRIMARY KEY,
            price_eur_mwh REAL NOT NULL,
            zone TEXT NOT NULL,
            fetched_at TEXT
        )
    """)
    columns = {
        row[1]
        for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    }
    if "fetched_at" not in columns:
        conn.execute(f'ALTER TABLE "{table_name}" ADD COLUMN fetched_at TEXT')


def write_cache(df: pd.DataFrame, zone: str) -> None:
    """Write DataFrame to SQLite table and CSV file."""
    if df.empty:
        return

    # Keep analytics on the filled frame, but only persist source-backed rows so
    # synthetic bars never poison future cache reads.
    persist_df = df.loc[~df["filled"]] if "filled" in df.columns else df
    if persist_df.empty:
        logger.warning("Skipping cache write for %s: no source-backed rows to persist", zone)
        return

    table_name = f"da_prices_{zone.lower()}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = CACHE_DIR / f"{table_name}.csv"
    fetched_at = pd.Timestamp.now(tz="UTC").isoformat()
    export = persist_df.reset_index()
    export["timestamp"] = export["timestamp"].map(lambda t: t.isoformat())
    export["zone"] = zone
    export["fetched_at"] = fetched_at
    export = export[["timestamp", "price_eur_mwh", "zone", "fetched_at"]]
    export.to_csv(csv_path, index=False)
    logger.info("Wrote CSV cache: %s", csv_path)

    # SQLite
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_price_cache_schema(conn, table_name)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                zone TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL
            )
        """)
        rows = [
            (row.timestamp.isoformat() if hasattr(row.timestamp, "isoformat") else str(row.timestamp),
             float(row.price_eur_mwh), zone, fetched_at)
            for row in persist_df.reset_index().itertuples(index=False)
        ]
        conn.executemany(
            f'INSERT OR REPLACE INTO "{table_name}" '
            "(timestamp, price_eur_mwh, zone, fetched_at) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.execute(
            'INSERT OR REPLACE INTO cache_metadata (zone, updated_at) VALUES (?, ?)',
            (zone, pd.Timestamp.now(tz="UTC").isoformat()),
        )


def read_cache(
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    """Read from SQLite cache. Return None on miss."""
    if not DB_PATH.exists():
        return None

    table_name = f"da_prices_{zone.lower()}"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = (
                f'SELECT timestamp, price_eur_mwh, fetched_at FROM "{table_name}" '
                "WHERE timestamp >= ? AND timestamp < ?"
            )
            df = pd.read_sql_query(
                query, conn,
                params=(start.isoformat(), end.isoformat()),
                parse_dates=["timestamp"],
            )
    except (sqlite3.DatabaseError, pd.errors.DatabaseError, pd.errors.ParserError, ValueError):
        return None

    if df.empty:
        return None

    if "fetched_at" not in df.columns or df["fetched_at"].isna().any():
        logger.info("Cache for %s uses old schema; treating requested slice as stale", zone)
        return None

    fetched_at = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce")
    if fetched_at.isna().any():
        logger.info("Cache for %s has invalid fetched_at values; treating as stale", zone)
        return None

    ttl = pd.Timedelta(hours=PRICE_CACHE_TTL_HOURS)
    stale_rows = fetched_at < (pd.Timestamp.now(tz="UTC") - ttl)
    if stale_rows.any():
        logger.info(
            "Cache stale for %s: %d rows in requested slice exceed ttl=%s",
            zone,
            int(stale_rows.sum()),
            ttl,
        )
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    df["zone"] = zone
    df = df.drop(columns=["fetched_at"])

    interval = _expected_cache_interval(zone, df.index)
    if _index_spans_known_resolution_boundary(zone, df.index):
        split_points = [start, *_zone_resolution_boundaries(zone, start, end), end]
        for segment_start, segment_end in itertools.pairwise(split_points):
            if _cache_segment_has_gaps(df, zone, segment_start, segment_end, interval):
                return None
    else:
        if _cache_segment_has_gaps(df, zone, start, end, interval):
            return None

    return df


# ── Unified entry point ──────────────────────────────────────────────────────

def fetch_prices(
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Unified entry point: auto-routes to ENTSO-E or Elexon based on zone.

    Args:
        zone: Bidding zone code (e.g. 'DE_LU', 'GB').
        start: Start timestamp (tz-aware or naive).
        end: End timestamp.
        use_cache: Check SQLite cache first if True.

    Returns:
        DataFrame with columns: [timestamp (index), price_eur_mwh, zone].
    """
    _validate_zone(zone)
    start_utc = _to_utc_timestamp(start)
    end_utc = _to_utc_timestamp(end)

    if end_utc <= start_utc:
        raise ValueError("end must be later than start.")

    # Cache check
    if use_cache:
        cached = read_cache(zone, start_utc, end_utc)
        if cached is not None:
            logger.info("Cache hit for %s [%s, %s)", zone, start_utc, end_utc)
            return cached

    # Route to the right API
    if is_elexon_zone(zone):
        df = fetch_elexon_prices(start_utc, end_utc)
    else:
        df = fetch_entsoe_prices(zone, start_utc, end_utc)

    if df.empty:
        logger.warning("No data returned for zone %s", zone)
        df["zone"] = zone
        return df

    # Clean and enrich
    df = clean_prices(df, zone=zone, expected_start=start_utc, expected_end=end_utc)
    df["zone"] = zone

    # Cache write
    write_cache(df, zone)
    return df


# ── Generation data fetchers ──────────────────────────────────────────────────

ELEXON_FUELINST_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELINST"

_RENEWABLE_COLS = [
    "solar_mw", "wind_onshore_mw", "wind_offshore_mw",
    "total_renewable_mw", "total_generation_mw", "renewable_pct",
]
_NON_GENERATION_COLUMN_TERMS = ("consumption", "load", "demand")


@retry(
    max_retries=3, backoff=2.0,
    auth_check=lambda exc: _looks_like_auth_error(exc),
)
def _call_entsoe_generation(
    client: EntsoePandasClient, zone: str,
    start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    """Call entsoe-py generation query with retry. Auth bypasses retry."""
    return client.query_generation(zone, start=start, end=end)


def fetch_generation_data(
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch actual generation by fuel type from ENTSO-E.

    Args:
        zone: Bidding zone code.
        start: Start timestamp (UTC).
        end: End timestamp (UTC).

    Returns:
        DataFrame with columns: [solar_mw, wind_onshore_mw, wind_offshore_mw,
        total_renewable_mw, total_generation_mw, renewable_pct].
    """
    if is_elexon_zone(zone):
        return fetch_elexon_generation(start, end)

    try:
        client = EntsoePandasClient(api_key=get_api_key())
    except OSError as exc:
        raise DataSourceAuthError(
            "ENTSO-E API key is missing or invalid. Check your .env configuration."
        ) from exc
    start_q = start.tz_convert(DEFAULT_QUERY_TIMEZONE)
    end_q = end.tz_convert(DEFAULT_QUERY_TIMEZONE)

    logger.info("Fetching generation data for %s", zone)
    try:
        raw = _call_entsoe_generation(client, zone, start_q, end_q)
    except requests.RequestException as exc:
        logger.error(
            "ENTSO-E generation request failed for %s: %s",
            zone, _scrub_secrets(str(exc)),
        )
        raise DataSourceNetworkError(
            f"Generation data request failed for {zone}. Please retry."
        ) from None
    except Exception as exc:
        logger.error(
            "ENTSO-E generation fetch failed for %s: %s",
            zone, _scrub_secrets(str(exc)),
        )
        if _looks_like_auth_error(exc):
            raise DataSourceAuthError(
                "ENTSO-E API authentication failed while fetching generation data."
            ) from None
        raise DataSourceParseError(
            f"Generation data for {zone} could not be parsed or was unavailable."
        ) from None

    if raw.empty:
        return pd.DataFrame(columns=_RENEWABLE_COLS)

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [
            "_".join(str(c) for c in col).strip("_") for col in raw.columns
        ]

    raw = raw.apply(pd.to_numeric, errors="coerce")
    df = pd.DataFrame(index=raw.index)
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    # Map common ENTSO-E generation types
    generation_cols = [
        c for c in raw.columns
        if not any(term in c.lower() for term in _NON_GENERATION_COLUMN_TERMS)
    ]
    solar_cols = [c for c in generation_cols if "solar" in c.lower()]
    wind_on_cols = [
        c for c in generation_cols
        if "wind" in c.lower() and "offshore" not in c.lower()
    ]
    wind_off_cols = [
        c for c in generation_cols
        if "wind" in c.lower() and "offshore" in c.lower()
    ]

    df["solar_mw"] = raw[solar_cols].sum(axis=1) if solar_cols else 0.0
    df["wind_onshore_mw"] = raw[wind_on_cols].sum(axis=1) if wind_on_cols else 0.0
    df["wind_offshore_mw"] = raw[wind_off_cols].sum(axis=1) if wind_off_cols else 0.0
    df["total_renewable_mw"] = df["solar_mw"] + df["wind_onshore_mw"] + df["wind_offshore_mw"]
    df["total_generation_mw"] = raw[generation_cols].sum(axis=1) if generation_cols else 0.0
    df["renewable_pct"] = (
        df["total_renewable_mw"] / df["total_generation_mw"].replace(0, float("nan")) * 100
    ).fillna(0)

    return df


def fetch_elexon_generation(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch GB generation mix from Elexon FUELINST dataset.

    Args:
        start: Start timestamp (UTC).
        end: End timestamp (UTC).

    Returns:
        DataFrame with same schema as fetch_generation_data().
    """
    logger.info("Fetching Elexon FUELINST data...")
    try:
        resp = requests.get(
            ELEXON_FUELINST_URL,
            params={
                # Use business-time windowing so we fetch by settlement period
                # rather than by the later publish timestamp.
                "from": start.isoformat(),
                "to": end.isoformat(),
                "format": "json",
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("Elexon FUELINST request failed: %s", exc)
        return pd.DataFrame(columns=_RENEWABLE_COLS)
    except ValueError as exc:
        logger.warning("Elexon FUELINST unavailable: %s", exc)
        return pd.DataFrame(columns=_RENEWABLE_COLS)

    records = data if isinstance(data, list) else data.get("data", [])
    if not records:
        return pd.DataFrame(columns=_RENEWABLE_COLS)

    raw = pd.DataFrame(records)
    raw["timestamp"] = pd.to_datetime(raw["startTime"], utc=True)
    raw = raw.set_index("timestamp").sort_index()

    # Pivot fuel types to columns
    if "fuelType" in raw.columns and "generation" in raw.columns:
        pivot = raw.pivot_table(
            values="generation", index=raw.index, columns="fuelType", aggfunc="sum",
        )
    else:
        return pd.DataFrame(columns=_RENEWABLE_COLS)

    df = pd.DataFrame(index=pivot.index)
    df.index.name = "timestamp"

    wind_cols = [c for c in pivot.columns if "WIND" in c.upper()]
    solar_cols = [c for c in pivot.columns if "SOLAR" in c.upper()]

    df["solar_mw"] = pivot[solar_cols].sum(axis=1) if solar_cols else 0.0
    df["wind_onshore_mw"] = pivot[wind_cols].sum(axis=1) if wind_cols else 0.0
    df["wind_offshore_mw"] = 0.0  # Elexon FUELINST doesn't split on/offshore
    df["total_renewable_mw"] = df["solar_mw"] + df["wind_onshore_mw"]
    df["total_generation_mw"] = pivot.sum(axis=1)
    df["renewable_pct"] = (
        df["total_renewable_mw"] / df["total_generation_mw"].replace(0, float("nan")) * 100
    ).fillna(0)

    return df


# ── Fingrid Open Data API (Finland FCR/aFRR) ─────────────────────────────────


@retry(max_retries=3, backoff=2.0, exceptions=(requests.RequestException, ValueError))
def _call_fingrid_api(
    dataset_id: int,
    params: dict[str, Any],
    headers: dict[str, str],
) -> Any:
    """Fetch one Fingrid API page with retries.

    Auth failures (401/403) raise DataSourceAuthError and bypass the retry
    loop — retrying a missing/invalid API key cannot succeed.
    """
    resp = requests.get(
        f"{FINGRID_BASE_URL}/datasets/{dataset_id}/data",
        params=params,
        headers=headers,
        timeout=30,
    )
    _raise_if_auth_failed(resp, "Fingrid", "Set FINGRID_API_KEY in .env.")
    resp.raise_for_status()
    return resp.json()


def fetch_fingrid_data(
    dataset_id: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch data from Fingrid Open Data API.

    Args:
        dataset_id: Fingrid dataset ID (e.g. 318 for FCR-N price).
        start: Start timestamp (UTC).
        end: End timestamp (UTC).

    Returns:
        DataFrame with columns: [timestamp, value]. Timestamps in UTC.
    """
    all_rows: list[dict[str, Any]] = []
    page = 1
    page_size = 20000
    api_key = get_fingrid_api_key()
    headers = {"x-api-key": api_key} if api_key else {}
    if not api_key:
        logger.warning(
            "Fingrid v2 endpoint hit without FINGRID_API_KEY; attempting unauthenticated access"
        )

    while True:
        params = {
            "startTime": start.isoformat(),
            "endTime": end.isoformat(),
            "format": "json",
            "pageSize": page_size,
            "page": page,
        }
        logger.info(
            "Fetching Fingrid dataset %d page %d", dataset_id, page,
        )
        try:
            payload = _call_fingrid_api(dataset_id, params, headers)
        except (requests.RequestException, ValueError) as exc:
            # Keep already-collected pages: a transient failure on a later page
            # shouldn't drop earlier successful pages on the floor. Surface as
            # a warning so the user knows the slice is partial.
            logger.warning(
                "Fingrid fetch failed for dataset %d on page %d after retries; "
                "returning %d rows collected from earlier pages (data may be "
                "incomplete): %s",
                dataset_id, page, len(all_rows), exc,
            )
            break

        records = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list) or not records:
            break

        all_rows.extend(records)
        if len(records) < page_size:
            break
        page += 1
        time.sleep(0.5)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "value"])

    df = pd.DataFrame(all_rows)
    time_col = "startTime" if "startTime" in df.columns else "start_time"
    if time_col not in df.columns:
        # Try first column that looks like a timestamp
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                time_col = col
                break
    df["timestamp"] = pd.to_datetime(df[time_col], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["timestamp", "value"]].dropna().sort_values("timestamp")
    df = df.groupby("timestamp", as_index=False)["value"].mean()
    return df.reset_index(drop=True)


def fetch_fingrid_fcr_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch Finland FCR-N and FCR-D prices from Fingrid.

    Args:
        start: Start timestamp (UTC).
        end: End timestamp (UTC).

    Returns:
        DataFrame with columns:
        [timestamp, fcr_n_price, fcr_d_up_price, fcr_d_down_price]
        All prices in EUR/MW/h.
    """
    datasets = {
        "fcr_n_price": 317,
        "fcr_d_up_price": 318,
        "fcr_d_down_price": 283,
    }
    merged: pd.DataFrame | None = None
    for col_name, ds_id in datasets.items():
        df = fetch_fingrid_data(ds_id, start, end)
        if df.empty:
            continue
        series = df.groupby("timestamp")["value"].mean().rename(col_name)
        if merged is None:
            merged = series.to_frame()
        else:
            merged = merged.join(series, how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame(
            columns=["timestamp", "fcr_n_price", "fcr_d_up_price", "fcr_d_down_price"],
        )

    merged = merged.sort_index()
    merged.index.name = "timestamp"
    return merged.reset_index()


def fetch_fingrid_afrr_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch Finland aFRR capacity prices from Fingrid.

    Args:
        start: Start timestamp (UTC).
        end: End timestamp (UTC).

    Returns:
        DataFrame with columns: [timestamp, afrr_up_price, afrr_down_price]
        Prices in EUR/MW/h.
    """
    datasets = {
        "afrr_up_price": 52,
        "afrr_down_price": 51,
    }
    merged: pd.DataFrame | None = None
    for col_name, ds_id in datasets.items():
        df = fetch_fingrid_data(ds_id, start, end)
        if df.empty:
            continue
        series = df.groupby("timestamp")["value"].mean().rename(col_name)
        if merged is None:
            merged = series.to_frame()
        else:
            merged = merged.join(series, how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame(
            columns=["timestamp", "afrr_up_price", "afrr_down_price"],
        )

    merged = merged.sort_index()
    merged.index.name = "timestamp"
    return merged.reset_index()


# ── Regelleistung.net (Germany FCR/aFRR) ─────────────────────────────────────

REGELLEISTUNG_AUTO_FETCH_ENABLED = True


@retry(max_retries=3, backoff=2.0, exceptions=(requests.RequestException,))
def _call_regelleistung_api(
    product: str, market: str, date_str: str,
) -> bytes:
    """Download one day of Regelleistung tender results as xlsx bytes.

    Auth failures (401/403) bypass the retry loop and surface as
    DataSourceAuthError so the UI can prompt the user instead of silently
    looping. The endpoint is keyless today, so this guards against future
    policy changes or rate-limit auth walls.
    """
    resp = requests.get(
        REGELLEISTUNG_API_URL,
        params={
            "productTypes": product,
            "market": market,
            "exportFormat": "xlsx",
            "date": date_str,
        },
        timeout=30,
    )
    _raise_if_auth_failed(
        resp, "Regelleistung",
        "Endpoint may have moved or now requires authentication.",
    )
    resp.raise_for_status()
    return resp.content


def _parse_regelleistung_xlsx(
    content: bytes, product: str, target_date: str,
) -> pd.DataFrame:
    """Parse Regelleistung xlsx export into standard ancillary format."""
    from io import BytesIO

    import openpyxl

    wb = openpyxl.load_workbook(BytesIO(content), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if len(rows) < 2:
        return pd.DataFrame(
            columns=[
                "timestamp", "date", "product", "time_block",
                "capacity_price_eur_mw", "direction",
            ],
        )

    headers = [str(h).strip().lower() if h else "" for h in rows[0]]
    records = []
    for row in rows[1:]:
        row_dict = dict(zip(headers, row, strict=False))
        price = None
        price_is_block_total = False
        price_keys = (
            "germany_settlementcapacity_price_[eur/mw]",
            "germany_average_capacity_price_[(eur/mw)/h]",
            "capacity price [eur/mw]",
            "capacity_price",
            "price",
            "ergebnispreis",
        )
        for key in price_keys:
            if key in row_dict and row_dict[key] is not None:
                try:
                    price = float(row_dict[key])
                except (ValueError, TypeError):
                    continue
                price_is_block_total = key.endswith("settlementcapacity_price_[eur/mw]")
                break
        if price is None:
            continue

        direction = "Symmetric"
        product_label = ""
        for key in ("productname", "product", "product_name"):
            if row_dict.get(key):
                product_label = str(row_dict[key]).strip()
                break
        for key in ("direction", "richtung", "productname", "product", "product_name"):
            if row_dict.get(key):
                val = str(row_dict[key]).strip().upper()
                tokens = {token for token in re.split(r"[^A-Z]+", val) if token}
                if "UP" in tokens or "POS" in tokens:
                    direction = "Up"
                elif "DOWN" in tokens or "NEG" in tokens:
                    direction = "Down"
                break

        time_block = ""
        for key in (
            "time_block", "von", "from", "delivery_date", "productname", "product",
        ):
            if row_dict.get(key):
                time_block = row_dict[key]
                break
        current_block = re.search(
            r"(?:^|_)(\d{2})_(\d{2})(?:$|_)", str(time_block).strip(),
        )
        block_hours = None
        if current_block:
            start_hour = int(current_block.group(1))
            end_hour = int(current_block.group(2))
            block_hours = (end_hour - start_hour) % 24 or 24
            time_block = f"{start_hour:02d}:00-{end_hour:02d}:00"

        try:
            timestamp = _parse_regelleistung_time_block_start(target_date, time_block)
        except ValueError as exc:
            if str(time_block).strip():
                logger.warning(
                    "Skipping Regelleistung row with unparseable time_block=%r "
                    "for %s on %s: %s",
                    time_block,
                    product,
                    target_date,
                    exc,
                )
            continue

        if price_is_block_total:
            if block_hours is None or block_hours <= 0:
                logger.warning(
                    "Skipping Regelleistung %s row with EUR/MW block price but "
                    "unknown duration: %r",
                    product,
                    product_label or time_block,
                )
                continue
            price /= block_hours

        records.append({
            "timestamp": timestamp,
            "date": target_date,
            "product": product,
            "time_block": "" if time_block is None else str(time_block),
            "capacity_price_eur_mw": price,
            "direction": direction,
        })

    return pd.DataFrame(records)


def fetch_regelleistung_results(
    product: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    """Fetch German balancing auction results from regelleistung.net REST API.

    Args:
        product: One of "FCR", "aFRR".
        start: Start timestamp (UTC).
        end: End timestamp (UTC).

    Returns:
        DataFrame with columns:
        [timestamp, date, product, time_block, capacity_price_eur_mw, direction]
        or None if fetching fails.
    """
    market = "CAPACITY"
    start_local = start.tz_convert("Europe/Berlin").normalize()
    end_local = (end - pd.Timedelta(seconds=1)).tz_convert("Europe/Berlin").normalize()
    date_range = pd.date_range(start_local, end_local, freq="D")

    all_frames: list[pd.DataFrame] = []
    for day in date_range:
        date_str = day.strftime("%Y-%m-%d")
        try:
            content = _call_regelleistung_api(product, market, date_str)
            df = _parse_regelleistung_xlsx(content, product, date_str)
            if not df.empty:
                all_frames.append(df)
        except requests.RequestException as exc:
            logger.warning(
                "Regelleistung fetch failed for %s on %s: %s", product, date_str, exc,
            )
        except (ValueError, KeyError, TypeError, DataSourceParseError) as exc:
            logger.warning(
                "Regelleistung parse failed for %s on %s: %s", product, date_str, exc,
            )
        # DataSourceAuthError (and any other unexpected error) propagates so
        # the UI can surface a root-cause message instead of "no data".

    if not all_frames:
        logger.warning(
            "No Regelleistung data retrieved for %s; "
            "upload a manual DE_%s CSV as fallback",
            product, product,
        )
        return None

    result = pd.concat(all_frames, ignore_index=True)
    logger.info(
        "Fetched %d Regelleistung %s records (%s to %s)",
        len(result), product, date_range[0].date(), date_range[-1].date(),
    )
    return result


# ── Netztransparenz.de (Germany reBAP / NRV-Saldo) ──────────────────────────

NETZTRANSPARENZ_CSV_DOWNLOAD_ROUTE = (
    f"{NETZTRANSPARENZ_BASE_URL}/DesktopModules/LotesCharts/"
    "CsvDownloadHandler.ashx"
)
NETZTRANSPARENZ_IMBALANCE_ZONE = "DE_LU"
IMBALANCE_SOURCE_NETZTRANSPARENZ = "Netztransparenz.de"
_NETZTRANSPARENZ_REBAP_TOLERANCE_EUR_MWH = 1e-2
_NETZTRANSPARENZ_DOWNLOAD_BASE_SETTINGS: dict[str, Any] = {
    "DataType": 20,
    "CultureName": "de-DE",
    "DiagramType": "line",
    "TimeInterval": 15,
    "CsvColumns": ["50Hertz", "Amprion", "TenneT TSO", "TransnetBW"],
    "TsoIds": [0],
    "NrvDirection": 0,
    "WebApiBaseUri": (
        "https://lotes-UNB-svc-netzt.corp.transmission-it.de/StatistikApi/"
    ),
}
_NETZTRANSPARENZ_NRV_SETTINGS: dict[str, Any] = {
    **_NETZTRANSPARENZ_DOWNLOAD_BASE_SETTINGS,
    "ProduktId": 6,
    "Title": "NRV-Saldo qualitätsgesichert",
    "DataUnit": "MW",
    "WebApiRoute": "NrvSaldo/nrvsaldo/qualitaetsgesichert",
}
_NETZTRANSPARENZ_REBAP_SETTINGS: dict[str, Any] = {
    **_NETZTRANSPARENZ_DOWNLOAD_BASE_SETTINGS,
    "ProduktId": 10,
    "Title": "reBAP unterdeckt",
    "DataUnit": "EUR/MWh",
    "WebApiRoute": "NrvSaldo/rebap/qualitaetsgesichert",
}


def _netztransparenz_local_date_bounds(
    start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[str, str]:
    """Return Berlin-local date bounds for the CSV handler's [from, to)."""
    start_local = pd.Timestamp(start).tz_convert("Europe/Berlin")
    end_local = pd.Timestamp(end).tz_convert("Europe/Berlin")
    return start_local.strftime("%Y-%m-%d"), end_local.strftime("%Y-%m-%d")


def _netztransparenz_download_request(
    *, start: pd.Timestamp, end: pd.Timestamp, settings: dict[str, Any],
) -> str:
    """Base64 request payload consumed by Netztransparenz CsvDownloadHandler."""
    local_from, local_to = _netztransparenz_local_date_bounds(start, end)
    request = {
        "LocalFrom": local_from,
        "LocalTo": local_to,
        # "cet" means the site's ME(S)Z option: local German time with CET/CEST
        # labels in the returned CSV, which lets us disambiguate DST repeats.
        "ResultTimeZone": "cet",
        "Settings": settings,
    }
    return base64.b64encode(
        json.dumps(request, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).decode("ascii")


@retry(max_retries=3, backoff=2.0, exceptions=(requests.RequestException,))
def _call_netztransparenz_csv(
    *, start: pd.Timestamp, end: pd.Timestamp, settings: dict[str, Any],
) -> str:
    """Download one Netztransparenz CSV via the public chart download handler."""
    encoded_request = _netztransparenz_download_request(
        start=start, end=end, settings=settings,
    )
    resp = requests.get(
        NETZTRANSPARENZ_CSV_DOWNLOAD_ROUTE,
        params={"request": encoded_request},
        timeout=30,
    )
    _raise_if_auth_failed(
        resp,
        "Netztransparenz",
        "The public CSV endpoint may have changed.",
    )
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "").lower()
    text = resp.content.decode("utf-8-sig", errors="replace")
    if "csv" not in content_type and "Datum;" not in text[:200]:
        raise DataSourceParseError(
            "Netztransparenz CSV download returned a non-CSV response.",
        )
    return text


def _read_netztransparenz_csv_text(content: str, *, source_name: str) -> pd.DataFrame:
    """Read a semicolon/comma-decimal Netztransparenz CSV string."""
    try:
        return pd.read_csv(io.StringIO(content), sep=";", decimal=",")
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataSourceParseError(
            f"Could not parse Netztransparenz {source_name} CSV: {exc}"
        ) from exc


def _netztransparenz_timestamp_utc(
    df: pd.DataFrame, *, source_name: str,
) -> pd.Series:
    """Build UTC timestamps from Netztransparenz Datum/von/Zeitzone columns."""
    required = {"Datum", "Zeitzone", "von"}
    missing = required - set(df.columns)
    if missing:
        raise DataSourceParseError(
            f"{source_name} is missing required column(s): {sorted(missing)}"
        )
    zone_labels = df["Zeitzone"].astype(str).str.strip().str.upper()
    unknown = set(zone_labels) - {"CET", "CEST"}
    if unknown:
        raise DataSourceParseError(
            f"{source_name} has unsupported Zeitzone value(s): {sorted(unknown)}",
        )
    local_naive = pd.to_datetime(
        df["Datum"].astype(str).str.strip() + " " + df["von"].astype(str).str.strip(),
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    )
    if local_naive.isna().any():
        raise DataSourceParseError(
            f"{source_name} has {int(local_naive.isna().sum())} "
            "unparseable timestamp row(s)",
        )
    is_dst = (zone_labels == "CEST").to_numpy()
    try:
        return local_naive.dt.tz_localize(
            "Europe/Berlin", ambiguous=is_dst, nonexistent="raise",
        ).dt.tz_convert("UTC")
    except (ValueError, TypeError) as exc:
        raise DataSourceParseError(
            f"{source_name} has invalid Europe/Berlin local timestamp(s): {exc}"
        ) from exc


def _validate_netztransparenz_regular_15min(
    ts: pd.Series, *, source_name: str,
) -> None:
    """Require a unique, regular 15-minute UTC time axis."""
    ordered = pd.Series(pd.DatetimeIndex(ts).sort_values())
    if ordered.duplicated().any():
        raise DataSourceParseError(f"{source_name} contains duplicate timestamps")
    if len(ordered) < 2:
        raise DataSourceParseError(f"{source_name} needs at least two timestamps")
    deltas = ordered.diff().dropna().dt.total_seconds()
    bad = deltas[deltas != 900.0]
    if not bad.empty:
        raise DataSourceParseError(
            f"{source_name} is not a regular 15-minute series; "
            f"unexpected gap seconds: {sorted(set(bad.astype(int)))[:5]}",
        )


def _netztransparenz_rebap_price(rebap: pd.DataFrame) -> pd.Series:
    """Return the validated signed German reBAP cash-flow price."""
    missing = {"reBAP unterdeckt", "reBAP ueberdeckt"} - set(rebap.columns)
    if missing:
        raise DataSourceParseError(
            f"reBAP is missing required column(s): {sorted(missing)}",
        )
    under = pd.to_numeric(rebap["reBAP unterdeckt"], errors="coerce")
    over = pd.to_numeric(rebap["reBAP ueberdeckt"], errors="coerce")
    if under.isna().any() or over.isna().any():
        raise DataSourceParseError("reBAP file contains non-numeric price value(s)")
    diff = float((under - over).abs().max())
    if diff > _NETZTRANSPARENZ_REBAP_TOLERANCE_EUR_MWH + 1e-12:
        raise DataSourceParseError(
            "reBAP unterdeckt and ueberdeckt columns differ; expected the "
            f"symmetric German reBAP export (max diff {diff:g})",
        )
    return under


def _convert_netztransparenz_imbalance_exports(
    *, nrv_csv: str, rebap_csv: str, zone: str = NETZTRANSPARENZ_IMBALANCE_ZONE,
) -> pd.DataFrame:
    """Convert raw Netztransparenz CSV text into the dedicated imbalance frame."""
    nrv = _read_netztransparenz_csv_text(nrv_csv, source_name="NRV-Saldo")
    rebap = _read_netztransparenz_csv_text(rebap_csv, source_name="reBAP")
    if "Deutschland" not in nrv.columns:
        raise DataSourceParseError(
            "NRV-Saldo file is missing the 'Deutschland' MW column",
        )

    nrv_ts = _netztransparenz_timestamp_utc(nrv, source_name="NRV-Saldo")
    rebap_ts = _netztransparenz_timestamp_utc(rebap, source_name="reBAP")
    _validate_netztransparenz_regular_15min(nrv_ts, source_name="NRV-Saldo")
    _validate_netztransparenz_regular_15min(rebap_ts, source_name="reBAP")

    nrv_work = pd.DataFrame({
        "timestamp": nrv_ts,
        "system_imbalance_volume_mw": pd.to_numeric(
            nrv["Deutschland"], errors="coerce",
        ),
    })
    rebap_work = pd.DataFrame({
        "timestamp": rebap_ts,
        "imbalance_price_eur_mwh": _netztransparenz_rebap_price(rebap),
    })
    if nrv_work["system_imbalance_volume_mw"].isna().any():
        raise DataSourceParseError(
            "NRV-Saldo file contains non-numeric Deutschland value(s)",
        )

    merged = nrv_work.merge(rebap_work, on="timestamp", how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise DataSourceParseError(
            "NRV-Saldo and reBAP timestamps do not align exactly: "
            f"{counts}",
        )
    merged = merged.sort_values("timestamp")
    out = pd.DataFrame({
        "zone": zone,
        "imbalance_price_eur_mwh": merged["imbalance_price_eur_mwh"].astype(float),
        "system_imbalance_volume_mw": merged[
            "system_imbalance_volume_mw"
        ].astype(float),
    })
    out.index = pd.DatetimeIndex(merged["timestamp"], name="timestamp")
    return out


def fetch_netztransparenz_imbalance(
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    """Fetch German NRV-Saldo + reBAP from Netztransparenz into imbalance frame.

    This is a keyless public CSV download for the DE_LU balancing area. The
    returned frame is ready for ``persist_imbalance_frame(...,
    source=IMBALANCE_SOURCE_NETZTRANSPARENZ)`` and therefore flows through the
    same provenance/Data Trust/cockpit overlay path as manual uploads.
    """
    if zone != NETZTRANSPARENZ_IMBALANCE_ZONE:
        return None
    if pd.Timestamp(end) <= pd.Timestamp(start):
        raise DataSourceParseError("Netztransparenz fetch end must be after start")
    nrv_csv = _call_netztransparenz_csv(
        start=start, end=end, settings=_NETZTRANSPARENZ_NRV_SETTINGS,
    )
    rebap_csv = _call_netztransparenz_csv(
        start=start, end=end, settings=_NETZTRANSPARENZ_REBAP_SETTINGS,
    )
    return _convert_netztransparenz_imbalance_exports(
        nrv_csv=nrv_csv,
        rebap_csv=rebap_csv,
        zone=zone,
    )


# ── Elexon System Prices (GB Balancing) ───────────────────────────────────────

ELEXON_SYSTEM_PRICES_ENDPOINT = (
    f"{ELEXON_BASE_URL}/balancing/settlement/system-prices"
)


def fetch_elexon_system_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch GB system prices (SBP/SSP) from Elexon.

    Args:
        start: Start timestamp (UTC).
        end: End timestamp (UTC).

    Returns:
        DataFrame with columns:
        [timestamp, system_buy_price_gbp, system_sell_price_gbp,
         system_buy_price_eur, system_sell_price_eur, spread_eur]
    """
    all_records: list[dict[str, Any]] = []
    from_date_str, to_date_str = _elexon_date_bounds(start, end)
    range_label = f"[{from_date_str}, {to_date_str})"

    logger.info("Fetching Elexon system prices for %s", range_label)
    try:
        data = _call_elexon_system_prices_api(from_date_str, to_date_str)
        all_records.extend(_extract_elexon_records(data, range_label))
    except requests.RequestException as exc:
        logger.warning("Elexon system prices failed for %s: %s", range_label, exc)
    except (TypeError, ValueError, KeyError, DataSourceParseError) as exc:
        logger.warning("Elexon system prices failed for %s: %s", range_label, exc)

    if not all_records:
        return pd.DataFrame(columns=[
            "timestamp", "system_buy_price_gbp", "system_sell_price_gbp",
            "system_buy_price_eur", "system_sell_price_eur", "spread_eur",
        ])

    df = pd.DataFrame(all_records)

    # Build timestamp from settlement date + period
    if "settlementDate" in df.columns and "settlementPeriod" in df.columns:
        df["timestamp"] = _gb_settlement_period_to_utc(
            df["settlementDate"],
            df["settlementPeriod"],
        )
    elif "startTime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["startTime"], utc=True)
    else:
        return pd.DataFrame(columns=[
            "timestamp", "system_buy_price_gbp", "system_sell_price_gbp",
            "system_buy_price_eur", "system_sell_price_eur", "spread_eur",
        ])

    sbp_col = next((c for c in df.columns if "systemBuyPrice" in c or "system_buy_price" in c.lower()), None)
    ssp_col = next((c for c in df.columns if "systemSellPrice" in c or "system_sell_price" in c.lower()), None)

    if sbp_col is None or ssp_col is None:
        logger.warning("Elexon system prices: missing SBP/SSP columns in %s", list(df.columns))
        return pd.DataFrame(columns=[
            "timestamp", "system_buy_price_gbp", "system_sell_price_gbp",
            "system_buy_price_eur", "system_sell_price_eur", "spread_eur",
        ])

    out = pd.DataFrame()
    out["timestamp"] = df["timestamp"]
    out["system_buy_price_gbp"] = pd.to_numeric(df[sbp_col], errors="coerce")
    out["system_sell_price_gbp"] = pd.to_numeric(df[ssp_col], errors="coerce")
    out["system_buy_price_eur"] = _convert_gbp_series_to_eur(
        out["system_buy_price_gbp"], out["timestamp"],
    )
    out["system_sell_price_eur"] = _convert_gbp_series_to_eur(
        out["system_sell_price_gbp"], out["timestamp"],
    )
    out["spread_eur"] = out["system_buy_price_eur"] - out["system_sell_price_eur"]
    out = out.set_index("timestamp").sort_index()

    if end > start:
        out = out[(out.index >= start) & (out.index < end)]

    return out


# ── ENTSO-E Imbalance Prices ─────────────────────────────────────────────────

def fetch_entsoe_imbalance_prices(
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    """Fetch imbalance prices from ENTSO-E where available.

    Args:
        zone: Bidding zone code.
        start: Start timestamp (UTC).
        end: End timestamp (UTC).

    Returns:
        DataFrame with columns: [timestamp (index), imbalance_price_long,
        imbalance_price_short] or None if data not available.
    """
    try:
        client = EntsoePandasClient(api_key=get_api_key())
    except OSError as exc:
        # Missing API key was previously logged at WARNING and silently
        # turned into None. DA/IDA both raise here so the sidebar can
        # surface a "set ENTSOE_API_KEY in .env" hint; mirror that
        # behaviour so the user is not left wondering why imbalance
        # data never appears.
        raise DataSourceAuthError(
            "ENTSO-E API key missing or invalid for imbalance prices. "
            "Set ENTSOE_API_KEY in .env."
        ) from exc

    start_q = start.tz_convert(DEFAULT_QUERY_TIMEZONE)
    end_q = end.tz_convert(DEFAULT_QUERY_TIMEZONE)

    logger.info("Fetching ENTSO-E imbalance prices for %s", zone)
    try:
        raw = client.query_imbalance_prices(zone, start=start_q, end=end_q)
    except requests.RequestException as exc:
        logger.warning(
            "ENTSO-E imbalance request failed for %s: %s",
            zone, _scrub_secrets(str(exc)),
        )
        return None
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning(
            "ENTSO-E imbalance data unavailable for %s: %s",
            zone, _scrub_secrets(str(exc)),
        )
        return None
    except Exception as exc:
        logger.warning(
            "ENTSO-E imbalance fetch unavailable for %s: %s",
            zone, _scrub_secrets(str(exc)),
        )
        return None

    if raw is None or (isinstance(raw, (pd.DataFrame, pd.Series)) and raw.empty):
        return None

    if isinstance(raw, pd.Series):
        raw = raw.to_frame("imbalance_price_long")

    if isinstance(raw, pd.DataFrame):
        raw.index = raw.index.tz_convert("UTC")
        raw.index.name = "timestamp"
        # Try to map to standard columns
        out = pd.DataFrame(index=raw.index)
        if raw.shape[1] >= 2:
            out["imbalance_price_long"] = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
            out["imbalance_price_short"] = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
        elif raw.shape[1] == 1:
            out["imbalance_price_long"] = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
            out["imbalance_price_short"] = out["imbalance_price_long"]
        return out

    return None


# ── ENTSO-E Intraday Auction Prices ──────────────────────────────────────────

# IDA = Intraday Auction. ENTSO-E publishes opening (IDA1) and reopening
# (IDA2 / IDA3) auction prices for the zones that participate in SIDC. Other
# zones still trade intraday continuously but those rolling weighted averages
# are not exposed through entsoe-py — we treat them as unavailable here.
INTRADAY_SUPPORTED_ZONES: set[str] = {"DE_LU", "NL", "BE", "FR", "AT", "IT_NORD"}


try:
    from entsoe.exceptions import NoMatchingDataError as _EntsoeNoMatchingDataError
except ImportError:  # pragma: no cover — defensive against older entsoe-py
    _EntsoeNoMatchingDataError = ()


def _ida_cache_table(zone: str, sequence: int) -> str:
    """SQLite table name for one zone x IDA round."""
    return f"ida_prices_{zone.lower()}_seq{sequence}"


IDA_SOURCE_ENTSOE = "ENTSO-E intraday auction"
IDA_SOURCE_MANUAL = "Manual CSV"
IDA_SOURCE_MIXED = "Mixed"
IDA_SOURCE_LEGACY = "Unknown (pre-provenance cache)"
_IDA_SOURCE_TABLE = "ida_price_sources"


def write_intraday_cache(
    df: pd.DataFrame, zone: str, sequence: int,
    *, source: str = IDA_SOURCE_ENTSOE,
) -> None:
    """Persist IDA prices to SQLite. Auction prints are final once published,
    so we INSERT OR REPLACE on the timestamp key without freshness tracking.

    ``source`` (``IDA_SOURCE_ENTSOE`` for the live fetch, ``IDA_SOURCE_MANUAL``
    for an uploaded CSV) is stored PER ROW, so a table that mixes manually
    uploaded and live-fetched intervals is labelled ``Mixed`` in the durable
    ``ida_price_sources`` sidecar rather than inheriting only the last write's
    source. Data Trust reads that sidecar so provenance survives a restart.
    """
    if df.empty:
        return
    table = _ida_cache_table(zone, sequence)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{table}" '
            "(timestamp TEXT PRIMARY KEY, intraday_price_eur_mwh REAL NOT NULL, "
            "source TEXT)"
        )
        _ensure_intraday_source_column(conn, table)
        rows = [
            (ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
             float(price), source)
            for ts, price in df["intraday_price_eur_mwh"].items()
        ]
        conn.executemany(
            f'INSERT OR REPLACE INTO "{table}" '
            "(timestamp, intraday_price_eur_mwh, source) VALUES (?, ?, ?)",
            rows,
        )
        _record_intraday_source(conn, table, zone, sequence)


def _ensure_intraday_source_column(conn: sqlite3.Connection, table: str) -> None:
    """Add the per-row ``source`` column to a pre-provenance IDA price table.

    Older caches (and the synthetic seed table) predate per-row provenance;
    their rows are backfilled with a clearly-labelled legacy sentinel so they
    are never silently attributed to a real source.
    """
    existing = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
    if "source" not in existing:
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN source TEXT')
    conn.execute(
        f'UPDATE "{table}" SET source = ? WHERE source IS NULL',
        (IDA_SOURCE_LEGACY,),
    )


def _record_intraday_source(
    conn: sqlite3.Connection, table: str, zone: str, sequence: int,
) -> None:
    """Upsert a provenance row reflecting the FULL cached extent of one
    (zone, sequence) IDA table after a write. The label is derived from the
    DISTINCT per-row sources: a single source when uniform, else ``Mixed``."""
    n, first, last = conn.execute(
        f'SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM "{table}"'
    ).fetchone()
    distinct = sorted(
        row[0] for row in conn.execute(f'SELECT DISTINCT source FROM "{table}"')
        if row[0] is not None
    )
    if not distinct:
        label = IDA_SOURCE_LEGACY
    elif len(distinct) == 1:
        label = distinct[0]
    else:
        label = IDA_SOURCE_MIXED
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{_IDA_SOURCE_TABLE}" '
        "(zone TEXT, sequence INTEGER, source TEXT, rows INTEGER, "
        "first_timestamp TEXT, last_timestamp TEXT, imported_at TEXT, "
        "PRIMARY KEY (zone, sequence))"
    )
    conn.execute(
        f'INSERT OR REPLACE INTO "{_IDA_SOURCE_TABLE}" '
        "(zone, sequence, source, rows, first_timestamp, last_timestamp, imported_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (zone, int(sequence), label, int(n), first, last,
         datetime.now(UTC).isoformat()),
    )


def read_intraday_sources() -> dict[tuple[str, int], dict[str, Any]]:
    """Return durable IDA provenance keyed by (zone, sequence).

    Each value has ``source``, ``rows``, ``first`` / ``last`` (UTC Timestamps),
    and ``imported_at``. Empty when the sidecar table does not exist yet.
    """
    if not DB_PATH.exists():
        return {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                f'SELECT zone, sequence, source, rows, first_timestamp, '
                f'last_timestamp, imported_at FROM "{_IDA_SOURCE_TABLE}"'
            )
            recorded = cur.fetchall()
    except sqlite3.DatabaseError:
        return {}
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for zone, sequence, source, rows, first, last, imported_at in recorded:
        out[(str(zone), int(sequence))] = {
            "source": source,
            "rows": int(rows),
            "first": pd.to_datetime(first, utc=True, errors="coerce"),
            "last": pd.to_datetime(last, utc=True, errors="coerce"),
            "imported_at": imported_at,
        }
    return out


def read_intraday_cache(
    zone: str, start: pd.Timestamp, end: pd.Timestamp, *, sequence: int = 1,
) -> pd.DataFrame | None:
    """Return cached IDA rows in [start, end) or None when the table is empty.

    Lets the UI render IDA analytics across browser refreshes without
    re-hitting the ENTSO-E API. The caller still owns the "did we cover
    the requested window" decision — the helper just returns what is
    persisted.
    """
    if not DB_PATH.exists():
        return None
    table = _ida_cache_table(zone, sequence)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                f'SELECT timestamp, intraday_price_eur_mwh FROM "{table}" '
                "WHERE timestamp >= ? AND timestamp < ?",
                conn,
                params=(start.isoformat(), end.isoformat()),
            )
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        return None
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    df.index.name = "timestamp"
    return df


INTRADAY_CSV_COLUMNS = ("timestamp", "ida_price_eur_mwh", "sequence", "zone")


def generate_intraday_template_csv() -> str:
    """Return a sample IDA price upload CSV (header + a few rows).

    Columns: ``timestamp`` (UTC ISO 8601 — include a ``+00:00`` offset, or a
    naive value is interpreted as UTC), ``ida_price_eur_mwh``, and the
    optional ``sequence`` (1/2/3, default 1) and ``zone`` (defaults to the
    active bidding zone) columns. This is the manual fallback for zones or
    windows where the ENTSO-E intraday-auction endpoint returns no data.
    """
    return (
        "timestamp,ida_price_eur_mwh,sequence,zone\n"
        "2026-01-01T00:00:00+00:00,72.5,1,DE_LU\n"
        "2026-01-01T01:00:00+00:00,68.0,1,DE_LU\n"
        "2026-01-01T02:00:00+00:00,65.4,1,DE_LU\n"
    )


def _coerce_intraday_sequence(
    values: pd.Series, default_sequence: int,
) -> pd.Series:
    """Map a raw ``sequence`` column to numeric IDA rounds.

    Blank / NA cells fall back to ``default_sequence``. Bare ``1/2/3`` and
    ``IDA1/IDA2/IDA3`` labels (case-insensitive, optional separator) are
    accepted. Any other non-blank value raises ``ValueError`` rather than
    being silently coerced to the default round — a non-numeric ``sequence``
    must never quietly send IDA2/IDA3 data into the IDA1 table.
    """
    text = values.astype("string").str.strip()
    is_blank = text.isna() | (text.str.len() == 0)
    cleaned = text.str.upper().str.replace(r"^IDA[\s_-]*", "", regex=True)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    invalid = (~is_blank) & numeric.isna()
    if invalid.any():
        bad = sorted(set(text[invalid].dropna().tolist()))
        raise ValueError(
            f"IDA 'sequence' values not understood: {bad}. "
            "Use 1/2/3 or IDA1/IDA2/IDA3."
        )
    return numeric.where(~is_blank, other=float(default_sequence))


def parse_intraday_csv(
    content: str,
    *,
    default_zone: str | None = None,
    default_sequence: int = 1,
) -> pd.DataFrame:
    """Parse a manual IDA price CSV into a normalised long frame.

    Required columns: ``timestamp`` and ``ida_price_eur_mwh`` (header match
    is case-insensitive). Optional ``sequence`` (``1/2/3`` or ``IDA1/2/3``)
    and ``zone`` columns fall back to ``default_sequence`` / ``default_zone``
    when absent or blank. Timestamps are coerced to UTC (a naive value is
    assumed UTC). Negative prices are retained; only unparseable
    timestamp/price rows are dropped.

    Args:
        content: Raw CSV text.
        default_zone: Zone for rows without a ``zone`` value. Usually the
            active dashboard zone.
        default_sequence: IDA round for rows without a ``sequence`` value.

    Returns:
        UTC-indexed frame with columns ``[zone, sequence,
        intraday_price_eur_mwh]``, deduplicated on (zone, sequence,
        timestamp) keeping the last row, sorted by timestamp.

    Raises:
        ValueError: missing required columns, an unknown zone, a sequence
            outside {1, 2, 3}, or no parseable rows.
    """
    try:
        raw = pd.read_csv(io.StringIO(content))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise ValueError(f"Could not parse IDA CSV: {exc}") from exc

    cols = {str(c).strip().lower(): c for c in raw.columns}
    if "timestamp" not in cols or "ida_price_eur_mwh" not in cols:
        raise ValueError(
            "IDA CSV must have at least 'timestamp' and 'ida_price_eur_mwh' columns."
        )

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(raw[cols["timestamp"]], utc=True, errors="coerce"),
        "intraday_price_eur_mwh": pd.to_numeric(
            raw[cols["ida_price_eur_mwh"]], errors="coerce",
        ),
    })
    # Carry raw sequence/zone so validation runs only on rows that survive the
    # timestamp/price dropna (garbage in an already-dropped row must not fail
    # the whole upload).
    if "sequence" in cols:
        out["_seq_raw"] = raw[cols["sequence"]]
    if "zone" in cols:
        zone_ser = raw[cols["zone"]].astype("string").str.strip()
        out["zone"] = zone_ser.where(zone_ser.str.len() > 0, other=default_zone)
    else:
        out["zone"] = default_zone

    out = out.dropna(subset=["timestamp", "intraday_price_eur_mwh"])
    if out.empty:
        raise ValueError(
            "No valid IDA rows after parsing — check the timestamp and "
            "ida_price_eur_mwh columns."
        )
    if out["zone"].isna().any():
        raise ValueError(
            "IDA CSV has rows without a zone and no default zone was provided."
        )
    out["zone"] = out["zone"].astype(str)

    if "_seq_raw" in out.columns:
        seq = _coerce_intraday_sequence(out.pop("_seq_raw"), default_sequence)
    else:
        seq = pd.Series(float(default_sequence), index=out.index)
    # Reject fractional sequences (e.g. 1.5) BEFORE the int cast, which would
    # otherwise silently truncate them to a valid round (1.5 -> IDA1).
    if not (seq.dropna() % 1 == 0).all():
        raise ValueError("IDA 'sequence' must be a whole number (1, 2 or 3).")
    out["sequence"] = seq.astype(int)
    bad_seq = sorted(int(s) for s in set(out["sequence"].unique()) - {1, 2, 3})
    if bad_seq:
        raise ValueError(f"IDA 'sequence' must be 1, 2 or 3; got {bad_seq}.")
    for zone in out["zone"].unique():
        _validate_zone(zone)

    out = (
        out.drop_duplicates(subset=["zone", "sequence", "timestamp"], keep="last")
        .set_index("timestamp")
        .sort_index()
    )
    out.index.name = "timestamp"
    return out[["zone", "sequence", "intraday_price_eur_mwh"]]


def persist_intraday_frame(long_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Write a parsed IDA long frame to SQLite, one table per (zone, sequence).

    Reuses ``write_intraday_cache`` so manually uploaded prices land in the
    same ``ida_prices_{zone}_seq{n}`` tables the live fetch and the cockpit
    already read, with no separate code path downstream.

    Args:
        long_df: Output of :func:`parse_intraday_csv`.

    Returns:
        One summary dict per (zone, sequence) group with keys ``zone``,
        ``sequence``, ``rows``, ``first``, ``last``.
    """
    summaries: list[dict[str, Any]] = []
    if long_df is None or long_df.empty:
        return summaries
    for (zone, sequence), grp in long_df.groupby(["zone", "sequence"], sort=True):
        frame = grp[["intraday_price_eur_mwh"]]
        write_intraday_cache(
            frame, str(zone), int(sequence), source=IDA_SOURCE_MANUAL,
        )
        idx = pd.DatetimeIndex(frame.index)
        summaries.append({
            "zone": str(zone),
            "sequence": int(sequence),
            "rows": len(frame),
            "first": idx.min(),
            "last": idx.max(),
        })
    return summaries


# ── Import zone-name safety (shared by the unified capacity + activation imports)
# Raw CSV ``zone`` values flow into per-zone SQLite table names
# (``capacity_prices_{zone}`` / ``activation_prices_{zone}``), so validate they
# are table-name safe BEFORE interpolation. This is deliberately NOT the
# supported-bidding-zone check (``_validate_zone``): import-first may legitimately
# ingest a zone code not yet wired into config, so we enforce only table-name
# safety (letters/digits/underscore), not business support.
_IMPORT_ZONE_SAFE_RE = re.compile(r"[A-Za-z0-9_]+")


def validate_import_zone(zone: object) -> str:
    """Return the stripped zone if table-name safe, else raise.

    Safe = non-empty and only ASCII letters, digits, and underscores, so it can
    be embedded in a ``{prefix}_{zone}`` SQLite table name without quoting or
    injection risk. Raises ``DataSourceParseError`` on an empty or unsafe value.
    Used by both unified importers' table-name helpers (the choke point) AND
    their parsers (early, friendly error), so a malformed/malicious zone can
    never reach raw SQL.
    """
    zone_str = str(zone).strip()
    if not _IMPORT_ZONE_SAFE_RE.fullmatch(zone_str):
        raise DataSourceParseError(
            f"Invalid or unsafe import zone {zone!r}; a zone must be non-empty "
            "and contain only letters, digits, and underscores."
        )
    return zone_str


# ── Unified reserve-capacity cache + provenance (Step 2 / 6b) ──────────────────
# Capacity parity with the IDA cache above: one table per zone holding all
# (product, direction) streams, plus a provenance sidecar keyed per
# (zone, product, direction) so a market stream with a distinct source is
# labelled independently (e.g. DE_LU FCR symmetric = Manual CSV vs DE_LU aFRR up
# = TSO API). The parser lives in ``ancillary.parse_capacity_import_csv``.

CAPACITY_SOURCE_MANUAL = "Manual CSV"
CAPACITY_SOURCE_MIXED = "Mixed"
CAPACITY_SOURCE_LEGACY = "Unknown (pre-provenance cache)"
_CAPACITY_SOURCE_TABLE = "capacity_price_sources"


def _capacity_cache_table(zone: str) -> str:
    """SQLite table name for one zone's reserve-capacity prices.

    Validates the zone is table-name safe (the choke point that also guards a
    caller bypassing the parser, e.g. ``persist_capacity_frame`` directly).
    """
    return f"capacity_prices_{validate_import_zone(zone).lower()}"


def write_capacity_cache(
    df: pd.DataFrame, zone: str, *, source: str = CAPACITY_SOURCE_MANUAL,
) -> None:
    """Persist reserve-capacity rows for one zone (keep-last on the key).

    The key is ``(timestamp, product, direction)``, so re-importing the same
    block overwrites it (``INSERT OR REPLACE``) rather than duplicating. The
    per-row ``source`` feeds the ``capacity_price_sources`` sidecar, which is
    re-derived per (product, direction) AFTER the write — so an overwrite that
    changes the source updates provenance too (no stale label on fresh data).
    ``df`` must carry ``product_type``, ``direction``, ``capacity_price_eur_mw``
    columns on a timestamp index (the ``ancillary.parse_capacity_import_csv``
    output).
    """
    if df is None or df.empty:
        return
    if not {"product_type", "direction", "capacity_price_eur_mw"}.issubset(df.columns):
        return
    rows = []
    for ts, row in df.iterrows():
        price = row["capacity_price_eur_mw"]
        if pd.isna(price):
            continue
        direction = row.get("direction", "")
        direction_str = "" if pd.isna(direction) else str(direction).strip()
        rows.append((
            ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            str(row["product_type"]), direction_str,
            float(price), source,
        ))
    if not rows:
        return
    table = _capacity_cache_table(zone)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{table}" '
            "(timestamp TEXT, product TEXT, direction TEXT, "
            "capacity_price_eur_mw REAL NOT NULL, source TEXT, "
            "PRIMARY KEY (timestamp, product, direction))"
        )
        conn.executemany(
            f'INSERT OR REPLACE INTO "{table}" '
            "(timestamp, product, direction, capacity_price_eur_mw, source) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        _record_capacity_sources(conn, table, zone)


def _record_capacity_sources(
    conn: sqlite3.Connection, table: str, zone: str,
) -> None:
    """Re-derive the provenance sidecar per (zone, product, direction).

    Mirrors ``_record_intraday_source`` but at (product, direction) granularity:
    a stream's label is the single source when uniform across its rows, else
    ``Mixed``. Re-deriving from the table after each write keeps provenance in
    lockstep with the (possibly overwritten) data.
    """
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{_CAPACITY_SOURCE_TABLE}" '
        "(zone TEXT, product TEXT, direction TEXT, source TEXT, rows INTEGER, "
        "first_timestamp TEXT, last_timestamp TEXT, imported_at TEXT, "
        "PRIMARY KEY (zone, product, direction))"
    )
    now = datetime.now(UTC).isoformat()
    groups = conn.execute(
        f'SELECT product, direction, COUNT(*), MIN(timestamp), MAX(timestamp) '
        f'FROM "{table}" GROUP BY product, direction'
    ).fetchall()
    for product, direction, n, first, last in groups:
        distinct = sorted(
            r[0] for r in conn.execute(
                f'SELECT DISTINCT source FROM "{table}" '
                "WHERE product = ? AND direction = ?",
                (product, direction),
            ) if r[0] is not None
        )
        if not distinct:
            label = CAPACITY_SOURCE_LEGACY
        elif len(distinct) == 1:
            label = distinct[0]
        else:
            label = CAPACITY_SOURCE_MIXED
        conn.execute(
            f'INSERT OR REPLACE INTO "{_CAPACITY_SOURCE_TABLE}" '
            "(zone, product, direction, source, rows, first_timestamp, "
            "last_timestamp, imported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (zone, product, direction, label, int(n), first, last, now),
        )


def read_capacity_sources() -> dict[tuple[str, str, str], dict[str, Any]]:
    """Return durable reserve-capacity provenance keyed by
    (zone, product, direction). Empty when the sidecar does not exist yet."""
    if not DB_PATH.exists():
        return {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            recorded = conn.execute(
                f'SELECT zone, product, direction, source, rows, '
                f'first_timestamp, last_timestamp, imported_at '
                f'FROM "{_CAPACITY_SOURCE_TABLE}"'
            ).fetchall()
    except sqlite3.DatabaseError:
        return {}
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for zone, product, direction, source, rows, first, last, imported_at in recorded:
        out[(str(zone), str(product), str(direction))] = {
            "source": source,
            "rows": int(rows),
            "first": pd.to_datetime(first, utc=True, errors="coerce"),
            "last": pd.to_datetime(last, utc=True, errors="coerce"),
            "imported_at": imported_at,
        }
    return out


def read_capacity_cache(
    zone: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Return cached reserve-capacity rows for a zone, or None when empty.

    Columns ``product_type, direction, capacity_price_eur_mw`` on a UTC
    timestamp index — the shape ``build_ancillary_dataset`` / the cockpit
    capacity helpers consume. ``[start, end)`` filters when both are given.
    """
    if not DB_PATH.exists():
        return None
    table = _capacity_cache_table(zone)
    query = (
        f'SELECT timestamp, product, direction, capacity_price_eur_mw FROM "{table}"'
    )
    params: tuple = ()
    if start is not None and end is not None:
        query += " WHERE timestamp >= ? AND timestamp < ?"
        params = (start.isoformat(), end.isoformat())
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(query, conn, params=params)
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        return None
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    df.index.name = "timestamp"
    return df.rename(columns={"product": "product_type"})


def persist_capacity_frame(
    long_df: pd.DataFrame, *, source: str = CAPACITY_SOURCE_MANUAL,
) -> list[dict[str, Any]]:
    """Write a parsed capacity frame to SQLite, one table per zone.

    Args:
        long_df: Output of ``ancillary.parse_capacity_import_csv`` (timestamp
            index; ``zone``, ``product_type``, ``direction``,
            ``capacity_price_eur_mw`` columns).
        source: Provenance label stored per row.

    Returns:
        One summary dict per (zone, product, direction) written.
    """
    summaries: list[dict[str, Any]] = []
    if long_df is None or long_df.empty or "zone" not in long_df.columns:
        return summaries
    for zone, grp in long_df.groupby("zone", sort=True):
        zone_str = str(zone).strip()
        if not zone_str:
            continue
        write_capacity_cache(grp, zone_str, source=source)
        for (product, direction), sub in grp.groupby(
            ["product_type", "direction"], sort=True,
        ):
            summaries.append({
                "zone": zone_str,
                "product": str(product),
                "direction": str(direction),
                "rows": len(sub),
                "source": source,
            })
    return summaries


# ── Unified activation-energy cache + provenance (Step 3 / 3b) ─────────────────
# Energy-leg parity with the capacity cache above: one table per zone holding all
# (product, direction) activation streams + a provenance sidecar keyed per
# (zone, product, direction). ``system_activated_volume_mw`` is stored
# SYSTEM-level exactly as imported (red-line: the asset/capture share is a model
# assumption, never multiplied in at persistence). Parser lives in
# ``ancillary.parse_activation_import_csv``.

ACTIVATION_SOURCE_MANUAL = "Manual CSV"
ACTIVATION_SOURCE_MIXED = "Mixed"
ACTIVATION_SOURCE_LEGACY = "Unknown (pre-provenance cache)"
_ACTIVATION_SOURCE_TABLE = "activation_price_sources"


def _activation_cache_table(zone: str) -> str:
    """SQLite table name for one zone's activation-energy prices.

    Validates the zone is table-name safe (the choke point that also guards a
    caller bypassing the parser, e.g. ``persist_activation_frame`` directly).
    """
    return f"activation_prices_{validate_import_zone(zone).lower()}"


def write_activation_cache(
    df: pd.DataFrame, zone: str, *, source: str = ACTIVATION_SOURCE_MANUAL,
) -> None:
    """Persist activation-energy rows for one zone (keep-last on the key).

    Key ``(timestamp, product, direction)`` — re-importing the same interval
    overwrites it (``INSERT OR REPLACE``). The per-row ``source`` feeds the
    ``activation_price_sources`` sidecar, re-derived per (product, direction)
    AFTER the write so an overwrite that changes the source refreshes provenance.
    ``df`` must carry ``product_type``, ``direction``, ``activation_price_eur_mwh``,
    ``system_activated_volume_mw`` on a timestamp index (the
    ``ancillary.parse_activation_import_csv`` output).
    """
    if df is None or df.empty:
        return
    needed = {
        "product_type", "direction",
        "activation_price_eur_mwh", "system_activated_volume_mw",
    }
    if not needed.issubset(df.columns):
        return
    rows = []
    for ts, row in df.iterrows():
        price = row["activation_price_eur_mwh"]
        volume = row["system_activated_volume_mw"]
        if pd.isna(price) or pd.isna(volume):
            continue
        direction = row.get("direction", "")
        direction_str = "" if pd.isna(direction) else str(direction).strip()
        rows.append((
            ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            str(row["product_type"]), direction_str,
            float(price), float(volume), source,
        ))
    if not rows:
        return
    table = _activation_cache_table(zone)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{table}" '
            "(timestamp TEXT, product TEXT, direction TEXT, "
            "activation_price_eur_mwh REAL NOT NULL, "
            "system_activated_volume_mw REAL NOT NULL, source TEXT, "
            "PRIMARY KEY (timestamp, product, direction))"
        )
        conn.executemany(
            f'INSERT OR REPLACE INTO "{table}" '
            "(timestamp, product, direction, activation_price_eur_mwh, "
            "system_activated_volume_mw, source) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        _record_activation_sources(conn, table, zone)


def _record_activation_sources(
    conn: sqlite3.Connection, table: str, zone: str,
) -> None:
    """Re-derive the activation provenance sidecar per (zone, product, direction).

    Same logic as ``_record_capacity_sources``: a stream's label is the single
    source when uniform, else ``Mixed``; re-derived from the table after each
    write so provenance stays in lockstep with overwritten data.
    """
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{_ACTIVATION_SOURCE_TABLE}" '
        "(zone TEXT, product TEXT, direction TEXT, source TEXT, rows INTEGER, "
        "first_timestamp TEXT, last_timestamp TEXT, imported_at TEXT, "
        "PRIMARY KEY (zone, product, direction))"
    )
    now = datetime.now(UTC).isoformat()
    groups = conn.execute(
        f'SELECT product, direction, COUNT(*), MIN(timestamp), MAX(timestamp) '
        f'FROM "{table}" GROUP BY product, direction'
    ).fetchall()
    for product, direction, n, first, last in groups:
        distinct = sorted(
            r[0] for r in conn.execute(
                f'SELECT DISTINCT source FROM "{table}" '
                "WHERE product = ? AND direction = ?",
                (product, direction),
            ) if r[0] is not None
        )
        if not distinct:
            label = ACTIVATION_SOURCE_LEGACY
        elif len(distinct) == 1:
            label = distinct[0]
        else:
            label = ACTIVATION_SOURCE_MIXED
        conn.execute(
            f'INSERT OR REPLACE INTO "{_ACTIVATION_SOURCE_TABLE}" '
            "(zone, product, direction, source, rows, first_timestamp, "
            "last_timestamp, imported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (zone, product, direction, label, int(n), first, last, now),
        )


def read_activation_sources() -> dict[tuple[str, str, str], dict[str, Any]]:
    """Return durable activation provenance keyed by (zone, product, direction).
    Empty when the sidecar does not exist yet."""
    if not DB_PATH.exists():
        return {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            recorded = conn.execute(
                f'SELECT zone, product, direction, source, rows, '
                f'first_timestamp, last_timestamp, imported_at '
                f'FROM "{_ACTIVATION_SOURCE_TABLE}"'
            ).fetchall()
    except sqlite3.DatabaseError:
        return {}
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for zone, product, direction, source, rows, first, last, imported_at in recorded:
        out[(str(zone), str(product), str(direction))] = {
            "source": source,
            "rows": int(rows),
            "first": pd.to_datetime(first, utc=True, errors="coerce"),
            "last": pd.to_datetime(last, utc=True, errors="coerce"),
            "imported_at": imported_at,
        }
    return out


def read_activation_cache(
    zone: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Return cached activation-energy rows for a zone, or None when empty.

    Columns ``product_type, direction, activation_price_eur_mwh,
    system_activated_volume_mw`` on a UTC timestamp index. ``[start, end)``
    filters when both are given.
    """
    if not DB_PATH.exists():
        return None
    table = _activation_cache_table(zone)
    query = (
        f'SELECT timestamp, product, direction, activation_price_eur_mwh, '
        f'system_activated_volume_mw FROM "{table}"'
    )
    params: tuple = ()
    if start is not None and end is not None:
        query += " WHERE timestamp >= ? AND timestamp < ?"
        params = (start.isoformat(), end.isoformat())
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(query, conn, params=params)
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        return None
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    df.index.name = "timestamp"
    return df.rename(columns={"product": "product_type"})


def persist_activation_frame(
    long_df: pd.DataFrame, *, source: str = ACTIVATION_SOURCE_MANUAL,
) -> list[dict[str, Any]]:
    """Write a parsed activation frame to SQLite, one table per zone.

    Args:
        long_df: Output of ``ancillary.parse_activation_import_csv`` (timestamp
            index; ``zone``, ``product_type``, ``direction``,
            ``activation_price_eur_mwh``, ``system_activated_volume_mw`` columns).
        source: Provenance label stored per row.

    Returns:
        One summary dict per (zone, product, direction) written.
    """
    summaries: list[dict[str, Any]] = []
    if long_df is None or long_df.empty or "zone" not in long_df.columns:
        return summaries
    for zone, grp in long_df.groupby("zone", sort=True):
        zone_str = str(zone).strip()
        if not zone_str:
            continue
        write_activation_cache(grp, zone_str, source=source)
        for (product, direction), sub in grp.groupby(
            ["product_type", "direction"], sort=True,
        ):
            summaries.append({
                "zone": zone_str,
                "product": str(product),
                "direction": str(direction),
                "rows": len(sub),
                "source": source,
            })
    return summaries


# ── Unified reBAP / imbalance cache + provenance (Step 4 / 4b) ────────────────
# Passive imbalance-settlement parity with the activation cache above: one table
# per zone plus a provenance sidecar keyed by zone. ``system_imbalance_volume_mw``
# is stored SYSTEM/area-level exactly as imported (red-line: asset imbalance /
# capture share is a model assumption, never multiplied in at persistence).
# Parser lives in ``ancillary.parse_imbalance_import_csv``.

IMBALANCE_SOURCE_MANUAL = "Manual CSV"
IMBALANCE_SOURCE_MIXED = "Mixed"
IMBALANCE_SOURCE_LEGACY = "Unknown (pre-provenance cache)"
_IMBALANCE_SOURCE_TABLE = "imbalance_price_sources"


def _imbalance_cache_table(zone: str) -> str:
    """SQLite table name for one zone's reBAP / imbalance prices.

    Validates the zone is table-name safe (the choke point that also guards a
    caller bypassing the parser, e.g. ``persist_imbalance_frame`` directly).
    """
    return f"imbalance_prices_{validate_import_zone(zone).lower()}"


def write_imbalance_cache(
    df: pd.DataFrame, zone: str, *, source: str = IMBALANCE_SOURCE_MANUAL,
) -> None:
    """Persist imbalance-settlement rows for one zone (keep-last on timestamp).

    Key ``timestamp`` — re-importing the same settlement interval overwrites it
    (``INSERT OR REPLACE``). The per-row ``source`` feeds the
    ``imbalance_price_sources`` sidecar, re-derived AFTER the write so an
    overwrite that changes the source refreshes provenance. ``df`` must carry
    ``imbalance_price_eur_mwh`` and ``system_imbalance_volume_mw`` on a
    timestamp index (the ``ancillary.parse_imbalance_import_csv`` output).
    """
    if df is None or df.empty:
        return
    needed = {"imbalance_price_eur_mwh", "system_imbalance_volume_mw"}
    if not needed.issubset(df.columns):
        return
    rows = []
    for ts, row in df.iterrows():
        price = row["imbalance_price_eur_mwh"]
        volume = row["system_imbalance_volume_mw"]
        if pd.isna(price) or pd.isna(volume):
            continue
        rows.append((
            ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            float(price), float(volume), source,
        ))
    if not rows:
        return
    table = _imbalance_cache_table(zone)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{table}" '
            "(timestamp TEXT PRIMARY KEY, "
            "imbalance_price_eur_mwh REAL NOT NULL, "
            "system_imbalance_volume_mw REAL NOT NULL, source TEXT)"
        )
        conn.executemany(
            f'INSERT OR REPLACE INTO "{table}" '
            "(timestamp, imbalance_price_eur_mwh, "
            "system_imbalance_volume_mw, source) VALUES (?, ?, ?, ?)",
            rows,
        )
        _record_imbalance_source(conn, table, zone)


def _record_imbalance_source(
    conn: sqlite3.Connection, table: str, zone: str,
) -> None:
    """Re-derive the imbalance provenance sidecar for one zone."""
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{_IMBALANCE_SOURCE_TABLE}" '
        "(zone TEXT PRIMARY KEY, source TEXT, rows INTEGER, "
        "first_timestamp TEXT, last_timestamp TEXT, imported_at TEXT)"
    )
    row = conn.execute(
        f'SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM "{table}"'
    ).fetchone()
    if row is None:
        return
    n, first, last = row
    if not n:
        return
    distinct = sorted(
        r[0] for r in conn.execute(
            f'SELECT DISTINCT source FROM "{table}"',
        ) if r[0] is not None
    )
    if not distinct:
        label = IMBALANCE_SOURCE_LEGACY
    elif len(distinct) == 1:
        label = distinct[0]
    else:
        label = IMBALANCE_SOURCE_MIXED
    conn.execute(
        f'INSERT OR REPLACE INTO "{_IMBALANCE_SOURCE_TABLE}" '
        "(zone, source, rows, first_timestamp, last_timestamp, imported_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (zone, label, int(n), first, last, datetime.now(UTC).isoformat()),
    )


def read_imbalance_sources() -> dict[str, dict[str, Any]]:
    """Return durable imbalance provenance keyed by zone.

    Empty when the sidecar does not exist yet.
    """
    if not DB_PATH.exists():
        return {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            recorded = conn.execute(
                f'SELECT zone, source, rows, first_timestamp, last_timestamp, '
                f'imported_at FROM "{_IMBALANCE_SOURCE_TABLE}"'
            ).fetchall()
    except sqlite3.DatabaseError:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for zone, source, rows, first, last, imported_at in recorded:
        out[str(zone)] = {
            "source": source,
            "rows": int(rows),
            "first": pd.to_datetime(first, utc=True, errors="coerce"),
            "last": pd.to_datetime(last, utc=True, errors="coerce"),
            "imported_at": imported_at,
        }
    return out


def read_imbalance_cache(
    zone: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Return cached imbalance-settlement rows for a zone, or None when empty.

    Columns ``imbalance_price_eur_mwh`` and ``system_imbalance_volume_mw`` on a
    UTC timestamp index. ``[start, end)`` filters when both are given.
    """
    if not DB_PATH.exists():
        return None
    table = _imbalance_cache_table(zone)
    query = (
        f'SELECT timestamp, imbalance_price_eur_mwh, system_imbalance_volume_mw '
        f'FROM "{table}"'
    )
    params: tuple = ()
    if start is not None and end is not None:
        query += " WHERE timestamp >= ? AND timestamp < ?"
        params = (start.isoformat(), end.isoformat())
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(query, conn, params=params)
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        return None
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    df.index.name = "timestamp"
    return df


def persist_imbalance_frame(
    long_df: pd.DataFrame, *, source: str = IMBALANCE_SOURCE_MANUAL,
) -> list[dict[str, Any]]:
    """Write a parsed imbalance frame to SQLite, one table per zone.

    Args:
        long_df: Output of ``ancillary.parse_imbalance_import_csv`` (timestamp
            index; ``zone``, ``imbalance_price_eur_mwh``,
            ``system_imbalance_volume_mw`` columns).
        source: Provenance label stored per row.

    Returns:
        One summary dict per zone written.
    """
    summaries: list[dict[str, Any]] = []
    if long_df is None or long_df.empty or "zone" not in long_df.columns:
        return summaries
    for zone, grp in long_df.groupby("zone", sort=True):
        zone_str = str(zone).strip()
        if not zone_str:
            continue
        write_imbalance_cache(grp, zone_str, source=source)
        summaries.append({
            "zone": zone_str,
            "rows": len(grp),
            "source": source,
        })
    return summaries


def fetch_intraday_prices(
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    sequence: int = 1,
) -> pd.DataFrame | None:
    """Fetch ENTSO-E intraday auction (IDA) prices for a zone.

    Args:
        zone: Bidding zone code.
        start: Start timestamp (UTC).
        end: End timestamp (UTC).
        sequence: IDA round — 1 (15:00 D-1), 2 (22:00 D-1), or 3 (10:00 day-of).

    Returns:
        DataFrame indexed by UTC timestamp with column
        ``intraday_price_eur_mwh``, or None when the zone does not publish
        intraday auction results.
    """
    if zone not in INTRADAY_SUPPORTED_ZONES:
        logger.info("Intraday auction prices not available for %s", zone)
        return None
    if sequence not in (1, 2, 3):
        raise ValueError(f"IDA sequence must be 1, 2 or 3 (got {sequence})")

    try:
        client = EntsoePandasClient(api_key=get_api_key())
    except OSError as exc:
        raise DataSourceAuthError(
            "ENTSO-E API key missing or invalid. Set ENTSOE_API_KEY in .env."
        ) from exc

    start_q = start.tz_convert(DEFAULT_QUERY_TIMEZONE)
    end_q = end.tz_convert(DEFAULT_QUERY_TIMEZONE)

    logger.info("Fetching ENTSO-E IDA%d prices for %s", sequence, zone)
    try:
        series = client.query_intraday_prices(
            zone, start=start_q, end=end_q, sequence=sequence,
        )
    except _EntsoeNoMatchingDataError as exc:
        # entsoe-py raises NoMatchingDataError (Exception subclass, NOT
        # ValueError) when a supported zone has no IDA print in the window.
        # That is "no data," not an error — let the UI render a friendly
        # message rather than a stack trace.
        logger.info(
            "ENTSO-E IDA%d: no data for %s in window: %s",
            sequence, zone, _scrub_secrets(str(exc)),
        )
        return None
    except requests.HTTPError as exc:
        # entsoe-py propagates ENTSO-E 401/403 as HTTPError. Classify as
        # an auth error so the sidebar can prompt for ENTSOE_API_KEY
        # instead of surfacing a generic network failure.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise DataSourceAuthError(
                f"ENTSO-E auth failed (HTTP {status}) on IDA{sequence} for "
                f"{zone}. Check ENTSOE_API_KEY in .env."
            ) from None
        raise DataSourceNetworkError(
            f"ENTSO-E IDA{sequence} fetch failed for {zone}: "
            f"{_scrub_secrets(str(exc))}"
        ) from None
    except requests.RequestException as exc:
        raise DataSourceNetworkError(
            f"ENTSO-E IDA{sequence} fetch failed for {zone}: "
            f"{_scrub_secrets(str(exc))}"
        ) from None
    except (ValueError, TypeError, KeyError) as exc:
        # Some older entsoe-py releases raised ValueError for missing data;
        # keep the soft-fail to stay compatible with installations pinned
        # below 0.7.11.
        logger.info(
            "ENTSO-E IDA%d unavailable for %s: %s",
            sequence, zone, _scrub_secrets(str(exc)),
        )
        return None
    except Exception as exc:
        # entsoe-py custom exceptions (e.g. InvalidTokenError) can carry the
        # request URL — including securityToken — in their stringification.
        # Classify auth-flavoured strings and otherwise wrap as parse error;
        # never let the raw exception escape to Streamlit.
        scrubbed = _scrub_secrets(str(exc))
        logger.error(
            "ENTSO-E IDA%d fetch failed for %s: %s", sequence, zone, scrubbed,
        )
        if _looks_like_auth_error(exc):
            raise DataSourceAuthError(
                f"ENTSO-E auth failed on IDA{sequence} for {zone}. "
                "Check ENTSOE_API_KEY in .env."
            ) from None
        raise DataSourceParseError(
            f"ENTSO-E IDA{sequence} returned data for {zone} in an "
            f"unexpected format: {scrubbed}"
        ) from None

    if series is None or (isinstance(series, (pd.DataFrame, pd.Series)) and series.empty):
        return None

    df = series.to_frame(name="intraday_price_eur_mwh") if isinstance(series, pd.Series) else series
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    df["intraday_price_eur_mwh"] = pd.to_numeric(
        df.iloc[:, 0], errors="coerce",
    )
    out = df[["intraday_price_eur_mwh"]]
    # Persist to SQLite so browser refresh / page re-render doesn't trigger
    # a second slow ENTSO-E call. IDA prints are final once published, so
    # we can opportunistically cache without freshness logic.
    try:
        write_intraday_cache(out, zone, sequence)
    except sqlite3.DatabaseError as exc:  # pragma: no cover — defensive
        logger.warning("Could not persist IDA%d cache for %s: %s", sequence, zone, exc)
    return out


# ── REE ESIOS (Spain) ─────────────────────────────────────────────────────────

# Stable indicator IDs at ESIOS for the most common balancing-reserve products
# used in the Spanish electricity market. Capacity prices are EUR/MW, energy
# prices are EUR/MWh; ESIOS publishes them hourly. IDs are stable across
# ESIOS API versions but new SRS rules (Nov 2024) added 15-min indicators
# alongside the legacy hourly ones — we use the hourly set so cadence
# matches DA prices.
ESIOS_INDICATORS: dict[str, dict[str, int]] = {
    "secondary_up_capacity":   {"id": 634, "unit": "EUR/MW"},
    "secondary_down_capacity": {"id": 635, "unit": "EUR/MW"},
    "tertiary_up_energy":      {"id": 672, "unit": "EUR/MWh"},
    "tertiary_down_energy":    {"id": 673, "unit": "EUR/MWh"},
}


@retry(max_retries=3, backoff=2.0, exceptions=(requests.RequestException, ValueError))
def _call_esios_api(
    indicator_id: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """One ESIOS indicator GET request with retries.

    401/403 bypass retry as DataSourceAuthError so the UI can prompt for
    ESIOS_API_KEY rather than looping a guaranteed failure.
    """
    api_key = get_esios_api_key()
    if not api_key:
        raise DataSourceAuthError(
            "ESIOS_API_KEY missing. Request one at consultasios@ree.es "
            "and add it to .env."
        )
    headers = {
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }
    params = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    resp = requests.get(
        f"{ESIOS_BASE_URL}/indicators/{indicator_id}",
        params=params, headers=headers, timeout=30,
    )
    _raise_if_auth_failed(resp, "ESIOS", "Check ESIOS_API_KEY in .env.")
    resp.raise_for_status()
    return resp.json()


def _parse_esios_indicator(
    payload: dict[str, Any], column: str,
) -> pd.DataFrame:
    """Convert an ESIOS indicator JSON payload to a UTC-indexed DataFrame.

    The response shape is ``{"indicator": {"values": [{"datetime", "value",
    "geo_id", ...}, ...]}}``; we keep only the time series and drop geo
    breakdowns since BESS revenue valuation is at system level.
    """
    indicator = payload.get("indicator", {}) if isinstance(payload, dict) else {}
    values = indicator.get("values") or []
    if not values:
        return pd.DataFrame(columns=["timestamp", column])

    df = pd.DataFrame(values)
    if "datetime" not in df.columns or "value" not in df.columns:
        raise DataSourceParseError(
            f"ESIOS indicator payload missing datetime/value columns: "
            f"{list(df.columns)}"
        )
    df["timestamp"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df[column] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", column])
    # ESIOS sometimes returns rows per geo_id; collapse to one row per timestamp.
    # If per-geo values diverge meaningfully on the same timestamp, log a
    # warning — the simple mean can silently hide real geographic variation.
    # The caller can request explicit geo_ids / geo_agg ESIOS params to
    # override this default.
    if "geo_id" in df.columns:
        std_per_ts = df.groupby("timestamp")[column].transform("std").fillna(0)
        n_divergent = int((std_per_ts > 1.0).sum())
        if n_divergent > 0:
            logger.warning(
                "ESIOS indicator returned divergent per-geo values for %d "
                "row(s); collapsing by mean. Pass explicit geo_ids if "
                "geography matters for this product.",
                n_divergent,
            )
    df = df.groupby("timestamp", as_index=False)[column].mean()
    return df[["timestamp", column]].sort_values("timestamp").reset_index(drop=True)


def fetch_esios_indicator(
    indicator_id: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    column: str = "value",
) -> pd.DataFrame:
    """Fetch a single ESIOS indicator time series.

    Args:
        indicator_id: ESIOS indicator ID (see ``ESIOS_INDICATORS`` for the
            most common balancing-reserve products).
        start: Start timestamp (UTC).
        end: Exclusive end timestamp (UTC).
        column: Output column name (default ``"value"``).

    Returns:
        DataFrame with columns ``[timestamp, <column>]`` (UTC, hourly).
    """
    logger.info("Fetching ESIOS indicator %d", indicator_id)
    try:
        payload = _call_esios_api(indicator_id, start, end)
    except requests.RequestException as exc:
        # ESIOS passes the API key via ``x-api-key`` header (not URL), so
        # the exception string typically does NOT carry credentials.
        # The request URL still embeds query params (start_date, end_date,
        # indicator_id) — not sensitive, but Streamlit panels render
        # __cause__ chains, leaking the full upstream URL into the UI.
        # Match the ENTSO-E pattern from PR2: scrub + ``from None`` so
        # only the cleaned message reaches the user.
        raise DataSourceNetworkError(
            f"ESIOS indicator {indicator_id} fetch failed: "
            f"{_scrub_secrets(str(exc))}"
        ) from None
    return _parse_esios_indicator(payload, column)


def fetch_esios_ancillary_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch the ESIOS bundle of secondary/tertiary reserve prices.

    Returns:
        DataFrame with columns:
        ``[timestamp, secondary_up_capacity_eur_mw,
        secondary_down_capacity_eur_mw, tertiary_up_energy_eur_mwh,
        tertiary_down_energy_eur_mwh]``.

        Individual indicator failures are logged but don't fail the call —
        the user still gets whatever indicators ESIOS returned. An
        all-empty result returns an empty frame with the schema columns
        present so downstream merges remain stable.
    """
    column_map = {
        "secondary_up_capacity":   "secondary_up_capacity_eur_mw",
        "secondary_down_capacity": "secondary_down_capacity_eur_mw",
        "tertiary_up_energy":      "tertiary_up_energy_eur_mwh",
        "tertiary_down_energy":    "tertiary_down_energy_eur_mwh",
    }
    merged: pd.DataFrame | None = None
    for slug, output_col in column_map.items():
        meta = ESIOS_INDICATORS[slug]
        try:
            df = fetch_esios_indicator(meta["id"], start, end, column=output_col)
        except (DataSourceNetworkError, DataSourceParseError) as exc:
            logger.warning(
                "ESIOS indicator %d (%s) unavailable: %s",
                meta["id"], slug, exc,
            )
            continue
        if df.empty:
            continue
        merged = df if merged is None else merged.merge(df, on="timestamp", how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame(columns=["timestamp", *column_map.values()])
    return merged.sort_values("timestamp").reset_index(drop=True)


# ── Connection tests ──────────────────────────────────────────────────────────

def test_elexon_connection() -> None:
    """Test Elexon API with a 1-day request. No API key needed."""
    end = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.Timedelta(days=1)

    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Elexon API for yesterday's GB data...")

    df = fetch_elexon_prices(start, end)
    if df.empty:
        logger.warning("Elexon returned no data.")
        return

    logger.info("Elexon test successful: %d rows", len(df))
    logger.info("Price stats (EUR/MWh):\n%s", df["price_eur_mwh"].describe())
    logger.info("First 5 rows:\n%s", df.head())


def test_entsoe_connection() -> None:
    """Test ENTSO-E API with a 7-day request for DE_LU and FR.

    Requires valid ENTSOE_API_KEY in .env.
    """
    end = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.Timedelta(days=7)

    logging.basicConfig(level=logging.INFO)
    client = EntsoePandasClient(api_key=get_api_key())

    for zone in ["DE_LU", "FR"]:
        logger.info("Testing ENTSO-E for %s...", zone)
        df = fetch_entsoe_prices(zone, start, end, client=client)
        if df.empty:
            logger.warning("No data for %s", zone)
            continue
        logger.info("%s: %d rows", zone, len(df))
        logger.info("Price stats:\n%s", df["price_eur_mwh"].describe())
        logger.info("First 5 rows:\n%s", df.head())
