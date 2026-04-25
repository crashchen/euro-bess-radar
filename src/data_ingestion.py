"""Unified data ingestion for eu-bess-pulse: ENTSO-E + Elexon APIs."""

from __future__ import annotations

from datetime import date
import functools
import logging
import sqlite3
import time
from typing import Any, Callable

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
    FINGRID_BASE_URL,
    GBP_EUR_YEARLY,
    PRICE_CACHE_TTL_HOURS,
    REGELLEISTUNG_API_URL,
    get_api_key,
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


# ── Retry decorator ──────────────────────────────────────────────────────────

def retry(
    max_retries: int = 3,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[..., Any]:
    """Decorator: retry with exponential backoff on exception."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        wait = backoff ** attempt
                        logger.warning(
                            "%s failed (attempt %d/%d), retrying in %.1fs: %s",
                            func.__name__, attempt + 1, max_retries, wait, exc,
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
            for segment_start, segment_end in zip(split_points, split_points[1:]):
                interval = _expected_interval_for_segment(
                    zone,
                    segment_start,
                    inferred or fallback,
                )
                expected_points = max(int(round((segment_end - segment_start) / interval)), 1)
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
        expected_points = max(int(round((end - start) / interval)), 1)
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
    expected_points = max(int(round((segment_end - segment_start) / interval)), 1)
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
        Cleaned DataFrame with gaps forward-filled, no NaN remaining, plus a
        `filled` flag marking rows that were synthetic or missing in source data.
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
    df["filled"] = df["price_eur_mwh"].isna()
    df["price_eur_mwh"] = df["price_eur_mwh"].ffill().bfill()
    df["filled"] = df["filled"].astype(bool)
    return df


# ── ENTSO-E fetcher ──────────────────────────────────────────────────────────

@retry(max_retries=3, backoff=2.0)
def _call_entsoe_api(
    client: EntsoePandasClient, zone: str,
    start: pd.Timestamp, end: pd.Timestamp,
) -> pd.Series:
    """Call entsoe-py with retry logic."""
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
        logger.exception("ENTSO-E request failed for %s", zone)
        raise DataSourceNetworkError(
            f"ENTSO-E request failed for {zone}. Please retry."
        ) from exc
    except Exception as exc:
        logger.exception("ENTSO-E fetch failed for %s", zone)
        if _looks_like_auth_error(exc):
            raise DataSourceAuthError(
                "ENTSO-E API authentication failed. Check your token and permissions."
            ) from exc
        raise DataSourceParseError(
            f"ENTSO-E returned data for {zone} in an unexpected format."
        ) from exc

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
        for segment_start, segment_end in zip(split_points, split_points[1:]):
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


@retry(max_retries=3, backoff=2.0)
def _call_entsoe_generation(
    client: EntsoePandasClient, zone: str,
    start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    """Call entsoe-py generation query with retry."""
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
        logger.exception("ENTSO-E generation request failed for %s", zone)
        raise DataSourceNetworkError(
            f"Generation data request failed for {zone}. Please retry."
        ) from exc
    except Exception as exc:
        logger.exception("ENTSO-E generation fetch failed for %s", zone)
        if _looks_like_auth_error(exc):
            raise DataSourceAuthError(
                "ENTSO-E API authentication failed while fetching generation data."
            ) from exc
        raise DataSourceParseError(
            f"Generation data for {zone} could not be parsed or was unavailable."
        ) from exc

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
    """Fetch one Fingrid API page with retries."""
    resp = requests.get(
        f"{FINGRID_BASE_URL}/datasets/{dataset_id}/data",
        params=params,
        headers=headers,
        timeout=30,
    )
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
            logger.warning(
                "Fingrid fetch failed for dataset %d on page %d after retries; "
                "discarding %d collected rows and returning empty data: %s",
                dataset_id,
                page,
                len(all_rows),
                exc,
            )
            return pd.DataFrame(columns=["timestamp", "value"])

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
    """Download one day of Regelleistung tender results as xlsx bytes."""
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
    resp.raise_for_status()
    return resp.content


def _parse_regelleistung_xlsx(
    content: bytes, product: str, target_date: str,
) -> pd.DataFrame:
    """Parse Regelleistung xlsx export into standard ancillary format."""
    import openpyxl
    from io import BytesIO

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
        row_dict = dict(zip(headers, row))
        price = None
        for key in ("capacity price [eur/mw]", "capacity_price", "price", "ergebnispreis"):
            if key in row_dict and row_dict[key] is not None:
                try:
                    price = float(row_dict[key])
                except (ValueError, TypeError):
                    continue
                break
        if price is None:
            continue

        direction = "Symmetric"
        for key in ("direction", "richtung", "product_name"):
            if key in row_dict and row_dict[key]:
                val = str(row_dict[key]).strip().upper()
                if "UP" in val or "POS" in val:
                    direction = "Up"
                elif "DOWN" in val or "NEG" in val:
                    direction = "Down"
                break

        time_block = ""
        for key in ("time_block", "von", "from", "delivery_date"):
            if key in row_dict and row_dict[key]:
                time_block = row_dict[key]
                break

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
        except Exception as exc:
            logger.warning(
                "Regelleistung parse failed for %s on %s: %s", product, date_str, exc,
            )

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
        logger.warning("Cannot create ENTSO-E client: %s", exc)
        return None

    start_q = start.tz_convert(DEFAULT_QUERY_TIMEZONE)
    end_q = end.tz_convert(DEFAULT_QUERY_TIMEZONE)

    logger.info("Fetching ENTSO-E imbalance prices for %s", zone)
    try:
        raw = client.query_imbalance_prices(zone, start=start_q, end=end_q)
    except requests.RequestException as exc:
        logger.warning("ENTSO-E imbalance request failed for %s: %s", zone, exc)
        return None
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("ENTSO-E imbalance data unavailable for %s: %s", zone, exc)
        return None
    except Exception as exc:
        logger.warning("ENTSO-E imbalance fetch unavailable for %s: %s", zone, exc)
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
