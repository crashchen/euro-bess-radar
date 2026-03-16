"""Unified data ingestion for eu-bess-pulse: ENTSO-E + Elexon APIs."""

from __future__ import annotations

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
    GBP_TO_EUR,
    get_api_key,
    is_elexon_zone,
)

logger = logging.getLogger(__name__)


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


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw price data: handle gaps, outliers, timezone normalisation.

    Args:
        df: DataFrame with DatetimeIndex named 'timestamp' and column 'price_eur_mwh'.

    Returns:
        Cleaned DataFrame with gaps forward-filled, no NaN remaining.
    """
    if df.empty:
        return df

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if len(df) >= 2:
        deltas = pd.Series(df.index).diff().dropna()
        positive = deltas[deltas > pd.Timedelta(0)]
        freq = positive.mode().iloc[0] if not positive.empty else pd.Timedelta(hours=1)
    else:
        freq = pd.Timedelta(hours=1)

    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_idx)
    df.index.name = "timestamp"

    df["price_eur_mwh"] = df["price_eur_mwh"].ffill().bfill()
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
        client = EntsoePandasClient(api_key=get_api_key())

    start_q = start.tz_convert(DEFAULT_QUERY_TIMEZONE)
    end_q = end.tz_convert(DEFAULT_QUERY_TIMEZONE)

    logger.info("Fetching ENTSO-E data for %s [%s, %s)", zone, start_q, end_q)
    raw = _call_entsoe_api(client, zone, start_q, end_q)

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

@retry(max_retries=3, backoff=2.0, exceptions=(requests.RequestException,))
def _call_elexon_api(date_str: str, next_date_str: str) -> list[dict[str, Any]]:
    """Fetch one day of Elexon market index data."""
    resp = requests.get(
        ELEXON_MARKET_INDEX_ENDPOINT,
        params={"from": date_str, "to": next_date_str, "format": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


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
        Original GBP/MWh prices converted to EUR using GBP_TO_EUR rate.
        30-min settlement periods, UTC timestamps.
    """
    all_records: list[dict[str, Any]] = []
    current = start.normalize()
    end_date = end.normalize()

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        next_date = current + pd.Timedelta(days=1)
        next_date_str = next_date.strftime("%Y-%m-%d")

        logger.info("Fetching Elexon data for %s", date_str)
        try:
            data = _call_elexon_api(date_str, next_date_str)
            if isinstance(data, list):
                all_records.extend(data)
            elif isinstance(data, dict) and "data" in data:
                all_records.extend(data["data"])
        except Exception:
            logger.warning("Failed to fetch Elexon data for %s", date_str)

        current = next_date
        time.sleep(0.5)

    if not all_records:
        logger.warning("Elexon returned no data for [%s, %s)", start, end)
        return pd.DataFrame(columns=["price_eur_mwh"])

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["startTime"], utc=True)
    df["price_gbp"] = pd.to_numeric(df["price"], errors="coerce")

    # Each settlement period has multiple data providers (e.g. one with a
    # real price and one reporting 0).  Drop zeros, then average any
    # remaining duplicates per timestamp.
    df = df[df["price_gbp"] > 0]
    df = df.groupby("timestamp", as_index=True)["price_gbp"].mean().to_frame()
    df["price_eur_mwh"] = df["price_gbp"] * GBP_TO_EUR
    df = df[["price_eur_mwh"]].sort_index()
    return df


# ── Cache (SQLite + CSV) ─────────────────────────────────────────────────────

def write_cache(df: pd.DataFrame, zone: str) -> None:
    """Write DataFrame to SQLite table and CSV file."""
    if df.empty:
        return

    table_name = f"da_prices_{zone.lower()}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = CACHE_DIR / f"{table_name}.csv"
    export = df.reset_index()
    export["timestamp"] = export["timestamp"].map(lambda t: t.isoformat())
    export["zone"] = zone
    export.to_csv(csv_path, index=False)
    logger.info("Wrote CSV cache: %s", csv_path)

    # SQLite
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                timestamp TEXT PRIMARY KEY,
                price_eur_mwh REAL NOT NULL,
                zone TEXT NOT NULL
            )
        """)
        rows = [
            (row.timestamp.isoformat() if hasattr(row.timestamp, "isoformat") else str(row.timestamp),
             float(row.price_eur_mwh), zone)
            for row in df.reset_index().itertuples(index=False)
        ]
        conn.executemany(
            f'INSERT OR REPLACE INTO "{table_name}" (timestamp, price_eur_mwh, zone) VALUES (?, ?, ?)',
            rows,
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
                f'SELECT timestamp, price_eur_mwh FROM "{table_name}" '
                "WHERE timestamp >= ? AND timestamp < ?"
            )
            df = pd.read_sql_query(
                query, conn,
                params=(start.isoformat(), end.isoformat()),
                parse_dates=["timestamp"],
            )
    except Exception:
        return None

    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    df["zone"] = zone

    # Validate coverage: reject partial cache (span + density)
    requested_hours = (end - start).total_seconds() / 3600
    if requested_hours > 48:
        cached_span = (df.index.max() - df.index.min()).total_seconds() / 3600
        # Span check: cached range must cover >=80% of the requested range
        if cached_span < requested_hours * 0.8:
            logger.info(
                "Cache span too short for %s: %.0fh cached vs %.0fh requested",
                zone, cached_span, requested_hours,
            )
            return None
        # Density check: expect at least 1 row per 2 hours on average
        expected_rows = requested_hours / 2
        if len(df) < expected_rows:
            logger.info(
                "Cache too sparse for %s: %d rows vs %.0f expected minimum",
                zone, len(df), expected_rows,
            )
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
    df = clean_prices(df)
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

    client = EntsoePandasClient(api_key=get_api_key())
    start_q = start.tz_convert(DEFAULT_QUERY_TIMEZONE)
    end_q = end.tz_convert(DEFAULT_QUERY_TIMEZONE)

    logger.info("Fetching generation data for %s", zone)
    try:
        raw = _call_entsoe_generation(client, zone, start_q, end_q)
    except Exception as exc:
        logger.warning("Generation data unavailable for %s: %s", zone, exc)
        return pd.DataFrame(columns=_RENEWABLE_COLS)

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
    solar_cols = [c for c in raw.columns if "solar" in c.lower()]
    wind_on_cols = [c for c in raw.columns if "wind" in c.lower() and "offshore" not in c.lower()]
    wind_off_cols = [c for c in raw.columns if "wind" in c.lower() and "offshore" in c.lower()]

    df["solar_mw"] = raw[solar_cols].sum(axis=1) if solar_cols else 0.0
    df["wind_onshore_mw"] = raw[wind_on_cols].sum(axis=1) if wind_on_cols else 0.0
    df["wind_offshore_mw"] = raw[wind_off_cols].sum(axis=1) if wind_off_cols else 0.0
    df["total_renewable_mw"] = df["solar_mw"] + df["wind_onshore_mw"] + df["wind_offshore_mw"]
    df["total_generation_mw"] = raw.sum(axis=1)
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
                "publishDateTimeFrom": start.isoformat(),
                "publishDateTimeTo": end.isoformat(),
                "format": "json",
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
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

FINGRID_BASE_URL = "https://data.fingrid.fi/api/data"


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

    while True:
        params = {
            "datasets": str(dataset_id),
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
            resp = requests.get(
                FINGRID_BASE_URL, params=params, timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("Fingrid fetch failed for dataset %d: %s", dataset_id, exc)
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
        "fcr_n_price": 318,
        "fcr_d_up_price": 319,
        "fcr_d_down_price": 320,
    }
    merged: pd.DataFrame | None = None
    for col_name, ds_id in datasets.items():
        df = fetch_fingrid_data(ds_id, start, end)
        if df.empty:
            continue
        series = df.set_index("timestamp")["value"].rename(col_name)
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
        "afrr_up_price": 795,
        "afrr_down_price": 796,
    }
    merged: pd.DataFrame | None = None
    for col_name, ds_id in datasets.items():
        df = fetch_fingrid_data(ds_id, start, end)
        if df.empty:
            continue
        series = df.set_index("timestamp")["value"].rename(col_name)
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

REGELLEISTUNG_BASE_URL = (
    "https://www.regelleistung.net/apps/datacenter/tendering-files"
)


def fetch_regelleistung_results(
    product: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    """Attempt to fetch German balancing auction results from regelleistung.net.

    Args:
        product: One of "FCR", "aFRR", "mFRR".
        start: Start timestamp (UTC).
        end: End timestamp (UTC).

    Returns:
        DataFrame with columns:
        [date, product, time_block, capacity_price_eur_mw, direction]
        or None if fetching fails.
    """
    logger.info(
        "Attempting regelleistung.net fetch for %s [%s, %s)",
        product, start, end,
    )
    try:
        resp = requests.get(
            REGELLEISTUNG_BASE_URL,
            params={
                "productType": product,
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d"),
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(
            "Regelleistung.net auto-fetch failed for %s: %s. "
            "Please download manually from %s",
            product, exc, REGELLEISTUNG_BASE_URL,
        )
        return None

    if not isinstance(data, list) or not data:
        logger.warning("Regelleistung.net returned no data for %s", product)
        return None

    df = pd.DataFrame(data)
    # Standardise column names (best-effort)
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "date" in cl or "delivery" in cl:
            col_map[col] = "date"
        elif "price" in cl and "capacity" in cl:
            col_map[col] = "capacity_price_eur_mw"
        elif "product" in cl:
            col_map[col] = "product"
        elif "direction" in cl:
            col_map[col] = "direction"
    df = df.rename(columns=col_map)

    for required in ["date", "capacity_price_eur_mw"]:
        if required not in df.columns:
            logger.warning(
                "Regelleistung.net response missing '%s' column", required,
            )
            return None

    df["product"] = df.get("product", product)
    df["direction"] = df.get("direction", "symmetric" if product == "FCR" else "")
    df["time_block"] = df.get("time_block", "")
    df["capacity_price_eur_mw"] = pd.to_numeric(
        df["capacity_price_eur_mw"], errors="coerce",
    )
    return df[["date", "product", "time_block", "capacity_price_eur_mw", "direction"]]


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
    current = start.normalize()
    end_date = end.normalize()

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        logger.info("Fetching Elexon system prices for %s", date_str)
        try:
            resp = requests.get(
                ELEXON_SYSTEM_PRICES_ENDPOINT,
                params={"settlementDate": date_str, "format": "json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            records = data if isinstance(data, list) else data.get("data", [])
            all_records.extend(records)
        except Exception as exc:
            logger.warning("Elexon system prices failed for %s: %s", date_str, exc)

        current += pd.Timedelta(days=1)
        time.sleep(0.5)

    if not all_records:
        return pd.DataFrame(columns=[
            "timestamp", "system_buy_price_gbp", "system_sell_price_gbp",
            "system_buy_price_eur", "system_sell_price_eur", "spread_eur",
        ])

    df = pd.DataFrame(all_records)

    # Build timestamp from settlement date + period
    if "settlementDate" in df.columns and "settlementPeriod" in df.columns:
        df["minutes"] = (df["settlementPeriod"].astype(int) - 1) * 30
        df["timestamp"] = pd.to_datetime(df["settlementDate"]) + pd.to_timedelta(
            df["minutes"], unit="m",
        )
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
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
    out["system_buy_price_eur"] = out["system_buy_price_gbp"] * GBP_TO_EUR
    out["system_sell_price_eur"] = out["system_sell_price_gbp"] * GBP_TO_EUR
    out["spread_eur"] = out["system_buy_price_eur"] - out["system_sell_price_eur"]
    out = out.set_index("timestamp").sort_index()
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
    except Exception as exc:
        logger.warning("Cannot create ENTSO-E client: %s", exc)
        return None

    start_q = start.tz_convert(DEFAULT_QUERY_TIMEZONE)
    end_q = end.tz_convert(DEFAULT_QUERY_TIMEZONE)

    logger.info("Fetching ENTSO-E imbalance prices for %s", zone)
    try:
        raw = client.query_imbalance_prices(zone, start=start_q, end=end_q)
    except Exception as exc:
        logger.warning("ENTSO-E imbalance data unavailable for %s: %s", zone, exc)
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
