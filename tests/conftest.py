"""Shared test fixtures for eu-bess-pulse tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def mock_entsoe_series() -> pd.Series:
    """Synthetic ENTSO-E day-ahead price series (24h, hourly, UTC)."""
    idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    prices = [50.0 + (i % 24) * 2.0 for i in range(24)]
    return pd.Series(prices, index=idx, name="price")


@pytest.fixture
def mock_elexon_json() -> list[dict]:
    """Synthetic Elexon market index JSON — two providers per period.

    Mirrors real Elexon data: each settlement period has one record with a
    real price and a second record with price=0 from a different provider.
    48 settlement periods → 96 raw records → 48 rows after dedup.
    """
    base = pd.Timestamp("2025-01-01", tz="UTC")
    records = []
    for i in range(48):
        ts = base + pd.Timedelta(minutes=30 * i)
        real_price = 40.0 + (i % 24) * 1.5
        # Provider with the real price
        records.append({
            "startTime": ts.isoformat(),
            "settlementDate": "2025-01-01",
            "settlementPeriod": i + 1,
            "dataProvider": "APXMIDP",
            "price": real_price,
            "volume": 100.0,
        })
        # Provider reporting zero
        records.append({
            "startTime": ts.isoformat(),
            "settlementDate": "2025-01-01",
            "settlementPeriod": i + 1,
            "dataProvider": "N2EXMIDP",
            "price": 0.0,
            "volume": 0.0,
        })
    return records


@pytest.fixture
def mock_price_df() -> pd.DataFrame:
    """Cleaned price DataFrame for cache roundtrip tests."""
    idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"price_eur_mwh": [50.0 + i for i in range(24)]}, index=idx)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def dst_spring_forward_prices() -> pd.DataFrame:
    """Hourly UTC prices spanning the 2025-03-30 Europe/Berlin spring-forward.

    Local time skips from 02:00 to 03:00 CET → CEST, so any analytics that
    groups by local-day must handle a short day. Fixture covers 2025-03-29
    00:00 UTC through 2025-03-31 00:00 UTC (48h UTC, 47h local).
    """
    idx = pd.date_range(
        "2025-03-29T00:00:00Z", "2025-03-31T00:00:00Z",
        freq="h", inclusive="left", tz="UTC",
    )
    prices = [40.0 + (i % 24) * 1.5 for i in range(len(idx))]
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def dst_fall_back_prices() -> pd.DataFrame:
    """Hourly UTC prices spanning the 2025-10-26 Europe/Berlin fall-back.

    Local time repeats 02:00 to 03:00 CEST → CET, so analytics that groups
    by local-day must handle a long day (25 local hours) and a tz-ambiguous
    timestamp.
    """
    idx = pd.date_range(
        "2025-10-25T00:00:00Z", "2025-10-27T00:00:00Z",
        freq="h", inclusive="left", tz="UTC",
    )
    prices = [50.0 + (i % 24) * 1.2 for i in range(len(idx))]
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def de_lu_15min_prices() -> pd.DataFrame:
    """DE_LU 15-minute resolution prices (96 intervals/day, post-Oct 2025)."""
    idx = pd.date_range(
        "2025-11-01T00:00:00Z", periods=96, freq="15min", tz="UTC",
    )
    prices = [60.0 + (i % 96) * 0.5 for i in range(96)]
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def sparse_hourly_prices() -> pd.DataFrame:
    """Hourly data with a real internal gap (one missing timestamp).

    Used to exercise the path where clean_prices / analytics must surface
    a missing interval without forward-fill masking it.
    """
    idx = pd.DatetimeIndex(
        [
            "2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z",
            # 02:00 missing
            "2025-01-01T03:00:00Z", "2025-01-01T04:00:00Z",
            "2025-01-01T05:00:00Z",
        ],
        name="timestamp",
    )
    df = pd.DataFrame({"price_eur_mwh": [10.0, 11.0, 13.0, 14.0, 15.0]}, index=idx)
    return df
