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
