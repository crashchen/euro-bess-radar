"""Tests for the screening DA-price forecast (9.2b Stage-0 prep)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.da_forecast import DA_FORECAST_COL, build_da_price_forecast
from src.ida_forecast import FORECAST_COL as IDA_FORECAST_COL
from src.ida_forecast import build_ida_forecast


def _da_history(days: int = 4) -> pd.DataFrame:
    idx = pd.date_range("2025-03-01", periods=24 * days, freq="h", tz="UTC")
    shape = np.array(
        [20, 18, 16, 15, 15, 18, 30, 55, 70, 60, 45, 40,
         38, 42, 50, 65, 85, 95, 80, 60, 45, 35, 28, 24],
        dtype=float,
    )
    df = pd.DataFrame({"price_eur_mwh": np.tile(shape, days)}, index=idx)
    df.index.name = "timestamp"
    return df


def test_defaults_to_walk_forward_and_relabels() -> None:
    # Default must be walk-forward: this helper feeds the real-time reserve
    # commitment, so it must not peek at the target day or any later day.
    df = _da_history(3)
    dates = sorted(set(pd.DatetimeIndex(df.index).date))
    fc, meta = build_da_price_forecast(df, target_dates=dates)
    assert meta["forecast_mode"] == "walk_forward"
    assert DA_FORECAST_COL in fc.columns
    assert IDA_FORECAST_COL not in fc.columns
    # Walk-forward drops the first day (no prior history): 2 of 3 days remain.
    forecast_dates = set(pd.DatetimeIndex(fc.index).tz_convert("UTC").date)
    assert dates[0] not in forecast_dates
    assert len(forecast_dates) == 2


def test_values_match_ida_machinery_under_same_mode() -> None:
    df = _da_history(4)
    dates = sorted(set(pd.DatetimeIndex(df.index).date))
    fc, _ = build_da_price_forecast(df, target_dates=dates, tz="Europe/Berlin")
    ref, _ = build_ida_forecast(
        df, target_dates=dates, tz="Europe/Berlin", value_col="price_eur_mwh",
        forecast_mode="walk_forward",
    )
    assert np.allclose(
        fc[DA_FORECAST_COL].to_numpy(), ref[IDA_FORECAST_COL].to_numpy(),
    )


def test_explicit_loo_excludes_target_day_own_prices() -> None:
    # loo is still available explicitly (skill/backtest) and must exclude the
    # target day's own realised prices.
    df = _da_history(3)
    idx = pd.DatetimeIndex(df.index)
    day0 = idx.normalize() == pd.Timestamp("2025-03-01", tz="UTC")
    df.loc[day0, "price_eur_mwh"] += 500.0
    dates = sorted(set(idx.date))
    fc, meta = build_da_price_forecast(df, target_dates=dates, forecast_mode="loo")
    assert meta["forecast_mode"] == "loo"
    day0_fc = fc[pd.DatetimeIndex(fc.index).normalize() == pd.Timestamp(
        "2025-03-01", tz="UTC",
    )]
    # loo forecasts day 0 from the OTHER days -> stays near the un-inflated level.
    assert day0_fc[DA_FORECAST_COL].max() < 200.0


def test_empty_history_returns_empty_forecast() -> None:
    fc, meta = build_da_price_forecast(
        pd.DataFrame(), target_dates=[pd.Timestamp("2025-03-01").date()],
    )
    assert fc.empty
    assert meta["coverage"] == 0.0


def test_invalid_forecast_mode_raises() -> None:
    df = _da_history(2)
    dates = sorted(set(pd.DatetimeIndex(df.index).date))
    with pytest.raises(ValueError, match="forecast_mode"):
        build_da_price_forecast(df, target_dates=dates, forecast_mode="bogus")
