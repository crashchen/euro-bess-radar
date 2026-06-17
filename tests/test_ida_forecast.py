"""Tests for the hour-of-week/day IDA forecast generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ida_forecast import FORECAST_COL, build_ida_forecast
from src.simulation import available_local_dates


def _make_history(days: int = 5, *, anomaly_day: int | None = None) -> pd.DataFrame:
    """`days` of hourly IDA with a fixed daily shape; one day optionally
    inverted to act as an out-of-climatology anomaly."""
    shape = np.array(
        [10, 10, 10, 10, 10, 12, 20, 45, 60, 50, 30, 25,
         22, 28, 45, 72, 92, 80, 55, 40, 30, 20, 15, 12],
        dtype=float,
    )
    idx = pd.date_range("2026-03-16", periods=24 * days, freq="h", tz="UTC")
    values = np.tile(shape, days)
    if anomaly_day is not None:
        start = anomaly_day * 24
        values[start:start + 24] = shape[::-1]
    df = pd.DataFrame({"intraday_price_eur_mwh": values}, index=idx)
    df.index.name = "timestamp"
    return df


def test_hour_of_day_full_coverage_on_short_window() -> None:
    df = _make_history(days=4)
    dates = available_local_dates(df, tz="UTC")
    fc, meta = build_ida_forecast(df, target_dates=dates, tz="UTC")
    assert meta["coverage"] == pytest.approx(1.0)
    assert meta["bucket"] == "hour_of_day"
    assert meta["n_buckets_requested"] == 24
    assert meta["n_buckets_filled"] == 24
    assert meta["fallback_points"] == 0
    assert len(fc) == 24 * len(dates)
    assert list(fc.columns) == [FORECAST_COL, "bucket", "n_samples"]


def test_forecast_index_aligns_to_realised_timestamps() -> None:
    df = _make_history(days=3)
    dates = available_local_dates(df, tz="UTC")
    fc, _ = build_ida_forecast(df, target_dates=dates, tz="UTC")
    assert fc.index.equals(df.index)


def test_leave_one_day_out_excludes_target_day_anomaly() -> None:
    # Day index 2 is inverted. With leave-one-day-out its own forecast must
    # NOT reflect the inversion (it is built from the other, normal days),
    # so the forecast keeps the normal shape.
    df = _make_history(days=5, anomaly_day=2)
    dates = available_local_dates(df, tz="UTC")
    anomaly_date = dates[2]

    loo, _ = build_ida_forecast(df, target_dates=[anomaly_date], tz="UTC")
    realised_day = df.loc[df.index.normalize() == pd.Timestamp(anomaly_date, tz="UTC")]
    # Morning (hour 6-9) is cheap in the normal shape but expensive in the
    # inverted realised day — the LOO forecast should track the normal shape.
    fc_morning = loo[FORECAST_COL].to_numpy()[6:10].mean()
    realised_morning = realised_day["intraday_price_eur_mwh"].to_numpy()[6:10].mean()
    assert fc_morning < realised_morning - 5.0


def test_leave_one_day_in_uses_target_day() -> None:
    df = _make_history(days=5, anomaly_day=2)
    dates = available_local_dates(df, tz="UTC")
    anomaly_date = dates[2]
    with_self, _ = build_ida_forecast(
        df, target_dates=[anomaly_date], tz="UTC", leave_one_day_out=False,
    )
    without_self, _ = build_ida_forecast(
        df, target_dates=[anomaly_date], tz="UTC", leave_one_day_out=True,
    )
    # Including the target day pulls the climatology toward its own values,
    # so the two forecasts must differ on the anomalous day.
    assert not np.allclose(
        with_self[FORECAST_COL].to_numpy(),
        without_self[FORECAST_COL].to_numpy(),
    )


def test_hour_of_week_degrades_on_sub_week_window() -> None:
    # 3 distinct weekdays, one sample each -> leave-one-day-out empties every
    # hour-of-week bucket and everything falls back to the global mean.
    df = _make_history(days=3)
    dates = available_local_dates(df, tz="UTC")
    _, meta = build_ida_forecast(
        df, target_dates=dates, tz="UTC", bucket="hour_of_week",
    )
    assert meta["coverage"] == pytest.approx(0.0)
    assert meta["fallback_points"] == meta["n_target_points"]
    # All forecast points are global-mean fallbacks, so no bucket is
    # actually backed by history even though many buckets appear.
    assert meta["n_buckets_filled"] == 0
    assert meta["n_buckets_requested"] > 0


def test_empty_or_missing_history_returns_empty_frame() -> None:
    empty = pd.DataFrame(columns=["intraday_price_eur_mwh"])
    fc, meta = build_ida_forecast(empty, target_dates=[], tz="UTC")
    assert fc.empty
    assert meta["coverage"] == 0.0
    assert meta["n_target_points"] == 0

    df = _make_history(days=2)
    fc2, meta2 = build_ida_forecast(df, target_dates=[], tz="UTC")
    assert fc2.empty
    assert meta2["n_target_points"] == 0


def test_single_loaded_day_has_no_out_of_sample_climatology() -> None:
    df = _make_history(days=1)
    dates = available_local_dates(df, tz="UTC")
    fc, meta = build_ida_forecast(df, target_dates=dates, tz="UTC")
    # Only one day loaded -> leave-one-day-out leaves no climatology.
    assert fc.empty
    assert meta["n_target_points"] == 0


def test_invalid_bucket_raises() -> None:
    df = _make_history(days=2)
    dates = available_local_dates(df, tz="UTC")
    with pytest.raises(ValueError, match="bucket must be one of"):
        build_ida_forecast(df, target_dates=dates, tz="UTC", bucket="nope")
