"""Tests for the hour-of-week/day IDA forecast generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ida_forecast import (
    FORECAST_COL,
    build_ida_forecast,
    compute_forecast_skill,
)
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


def test_in_sample_mode_uses_target_day() -> None:
    df = _make_history(days=5, anomaly_day=2)
    dates = available_local_dates(df, tz="UTC")
    anomaly_date = dates[2]
    with_self, _ = build_ida_forecast(
        df, target_dates=[anomaly_date], tz="UTC", forecast_mode="in_sample",
    )
    without_self, _ = build_ida_forecast(
        df, target_dates=[anomaly_date], tz="UTC", forecast_mode="loo",
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


def test_invalid_forecast_mode_raises() -> None:
    df = _make_history(days=2)
    dates = available_local_dates(df, tz="UTC")
    with pytest.raises(ValueError, match="forecast_mode must be one of"):
        build_ida_forecast(df, target_dates=dates, tz="UTC", forecast_mode="nope")


def test_walk_forward_drops_first_day_and_uses_only_prior_days() -> None:
    df = _make_history(days=4)
    dates = available_local_dates(df, tz="UTC")
    fc, meta = build_ida_forecast(
        df, target_dates=dates, tz="UTC", forecast_mode="walk_forward",
    )
    assert meta["forecast_mode"] == "walk_forward"
    # The earliest target day has no prior history and is dropped; the
    # remaining 3 days are forecast from strictly-earlier days only.
    forecast_dates = {ts.tz_convert("UTC").date() for ts in fc.index}
    assert dates[0] not in forecast_dates
    assert len(forecast_dates) == 3


# ── Forecast skill scoring ──────────────────────────────────────────────────

def _make_realised_and_da(days: int = 2):
    """Realised IDA (one shape) + DA prices (a DIFFERENT shape) over `days`."""
    ida_shape = np.array(
        [10, 10, 10, 10, 10, 12, 20, 45, 60, 50, 30, 25,
         22, 28, 45, 72, 92, 80, 55, 40, 30, 20, 15, 12],
        dtype=float,
    )
    da_shape = ida_shape[::-1].copy()
    idx = pd.date_range("2026-03-16", periods=24 * days, freq="h", tz="UTC")
    realised = pd.DataFrame(
        {"intraday_price_eur_mwh": np.tile(ida_shape, days)}, index=idx,
    )
    da = pd.DataFrame({"price_eur_mwh": np.tile(da_shape, days)}, index=idx)
    realised.index.name = da.index.name = "timestamp"
    return realised, da


def test_skill_perfect_forecast_has_zero_error_and_full_skill() -> None:
    realised, da = _make_realised_and_da(days=2)
    forecast = realised.rename(columns={"intraday_price_eur_mwh": FORECAST_COL})
    skill = compute_forecast_skill(forecast, realised, da_prices=da, tz="UTC")
    assert skill["n_points"] == 48
    assert skill["mae"] == pytest.approx(0.0, abs=1e-9)
    assert skill["bias"] == pytest.approx(0.0, abs=1e-9)
    assert skill["rmse"] == pytest.approx(0.0, abs=1e-9)
    # A perfect forecast beats the (non-trivial) DA-as-IDA baseline fully.
    assert skill["naive_da_mae"] > 0.0
    assert skill["skill_vs_da"] == pytest.approx(1.0, abs=1e-9)


def test_skill_detects_constant_bias() -> None:
    realised, _ = _make_realised_and_da(days=1)
    forecast = (realised + 5.0).rename(
        columns={"intraday_price_eur_mwh": FORECAST_COL},
    )
    skill = compute_forecast_skill(forecast, realised, tz="UTC")
    assert skill["mae"] == pytest.approx(5.0, abs=1e-9)
    assert skill["bias"] == pytest.approx(5.0, abs=1e-9)  # forecast prints high
    assert skill["naive_da_mae"] is None  # no DA passed
    assert skill["skill_vs_da"] is None


def test_skill_zero_when_forecast_equals_da_baseline() -> None:
    realised, da = _make_realised_and_da(days=2)
    # Forecast == DA -> same error as the naive baseline -> zero skill.
    forecast = da.rename(columns={"price_eur_mwh": FORECAST_COL})
    skill = compute_forecast_skill(forecast, realised, da_prices=da, tz="UTC")
    assert skill["skill_vs_da"] == pytest.approx(0.0, abs=1e-9)


def test_skill_negative_when_forecast_worse_than_da() -> None:
    realised, da = _make_realised_and_da(days=2)
    # A forecast further from realised than DA is -> negative skill.
    worse = (2.0 * da["price_eur_mwh"] - realised["intraday_price_eur_mwh"])
    forecast = pd.DataFrame({FORECAST_COL: worse}, index=realised.index)
    skill = compute_forecast_skill(forecast, realised, da_prices=da, tz="UTC")
    assert skill["skill_vs_da"] < 0.0


def test_skill_by_hour_profile_has_one_row_per_local_hour() -> None:
    realised, _ = _make_realised_and_da(days=2)
    forecast = (realised + 3.0).rename(
        columns={"intraday_price_eur_mwh": FORECAST_COL},
    )
    skill = compute_forecast_skill(forecast, realised, tz="UTC")
    by_hour = skill["by_hour"]
    assert list(by_hour.columns) == ["hour", "mae", "n"]
    assert len(by_hour) == 24
    assert (by_hour["n"] == 2).all()  # 2 days -> 2 samples per hour
    assert np.allclose(by_hour["mae"], 3.0)


def test_skill_empty_inputs_return_empty_metrics() -> None:
    realised, _ = _make_realised_and_da(days=1)
    empty_fc = pd.DataFrame(columns=[FORECAST_COL])
    skill = compute_forecast_skill(empty_fc, realised, tz="UTC")
    assert skill["n_points"] == 0
    assert skill["by_hour"].empty
