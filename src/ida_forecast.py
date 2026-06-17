"""Hour-of-week IDA price forecasting for the sequential DA+ID policy.

The sequential dispatch policy (`dispatch.solve_sequential_da_id_dispatch`)
needs an IDA price *forecast* — what the desk believes IDA will print when
it commits its rebid — separate from the realised IDA price used for
settlement. This module builds a screening-grade forecast from a
hour-of-week climatology of the loaded IDA history.

Anti-leakage: when forecasting a given local day, that day's own realised
prices are excluded from the climatology (leave-one-day-out), so the
forecast for day D never sees day D. This keeps the cockpit's
forecast-error numbers honest as a backtest rather than an in-sample fit.

The forecast is intentionally simple: a climatology mean bucketed by
hour only (hour-of-day or hour-of-week). It does NOT distinguish the
minute-of-hour of 15-min data, and is NOT a trading-grade IDA price
model — it exists only to demonstrate the *gap* between a realistic
forecast-driven policy and the perfect-foresight ceiling. Surface it in
the UI as "climatology forecast, hourly bucketed".
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.analytics import _to_local

IDA_VALUE_COL = "intraday_price_eur_mwh"
FORECAST_COL = "forecast_ida_eur_mwh"
_VALID_BUCKETS = ("hour_of_day", "hour_of_week")
_VALID_FORECAST_MODES = ("loo", "walk_forward", "in_sample")


def _bucket_of(local_index: pd.DatetimeIndex, bucket: str) -> np.ndarray:
    """Map a local DatetimeIndex to its climatology bucket.

    ``hour_of_day`` (0..23) is robust on short windows because every day
    contributes to the same 24 buckets. ``hour_of_week`` (0..167) captures
    weekday vs weekend structure but needs several weeks of history before
    leave-one-day-out leaves enough same-weekday samples per bucket.
    """
    if bucket == "hour_of_week":
        return local_index.weekday * 24 + local_index.hour
    return np.asarray(local_index.hour)


def build_ida_forecast(
    ida_history: pd.DataFrame,
    *,
    target_dates: list[date],
    tz: str | None = None,
    value_col: str = IDA_VALUE_COL,
    bucket: str = "hour_of_day",
    forecast_mode: str = "loo",
) -> tuple[pd.DataFrame, dict]:
    """Forecast IDA prices at each target day's own timestamps.

    For every local date in ``target_dates`` the forecast at each interval
    is the mean realised price of the history rows sharing that interval's
    climatology bucket (``hour_of_day`` by default, or ``hour_of_week``).
    ``forecast_mode`` controls which history a day is allowed to see:

    - ``"loo"`` (default): leave-one-day-out cross-validation — every day
      EXCEPT the target day. This is NOT walk-forward: forecasting Jan 1
      may use Jan 2/Jan 3 from the loaded sample. It is an unbiased
      screening estimate of forecast skill, not what a desk could have
      known in real time.
    - ``"walk_forward"``: only days strictly BEFORE the target day, so it
      reflects information available at commit time (the first day(s) have
      no history and are dropped).
    - ``"in_sample"``: all days including the target day (optimistic; for
      diagnostics only).

    Empty buckets (no historical sample after exclusion) fall back to the
    climatology's global mean, and those points are flagged with
    ``n_samples == 0`` so callers can judge forecast reliability.

    Args:
        ida_history: UTC-indexed frame with ``value_col``. Typically the
            full loaded IDA1 series for the zone.
        target_dates: Local calendar dates to produce a forecast for.
        tz: IANA timezone for the bucket mapping (the zone's local time).
            None groups on the index as-is (UTC).
        value_col: Realised IDA price column.
        bucket: ``"hour_of_day"`` (robust on short windows) or
            ``"hour_of_week"`` (weekday-aware, needs several weeks).
        forecast_mode: ``"loo"`` (default), ``"walk_forward"``, or
            ``"in_sample"`` (see above).

    Returns:
        ``(forecast_df, metadata)``. ``forecast_df`` is UTC-indexed with
        columns ``[FORECAST_COL, "bucket", "n_samples"]`` covering only the
        timestamps present for the target dates. ``metadata`` has
        ``coverage`` (fraction of target points backed by >=1 sample),
        ``n_target_points``, ``n_buckets_filled``, ``n_buckets_requested``,
        ``global_mean``, ``bucket``, ``forecast_mode``, and
        ``fallback_points``.
    """
    if bucket not in _VALID_BUCKETS:
        raise ValueError(f"bucket must be one of {_VALID_BUCKETS}, got {bucket!r}")
    if forecast_mode not in _VALID_FORECAST_MODES:
        raise ValueError(
            f"forecast_mode must be one of {_VALID_FORECAST_MODES}, got {forecast_mode!r}"
        )
    empty_meta = {
        "coverage": 0.0,
        "n_target_points": 0,
        "n_buckets_requested": 0,
        "n_buckets_filled": 0,
        "global_mean": float("nan"),
        "bucket": bucket,
        "forecast_mode": forecast_mode,
        "fallback_points": 0,
    }
    if ida_history is None or ida_history.empty or value_col not in ida_history.columns:
        return _empty_forecast_frame(), empty_meta
    if not target_dates:
        return _empty_forecast_frame(), empty_meta

    local = _to_local(ida_history[[value_col]].dropna(), tz).sort_index()
    if local.empty:
        return _empty_forecast_frame(), empty_meta

    local_index = pd.DatetimeIndex(local.index)
    local = local.assign(
        _bucket=_bucket_of(local_index, bucket),
        _date=local_index.date,
    )
    target_set = set(target_dates)

    frames: list[pd.DataFrame] = []
    for target in sorted(target_set):
        day_rows = local[local["_date"] == target]
        if day_rows.empty:
            continue
        if forecast_mode == "walk_forward":
            clim = local[local["_date"] < target]
        elif forecast_mode == "in_sample":
            clim = local
        else:  # "loo"
            clim = local[local["_date"] != target]
        if clim.empty:
            # No usable climatology (only the target day, or no prior days
            # in walk-forward mode) — skip; the day is excluded downstream.
            continue
        bucket_mean = clim.groupby("_bucket")[value_col].mean()
        bucket_count = clim.groupby("_bucket")[value_col].size()
        global_mean = float(clim[value_col].mean())

        buckets = day_rows["_bucket"].to_numpy()
        forecast_vals = bucket_mean.reindex(buckets).to_numpy()
        sample_counts = bucket_count.reindex(buckets).fillna(0).to_numpy(dtype=int)
        forecast_vals = np.where(np.isnan(forecast_vals), global_mean, forecast_vals)

        frame = pd.DataFrame(
            {
                FORECAST_COL: forecast_vals,
                "bucket": buckets,
                "n_samples": sample_counts,
            },
            index=pd.DatetimeIndex(day_rows.index).tz_convert("UTC"),
        )
        frames.append(frame)

    if not frames:
        return _empty_forecast_frame(), empty_meta

    forecast_df = pd.concat(frames).sort_index()
    forecast_df.index.name = "timestamp"
    n_target = len(forecast_df)
    fallback_points = int((forecast_df["n_samples"] == 0).sum())
    metadata = {
        "coverage": float((forecast_df["n_samples"] > 0).mean()),
        "n_target_points": n_target,
        # Distinct buckets appearing in the target window vs. those backed
        # by >=1 historical sample. Reporting both stops the UI from
        # implying a forecast is well-supported when every point is a
        # global-mean fallback (n_buckets_filled == 0).
        "n_buckets_requested": int(forecast_df["bucket"].nunique()),
        "n_buckets_filled": int(
            forecast_df.loc[forecast_df["n_samples"] > 0, "bucket"].nunique()
        ),
        "global_mean": float(forecast_df[FORECAST_COL].mean()),
        "bucket": bucket,
        "forecast_mode": forecast_mode,
        "fallback_points": fallback_points,
    }
    return forecast_df, metadata


def _empty_forecast_frame() -> pd.DataFrame:
    out = pd.DataFrame(columns=[FORECAST_COL, "bucket", "n_samples"])
    out.index = pd.DatetimeIndex([], name="timestamp")
    return out
