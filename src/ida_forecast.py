"""Hourly IDA price forecasting for the sequential DA+ID policy.

The sequential dispatch policy (`dispatch.solve_sequential_da_id_dispatch`)
needs an IDA price *forecast* — what the desk believes IDA will print when
it commits its rebid — separate from the realised IDA price used for
settlement. This module builds a screening-grade forecast from an
hourly climatology of the loaded IDA history.

Forecast information modes: the default leave-one-day-out mode excludes the
target day's own realised prices but may use later days in the loaded sample;
walk-forward mode uses only days strictly before the target; in-sample mode
is diagnostic only. This keeps the cockpit explicit about whether it is
showing forecast-skill backtesting or a stricter prior-information view.

The forecast is intentionally simple: a climatology mean bucketed by
hour only (hour-of-day or hour-of-week). It does NOT distinguish the
minute-of-hour of 15-min data, and is NOT a trading-grade IDA price
model — it exists only to demonstrate the *gap* between a forecast-driven
screening policy and the perfect-foresight ceiling. Surface it in the UI as
"climatology forecast, hourly bucketed".
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


SKILL_BY_HOUR_COLUMNS = ["hour", "mae", "n"]


def compute_forecast_skill(
    forecast_df: pd.DataFrame,
    realised: pd.DataFrame,
    *,
    value_col: str = IDA_VALUE_COL,
    da_prices: pd.DataFrame | None = None,
    tz: str | None = None,
) -> dict:
    """Score a climatology IDA forecast against the realised IDA prints.

    Aligns ``forecast_df[FORECAST_COL]`` with the realised IDA series on the
    timestamp index and reports price-space error metrics, so a user can see
    how trustworthy the forecast is *before* reading the revenue panel or
    setting a rebid deadband. When ``da_prices`` is given, it also scores the
    forecast against the DA-as-IDA naive baseline (the "no rebid signal"
    null hypothesis): ``skill_vs_da = 1 - MAE_forecast / MAE_da`` (positive ⇒
    the climatology beats just assuming IDA == DA).

    Args:
        forecast_df: Output of :func:`build_ida_forecast` (UTC-indexed).
        realised: UTC-indexed realised IDA frame with ``value_col``.
        value_col: Realised IDA price column.
        da_prices: Optional UTC-indexed DA frame with ``price_eur_mwh`` for
            the naive-baseline skill score.
        tz: Local timezone for the hour-of-day error profile.

    Returns:
        Dict with ``n_points``, ``mae``, ``bias`` (mean signed
        forecast - realised; >0 means the forecast prints high), ``rmse``,
        ``realised_std`` (context for the MAE), ``naive_da_mae`` /
        ``skill_vs_da`` (None without DA or when the naive MAE is ~0), and a
        ``by_hour`` DataFrame (local hour-of-day MAE + sample count).
    """
    empty = {
        "n_points": 0, "mae": float("nan"), "bias": float("nan"),
        "rmse": float("nan"), "realised_std": float("nan"),
        "naive_da_mae": None, "skill_vs_da": None,
        "by_hour": pd.DataFrame(columns=SKILL_BY_HOUR_COLUMNS),
    }
    if (
        forecast_df is None or forecast_df.empty
        or realised is None or realised.empty
        or FORECAST_COL not in forecast_df.columns
        or value_col not in realised.columns
    ):
        return empty

    aligned = forecast_df[[FORECAST_COL]].join(realised[[value_col]], how="inner")
    if da_prices is not None and "price_eur_mwh" in getattr(da_prices, "columns", []):
        aligned = aligned.join(
            da_prices[["price_eur_mwh"]].rename(columns={"price_eur_mwh": "_da"}),
            how="left",
        )
    aligned = aligned.dropna(subset=[FORECAST_COL, value_col])
    if aligned.empty:
        return empty

    err = aligned[FORECAST_COL] - aligned[value_col]
    abs_err = err.abs()
    naive_da_mae = None
    skill_vs_da = None
    if "_da" in aligned.columns:
        da_rows = aligned.dropna(subset=["_da"])
        if not da_rows.empty:
            naive_da_mae = float((da_rows["_da"] - da_rows[value_col]).abs().mean())
            fc_mae_sub = float(
                (da_rows[FORECAST_COL] - da_rows[value_col]).abs().mean()
            )
            if naive_da_mae > 1e-9:
                skill_vs_da = 1.0 - fc_mae_sub / naive_da_mae

    utc_index = pd.DatetimeIndex(aligned.index)
    local_index = utc_index.tz_convert(tz) if tz else utc_index
    by_hour = (
        pd.DataFrame({"hour": local_index.hour, "abs_err": abs_err.to_numpy()})
        .groupby("hour")["abs_err"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "mae", "size": "n"})
    )
    return {
        "n_points": len(aligned),
        "mae": float(abs_err.mean()),
        "bias": float(err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "realised_std": float(aligned[value_col].std(ddof=0)),
        "naive_da_mae": naive_da_mae,
        "skill_vs_da": skill_vs_da,
        "by_hour": by_hour[SKILL_BY_HOUR_COLUMNS],
    }
