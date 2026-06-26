"""Block-of-day climatology skill for reserve capacity prices (Phase 9.2b prep).

Phase 9.2b (forecast-driven realistic triple-joint) will commit reserve
capacity under a price *forecast* BEFORE the DA auction — reserve auctions
clear on D-1 morning (FCR ~08:00, aFRR ~09:00 CET), ahead of the noon DA gate.
Before building that solver, this module answers the prerequisite question:
*are reserve capacity prices forecastable at all?* It scores a block-of-day
climatology (the native 4h FCR/aFRR product granularity) against realised
prices, with a flat sample-mean naive baseline.

This is a skill DIAGNOSTIC, not a dispatch model. Reserve activation energy is
never modelled in this project (capacity headroom only). The leave-one-day-out
default is an unbiased skill estimate, NOT what a desk knew in real time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import _to_local

RESERVE_VALUE_COL = "capacity_price_eur_mw"
RESERVE_FORECAST_COL = "forecast_capacity_eur_mw"
RESERVE_BLOCK_HOURS = 4
N_BLOCKS_PER_DAY = 24 // RESERVE_BLOCK_HOURS  # 6
SKILL_BY_BLOCK_COLUMNS = ["block", "mae", "n"]
_VALID_FORECAST_MODES = ("loo", "walk_forward", "in_sample")


def _block_of_day(local_index: pd.DatetimeIndex) -> np.ndarray:
    """4h block index (0..5) for each local timestamp."""
    return np.asarray(local_index.hour) // RESERVE_BLOCK_HOURS


def _empty_skill(forecast_mode: str) -> dict:
    return {
        "n_points": 0,
        "mae": float("nan"),
        "bias": float("nan"),
        "rmse": float("nan"),
        "realised_std": float("nan"),
        "naive_mean_mae": None,
        "skill_vs_mean": None,
        "coverage": 0.0,
        "n_blocks_filled": 0,
        "n_blocks_requested": 0,
        "forecast_mode": forecast_mode,
        "by_block": pd.DataFrame(columns=SKILL_BY_BLOCK_COLUMNS),
    }


def compute_reserve_forecast_skill(
    history: pd.DataFrame,
    *,
    tz: str | None = None,
    value_col: str = RESERVE_VALUE_COL,
    forecast_mode: str = "loo",
) -> dict:
    """Score a block-of-day climatology against realised reserve prices.

    For each local day the forecast at each interval is the mean realised
    capacity price of the history rows sharing that interval's 4h block.
    ``forecast_mode`` controls which history a day may see:

    - ``"loo"`` (default): leave-one-day-out — every day EXCEPT the target.
      Unbiased skill estimate, NOT walk-forward (may use later days).
    - ``"walk_forward"``: only days strictly before the target (drops the
      earliest day, reflects information available at commit time).
    - ``"in_sample"``: all days including the target (diagnostics only).

    The naive baseline is the flat sample mean of the same climatology days
    (there is no DA-as-IDA analog for reserve): ``skill_vs_mean =
    1 - MAE_climatology / MAE_flat`` (>0 ⇒ the block climatology beats a flat
    mean, i.e. reserve prices carry block-of-day structure worth forecasting;
    ≤0 ⇒ a forecast-driven reserve commitment rests on thin ice).

    Args:
        history: UTC-indexed frame with ``value_col`` (one reserve product's
            capacity price series, EUR/MW/h).
        tz: IANA timezone for block-of-day bucketing (None groups on UTC).
        value_col: Realised capacity-price column.
        forecast_mode: ``"loo"`` (default), ``"walk_forward"``, ``"in_sample"``.

    Returns:
        Dict with ``n_points``, ``mae``, ``bias`` (signed forecast - realised,
        >0 ⇒ forecast prints high), ``rmse``, ``realised_std``,
        ``naive_mean_mae`` / ``skill_vs_mean`` (None when the naive MAE is ~0),
        ``coverage`` (fraction of points backed by >=1 historical sample),
        ``n_blocks_filled`` / ``n_blocks_requested``, ``forecast_mode``, and a
        ``by_block`` DataFrame (per-block MAE + sample count).
    """
    if forecast_mode not in _VALID_FORECAST_MODES:
        raise ValueError(
            f"forecast_mode must be one of {_VALID_FORECAST_MODES}, got {forecast_mode!r}"
        )
    empty = _empty_skill(forecast_mode)
    if history is None or history.empty or value_col not in history.columns:
        return empty

    local = _to_local(history[[value_col]].dropna(), tz).sort_index()
    if local.empty:
        return empty
    idx = pd.DatetimeIndex(local.index)
    local = local.assign(_block=_block_of_day(idx), _date=idx.date)

    # Pre-aggregate once. The original implementation re-filtered and
    # regrouped the full history for every target day; this keeps the same
    # LOO / walk-forward semantics while scaling with days x 6 native blocks.
    day_block = local.groupby(["_date", "_block"])[value_col].agg(["sum", "count"])
    dates = sorted(local["_date"].unique())
    block_index = pd.Index(range(N_BLOCKS_PER_DAY), name="_block")
    # fillna AFTER reindex: unstack leaves NaN for (date, block) pairs absent
    # from the history (a day missing an entire 4h block), and reindex's
    # fill_value only fills newly-added labels, not those pre-existing NaNs.
    day_block_sum = (
        day_block["sum"]
        .unstack("_block")
        .reindex(index=dates, columns=block_index)
        .fillna(0.0)
    )
    day_block_count = (
        day_block["count"]
        .unstack("_block")
        .reindex(index=dates, columns=block_index)
        .fillna(0)
        .astype(int)
    )
    total_block_sum = day_block_sum.sum(axis=0)
    total_block_count = day_block_count.sum(axis=0)
    total_sum = float(total_block_sum.sum())
    total_count = int(total_block_count.sum())
    cumulative_sum = pd.Series(0.0, index=block_index)
    cumulative_count = pd.Series(0, index=block_index, dtype=int)
    cumulative_total_sum = 0.0
    cumulative_total_count = 0

    frames: list[pd.DataFrame] = []
    for target, day in local.groupby("_date", sort=True):
        target_block_sum = day_block_sum.loc[target]
        target_block_count = day_block_count.loc[target]
        if forecast_mode == "walk_forward":
            block_sum = cumulative_sum
            block_count = cumulative_count
            global_sum = cumulative_total_sum
            global_count = cumulative_total_count
        elif forecast_mode == "in_sample":
            block_sum = total_block_sum
            block_count = total_block_count
            global_sum = total_sum
            global_count = total_count
        else:  # "loo"
            block_sum = total_block_sum - target_block_sum
            block_count = total_block_count - target_block_count
            global_sum = total_sum - float(target_block_sum.sum())
            global_count = total_count - int(target_block_count.sum())

        if global_count <= 0:
            if forecast_mode == "walk_forward":
                cumulative_sum = cumulative_sum + target_block_sum
                cumulative_count = cumulative_count + target_block_count
                cumulative_total_sum += float(target_block_sum.sum())
                cumulative_total_count += int(target_block_count.sum())
            continue
        global_mean = global_sum / global_count
        block_mean = block_sum / block_count.replace(0, np.nan)
        blocks = day["_block"].to_numpy()
        forecast = block_mean.reindex(blocks).to_numpy(dtype=float)
        counts = block_count.reindex(blocks).fillna(0).to_numpy(dtype=int)
        forecast = np.where(np.isnan(forecast), global_mean, forecast)
        frames.append(pd.DataFrame(
            {
                "forecast": forecast,
                "naive": global_mean,
                "realised": day[value_col].to_numpy(dtype=float),
                "block": blocks,
                "n_samples": counts,
            },
            index=pd.DatetimeIndex(day.index),
        ))
        if forecast_mode == "walk_forward":
            cumulative_sum = cumulative_sum + target_block_sum
            cumulative_count = cumulative_count + target_block_count
            cumulative_total_sum += float(target_block_sum.sum())
            cumulative_total_count += int(target_block_count.sum())

    if not frames:
        return empty

    aligned = pd.concat(frames).sort_index()
    err = aligned["forecast"] - aligned["realised"]
    abs_err = err.abs()
    naive_mae = float((aligned["naive"] - aligned["realised"]).abs().mean())
    mae = float(abs_err.mean())
    skill_vs_mean = (1.0 - mae / naive_mae) if naive_mae > 1e-9 else None

    by_block = (
        aligned.assign(abs_err=abs_err)
        .groupby("block")["abs_err"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "mae", "size": "n"})
    )
    return {
        "n_points": len(aligned),
        "mae": mae,
        "bias": float(err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "realised_std": float(aligned["realised"].std(ddof=0)),
        "naive_mean_mae": naive_mae,
        "skill_vs_mean": skill_vs_mean,
        "coverage": float((aligned["n_samples"] > 0).mean()),
        "n_blocks_filled": int(aligned.loc[aligned["n_samples"] > 0, "block"].nunique()),
        "n_blocks_requested": int(aligned["block"].nunique()),
        "forecast_mode": forecast_mode,
        "by_block": by_block[SKILL_BY_BLOCK_COLUMNS],
    }


def _empty_reserve_forecast_frame() -> pd.DataFrame:
    out = pd.DataFrame(columns=[RESERVE_FORECAST_COL, "block", "n_samples"])
    out.index = pd.DatetimeIndex([], name="timestamp")
    return out


def build_reserve_price_forecast(
    history: pd.DataFrame,
    *,
    target_dates: list,
    tz: str | None = None,
    value_col: str = RESERVE_VALUE_COL,
    forecast_mode: str = "walk_forward",
) -> tuple[pd.DataFrame, dict]:
    """Block-of-day climatology forecast of reserve prices at target intervals.

    For each local date in ``target_dates`` the forecast at each interval is the
    mean realised capacity price of the history rows sharing that interval's 4h
    block. Feeds the 9.2b Stage-0 reserve commitment, so it defaults to
    ``forecast_mode="walk_forward"`` (prior days only) — the commitment must not
    see the target day's NOR any later day's realised price. ``"loo"`` (may use
    future days) and ``"in_sample"`` are for diagnostics only. Empty blocks fall
    back to the climatology global mean, flagged ``n_samples == 0``.

    Returns ``(forecast_df, metadata)``. ``forecast_df`` is UTC-indexed with
    ``[RESERVE_FORECAST_COL, "block", "n_samples"]`` over the target dates'
    timestamps; ``metadata`` carries ``coverage`` / ``n_blocks_filled`` /
    ``n_blocks_requested`` / ``fallback_points`` / ``forecast_mode`` /
    ``global_mean`` / ``n_target_points``. Walk-forward's first day (no prior
    history) is simply absent from the frame — never silently in-sampled.
    """
    if forecast_mode not in _VALID_FORECAST_MODES:
        raise ValueError(
            f"forecast_mode must be one of {_VALID_FORECAST_MODES}, got {forecast_mode!r}"
        )
    empty_meta = {
        "coverage": 0.0, "n_target_points": 0, "n_blocks_requested": 0,
        "n_blocks_filled": 0, "global_mean": float("nan"),
        "forecast_mode": forecast_mode, "fallback_points": 0,
    }
    if history is None or history.empty or value_col not in history.columns:
        return _empty_reserve_forecast_frame(), empty_meta
    if not target_dates:
        return _empty_reserve_forecast_frame(), empty_meta

    local = _to_local(history[[value_col]].dropna(), tz).sort_index()
    if local.empty:
        return _empty_reserve_forecast_frame(), empty_meta
    idx = pd.DatetimeIndex(local.index)
    local = local.assign(_block=_block_of_day(idx), _date=idx.date)
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
            # walk-forward first day (no prior history) — dropped, not in-sampled.
            continue
        block_mean = clim.groupby("_block")[value_col].mean()
        block_count = clim.groupby("_block")[value_col].size()
        global_mean = float(clim[value_col].mean())

        blocks = day_rows["_block"].to_numpy()
        forecast = block_mean.reindex(blocks).to_numpy(dtype=float)
        counts = block_count.reindex(blocks).fillna(0).to_numpy(dtype=int)
        forecast = np.where(np.isnan(forecast), global_mean, forecast)
        frames.append(pd.DataFrame(
            {RESERVE_FORECAST_COL: forecast, "block": blocks, "n_samples": counts},
            index=pd.DatetimeIndex(day_rows.index).tz_convert("UTC"),
        ))

    if not frames:
        return _empty_reserve_forecast_frame(), empty_meta

    forecast_df = pd.concat(frames).sort_index()
    forecast_df.index.name = "timestamp"
    fallback_points = int((forecast_df["n_samples"] == 0).sum())
    metadata = {
        "coverage": float((forecast_df["n_samples"] > 0).mean()),
        "n_target_points": len(forecast_df),
        "n_blocks_requested": int(forecast_df["block"].nunique()),
        "n_blocks_filled": int(
            forecast_df.loc[forecast_df["n_samples"] > 0, "block"].nunique()
        ),
        "global_mean": float(forecast_df[RESERVE_FORECAST_COL].mean()),
        "forecast_mode": forecast_mode,
        "fallback_points": fallback_points,
    }
    return forecast_df, metadata
