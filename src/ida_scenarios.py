"""IDA price scenario generation for the stochastic DA+ID MILP (Increment A).

The stochastic dispatch upgrade (``docs/design/stochastic-milp-v1.md``)
commits the Stage-1 DA schedule against a *distribution* of IDA price paths
instead of a single point forecast. This module builds that distribution —
the scenario generator ONLY; no solver code lives here.

v1 mode is ``error_resample`` (locked in the scope review): each scenario is
the climatology point forecast (the same forecast the sequential policy
already uses, from :mod:`src.ida_forecast`) plus a resampled historical daily
*error path* (realised - forecast, whole days). This preserves the cross-hour
error correlation a per-interval noise model would destroy.

Three contract points from §3 of the design doc are load-bearing and tested:

- **Mean-centring**: the sampled error paths are re-centred so their mean path
  is exactly zero, hence the scenario mean equals the base forecast per
  interval. This is what makes the ``rebid_cap = inf`` decoupling identity
  (§8-1) exact: without it a finite resample has a nonzero mean-error path and
  the decoupled solution would target the scenario-mean path, not the base
  forecast.
- **Resolution / grid partitioning**: error paths are sampled ONLY from
  history days whose local ``(hour, minute)`` sequence matches the target day
  (``_shape_key``). This keeps a 60-min error path off a 15-min day (DE_LU
  switched market time unit in Oct 2025), keeps a DST-short/long day off a
  normal day, AND keeps a sparse day (a normal day missing an hour, same COUNT
  as a spring-forward day but a different grid) from being force-aligned — no
  interpolation, which would fabricate sub-hour structure.
- **Forecast-mode history discipline**: the error pool obeys the target day's
  ``forecast_mode`` allowed history exactly as :func:`ida_forecast.build_ida_forecast`
  does (``loo`` default excludes only the target day; ``walk_forward`` uses
  strictly earlier days; ``in_sample`` is diagnostic). Under ``loo`` the
  target day is never in its own pool.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.analytics import _to_local
from src.ida_forecast import (
    _VALID_BUCKETS,
    _VALID_FORECAST_MODES,
    FORECAST_COL,
    IDA_VALUE_COL,
    _bucket_of,
)

# Column carrying the base point forecast in each day's bundle, matching
# ``ida_forecast.FORECAST_COL`` so the solver can consume either interchangeably.
SCENARIO_BASE_COL = FORECAST_COL


def build_ida_scenarios(
    ida_history: pd.DataFrame,
    *,
    target_dates: list[date],
    n_scenarios: int = 10,
    tz: str | None = None,
    value_col: str = IDA_VALUE_COL,
    bucket: str = "hour_of_day",
    forecast_mode: str = "loo",
    seed: int | None = None,
    mean_centered: bool = True,
) -> tuple[dict[date, dict], dict]:
    """Build an ``error_resample`` IDA scenario bundle per target day.

    For every local date in ``target_dates`` this returns a bundle carrying the
    base point forecast, ``n_scenarios`` full-day price paths (base forecast +
    a resampled, mean-centred historical error path), and equal weights, plus
    generation metadata. No solver is involved.

    Args:
        ida_history: UTC-indexed frame with ``value_col`` (the loaded IDA1
            series for the zone).
        target_dates: Local calendar dates to generate scenarios for.
        n_scenarios: Number of scenarios ``S`` per day (equal-weighted in v1).
        tz: IANA timezone for the local-day grouping and climatology buckets.
        value_col: Realised IDA price column.
        bucket: Climatology bucket, ``"hour_of_day"`` (default) or
            ``"hour_of_week"`` (see :func:`ida_forecast.build_ida_forecast`).
        forecast_mode: History discipline, ``"loo"`` / ``"walk_forward"`` /
            ``"in_sample"`` (same semantics and caveats as the forecast).
        seed: Seed for the resampling RNG; recorded in metadata for
            reproducibility.
        mean_centered: When True (default) re-centre the sampled error paths so
            the scenario mean equals the base forecast exactly (§3).

    Returns:
        ``(scenarios_by_date, metadata)``. ``scenarios_by_date`` maps each
        generated date to a dict with ``base_forecast`` (UTC-indexed Series),
        ``scenarios`` (``(S, n_intervals)`` ndarray), ``timestamps``
        (UTC DatetimeIndex), ``weights`` (``(S,)`` ndarray summing to 1),
        ``pool_size`` (distinct same-shape error days available),
        ``sampled_with_replacement`` (True when ``pool_size < S``),
        ``base_coverage`` (fraction of intervals whose base forecast is
        bucket-backed vs global-mean fallback), and ``base_fallback_points``.
        ``metadata`` carries ``mode``, ``forecast_mode``, ``n_scenarios``,
        ``seed``, ``mean_centered``, ``bucket``, ``n_target_days``,
        ``n_days_generated``, ``min_pool_size``, ``days_with_replacement``,
        ``min_base_coverage``, and ``total_fallback_points``.
    """
    if bucket not in _VALID_BUCKETS:
        raise ValueError(f"bucket must be one of {_VALID_BUCKETS}, got {bucket!r}")
    if forecast_mode not in _VALID_FORECAST_MODES:
        raise ValueError(
            f"forecast_mode must be one of {_VALID_FORECAST_MODES}, got "
            f"{forecast_mode!r}"
        )
    if n_scenarios < 1:
        raise ValueError(f"n_scenarios must be >= 1, got {n_scenarios}")

    meta = _base_metadata(n_scenarios, forecast_mode, bucket, seed, mean_centered)
    if (
        ida_history is None or ida_history.empty
        or value_col not in ida_history.columns or not target_dates
    ):
        return {}, meta

    local = _to_local(ida_history[[value_col]].dropna(), tz).sort_index()
    if local.empty:
        return {}, meta
    local = local.assign(
        _bucket=_bucket_of(local.index, bucket), _date=local.index.date,
    )

    rng = np.random.default_rng(seed)
    out: dict[date, dict] = {}
    meta["n_target_days"] = len(set(target_dates))
    for target in sorted(set(target_dates)):
        bundle = _build_day_bundle(
            local, target, value_col, forecast_mode, n_scenarios, rng,
            mean_centered,
        )
        if bundle is not None:
            out[target] = bundle

    meta["n_days_generated"] = len(out)
    if out:
        meta["min_pool_size"] = min(b["pool_size"] for b in out.values())
        meta["days_with_replacement"] = sum(
            b["sampled_with_replacement"] for b in out.values()
        )
        meta["min_base_coverage"] = min(b["base_coverage"] for b in out.values())
        meta["total_fallback_points"] = sum(
            b["base_fallback_points"] for b in out.values()
        )
    return out, meta


def _base_metadata(
    n_scenarios: int, forecast_mode: str, bucket: str, seed: int | None,
    mean_centered: bool,
) -> dict:
    """Metadata skeleton shared by empty and populated returns."""
    return {
        "mode": "error_resample",
        "forecast_mode": forecast_mode,
        "n_scenarios": n_scenarios,
        "seed": seed,
        "mean_centered": mean_centered,
        "bucket": bucket,
        "n_target_days": 0,
        "n_days_generated": 0,
        "min_pool_size": 0,
        "days_with_replacement": 0,
        "min_base_coverage": float("nan"),
        "total_fallback_points": 0,
    }


def _allowed_climatology(
    local: pd.DataFrame, target: date, forecast_mode: str,
) -> pd.DataFrame:
    """History rows a target day may see under its forecast mode (§3)."""
    if forecast_mode == "walk_forward":
        return local[local["_date"] < target]
    if forecast_mode == "in_sample":
        return local
    return local[local["_date"] != target]  # loo


def _build_day_bundle(
    local: pd.DataFrame, target: date, value_col: str, forecast_mode: str,
    n_scenarios: int, rng: np.random.Generator, mean_centered: bool,
) -> dict | None:
    """Assemble one target day's scenario bundle, or None when unsupported.

    Returns None when the day has no rows, no usable climatology, or no
    same-shape error day in the allowed history (rather than force-aligning a
    day with a different interval grid).
    """
    day_rows = local[local["_date"] == target]
    if day_rows.empty:
        return None
    clim = _allowed_climatology(local, target, forecast_mode)
    if clim.empty:
        return None

    bucket_mean = clim.groupby("_bucket")[value_col].mean()
    bucket_count = clim.groupby("_bucket")[value_col].size()
    global_mean = float(clim[value_col].mean())
    target_buckets = day_rows["_bucket"].to_numpy()
    base = _bucket_forecast(target_buckets, bucket_mean, global_mean)
    sample_counts = bucket_count.reindex(target_buckets).fillna(0).to_numpy(dtype=int)

    pool = _error_pool(clim, bucket_mean, global_mean, value_col, _shape_key(day_rows))
    if not pool:
        return None

    errors, with_replacement = _sample_errors(pool, n_scenarios, rng, mean_centered)
    timestamps = day_rows.index.tz_convert("UTC")
    timestamps.name = "timestamp"
    return {
        "base_forecast": pd.Series(base, index=timestamps, name=SCENARIO_BASE_COL),
        "scenarios": base[None, :] + errors,
        "timestamps": timestamps,
        "weights": np.full(n_scenarios, 1.0 / n_scenarios),
        "pool_size": len(pool),
        "sampled_with_replacement": with_replacement,
        # Base-forecast bucket support (§3, build_ida_forecast style): a day
        # whose base is mostly global-mean fallback is labelled downstream.
        "base_coverage": float((sample_counts > 0).mean()),
        "base_fallback_points": int((sample_counts == 0).sum()),
    }


def _bucket_forecast(
    buckets: np.ndarray, bucket_mean: pd.Series, global_mean: float,
) -> np.ndarray:
    """Map buckets to their climatology mean, empty buckets to the global mean."""
    vals = bucket_mean.reindex(buckets).to_numpy(dtype=float)
    return np.where(np.isnan(vals), global_mean, vals)


def _shape_key(day: pd.DataFrame) -> tuple[tuple[int, int], ...]:
    """Ordered local ``(hour, minute)`` sequence identifying a day's grid.

    Two days share a key iff their intervals line up position-by-position in
    local time — which is exactly the condition for a positional error-path
    add to be valid. This subsumes resolution (15-min vs 60-min differ),
    DST length (23-/25-hour days differ from 24-hour), AND sparse gaps (a
    normal day missing an hour differs from a DST day of the same COUNT).
    Offset is deliberately NOT in the key: the climatology pools summer↔winter
    by local hour, so a CEST day is a valid error source for a CET target.
    """
    idx = day.index
    return tuple(zip(idx.hour.tolist(), idx.minute.tolist(), strict=True))


def _error_pool(
    clim: pd.DataFrame, bucket_mean: pd.Series, global_mean: float,
    value_col: str, target_key: tuple[tuple[int, int], ...],
) -> list[np.ndarray]:
    """Whole-day error paths (realised - forecast) for same-shape days only.

    Only history days whose local ``(hour, minute)`` sequence matches the
    target day contribute (see :func:`_shape_key`), so a resolution, DST-length,
    or sparse-gap mismatch drops the day instead of being force-aligned. Errors
    are positional within the day, valid because the shared key guarantees the
    same local-time ordering.
    """
    pool: list[np.ndarray] = []
    for _, day in clim.groupby("_date", sort=True):
        if _shape_key(day) != target_key:
            continue
        forecast = _bucket_forecast(day["_bucket"].to_numpy(), bucket_mean, global_mean)
        pool.append(day[value_col].to_numpy(dtype=float) - forecast)
    return pool


def _sample_errors(
    pool: list[np.ndarray], n_scenarios: int, rng: np.random.Generator,
    mean_centered: bool,
) -> tuple[np.ndarray, bool]:
    """Sample S error paths (with replacement only when the pool is too small).

    Mean-centring subtracts the sampled paths' mean path, so with equal weights
    the scenario mean equals the base forecast exactly (§3). The shift is one
    per-interval constant across scenarios, so each path's cross-hour shape is
    preserved (it is no longer a raw historical error, which metadata records).
    """
    with_replacement = len(pool) < n_scenarios
    idx = rng.choice(len(pool), size=n_scenarios, replace=with_replacement)
    errors = np.array([pool[i] for i in idx], dtype=float)
    if mean_centered:
        errors = errors - errors.mean(axis=0, keepdims=True)
    return errors, with_replacement
