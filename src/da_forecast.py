"""Screening DA-price forecast for the 9.2b reserve-commit stage (prep).

Phase 9.2b commits reserve capacity BEFORE the DA auction clears (reserve
auctions gate D-1 morning, DA at noon). Its Stage-0 reserve sizing must
therefore weigh reserve capacity revenue against the *expected* DA arbitrage
opportunity, NOT the realised DA print — otherwise 9.2b silently degenerates
into a variant of the 9.2a perfect-foresight ceiling.

This module produces that expectation as a SCREENING DA-price forecast by
reusing the IDA hour-of-day / hour-of-week climatology machinery on the DA
series. It is a screening forecast (climatology, hourly bucketed), not a
trading-grade DA model.

Because this helper exists for the reserve-first COMMITMENT, it defaults to
``forecast_mode="walk_forward"`` (only days strictly before the target), so
Stage 0 never peeks at the target day's realised DA price NOR at later days'
prices — i.e. it sees only what a desk could know at reserve-commit time. The
unbiased leave-one-day-out mode (``"loo"``, which may use future days) is still
available but must be requested explicitly, and is only appropriate for
skill/backtest reporting, never for the realistic commitment path.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.ida_forecast import FORECAST_COL as _IDA_FORECAST_COL
from src.ida_forecast import build_ida_forecast

DA_VALUE_COL = "price_eur_mwh"
DA_FORECAST_COL = "forecast_da_eur_mwh"


def build_da_price_forecast(
    da_history: pd.DataFrame,
    *,
    target_dates: list[date],
    tz: str | None = None,
    bucket: str = "hour_of_day",
    forecast_mode: str = "walk_forward",
) -> tuple[pd.DataFrame, dict]:
    """Climatology DA-price forecast at each target day's own timestamps.

    Thin wrapper over :func:`ida_forecast.build_ida_forecast` applied to the DA
    price series and relabelled to a DA-specific forecast column, so the 9.2b
    Stage-0 reserve commitment can size headroom against *expected* DA
    arbitrage. ``forecast_mode`` defaults to ``"walk_forward"`` (prior days
    only) because this helper feeds a real-time commitment: the default must not
    leak the target day's or any later day's realised prices. ``"loo"``
    (leave-one-day-out, may use future days — unbiased skill but not real-time)
    and ``"in_sample"`` (diagnostics) are available only when requested.

    Args:
        da_history: UTC-indexed frame with ``price_eur_mwh`` (the loaded DA
            series for the zone).
        target_dates: Local calendar dates to produce a forecast for.
        tz: IANA timezone for the climatology bucketing (None groups on UTC).
        bucket: ``"hour_of_day"`` (robust on short windows) or
            ``"hour_of_week"`` (weekday-aware, needs several weeks).
        forecast_mode: ``"loo"`` (default), ``"walk_forward"``, ``"in_sample"``.

    Returns:
        ``(forecast_df, metadata)``. ``forecast_df`` is UTC-indexed with columns
        ``[DA_FORECAST_COL, "bucket", "n_samples"]`` covering the target dates'
        timestamps; ``metadata`` carries ``coverage`` / ``n_buckets_filled`` /
        ``n_buckets_requested`` / ``fallback_points`` / ``forecast_mode`` so the
        caller can judge forecast support (same shape as the IDA forecast).
    """
    forecast_df, metadata = build_ida_forecast(
        da_history,
        target_dates=target_dates,
        tz=tz,
        value_col=DA_VALUE_COL,
        bucket=bucket,
        forecast_mode=forecast_mode,
    )
    if _IDA_FORECAST_COL in forecast_df.columns:
        forecast_df = forecast_df.rename(
            columns={_IDA_FORECAST_COL: DA_FORECAST_COL},
        )
    return forecast_df, metadata
