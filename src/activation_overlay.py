"""Activation-energy replay overlay (Step 3c-2a).

A SCREENING estimate of the energy-leg cash flow a BESS would have earned if it
had provided ``reserve_mw`` of a reserve product and been activated pro-rata to
the historical SYSTEM activated volume. It is a historical REPLAY OVERLAY only:

- It does NOT consume modelled SoC, does NOT occupy MILP power headroom, and is
  NOT co-optimized with DA/IDA/reserve dispatch.
- It is therefore NOT additive to the strategy-comparison total, and does NOT
  represent live trading or aggregator dispatch.

Key semantics (locked with review):

- ``system_activated_volume_mw`` is a non-negative SYSTEM-level quantity; a
  negative value is treated as zero, never as a sign.
- The asset's activated power is its capture share of the system volume, capped
  by the committed reserve: ``min(reserve_mw, capture_share * max(vol, 0))``.
- ``activation_price_eur_mwh`` is the cash-flow price already expressed in the
  import. Direction (up/down) is a breakdown label and a hook for future
  SoC/headroom coupling — it does NOT flip the price sign here (so a down
  activation at a negative price yields a negative cash flow, with no double
  sign error).

The capture share is a global model assumption, surfaced in the cockpit's audit
panel by the consuming UI increment (3c-2b); this module is a pure calculation.
"""

from __future__ import annotations

import pandas as pd

_REQUIRED_COLUMNS = {
    "product_type", "direction",
    "activation_price_eur_mwh", "system_activated_volume_mw",
}
_OVERLAY_COLUMNS = ["product", "direction", "activated_mwh", "activation_overlay_eur"]


def _interval_hours(index: pd.DatetimeIndex, default: float = 1.0) -> float:
    """Best-effort interval length (hours) from one stream's UTC timestamps.

    Uses the modal positive gap between consecutive sorted timestamps, so a 4h
    block series resolves to 4.0 and an hourly series to 1.0. Falls back to
    ``default`` when there are fewer than two distinct timestamps.
    """
    if index is None or len(index) < 2:
        return default
    ordered = pd.DatetimeIndex(index).sort_values()
    deltas = ordered.to_series().diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return default
    return float(deltas.mode().iloc[0]) / 3600.0


def _empty_result(reserve_mw: float, capture_share: float) -> dict:
    return {
        "activation_energy_overlay_eur": 0.0,
        "by_stream": pd.DataFrame(columns=_OVERLAY_COLUMNS),
        "reserve_mw": float(reserve_mw),
        "capture_share": float(capture_share),
    }


def compute_activation_overlay(
    activation_df: pd.DataFrame | None,
    *,
    reserve_mw: float,
    capture_share: float,
    interval_hours: float | None = None,
) -> dict:
    """Compute the activation-energy replay overlay.

    Args:
        activation_df: Activation frame from ``data_ingestion.read_activation_cache``
            (UTC index; ``product_type``, ``direction``, ``activation_price_eur_mwh``,
            ``system_activated_volume_mw``). May be None/empty.
        reserve_mw: Committed reserve power (MW); the per-interval asset activated
            power is capped at this. Negatives are floored to 0.
        capture_share: This asset's assumed slice of the SYSTEM activated volume
            (>= 0; floored to 0). ``0`` yields a zero overlay. Not auto-clamped
            above 1 — the ``reserve_mw`` cap bounds the delivered power.
        interval_hours: Optional explicit interval length (h); inferred per stream
            from the timestamp spacing when None.

    Returns:
        ``{"activation_energy_overlay_eur": float, "by_stream": DataFrame,
        "reserve_mw": float, "capture_share": float}``. ``by_stream`` has one row
        per (product, direction) with ``activated_mwh`` and
        ``activation_overlay_eur``. The total is an OVERLAY, deliberately NOT named
        "revenue"/"strategy" — it is not additive to the strategy-comparison total.
    """
    reserve_mw = max(float(reserve_mw), 0.0)
    capture_share = max(float(capture_share), 0.0)
    if activation_df is None or activation_df.empty:
        return _empty_result(reserve_mw, capture_share)
    if not _REQUIRED_COLUMNS.issubset(activation_df.columns):
        return _empty_result(reserve_mw, capture_share)

    rows: list[dict[str, object]] = []
    for (product, direction), grp in activation_df.groupby(
        ["product_type", "direction"], sort=True,
    ):
        dt = interval_hours if interval_hours is not None else _interval_hours(grp.index)
        # Non-negative system quantity; never interpret a negative as a sign.
        volume = grp["system_activated_volume_mw"].clip(lower=0.0)
        price = grp["activation_price_eur_mwh"]
        # Asset's slice of the system volume, capped by the committed reserve.
        asset_mw = (capture_share * volume).clip(upper=reserve_mw)
        activated_mwh = asset_mw * dt
        # Price is already a cash-flow price; direction does NOT flip its sign.
        cashflow = activated_mwh * price
        rows.append({
            "product": str(product),
            "direction": str(direction),
            "activated_mwh": float(activated_mwh.sum()),
            "activation_overlay_eur": float(cashflow.sum()),
        })

    by_stream = pd.DataFrame(rows, columns=_OVERLAY_COLUMNS)
    total = float(by_stream["activation_overlay_eur"].sum())
    return {
        "activation_energy_overlay_eur": total,
        "by_stream": by_stream,
        "reserve_mw": reserve_mw,
        "capture_share": capture_share,
    }
