"""Passive imbalance-settlement replay overlay (Step 4d-1).

A SCREENING estimate of the cash flow a BESS/portfolio would have seen if it
held a small passive imbalance position that helps the system imbalance. This is
a historical REPLAY OVERLAY only:

- It does NOT consume modelled SoC, does NOT occupy MILP power headroom, and is
  NOT co-optimized with DA/IDA/reserve dispatch.
- It is therefore NOT additive to the strategy-comparison total, and does NOT
  represent live BRP control or aggregator dispatch.

Key semantics (validated against German Netztransparenz reBAP + NRV-Saldo CSVs):

- ``system_imbalance_volume_mw > 0`` means the German system is short
  (Unterdeckung), so a positive BESS net dispatch (discharging / extra
  injection) helps the system.
- ``system_imbalance_volume_mw < 0`` means the German system is long
  (Uberdeckung), so a negative BESS net dispatch (charging / less injection)
  helps the system.
- ``imbalance_price_eur_mwh`` is the published signed cash-flow settlement price
  already. Do NOT apply a second sign flip. A helping position can still lose
  money when the published price sign makes that interval unattractive.

The capture share is a global model assumption, to be surfaced by the consuming
UI increment (4d-2); this module is a pure calculation.
"""

from __future__ import annotations

import math

import pandas as pd

_REQUIRED_COLUMNS = {"imbalance_price_eur_mwh", "system_imbalance_volume_mw"}
_OVERLAY_COLUMNS = [
    "system_state", "asset_imbalance_mwh", "imbalance_overlay_eur",
]


def _interval_hours(index: pd.DatetimeIndex, default: float = 0.25) -> float:
    """Best-effort interval length (hours) from one stream's UTC timestamps."""
    if index is None or len(index) < 2:
        return default
    ordered = pd.DatetimeIndex(index).sort_values()
    deltas = pd.Series(ordered).diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return default
    return float(deltas.mode().iloc[0]) / 3600.0


def _empty_result(power_mw: float, capture_share: float) -> dict:
    return {
        "imbalance_settlement_overlay_eur": 0.0,
        "by_system_state": pd.DataFrame(columns=_OVERLAY_COLUMNS),
        "power_mw": float(power_mw),
        "capture_share": float(capture_share),
        "sign_convention": "positive_system_imbalance_means_system_short",
    }


def compute_imbalance_overlay(
    imbalance_df: pd.DataFrame | None,
    *,
    power_mw: float,
    capture_share: float,
    interval_hours: float | None = None,
) -> dict:
    """Compute the passive imbalance-settlement replay overlay.

    Args:
        imbalance_df: Imbalance frame from
            ``data_ingestion.read_imbalance_cache`` (UTC index;
            ``imbalance_price_eur_mwh``, ``system_imbalance_volume_mw``). May be
            None/empty.
        power_mw: Physical BESS power cap (MW). The asset's passive imbalance
            position is capped at this. Negatives are floored to 0.
        capture_share: This asset/portfolio's assumed slice of the SYSTEM
            imbalance magnitude (>= 0; floored to 0). ``0`` yields a zero
            overlay. Not auto-clamped above 1; ``power_mw`` bounds the position.
        interval_hours: Optional explicit interval length (h); inferred from
            timestamp spacing when None. German reBAP/NRV files are 15-min, so
            the fallback is 0.25h.

    Returns:
        ``{"imbalance_settlement_overlay_eur": float, "by_system_state":
        DataFrame, "power_mw": float, "capture_share": float,
        "sign_convention": str}``. ``by_system_state`` aggregates the signed
        asset imbalance MWh and signed cash flow for ``system_short``,
        ``system_long``, and/or ``neutral`` intervals.

    Formula:
        ``asset_net_dispatch_mw = sign(system_imbalance_volume_mw)
        * min(power_mw, capture_share * abs(system_imbalance_volume_mw))``;
        positive dispatch means discharging / extra injection. Cash flow is then
        ``asset_net_dispatch_mw * dt * imbalance_price_eur_mwh``.
    """
    power_mw = max(float(power_mw), 0.0)
    capture_share = max(float(capture_share), 0.0)
    if imbalance_df is None or imbalance_df.empty:
        return _empty_result(power_mw, capture_share)
    if not _REQUIRED_COLUMNS.issubset(imbalance_df.columns):
        return _empty_result(power_mw, capture_share)

    price = pd.to_numeric(imbalance_df["imbalance_price_eur_mwh"], errors="coerce")
    system_volume = pd.to_numeric(
        imbalance_df["system_imbalance_volume_mw"], errors="coerce",
    )
    valid = price.notna() & system_volume.notna()
    if not bool(valid.any()):
        return _empty_result(power_mw, capture_share)

    price = price.loc[valid]
    system_volume = system_volume.loc[valid]
    idx = pd.DatetimeIndex(imbalance_df.loc[valid].index)
    dt = interval_hours if interval_hours is not None else _interval_hours(idx)
    if not math.isfinite(float(dt)) or float(dt) <= 0:
        dt = 0.25

    magnitude = (capture_share * system_volume.abs()).clip(upper=power_mw)
    direction = system_volume.apply(lambda v: 1.0 if v > 0 else (-1.0 if v < 0 else 0.0))
    asset_mw = magnitude * direction
    asset_mwh = asset_mw * float(dt)
    cashflow = asset_mwh * price

    state = pd.Series("neutral", index=system_volume.index)
    state = state.mask(system_volume > 0, "system_short")
    state = state.mask(system_volume < 0, "system_long")
    work = pd.DataFrame({
        "system_state": state,
        "asset_imbalance_mwh": asset_mwh,
        "imbalance_overlay_eur": cashflow,
    })
    by_state = (
        work.groupby("system_state", sort=True, as_index=False)
        .agg({
            "asset_imbalance_mwh": "sum",
            "imbalance_overlay_eur": "sum",
        })
        .reindex(columns=_OVERLAY_COLUMNS)
    )
    return {
        "imbalance_settlement_overlay_eur": float(by_state["imbalance_overlay_eur"].sum()),
        "by_system_state": by_state,
        "power_mw": power_mw,
        "capture_share": capture_share,
        "sign_convention": "positive_system_imbalance_means_system_short",
    }
