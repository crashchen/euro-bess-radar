"""Coverage classification from raw prices (contract §4.3, red-lines #17/#21).

Completeness is an **expected-grid exact match** per consumed leg, derived before
any scalar collapse or zero-fill (reviews 36/55). A duplicate, NaN/Inf, missing, or
extra row makes a day ``missing`` — never a silent €0 and never a solver failure.
The classification order is pinned: data-completeness + reserve-coverage gates run
first (→ ``missing``); only a day that passes them and then fails the solver is
``solver_failed`` (§4.3).
"""

from __future__ import annotations

import datetime as _dt
import math
from collections import Counter

import pandas as pd

from src.project_case import grid
from src.project_case.schema import (
    ReserveCoverageAudit,
    ReserveCoverageEntry,
)


class AdapterUnavailableError(Exception):
    """A model-support / no-valid-day outcome: emit **no** StrategyRunResult.

    Distinct from ``ProjectCaseValidationError`` (a schema/invariant violation).
    Raised when a consumed leg lacks a registry entry, or the final ``valid_dates``
    set is empty (§4.3, §5, red-lines #17/#18/#21).
    """


def _utc_series(series: pd.Series, name: str) -> pd.Series:
    """Return ``series`` with a tz-aware UTC DatetimeIndex (float values)."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    idx = pd.DatetimeIndex(series.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return pd.Series(series.to_numpy(dtype=float), index=idx)


def _local_dates(idx: pd.DatetimeIndex, tz: str) -> pd.Index:
    return pd.DatetimeIndex(idx.tz_convert(tz)).date


def classify_leg_complete_dates(
    series: pd.Series,
    *,
    zone: str,
    leg: str,
    evaluation_dates: tuple[_dt.date, ...],
) -> frozenset[_dt.date]:
    """Dates with an exact expected-grid match for one energy leg (DA or IDA).

    Raises :class:`AdapterUnavailableError` when the leg/zone has no registry entry.
    Energy prices may be negative; only finiteness + grid identity are required.
    """
    if grid.profile_id(leg, zone) is None:
        raise AdapterUnavailableError(
            f"market-grid registry has no {leg} calendar for zone {zone}"
        )
    tz = grid._zone_tz(zone)
    s = _utc_series(series, f"{leg} price series")
    local = _local_dates(s.index, tz)
    complete: set[_dt.date] = set()
    for target in evaluation_dates:
        expected = grid.expected_timestamps(leg, zone, target)
        if expected is None:  # pragma: no cover - support checked above
            raise AdapterUnavailableError(
                f"no {leg} grid for {zone} {target}"
            )
        mask = local == target
        day_index = s.index[mask]
        day_values = s.to_numpy()[mask]
        if Counter(day_index) != Counter(expected):
            continue  # missing / extra / duplicate timestamp
        if not all(math.isfinite(v) for v in day_values):
            continue  # NaN / Inf at an expected point
        complete.add(target)
    return frozenset(complete)


def build_reserve_coverage_audit(
    block_price_series: pd.Series,
    *,
    zone: str,
    evaluation_dates: tuple[_dt.date, ...],
) -> ReserveCoverageAudit:
    """Build a per-day :class:`ReserveCoverageAudit` from raw block prices.

    ``block_price_series`` is a UTC-indexed series of block-start prices
    (EUR/MW/h). A block is *present* only when the raw input has **exactly one**
    row at its block-start timestamp with a finite ``block_price_eur_mw_h >= 0``
    (red-line #21). One entry per evaluation date; a fully-missing day is retained
    with ``present = ∅``.
    """
    if grid.reserve_profile_id(zone) is None:
        raise AdapterUnavailableError(
            f"market-grid registry has no reserve calendar for zone {zone}"
        )
    s = _utc_series(block_price_series, "reserve block price series")
    counts = Counter(s.index)
    tz = grid._zone_tz(zone)
    local = _local_dates(s.index, tz)
    entries: list[ReserveCoverageEntry] = []
    for target in evaluation_dates:
        blocks = grid.reserve_blocks(zone, target)
        if blocks is None:  # pragma: no cover - support checked above
            raise AdapterUnavailableError(f"no reserve blocks for {zone} {target}")
        required = tuple(bid for bid, _ in blocks)
        durations = {bid: float(dur) for bid, dur in blocks}
        canonical_starts = {pd.Timestamp(bid) for bid in required}
        day_ts = s.index[local == target]
        if any(ts not in canonical_starts for ts in day_ts):
            # A day carrying any non-canonical / extra reserve row is malformed and
            # untrustworthy -> fully uncovered (red-line #17/#21, no pass-through).
            present: list[str] = []
        else:
            present = [
                bid for bid in required if _block_is_present(s, counts, pd.Timestamp(bid))
            ]
        present_set = frozenset(present)
        missing = tuple(b for b in required if b not in present_set)
        entries.append(
            ReserveCoverageEntry(
                date=target,
                required_blocks=required,
                present_blocks=tuple(present),
                missing_blocks=missing,
                settlement_duration_hours_by_block=durations,
            )
        )
    return ReserveCoverageAudit(tuple(entries))


def _block_is_present(s: pd.Series, counts: Counter, ts: pd.Timestamp) -> bool:
    """A block is present iff exactly one canonical row with finite price >= 0."""
    if counts.get(ts, 0) != 1:
        return False  # missing or duplicate row
    value = float(s.loc[ts])
    return math.isfinite(value) and value >= 0.0


def reserve_scalar_price(
    block_price_series: pd.Series,
    *,
    zone: str,
    pricing_dates: tuple[_dt.date, ...],
) -> float:
    """Duration-weighted mean of complete blocks over ``pricing_dates`` (§5).

    ``sum(block_price * explicit_block_duration) / sum(explicit_block_duration)``
    using canonical product-block durations — never the gap to the adjacent
    surviving timestamp (``duration_weighted_mean_complete_blocks_v1``).
    """
    s = _utc_series(block_price_series, "reserve block price series")
    counts = Counter(s.index)
    num = 0.0
    den = 0.0
    for target in pricing_dates:
        blocks = grid.reserve_blocks(zone, target)
        if blocks is None:  # pragma: no cover
            raise AdapterUnavailableError(f"no reserve blocks for {zone} {target}")
        for bid, dur in blocks:
            ts = pd.Timestamp(bid)
            if counts.get(ts, 0) != 1:
                raise AdapterUnavailableError(
                    f"reserve scalar over incomplete block {bid} (pre-gate violated)"
                )
            price = float(s.loc[ts])
            if not (math.isfinite(price) and price >= 0.0):
                raise AdapterUnavailableError(
                    f"reserve scalar over invalid block price at {bid}"
                )
            num += price * float(dur)
            den += float(dur)
    if den <= 0.0:
        raise AdapterUnavailableError("reserve scalar has no priced blocks")
    return num / den
