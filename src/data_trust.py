"""Source traceability and data-quality summaries for dashboard outputs."""

from __future__ import annotations

import pandas as pd

from src.ancillary import list_capacity_products
from src.config import get_zone_timezone, is_elexon_zone
from src.data_ingestion import (
    read_capacity_sources,
    read_intraday_sources,
    summarize_price_data_quality,
)

QUALITY_COLUMNS = [
    "zone", "source", "timezone", "first_timestamp_utc", "last_timestamp_utc",
    "total_intervals", "valid_intervals", "coverage_pct",
    "source_gap_intervals", "imputed_intervals", "missing_intervals",
    "source_gap_pct", "imputed_pct", "missing_pct", "max_source_gap_hours",
]

INTRADAY_SOURCE_COLUMNS = [
    "zone", "sequence", "source", "rows",
    "first_timestamp_utc", "last_timestamp_utc", "imported_at",
]

COVERAGE_MATRIX_COLUMNS = ["zone", "DA", "IDA1", "IDA2", "IDA3", "reserve_capacity"]

CAPACITY_SOURCE_COLUMNS = [
    "zone", "product", "direction", "source", "rows",
    "first_timestamp_utc", "last_timestamp_utc", "imported_at",
]

# Displayed when a (zone, stream) cell has no loaded/cached data.
_NO_DATA = "—"


def build_capacity_source_table(
    sources: dict[tuple[str, str, str], dict] | None = None,
) -> pd.DataFrame:
    """Build an audit table of cached reserve-capacity provenance.

    Mirrors :func:`build_intraday_source_table` but at the
    ``(zone, product, direction)`` granularity of the durable
    ``capacity_price_sources`` sidecar (written by ``write_capacity_cache``), so
    each reserve stream's source survives a restart and a re-import that changes
    the source relabels that stream instead of leaving a stale label.

    Args:
        sources: Optional pre-built mapping ``(zone, product, direction) ->
            {"source", "rows", "first", "last", "imported_at"}``. Read from the
            database when None (the normal path; the argument exists for testing).

    Returns:
        One row per (zone, product, direction), sorted.
    """
    if sources is None:
        sources = read_capacity_sources()
    rows: list[dict[str, object]] = []
    for (zone, product, direction), meta in sorted((sources or {}).items()):
        rows.append({
            "zone": str(zone),
            "product": str(product),
            "direction": str(direction),
            "source": meta.get("source", "Manual CSV"),
            "rows": int(meta.get("rows", 0)),
            "first_timestamp_utc": meta.get("first", pd.NaT),
            "last_timestamp_utc": meta.get("last", pd.NaT),
            "imported_at": meta.get("imported_at"),
        })
    if not rows:
        return pd.DataFrame(columns=CAPACITY_SOURCE_COLUMNS)
    return pd.DataFrame(rows, columns=CAPACITY_SOURCE_COLUMNS)


def build_intraday_source_table(
    sources: dict[tuple[str, int], dict] | None = None,
) -> pd.DataFrame:
    """Build an audit table of cached IDA price provenance.

    Provenance is read from the durable ``ida_price_sources`` SQLite sidecar
    (written by ``write_intraday_cache``) so it survives a session/server
    restart — manually uploaded IDA prices stay labelled ``Manual CSV`` even
    after the uploading session is gone, and a later live fetch relabels the
    same (zone, sequence) instead of leaving a stale manual label.

    Args:
        sources: Optional pre-built provenance mapping ``(zone, sequence) ->
            {"source", "rows", "first", "last", "imported_at"}``. When None,
            it is read from the database (the normal path; the argument exists
            for testing).

    Returns:
        One row per (zone, sequence), sorted by zone then sequence.
    """
    if sources is None:
        sources = read_intraday_sources()
    rows: list[dict[str, object]] = []
    for (zone, sequence), meta in sorted((sources or {}).items()):
        rows.append({
            "zone": str(zone),
            "sequence": int(sequence),
            "source": meta.get("source", "Manual CSV"),
            "rows": int(meta.get("rows", 0)),
            "first_timestamp_utc": meta.get("first", pd.NaT),
            "last_timestamp_utc": meta.get("last", pd.NaT),
            "imported_at": meta.get("imported_at"),
        })
    if not rows:
        return pd.DataFrame(columns=INTRADAY_SOURCE_COLUMNS)
    return pd.DataFrame(rows, columns=INTRADAY_SOURCE_COLUMNS)


def source_label_for_zone(zone: str) -> str:
    """Return the implemented day-ahead source label for a bidding zone."""
    return (
        "Elexon Insights API (GBP->EUR)"
        if is_elexon_zone(zone)
        else "ENTSO-E Transparency Platform"
    )


def build_zone_data_quality_table(
    zone_data: dict[str, pd.DataFrame],
    *,
    zone_timezones: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build one row per fetched zone with coverage and provenance metadata.

    Args:
        zone_data: Mapping of bidding-zone code to cleaned price DataFrame.
        zone_timezones: Optional timezone overrides. When omitted, project
            zone config is used.

    Returns:
        DataFrame suitable for display/export. Percent columns are stored as
        0-100 values for Streamlit table formatting.
    """
    zone_timezones = zone_timezones or {}
    rows: list[dict[str, object]] = []

    for zone, df in zone_data.items():
        if df is None:
            continue
        quality = summarize_price_data_quality(df)
        total = int(quality["total_intervals"])
        valid = int(quality["valid_intervals"])
        coverage_pct = (100.0 * valid / total) if total else 0.0

        if df.empty:
            first_ts = pd.NaT
            last_ts = pd.NaT
        else:
            idx = pd.DatetimeIndex(df.index)
            first_ts = idx.min()
            last_ts = idx.max()

        rows.append({
            "zone": zone,
            "source": source_label_for_zone(zone),
            "timezone": zone_timezones.get(zone, get_zone_timezone(zone)),
            "first_timestamp_utc": first_ts,
            "last_timestamp_utc": last_ts,
            "total_intervals": total,
            "valid_intervals": valid,
            "coverage_pct": round(coverage_pct, 2),
            "source_gap_intervals": int(quality["source_gap_intervals"]),
            "imputed_intervals": int(quality["imputed_intervals"]),
            "missing_intervals": int(quality["missing_intervals"]),
            "source_gap_pct": round(float(quality["source_gap_ratio"]) * 100.0, 2),
            "imputed_pct": round(float(quality["imputed_ratio"]) * 100.0, 2),
            "missing_pct": round(float(quality["missing_ratio"]) * 100.0, 2),
            "max_source_gap_hours": float(quality["max_source_gap_hours"]),
        })

    if not rows:
        return pd.DataFrame(columns=QUALITY_COLUMNS)
    return (
        pd.DataFrame(rows, columns=QUALITY_COLUMNS)
        .sort_values("zone")
        .reset_index(drop=True)
    )


def _da_coverage_cell(df: pd.DataFrame | None) -> str:
    """DA stream cell: coverage% + valid/total interval count, or no-data."""
    if df is None:
        return _NO_DATA
    quality = summarize_price_data_quality(df)
    total = int(quality["total_intervals"])
    valid = int(quality["valid_intervals"])
    if total == 0:
        return "empty"
    return f"{100.0 * valid / total:.0f}% ({valid}/{total})"


def _intraday_cell(
    intraday_sources: dict[tuple[str, int], dict] | None, zone: str, sequence: int,
) -> str:
    """IDA{n} stream cell: provenance label + cached row count, or no-data."""
    meta = (intraday_sources or {}).get((zone, sequence))
    if not meta:
        return _NO_DATA
    return f"{meta.get('source', '?')} ({int(meta.get('rows', 0))})"


def build_coverage_matrix(
    zone_data: dict[str, pd.DataFrame],
    *,
    intraday_sources: dict[tuple[str, int], dict] | None = None,
    ancillary_df: pd.DataFrame | None = None,
    capacity_sources: dict[tuple[str, str, str], dict] | None = None,
    primary_zone: str | None = None,
) -> pd.DataFrame:
    """Zone x data-stream coverage matrix for the Data Trust tab.

    One row per zone the run touches (loaded DA zones, plus zones with cached
    IDA provenance, plus the primary zone), with a short status per stream so a
    user can see at a glance what is loaded and where the gaps are before
    trusting any revenue/uplift number.

    Args:
        zone_data: Mapping of bidding-zone code to cleaned DA price DataFrame.
        intraday_sources: ``(zone, sequence) -> provenance`` mapping (from
            ``read_intraday_sources``); read from the DB sidecar when omitted.
        ancillary_df: Merged ancillary dataset for the *primary* zone (session
            per-country / auto-fetch capacity; not zone-tagged, so it backs the
            reserve cell for ``primary_zone`` only — the fallback when a zone has
            no persisted unified-import capacity).
        capacity_sources: ``(zone, product, direction) -> provenance`` mapping
            (from ``read_capacity_sources``); read from the DB sidecar when
            omitted. This is the zone-tagged unified-import capacity, so the
            reserve column is shown PER zone, not just the primary.
        primary_zone: The zone whose session ancillary capacity products back the
            reserve cell when that zone has no persisted unified-import capacity.

    Returns:
        Wide DataFrame with columns ``COVERAGE_MATRIX_COLUMNS``; cells hold
        ``DA`` coverage%, ``IDA{n}`` ``source (rows)``, a reserve product list,
        or ``"—"`` when a stream is absent. Empty (with columns) when nothing
        is loaded.
    """
    if intraday_sources is None:
        intraday_sources = read_intraday_sources()
    if capacity_sources is None:
        capacity_sources = read_capacity_sources()
    zone_data = zone_data or {}

    # Zone-tagged unified-import capacity (persisted) -> products per zone.
    persisted_by_zone: dict[str, set] = {}
    for (zone, product, _direction) in (capacity_sources or {}):
        persisted_by_zone.setdefault(str(zone), set()).add(str(product))
    # Session per-country / auto-fetch capacity is primary-zone only (fallback).
    session_products = list_capacity_products(ancillary_df)

    zones = (
        set(zone_data)
        | {z for (z, _seq) in (intraday_sources or {})}
        | set(persisted_by_zone)
    )
    if primary_zone and session_products:
        zones.add(primary_zone)
    if not zones:
        return pd.DataFrame(columns=COVERAGE_MATRIX_COLUMNS)

    rows: list[dict[str, object]] = []
    for zone in sorted(zones):
        persisted = sorted(persisted_by_zone.get(zone, set()))
        if persisted:
            reserve = ", ".join(persisted)
        elif zone == primary_zone and session_products:
            reserve = ", ".join(session_products)
        else:
            reserve = _NO_DATA
        rows.append({
            "zone": zone,
            "DA": _da_coverage_cell(zone_data.get(zone)),
            "IDA1": _intraday_cell(intraday_sources, zone, 1),
            "IDA2": _intraday_cell(intraday_sources, zone, 2),
            "IDA3": _intraday_cell(intraday_sources, zone, 3),
            "reserve_capacity": reserve,
        })
    return pd.DataFrame(rows, columns=COVERAGE_MATRIX_COLUMNS)
