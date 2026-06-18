"""Source traceability and data-quality summaries for dashboard outputs."""

from __future__ import annotations

import pandas as pd

from src.config import get_zone_timezone, is_elexon_zone
from src.data_ingestion import summarize_price_data_quality

QUALITY_COLUMNS = [
    "zone", "source", "timezone", "first_timestamp_utc", "last_timestamp_utc",
    "total_intervals", "valid_intervals", "coverage_pct",
    "source_gap_intervals", "imputed_intervals", "missing_intervals",
    "source_gap_pct", "imputed_pct", "missing_pct", "max_source_gap_hours",
]

INTRADAY_SOURCE_COLUMNS = [
    "zone", "sequence", "source", "rows",
    "first_timestamp_utc", "last_timestamp_utc",
]


def build_intraday_source_table(
    manual_sources: dict[tuple[str, int], dict] | None,
) -> pd.DataFrame:
    """Build an audit table of manually uploaded IDA price sources.

    ENTSO-E currently returns no intraday-auction data for the tested SIDC
    zones/windows, so IDA prices in the cockpit/uplift panels may come from
    a manual CSV upload instead of the API. This surfaces those uploads with
    ``source = "Manual CSV"`` so the IDA-driven numbers are traceable rather
    than appearing to be API-sourced.

    Args:
        manual_sources: Session mapping ``(zone, sequence) -> {"rows",
            "first", "last"}`` populated by the sidebar IDA upload handler.

    Returns:
        One row per uploaded (zone, sequence), sorted by zone then sequence.
    """
    rows: list[dict[str, object]] = []
    for (zone, sequence), meta in sorted((manual_sources or {}).items()):
        rows.append({
            "zone": str(zone),
            "sequence": int(sequence),
            "source": "Manual CSV",
            "rows": int(meta.get("rows", 0)),
            "first_timestamp_utc": meta.get("first", pd.NaT),
            "last_timestamp_utc": meta.get("last", pd.NaT),
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
