"""Excel report export for eu-bess-pulse."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from src.analytics import build_price_heatmap
from src.config import CACHE_DIR


# ── Styling helpers ──────────────────────────────────────────────────────────

_HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
_HEADER_FILL = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
_PRICE_FMT = "#,##0.00"
_PCT_FMT = "0.0%"


def _write_header_row(ws, row: int, headers: list[str]) -> None:
    """Write a styled header row."""
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=row, column=col, value=header)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = Alignment(horizontal="center")


def _auto_column_width(ws) -> None:
    """Auto-fit column widths based on content."""
    for col_cells in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col_cells[0].column)
        for cell in col_cells:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max_len + 3, 30)


def _write_kv_pair(ws, row: int, key: str, value, fmt: str | None = None) -> int:
    """Write a key-value pair to the summary sheet."""
    ws.cell(row=row, column=1, value=key).font = Font(bold=True)
    cell = ws.cell(row=row, column=2, value=value)
    if fmt:
        cell.number_format = fmt
    return row + 1


# ── Sheet builders ───────────────────────────────────────────────────────────

def _build_summary_sheet(
    ws, zone: str, price_df: pd.DataFrame,
    percentiles: dict[str, float],
    revenue_estimate: dict[str, float],
    negative_stats: dict[str, float],
    tz: str | None = None,
) -> None:
    """Populate the Summary sheet with key-value metrics."""
    ws.title = "Summary"
    row = 1
    ws.cell(row=row, column=1, value="BESS Pulse — Market Report").font = Font(
        bold=True, size=14
    )
    row = 3

    dates = price_df.index.tz_convert(tz) if tz else price_df.index
    row = _write_kv_pair(ws, row, "Zone", zone)
    row = _write_kv_pair(ws, row, "Timezone", tz or "UTC")
    row = _write_kv_pair(ws, row, "Date Range Start", str(dates.min().date()))
    row = _write_kv_pair(ws, row, "Date Range End", str(dates.max().date()))
    total_days = (dates.max().date() - dates.min().date()).days + 1
    row = _write_kv_pair(ws, row, "Total Days", total_days)
    row += 1

    row = _write_kv_pair(ws, row, "Avg Price (EUR/MWh)", round(price_df["price_eur_mwh"].mean(), 2), _PRICE_FMT)
    row = _write_kv_pair(ws, row, "Median Price (EUR/MWh)", round(price_df["price_eur_mwh"].median(), 2), _PRICE_FMT)
    row += 1

    row = _write_kv_pair(ws, row, "P50 Spread", round(percentiles["p50"], 2), _PRICE_FMT)
    row = _write_kv_pair(ws, row, "P75 Spread", round(percentiles["p75"], 2), _PRICE_FMT)
    row = _write_kv_pair(ws, row, "P90 Spread", round(percentiles["p90"], 2), _PRICE_FMT)
    row = _write_kv_pair(ws, row, "Mean Spread", round(percentiles["mean"], 2), _PRICE_FMT)
    row += 1

    row = _write_kv_pair(ws, row, "Est. Annual Revenue (EUR/MW)", revenue_estimate["annual_revenue_eur_per_mw"], _PRICE_FMT)
    row = _write_kv_pair(ws, row, "Capture Rate Assumption", revenue_estimate["capture_rate_assumption"])
    row += 1

    row = _write_kv_pair(ws, row, "Negative Price Hours", negative_stats["total_negative_hours"])
    row = _write_kv_pair(ws, row, "Negative Price %", negative_stats["pct_negative"] / 100, _PCT_FMT)
    if negative_stats["total_negative_hours"] > 0:
        row = _write_kv_pair(ws, row, "Avg Negative Price", negative_stats["avg_negative_price"], _PRICE_FMT)
        row = _write_kv_pair(ws, row, "Most Negative Price", negative_stats["most_negative_price"], _PRICE_FMT)

    _auto_column_width(ws)


def _build_table_sheet(ws, title: str, df: pd.DataFrame) -> None:
    """Write a DataFrame as a formatted table sheet."""
    ws.title = title
    headers = list(df.columns)
    _write_header_row(ws, 1, headers)

    for r_idx, row_data in enumerate(df.itertuples(index=False), 2):
        for c_idx, value in enumerate(row_data, 1):
            cell = ws.cell(row=r_idx, column=c_idx)
            if isinstance(value, float):
                cell.value = round(value, 2)
                cell.number_format = _PRICE_FMT
            elif isinstance(value, pd.Timestamp):
                cell.value = str(value)
            else:
                cell.value = value

    _auto_column_width(ws)


def _build_heatmap_sheet(ws, title: str, heatmap: pd.DataFrame) -> None:
    """Write a heatmap matrix to a sheet."""
    ws.title = title
    # Header row: "Hour" + month columns
    headers = ["Hour"] + list(heatmap.columns)
    _write_header_row(ws, 1, headers)

    for r_idx, (hour, row_data) in enumerate(heatmap.iterrows(), 2):
        ws.cell(row=r_idx, column=1, value=hour)
        for c_idx, value in enumerate(row_data, 2):
            cell = ws.cell(row=r_idx, column=c_idx)
            if pd.notna(value):
                cell.value = round(float(value), 2)
                cell.number_format = _PRICE_FMT

    _auto_column_width(ws)


# ── Public API ───────────────────────────────────────────────────────────────

def export_to_excel(
    zone: str,
    price_df: pd.DataFrame,
    daily_spreads: pd.DataFrame,
    monthly_spreads: pd.DataFrame,
    percentiles: dict[str, float],
    revenue_estimate: dict[str, float],
    negative_stats: dict[str, float],
    output_path: Path | None = None,
    tz: str | None = None,
) -> Path:
    """Export all analytics to a formatted Excel workbook.

    Args:
        zone: Bidding zone code.
        price_df: Cleaned price DataFrame.
        daily_spreads: Daily spread DataFrame.
        monthly_spreads: Monthly aggregated spreads.
        percentiles: Spread percentile dict.
        revenue_estimate: Revenue estimate dict.
        negative_stats: Negative price stats dict.
        output_path: Optional output path. Auto-generated if None.
        tz: IANA timezone for local-time date display and heatmap.

    Returns:
        Path to the created .xlsx file.
    """
    wb = Workbook()

    # Sheet 1: Summary
    _build_summary_sheet(
        wb.active, zone, price_df, percentiles, revenue_estimate, negative_stats,
        tz=tz,
    )

    # Sheet 2: Daily Spreads
    _build_table_sheet(wb.create_sheet(), "Daily Spreads", daily_spreads)

    # Sheet 3: Monthly Summary
    _build_table_sheet(wb.create_sheet(), "Monthly Summary", monthly_spreads)

    # Sheet 4: Hourly Prices
    hourly = price_df[["price_eur_mwh"]].reset_index()
    hourly["timestamp"] = hourly["timestamp"].astype(str)
    _build_table_sheet(wb.create_sheet(), "Hourly Prices", hourly)

    # Sheet 5: Price Heatmap (local time)
    heatmap = build_price_heatmap(price_df, tz=tz)
    _build_heatmap_sheet(wb.create_sheet(), "Price Heatmap", heatmap)

    # Determine output path
    if output_path is None:
        dates = price_df.index.tz_convert(tz) if tz else price_df.index
        start_str = dates.min().strftime("%Y%m%d")
        end_str = dates.max().strftime("%Y%m%d")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        output_path = CACHE_DIR / f"{zone}_{start_str}_{end_str}_report.xlsx"

    wb.save(output_path)
    return output_path


def export_to_bytes(
    zone: str,
    price_df: pd.DataFrame,
    daily_spreads: pd.DataFrame,
    monthly_spreads: pd.DataFrame,
    percentiles: dict[str, float],
    revenue_estimate: dict[str, float],
    negative_stats: dict[str, float],
    tz: str | None = None,
) -> bytes:
    """Export to in-memory bytes for Streamlit download button.

    Args:
        Same as export_to_excel, plus tz for local-time display.

    Returns:
        Bytes content of the .xlsx file.
    """
    wb = Workbook()
    _build_summary_sheet(
        wb.active, zone, price_df, percentiles, revenue_estimate, negative_stats,
        tz=tz,
    )
    _build_table_sheet(wb.create_sheet(), "Daily Spreads", daily_spreads)
    _build_table_sheet(wb.create_sheet(), "Monthly Summary", monthly_spreads)

    hourly = price_df[["price_eur_mwh"]].reset_index()
    hourly["timestamp"] = hourly["timestamp"].astype(str)
    _build_table_sheet(wb.create_sheet(), "Hourly Prices", hourly)

    heatmap = build_price_heatmap(price_df, tz=tz)
    _build_heatmap_sheet(wb.create_sheet(), "Price Heatmap", heatmap)

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
