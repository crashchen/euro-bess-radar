"""Excel and PDF report export for eu-bess-pulse."""

from __future__ import annotations

import logging
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from src.analytics import build_price_heatmap
from src.config import CACHE_DIR

logger = logging.getLogger(__name__)


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
    if "total_eur" in revenue_estimate:
        row = _write_kv_pair(ws, row, "Headline Annual Revenue (EUR)", revenue_estimate["total_eur"], _PRICE_FMT)
    if "headline_total_mode" in revenue_estimate:
        row = _write_kv_pair(ws, row, "Headline Total Mode", revenue_estimate["headline_total_mode"])
    if "gross_additive_total_eur" in revenue_estimate:
        row = _write_kv_pair(
            ws, row,
            "Gross Additive Total (Reference, EUR)",
            revenue_estimate["gross_additive_total_eur"],
            _PRICE_FMT,
        )
    if revenue_estimate.get("capacity_stack_warning"):
        row = _write_kv_pair(
            ws, row,
            "Capacity Stack Warning",
            revenue_estimate["capacity_stack_warning"],
        )
    if "source_revenues" in revenue_estimate:
        for source, value in revenue_estimate["source_revenues"].items():
            row = _write_kv_pair(ws, row, f"{source} Revenue (EUR)", value, _PRICE_FMT)
    if "power_mw" in revenue_estimate:
        row = _write_kv_pair(ws, row, "BESS Power (MW)", revenue_estimate["power_mw"])
    if "duration_hours" in revenue_estimate:
        row = _write_kv_pair(ws, row, "BESS Duration (h)", revenue_estimate["duration_hours"])
    if "roundtrip_efficiency" in revenue_estimate:
        row = _write_kv_pair(ws, row, "Round-Trip Efficiency", revenue_estimate["roundtrip_efficiency"], _PCT_FMT)
    row = _write_kv_pair(ws, row, "Modeled Cycles per Day", revenue_estimate["cycles_per_day_assumption"])
    row = _write_kv_pair(ws, row, "Capture Rate Assumption", revenue_estimate["capture_rate_assumption"])
    row += 1

    row = _write_kv_pair(ws, row, "Negative Price Hours", negative_stats["negative_hours"])
    row = _write_kv_pair(
        ws, row, "Negative Price Intervals", negative_stats["negative_intervals"],
    )
    row = _write_kv_pair(
        ws, row, "Negative Price % of Intervals", negative_stats["pct_negative"] / 100,
        _PCT_FMT,
    )
    if negative_stats["negative_intervals"] > 0:
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

def _write_excel_workbook(
    writer: pd.ExcelWriter,
    zone: str,
    price_df: pd.DataFrame,
    daily_spreads: pd.DataFrame,
    monthly_spreads: pd.DataFrame,
    percentiles: dict[str, float],
    revenue_estimate: dict[str, float],
    negative_stats: dict[str, float],
    tz: str | None = None,
) -> None:
    """Populate the workbook opened via pandas ExcelWriter."""
    wb = writer.book
    summary_ws = wb.active if wb.active is not None else wb.create_sheet()

    _build_summary_sheet(
        summary_ws,
        zone,
        price_df,
        percentiles,
        revenue_estimate,
        negative_stats,
        tz=tz,
    )
    _build_table_sheet(wb.create_sheet(), "Daily Spreads", daily_spreads)
    _build_table_sheet(wb.create_sheet(), "Monthly Summary", monthly_spreads)

    hourly = price_df[["price_eur_mwh"]].reset_index()
    hourly["timestamp"] = hourly["timestamp"].astype(str)
    _build_table_sheet(wb.create_sheet(), "Hourly Prices", hourly)

    heatmap = build_price_heatmap(price_df, tz=tz)
    _build_heatmap_sheet(wb.create_sheet(), "Price Heatmap", heatmap)

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
    if output_path is None:
        dates = price_df.index.tz_convert(tz) if tz else price_df.index
        start_str = dates.min().strftime("%Y%m%d")
        end_str = dates.max().strftime("%Y%m%d")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        output_path = CACHE_DIR / f"{zone}_{start_str}_{end_str}_report.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        _write_excel_workbook(
            writer,
            zone,
            price_df,
            daily_spreads,
            monthly_spreads,
            percentiles,
            revenue_estimate,
            negative_stats,
            tz=tz,
        )
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
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _write_excel_workbook(
            writer,
            zone,
            price_df,
            daily_spreads,
            monthly_spreads,
            percentiles,
            revenue_estimate,
            negative_stats,
            tz=tz,
        )
    return buf.getvalue()


# ── PDF export ──────────────────────────────────────────────────────────────


def _render_figure_to_image(
    fig: Any,
    width: int = 1200,
    height: int = 500,
) -> bytes:
    """Convert a Plotly figure to PNG bytes via kaleido."""
    return fig.to_image(format="png", width=width, height=height)


def _build_pdf_report(
    zone: str,
    price_df: pd.DataFrame,
    percentiles: dict[str, float],
    revenue_estimate: dict[str, float],
    negative_stats: dict[str, float],
    tz: str | None = None,
    figures: dict[str, Any] | None = None,
) -> bytes:
    """Lay out a multi-page PDF report using fpdf2."""
    from fpdf import FPDF

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Page 1: Summary ─────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 14, "BESS Pulse - Market Report", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    dates = price_df.index.tz_convert(tz) if tz else price_df.index
    start_str = str(dates.min().date())
    end_str = str(dates.max().date())
    total_days = (dates.max().date() - dates.min().date()).days + 1

    rows: list[tuple[str, str]] = [
        ("Zone", zone),
        ("Timezone", tz or "UTC"),
        ("Date Range", f"{start_str}  to  {end_str}"),
        ("Total Days", str(total_days)),
        ("", ""),
        ("Avg Price (EUR/MWh)", f"{price_df['price_eur_mwh'].mean():.2f}"),
        ("Median Price (EUR/MWh)", f"{price_df['price_eur_mwh'].median():.2f}"),
        ("", ""),
        ("P50 Spread", f"{percentiles['p50']:.2f}"),
        ("P75 Spread", f"{percentiles['p75']:.2f}"),
        ("P90 Spread", f"{percentiles['p90']:.2f}"),
        ("Mean Spread", f"{percentiles['mean']:.2f}"),
        ("", ""),
        ("Est. Annual Revenue (EUR/MW)",
         f"{revenue_estimate['annual_revenue_eur_per_mw']:,.0f}"),
    ]

    if "total_eur" in revenue_estimate:
        rows.append(("Headline Annual Revenue (EUR)",
                      f"{revenue_estimate['total_eur']:,.0f}"))
    if "headline_total_mode" in revenue_estimate:
        rows.append(("Headline Total Mode",
                      str(revenue_estimate["headline_total_mode"])))
    if "gross_additive_total_eur" in revenue_estimate:
        rows.append(("Gross Additive Total (Reference, EUR)",
                      f"{revenue_estimate['gross_additive_total_eur']:,.0f}"))
    if revenue_estimate.get("capacity_stack_warning"):
        rows.append(("Note", str(revenue_estimate["capacity_stack_warning"])))
    if "source_revenues" in revenue_estimate:
        rows.append(("", ""))
        for source, value in revenue_estimate["source_revenues"].items():
            rows.append((f"{source} Revenue (EUR)", f"{value:,.0f}"))

    rows.append(("", ""))
    rows.append(("Modeled Cycles per Day",
                  str(revenue_estimate.get("cycles_per_day_assumption", ""))))
    rows.append(("Capture Rate Assumption",
                  str(revenue_estimate.get("capture_rate_assumption", ""))))
    if "roundtrip_efficiency" in revenue_estimate:
        rows.append(("Round-Trip Efficiency",
                      f"{revenue_estimate['roundtrip_efficiency']:.0%}"))
    rows.append(("", ""))
    rows.append(("Negative Price Hours",
                  str(negative_stats.get("negative_hours", 0))))
    rows.append(("Negative Price Intervals",
                  str(negative_stats.get("negative_intervals", 0))))
    rows.append(("Negative Price % of Intervals",
                  f"{negative_stats.get('pct_negative', 0):.1f}%"))

    col_w = 100
    for label, value in rows:
        if not label and not value:
            pdf.ln(3)
            continue
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w, 7, label)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")

    # ── Chart pages ─────────────────────────────────────────────────────
    if figures:
        chart_order = [
            ("Price Time Series", "price_ts"),
            ("Price Heatmap", "price_heatmap"),
            ("Daily Spread", "spread_ts"),
            ("Revenue Breakdown", "revenue_bar"),
            ("Revenue Waterfall", "revenue_waterfall"),
            ("Spread Heatmap", "spread_heatmap"),
            ("Monthly Seasonality", "monthly_seasonality"),
        ]
        for title, key in chart_order:
            fig = figures.get(key)
            if fig is None:
                continue
            try:
                img_bytes = _render_figure_to_image(fig)
            except Exception:
                logger.warning("Failed to render chart '%s' for PDF", key)
                continue

            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes)
                tmp.flush()
                page_w = pdf.w - pdf.l_margin - pdf.r_margin
                page_h = pdf.h - pdf.get_y() - 15
                img_w = page_w
                img_h = img_w * 500 / 1200
                if img_h > page_h:
                    img_h = page_h
                    img_w = img_h * 1200 / 500
                pdf.image(tmp.name, w=img_w, h=img_h)
                Path(tmp.name).unlink(missing_ok=True)

    return bytes(pdf.output())


def export_to_pdf_bytes(
    zone: str,
    price_df: pd.DataFrame,
    daily_spreads: pd.DataFrame,
    monthly_spreads: pd.DataFrame,
    percentiles: dict[str, float],
    revenue_estimate: dict[str, float],
    negative_stats: dict[str, float],
    tz: str | None = None,
    figures: dict[str, Any] | None = None,
) -> bytes:
    """Export analytics to a PDF report as in-memory bytes.

    Args:
        zone: Bidding zone code.
        price_df: Cleaned price DataFrame.
        daily_spreads: Daily spread DataFrame.
        monthly_spreads: Monthly aggregated spreads.
        percentiles: Spread percentile dict.
        revenue_estimate: Revenue estimate dict.
        negative_stats: Negative price stats dict.
        tz: IANA timezone for local-time date display.
        figures: Optional dict mapping chart keys to Plotly figures.
            Recognized keys: price_ts, price_heatmap, spread_ts,
            revenue_bar, revenue_waterfall, spread_heatmap,
            monthly_seasonality.

    Returns:
        Bytes content of the PDF file.
    """
    return _build_pdf_report(
        zone=zone,
        price_df=price_df,
        percentiles=percentiles,
        revenue_estimate=revenue_estimate,
        negative_stats=negative_stats,
        tz=tz,
        figures=figures,
    )
