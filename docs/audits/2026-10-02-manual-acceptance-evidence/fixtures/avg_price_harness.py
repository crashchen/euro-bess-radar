"""Synthetic average-price harness for manual checks 32-34 (2026-10-02).

Run from a clean `git archive e28ee9c` checkout:
  PYTHONPATH=. streamlit run <this file> --server.address 127.0.0.1 --server.port 8614
Production renderers (Market Overview, Zone Comparison) and production
Excel/PDF exporters receive in-memory frames. The analytics chain mirrors
app.py lines 146-232. No market cache is read or written: DB/cache paths are
patched to a scratch directory as a defensive extra.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st

from src import config, data_ingestion, export
from src.analytics import (
    calculate_daily_spreads,
    calculate_monthly_spreads_from_daily,
    calculate_negative_price_hours,
    calculate_spread_percentiles,
    estimate_annual_arbitrage_revenue,
    filter_to_complete_local_days,
)
from src.export import export_to_bytes, export_to_pdf_bytes
from src.pages import market_overview, zone_comparison
from src.ui_theme import cockpit_chart_template, inject_global_cockpit_theme

TZ = "Europe/Berlin"


def _frame(index: pd.DatetimeIndex, prices) -> pd.DataFrame:
    return pd.DataFrame(
        {"price_eur_mwh": np.asarray(prices, dtype=float), "filled": False, "imputed": False},
        index=index.tz_convert("UTC").rename("timestamp"),
    )


def _local(start: str, end: str, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=freq, inclusive="left", tz=TZ)


def fixtures() -> dict[str, pd.DataFrame]:
    hourly = _local("2025-09-01", "2025-10-01", "h")
    quarter = _local("2025-10-01", "2025-10-11", "15min")
    sdac = pd.concat([_frame(hourly, 10.0), _frame(quarter, 100.0)])

    base = _local("2026-03-02", "2026-03-07", "h")
    shape = 50.0 + 30.0 * np.sin(np.arange(len(base)) * 2 * np.pi / 24)
    missing = _frame(base, shape).drop(index=_frame(base, shape).index[30])
    singleton = _frame(base[:1], [42.0])
    two_hour = _frame(_local("2026-03-02", "2026-03-07", "2h"), 55.0)

    partial = shape.copy()
    partial[[5, 6, 7]] = np.nan
    partial[[50, 51]] = np.inf
    partial_frame = _frame(base, partial)
    none_finite = _frame(base, np.where(np.arange(len(base)) % 2, np.nan, np.inf))
    return {
        "32 SDAC 30d hourly@10 + 10d 15min@100": sdac,
        "33a internal missing timestamp": missing,
        "33b singleton": singleton,
        "33c regular 2-hour cadence": two_hour,
        "34a partial NaN/inf (3 NaN + 2 inf of 120)": partial_frame,
        "34b every price non-finite": none_finite,
    }


def main() -> None:
    st.set_page_config(page_title="Avg price harness", layout="wide")
    inject_global_cockpit_theme()
    st.sidebar.header("Synthetic avg-price fixture")
    fx = fixtures()
    name = st.sidebar.radio("Fixture", list(fx))
    primary_df = fx[name]
    zone = "DE_LU"
    other = _frame(_local("2025-09-01", "2025-09-06", "h"), 60.0)
    zone_data = {zone: primary_df, "FR": other}
    template = cockpit_chart_template()
    st.sidebar.caption(f"{len(primary_df)} rows; comparison zone FR = clean hourly @60")

    daily = calculate_daily_spreads(primary_df, tz=TZ, duration_hours=1)
    monthly = calculate_monthly_spreads_from_daily(daily)
    pct = calculate_spread_percentiles(daily)
    complete = filter_to_complete_local_days(primary_df, tz=TZ)
    neg = calculate_negative_price_hours(complete)
    revenue = estimate_annual_arbitrage_revenue(
        daily, power_mw=10, duration_hours=1, roundtrip_efficiency=0.88, capture_rate=0.7,
    )
    export_revenue = revenue.copy()
    export_revenue.update(
        power_mw=10, duration_hours=1, roundtrip_efficiency=0.88,
        capture_basis="sidebar capture haircut applied to screening revenue",
    )
    tab_mo, tab_zc = st.tabs(["Market Overview", "Zone Comparison"])
    with tab_mo:
        market_overview.render(
            primary_zone=zone, primary_df=primary_df, daily_spreads=daily,
            percentiles=pct, neg_stats=neg, duration_hours=1, zone_tz=TZ,
            chart_template=template, report_figures={},
        )
    with tab_zc:
        zone_comparison.render(
            zone_data=zone_data, duration_hours=1, capture_rate=0.7, efficiency=0.88,
            power_mw=10, use_lp_dispatch=False, capex_eur_kwh=0, chart_template=template,
        )
    xlsx = export_to_bytes(
        zone=zone, price_df=primary_df, daily_spreads=daily, monthly_spreads=monthly,
        percentiles=pct, revenue_estimate=export_revenue, negative_stats=neg, tz=TZ,
        project_case_result=None,
    )
    pdf = export_to_pdf_bytes(
        zone=zone, price_df=primary_df, daily_spreads=daily, monthly_spreads=monthly,
        percentiles=pct, revenue_estimate=export_revenue, negative_stats=neg, tz=TZ,
        figures=None,
    )
    c1, c2 = st.columns(2)
    c1.download_button("Export to Excel", xlsx, file_name=f"{zone}_report.xlsx")
    c2.download_button("Export to PDF", pdf, file_name=f"{zone}_report.pdf")


scratch = Path(mkdtemp(prefix="avg-price-harness-"))
with (
    patch.object(config, "CACHE_DIR", scratch),
    patch.object(config, "DB_PATH", scratch / "synthetic.db"),
    patch.object(data_ingestion, "CACHE_DIR", scratch),
    patch.object(data_ingestion, "DB_PATH", scratch / "synthetic.db"),
    patch.object(export, "CACHE_DIR", scratch),
):
    main()
