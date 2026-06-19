"""BESS Pulse — European BESS Market Screening Dashboard."""

from __future__ import annotations

import logging
import time

import pandas as pd
import streamlit as st

from src.analytics import (
    calculate_daily_dispatch,
    calculate_daily_spreads,
    calculate_monthly_spreads_from_daily,
    calculate_negative_price_hours,
    calculate_spread_percentiles,
    estimate_annual_arbitrage_revenue,
    filter_to_complete_local_days,
)
from src.ancillary import build_ancillary_dataset
from src.assumptions import build_assumptions_table
from src.components.sidebar import (
    _format_data_error,
    load_zone_data,
    render_sidebar,
)
from src.config import get_zone_timezone
from src.data_ingestion import (
    DataSourceAuthError,
    DataSourceNetworkError,
    DataSourceParseError,
    build_zone_query_window,
    read_intraday_cache,
)
from src.export import export_to_bytes, export_to_pdf_bytes
from src.pages import (
    ancillary_services,
    data_trust,
    forward_scenarios,
    heatmaps,
    market_overview,
    renewable_correlation,
    revenue_estimation,
    simulation_cockpit,
    zone_comparison,
)
from src.ui_theme import cockpit_chart_template, inject_global_cockpit_theme

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="BESS Pulse", layout="wide", page_icon="\u26a1")
inject_global_cockpit_theme()

# ── Sidebar ──────────────────────────────────────────────────────────────────
params = render_sidebar()

selected_zones = params["selected_zones"]
start_date = params["start_date"]
end_date = params["end_date"]
power_mw = params["power_mw"]
duration_hours = params["duration_hours"]
efficiency = params["efficiency"]
capture_rate = params["capture_rate"]
capex_eur_kwh = params["capex_eur_kwh"]
use_lp_dispatch = params["use_lp_dispatch"]
force_refresh = params["force_refresh"]
chart_template = cockpit_chart_template()
fetch_btn = params["fetch_btn"]
current_zone_date_scope = params["current_zone_date_scope"]

# ── Data fetching ────────────────────────────────────────────────────────────

if fetch_btn or "zone_data" in st.session_state:
    refresh_token = st.session_state.get("refresh_token", 0)
    if fetch_btn:
        if force_refresh:
            refresh_token = time.time_ns()
        else:
            refresh_token = 0

        zone_data: dict[str, pd.DataFrame] = {}
        progress = st.sidebar.progress(0, text="Fetching...")

        for i, zone in enumerate(selected_zones):
            with st.spinner(f"Fetching data for {zone}..."):
                try:
                    df = load_zone_data(
                        zone,
                        str(start_date),
                        str(end_date),
                        force_refresh=force_refresh,
                        refresh_token=refresh_token,
                    )
                    if not df.empty:
                        zone_data[zone] = df
                    else:
                        st.warning(f"No data returned for {zone}")
                except (DataSourceAuthError, DataSourceNetworkError, DataSourceParseError, ValueError) as exc:
                    logger.warning("Failed to fetch %s", zone, exc_info=exc)
                    st.error(f"Failed to fetch {zone}: {_format_data_error(exc)}")
            progress.progress((i + 1) / len(selected_zones), text=f"Done: {zone}")

        progress.empty()
        st.session_state["zone_data"] = zone_data
        st.session_state["selected_zones"] = selected_zones
        st.session_state["fetched_zone_date_scope"] = current_zone_date_scope
        st.session_state["refresh_token"] = refresh_token
    else:
        zone_data = st.session_state.get("zone_data", {})
        selected_zones = st.session_state.get("selected_zones", selected_zones)
        if st.session_state.get("fetched_zone_date_scope") != current_zone_date_scope:
            zone_data = {}

    if not zone_data:
        st.warning("No data available. Check zone selection and date range.")
        st.stop()

    available_zones = list(zone_data.keys())
    default_zone = (
        selected_zones[0]
        if selected_zones and selected_zones[0] in zone_data
        else available_zones[0]
    )

    if len(available_zones) > 1:
        primary_zone = st.selectbox(
            "Display Zone",
            options=available_zones,
            index=available_zones.index(default_zone),
            help="Select which fetched zone to show in the single-zone tabs below.",
        )
        st.caption(
            "Market Overview, Heatmaps, Revenue Estimation, and Renewable Correlation "
            "show the selected display zone. Zone Comparison uses all fetched zones."
        )
    else:
        primary_zone = default_zone

    # ── Compute analytics for selected display zone ──────────────────────
    primary_df = zone_data[primary_zone]
    zone_tz = get_zone_timezone(primary_zone)

    if use_lp_dispatch:
        daily_spreads = calculate_daily_dispatch(
            primary_df, tz=zone_tz, duration_hours=duration_hours,
            power_mw=power_mw, efficiency=efficiency,
        )
    else:
        daily_spreads = calculate_daily_spreads(
            primary_df, tz=zone_tz, duration_hours=duration_hours,
        )
    monthly_spreads = calculate_monthly_spreads_from_daily(daily_spreads)
    percentiles = calculate_spread_percentiles(daily_spreads)
    # Keep negative-price stats and heatmaps consistent with daily-spread
    # filtering: a day with any NaN price is excluded everywhere.
    complete_df = filter_to_complete_local_days(primary_df, tz=zone_tz)
    neg_stats = calculate_negative_price_hours(complete_df)
    revenue = estimate_annual_arbitrage_revenue(
        daily_spreads,
        power_mw=power_mw,
        duration_hours=duration_hours,
        roundtrip_efficiency=efficiency,
        capture_rate=capture_rate,
    )

    # Collect Plotly figures for PDF export
    report_figures: dict[str, object] = {}

    # Build ancillary dataset for Revenue tab
    stored_ancillary_zone = st.session_state.get("ancillary_zone")
    stored_ancillary_dates = st.session_state.get("ancillary_dates")
    current_ancillary_dates = (str(start_date), str(end_date))
    ancillary_scope_matches = (
        stored_ancillary_zone == primary_zone
        and stored_ancillary_dates == current_ancillary_dates
    )
    manual_anc_df = st.session_state.get("ancillary_df") if ancillary_scope_matches else None
    auto_fetch_results = (
        st.session_state.get("auto_fetch_results", {})
        if ancillary_scope_matches
        else {}
    )
    anc_df = build_ancillary_dataset(manual_anc_df, auto_fetch_results)
    intraday_cache_key = f"intraday_cache::{primary_zone}::{start_date}::{end_date}"
    intraday_df = st.session_state.get(intraday_cache_key)
    if intraday_df is None:
        # Cache-first read so the Simulation Cockpit sees IDA1 prices (live
        # fetch OR manual CSV upload) without first visiting the Revenue tab.
        try:
            _ida_start, _ida_end = build_zone_query_window(
                primary_zone, start_date, end_date,
            )
            intraday_df = read_intraday_cache(
                primary_zone, _ida_start, _ida_end, sequence=1,
            )
            if intraday_df is not None and not intraday_df.empty:
                st.session_state[intraday_cache_key] = intraday_df
        except ValueError:
            intraday_df = None

    export_revenue = revenue.copy()
    export_revenue["power_mw"] = power_mw
    export_revenue["duration_hours"] = duration_hours
    export_revenue["roundtrip_efficiency"] = efficiency

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_names = [
        "Market Overview", "Heatmaps", "Revenue Estimation",
        "Renewable Correlation", "Zone Comparison", "Ancillary Services",
        "Simulation Cockpit", "Forward Scenarios", "Data Trust",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        market_overview.render(
            primary_zone=primary_zone,
            primary_df=primary_df,
            daily_spreads=daily_spreads,
            percentiles=percentiles,
            neg_stats=neg_stats,
            duration_hours=duration_hours,
            zone_tz=zone_tz,
            chart_template=chart_template,
            report_figures=report_figures,
        )

    with tabs[1]:
        heatmaps.render(
            primary_zone=primary_zone,
            primary_df=complete_df,
            duration_hours=duration_hours,
            zone_tz=zone_tz,
            chart_template=chart_template,
            report_figures=report_figures,
        )

    with tabs[2]:
        export_revenue = revenue_estimation.render(
            primary_zone=primary_zone,
            primary_df=primary_df,
            daily_spreads=daily_spreads,
            monthly_spreads=monthly_spreads,
            percentiles=percentiles,
            revenue=revenue,
            start_date=start_date,
            end_date=end_date,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            capture_rate=capture_rate,
            capex_eur_kwh=capex_eur_kwh,
            use_lp_dispatch=use_lp_dispatch,
            zone_tz=zone_tz,
            chart_template=chart_template,
            report_figures=report_figures,
            export_revenue=export_revenue,
            auto_fetch_results=auto_fetch_results,
        )

    with tabs[3]:
        renewable_correlation.render(
            primary_zone=primary_zone,
            primary_df=primary_df,
            start_date=start_date,
            end_date=end_date,
            duration_hours=duration_hours,
            zone_tz=zone_tz,
            chart_template=chart_template,
            refresh_token=refresh_token,
        )

    with tabs[4]:
        zone_comparison.render(
            zone_data=zone_data,
            duration_hours=duration_hours,
            capture_rate=capture_rate,
            efficiency=efficiency,
            power_mw=power_mw,
            use_lp_dispatch=use_lp_dispatch,
            capex_eur_kwh=capex_eur_kwh,
            chart_template=chart_template,
        )

    with tabs[5]:
        ancillary_services.render(
            primary_zone=primary_zone,
            start_date=start_date,
            end_date=end_date,
            anc_df=anc_df,
        )

    with tabs[6]:
        simulation_cockpit.render(
            primary_zone=primary_zone,
            primary_df=primary_df,
            intraday_df=intraday_df,
            anc_df=anc_df,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            capture_rate=capture_rate,
            capex_eur_kwh=capex_eur_kwh,
            zone_tz=zone_tz,
            chart_template=chart_template,
        )

    with tabs[7]:
        forward_scenarios.render(
            zone_data=zone_data,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            capture_rate=capture_rate,
            chart_template=chart_template,
        )

    with tabs[8]:
        _fmode_label = st.session_state.get("forecast_policy_mode")
        _forecast_mode = {
            "Walk-forward": "walk_forward",
            "LOO cross-validation": "loo",
        }.get(_fmode_label)
        _fbucket_label = st.session_state.get("forecast_policy_bucket")
        _forecast_bucket = {
            "Hour-of-week": "hour_of_week",
            "Hour-of-day": "hour_of_day",
        }.get(_fbucket_label)
        assumptions = build_assumptions_table(
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            capture_rate=capture_rate,
            capex_eur_kwh=capex_eur_kwh,
            use_lp_dispatch=use_lp_dispatch,
            deadband_eur_per_mw=st.session_state.get("forecast_policy_deadband"),
            forecast_mode=_forecast_mode,
            forecast_bucket=_forecast_bucket,
        )
        data_trust.render(
            zone_data=zone_data,
            zone_timezones={zone: get_zone_timezone(zone) for zone in zone_data},
            assumptions=assumptions,
        )

    # ── Export button ────────────────────────────────────────────────────
    st.divider()
    xlsx_bytes = export_to_bytes(
        zone=primary_zone,
        price_df=primary_df,
        daily_spreads=daily_spreads,
        monthly_spreads=monthly_spreads,
        percentiles=percentiles,
        revenue_estimate=export_revenue,
        negative_stats=neg_stats,
        tz=zone_tz,
    )
    include_pdf_charts = st.checkbox(
        "Include charts in PDF",
        value=False,
        help=(
            "Chart pages require Kaleido/Chrome static rendering and can be slower "
            "or unavailable on some local machines. Leave off for a stable text-only PDF."
        ),
    )
    pdf_figures = report_figures if include_pdf_charts else None
    if not include_pdf_charts:
        st.caption("PDF export will include the summary tables only. Enable chart export to add Plotly chart pages.")

    pdf_bytes = export_to_pdf_bytes(
        zone=primary_zone,
        price_df=primary_df,
        daily_spreads=daily_spreads,
        monthly_spreads=monthly_spreads,
        percentiles=percentiles,
        revenue_estimate=export_revenue,
        negative_stats=neg_stats,
        tz=zone_tz,
        figures=pdf_figures,
    )
    exp_col1, exp_col2 = st.columns(2)
    exp_col1.download_button(
        label="\U0001f4e5 Export to Excel",
        data=xlsx_bytes,
        file_name=f"{primary_zone}_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    exp_col2.download_button(
        label="\U0001f4c4 Export to PDF",
        data=pdf_bytes,
        file_name=f"{primary_zone}_report.pdf",
        mime="application/pdf",
        width="stretch",
    )

else:
    st.title("\u26a1 BESS Pulse")
    st.markdown(
        "Select one or more bidding zones in the sidebar and click **Fetch Data** "
        "to begin the analysis."
    )
