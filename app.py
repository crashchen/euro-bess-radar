"""BESS Pulse — European BESS Market Screening Dashboard."""

from __future__ import annotations

import logging
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    analyze_renewable_bess_signal,
    build_price_heatmap,
    build_renewable_price_scatter,
    build_spread_heatmap,
    calculate_daily_spreads,
    calculate_monthly_spreads_from_daily,
    calculate_negative_price_hours,
    calculate_spread_percentiles,
    compare_zones,
    estimate_annual_arbitrage_revenue,
)
from src.ancillary import (
    ANCILLARY_TEMPLATES,
    build_ancillary_dataset,
    calculate_ancillary_revenue,
    generate_template_csv,
    merge_revenue_stack,
    parse_ancillary_csv,
)
from src.ancillary_fetchers import get_available_fetchers, run_auto_fetch
from src.config import (
    ALL_ZONES,
    ANCILLARY_CAPACITY_AVAILABILITY,
    ANCILLARY_ENERGY_ACTIVATION_SHARE,
    ZONE_TIMEZONES,
    get_zone_timezone,
    is_elexon_zone,
)
from src.data_ingestion import (
    DataSourceAuthError,
    DataSourceNetworkError,
    DataSourceParseError,
    build_zone_query_window,
    fetch_generation_data,
    fetch_prices,
)
from src.export import export_to_bytes, export_to_pdf_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="BESS Pulse", layout="wide", page_icon="\u26a1")

ANCILLARY_STATE_KEYS = (
    "ancillary_df",
    "ancillary_template",
    "auto_fetch_results",
    "ancillary_zone",
    "ancillary_dates",
)


def _clear_stale_ancillary_state() -> None:
    """Remove ancillary data that no longer matches the active sidebar scope."""
    for key in ANCILLARY_STATE_KEYS:
        st.session_state.pop(key, None)


def _store_ancillary_scope(zone: str, start: object, end: object) -> None:
    """Persist the zone/date scope associated with ancillary data in session state."""
    st.session_state["ancillary_zone"] = zone
    st.session_state["ancillary_dates"] = (str(start), str(end))


def _clear_stale_price_state() -> None:
    """Remove fetched price data when the sidebar scope changes."""
    for key in ("zone_data", "selected_zones", "fetched_zone_date_scope", "refresh_token"):
        st.session_state.pop(key, None)


def _parse_and_store_ancillary_upload(
    uploaded_file,
    template_key: str,
    zone: str,
    start: object,
    end: object,
) -> None:
    """Parse one uploaded ancillary CSV and store it under the active scope."""
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig")
        parsed = parse_ancillary_csv(content, template_key)
        st.session_state["ancillary_df"] = parsed
        st.session_state["ancillary_template"] = template_key
        _store_ancillary_scope(zone, start, end)
        st.success(f"{template_key} loaded: {len(parsed)} rows")
    except UnicodeDecodeError:
        st.error("Parse error: the uploaded file is not valid UTF-8/CSV text.")
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Parse error: {exc}")


def _run_and_store_ancillary_fetch(zone: str, start: object, end: object) -> None:
    """Run the configured ancillary auto-fetchers and store successful results."""
    fetchers = get_available_fetchers(zone)
    with st.spinner(f"Fetching from {', '.join(f['source'] for f in fetchers)}..."):
        auto_start, auto_end = build_zone_query_window(zone, start, end)
        results = run_auto_fetch(zone, auto_start, auto_end)
        if results:
            st.session_state["auto_fetch_results"] = results
            _store_ancillary_scope(zone, start, end)
            st.success(f"Fetched {len(results)} dataset(s): " + ", ".join(results.keys()))
        else:
            st.warning("No data returned. Try manual CSV upload instead.")

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("BESS Pulse")
st.sidebar.caption("European BESS Market Screening")

# Zone selector
zone_options = {v: f"{k} ({v})" for k, v in ALL_ZONES.items()}
selected_zones = st.sidebar.multiselect(
    "Bidding Zones",
    options=list(zone_options.keys()),
    default=["DE_LU"],
    format_func=lambda x: zone_options[x],
)

# Date range (inclusive local calendar dates per selected bidding zone)
col1, col2 = st.sidebar.columns(2)
default_end = pd.Timestamp.now().normalize()
default_start = default_end - pd.Timedelta(days=30)
start_date = col1.date_input("Start", value=default_start.date())
end_date = col2.date_input("End", value=default_end.date())
primary_zone_for_fetch = selected_zones[0] if selected_zones else "DE_LU"

current_zone_date_scope = (tuple(selected_zones), str(start_date), str(end_date))
previous_zone_date_scope = st.session_state.get("zone_date_scope")
if previous_zone_date_scope is not None and previous_zone_date_scope != current_zone_date_scope:
    _clear_stale_ancillary_state()
    _clear_stale_price_state()
st.session_state["zone_date_scope"] = current_zone_date_scope

# BESS parameters — wrapped in a form so slider/input changes don't trigger
# a full Streamlit re-run until the user clicks "Apply".
st.sidebar.subheader("BESS Parameters")
with st.sidebar.form("bess_params"):
    power_mw = st.number_input("Power (MW)", min_value=0.1, value=10.0, step=1.0)
    duration_hours = st.selectbox("Duration (h)", [1, 2, 4], index=0)
    efficiency = st.slider("Efficiency (%)", 85, 95, 88) / 100.0
    capture_rate = st.slider("Capture (%)", 30, 100, 70) / 100.0
    capex_eur_kwh = st.number_input(
        "CapEx (EUR/kWh)",
        min_value=0.0, value=0.0, step=50.0,
        help="Enter installed CapEx to calculate payback period. Leave 0 to skip.",
    )
    st.form_submit_button("Apply", type="primary")
force_refresh = st.sidebar.checkbox(
    "Force Refresh",
    value=False,
    help="Bypass Streamlit and local price caches for the next fetch only.",
)

# Chart theme
chart_template = st.sidebar.selectbox(
    "Chart Theme", ["plotly_dark", "plotly_white", "plotly"], index=0,
)

# ── Ancillary services upload ────────────────────────────────────────────────

with st.sidebar.expander("Ancillary Services Data"):
    anc_template = st.selectbox(
        "Template", list(ANCILLARY_TEMPLATES.keys()),
        format_func=lambda k: f"{k} — {ANCILLARY_TEMPLATES[k]['description'][:40]}",
    )
    anc_file = st.file_uploader("Upload CSV", type=["csv"], key="anc_upload")

    if anc_file is not None:
        if st.button("Parse & Import"):
            _parse_and_store_ancillary_upload(
                anc_file,
                anc_template,
                primary_zone_for_fetch,
                start_date,
                end_date,
            )

    if "ancillary_df" in st.session_state:
        st.caption(f"Loaded: {st.session_state['ancillary_template']} "
                   f"({len(st.session_state['ancillary_df'])} rows)")

    # Template download
    tmpl_key = st.selectbox(
        "Download template",
        list(ANCILLARY_TEMPLATES.keys()),
        key="tmpl_select",
    )
    st.download_button(
        label=f"\U0001f4e5 Download {tmpl_key} template",
        data=generate_template_csv(tmpl_key),
        file_name=f"{tmpl_key}_template.csv",
        mime="text/csv",
        key="tmpl_download",
    )


# ── Auto-fetch ancillary data ───────────────────────────────────────────────

with st.sidebar.expander("Auto-Fetch Ancillary Data"):
    fetchers = get_available_fetchers(primary_zone_for_fetch)

    if fetchers:
        st.markdown("**Available for this zone:**")
        for f in fetchers:
            st.caption(f"\U0001f4e1 {f['name']} ({f['source']})")
        if st.button(
            f"\u26a1 Fetch ancillary data for {primary_zone_for_fetch}",
            key="auto_fetch_ancillary",
        ):
            _run_and_store_ancillary_fetch(primary_zone_for_fetch, start_date, end_date)
    else:
        st.info("No auto-fetch available for this zone. Use CSV upload.")


fetch_btn = st.sidebar.button("Fetch Data", type="primary", width="stretch")


# ── Data fetching ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_zone_data(
    zone: str,
    start: str,
    end: str,
    force_refresh: bool = False,
    refresh_token: int = 0,
) -> pd.DataFrame:
    """Fetch price data for a zone (cached by Streamlit)."""
    del refresh_token
    api_start, api_end = build_zone_query_window(zone, start, end)
    return fetch_prices(
        zone=zone,
        start=api_start,
        end=api_end,
        use_cache=not force_refresh,
    )


@st.cache_data(show_spinner=False)
def load_generation(zone: str, start: str, end: str, refresh_token: int = 0) -> pd.DataFrame:
    """Fetch generation data for a zone (cached by Streamlit)."""
    del refresh_token
    api_start, api_end = build_zone_query_window(zone, start, end)
    return fetch_generation_data(
        zone=zone,
        start=api_start,
        end=api_end,
    )


def _format_data_error(exc: Exception) -> str:
    """Convert internal fetch errors into concise user-facing messages."""
    if isinstance(exc, DataSourceAuthError):
        return str(exc)
    if isinstance(exc, DataSourceNetworkError):
        return str(exc)
    if isinstance(exc, DataSourceParseError):
        return str(exc)
    if isinstance(exc, ValueError):
        return str(exc)
    return "Unexpected data fetch failure. Check logs for details."


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

    # ── Compute analytics for selected display zone ──────────────────────────
    primary_df = zone_data[primary_zone]
    zone_tz = get_zone_timezone(primary_zone)

    daily_spreads = calculate_daily_spreads(
        primary_df, tz=zone_tz, duration_hours=duration_hours,
    )
    monthly_spreads = calculate_monthly_spreads_from_daily(daily_spreads)
    percentiles = calculate_spread_percentiles(daily_spreads)
    neg_stats = calculate_negative_price_hours(primary_df)
    revenue = estimate_annual_arbitrage_revenue(
        daily_spreads,
        power_mw=power_mw,
        duration_hours=duration_hours,
        roundtrip_efficiency=efficiency,
        capture_rate=capture_rate,
    )

    # Collect Plotly figures for PDF export
    report_figures: dict[str, object] = {}

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_names = [
        "Market Overview", "Heatmaps", "Revenue Estimation",
        "Renewable Correlation", "Zone Comparison", "Ancillary Services",
    ]
    tabs = st.tabs(tab_names)

    # ── Tab 1: Market Overview ───────────────────────────────────────────────
    with tabs[0]:
        st.subheader(f"Market Overview — {primary_zone}")
        source = "Elexon (GBP\u2192EUR)" if is_elexon_zone(primary_zone) else "ENTSO-E"
        st.caption(f"Data source: {source}")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg Price", f"\u20ac{primary_df['price_eur_mwh'].mean():.2f}/MWh")
        k2.metric("Avg Ordered Spread", f"\u20ac{percentiles['mean']:.2f}/MWh")
        k3.metric("P90 Ordered Spread", f"\u20ac{percentiles['p90']:.2f}/MWh")
        k4.metric(
            "Neg Price Hours",
            f"{neg_stats['negative_hours']:.1f}h",
            delta=f"{neg_stats['pct_negative']:.1f}% of intervals",
        )
        st.caption(
            f"Spreads use chronology-aware {duration_hours}h charge/discharge windows "
            f"in {zone_tz}."
        )

        price_plot_df = primary_df.reset_index()
        fig_price = px.line(
            price_plot_df,
            x="timestamp", y="price_eur_mwh",
            title="Day-Ahead Prices",
            labels={"price_eur_mwh": "EUR/MWh", "timestamp": ""},
            template=chart_template,
        )
        fig_price.update_traces(opacity=0.4, name="Hourly", showlegend=True)
        ma_series = primary_df["price_eur_mwh"].rolling(
            window=24 * 30, min_periods=24,
        ).mean()
        fig_price.add_scatter(
            x=primary_df.index, y=ma_series,
            mode="lines", name="30-Day MA",
            line=dict(color="#E74C3C", width=2),
        )
        fig_price.update_xaxes(rangeslider_visible=True)
        report_figures["price_ts"] = fig_price
        st.plotly_chart(fig_price, width="stretch")

        spread_plot_df = daily_spreads.copy()
        spread_plot_df["date"] = pd.to_datetime(spread_plot_df["date"])

        fig_spread = px.bar(
            spread_plot_df,
            x="date", y="spread",
            title=f"Daily Ordered Spread ({duration_hours}h windows)",
            labels={"spread": "EUR/MWh", "date": ""},
            color="spread",
            color_continuous_scale="Viridis",
            template=chart_template,
        )
        fig_spread.update_xaxes(rangeslider_visible=True, type="date")
        report_figures["spread_ts"] = fig_spread
        st.plotly_chart(fig_spread, width="stretch")

    # ── Tab 2: Heatmaps ─────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader(f"Heatmaps — {primary_zone}")

        price_hm = build_price_heatmap(primary_df, tz=zone_tz)
        fig_phm = px.imshow(
            price_hm,
            title=f"Average Price by Hour & Month ({zone_tz})",
            labels=dict(x="Month", y="Hour (local)", color="EUR/MWh"),
            color_continuous_scale="Viridis",
            aspect="auto",
            template=chart_template,
        )
        fig_phm.update_yaxes(dtick=1)
        report_figures["price_heatmap"] = fig_phm
        st.plotly_chart(fig_phm, width="stretch")

        spread_hm = build_spread_heatmap(
            primary_df, tz=zone_tz, duration_hours=duration_hours,
        )
        fig_shm = px.imshow(
            spread_hm,
            title=f"Selected-Window Spread Signal ({duration_hours}h windows, {zone_tz})",
            labels=dict(x="Month", y="Hour (local)", color="Signed signal (EUR/MWh)"),
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            aspect="auto",
            template=chart_template,
        )
        fig_shm.update_yaxes(dtick=1)
        report_figures["spread_heatmap"] = fig_shm
        st.plotly_chart(fig_shm, width="stretch")
        st.caption(
            "Negative hours mark windows selected for charging; positive hours mark "
            "windows selected for discharging. Color intensity shows the average signed "
            "ordered-spread signal assigned to those selected windows, not per-hour revenue attribution."
        )

        # Charge/Discharge hour frequency
        charge_freq = (spread_hm < 0).sum(axis=1)
        discharge_freq = (spread_hm > 0).sum(axis=1)
        total_months = spread_hm.shape[1]
        freq_df = pd.DataFrame({
            "Hour": range(24),
            "Charge %": (charge_freq.values / total_months * 100).round(1),
            "Discharge %": (discharge_freq.values / total_months * 100).round(1),
        })
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Bar(
            x=freq_df["Hour"], y=freq_df["Charge %"],
            name="Charge", marker_color="#3498DB",
        ))
        fig_freq.add_trace(go.Bar(
            x=freq_df["Hour"], y=freq_df["Discharge %"],
            name="Discharge", marker_color="#E74C3C",
        ))
        fig_freq.update_layout(
            title=f"Charge/Discharge Hour Selection Frequency ({duration_hours}h windows)",
            xaxis_title="Hour (local)",
            yaxis_title="% of months selected",
            barmode="group",
            template=chart_template,
        )
        st.plotly_chart(fig_freq, width="stretch")

    # ── Tab 3: Revenue Estimation ────────────────────────────────────────────
    with tabs[2]:
        st.subheader(f"Revenue Estimation — {primary_zone}")
        sample_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
        if sample_days < 365:
            st.warning(
                f"Annualisation note: this revenue is extrapolated from the selected "
                f"sample window ({sample_days} days, {start_date} to {end_date}). "
                "For market screening, prefer at least 12 months of data to reduce "
                "seasonality bias."
            )
        else:
            st.caption(
                f"Annualisation note: this revenue is extrapolated from the selected "
                f"sample window ({sample_days} days)."
            )

        with st.expander("How ancillary works"):
            st.markdown(
                """
                This ancillary layer is a screening model that sits on top of day-ahead arbitrage.

                - You can load ancillary data by manual CSV upload or zone-specific auto-fetch.
                - If both are present, manual uploads override auto-fetched rows for the same product name only.
                - Reserve products are kept separate where possible, for example `FCR-N`, `FCR-D Up`, `FCR-D Down`, `aFRR Up`, and `aFRR Down`.
                - Capacity-style products are annualised from average `EUR/MW` prices using the selected BESS power and a fixed `{ANCILLARY_CAPACITY_AVAILABILITY:.0%}` availability assumption.
                - Explicit single-sided energy prices are annualised using the selected BESS power and a simplified `{ANCILLARY_ENERGY_ACTIVATION_SHARE:.0%}` activation-hours assumption.
                - Two-sided balancing or system-price signals, such as GB `system buy` and `system sell`, are stored and shown but are not auto-monetised because dispatch direction and activation volume are still unknown.
                - The revenue stack chart shows `DA Arbitrage` plus each ancillary product as separate colored components.
                - This output is intended for market screening and prioritisation, not as a dispatch-grade settlement model.
                """
            )

        # Check for ancillary data
        stored_ancillary_zone = st.session_state.get("ancillary_zone")
        stored_ancillary_dates = st.session_state.get("ancillary_dates")
        current_ancillary_dates = (str(start_date), str(end_date))
        ancillary_scope_matches = (
            stored_ancillary_zone == primary_zone
            and stored_ancillary_dates == current_ancillary_dates
        )
        ancillary_scope_mismatch = (
            (st.session_state.get("ancillary_df") is not None)
            or bool(st.session_state.get("auto_fetch_results"))
        ) and not ancillary_scope_matches

        manual_anc_df = st.session_state.get("ancillary_df") if ancillary_scope_matches else None
        auto_fetch_results = (
            st.session_state.get("auto_fetch_results", {})
            if ancillary_scope_matches
            else {}
        )
        anc_df = build_ancillary_dataset(manual_anc_df, auto_fetch_results)
        anc_rev = None
        stack = None
        anc_source = None
        export_revenue = revenue.copy()
        export_revenue["power_mw"] = power_mw
        export_revenue["duration_hours"] = duration_hours
        export_revenue["roundtrip_efficiency"] = efficiency

        if ancillary_scope_mismatch:
            st.info(
                "Ancillary data was loaded for a different zone/window. Re-fetch or "
                "re-upload to include ancillary revenue."
            )

        if anc_df is not None and not anc_df.empty:
            anc_rev = calculate_ancillary_revenue(anc_df, power_mw, duration_hours)
            stack = merge_revenue_stack(revenue, anc_rev, power_mw=power_mw)
            export_revenue.update(stack)
            export_revenue["annual_revenue_eur"] = stack["total_eur"]
            export_revenue["annual_revenue_eur_per_mw"] = stack["total_per_mw"]
            export_revenue["da_only_annual_revenue_eur"] = revenue["annual_revenue_eur"]
            export_revenue["da_only_annual_revenue_eur_per_mw"] = revenue[
                "annual_revenue_eur_per_mw"
            ]
            if manual_anc_df is not None and not manual_anc_df.empty:
                anc_source = "manual upload"
            elif auto_fetch_results:
                anc_source = f"auto-fetch ({len(auto_fetch_results)} dataset(s))"

        r1, r2, r3 = st.columns(3)
        if stack:
            r1.metric(
                "Headline Annual Revenue",
                f"\u20ac{stack['total_eur']:,.0f}",
                help=stack.get("headline_total_mode", "combined screening total"),
            )
            r2.metric("DA Arbitrage", f"\u20ac{stack['da_arbitrage_eur']:,.0f}",
                       delta=f"{stack['da_pct']:.0f}% of gross reference")
            r3.metric(
                "Ancillary Standalone",
                f"\u20ac{stack['standalone_ancillary_eur']:,.0f}",
                delta=f"{stack['ancillary_pct']:.0f}% of gross reference",
            )

            if stack.get("capacity_stack_warning"):
                st.warning(stack["capacity_stack_warning"])
                st.caption(
                    f"Gross additive reference, not co-optimized: "
                    f"\u20ac{stack['gross_additive_total_eur']:,.0f}/yr."
                )

            component_rows = [
                {
                    "Source": source,
                    "Standalone Annual Revenue (EUR)": value,
                    "Revenue Type": stack.get("product_revenue_types", {}).get(source, "energy")
                    if source != "DA Arbitrage" else "DA",
                }
                for source, value in stack["source_revenues"].items()
                if value > 0
            ]
            if component_rows:
                st.table(pd.DataFrame(component_rows))

            fig_stack = go.Figure()
            palette = px.colors.qualitative.Bold + px.colors.qualitative.Safe
            for idx, row in enumerate(component_rows):
                fig_stack.add_trace(go.Bar(
                    name=row["Source"],
                    y=["Annual Revenue"],
                    x=[row["Standalone Annual Revenue (EUR)"]],
                    orientation="h",
                    marker_color=palette[idx % len(palette)],
                ))
            fig_stack.update_layout(
                barmode="stack",
                title="Revenue Stack by Product",
                template=chart_template,
                xaxis_title="EUR",
                yaxis_title="",
                legend_title_text="Source",
            )
            report_figures["revenue_bar"] = fig_stack
            st.plotly_chart(fig_stack, width="stretch")
            if anc_source:
                st.caption(f"Ancillary valuation source: {anc_source}")
            if auto_fetch_results and anc_rev["total_ancillary_eur"] == 0:
                st.info(
                    "Auto-fetched balancing/system-price datasets are loaded, but the current "
                    "model only monetises explicit capacity prices and single-sided energy prices."
                )
        else:
            r1.metric(
                "Est. Annual Revenue",
                f"\u20ac{revenue['annual_revenue_eur']:,.0f}",
                help=(
                    f"At {power_mw} MW, {duration_hours}h, {efficiency*100:.0f}% eff, "
                    f"{revenue['cycles_per_day_assumption']:.1f} modeled cycle/day, "
                    f"{revenue['capture_rate_assumption']:.0%} capture"
                ),
            )
            r2.metric("Revenue per MW", f"\u20ac{revenue['annual_revenue_eur_per_mw']:,.0f}/MW/yr")
            r3.metric("Avg Daily Revenue", f"\u20ac{revenue['avg_daily_revenue']:,.0f}")
            st.info("Upload or auto-fetch ancillary services data to see the full revenue stack.")

        # CapEx / Payback
        if capex_eur_kwh > 0:
            total_capex = capex_eur_kwh * power_mw * duration_hours * 1000
            annual_rev = (
                stack["total_eur"] if stack
                else revenue["annual_revenue_eur"]
            )
            payback_years = total_capex / annual_rev if annual_rev > 0 else float("inf")
            st.divider()
            p1, p2, p3 = st.columns(3)
            p1.metric("Total CapEx", f"\u20ac{total_capex:,.0f}")
            p2.metric("Annual Revenue", f"\u20ac{annual_rev:,.0f}")
            p3.metric(
                "Simple Payback",
                f"{payback_years:.1f} years" if payback_years < 100 else "N/A",
            )

        # Revenue waterfall
        st.divider()
        theoretical_spread = percentiles["mean"]
        eff_loss = theoretical_spread * (1 - efficiency)
        post_eff = theoretical_spread - eff_loss
        capture_loss = post_eff * (1 - capture_rate)
        realized_spread = post_eff - capture_loss
        wf_measures = ["relative", "relative", "total", "relative", "total"]
        wf_labels = [
            "Avg Ordered Spread",
            "Efficiency Loss",
            "Post-Efficiency",
            "Capture Discount",
            "Realized Spread",
        ]
        wf_values = [
            theoretical_spread,
            -eff_loss,
            post_eff,
            -capture_loss,
            realized_spread,
        ]
        fig_wf = go.Figure(go.Waterfall(
            x=wf_labels,
            y=wf_values,
            measure=wf_measures,
            connector={"line": {"color": "rgba(150,150,150,0.4)"}},
            decreasing={"marker": {"color": "#E74C3C"}},
            increasing={"marker": {"color": "#2ECC71"}},
            totals={"marker": {"color": "#2E86C1"}},
            textposition="outside",
            text=[f"\u20ac{v:+.1f}" if m == "relative" else f"\u20ac{v:.1f}"
                  for v, m in zip(wf_values, wf_measures)],
        ))
        fig_wf.update_layout(
            title="Revenue Attribution Waterfall (EUR/MWh per cycle)",
            template=chart_template,
            yaxis_title="EUR/MWh",
            showlegend=False,
        )
        report_figures["revenue_waterfall"] = fig_wf
        st.plotly_chart(fig_wf, width="stretch")

        # Sensitivity table
        st.markdown("**Duration Sensitivity (per MW)**")
        sens_rows = []
        for dur in [1, 2, 4]:
            spreads_d = calculate_daily_spreads(
                primary_df, tz=zone_tz, duration_hours=dur,
            )
            rev_d = estimate_annual_arbitrage_revenue(
                spreads_d, power_mw=1.0, duration_hours=dur,
                roundtrip_efficiency=efficiency,
                capture_rate=capture_rate,
            )
            sens_rows.append({
                "Duration (h)": dur,
                "Annual Revenue (EUR/MW)": f"\u20ac{rev_d['annual_revenue_eur_per_mw']:,.0f}",
                "Avg Daily (EUR/MW)": f"\u20ac{rev_d['avg_daily_revenue']:,.0f}",
            })
        st.table(pd.DataFrame(sens_rows))

        # Spread distribution
        fig_hist = px.histogram(
            daily_spreads, x="spread", nbins=30,
            title="Ordered Spread Distribution",
            labels={"spread": "Daily Ordered Spread (EUR/MWh)", "count": "Days"},
            template=chart_template,
        )
        fig_hist.add_vline(x=percentiles["p50"], line_dash="dash",
                           annotation_text=f"P50: {percentiles['p50']:.1f}")
        fig_hist.add_vline(x=percentiles["p90"], line_dash="dash", line_color="red",
                           annotation_text=f"P90: {percentiles['p90']:.1f}")
        st.plotly_chart(fig_hist, width="stretch")

        # Monthly revenue seasonality & volatility
        if not monthly_spreads.empty and len(monthly_spreads) >= 2:
            st.divider()
            st.markdown("**Monthly Revenue Seasonality & Risk**")

            monthly_rev = monthly_spreads["avg_spread"].values
            spread_cv = float(monthly_rev.std() / monthly_rev.mean()) if monthly_rev.mean() > 0 else 0
            best_month = monthly_spreads.loc[monthly_spreads["avg_spread"].idxmax()]
            worst_month = monthly_spreads.loc[monthly_spreads["avg_spread"].idxmin()]
            zero_spread_days = int((daily_spreads["spread"] <= 0).sum())

            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Spread CV", f"{spread_cv:.2f}",
                       help="Coefficient of variation — lower = more stable revenue")
            v2.metric("Best Month", f"{best_month['year_month']}",
                       delta=f"\u20ac{best_month['avg_spread']:.1f}/MWh")
            v3.metric("Worst Month", f"{worst_month['year_month']}",
                       delta=f"\u20ac{worst_month['avg_spread']:.1f}/MWh",
                       delta_color="inverse")
            v4.metric("Zero-Spread Days", f"{zero_spread_days}",
                       delta=f"{zero_spread_days / len(daily_spreads) * 100:.0f}% of total",
                       delta_color="inverse")

            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly_spreads["year_month"],
                y=monthly_spreads["avg_spread"],
                name="Avg Spread",
                marker_color="#2E86C1",
            ))
            fig_monthly.add_trace(go.Scatter(
                x=monthly_spreads["year_month"],
                y=monthly_spreads["max_spread"],
                mode="markers+lines",
                name="Max Spread",
                marker=dict(color="#2ECC71", size=6),
                line=dict(dash="dot"),
            ))
            fig_monthly.add_trace(go.Scatter(
                x=monthly_spreads["year_month"],
                y=monthly_spreads["min_spread"],
                mode="markers+lines",
                name="Min Spread",
                marker=dict(color="#E74C3C", size=6),
                line=dict(dash="dot"),
            ))
            fig_monthly.update_layout(
                title=f"Monthly Spread Breakdown ({duration_hours}h windows)",
                template=chart_template,
                xaxis_title="Month",
                yaxis_title="EUR/MWh",
                legend=dict(orientation="h", y=-0.15),
            )
            report_figures["monthly_seasonality"] = fig_monthly
            st.plotly_chart(fig_monthly, width="stretch")

    # ── Tab 4: Renewable Correlation ─────────────────────────────────────────
    with tabs[3]:
        st.subheader(f"Renewable Correlation — {primary_zone}")
        st.caption(
            "This page asks whether renewables compress hourly prices and whether "
            "high-renewable days widen capturable ordered spreads. Price diagnostics "
            "use hourly renewable share; spread diagnostics use local-day average "
            "renewable share."
        )

        try:
            with st.spinner("Fetching generation data..."):
                gen_df = load_generation(
                    primary_zone,
                    str(start_date),
                    str(end_date),
                    refresh_token=refresh_token,
                )
        except (DataSourceAuthError, DataSourceNetworkError, DataSourceParseError, ValueError) as exc:
            logger.warning("Failed to fetch generation data for %s", primary_zone, exc_info=exc)
            st.error(f"Failed to fetch generation data: {_format_data_error(exc)}")
            gen_df = pd.DataFrame()

        if gen_df.empty:
            st.info("Generation data not available for this zone.")
        else:
            signal = analyze_renewable_bess_signal(
                primary_df,
                gen_df,
                tz=zone_tz,
                duration_hours=duration_hours,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "RE-Price Corr",
                (
                    f"{signal['correlation_renewable_price']:.3f}"
                    if signal["correlation_renewable_price"] is not None
                    else "N/A"
                ),
            )
            c2.metric(
                "Avg Price (High RE Hours)",
                (
                    f"\u20ac{signal['avg_price_high_renewable']:.1f}"
                    if signal["avg_price_high_renewable"] is not None
                    else "N/A"
                ),
            )
            c3.metric(
                "Avg Price (Low RE Hours)",
                (
                    f"\u20ac{signal['avg_price_low_renewable']:.1f}"
                    if signal["avg_price_low_renewable"] is not None
                    else "N/A"
                ),
            )
            c4.metric(
                "Spread Uplift (High RE Days vs Low RE Days)",
                (
                    f"\u20ac{signal['spread_uplift_high_vs_low_renewable']:.1f}/MWh"
                    if signal["spread_uplift_high_vs_low_renewable"] is not None
                    else "N/A"
                ),
            )

            if (
                signal["correlation_renewable_price"] is None
                and signal["spread_uplift_high_vs_low_renewable"] is None
            ):
                st.info(
                    "There is not enough overlapping price and generation data to build "
                    "reliable renewable diagnostics for this sample window."
                )
            elif signal["spread_uplift_high_vs_low_renewable"] is None:
                st.info(
                    "Hourly renewable diagnostics are available, but there are not enough "
                    "local-day observations to evaluate spread uplift by renewable quartile."
                )
            elif signal["correlation_renewable_price"] is None:
                st.info(
                    "Daily renewable-spread diagnostics are available, but hourly price vs "
                    "renewable overlap is too sparse for a robust correlation view."
                )

            # Scatter plot
            scatter_df = build_renewable_price_scatter(primary_df, gen_df, tz=zone_tz)
            if signal["hourly_points"] >= 10 and not scatter_df.empty:
                fig_scatter = px.scatter(
                    scatter_df.reset_index(),
                    x="renewable_pct", y="price_eur_mwh",
                    color="hour",
                    title="Renewable % vs Price",
                    labels={"renewable_pct": "Renewable %", "price_eur_mwh": "EUR/MWh"},
                    color_continuous_scale="Viridis",
                    opacity=0.5,
                    template=chart_template,
                )
                st.plotly_chart(fig_scatter, width="stretch")
            else:
                st.info("Not enough hourly overlap to show the renewable-vs-price scatter.")

            qcol1, qcol2 = st.columns(2)
            with qcol1:
                if signal["price_by_renewable_quartile"]:
                    q_df = pd.DataFrame([
                        {
                            "Quartile": quartile,
                            "Avg Price (EUR/MWh)": signal["price_by_renewable_quartile"][quartile],
                        }
                        for quartile in ["Q1", "Q2", "Q3", "Q4"]
                    ])
                    fig_price_q = px.bar(
                        q_df,
                        x="Quartile",
                        y="Avg Price (EUR/MWh)",
                        title="Average Price by Renewable Quartile",
                        color="Avg Price (EUR/MWh)",
                        color_continuous_scale="RdYlGn_r",
                        template=chart_template,
                    )
                    st.plotly_chart(fig_price_q, width="stretch")
                else:
                    st.info("Not enough hourly renewable variation to build price quartiles.")

            with qcol2:
                if signal["spread_by_renewable_quartile"]:
                    spread_q_df = pd.DataFrame([
                        {
                            "Quartile": quartile,
                            "Avg Ordered Spread (EUR/MWh)": (
                                signal["spread_by_renewable_quartile"][quartile]
                            ),
                        }
                        for quartile in ["Q1", "Q2", "Q3", "Q4"]
                    ])
                    fig_spread_q = px.bar(
                        spread_q_df,
                        x="Quartile",
                        y="Avg Ordered Spread (EUR/MWh)",
                        title=f"Average Ordered Spread by Daily Renewable Quartile ({duration_hours}h)",
                        color="Avg Ordered Spread (EUR/MWh)",
                        color_continuous_scale="Viridis",
                        template=chart_template,
                    )
                    st.plotly_chart(fig_spread_q, width="stretch")
                else:
                    st.info(
                        "Not enough daily renewable variation to build spread quartiles."
                    )

    # ── Tab 5: Zone Comparison ───────────────────────────────────────────────
    with tabs[4]:
        if len(zone_data) < 2:
            st.info("Select multiple zones to see comparison.")
        else:
            st.subheader("Zone Comparison")
            comp = compare_zones(
                zone_data,
                zone_timezones=ZONE_TIMEZONES,
                duration_hours=duration_hours,
                capture_rate=capture_rate,
                roundtrip_efficiency=efficiency,
            )
            if comp.empty:
                st.warning(
                    "No comparison rows could be built from the fetched zone data."
                )
            else:
                comp = comp.sort_values(
                    "estimated_annual_revenue_per_mw", ascending=False,
                ).reset_index(drop=True)

                # Risk/Reward scatter
                fig_rr = px.scatter(
                    comp,
                    x="p90_spread",
                    y="estimated_annual_revenue_per_mw",
                    size="negative_pct",
                    text="zone",
                    title="Zone Screening: Risk/Reward Frontier",
                    labels={
                        "p90_spread": "P90 Spread (EUR/MWh)",
                        "estimated_annual_revenue_per_mw": "Est. Annual Revenue (EUR/MW/yr)",
                        "negative_pct": "Negative Price %",
                    },
                    template=chart_template,
                    size_max=40,
                )
                fig_rr.update_traces(textposition="top center")
                st.plotly_chart(fig_rr, width="stretch")

                # Numeric table (keep sortable)
                st.dataframe(
                    comp,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "zone": "Zone",
                        "avg_price": st.column_config.NumberColumn(
                            "Avg Price", format="\u20ac%.2f",
                        ),
                        "std_price": st.column_config.NumberColumn(
                            "Std Dev", format="%.2f",
                        ),
                        "avg_spread": st.column_config.NumberColumn(
                            "Avg Spread", format="\u20ac%.2f",
                        ),
                        "p50_spread": st.column_config.NumberColumn(
                            "P50 Spread", format="\u20ac%.2f",
                        ),
                        "p90_spread": st.column_config.NumberColumn(
                            "P90 Spread", format="\u20ac%.2f",
                        ),
                        "negative_pct": st.column_config.NumberColumn(
                            "Neg Price %", format="%.1f%%",
                        ),
                        "estimated_annual_revenue_per_mw": st.column_config.NumberColumn(
                            "Revenue (EUR/MW/yr)", format="\u20ac%,.0f",
                        ),
                    },
                )

                all_daily = []
                for zone, df in zone_data.items():
                    ds = calculate_daily_spreads(
                        df,
                        tz=get_zone_timezone(zone),
                        duration_hours=duration_hours,
                    )
                    if ds.empty:
                        continue
                    ds["zone"] = zone
                    all_daily.append(ds)

                if all_daily:
                    combined = pd.concat(all_daily, ignore_index=True)
                    fig_comp = px.line(
                        combined, x="date", y="spread", color="zone",
                        title=f"Daily Ordered Spread Comparison ({duration_hours}h windows)",
                        labels={"spread": "EUR/MWh", "date": ""},
                        template=chart_template,
                    )
                    st.plotly_chart(fig_comp, width="stretch")

    # ── Tab 6: Ancillary Services ─────────────────────────────────────────────
    with tabs[5]:
        st.subheader(f"Ancillary Services — {primary_zone}")

        anc_col1, anc_col2 = st.columns(2)
        with anc_col1:
            st.markdown("**Auto-Fetch**")
            tab_fetchers = get_available_fetchers(primary_zone)
            if tab_fetchers:
                for f in tab_fetchers:
                    st.caption(f"\U0001f4e1 {f['name']} ({f['source']})")
                if st.button(
                    f"\u26a1 Fetch ancillary for {primary_zone}",
                    key="tab_auto_fetch",
                ):
                    _run_and_store_ancillary_fetch(primary_zone, start_date, end_date)
            else:
                st.info("No auto-fetch available for this zone.")

        with anc_col2:
            st.markdown("**Manual CSV Upload**")
            tab_template = st.selectbox(
                "Template", list(ANCILLARY_TEMPLATES.keys()),
                format_func=lambda k: f"{k} — {ANCILLARY_TEMPLATES[k]['description'][:40]}",
                key="tab_anc_template",
            )
            tab_file = st.file_uploader("Upload CSV", type=["csv"], key="tab_anc_upload")
            if tab_file is not None:
                if st.button("Parse & Import", key="tab_anc_parse"):
                    _parse_and_store_ancillary_upload(
                        tab_file,
                        tab_template,
                        primary_zone,
                        start_date,
                        end_date,
                    )

        # Show loaded ancillary data summary
        st.divider()
        if anc_df is not None and not anc_df.empty:
            st.markdown(f"**Loaded ancillary data:** {len(anc_df)} rows")
            if "product_type" in anc_df.columns:
                products = anc_df["product_type"].unique()
                st.caption(f"Products: {', '.join(str(p) for p in products)}")
            with st.expander("Preview ancillary data"):
                st.dataframe(anc_df.head(50), hide_index=True)
        else:
            st.info(
                "No ancillary data loaded. Use auto-fetch or manual CSV upload above."
            )

    # ── Export button ────────────────────────────────────────────────────────
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
    pdf_bytes = export_to_pdf_bytes(
        zone=primary_zone,
        price_df=primary_df,
        daily_spreads=daily_spreads,
        monthly_spreads=monthly_spreads,
        percentiles=percentiles,
        revenue_estimate=export_revenue,
        negative_stats=neg_stats,
        tz=zone_tz,
        figures=report_figures,
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
