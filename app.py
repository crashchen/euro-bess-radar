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
    calculate_monthly_spreads,
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
from src.export import export_to_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="BESS Pulse", layout="wide", page_icon="\u26a1")

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

# BESS parameters
st.sidebar.subheader("BESS Parameters")
power_mw = st.sidebar.number_input("Power (MW)", min_value=0.1, value=10.0, step=1.0)
duration_hours = st.sidebar.selectbox("Duration (h)", [1, 2, 4], index=0)
efficiency = st.sidebar.slider("Efficiency (%)", 85, 95, 88) / 100.0
capture_rate = st.sidebar.slider("Capture (%)", 30, 100, 70) / 100.0
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
            try:
                content = anc_file.getvalue().decode("utf-8-sig")
                anc_df = parse_ancillary_csv(content, anc_template)
                st.session_state["ancillary_df"] = anc_df
                st.session_state["ancillary_template"] = anc_template
                st.success(f"{anc_template} loaded: {len(anc_df)} rows")
            except UnicodeDecodeError:
                st.error("Parse error: the uploaded file is not valid UTF-8/CSV text.")
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(f"Parse error: {exc}")

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
    primary_zone_for_fetch = selected_zones[0] if selected_zones else "DE_LU"
    fetchers = get_available_fetchers(primary_zone_for_fetch)

    if fetchers:
        st.markdown("**Available for this zone:**")
        for f in fetchers:
            st.caption(f"\U0001f4e1 {f['name']} ({f['source']})")
        if st.button(
            f"\u26a1 Fetch ancillary data for {primary_zone_for_fetch}",
            key="auto_fetch_ancillary",
        ):
            with st.spinner(
                f"Fetching from {', '.join(f['source'] for f in fetchers)}..."
            ):
                auto_start, auto_end = build_zone_query_window(
                    primary_zone_for_fetch,
                    start_date,
                    end_date,
                )
                results = run_auto_fetch(primary_zone_for_fetch, auto_start, auto_end)
                if results:
                    st.session_state["auto_fetch_results"] = results
                    st.success(
                        f"Fetched {len(results)} dataset(s): "
                        + ", ".join(results.keys())
                    )
                else:
                    st.warning(
                        "No data returned. Try manual CSV upload instead."
                    )
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
            load_zone_data.clear()
            load_generation.clear()
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
        st.session_state["refresh_token"] = refresh_token
    else:
        zone_data = st.session_state.get("zone_data", {})
        selected_zones = st.session_state.get("selected_zones", selected_zones)

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
    monthly_spreads = calculate_monthly_spreads(
        primary_df, tz=zone_tz, duration_hours=duration_hours,
    )
    percentiles = calculate_spread_percentiles(daily_spreads)
    neg_stats = calculate_negative_price_hours(primary_df)
    revenue = estimate_annual_arbitrage_revenue(
        daily_spreads,
        power_mw=power_mw,
        duration_hours=duration_hours,
        roundtrip_efficiency=efficiency,
        capture_rate=capture_rate,
    )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_names = [
        "Market Overview", "Heatmaps", "Revenue Estimation",
        "Renewable Correlation", "Zone Comparison",
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

        fig_price = px.line(
            primary_df.reset_index(),
            x="timestamp", y="price_eur_mwh",
            title="Day-Ahead Prices",
            labels={"price_eur_mwh": "EUR/MWh", "timestamp": ""},
            template=chart_template,
        )
        fig_price.update_xaxes(rangeslider_visible=True)
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
        st.plotly_chart(fig_shm, width="stretch")
        st.caption(
            "Negative hours mark windows selected for charging; positive hours mark "
            "windows selected for discharging. Color intensity shows the average signed "
            "ordered-spread signal assigned to those selected windows, not per-hour revenue attribution."
        )

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
                - Explicit single-sided energy prices are annualised using the selected BESS power, duration, and a simplified `{ANCILLARY_ENERGY_ACTIVATION_SHARE:.0%}` activation-hours assumption.
                - Two-sided balancing or system-price signals, such as GB `system buy` and `system sell`, are stored and shown but are not auto-monetised because dispatch direction and activation volume are still unknown.
                - The revenue stack chart shows `DA Arbitrage` plus each ancillary product as separate colored components.
                - This output is intended for market screening and prioritisation, not as a dispatch-grade settlement model.
                """
            )

        # Check for ancillary data
        manual_anc_df = st.session_state.get("ancillary_df")
        auto_fetch_results = st.session_state.get("auto_fetch_results", {})
        anc_df = build_ancillary_dataset(manual_anc_df, auto_fetch_results)
        anc_rev = None
        stack = None
        anc_source = None
        export_revenue = revenue.copy()

        if anc_df is not None and not anc_df.empty:
            anc_rev = calculate_ancillary_revenue(anc_df, power_mw, duration_hours)
            stack = merge_revenue_stack(revenue, anc_rev)
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
            r1.metric("Total Annual Revenue", f"\u20ac{stack['total_eur']:,.0f}")
            r2.metric("DA Arbitrage", f"\u20ac{stack['da_arbitrage_eur']:,.0f}",
                       delta=f"{stack['da_pct']:.0f}% of total")
            r3.metric("Ancillary Services", f"\u20ac{stack['total_eur'] - stack['da_arbitrage_eur']:,.0f}",
                       delta=f"{stack['ancillary_pct']:.0f}% of total")

            component_rows = [
                {"Source": source, "Annual Revenue (EUR)": value}
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
                    x=[row["Annual Revenue (EUR)"]],
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
            )
            if comp.empty:
                st.warning(
                    "No comparison rows could be built from the fetched zone data."
                )
            else:
                comp = comp.sort_values(
                    "estimated_annual_revenue_per_mw", ascending=False,
                ).reset_index(drop=True)
                comp_display = comp.copy()
                comp_display["avg_price"] = comp_display["avg_price"].map(
                    lambda x: f"\u20ac{x:.2f}"
                )
                comp_display["std_price"] = comp_display["std_price"].map(
                    lambda x: f"{x:.2f}"
                )
                comp_display["avg_spread"] = comp_display["avg_spread"].map(
                    lambda x: f"\u20ac{x:.2f}"
                )
                comp_display["p50_spread"] = comp_display["p50_spread"].map(
                    lambda x: f"\u20ac{x:.2f}"
                )
                comp_display["p90_spread"] = comp_display["p90_spread"].map(
                    lambda x: f"\u20ac{x:.2f}"
                )
                comp_display["negative_pct"] = comp_display["negative_pct"].map(
                    lambda x: f"{x:.1f}%"
                )
                comp_display["estimated_annual_revenue_per_mw"] = (
                    comp_display["estimated_annual_revenue_per_mw"].map(
                        lambda x: f"\u20ac{x:,.0f}"
                    )
                )
                st.dataframe(comp_display, width="stretch", hide_index=True)

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
    st.download_button(
        label="Export to Excel",
        data=xlsx_bytes,
        file_name=f"{primary_zone}_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

else:
    st.title("\u26a1 BESS Pulse")
    st.markdown(
        "Select one or more bidding zones in the sidebar and click **Fetch Data** "
        "to begin the analysis."
    )
