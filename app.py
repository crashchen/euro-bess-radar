"""BESS Pulse — European BESS Market Screening Dashboard."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    analyze_price_renewable_correlation,
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
    calculate_ancillary_revenue,
    generate_template_csv,
    merge_revenue_stack,
    parse_ancillary_csv,
)
from src.ancillary_fetchers import get_available_fetchers, run_auto_fetch
from src.config import ALL_ZONES, ZONE_TIMEZONES, get_zone_timezone, is_elexon_zone
from src.data_ingestion import (
    build_zone_query_window,
    fetch_generation_data,
    fetch_prices,
)
from src.export import export_to_bytes

logging.basicConfig(level=logging.INFO)

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
            except Exception as e:
                st.error(f"Parse error: {e}")

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
def load_zone_data(zone: str, start: str, end: str) -> pd.DataFrame:
    """Fetch price data for a zone (cached by Streamlit)."""
    api_start, api_end = build_zone_query_window(zone, start, end)
    return fetch_prices(
        zone=zone,
        start=api_start,
        end=api_end,
        use_cache=True,
    )


@st.cache_data(show_spinner=False)
def load_generation(zone: str, start: str, end: str) -> pd.DataFrame:
    """Fetch generation data for a zone (cached by Streamlit)."""
    api_start, api_end = build_zone_query_window(zone, start, end)
    return fetch_generation_data(
        zone=zone,
        start=api_start,
        end=api_end,
    )


if fetch_btn or "zone_data" in st.session_state:
    if fetch_btn:
        zone_data: dict[str, pd.DataFrame] = {}
        progress = st.sidebar.progress(0, text="Fetching...")

        for i, zone in enumerate(selected_zones):
            with st.spinner(f"Fetching data for {zone}..."):
                try:
                    df = load_zone_data(zone, str(start_date), str(end_date))
                    if not df.empty:
                        zone_data[zone] = df
                    else:
                        st.warning(f"No data returned for {zone}")
                except Exception as e:
                    st.error(f"Failed to fetch {zone}: {e}")
            progress.progress((i + 1) / len(selected_zones), text=f"Done: {zone}")

        progress.empty()
        st.session_state["zone_data"] = zone_data
        st.session_state["selected_zones"] = selected_zones
    else:
        zone_data = st.session_state.get("zone_data", {})
        selected_zones = st.session_state.get("selected_zones", selected_zones)

    if not zone_data:
        st.warning("No data available. Check zone selection and date range.")
        st.stop()

    # ── Compute analytics for primary zone ───────────────────────────────────
    primary_zone = selected_zones[0] if selected_zones else list(zone_data.keys())[0]
    primary_df = zone_data[primary_zone]
    zone_tz = get_zone_timezone(primary_zone)

    daily_spreads = calculate_daily_spreads(primary_df, tz=zone_tz)
    monthly_spreads = calculate_monthly_spreads(primary_df, tz=zone_tz)
    percentiles = calculate_spread_percentiles(daily_spreads)
    neg_stats = calculate_negative_price_hours(primary_df)
    revenue = estimate_annual_arbitrage_revenue(
        daily_spreads,
        power_mw=power_mw,
        duration_hours=duration_hours,
        roundtrip_efficiency=efficiency,
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
        k2.metric("Avg Spread", f"\u20ac{percentiles['mean']:.2f}/MWh")
        k3.metric("P90 Spread", f"\u20ac{percentiles['p90']:.2f}/MWh")
        k4.metric("Neg Hours", f"{neg_stats['pct_negative']:.1f}%")

        fig_price = px.line(
            primary_df.reset_index(),
            x="timestamp", y="price_eur_mwh",
            title="Day-Ahead Prices",
            labels={"price_eur_mwh": "EUR/MWh", "timestamp": ""},
            template=chart_template,
        )
        fig_price.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_price, width="stretch")

        fig_spread = px.bar(
            daily_spreads,
            x="date", y="spread",
            title="Daily Spread (Max - Min)",
            labels={"spread": "EUR/MWh", "date": ""},
            color="spread",
            color_continuous_scale="Viridis",
            template=chart_template,
        )
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

        spread_hm = build_spread_heatmap(primary_df, tz=zone_tz)
        fig_shm = px.imshow(
            spread_hm,
            title=f"Spread Contribution — Deviation from Daily Mean ({zone_tz})",
            labels=dict(x="Month", y="Hour (local)", color="EUR/MWh"),
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            aspect="auto",
            template=chart_template,
        )
        fig_shm.update_yaxes(dtick=1)
        st.plotly_chart(fig_shm, width="stretch")

    # ── Tab 3: Revenue Estimation ────────────────────────────────────────────
    with tabs[2]:
        st.subheader(f"Revenue Estimation — {primary_zone}")

        # Check for ancillary data
        anc_df = st.session_state.get("ancillary_df")
        anc_rev = None
        stack = None

        if anc_df is not None and not anc_df.empty:
            anc_rev = calculate_ancillary_revenue(anc_df, power_mw, duration_hours)
            stack = merge_revenue_stack(revenue, anc_rev)

        r1, r2, r3 = st.columns(3)
        if stack:
            r1.metric("Total Annual Revenue", f"\u20ac{stack['total_eur']:,.0f}")
            r2.metric("DA Arbitrage", f"\u20ac{stack['da_arbitrage_eur']:,.0f}",
                       delta=f"{stack['da_pct']:.0f}% of total")
            r3.metric("Ancillary Services", f"\u20ac{stack['total_eur'] - stack['da_arbitrage_eur']:,.0f}",
                       delta=f"{stack['ancillary_pct']:.0f}% of total")

            # Stacked bar
            stack_df = pd.DataFrame([{
                "DA Arbitrage": stack["da_arbitrage_eur"],
                "FCR": stack["fcr_eur"],
                "aFRR": stack["afrr_eur"],
                "mFRR": stack["mfrr_eur"],
            }]).T.reset_index()
            stack_df.columns = ["Source", "EUR"]
            fig_stack = px.bar(
                stack_df, x="EUR", y="Source", orientation="h",
                title="Revenue Stack",
                color="Source",
                color_discrete_sequence=["#2196F3", "#4CAF50", "#FF9800", "#F44336"],
                template=chart_template,
            )
            st.plotly_chart(fig_stack, width="stretch")
        else:
            r1.metric(
                "Est. Annual Revenue",
                f"\u20ac{revenue['annual_revenue_eur']:,.0f}",
                help=f"At {power_mw} MW, {duration_hours}h, {efficiency*100:.0f}% eff, 70% capture",
            )
            r2.metric("Revenue per MW", f"\u20ac{revenue['annual_revenue_eur_per_mw']:,.0f}/MW/yr")
            r3.metric("Avg Daily Revenue", f"\u20ac{revenue['avg_daily_revenue']:,.0f}")
            st.info("Upload ancillary services data to see the full revenue stack.")

        # Sensitivity table
        st.markdown("**Duration Sensitivity (per MW)**")
        sens_rows = []
        for dur in [1, 2, 4]:
            rev_d = estimate_annual_arbitrage_revenue(
                daily_spreads, power_mw=1.0, duration_hours=dur,
                roundtrip_efficiency=efficiency,
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
            title="Spread Distribution",
            labels={"spread": "Daily Spread (EUR/MWh)", "count": "Days"},
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

        with st.spinner("Fetching generation data..."):
            gen_df = load_generation(primary_zone, str(start_date), str(end_date))

        if gen_df.empty:
            st.info("Generation data not available for this zone.")
        else:
            corr = analyze_price_renewable_correlation(primary_df, gen_df)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Wind-Price Corr", f"{corr['correlation_wind_price']:.3f}")
            c2.metric("Solar-Price Corr", f"{corr['correlation_solar_price']:.3f}")
            c3.metric("Avg Price (High RE)", f"\u20ac{corr['avg_price_high_renewable']:.1f}")
            c4.metric("Avg Price (Low RE)", f"\u20ac{corr['avg_price_low_renewable']:.1f}")

            st.caption("Negative correlation = BESS-friendly market")

            # Scatter plot
            scatter_df = build_renewable_price_scatter(primary_df, gen_df, tz=zone_tz)
            if not scatter_df.empty:
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

            # Box plot by quartile
            if corr["price_spread_by_renewable_quartile"]:
                q_df = pd.DataFrame([
                    {"Quartile": k, "Avg Price (EUR/MWh)": v}
                    for k, v in corr["price_spread_by_renewable_quartile"].items()
                ])
                fig_box = px.bar(
                    q_df, x="Quartile", y="Avg Price (EUR/MWh)",
                    title="Average Price by Renewable Quartile",
                    color="Avg Price (EUR/MWh)",
                    color_continuous_scale="RdYlGn_r",
                    template=chart_template,
                )
                st.plotly_chart(fig_box, width="stretch")

    # ── Tab 5: Zone Comparison ───────────────────────────────────────────────
    with tabs[4]:
        if len(zone_data) < 2:
            st.info("Select multiple zones to see comparison.")
        else:
            st.subheader("Zone Comparison")
            comp = compare_zones(zone_data, zone_timezones=ZONE_TIMEZONES)
            st.dataframe(
                comp.style.format({
                    "avg_price": "\u20ac{:.2f}",
                    "std_price": "{:.2f}",
                    "avg_spread": "\u20ac{:.2f}",
                    "p50_spread": "\u20ac{:.2f}",
                    "p90_spread": "\u20ac{:.2f}",
                    "negative_pct": "{:.1f}%",
                    "estimated_annual_revenue_per_mw": "\u20ac{:,.0f}",
                }),
                width="stretch",
            )

            all_daily = []
            for zone, df in zone_data.items():
                ds = calculate_daily_spreads(df, tz=get_zone_timezone(zone))
                ds["zone"] = zone
                all_daily.append(ds)
            combined = pd.concat(all_daily, ignore_index=True)

            fig_comp = px.line(
                combined, x="date", y="spread", color="zone",
                title="Daily Spread Comparison",
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
        revenue_estimate=revenue,
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
