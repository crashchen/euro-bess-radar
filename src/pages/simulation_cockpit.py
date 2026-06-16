"""Tab 7: Simulation Cockpit — interval-level BESS dispatch replay."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.simulation import (
    available_local_dates,
    simulate_da_id_replay,
    simulate_da_milp_replay,
)


def render(
    *,
    primary_zone: str,
    primary_df: pd.DataFrame,
    intraday_df: pd.DataFrame | None,
    anc_df: pd.DataFrame | None,
    power_mw: float,
    duration_hours: int,
    efficiency: float,
    capture_rate: float,
    capex_eur_kwh: float,
    zone_tz: str,
    chart_template: str,
) -> None:
    """Render an enspired-like historical simulation cockpit."""
    st.subheader(f"Simulation Cockpit — {primary_zone}")
    st.caption(
        "Backtest replay, not live trading. This cockpit replays one local "
        "day through the project MILP dispatch model using already-loaded "
        "market data; it is not actual enspired dispatch or asset telemetry."
    )

    dates = available_local_dates(primary_df, tz=zone_tz)
    if not dates:
        st.info("Fetch day-ahead prices first to run a simulation replay.")
        return

    c1, c2, c3 = st.columns([1.2, 1.4, 1.4])
    selected_day = c1.selectbox(
        "Simulation day",
        options=dates,
        index=len(dates) - 1,
        format_func=lambda d: d.isoformat(),
        help="Local market day in the selected bidding-zone timezone.",
    )
    mode = c2.selectbox(
        "Replay mode",
        options=["DA MILP Replay", "DA + IDA1 Replay"],
        help=(
            "DA replay uses day-ahead prices only. DA + IDA1 replay requires "
            "IDA1 data already fetched in the Revenue Estimation tab."
        ),
    )
    has_capacity_anc = (
        anc_df is not None
        and not anc_df.empty
        and "capacity_price_eur_mw" in anc_df.columns
        and anc_df["capacity_price_eur_mw"].notna().any()
    )
    reserve_overlay = c3.checkbox(
        "Reserve headroom overlay",
        value=bool(has_capacity_anc),
        disabled=not has_capacity_anc,
        help=(
            "Shows uncommitted physical power headroom when reserve capacity "
            "data exists. It does not model FCR/aFRR activation energy."
        ),
    )

    if mode == "DA + IDA1 Replay":
        if intraday_df is None or intraday_df.empty:
            st.warning(
                "IDA1 is not loaded for this zone/date window. Open Revenue "
                "Estimation → Intraday Uplift (IDA1), fetch IDA1 prices, then "
                "return to this cockpit."
            )
            return
        result = simulate_da_id_replay(
            primary_df,
            intraday_df,
            simulation_date=selected_day,
            tz=zone_tz,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            capture_rate=capture_rate,
            capex_eur_kwh=capex_eur_kwh,
            soc_init_frac=0.5,
            include_reserve_headroom=reserve_overlay,
        )
    else:
        result = simulate_da_milp_replay(
            primary_df,
            simulation_date=selected_day,
            tz=zone_tz,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            capture_rate=capture_rate,
            capex_eur_kwh=capex_eur_kwh,
            soc_init_frac=0.5,
            include_reserve_headroom=reserve_overlay,
        )

    summary = result["summary"]
    ts = result["timeseries"]
    if ts.empty:
        st.warning(summary.get("reason") or "No simulation result for this day.")
        return

    _render_kpis(summary)
    st.caption(
        f"Annualized values scale this single simulated day by 365.25 and "
        f"divide by {power_mw:.1f} MW. Use it as an operational replay, not "
        "a bankable annual forecast."
    )

    left, right = st.columns([1.4, 1.0])
    with left:
        _plot_soc(ts, chart_template)
        _plot_dispatch(ts, chart_template)
        _plot_power_allocation(ts, reserve_overlay, chart_template)
    with right:
        _render_health_panel(summary)
        _plot_revenue(ts, chart_template)
        _plot_price(ts, mode, chart_template)

    with st.expander("Simulation interval data", expanded=False):
        st.dataframe(ts, width="stretch", hide_index=True)


def _render_kpis(summary: dict) -> None:
    r1 = st.columns(3)
    r1[0].metric("Total Revenue", f"€{summary['total_revenue_eur']:,.0f}")
    r1[1].metric("EUR/MW/Year", f"€{summary['annualized_eur_per_mw']:,.0f}")
    r1[2].metric("Number of Trades", f"{summary['number_of_trades']:,}")

    r2 = st.columns(3)
    r2[0].metric("Traded Volume", f"{summary['traded_volume_mwh']:,.1f} MWh")
    r2[1].metric("Physical Throughput", f"{summary['physical_throughput_mwh']:,.1f} MWh")
    r2[2].metric("Rebalancing Factor", f"{summary['rebalancing_factor']:.2f}")


def _render_health_panel(summary: dict) -> None:
    st.markdown("**Battery Health Snapshot**")
    h1, h2 = st.columns(2)
    h1.metric("Daily FCE", f"{summary['daily_fce']:.2f}")
    h2.metric("Avg C-rate", f"{summary['avg_c_rate']:.2f}")
    h3, h4 = st.columns(2)
    h3.metric("Max DoD", f"{summary['max_depth_of_discharge_pct']:.1f}%")
    h4.metric("SoH Delta", f"{summary['soh_delta_pct']:.4f}%")
    st.metric(
        "Degradation Cost",
        f"€{summary['degradation_cost_eur']:,.0f}",
        help="Screening-level FEC amortisation from the CapEx input. Zero if CapEx is zero.",
    )


def _plot_soc(ts: pd.DataFrame, chart_template: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["soc_mwh"],
        mode="lines", fill="tozeroy", name="SoC",
        line=dict(color="#00A3FF", width=2),
    ))
    fig.update_layout(
        title="State of Charge Progression",
        xaxis_title="",
        yaxis_title="MWh",
        template=chart_template,
        height=300,
    )
    st.plotly_chart(fig, width="stretch")


def _plot_dispatch(ts: pd.DataFrame, chart_template: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ts["local_time"], y=-ts["p_charge_mw"],
        name="Charge", marker_color="#D81B60",
    ))
    fig.add_trace(go.Bar(
        x=ts["local_time"], y=ts["p_discharge_mw"],
        name="Discharge", marker_color="#00A3FF",
    ))
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["net_dispatch_mw"],
        mode="lines", name="Net setpoint",
        line=dict(color="#D0D4DC", width=2),
    ))
    fig.update_layout(
        title="Physical Dispatch",
        xaxis_title="",
        yaxis_title="MW",
        barmode="relative",
        template=chart_template,
        height=340,
    )
    st.plotly_chart(fig, width="stretch")


def _plot_power_allocation(
    ts: pd.DataFrame,
    reserve_overlay: bool,
    chart_template: str,
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["available_discharge_mw"],
        mode="lines", name="Available discharge",
        line=dict(color="#0877BD", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=-ts["available_charge_mw"],
        mode="lines", name="Available charge",
        line=dict(color="#0A4C8A", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["net_dispatch_mw"],
        mode="lines", name="Physical position",
        line=dict(color="#FF2D95", width=2),
    ))
    if reserve_overlay and "reserve_headroom_mw" in ts.columns:
        fig.add_trace(go.Scatter(
            x=ts["local_time"], y=ts["reserve_headroom_mw"],
            mode="lines", name="Reserve headroom overlay",
            line=dict(color="#7FDBFF", width=1, dash="dot"),
        ))
    fig.update_layout(
        title="Dynamic Power Allocation",
        xaxis_title="",
        yaxis_title="MW",
        template=chart_template,
        height=340,
    )
    st.plotly_chart(fig, width="stretch")


def _plot_revenue(ts: pd.DataFrame, chart_template: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["cumulative_revenue_eur"],
        mode="lines", name="Cumulative revenue",
        line=dict(color="#D0D4DC", width=2),
    ))
    fig.add_trace(go.Bar(
        x=ts["local_time"], y=ts["interval_revenue_eur"],
        name="Interval revenue", marker_color="#FF2D95", opacity=0.55,
    ))
    fig.update_layout(
        title="Revenue Stream Replay",
        xaxis_title="",
        yaxis_title="EUR",
        template=chart_template,
        height=300,
    )
    st.plotly_chart(fig, width="stretch")


def _plot_price(ts: pd.DataFrame, mode: str, chart_template: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["price_eur_mwh"],
        mode="lines", name="DA price",
        line=dict(color="#FF2D95", width=2),
    ))
    if mode == "DA + IDA1 Replay" and "intraday_price_eur_mwh" in ts.columns:
        fig.add_trace(go.Scatter(
            x=ts["local_time"], y=ts["intraday_price_eur_mwh"],
            mode="lines", name="IDA1 price",
            line=dict(color="#00A3FF", width=2),
        ))
    fig.update_layout(
        title="Market Prices",
        xaxis_title="",
        yaxis_title="EUR/MWh",
        template=chart_template,
        height=300,
    )
    st.plotly_chart(fig, width="stretch")
