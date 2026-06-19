"""Tab 7: Simulation Cockpit — interval-level BESS dispatch replay."""

from __future__ import annotations

import math
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.export import cockpit_tables_to_excel
from src.simulation import (
    available_local_dates,
    build_dispatch_event_table,
    simulate_da_id_replay,
    simulate_da_milp_replay,
    simulate_replay_batch,
    simulate_sequential_da_id_batch,
)

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _cockpit_download_button(
    tables: dict, assumptions, *, key: str, label: str, file_name: str,
) -> None:
    """Offer an xlsx download of cockpit result tables (+ assumptions), or
    nothing when there is no data to export."""
    data = cockpit_tables_to_excel(tables, assumptions=assumptions)
    if data:
        st.download_button(
            label, data=data, file_name=file_name, mime=_XLSX_MIME, key=key,
        )

# Color semantics — assign concepts to hues so a reader can decode by color.
_C_PRICE = "#FF2D95"      # markets / DA price / interval revenue (magenta)
_C_PRICE_IDA = "#FFC233"  # IDA price (warm yellow, distinct from DA magenta)
_C_REVENUE = "#D0D4DC"    # cumulative revenue / neutral totals (light grey)
_C_SOC = "#7FB6FF"        # state of charge / energy storage (light cyan)
_C_CHARGE = "#D81B60"     # charging (deep magenta, "power in")
_C_DISCHARGE = "#00A3FF"  # discharging (cyan, "power out")
_C_NET = "#FFFFFF"        # net physical / final dispatch line (white)
_C_DA_POS = "#7FB6FF"     # DA planned position (light cyan, dotted)
_C_AVAIL = "#0A4C8A"      # available envelope fill (deep blue)
_C_REBID = "#FF2D95"      # rebid delta bars (magenta — financial event)
_BAR_OPACITY = 0.75


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
    assumptions: pd.DataFrame | None = None,
) -> None:
    """Render an enspired-like historical simulation cockpit."""
    _inject_cockpit_css()

    dates = available_local_dates(primary_df, tz=zone_tz)
    if not dates:
        st.info("Fetch day-ahead prices first to run a simulation replay.")
        return

    header_slot = st.empty()
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
    cockpit_capture_rate = c3.slider(
        "Capture rate haircut",
        min_value=0.5, max_value=1.0, value=1.0, step=0.05,
        help=(
            "1.0 = raw MILP perfect-foresight revenue (what the model achieves "
            "with full price knowledge). Values < 1.0 derate for forecast "
            f"slippage. The sidebar value ({capture_rate:.2f}) is intentionally "
            "ignored here so cockpit shows raw MILP output by default."
        ),
    )
    _render_cockpit_header(
        target=header_slot,
        primary_zone=primary_zone,
        selected_day=selected_day,
        mode=mode,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        capture_rate=cockpit_capture_rate,
    )

    if mode == "DA + IDA1 Replay":
        if intraday_df is None or intraday_df.empty:
            st.warning(
                "IDA1 is not loaded for this zone/date window. Open Revenue "
                "Estimation -> Intraday Uplift (IDA1), fetch IDA1 prices, then "
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
            capture_rate=cockpit_capture_rate,
            capex_eur_kwh=capex_eur_kwh,
            soc_init_frac=0.5,
        )
    else:
        result = simulate_da_milp_replay(
            primary_df,
            simulation_date=selected_day,
            tz=zone_tz,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            capture_rate=cockpit_capture_rate,
            capex_eur_kwh=capex_eur_kwh,
            soc_init_frac=0.5,
        )

    summary = result["summary"]
    ts = result["timeseries"]
    if ts.empty:
        st.warning(summary.get("reason") or "No simulation result for this day.")
        return

    _render_kpis(summary, mode=mode, power_mw=power_mw)

    _render_health_panel(summary)
    _plot_price(ts, mode, chart_template)
    _plot_soc(ts, chart_template, capacity_mwh=power_mw * duration_hours)
    _plot_dispatch(ts, chart_template)
    _plot_power_allocation(ts, chart_template, power_mw=power_mw)
    if mode == "DA + IDA1 Replay" and "da_position_mw" in ts.columns:
        _plot_wholesales(ts, chart_template, power_mw=power_mw)
    _plot_revenue(ts, chart_template)

    _render_event_table(build_dispatch_event_table(ts))
    _render_multi_day_summary(
        primary_df=primary_df,
        intraday_df=intraday_df,
        dates=dates,
        mode=mode,
        zone_tz=zone_tz,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        capture_rate=cockpit_capture_rate,
        capex_eur_kwh=capex_eur_kwh,
        chart_template=chart_template,
        assumptions=assumptions,
    )
    _render_forecast_policy_section(
        primary_df=primary_df,
        intraday_df=intraday_df,
        dates=dates,
        zone_tz=zone_tz,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        chart_template=chart_template,
        assumptions=assumptions,
    )

    with st.expander("Simulation interval data", expanded=False):
        st.dataframe(ts, width="stretch", hide_index=True)


def _inject_cockpit_css() -> None:
    st.markdown(
        """
        <style>
        .cockpit-hero {
            border: 1px solid rgba(0, 163, 255, 0.18);
            border-radius: 14px;
            padding: 12px 14px;
            margin: 4px 0 14px 0;
            background:
                radial-gradient(circle at 10% 14%, rgba(255,45,149,0.14), transparent 24%),
                radial-gradient(circle at 84% 16%, rgba(0,163,255,0.13), transparent 28%),
                linear-gradient(135deg, #090d12 0%, #101824 48%, #07090d 100%);
            box-shadow: 0 0 0 1px rgba(255,255,255,0.03) inset,
                        0 12px 34px rgba(0,0,0,0.30);
        }
        .cockpit-title {
            color: #f4f7fb;
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0;
        }
        .cockpit-subtitle {
            color: #a8b3c5;
            font-size: 0.78rem;
            margin-top: 4px;
            max-width: 980px;
        }
        .cockpit-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 10px;
        }
        .cockpit-pill {
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 999px;
            color: #dce8f7;
            background: rgba(255,255,255,0.055);
            padding: 4px 9px;
            font-size: 0.70rem;
            letter-spacing: 0.02em;
        }
        .cockpit-pill.alert {
            color: #ffc233;
            border-color: rgba(255,194,51,0.35);
            background: rgba(255,194,51,0.09);
        }
        .cockpit-kpi-grid {
            display: grid;
            grid-template-columns: repeat(8, minmax(0, 1fr));
            gap: 10px;
            margin: 8px 0 12px;
        }
        .cockpit-kpi-card {
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 14px;
            padding: 10px 12px;
            min-height: 78px;
            background: linear-gradient(180deg, rgba(20,29,42,0.96), rgba(8,12,18,0.96));
            box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        }
        .cockpit-kpi-card.primary {
            grid-column: span 2;
            min-height: 88px;
        }
        .cockpit-kpi-card.accent-magenta {
            background: linear-gradient(150deg, rgba(255,45,149,0.82), rgba(91,13,66,0.92));
        }
        .cockpit-kpi-card.accent-cyan {
            background: linear-gradient(150deg, rgba(0,163,255,0.74), rgba(4,51,91,0.95));
        }
        .cockpit-kpi-label {
            color: rgba(246,250,255,0.92);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.075em;
        }
        .cockpit-kpi-value {
            color: #ffffff;
            font-size: 1.42rem;
            font-weight: 850;
            letter-spacing: -0.045em;
            margin-top: 7px;
            line-height: 1.0;
        }
        .cockpit-kpi-card.primary .cockpit-kpi-value {
            font-size: 1.72rem;
        }
        .cockpit-kpi-help {
            color: rgba(235,242,252,0.82);
            font-size: 0.72rem;
            margin-top: 7px;
        }
        .cockpit-health-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 10px;
            margin-top: 8px;
        }
        .cockpit-health-card {
            border: 1px solid rgba(0,163,255,0.18);
            border-radius: 14px;
            padding: 10px 12px;
            margin-bottom: 12px;
            background: radial-gradient(circle at 50% 8%, rgba(0,163,255,0.18), transparent 48%),
                        linear-gradient(180deg, rgba(18,27,38,0.94), rgba(7,10,15,0.96));
        }
        .cockpit-health-value {
            color: #00a3ff;
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: -0.04em;
        }
        .cockpit-health-label {
            color: #a8b3c5;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.065em;
            margin-top: 4px;
        }
        @media (max-width: 1100px) {
            .cockpit-kpi-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
            .cockpit-health-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        }
        @media (max-width: 720px) {
            .cockpit-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .cockpit-health-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fmt_rebal(value: float) -> str:
    if not math.isfinite(value):
        return "Inf"
    return f"{value:.2f}"


def _render_cockpit_header(
    *,
    target,
    primary_zone: str,
    selected_day,
    mode: str,
    power_mw: float,
    duration_hours: int,
    efficiency: float,
    capture_rate: float,
) -> None:
    target.markdown(
        f"""
        <div class="cockpit-hero">
          <div class="cockpit-title">Simulation Cockpit | {escape(primary_zone)}</div>
          <div class="cockpit-subtitle">
            Historical replay of BESS dispatch and monetisation paths. This is
            a backtest cockpit, not live telemetry, not actual enspired dispatch,
            and not an executable trading instruction.
          </div>
          <div class="cockpit-pill-row">
            <span class="cockpit-pill alert">Historical Replay | Not Live Trading</span>
            <span class="cockpit-pill">{escape(mode)}</span>
            <span class="cockpit-pill">Local day {selected_day}</span>
            <span class="cockpit-pill">{power_mw:.1f} MW / {duration_hours}h</span>
            <span class="cockpit-pill">{efficiency:.0%} efficiency</span>
            <span class="cockpit-pill">{capture_rate:.0%} capture</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpis(summary: dict, *, mode: str, power_mw: float) -> None:
    annualized = summary["annualized_eur_per_mw"]
    if mode == "DA + IDA1 Replay" and "rebid_uplift_eur" in summary:
        optional_card = _kpi_card(
            "ID Rebid Uplift",
            f"EUR {summary['rebid_uplift_eur']:,.0f}",
            "Extra value vs DA-only",
            "accent-magenta",
        )
    else:
        optional_card = _kpi_card(
            "Power Basis",
            f"{power_mw:.1f} MW",
            "Annualised denominator",
            "",
        )

    st.markdown(
        f"""
        <div class="cockpit-kpi-grid">
          {_kpi_card("Day Revenue", f"EUR {summary['total_revenue_eur']:,.0f}",
                     "Selected local day", "primary accent-magenta")}
          {_kpi_card("EUR/MW/year", f"EUR {annualized:,.0f}",
                     "Single-day annualised", "primary accent-magenta")}
          {_kpi_card("Traded Volume", f"{summary['traded_volume_mwh']:,.1f} MWh",
                     "DA + ID financial volume", "accent-cyan")}
          {_kpi_card("Physical Throughput", f"{summary['physical_throughput_mwh']:,.1f} MWh",
                     "Battery charge + discharge", "accent-cyan")}
          {_kpi_card("Trades", f"{summary['number_of_trades']:,}",
                     "Contiguous dispatch blocks", "")}
          {optional_card}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Single-day x 365.25 extrapolation ~= EUR {summary['annualized_eur_per_mw']:,.0f}/MW/yr "
        f"(power_mw={power_mw:.1f}). Sample-size of 1, NOT a bankable annual figure - "
        "pick representative days or aggregate a multi-day window for a usable estimate."
    )


def _kpi_card(label: str, value: str, help_text: str, accent: str) -> str:
    class_name = f"cockpit-kpi-card {accent}".strip()
    return (
        f'<div class="{class_name}">'
        f'<div class="cockpit-kpi-label">{escape(label)}</div>'
        f'<div class="cockpit-kpi-value">{escape(value)}</div>'
        f'<div class="cockpit-kpi-help">{escape(help_text)}</div>'
        "</div>"
    )


def _render_health_panel(summary: dict) -> None:
    st.markdown(
        f"""
        <div class="cockpit-health-card">
          <div class="cockpit-kpi-label">Battery Health Snapshot</div>
          <div class="cockpit-health-grid">
            {_health_metric("Daily FCE", f"{summary['daily_fce']:.2f}")}
            {_health_metric("Avg C-rate", f"{summary['avg_c_rate']:.2f}")}
            {_health_metric("Max DoD", f"{summary['max_depth_of_discharge_pct']:.1f}%")}
            {_health_metric("SoH Delta", f"{summary['soh_delta_pct']:.4f}%")}
            {_health_metric("Degradation", f"EUR {summary['degradation_cost_eur']:,.0f}")}
            {_health_metric("Rebalancing", _fmt_rebal(summary["rebalancing_factor"]))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _health_metric(label: str, value: str) -> str:
    return (
        "<div>"
        f'<div class="cockpit-health-value">{escape(value)}</div>'
        f'<div class="cockpit-health-label">{escape(label)}</div>'
        "</div>"
    )


def _apply_panel_layout(
    fig: go.Figure,
    title: str,
    y_title: str,
    template: str,
    height: int = 280,
) -> None:
    """Shared layout: unified hover, compact margins, consistent height."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="#edf4ff", size=14)),
        xaxis_title="",
        yaxis_title=y_title,
        template=_cockpit_plot_template(template),
        height=height,
        hovermode="x unified",
        paper_bgcolor="#07090d",
        plot_bgcolor="#0b1118",
        font=dict(color="#dce8f7", size=11),
        margin=dict(l=46, r=22, t=48, b=28),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.16)",
        tickfont=dict(color="#a8b3c5"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.22)",
        linecolor="rgba(255,255,255,0.16)",
        tickfont=dict(color="#a8b3c5"),
    )


def _cockpit_plot_template(template: str) -> str:
    """Keep cockpit visuals dark even if the global dashboard theme is light."""
    return template if "dark" in template.lower() else "plotly_dark"


def _plot_price(ts: pd.DataFrame, mode: str, chart_template: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["price_eur_mwh"],
        mode="lines", name="DA price",
        line=dict(color=_C_PRICE, width=2, shape="hv"),
    ))
    if mode == "DA + IDA1 Replay" and "intraday_price_eur_mwh" in ts.columns:
        fig.add_trace(go.Scatter(
            x=ts["local_time"], y=ts["intraday_price_eur_mwh"],
            mode="lines", name="IDA1 price",
            line=dict(color=_C_PRICE_IDA, width=2, shape="hv"),
        ))
    _apply_panel_layout(fig, "Market Prices", "EUR / MWh", chart_template, height=240)
    st.plotly_chart(fig, width="stretch")


def _plot_soc(ts: pd.DataFrame, chart_template: str, *, capacity_mwh: float) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["soc_mwh"],
        mode="lines", fill="tozeroy", name="SoC",
        line=dict(color=_C_SOC, width=2),
        fillcolor="rgba(0,163,255,0.15)",
    ))
    if capacity_mwh > 0:
        fig.add_hline(
            y=capacity_mwh, line=dict(color=_C_SOC, width=1, dash="dash"),
            annotation_text=f"Capacity {capacity_mwh:.1f} MWh",
            annotation_position="top left",
        )
    _apply_panel_layout(fig, "State of Charge Progression", "MWh", chart_template, height=260)
    st.plotly_chart(fig, width="stretch")


def _plot_dispatch(ts: pd.DataFrame, chart_template: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ts["local_time"], y=-ts["p_charge_mw"],
        name="Charge", marker_color=_C_CHARGE, opacity=_BAR_OPACITY,
    ))
    fig.add_trace(go.Bar(
        x=ts["local_time"], y=ts["p_discharge_mw"],
        name="Discharge", marker_color=_C_DISCHARGE, opacity=_BAR_OPACITY,
    ))
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["net_dispatch_mw"],
        mode="lines", name="Net setpoint",
        line=dict(color=_C_NET, width=2, shape="hv"),
    ))
    fig.update_layout(barmode="relative")
    _apply_panel_layout(fig, "Physical Dispatch", "MW", chart_template, height=300)
    st.plotly_chart(fig, width="stretch")


def _plot_power_allocation(
    ts: pd.DataFrame, chart_template: str, *, power_mw: float
) -> None:
    """Filled-area 'power corridor' — the signature enspired visual."""
    fig = go.Figure()
    # Upper envelope: available discharge (positive side)
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["available_discharge_mw"],
        mode="lines", name="Available discharge",
        line=dict(color=_C_AVAIL, width=1),
        fill="tozeroy", fillcolor="rgba(10,76,138,0.35)",
    ))
    # Lower envelope: available charge (negated for the negative axis)
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=-ts["available_charge_mw"],
        mode="lines", name="Available charge",
        line=dict(color=_C_AVAIL, width=1),
        fill="tozeroy", fillcolor="rgba(10,76,138,0.35)",
    ))
    # Physical setpoint sits on top so the corridor frames it
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["net_dispatch_mw"],
        mode="lines", name="Physical position",
        line=dict(color=_C_REBID, width=2.5, shape="hv"),
    ))
    if power_mw > 0:
        fig.add_hline(
            y=power_mw, line=dict(color=_C_NET, width=1, dash="dot"),
            annotation_text=f"+{power_mw:.0f} MW limit",
            annotation_position="top left",
        )
        fig.add_hline(
            y=-power_mw, line=dict(color=_C_NET, width=1, dash="dot"),
            annotation_text=f"-{power_mw:.0f} MW limit",
            annotation_position="bottom left",
        )
    _apply_panel_layout(
        fig, "Dynamic Power Allocation", "MW", chart_template, height=320,
    )
    st.plotly_chart(fig, width="stretch")


def _plot_wholesales(
    ts: pd.DataFrame, chart_template: str, *, power_mw: float
) -> None:
    """DA position vs final physical vs ID rebid delta — enspired's signature panel."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ts["local_time"], y=ts["rebid_delta_mw"],
        name="ID rebid delta", marker_color=_C_REBID, opacity=_BAR_OPACITY,
    ))
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["da_position_mw"],
        mode="lines", name="DA position",
        line=dict(color=_C_DA_POS, width=2, dash="dot", shape="hv"),
    ))
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["ida_position_mw"],
        mode="lines", name="Physical (DA+ID)",
        line=dict(color=_C_NET, width=2.5, shape="hv"),
    ))
    if power_mw > 0:
        fig.add_hline(
            y=power_mw, line=dict(color=_C_AVAIL, width=1, dash="dash"),
            annotation_text=f"Max sell {power_mw:.0f} MW",
            annotation_position="top right",
        )
        fig.add_hline(
            y=-power_mw, line=dict(color=_C_AVAIL, width=1, dash="dash"),
            annotation_text=f"Max buy {power_mw:.0f} MW",
            annotation_position="bottom right",
        )
    _apply_panel_layout(
        fig, "Wholesales Optimization (DA vs ID rebid vs Physical)",
        "MW (positive = discharge / sell)", chart_template, height=320,
    )
    st.plotly_chart(fig, width="stretch")


def _plot_revenue(ts: pd.DataFrame, chart_template: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ts["local_time"], y=ts["interval_revenue_eur"],
        name="Interval revenue", marker_color=_C_PRICE, opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=ts["local_time"], y=ts["cumulative_revenue_eur"],
        mode="lines", name="Cumulative revenue",
        line=dict(color=_C_REVENUE, width=2.5),
    ))
    _apply_panel_layout(fig, "Revenue Stream Replay", "EUR", chart_template, height=260)
    st.plotly_chart(fig, width="stretch")


def _render_event_table(events: pd.DataFrame) -> None:
    with st.expander("Dispatch event table", expanded=False):
        st.caption(
            "Contiguous non-zero physical dispatch blocks. This is a replay "
            "diagnostic, not an exchange order blotter."
        )
        if events.empty:
            st.info("No non-zero physical dispatch events for this day.")
            return
        st.dataframe(
            events,
            width="stretch",
            hide_index=True,
            column_config={
                "duration_h": st.column_config.NumberColumn("Duration (h)", format="%.2f"),
                "avg_power_mw": st.column_config.NumberColumn("Avg MW", format="%.2f"),
                "energy_mwh": st.column_config.NumberColumn("Energy MWh", format="%.2f"),
                "avg_price_eur_mwh": st.column_config.NumberColumn(
                    "Avg EUR/MWh", format="%.2f",
                ),
                "revenue_eur": st.column_config.NumberColumn("Revenue EUR", format="%.0f"),
                "soc_start_pct": st.column_config.NumberColumn("Start SoC %", format="%.1f"),
                "soc_end_pct": st.column_config.NumberColumn("End SoC %", format="%.1f"),
                "avg_rebid_delta_mw": st.column_config.NumberColumn(
                    "Avg rebid MW", format="%.2f",
                ),
                "max_abs_rebid_delta_mw": st.column_config.NumberColumn(
                    "Max |rebid| MW", format="%.2f",
                ),
            },
        )


def _render_multi_day_summary(
    *,
    primary_df: pd.DataFrame,
    intraday_df: pd.DataFrame | None,
    dates: list,
    mode: str,
    zone_tz: str,
    power_mw: float,
    duration_hours: int,
    efficiency: float,
    capture_rate: float,
    capex_eur_kwh: float,
    chart_template: str,
    assumptions: pd.DataFrame | None = None,
) -> None:
    with st.expander("Multi-day replay summary", expanded=False):
        st.caption(
            "Runs the same interval-level replay across multiple loaded local "
            "days and aggregates daily KPI. This is still historical backtest "
            "replay, not live trading."
        )
        c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
        sample = c1.selectbox(
            "Replay sample",
            options=["7 latest days", "30 latest days", "90 latest days", "All loaded days"],
            index=1,
            key="simulation_batch_sample",
        )
        carry_soc = c2.checkbox(
            "Continuous SoC across days",
            value=True,
            key="simulation_batch_carry_soc",
            help=(
                "When on, day N+1 starts at the end-of-day SoC of day N. "
                "Off: every day resets to 50%% (legacy behaviour, biases "
                "evening-peak revenue downward)."
            ),
        )
        run = c3.button("Run multi-day replay", key="simulation_batch_run")
        if mode == "DA + IDA1 Replay" and (intraday_df is None or intraday_df.empty):
            st.info("Load IDA1 data before running a DA + IDA1 multi-day replay.")
            return
        if not run:
            st.info("Choose a replay sample and click Run to aggregate loaded days.")
            return

        limit = _sample_limit(sample)
        batch_dates = dates if limit is None else dates[-limit:]
        with st.spinner(f"Running {len(batch_dates)} daily replay(s)..."):
            batch = simulate_replay_batch(
                primary_df,
                mode=mode,
                intraday_df=intraday_df,
                tz=zone_tz,
                dates=batch_dates,
                power_mw=power_mw,
                duration_hours=duration_hours,
                efficiency=efficiency,
                capture_rate=capture_rate,
                capex_eur_kwh=capex_eur_kwh,
                carry_soc=carry_soc,
            )

        excluded = int(batch.attrs.get("excluded_days", 0))
        if batch.empty:
            st.warning(f"No valid replay days in this sample. Excluded days: {excluded}.")
            return
        carry_mode = str(batch.attrs.get("carry_mode", "per_day_reset"))
        _render_batch_kpis(
            batch, requested_days=len(batch_dates),
            excluded_days=excluded, carry_mode=carry_mode,
        )
        _plot_batch_summary(batch, chart_template)
        if len(batch) >= 3:
            _plot_rolling_summary(batch, chart_template)
        if len(batch) >= 7:
            _plot_weekday_heatmap(batch, chart_template)
        st.dataframe(batch, width="stretch", hide_index=True)
        _cockpit_download_button(
            {"Multi-day Replay": batch}, assumptions,
            key="dl_multi_day_replay",
            label="\U0001f4e5 Download multi-day replay (Excel)",
            file_name="cockpit_multiday_replay.xlsx",
        )


def _render_forecast_policy_section(
    *,
    primary_df: pd.DataFrame,
    intraday_df: pd.DataFrame | None,
    dates: list,
    zone_tz: str,
    power_mw: float,
    duration_hours: int,
    efficiency: float,
    chart_template: str,
    assumptions: pd.DataFrame | None = None,
) -> None:
    """Three-way DA-only / forecast-realised / perfect-foresight panel."""
    with st.expander("Forecast-driven IDA policy (vs perfect-foresight ceiling)", expanded=False):
        st.caption(
            "Compares a climatology-forecast rebid policy with the ex-post "
            "perfect-foresight ceiling; the gap is the forecast error cost. "
            "Climatology forecast, hourly bucketed — a screening estimate, "
            "NOT a trading-grade IDA price model. Values are raw solver "
            "outputs and do NOT apply the sidebar capture-rate haircut."
        )
        if intraday_df is None or intraday_df.empty:
            st.info("Load IDA1 data to run the forecast-driven policy comparison.")
            return

        c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.1, 1.0])
        sample = c1.selectbox(
            "Replay sample",
            options=["7 latest days", "30 latest days", "90 latest days", "All loaded days"],
            index=1,
            key="forecast_policy_sample",
        )
        bucket_label = c2.selectbox(
            "Forecast climatology",
            options=["Hour-of-day", "Hour-of-week"],
            index=0,
            key="forecast_policy_bucket",
            help=(
                "Hour-of-day is robust on short windows; hour-of-week is "
                "weekday-aware but needs several weeks of history."
            ),
        )
        mode_label = c3.selectbox(
            "Forecast information",
            options=["LOO cross-validation", "Walk-forward"],
            index=0,
            key="forecast_policy_mode",
            help=(
                "LOO uses every loaded day except the target day (may use "
                "FUTURE days — unbiased skill estimate, not what a desk knew "
                "in real time). Walk-forward uses only days strictly before "
                "the target (drops the earliest day with no history)."
            ),
        )
        deadband_eur_per_mw = c4.number_input(
            "Rebid deadband (EUR/MW/day)",
            min_value=0.0, max_value=50.0, value=0.0, step=0.5,
            key="forecast_policy_deadband",
            help=(
                "Risk gate: minimum FORECAST-predicted IDA uplift per MW per "
                "day needed to rebid; below it the desk holds its committed DA "
                "schedule. 0 = rebid on any forecast edge (screening "
                "baseline). Raise it to suppress churn on marginal days where "
                "a thin forecast edge can flip into a realised loss."
            ),
        )
        run = st.button("Run forecast policy", key="forecast_policy_run")
        if not run:
            st.info("Choose a sample and click Run to compare against the ceiling.")
            return

        bucket = "hour_of_week" if bucket_label == "Hour-of-week" else "hour_of_day"
        forecast_mode = "walk_forward" if mode_label == "Walk-forward" else "loo"
        # The solver gate is an absolute per-day EUR hurdle; the UI knob is
        # power-normalised so it is comparable across system sizes.
        min_rebid_uplift_eur = float(deadband_eur_per_mw) * power_mw
        limit = _sample_limit(sample)
        batch_dates = dates if limit is None else dates[-limit:]
        with st.spinner(f"Solving {len(batch_dates)} day(s) under forecast + ceiling..."):
            per_day, summary = simulate_sequential_da_id_batch(
                primary_df,
                intraday_df,
                dates=batch_dates,
                tz=zone_tz,
                power_mw=power_mw,
                duration_hours=duration_hours,
                efficiency=efficiency,
                bucket=bucket,
                forecast_mode=forecast_mode,
                min_rebid_uplift_eur=min_rebid_uplift_eur,
            )

        if per_day.empty:
            st.warning(
                f"No valid forecast-policy days in this sample. "
                f"Excluded: {summary['excluded_days']}. A leave-one-day-out "
                "forecast needs at least 2 loaded days; walk-forward needs at "
                "least one day before the target."
            )
            return

        _render_forecast_policy_kpis(summary)
        _plot_forecast_policy(per_day, chart_template)
        meta = summary["forecast_meta"]
        mode_note = {
            "loo": "LOO cross-validation (may use future days except target)",
            "walk_forward": "walk-forward (prior days only)",
            "in_sample": "in-sample (includes target day)",
        }.get(meta["forecast_mode"], meta["forecast_mode"])
        st.caption(
            f"Forecast support: {meta['n_buckets_filled']}/"
            f"{meta['n_buckets_requested']} buckets backed by history, "
            f"{meta['fallback_points']} global-mean fallback points "
            f"(coverage {meta['coverage']:.0%}, {mode_note}). Shape/order is "
            "the primary signal, but level error still affects the cycling "
            "decision via round-trip efficiency loss and VOM."
        )
        _render_forecast_skill(summary.get("forecast_skill", {}), chart_template)
        st.dataframe(per_day, width="stretch", hide_index=True)
        _cockpit_download_button(
            {"Sequential DA+ID": per_day}, assumptions,
            key="dl_forecast_policy",
            label="\U0001f4e5 Download forecast policy (Excel)",
            file_name="cockpit_forecast_policy.xlsx",
        )


def _render_forecast_skill(skill: dict, chart_template: str) -> None:
    """Price-space forecast accuracy panel (MAE/bias/RMSE + skill vs DA)."""
    if not skill or skill.get("n_points", 0) == 0:
        return
    st.markdown("**Forecast skill (price-space, vs realised IDA)**")
    cols = st.columns(4)
    cols[0].metric("MAE", f"EUR {skill['mae']:.1f}/MWh")
    cols[1].metric("Bias", f"EUR {skill['bias']:+.1f}/MWh")
    cols[2].metric("RMSE", f"EUR {skill['rmse']:.1f}/MWh")
    sk = skill.get("skill_vs_da")
    cols[3].metric(
        "Skill vs DA-as-IDA", "n/a" if sk is None else f"{sk:+.0%}",
    )
    st.caption(
        f"Over {skill['n_points']:,} aligned intervals; realised IDA std "
        f"EUR {skill['realised_std']:.1f}/MWh. Bias >0 = forecast prints high. "
        "Skill vs DA-as-IDA >0 means the climatology beats just assuming "
        "IDA == DA (the no-rebid-signal null); <=0 means it does not, so the "
        "forecast-driven rebid rests on thin ice — widen the deadband."
    )
    by_hour = skill.get("by_hour")
    if by_hour is not None and not by_hour.empty:
        fig = go.Figure(
            go.Bar(x=by_hour["hour"], y=by_hour["mae"], marker_color=_C_PRICE_IDA),
        )
        _apply_panel_layout(
            fig, "Forecast MAE by hour-of-day", "EUR/MWh", chart_template, height=240,
        )
        fig.update_xaxes(title="Local hour")
        st.plotly_chart(fig, width="stretch")


def _render_forecast_policy_kpis(summary: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("DA-only", f"EUR {summary['total_da_only_eur']:,.0f}")
    cols[1].metric(
        "Forecast realised",
        f"EUR {summary['total_realised_eur']:,.0f}",
        delta=f"{summary['total_captured_eur']:,.0f} vs DA-only",
    )
    cols[2].metric("Perfect-foresight ceiling", f"EUR {summary['total_ceiling_eur']:,.0f}")
    cols[3].metric(
        "Forecast error cost",
        f"EUR {summary['total_forecast_error_eur']:,.0f}",
    )
    rate = summary["capture_rate"]
    if rate is None:
        rate_txt = "n/a (no rebid opportunity in window)"
    else:
        rate_txt = f"{rate:.0%} of achievable IDA uplift captured"
    n_rebid = summary.get("n_rebid_days", 0)
    n_hold = summary.get("n_hold_days", 0)
    gate_txt = ""
    if summary.get("min_rebid_uplift_eur", 0.0) > 0.0:
        gate_txt = (
            f" Deadband: rebid on {n_rebid} day(s), held DA on {n_hold} "
            "day(s) below the hurdle."
        )
    st.caption(
        f"{rate_txt}. Achievable uplift (ceiling - DA-only): "
        f"EUR {summary['total_ceiling_uplift_eur']:,.0f}. Captured can be "
        f"negative when the forecast misleads the rebid into churn losses.{gate_txt}"
    )


def _plot_forecast_policy(per_day: pd.DataFrame, chart_template: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=per_day["date"], y=per_day["da_only_eur"],
        name="DA-only", marker_color=_C_REVENUE, opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=per_day["date"], y=per_day["realised_eur"],
        name="Forecast realised", marker_color=_C_PRICE_IDA, opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=per_day["date"], y=per_day["ceiling_eur"],
        mode="lines+markers", name="Perfect-foresight ceiling",
        line=dict(color=_C_DISCHARGE, width=2),
    ))
    fig.update_layout(
        title="DA-only vs Forecast-driven vs Perfect-foresight ceiling",
        xaxis_title="Local date",
        yaxis_title="EUR/day",
        barmode="group",
        template=chart_template,
        height=340,
        hovermode="x unified",
        margin=dict(l=40, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    st.plotly_chart(fig, width="stretch")


def _sample_limit(sample: str) -> int | None:
    if sample.startswith("7 "):
        return 7
    if sample.startswith("30 "):
        return 30
    if sample.startswith("90 "):
        return 90
    return None


def _render_batch_kpis(
    batch: pd.DataFrame,
    *,
    requested_days: int,
    excluded_days: int,
    carry_mode: str,
) -> None:
    avg_daily = float(batch["total_revenue_eur"].mean())
    avg_annualized = float(batch["annualized_eur_per_mw"].mean())
    best = batch.loc[batch["total_revenue_eur"].idxmax()]
    stress = batch.loc[batch["daily_fce"].idxmax()]
    cols = st.columns(5)
    cols[0].metric("Valid Days", f"{len(batch)} / {requested_days}")
    cols[1].metric("Avg Day Revenue", f"EUR {avg_daily:,.0f}")
    cols[2].metric("Avg Annualized", f"EUR {avg_annualized:,.0f}/MW/yr")
    cols[3].metric("Avg FCE/day", f"{batch['daily_fce'].mean():.2f}")
    cols[4].metric("Excluded Days", f"{excluded_days}")
    soc_note = {
        "continuous_horizon": (
            "Continuous-horizon MILP across the window — overnight SoC "
            "is free, terminal-neutral only at the segment end."
        ),
        "per_day_reset": "Each day resets to 50% SoC (per-day terminal-neutral).",
    }.get(carry_mode, "")
    st.caption(
        f"Best day: {best['date']} (EUR {best['total_revenue_eur']:,.0f}). "
        f"Highest-cycle day: {stress['date']} ({stress['daily_fce']:.2f} FCE). "
        f"{soc_note}"
    )


def _plot_rolling_summary(batch: pd.DataFrame, chart_template: str) -> None:
    """Rolling 7-day revenue and FCE — smooths weekday seasonality."""
    window = min(7, len(batch))
    roll_rev = batch["total_revenue_eur"].rolling(window, min_periods=1).mean()
    roll_fce = batch["daily_fce"].rolling(window, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=batch["date"], y=roll_rev,
        mode="lines+markers", name=f"{window}-day avg revenue",
        line=dict(color=_C_PRICE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=batch["date"], y=roll_fce,
        mode="lines+markers", name=f"{window}-day avg FCE", yaxis="y2",
        line=dict(color=_C_SOC, width=2, dash="dot"),
    ))
    fig.update_layout(
        title=f"Rolling {window}-day Revenue and Cycle Intensity",
        xaxis_title="Local date",
        yaxis_title="EUR/day (avg)",
        yaxis2=dict(title="FCE/day (avg)", overlaying="y", side="right"),
        template=chart_template,
        height=300,
        hovermode="x unified",
        margin=dict(l=40, r=50, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    st.plotly_chart(fig, width="stretch")


def _plot_weekday_heatmap(batch: pd.DataFrame, chart_template: str) -> None:
    """Weekday x ISO-week revenue heatmap — exposes weekly seasonality."""
    dates = pd.to_datetime(batch["date"])
    df = batch.assign(
        iso_week=dates.dt.strftime("%G-W%V"),
        weekday=dates.dt.day_name().str[:3],
    )
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = (
        df.pivot_table(
            index="weekday", columns="iso_week",
            values="total_revenue_eur", aggfunc="mean",
        )
        .reindex(weekday_order)
    )
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="Magma",
        colorbar=dict(title="EUR/day"),
        hovertemplate="%{y}, %{x}<br>EUR %{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Weekday x Week Revenue Heatmap",
        template=chart_template,
        height=300,
        margin=dict(l=40, r=50, t=50, b=30),
    )
    st.plotly_chart(fig, width="stretch")


def _plot_batch_summary(batch: pd.DataFrame, chart_template: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=batch["date"], y=batch["total_revenue_eur"],
        name="Daily revenue", marker_color=_C_PRICE, opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=batch["date"], y=batch["daily_fce"],
        mode="lines+markers", name="Daily FCE", yaxis="y2",
        line=dict(color=_C_SOC, width=2),
    ))
    fig.update_layout(
        title="Multi-day Replay: Revenue and Cycle Intensity",
        xaxis_title="Local date",
        yaxis_title="EUR/day",
        yaxis2=dict(title="FCE/day", overlaying="y", side="right"),
        template=chart_template,
        height=320,
        hovermode="x unified",
        margin=dict(l=40, r=50, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    st.plotly_chart(fig, width="stretch")
