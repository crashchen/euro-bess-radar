"""Tab 7: Simulation Cockpit — interval-level BESS dispatch replay."""

from __future__ import annotations

import math
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.activation_overlay import compute_activation_overlay
from src.ancillary import (
    capacity_price_for_product,
    capacity_price_series_for_product,
    list_capacity_products,
)
from src.assumptions import CAPTURE_PARAM_LABEL
from src.config import ANCILLARY_CAPACITY_AVAILABILITY
from src.data_ingestion import (
    read_activation_cache,
    read_capacity_cache,
    read_imbalance_cache,
)
from src.dispatch import solve_joint_capacity_batch
from src.export import cockpit_tables_to_excel
from src.imbalance_overlay import compute_imbalance_overlay
from src.reserve_forecast import RESERVE_VALUE_COL, compute_reserve_forecast_skill
from src.simulation import (
    DAYS_PER_YEAR,
    available_local_dates,
    build_dispatch_event_table,
    simulate_da_id_replay,
    simulate_da_id_reserve_ceiling_batch,
    simulate_da_milp_replay,
    simulate_replay_batch,
    simulate_sequential_da_id_batch,
    simulate_sequential_da_id_reserve_batch,
    simulate_stochastic_da_id_batch,
    simulate_stochastic_triple_batch,
)
from src.strategy_compare import (
    STOCHASTIC_POLICY_VALUE_LABEL,
    STOCHASTIC_POLICY_VALUE_RESERVE_LABEL,
    build_strategy_comparison,
)

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Stochastic policy panel (Increment D): fixed scenario count + seed so the
# opt-in run is reproducible; the load-bearing rebid cap is the one exposed knob.
_STOCHASTIC_N_SCENARIOS = 10
_STOCHASTIC_SEED = 0


def _cockpit_export_assumptions(
    assumptions: pd.DataFrame | None,
    *,
    capture_value: str,
    capture_affects: str,
    capture_label: str = "Cockpit capture haircut",
) -> pd.DataFrame | None:
    """Adapt the global assumptions for a cockpit export.

    The cockpit ignores the sidebar DA-slippage capture rate (it uses its own
    haircut, or none for the forecast panel), so the exported Assumptions
    sheet must override that row instead of inheriting the sidebar value and
    contradicting the numbers. Falls back to appending if the sidebar capture
    label has drifted, so the cockpit context is never silently dropped.
    """
    if assumptions is None or assumptions.empty:
        return assumptions
    out = assumptions.copy()
    mask = out["parameter"] == CAPTURE_PARAM_LABEL
    new_row = {
        "parameter": capture_label, "value": capture_value, "unit": "",
        "source": "Cockpit panel", "affects": capture_affects,
    }
    if mask.any():
        for col, val in new_row.items():
            out.loc[mask, col] = val
        return out
    return pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)


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
        anc_df=anc_df,
        primary_zone=primary_zone,
        dates=dates,
        zone_tz=zone_tz,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        chart_template=chart_template,
        assumptions=assumptions,
    )
    _render_reserve_forecast_skill_section(
        anc_df=anc_df, zone_tz=zone_tz, chart_template=chart_template,
    )
    _render_activation_overlay_section(
        primary_zone=primary_zone, dates=dates, zone_tz=zone_tz, power_mw=power_mw,
    )
    _render_imbalance_overlay_section(
        primary_zone=primary_zone, dates=dates, zone_tz=zone_tz, power_mw=power_mw,
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


def _fmt_strategy_bar_label(value: float) -> str:
    """Format strategy-comparison bar labels without surfacing NaN/inf."""
    if not math.isfinite(value):
        return "N/A"
    return f"{value:,.0f}"


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
        # `capture_rate` here is the cockpit's own haircut, not the sidebar's.
        export_assumptions = _cockpit_export_assumptions(
            assumptions,
            capture_value=f"{capture_rate:.0%}",
            capture_affects=(
                "Haircut on cockpit replay revenue (the sidebar capture rate "
                "is intentionally ignored in the cockpit)"
            ),
        )
        _cockpit_download_button(
            {"Multi-day Replay": batch}, export_assumptions,
            key="dl_multi_day_replay",
            label="\U0001f4e5 Download multi-day replay (Excel)",
            file_name="cockpit_multiday_replay.xlsx",
        )


def _slice_to_local_dates(
    df: pd.DataFrame, keep_dates: set, tz: str,
) -> pd.DataFrame:
    """Filter a UTC-indexed frame to rows whose local date is kept.

    Returns an empty frame for a non-datetime index so a caller windowing
    capacity prices omits the row rather than silently pricing off the full
    sample.
    """
    if df is None or df.empty:
        return df
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return df.iloc[0:0]
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local_dates = pd.Index(idx.tz_convert(tz).date)
    return df[local_dates.isin(set(keep_dates))]


# Provenance labels for the cockpit's capacity source (shown in the panel).
_CAP_SOURCE_CACHE = "Cached unified import"
_CAP_SOURCE_SESSION = "Session ancillary fallback"


def _resolve_capacity_dataset(
    primary_zone: str | None, anc_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, str]:
    """Cache-first reserve-capacity dataset for the cockpit + a provenance label.

    Prefers the persisted unified-import cache (``read_capacity_cache``) so
    uploaded capacity actually drives the 9.2b reserve rows; falls back to the
    session ancillary (per-country upload / auto-fetch) only when the cache holds
    no capacity for this zone. The full-zone frame is returned; the panel windows
    it per valid date via ``_slice_to_local_dates`` before pricing, so a
    full-sample mean never leaks into the comparison window.
    """
    if not primary_zone:
        return anc_df, _CAP_SOURCE_SESSION
    cached = read_capacity_cache(primary_zone)
    if cached is not None and not cached.empty and list_capacity_products(cached):
        return cached, _CAP_SOURCE_CACHE
    return anc_df, _CAP_SOURCE_SESSION


def _reserve_coopt_total(
    primary_df: pd.DataFrame,
    reserve_product: str | None,
    anc_df: pd.DataFrame | None,
    *,
    valid_dates: set,
    tz: str,
    power_mw: float,
    duration_hours: int,
    efficiency: float,
) -> tuple[float | None, str | None, float | None]:
    """DA + reserve-capacity co-opt window total for the strategy comparison.

    Returns ``(total_eur, label, capacity_price_eur_mw_h)``, or
    ``(None, None, None)`` when no reserve product / in-window capacity price /
    solvable day is available. The total sums ``solve_joint_capacity_batch``'s
    ``joint_total_revenue`` over EXACTLY the DA+ID rows' valid local days
    (``per_day["date"]``), so every strategy shares one annualisation
    denominator (``valid_days``). The capacity price is taken from ancillary
    rows INSIDE the same valid-date window, so an out-of-window capacity print
    cannot skew the fourth row — when no capacity rows overlap the window the
    row is omitted rather than priced off the full sample. The joint MILP makes
    reserve headroom compete with DA for power (not additive) and excludes
    activation energy, so the red-line is enforced by the solver, not labelling.
    """
    if not reserve_product or not valid_dates:
        return None, None, None
    # Price the reserve from capacity rows in the SAME window as the DA+ID rows
    # so an out-of-window capacity print cannot skew the fourth row.
    window_anc = _slice_to_local_dates(anc_df, valid_dates, tz)
    price = capacity_price_for_product(window_anc, reserve_product)
    if price is None:
        return None, None, None
    window_df = _slice_to_local_dates(primary_df, valid_dates, tz)
    if window_df is None or window_df.empty:
        return None, None, None
    joint = solve_joint_capacity_batch(
        window_df,
        capacity_price_eur_mw_h=price,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        tz=tz,
    )
    if joint.empty:
        return None, None, None
    # Restrict to the DA+ID valid day set so the reserve row shares the exact
    # annualisation denominator (the joint batch can solve DA-only days the
    # sequential DA+ID batch excluded for want of IDA data).
    joint = joint[joint["date"].isin(valid_dates)]
    if joint.empty:
        return None, None, None
    total = float(joint["joint_total_revenue"].sum())
    return total, f"DA + {reserve_product} co-opt (headroom)", float(price)


def _reserve_triple_totals(
    primary_df: pd.DataFrame,
    intraday_df: pd.DataFrame,
    reserve_series: pd.Series | None,
    *,
    valid_dates: set,
    tz: str,
    power_mw: float,
    duration_hours: int,
    efficiency: float,
    bucket: str,
) -> dict:
    """9.2a ceiling + 9.2b forecast-driven realistic totals over a shared window.

    Both the perfect-foresight ceiling (Phase 9.2a) and the forecast-driven
    realistic triple (Phase 9.2b reserve-first sequential) are priced off the
    SAME per-interval ``reserve_series`` (via
    :func:`simulation.align_reserve_price_to_index`), so ``realistic <= ceiling``
    and the cockpit's forecast-effect gap panel is internally consistent.

    The realistic policy is ALWAYS walk-forward (reserve is committed D-1 before
    the DA gate and must not see the target/future day) regardless of the
    panel's LOO/walk-forward toggle, which only controls the DA/IDA skill
    estimate. When the 9.2b batch keeps at least one day, the ceiling is taken
    from its ``total_global_ceiling_eur`` (so both rows share the walk-forward
    window + denominator); otherwise it falls back to the standalone 9.2a
    ceiling over the DA/IDA window. Returns Nones when ``reserve_series`` is
    missing/empty or no day is solvable.
    """
    out = {
        "triple_total": None, "realistic_total": None,
        "seq_per_day": None, "seq_summary": None,
        "triple_valid_days": None, "triple_da_baseline": None,
    }
    if reserve_series is None or reserve_series.empty or not valid_dates:
        return out
    per_day, summary = simulate_sequential_da_id_reserve_batch(
        primary_df, intraday_df, reserve_series,
        dates=sorted(valid_dates), tz=tz, power_mw=power_mw,
        duration_hours=duration_hours, efficiency=efficiency,
        bucket=bucket, forecast_mode="walk_forward",
    )
    if not per_day.empty:
        out.update(
            triple_total=summary["total_global_ceiling_eur"],
            realistic_total=summary["total_realised_eur"],
            triple_valid_days=summary["valid_days"],
            triple_da_baseline=summary["total_da_only_eur"],
            seq_per_day=per_day, seq_summary=summary,
        )
        return out
    # 9.2b excluded every day (e.g. a 1-day walk-forward window): fall back to
    # the standalone 9.2a ceiling over the DA/IDA window, still per-interval
    # priced. No realistic row in this degenerate case.
    ceiling = simulate_da_id_reserve_ceiling_batch(
        primary_df, intraday_df, reserve_series, dates=sorted(valid_dates),
        tz=tz, power_mw=power_mw, duration_hours=duration_hours,
        efficiency=efficiency,
    )
    if ceiling["solved_days"] > 0:
        out["triple_total"] = ceiling["total_eur"]
    return out


def _append_reserve_assumptions(
    assumptions: pd.DataFrame | None,
    *,
    product: str,
    capacity_price_eur_mw_h: float | None,
) -> pd.DataFrame | None:
    """Append reserve co-opt provenance rows to the export assumptions table."""
    if assumptions is None:
        return None
    price_txt = (
        f"{capacity_price_eur_mw_h:.2f}"
        if capacity_price_eur_mw_h is not None else "n/a"
    )
    rows = pd.DataFrame([
        {
            "parameter": "Reserve co-opt product",
            "value": str(product),
            "unit": "",
            "source": "Loaded ancillary capacity data",
            "affects": "Strategy comparison reserve (4th) row",
        },
        {
            "parameter": "Reserve capacity price",
            "value": price_txt,
            "unit": "EUR/MW/h",
            "source": "Duration-weighted mean of loaded capacity prices",
            "affects": "Reserve co-opt headroom payment",
        },
        {
            "parameter": "Reserve availability haircut",
            "value": f"{ANCILLARY_CAPACITY_AVAILABILITY:.2f}",
            "unit": "fraction",
            "source": "config.ANCILLARY_CAPACITY_AVAILABILITY",
            "affects": "Reserve capacity payment",
        },
        {
            "parameter": "Reserve activation energy",
            "value": "not modelled",
            "unit": "",
            "source": "solve_joint_capacity_batch scope",
            "affects": "Reserve co-opt is capacity headroom only",
        },
        {
            "parameter": "Reserve additivity with DA",
            "value": "co-opt headroom (not additive)",
            "unit": "",
            "source": "Joint MILP power-balance",
            "affects": "Reserve competes with DA for power; total is a joint optimum",
        },
    ])
    return pd.concat([assumptions, rows], ignore_index=True)


def _append_triple_assumptions(
    assumptions: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Append DA+IDA+reserve ceiling provenance to the export assumptions."""
    if assumptions is None:
        return None
    rows = pd.DataFrame([{
        "parameter": "DA+IDA1+reserve row type",
        "value": "perfect-foresight ceiling",
        "unit": "",
        "source": "solve_daily_da_id_reserve_dispatch",
        "affects": (
            "Cumulative ceiling: full DA/IDA/capacity knowledge, capacity "
            "headroom only, no activation energy, NOT forecast-driven"
        ),
    }])
    return pd.concat([assumptions, rows], ignore_index=True)


def _append_realistic_triple_assumptions(
    assumptions: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Append DA+IDA+reserve forecast-driven (9.2b) provenance to the export."""
    if assumptions is None:
        return None
    rows = pd.DataFrame([
        {
            "parameter": "DA+IDA1+reserve (forecast-driven) row type",
            "value": "reserve-first sequential, forecast-driven",
            "unit": "",
            "source": "simulate_sequential_da_id_reserve_batch",
            "affects": (
                "9.2b realistic row: reserve committed D-1 BEFORE the DA gate "
                "(reserve-first); DA remains a financial commitment, while "
                "physical IDA execution/rebid is capped by reserved headroom"
            ),
        },
        {
            "parameter": "Reserve commitment information",
            "value": "walk-forward (prior days only)",
            "unit": "",
            "source": "build_reserve_price_forecast forecast_mode",
            "affects": (
                "Reserve price forecast sees no target/future day — a "
                "real-time D-1 commitment, not an unbiased skill estimate"
            ),
        },
        {
            "parameter": "Forecast effect sign",
            "value": "signed (negative = forecast helped)",
            "unit": "",
            "source": "full gap = forecast effect + timing cost",
            "affects": "Gap attribution is not clamped",
        },
        {
            "parameter": "Reserve energy (9.2b)",
            "value": "not modelled",
            "unit": "",
            "source": "solve_sequential_da_id_reserve_dispatch scope",
            "affects": "Reserve is capacity headroom only; activation energy excluded",
        },
    ])
    return pd.concat([assumptions, rows], ignore_index=True)


def _stochastic_rebid_cap_mw(cap_pct: float, power_mw: float) -> float:
    """Convert the %-of-power rebid-cap slider to an absolute MW cap.

    The stochastic panel is DA+IDA1 only (no committed reserve), so the batch's
    ``rebid_cap >= max_reserve`` feasibility domain reduces to ``>= 0`` and the
    cap is simply a non-negative fraction of power. Pure helper so the
    conversion is unit-tested.
    """
    return max(0.0, float(cap_pct) / 100.0 * float(power_mw))


def _run_stochastic_policy_batch(
    primary_df: pd.DataFrame,
    intraday_df: pd.DataFrame | None,
    capacity_df: pd.DataFrame | None,
    reserve_product: str | None,
    *,
    valid_dates: set,
    tz: str | None,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    bucket: str,
    forecast_mode: str,
    rebid_cap_mw: float,
    min_rebid_uplift_eur: float,
) -> tuple[pd.DataFrame, dict, bool]:
    """Route the opt-in stochastic policy run (v2 §5, the §6-1.1 routing pin).

    Reserve mode is active iff the selected product's per-interval capacity
    series is non-empty AFTER windowing to the valid dates. The router OWNS
    that predicate's production chain — ``_slice_to_local_dates`` then
    ``capacity_price_series_for_product``, the exact rule the 5th/6th strategy
    rows use (NOT the full-sample availability check) — so a call site cannot
    accidentally feed it an unwindowed frame (Codex review, PR #49). When
    active, the v2 reserve-mode batch runs INSTEAD of the v1 DA+IDA1 batch;
    otherwise the LITERAL v1 path runs and no v2 machinery is invoked —
    equality by routing, not by numerics. In reserve mode the panel's
    forecast-mode toggle and deadband are deliberately NOT forwarded: the v2
    batch forces walk-forward and an inert deadband by construction
    (§2.3/§3), and its signature accepts neither knob.

    Returns ``(per_day, summary, reserve_mode)``.
    """
    window_anc = _slice_to_local_dates(capacity_df, valid_dates, tz)
    reserve_series = capacity_price_series_for_product(window_anc, reserve_product)
    reserve_mode = reserve_series is not None and not reserve_series.empty
    dates = sorted(valid_dates)
    if reserve_mode:
        per_day, summary = simulate_stochastic_triple_batch(
            primary_df, intraday_df, reserve_series, dates=dates, tz=tz,
            power_mw=power_mw, duration_hours=duration_hours,
            efficiency=efficiency, n_scenarios=_STOCHASTIC_N_SCENARIOS,
            bucket=bucket, seed=_STOCHASTIC_SEED, rebid_cap_mw=rebid_cap_mw,
        )
    else:
        per_day, summary = simulate_stochastic_da_id_batch(
            primary_df, intraday_df, dates=dates, tz=tz, power_mw=power_mw,
            duration_hours=duration_hours, efficiency=efficiency,
            n_scenarios=_STOCHASTIC_N_SCENARIOS, bucket=bucket,
            forecast_mode=forecast_mode, seed=_STOCHASTIC_SEED,
            rebid_cap_mw=rebid_cap_mw,
            min_rebid_uplift_eur=min_rebid_uplift_eur,
        )
    return per_day, summary, reserve_mode


def _append_stochastic_assumptions(
    assumptions: pd.DataFrame | None, *, summary: dict,
    reserve_mode: bool = False,
) -> pd.DataFrame | None:
    """Append stochastic policy-value provenance to the export assumptions."""
    if assumptions is None:
        return None
    cap = summary.get("rebid_cap_mw")
    cap_txt = "inf" if cap is None or not math.isfinite(float(cap)) else f"{float(cap):.1f}"
    rows = pd.DataFrame([
        {
            "parameter": "Stochastic policy value basis",
            "value": (
                "stochastic_realised - capped_9.2b_reserve_first_realised "
                "(both incl. reserve capacity at realised prices)"
                if reserve_mode
                else "stochastic_realised - capped_myopic_realised"
            ),
            "unit": "EUR (window)",
            "source": (
                "simulate_stochastic_triple_batch" if reserve_mode
                else "simulate_stochastic_da_id_batch"
            ),
            "affects": (
                "Robust headline DELTA (not a revenue total); common rebid cap, "
                "same valid-day window; screening diagnostic, NOT bankable"
            ),
        },
        {
            "parameter": "Stochastic rebid cap",
            "value": cap_txt,
            "unit": "MW",
            "source": "Cockpit stochastic panel (% of power)",
            "affects": "|stage2_net - da_net| <= cap (the load-bearing coupling)",
        },
        {
            "parameter": "Stochastic scenarios / seed",
            "value": f"S={summary.get('n_scenarios', '')}, seed={_STOCHASTIC_SEED}",
            "unit": "",
            "source": "build_ida_scenarios error_resample",
            "affects": "Scenario count + reproducible seed for the commitment MILP",
        },
        {
            "parameter": "Commitment/distribution split",
            "value": "diagnostic; stability reported per day",
            "unit": "",
            "source": "canonical_tiebreak_applied / tiebreak_stable",
            "affects": (
                "Use the tiebreak_stable result column; fallback days make only "
                "the split tie-sensitive, not the policy_value total"
            ),
        },
    ])
    if reserve_mode:
        rows = pd.concat([rows, _reserve_mode_stochastic_rows()], ignore_index=True)
    return pd.concat([assumptions, rows], ignore_index=True)


def _reserve_mode_stochastic_rows() -> pd.DataFrame:
    """v2 reserve-mode provenance rows for the export assumptions (§5)."""
    return pd.DataFrame([
        {
            "parameter": "Reserve-mode information set",
            "value": "walk-forward FORCED (DA forecast, reserve forecast, IDA scenarios)",
            "unit": "",
            "source": "simulate_stochastic_triple_batch (no forecast_mode knob)",
            "affects": (
                "The reserve gate is a real-time commitment: the panel's LOO "
                "toggle is ignored; the window loses its first day(s)"
            ),
        },
        {
            "parameter": "Reserve-mode deadband",
            "value": "inert (re-dispatches on all days, all arms)",
            "unit": "",
            "source": "v2 contract §3",
            "affects": "The rebid-deadband knob is ignored in reserve mode",
        },
        {
            "parameter": "Reserve granularity & zero-price skip",
            "value": (
                "per-interval r* (screening, not a 4h product commitment); "
                "a day with no positive forecast fee commits r* = 0"
            ),
            "unit": "MW",
            "source": "solve_stochastic_reserve_commitment",
            "affects": "Committed reserve pattern; per-arm avg in the per-day sheet",
        },
        {
            "parameter": "Reserve availability haircut",
            "value": f"{ANCILLARY_CAPACITY_AVAILABILITY:.2f}",
            "unit": "fraction",
            "source": "config.ANCILLARY_CAPACITY_AVAILABILITY",
            "affects": "Capacity fee on both the commitment objective and settlement",
        },
        {
            "parameter": "Stage-0 expected objective",
            "value": "DIAGNOSTIC only",
            "unit": "",
            "source": "v2 contract §2.4",
            "affects": (
                "Never a pinned identity or a UI revenue number; realised "
                "identities anchor post-execution"
            ),
        },
        {
            "parameter": "Stage-0 tie-break stability",
            "value": "per-day stage0_tiebreak_stable column; headline-level",
            "unit": "",
            "source": "canonical Stage-0 selector (v2 contract §2.2)",
            "affects": (
                "Stage-0 ties hit the HEADLINE (not just the split); fallback "
                "days are counted in n_stage0_fallback_days"
            ),
        },
    ])


def _reserve_history_for_product(
    anc_df: pd.DataFrame | None, product: str,
) -> pd.DataFrame:
    """One reserve product's capacity-price series for the skill report."""
    if anc_df is None or anc_df.empty or "product_type" not in anc_df:
        return pd.DataFrame()
    if RESERVE_VALUE_COL not in anc_df:
        return pd.DataFrame()
    labels = anc_df["product_type"].fillna("UNKNOWN").astype(str).str.strip()
    return anc_df[labels == str(product).strip()][[RESERVE_VALUE_COL]]


def _render_reserve_forecast_skill_section(
    *, anc_df: pd.DataFrame | None, zone_tz: str, chart_template: str,
) -> None:
    """Reserve capacity-price forecastability diagnostic (Phase 9.2b prep)."""
    products = list_capacity_products(anc_df)
    if not products:
        return
    with st.expander("Reserve price forecast skill (Phase 9.2b prep)", expanded=False):
        st.caption(
            "Can reserve capacity prices be forecast? Block-of-day climatology "
            "(6x 4h FCR/aFRR product blocks) scored against a flat sample-mean "
            "baseline. A skill diagnostic for the planned forecast-driven reserve "
            "commitment (9.2b commits reserve BEFORE DA, under a price forecast) "
            "— not a dispatch model; capacity headroom only, no activation energy."
        )
        c1, c2 = st.columns([1.4, 1.0])
        product = c1.selectbox(
            "Reserve product", options=products, index=0, key="reserve_skill_product",
        )
        mode_label = c2.selectbox(
            "Forecast information",
            options=["LOO cross-validation", "Walk-forward"],
            index=0,
            key="reserve_skill_mode",
            help=(
                "LOO uses every loaded day except the target (unbiased skill, "
                "may use future days). Walk-forward uses only prior days."
            ),
        )
        forecast_mode = "walk_forward" if mode_label == "Walk-forward" else "loo"
        history = _reserve_history_for_product(anc_df, product)
        skill = compute_reserve_forecast_skill(
            history, tz=zone_tz, forecast_mode=forecast_mode,
        )
        if skill["n_points"] == 0:
            st.info(
                "Not enough capacity-price history for this product to score a "
                "forecast (need at least two local days for leave-one-out)."
            )
            return
        cols = st.columns(4)
        cols[0].metric("MAE", f"EUR {skill['mae']:.2f}/MW/h")
        cols[1].metric("Bias", f"EUR {skill['bias']:+.2f}/MW/h")
        cols[2].metric("RMSE", f"EUR {skill['rmse']:.2f}/MW/h")
        sk = skill["skill_vs_mean"]
        cols[3].metric(
            "Skill vs flat mean", "n/a" if sk is None else f"{sk:+.0%}",
        )
        st.caption(
            f"Over {skill['n_points']:,} intervals; realised std EUR "
            f"{skill['realised_std']:.2f}/MW/h; {skill['n_blocks_filled']}/"
            f"{skill['n_blocks_requested']} blocks history-backed (coverage "
            f"{skill['coverage']:.0%}). Skill vs flat mean >0 means block-of-day "
            "climatology beats a flat average — reserve prices carry forecastable "
            "structure worth a forecast-driven commitment; <=0 means it does not, "
            "so 9.2b's reserve forecast would rest on thin ice."
        )
        by_block = skill["by_block"]
        if not by_block.empty:
            labels = [
                f"{int(b) * 4:02d}-{int(b) * 4 + 4:02d}" for b in by_block["block"]
            ]
            fig = go.Figure(go.Bar(
                x=labels, y=by_block["mae"], marker_color=_C_AVAIL,
            ))
            _apply_panel_layout(
                fig, "Forecast MAE by 4h block", "EUR/MW/h", chart_template,
                height=240,
            )
            fig.update_xaxes(title="Local 4h block")
            st.plotly_chart(fig, width="stretch")


def _render_activation_overlay_section(
    *, primary_zone: str | None, dates: list, zone_tz: str, power_mw: float,
) -> None:
    """Activation-energy replay overlay (Step 3c-2b).

    Gated on imported activation data for the zone. The energy leg of reserves —
    a historical REPLAY OVERLAY only: not co-optimized with DA/IDA/reserve, not
    additive to the strategy-comparison total, no SoC coupling, not aggregator
    dispatch. The capture-share knob is this asset's assumed slice of the SYSTEM
    activated volume; the headline figure is windowed to the loaded dates.
    """
    if not primary_zone:
        return
    activation = read_activation_cache(primary_zone)
    if activation is None or activation.empty:
        return
    with st.expander("Activation-energy overlay (historical replay)", expanded=False):
        st.caption(
            "Energy-leg cash flow if this asset had provided reserve and been "
            "activated pro-rata to the SYSTEM activated volume in the imported "
            "data. A historical REPLAY OVERLAY: NOT co-optimized with DA/IDA/"
            "reserve, NOT additive to the strategy-comparison total, no SoC "
            "coupling, not aggregator dispatch. Assumes a regular interval series "
            "(each row = average activated MW for that interval)."
        )
        pct = st.slider(
            "Activation capture share",
            min_value=0.0, max_value=10.0, value=1.0, step=0.5, format="%.1f%%",
            key="activation_capture_share_pct",
            help=(
                "This asset's assumed slice of the SYSTEM activated volume. "
                "Default 1% — for a single-site BESS this is a small screening "
                "assumption; raise it only for an aggregator/portfolio view. "
                f"Delivered power is capped by the committed {power_mw:.0f} MW."
            ),
        )
        capture_share = pct / 100.0
        windowed = _slice_to_local_dates(activation, set(dates), zone_tz)
        if windowed is None or windowed.empty:
            st.info(
                "No activation rows fall within the loaded simulation dates; the "
                "imported activation data covers a different window."
            )
            return
        result = compute_activation_overlay(
            windowed, reserve_mw=power_mw, capture_share=capture_share,
        )
        c1, c2 = st.columns([1.0, 1.0])
        c1.metric(
            "Activation-energy overlay",
            f"EUR {result['activation_energy_overlay_eur']:,.0f}",
        )
        c2.metric("Capture share", f"{capture_share:.1%}")
        st.caption(
            f"Over the loaded window, assuming the full {power_mw:.0f} MW could be "
            "committed to reserve. A SEPARATE, non-additive estimate from the "
            "strategy comparison above — not a strategy revenue row."
        )
        by_stream = result["by_stream"]
        if not by_stream.empty:
            st.dataframe(
                by_stream,
                width="stretch",
                hide_index=True,
                column_config={
                    "product": "Product",
                    "direction": "Direction",
                    "activated_mwh": st.column_config.NumberColumn(
                        "Activated MWh", format="%.1f",
                    ),
                    "activation_overlay_eur": st.column_config.NumberColumn(
                        "Overlay EUR", format="%.0f",
                    ),
                },
            )


def _render_imbalance_overlay_section(
    *, primary_zone: str | None, dates: list, zone_tz: str, power_mw: float,
) -> None:
    """Passive reBAP / imbalance-settlement replay overlay (Step 4d-2).

    Gated on imported imbalance data for the zone. Historical replay only:
    not co-optimized with DA/IDA/reserve, not additive to the strategy-comparison
    total, no SoC/energy sustainability coupling, and not live BRP control.
    """
    if not primary_zone:
        return
    imbalance = read_imbalance_cache(primary_zone)
    if imbalance is None or imbalance.empty:
        return
    with st.expander("reBAP / imbalance overlay (historical replay)", expanded=False):
        st.caption(
            "Passive imbalance-settlement cash flow if this asset/portfolio held "
            "a small position that helps the SYSTEM imbalance. Uses the German "
            "Netztransparenz convention: positive system imbalance = system short "
            "(discharge helps), negative = system long (charge helps). A historical "
            "REPLAY OVERLAY only: NOT co-optimized with DA/IDA/reserve, NOT "
            "additive to the strategy-comparison total, ignores SoC/energy "
            "sustainability, and is NOT live BRP control or aggregator dispatch."
        )
        pct = st.slider(
            "Imbalance capture share",
            min_value=0.0, max_value=10.0, value=1.0, step=0.5, format="%.1f%%",
            key="imbalance_capture_share_pct",
            help=(
                "This asset/portfolio's assumed slice of the SYSTEM imbalance "
                "magnitude. Default 1% — a conservative screening assumption. "
                f"The signed position is capped by the BESS {power_mw:.0f} MW "
                "power rating."
            ),
        )
        capture_share = pct / 100.0
        windowed = _slice_to_local_dates(imbalance, set(dates), zone_tz)
        if windowed is None or windowed.empty:
            st.info(
                "No imbalance rows fall within the loaded simulation dates; the "
                "imported reBAP/imbalance data covers a different window."
            )
            return
        result = compute_imbalance_overlay(
            windowed, power_mw=power_mw, capture_share=capture_share,
        )
        c1, c2 = st.columns([1.0, 1.0])
        c1.metric(
            "Imbalance settlement overlay",
            f"EUR {result['imbalance_settlement_overlay_eur']:,.0f}",
        )
        c2.metric("Capture share", f"{capture_share:.1%}")
        st.caption(
            f"Over the loaded window, signed BESS net dispatch is capped by "
            f"{power_mw:.0f} MW and priced at the published signed reBAP/"
            "imbalance price. A SEPARATE, non-additive diagnostic — not a "
            "strategy revenue row."
        )
        by_state = result["by_system_state"]
        if not by_state.empty:
            st.dataframe(
                by_state,
                width="stretch",
                hide_index=True,
                column_config={
                    "system_state": "System state",
                    "asset_imbalance_mwh": st.column_config.NumberColumn(
                        "Asset imbalance MWh", format="%.1f",
                    ),
                    "imbalance_overlay_eur": st.column_config.NumberColumn(
                        "Overlay EUR", format="%.0f",
                    ),
                },
            )


def _render_forecast_policy_section(
    *,
    primary_df: pd.DataFrame,
    intraday_df: pd.DataFrame | None,
    anc_df: pd.DataFrame | None,
    primary_zone: str,
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

        # Cache-first: imported unified capacity (SQLite) drives the reserve rows;
        # fall back to the session ancillary only when the cache is empty.
        capacity_df, capacity_source = _resolve_capacity_dataset(primary_zone, anc_df)
        capacity_products = list_capacity_products(capacity_df)

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
        reserve_product = None
        if capacity_products:
            reserve_product = st.selectbox(
                "Reserve product (adds reserve strategy rows)",
                options=capacity_products,
                index=0,
                key="forecast_policy_reserve_product",
                help=(
                    "Adds reserve-capacity strategy rows when data is "
                    "available: a DA + reserve headroom row, plus cumulative "
                    "DA + IDA1 + reserve ceiling/forecast-driven rows when "
                    "IDA and capacity prices overlap. Capacity headroom only; "
                    "no activation energy."
                ),
            )
            st.caption(f"Reserve capacity source: {capacity_source}.")

        include_stochastic = st.checkbox(
            "Include stochastic policy (scenario-aware DA commitment, slower)",
            value=False,
            key="forecast_policy_stochastic",
            help=(
                "Adds a Stochastic policy value row + attribution/risk panel: "
                "the scenario-aware DA commitment (anticipates the IDA rebid "
                "across a scenario set) vs the capped-myopic baseline, at a "
                "common rebid cap. Runs an extra MILP per day (~1-2 s/day), so "
                "it is opt-in."
            ),
        )
        stochastic_cap_pct = 50
        if include_stochastic:
            stochastic_cap_pct = st.slider(
                "IDA rebid cap (% of power)",
                min_value=0, max_value=100, value=50, step=5,
                key="forecast_policy_stochastic_cap",
                help=(
                    "The load-bearing coupling |stage2_net - da_net| <= cap. A "
                    "finite cap is where the scenario-aware commitment earns its "
                    "value; at 100% the rebid can largely undo the DA commit. "
                    "DA+IDA1 only unless reserve mode is active (see the note "
                    "below), in which case the same cap governs the v2 "
                    "DA+IDA1+reserve batch."
                ),
            )
            st.caption(
                "Reserve mode: when the selected reserve product has capacity "
                "prices INSIDE the comparison window (the same rule as the "
                "reserve strategy rows), the stochastic run switches to the v2 "
                "DA+IDA1+reserve batch — which IGNORES the rebid deadband "
                "(re-dispatches every day) and forces WALK-FORWARD forecasts "
                "regardless of the forecast-mode toggle. The deadband/LOO "
                "settings above apply only while the DA+IDA1-only path is "
                "active."
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

        valid_dates = set(per_day["date"])
        reserve_total, reserve_label, reserve_price = _reserve_coopt_total(
            primary_df,
            reserve_product,
            capacity_df,
            valid_dates=valid_dates,
            tz=zone_tz,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
        )
        # Rows 5 & 6: 9.2a perfect-foresight ceiling and 9.2b forecast-driven
        # realistic triple, BOTH priced off the SAME per-interval reserve series
        # (windowed to the DA+ID valid dates so an out-of-window print can't
        # leak), scored over the 9.2b walk-forward window. capacity_df is
        # cache-first (imported unified capacity), else the session ancillary.
        window_anc = _slice_to_local_dates(capacity_df, valid_dates, zone_tz)
        reserve_series = capacity_price_series_for_product(window_anc, reserve_product)
        triple = _reserve_triple_totals(
            primary_df, intraday_df, reserve_series,
            valid_dates=valid_dates, tz=zone_tz, power_mw=power_mw,
            duration_hours=duration_hours, efficiency=efficiency, bucket=bucket,
        )
        triple_label = (
            f"DA + IDA1 + {reserve_product} (co-opt ceiling)"
            if triple["triple_total"] is not None else None
        )
        realistic_label = (
            f"DA + IDA1 + {reserve_product} (forecast-driven realistic)"
            if triple["realistic_total"] is not None else None
        )
        # Opt-in stochastic policy value: run the 3-policy batch (capped-myopic
        # / co-opt / stochastic) over the SAME valid-day window at a common rebid
        # cap, then thread its robust policy_value into the comparison as ONE
        # delta row. DA+IDA1 only (no reserve) — reserve cancels in the delta and
        # is the separate triple rows. NO custom label: the default
        # STOCHASTIC_POLICY_VALUE_LABEL is what excludes the row from the bar
        # chart (see _strategy_chart_rows; Increment D guardrail).
        stoch_per_day = None
        stoch_summary = None
        stoch_reserve_mode = False
        if include_stochastic:
            with st.spinner(
                f"Solving {len(valid_dates)} day(s) x 3 policies "
                "(capped-myopic / co-opt / stochastic)..."
            ):
                stoch_per_day, stoch_summary, stoch_reserve_mode = (
                    _run_stochastic_policy_batch(
                        primary_df,
                        intraday_df,
                        capacity_df,
                        reserve_product,
                        valid_dates=valid_dates,
                        tz=zone_tz,
                        power_mw=power_mw,
                        duration_hours=duration_hours,
                        efficiency=efficiency,
                        bucket=bucket,
                        forecast_mode=forecast_mode,
                        rebid_cap_mw=_stochastic_rebid_cap_mw(
                            stochastic_cap_pct, power_mw,
                        ),
                        min_rebid_uplift_eur=min_rebid_uplift_eur,
                    )
                )
        policy_value_total = None
        policy_value_valid_days = None
        policy_value_label = None
        if stoch_summary is not None and stoch_summary["valid_days"] > 0:
            # One delta row, label switches with the routing (§5): the reserve-
            # mode baseline is the capped 9.2b reserve-first policy — a
            # DIFFERENT number from the v1 capped-myopic — and both labels are
            # excluded from the bar chart by _strategy_chart_rows.
            if stoch_reserve_mode:
                policy_value_total = stoch_summary["total_policy_value_v2_eur"]
                policy_value_label = STOCHASTIC_POLICY_VALUE_RESERVE_LABEL
            else:
                policy_value_total = stoch_summary["total_policy_value_eur"]
            policy_value_valid_days = stoch_summary["valid_days"]
        comparison = build_strategy_comparison(
            summary,
            power_mw=power_mw,
            reserve_coopt_total=reserve_total,
            reserve_label=reserve_label,
            triple_joint_total=triple["triple_total"],
            triple_joint_label=triple_label,
            realistic_triple_total=triple["realistic_total"],
            realistic_triple_label=realistic_label,
            triple_valid_days=triple["triple_valid_days"],
            triple_da_baseline=triple["triple_da_baseline"],
            policy_value_total=policy_value_total,
            policy_value_valid_days=policy_value_valid_days,
            policy_value_label=policy_value_label,
        )
        _render_strategy_comparison(
            comparison, chart_template,
            has_reserve=reserve_total is not None,
            has_triple=triple["triple_total"] is not None,
            has_realistic=triple["realistic_total"] is not None,
        )
        if reserve_product and reserve_total is None:
            st.caption(
                f"Reserve row omitted: no {reserve_product} capacity price "
                "overlaps this comparison window, so it cannot be priced "
                "co-temporally with the DA+ID rows."
            )
        if stoch_summary is not None and stoch_summary["valid_days"] > 0:
            _render_stochastic_attribution_panel(
                stoch_summary, power_mw=power_mw,
                reserve_mode=stoch_reserve_mode,
            )
        if triple["seq_summary"] is not None:
            _render_reserve_gap_panel(
                triple["seq_summary"], reserve_product, chart_template,
            )
        st.dataframe(per_day, width="stretch", hide_index=True)
        # This panel reports raw solver values — no capture haircut applied.
        export_assumptions = _cockpit_export_assumptions(
            assumptions,
            capture_label="Capture haircut",
            capture_value="not applied",
            capture_affects=(
                "The forecast-policy panel reports raw solver values; no "
                "capture haircut is applied"
            ),
        )
        if reserve_total is not None:
            export_assumptions = _append_reserve_assumptions(
                export_assumptions,
                product=reserve_product,
                capacity_price_eur_mw_h=reserve_price,
            )
        if triple["triple_total"] is not None:
            export_assumptions = _append_triple_assumptions(export_assumptions)
        if triple["realistic_total"] is not None:
            export_assumptions = _append_realistic_triple_assumptions(export_assumptions)
        if stoch_summary is not None and stoch_summary["valid_days"] > 0:
            export_assumptions = _append_stochastic_assumptions(
                export_assumptions, summary=stoch_summary,
                reserve_mode=stoch_reserve_mode,
            )
        export_tables = {"Strategy comparison": comparison, "Sequential DA+ID": per_day}
        if triple["seq_per_day"] is not None and not triple["seq_per_day"].empty:
            export_tables["Sequential DA+ID+reserve"] = triple["seq_per_day"]
        if stoch_per_day is not None and not stoch_per_day.empty:
            export_tables["Stochastic policy (per day)"] = stoch_per_day
        _cockpit_download_button(
            export_tables,
            export_assumptions,
            key="dl_forecast_policy",
            label="\U0001f4e5 Download forecast policy + comparison (Excel)",
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


def _strategy_chart_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    """Rows to plot in the 'annualised revenue by strategy' bar chart.

    The stochastic policy-value row is a value DELTA (stochastic minus
    capped-myopic realised), NOT a revenue total, so it must never appear among
    the annualised-revenue bars — a small delta plotted next to strategy totals
    would misread as a tiny strategy. It stays in the table; the bar chart drops
    it. When the delta row is absent the frame passes through unchanged.
    """
    if comparison is None or comparison.empty:
        return comparison
    delta_labels = [
        STOCHASTIC_POLICY_VALUE_LABEL, STOCHASTIC_POLICY_VALUE_RESERVE_LABEL,
    ]
    return comparison[
        ~comparison["strategy"].isin(delta_labels)
    ].reset_index(drop=True)


def _render_strategy_comparison(
    comparison: pd.DataFrame, chart_template: str, *,
    has_reserve: bool = False, has_triple: bool = False,
    has_realistic: bool = False,
) -> None:
    """Investment-framed comparison of the dispatch strategies."""
    if comparison is None or comparison.empty:
        return
    st.markdown("**Strategy comparison (annualised)**")
    notes = []
    if has_reserve:
        notes.append(
            "The reserve co-opt row is a DIFFERENT stream (capacity headroom, "
            "no IDA): a headroom-aware estimate that competes with DA for power "
            "(so it is NOT additive with DA) and excludes activation energy."
        )
    if has_triple:
        notes.append(
            "The DA+IDA1+reserve (co-opt ceiling) row is the cumulative "
            "perfect-foresight ceiling (full DA/IDA/capacity knowledge, "
            "capacity headroom only, no activation energy, not forecast-driven)."
        )
    if has_realistic:
        notes.append(
            "The DA+IDA1+reserve (forecast-driven realistic) row is the 9.2b "
            "reserve-first walk-forward policy — it sits BELOW the ceiling by "
            "the forecast + commitment-timing gap (see the gap panel). It and "
            "the ceiling are scored over the walk-forward window, which may "
            "span one fewer day than the DA/IDA rows."
        )
    if not has_reserve and not has_triple:
        notes.append(
            "A reserve (FCR/aFRR) co-optimisation strategy adds further rows "
            "when ancillary capacity prices are loaded."
        )
    has_policy_value = bool(
        comparison["strategy"].isin([
            STOCHASTIC_POLICY_VALUE_LABEL, STOCHASTIC_POLICY_VALUE_RESERVE_LABEL,
        ]).any()
    )
    if has_policy_value:
        notes.append(
            "The Stochastic policy value row is a value DELTA (scenario-aware "
            "commitment minus the capped myopic baseline named in its label, "
            "at the common rebid cap over the same valid days), NOT a revenue "
            "total — it has no DA uplift and is excluded from the bar chart "
            "above; see the attribution & risk panel."
        )
    window_note = (
        "DA/IDA rows use the forecast-policy window; reserve-first triple rows "
        "use their own walk-forward window. "
        if has_realistic
        else "Same window, strategies side by side. "
    )
    st.caption(
        f"{window_note}Annualised EUR/MW/yr = window revenue x 365.25 / valid "
        "days / power; uplift is vs the matching DA-only baseline. Raw solver "
        f"values (no capture haircut). {' '.join(notes)}"
    )
    chart_rows = _strategy_chart_rows(comparison)
    palette = [_C_REVENUE, _C_PRICE_IDA, _C_DISCHARGE, _C_AVAIL, _C_SOC, _C_CHARGE]
    colors = [palette[i % len(palette)] for i in range(len(chart_rows))]
    fig = go.Figure(go.Bar(
        x=chart_rows["strategy"],
        y=chart_rows["annualized_eur_per_mw"],
        marker_color=colors,
        text=[
            _fmt_strategy_bar_label(v)
            for v in chart_rows["annualized_eur_per_mw"]
        ],
        textposition="outside",
    ))
    _apply_panel_layout(
        fig, "Annualised revenue by strategy", "EUR/MW/yr", chart_template,
        height=280,
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True,
        column_config={
            "strategy": "Strategy",
            "window_revenue_eur": st.column_config.NumberColumn(
                "Window EUR", format="%.0f",
            ),
            "annualized_eur_per_mw": st.column_config.NumberColumn(
                "EUR/MW/yr", format="%.0f",
            ),
            "uplift_vs_da_pct": st.column_config.NumberColumn(
                "Uplift vs DA", format="%.1f%%",
            ),
        },
    )


def _render_reserve_gap_panel(
    summary: dict, reserve_product: str, chart_template: str,
) -> None:
    """Decompose the 9.2b realistic-vs-ceiling gap (forecast effect + timing).

    The full gap (perfect-foresight ceiling - forecast-driven realistic) splits
    into a SIGNED forecast effect (negative = the forecast HELPED) and a
    commitment-timing cost (reserve is locked D-1 before the DA gate). Reserve =
    capacity headroom only; no activation energy.
    """
    if summary is None or summary.get("valid_days", 0) <= 0:
        return
    realistic = summary["total_realised_eur"]
    ceiling = summary["total_global_ceiling_eur"]
    forecast_effect = summary["total_forecast_effect_eur"]
    timing_cost = summary["total_timing_cost_eur"]
    full_gap = summary["total_full_gap_eur"]
    st.markdown(f"**Forecast-driven reserve gap ({reserve_product}, Phase 9.2b)**")
    cols = st.columns(4)
    cols[0].metric("Forecast-driven realistic", f"EUR {realistic:,.0f}")
    cols[1].metric("Perfect-foresight ceiling", f"EUR {ceiling:,.0f}")
    cols[2].metric("Full gap", f"EUR {full_gap:,.0f}")
    cols[3].metric("Forecast effect", f"EUR {forecast_effect:+,.0f}")
    st.caption(
        "Reserve-first sequential (walk-forward): reserve committed D-1 under a "
        "price forecast; DA remains a financial commitment; physical IDA "
        "execution/rebid is capped by reserved headroom. Full gap (ceiling - "
        f"realistic) = forecast effect + timing cost = "
        f"EUR {forecast_effect:+,.0f} + EUR {timing_cost:,.0f}. Forecast effect "
        "is SIGNED: negative means the forecast HELPED (realistic beat the "
        "no-skill split); timing cost is the D-1 commitment-ordering penalty. "
        "Capacity headroom only — no activation energy."
    )
    fig = go.Figure(go.Bar(
        x=["Timing cost", "Forecast effect"],
        y=[timing_cost, forecast_effect],
        marker_color=[_C_AVAIL, _C_PRICE_IDA],
        text=[f"{timing_cost:,.0f}", f"{forecast_effect:+,.0f}"],
        textposition="outside",
    ))
    _apply_panel_layout(
        fig, "Gap decomposition (ceiling - realistic)", "EUR (window)",
        chart_template, height=240,
    )
    st.plotly_chart(fig, width="stretch")


def _render_stochastic_attribution_panel(
    summary: dict, *, power_mw: float, reserve_mode: bool = False,
) -> None:
    """Attribution + risk for the stochastic policy value (Increment D).

    The headline ``policy_value`` (stochastic minus capped-myopic realised) is
    robust. Its commitment/distribution split is reliable only on days where the
    S=1 co-opt and S=N stochastic commitments both accepted the canonical
    Stage-1 tie-break; fallback days are flagged below. The risk block is a
    dispersion diagnostic over the pooled per-(day, scenario) energy totals; the
    objective stays risk-neutral.
    """
    if summary is None or summary.get("valid_days", 0) <= 0:
        return
    pv = summary[
        "total_policy_value_v2_eur" if reserve_mode else "total_policy_value_eur"
    ]
    days = int(summary["valid_days"])
    cap = summary.get("rebid_cap_mw")
    cap_txt = (
        "inf" if cap is None or not math.isfinite(float(cap))
        else f"{float(cap):,.1f} MW"
    )
    per_mw_yr = (
        pv * DAYS_PER_YEAR / days / power_mw
        if power_mw > 0 and days > 0 else float("nan")
    )
    st.markdown("**Stochastic policy value — attribution & risk**")
    cols = st.columns(3)
    cols[0].metric("Policy value (window)", f"EUR {pv:,.0f}")
    cols[1].metric("Annualised", f"EUR {per_mw_yr:,.0f}/MW/yr")
    cols[2].metric("Rebid cap", cap_txt)
    scope_note = (
        "Includes reserve capacity at REALISED prices on both sides; reserve "
        "committed walk-forward before the DA gate; per-interval reserve "
        "(screening, not a 4h product commitment); deadband inert."
        if reserve_mode
        else "DA+IDA1 only (reserve co-opt is the separate triple rows)."
    )
    st.caption(
        f"Scenario-aware commitment minus the capped myopic baseline, common "
        f"rebid cap {cap_txt}, over {days} valid day(s). A screening diagnostic, "
        "NOT a bankable co-optimised revenue. The split below is an attribution "
        f"diagnostic; trust it only on tie-stable days. {scope_note}"
    )
    fallback_days = int(summary.get("n_tiebreak_fallback_days", 0) or 0)
    stable_days = max(days - fallback_days, 0)
    split = st.columns(2)
    split[0].metric(
        "Commitment value (co-opt - myopic)",
        f"EUR {summary['total_commitment_value_eur']:,.0f}",
    )
    split[1].metric(
        "Distribution value (stochastic - co-opt)",
        f"EUR {summary['total_distribution_value_eur']:,.0f}",
    )
    st.caption(f"Stage-1 split tie-stable on {stable_days}/{days} days")
    if fallback_days > 0:
        st.warning(
            "Commitment/distribution split is tie-SENSITIVE on "
            f"{fallback_days}/{days} day(s): canonical tie-break fell back to "
            "the pass-1 solution. Trust only the robust policy-value headline "
            "for those days."
        )
    if reserve_mode:
        # Stage-0 ties hit the HEADLINE, not just the split (§2.2/§3), so the
        # warning attaches to the policy-value metric itself.
        s0_fallback = int(summary.get("n_stage0_fallback_days", 0) or 0)
        s0_stable = max(days - s0_fallback, 0)
        st.caption(f"Stage-0 (reserve) tie-stable on {s0_stable}/{days} days")
        if s0_fallback > 0:
            st.warning(
                f"HEADLINE is non-canonical on {s0_fallback}/{days} day(s): "
                "the Stage-0 reserve tie-break fell back to the pass-1 vector, "
                "so equal-optimal reserve commitments may settle to different "
                "realised money on those days — treat the policy value there "
                "as one member of a tie set, not a canonical number."
            )
    risk = summary.get("risk_block") or {}
    if risk.get("n", 0) > 0:
        rcols = st.columns(4)
        rcols[0].metric("P10", f"EUR {risk['p10']:,.0f}")
        rcols[1].metric("P50", f"EUR {risk['p50']:,.0f}")
        rcols[2].metric("P90", f"EUR {risk['p90']:,.0f}")
        rcols[3].metric("CVaR@90", f"EUR {risk['cvar90']:,.0f}")
        st.caption(
            f"Scenario risk pooled over {risk['n']:,} (day, scenario) energy "
            "totals (excludes the certain reserve-capacity constant). Dispersion "
            "diagnostic — the objective is risk-neutral; CVaR@90 is the downside "
            "tail (mean of the worst 10%), not an upper bound."
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
        colorscale=[
            [0.0, "#132033"],
            [0.35, "#0f766e"],
            [0.7, "#22c55e"],
            [1.0, "#facc15"],
        ],
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
