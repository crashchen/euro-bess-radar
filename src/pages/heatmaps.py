"""Tab 2: Heatmaps — price heatmap, spread heatmap, charge/discharge frequency."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import build_price_heatmap, build_spread_heatmap
from src.ui_theme import apply_cockpit_plot_theme


def render(
    primary_zone: str,
    primary_df: pd.DataFrame,
    duration_hours: int,
    zone_tz: str,
    chart_template: str,
    report_figures: dict[str, object],
) -> None:
    """Render the Heatmaps tab."""
    st.subheader(f"Heatmaps — {primary_zone}")

    price_hm = build_price_heatmap(primary_df, tz=zone_tz)
    fig_phm = px.imshow(
        price_hm,
        title=f"Average Price by Hour & Month ({zone_tz})",
        labels=dict(x="Month", y="Hour (local)", color="EUR/MWh"),
        color_continuous_scale=[
            [0.0, "#07111f"],
            [0.35, "#0c4f7d"],
            [0.72, "#00cfff"],
            [1.0, "#ff2d95"],
        ],
        aspect="auto",
        template=chart_template,
    )
    fig_phm.update_yaxes(dtick=1)
    apply_cockpit_plot_theme(fig_phm)
    report_figures["price_heatmap"] = fig_phm
    st.plotly_chart(fig_phm, width="stretch")
    st.caption(
        "Each cell is the average day-ahead price for that local hour (rows) in "
        "that calendar month (columns). Read it top-to-bottom for the daily "
        "shape — dark low-price bands are the cheapest hours to charge, bright "
        "high-price bands the best hours to discharge — and left-to-right for "
        "seasonal drift. A strong vertical contrast between the daily low and "
        "high bands is the arbitrage signal a battery monetises."
    )

    spread_hm = build_spread_heatmap(
        primary_df, tz=zone_tz, duration_hours=duration_hours,
    )
    fig_shm = px.imshow(
        spread_hm,
        title=f"Selected-Window Spread Signal ({duration_hours}h windows, {zone_tz})",
        labels=dict(x="Month", y="Hour (local)", color="Signed signal (EUR/MWh)"),
        color_continuous_scale=[
            [0.0, "#ff2d95"],
            [0.5, "#111925"],
            [1.0, "#00cfff"],
        ],
        color_continuous_midpoint=0,
        aspect="auto",
        template=chart_template,
    )
    fig_shm.update_yaxes(dtick=1)
    apply_cockpit_plot_theme(fig_shm)
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
        name="Charge", marker_color="#ff2d95",
    ))
    fig_freq.add_trace(go.Bar(
        x=freq_df["Hour"], y=freq_df["Discharge %"],
        name="Discharge", marker_color="#00a3ff",
    ))
    fig_freq.update_layout(
        title=f"Charge/Discharge Hour Selection Frequency ({duration_hours}h windows)",
        xaxis_title="Hour (local)",
        yaxis_title="% of months selected",
        barmode="group",
        template=chart_template,
    )
    apply_cockpit_plot_theme(fig_freq)
    st.plotly_chart(fig_freq, width="stretch")
    st.caption(
        "For each local hour, the share of months in which the ordered-spread "
        "model selected that hour for charging (pink) versus discharging (blue). "
        "Tall, well-separated pink and blue clusters mean the charge/discharge "
        "timing is consistent across the sample; overlap at the same hour means "
        "the optimal timing shifts month to month. This is selection frequency, "
        "not revenue."
    )
