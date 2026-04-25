"""Tab 2: Heatmaps — price heatmap, spread heatmap, charge/discharge frequency."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import build_price_heatmap, build_spread_heatmap


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
