"""Tab 1: Market Overview — price time series, spread bars, key metrics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import is_elexon_zone
from src.data_ingestion import summarize_price_data_quality
from src.ui_theme import apply_cockpit_plot_theme


def render(
    primary_zone: str,
    primary_df: pd.DataFrame,
    daily_spreads: pd.DataFrame,
    percentiles: dict[str, float],
    neg_stats: dict[str, float],
    duration_hours: int,
    zone_tz: str,
    chart_template: str,
    report_figures: dict[str, object],
) -> None:
    """Render the Market Overview tab."""
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
    quality = summarize_price_data_quality(primary_df)
    excluded_days = int(daily_spreads.attrs.get("excluded_days_due_to_missing", 0))
    if quality["missing_intervals"] > 0:
        st.warning(
            "Data quality: "
            f"{quality['imputed_ratio']:.1%} of intervals were short-gap imputed; "
            f"{quality['missing_ratio']:.1%} remain missing and are excluded from "
            f"spread/dispatch analytics across {excluded_days} local day(s)."
        )
    elif quality["imputed_intervals"] > 0:
        st.caption(
            "Data quality: "
            f"{quality['imputed_ratio']:.1%} of intervals were short-gap imputed; "
            "no unresolved price gaps remain."
        )

    price_plot_df = primary_df.reset_index()
    fig_price = px.line(
        price_plot_df,
        x="timestamp", y="price_eur_mwh",
        title="Day-Ahead Prices",
        labels={"price_eur_mwh": "EUR/MWh", "timestamp": ""},
        template=chart_template,
    )
    fig_price.update_traces(
        opacity=0.72,
        name="Hourly",
        showlegend=True,
        line=dict(color="#ff2d95", width=1.7),
    )
    ma_series = primary_df["price_eur_mwh"].rolling(
        window=24 * 30, min_periods=24,
    ).mean()
    fig_price.add_scatter(
        x=primary_df.index, y=ma_series,
        mode="lines", name="30-Day MA",
        line=dict(color="#d0d4dc", width=2.2),
    )
    fig_price.update_xaxes(rangeslider_visible=True)
    apply_cockpit_plot_theme(fig_price)
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
        color_continuous_scale=[
            [0.0, "#18233a"],
            [0.42, "#0c6b9e"],
            [0.72, "#00cfff"],
            [1.0, "#ff2d95"],
        ],
        template=chart_template,
    )
    fig_spread.update_xaxes(rangeslider_visible=True, type="date")
    apply_cockpit_plot_theme(fig_spread)
    report_figures["spread_ts"] = fig_spread
    st.plotly_chart(fig_spread, width="stretch")
