"""Tab 4: Renewable Correlation — RE-price correlation, scatter, quartile analysis."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import (
    analyze_renewable_bess_signal,
    build_renewable_price_scatter,
)
from src.components.sidebar import _format_data_error, load_generation
from src.data_ingestion import (
    DataSourceAuthError,
    DataSourceNetworkError,
    DataSourceParseError,
)

logger = logging.getLogger(__name__)


def render(
    primary_zone: str,
    primary_df: pd.DataFrame,
    start_date,
    end_date,
    duration_hours: int,
    zone_tz: str,
    chart_template: str,
    refresh_token: int,
) -> None:
    """Render the Renewable Correlation tab."""
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
        return

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
