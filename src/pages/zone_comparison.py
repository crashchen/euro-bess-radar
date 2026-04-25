"""Tab 5: Zone Comparison — multi-zone screening, risk/reward frontier, daily spread overlay."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import (
    calculate_daily_spreads,
    compare_zones,
)
from src.config import ZONE_TIMEZONES, get_zone_timezone
from src.export import export_comparison_to_bytes


def render(
    zone_data: dict[str, pd.DataFrame],
    duration_hours: int,
    capture_rate: float,
    efficiency: float,
    power_mw: float,
    use_lp_dispatch: bool,
    capex_eur_kwh: float,
    chart_template: str,
) -> None:
    """Render the Zone Comparison tab."""
    if len(zone_data) < 2:
        st.info("Select multiple zones to see comparison.")
        return

    st.subheader("Zone Comparison")
    comp = compare_zones(
        zone_data,
        zone_timezones=ZONE_TIMEZONES,
        duration_hours=duration_hours,
        capture_rate=capture_rate,
        roundtrip_efficiency=efficiency,
        power_mw=power_mw,
        use_lp_dispatch=use_lp_dispatch,
        capex_eur_kwh=capex_eur_kwh,
    )
    if comp.empty:
        st.warning(
            "No comparison rows could be built from the fetched zone data."
        )
        return

    has_degradation = "net_revenue_per_mw" in comp.columns
    sort_col = (
        "net_revenue_per_mw" if has_degradation
        else "estimated_annual_revenue_per_mw"
    )
    comp = comp.sort_values(
        sort_col, ascending=False,
    ).reset_index(drop=True)

    # Risk/Reward scatter
    y_col = sort_col
    y_label = (
        "Net Revenue (EUR/MW/yr)" if has_degradation
        else "Est. Annual Revenue (EUR/MW/yr)"
    )
    scatter_kwargs: dict = {
        "x": "p90_spread",
        "y": y_col,
        "text": "zone",
        "title": "Zone Screening: Risk/Reward Frontier",
        "labels": {
            "p90_spread": "P90 Spread (EUR/MWh)",
            y_col: y_label,
        },
        "template": chart_template,
    }
    if has_degradation and comp["lcos_eur_mwh"].notna().any():
        scatter_kwargs["size"] = "lcos_eur_mwh"
        scatter_kwargs["size_max"] = 40
        scatter_kwargs["labels"]["lcos_eur_mwh"] = "LCOS (EUR/MWh)"
    else:
        scatter_kwargs["size"] = "negative_pct"
        scatter_kwargs["size_max"] = 40
        scatter_kwargs["labels"]["negative_pct"] = "Negative Price %"

    fig_rr = px.scatter(comp, **scatter_kwargs)
    fig_rr.update_traces(textposition="top center")
    st.plotly_chart(fig_rr, width="stretch")

    # Numeric table
    col_config: dict = {
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
        "dispatch_method": "Dispatch",
    }
    if has_degradation:
        col_config.update({
            "avg_cycles_per_day": st.column_config.NumberColumn(
                "Cycles/Day", format="%.2f",
            ),
            "net_revenue_per_mw": st.column_config.NumberColumn(
                "Net Rev (EUR/MW/yr)", format="\u20ac%,.0f",
            ),
            "lcos_eur_mwh": st.column_config.NumberColumn(
                "LCOS", format="\u20ac%.1f/MWh",
            ),
            "payback_years": st.column_config.NumberColumn(
                "Payback", format="%.1f yr",
            ),
            "effective_life_years": st.column_config.NumberColumn(
                "Lifetime", format="%.1f yr",
            ),
            "limiting_factor": "Limit",
        })

    st.dataframe(
        comp,
        width="stretch",
        hide_index=True,
        column_config=col_config,
    )

    # Download comparison
    comp_xlsx = export_comparison_to_bytes(comp)
    st.download_button(
        label="Download comparison (Excel)",
        data=comp_xlsx,
        file_name="zone_comparison.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Daily spread comparison chart
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
