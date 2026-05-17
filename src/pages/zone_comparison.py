"""Tab 5: Zone Comparison — multi-zone screening, risk/reward frontier, daily spread overlay."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    calculate_daily_spreads,
    compare_zones,
)
from src.config import ZONE_TIMEZONES, get_zone_timezone
from src.export import export_comparison_to_bytes
from src.portfolio import (
    build_daily_revenue_matrix,
    compute_correlation_matrix,
    compute_efficient_frontier,
    compute_max_sharpe_portfolio,
    compute_min_variance_portfolio,
    compute_zone_stats,
)


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

    _render_portfolio_section(
        zone_data=zone_data,
        duration_hours=duration_hours,
        power_mw=power_mw,
        efficiency=efficiency,
        capture_rate=capture_rate,
        use_lp_dispatch=use_lp_dispatch,
        chart_template=chart_template,
    )


def _render_portfolio_section(
    *,
    zone_data: dict[str, pd.DataFrame],
    duration_hours: int,
    power_mw: float,
    efficiency: float,
    capture_rate: float,
    use_lp_dispatch: bool,
    chart_template: str,
) -> None:
    """Cross-zone diversification: correlation, Sharpe, efficient frontier."""
    with st.expander("Portfolio Analysis (cross-zone diversification)", expanded=False):
        st.caption(
            "Treats each zone as a daily revenue-per-MW series and reports "
            "pairwise correlation, per-zone Sharpe-like ratio, and the "
            "long-only Markowitz efficient frontier. Weights are MW shares "
            "(sum = 1, all ≥ 0)."
        )

        rev_df = build_daily_revenue_matrix(
            zone_data,
            zone_timezones=ZONE_TIMEZONES,
            duration_hours=duration_hours,
            power_mw=power_mw,
            efficiency=efficiency,
            capture_rate=capture_rate,
            use_lp_dispatch=use_lp_dispatch,
        )
        if rev_df.empty:
            # Distinguish "no zone has a complete day" from "zones exist but
            # share no overlapping dates" — the user fix differs (fetch
            # longer range vs widen zone selection).
            if any((df is not None and not df.empty) for df in zone_data.values()):
                st.info(
                    "Selected zones have no overlapping complete local days. "
                    "Try widening the date range so the zones share a sample."
                )
            else:
                st.info("Need at least one zone with a complete day of data.")
            return
        if rev_df.shape[1] < 2:
            st.info(
                f"Portfolio analysis needs 2+ zones with overlapping dates "
                f"(currently {rev_df.shape[1]} zone with {len(rev_df)} aligned days)."
            )
            return

        n_aligned = len(rev_df)
        st.caption(
            f"Aligned sample: {n_aligned} days across "
            f"{rev_df.shape[1]} zones ({', '.join(rev_df.columns)}). "
            "Per-zone numbers below are computed on this intersection — "
            "they will differ from the Zone Comparison table above, which "
            "uses each zone's full available history."
        )
        if n_aligned < 90:
            st.warning(
                f"Only {n_aligned} aligned days — frontier and Sharpe values "
                "are statistically noisy. Treat as directional, not "
                "investment-grade. 180+ days recommended for meaningful "
                "correlation structure."
            )
        st.caption(
            "Annual std uses i.i.d. √N scaling, which understates true risk "
            "because EU DA revenue has weekly seasonality, DST 23/25-h days, "
            "and weather-driven autocorrelation. Use Sharpe as a *relative* "
            "ranking only."
        )

        # ── Correlation heatmap ─────────────────────────────────────────
        corr = compute_correlation_matrix(rev_df)
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0.0,
            zmin=-1, zmax=1,
            aspect="auto",
            title="Daily Revenue Correlation",
            template=chart_template,
        )
        st.plotly_chart(fig_corr, width="stretch")

        # ── Per-zone stats ──────────────────────────────────────────────
        stats = compute_zone_stats(rev_df)
        st.dataframe(
            stats,
            width="stretch",
            column_config={
                "mean_daily": st.column_config.NumberColumn(
                    "Mean (EUR/MW/day)", format="€%.0f",
                ),
                "std_daily": st.column_config.NumberColumn(
                    "Std (EUR/MW/day)", format="€%.0f",
                ),
                "sharpe_daily": st.column_config.NumberColumn(
                    "Sharpe (daily)", format="%.2f",
                ),
                "mean_annual": st.column_config.NumberColumn(
                    "Annualised Mean", format="€%,.0f",
                ),
                "std_annual": st.column_config.NumberColumn(
                    "Annualised Std", format="€%,.0f",
                ),
            },
        )

        # ── Efficient frontier ──────────────────────────────────────────
        frontier = compute_efficient_frontier(rev_df, n_points=30)
        min_var = compute_min_variance_portfolio(rev_df)
        max_sharpe = compute_max_sharpe_portfolio(rev_df)

        single_zone_pts = pd.DataFrame({
            "annual_risk": stats["std_annual"],
            "annual_return": stats["mean_annual"],
            "zone": stats.index,
        })

        fig_ef = go.Figure()
        if not frontier.empty:
            fig_ef.add_trace(go.Scatter(
                x=frontier["annual_risk"],
                y=frontier["annual_return"],
                mode="lines+markers",
                name="Efficient frontier",
                line=dict(color="#3498DB", width=2),
                marker=dict(size=4),
            ))
        fig_ef.add_trace(go.Scatter(
            x=single_zone_pts["annual_risk"],
            y=single_zone_pts["annual_return"],
            mode="markers+text",
            text=single_zone_pts["zone"],
            textposition="top center",
            name="Single zone",
            marker=dict(size=12, color="#888"),
        ))
        fig_ef.add_trace(go.Scatter(
            x=[min_var["annual_risk"]], y=[min_var["annual_return"]],
            mode="markers", name="Min variance",
            marker=dict(size=14, color="#27AE60", symbol="diamond"),
        ))
        fig_ef.add_trace(go.Scatter(
            x=[max_sharpe["annual_risk"]], y=[max_sharpe["annual_return"]],
            mode="markers", name="Max Sharpe",
            marker=dict(size=14, color="#E74C3C", symbol="star"),
        ))
        fig_ef.update_layout(
            title="Long-Only Markowitz Frontier",
            xaxis_title="Annualised Std (EUR/MW)",
            yaxis_title="Annualised Mean Revenue (EUR/MW)",
            template=chart_template,
        )
        st.plotly_chart(fig_ef, width="stretch")

        # ── Optimal weights table ───────────────────────────────────────
        weights_df = pd.DataFrame({
            "Min variance": min_var["weights"],
            "Max Sharpe": max_sharpe["weights"],
        })
        st.markdown("**Optimal long-only weights**")
        st.dataframe(
            weights_df.style.format("{:.1%}"),
            width="stretch",
        )
        st.caption(
            f"Min-variance portfolio: €{min_var['annual_return']:,.0f}/MW/yr at "
            f"€{min_var['annual_risk']:,.0f} std. "
            f"Max-Sharpe portfolio: €{max_sharpe['annual_return']:,.0f}/MW/yr at "
            f"€{max_sharpe['annual_risk']:,.0f} std (Sharpe ≈ {max_sharpe['sharpe']:.2f})."
        )
