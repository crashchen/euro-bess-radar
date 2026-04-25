"""Tab 3: Revenue Estimation — DA arbitrage, ancillary stack, degradation, sensitivity."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    calculate_daily_spreads,
    calculate_imbalance_spread,
    calculate_monthly_revenue,
    calculate_yearly_revenue_breakdown,
    estimate_annual_arbitrage_revenue,
)
from src.ancillary import (
    build_ancillary_dataset,
    calculate_ancillary_revenue,
    co_optimize_revenue_split,
    merge_revenue_stack,
)
from src.config import (
    ANCILLARY_CAPACITY_AVAILABILITY,
    ANCILLARY_ENERGY_ACTIVATION_SHARE,
)
from src.degradation import (
    calculate_annual_throughput_mwh,
    calculate_degradation_cost,
    calculate_levelized_cost_of_storage,
    calculate_net_revenue,
    estimate_battery_lifetime,
)
from src.scenario import (
    bootstrap_annual_revenue,
    calculate_npv_distribution,
    sensitivity_table,
)


def render(
    primary_zone: str,
    primary_df: pd.DataFrame,
    daily_spreads: pd.DataFrame,
    monthly_spreads: pd.DataFrame,
    percentiles: dict[str, float],
    revenue: dict,
    start_date,
    end_date,
    power_mw: float,
    duration_hours: int,
    efficiency: float,
    capture_rate: float,
    capex_eur_kwh: float,
    use_lp_dispatch: bool,
    zone_tz: str,
    chart_template: str,
    report_figures: dict[str, object],
    export_revenue: dict,
    auto_fetch_results: dict | None = None,
) -> dict:
    """Render the Revenue Estimation tab. Returns updated export_revenue."""
    st.subheader(f"Revenue Estimation — {primary_zone}")
    st.caption(
        f"Applied BESS case: {power_mw:.1f} MW / {duration_hours}h, "
        f"{efficiency:.0%} efficiency, {capture_rate:.0%} capture, "
        f"{'LP dispatch' if use_lp_dispatch else 'greedy dispatch'}."
    )
    sample_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    if sample_days < 365:
        st.warning(
            f"Annualisation note: this revenue is extrapolated from the selected "
            f"sample window ({sample_days} days, {start_date} to {end_date}). "
            "For market screening, prefer at least 12 months of data to reduce "
            "seasonality bias."
        )
    else:
        st.caption(
            f"Annualisation note: this revenue is extrapolated from the selected "
            f"sample window ({sample_days} days)."
        )

    with st.expander("How ancillary works"):
        st.markdown(
            f"""
            This ancillary layer is a screening model that sits on top of day-ahead arbitrage.

            - You can load ancillary data by manual CSV upload or zone-specific auto-fetch.
            - If both are present, manual uploads override auto-fetched rows for the same product name only.
            - Reserve products are kept separate where possible, for example `FCR-N`, `FCR-D Up`, `FCR-D Down`, `aFRR Up`, and `aFRR Down`.
            - Capacity-style products are annualised from average `EUR/MW` prices using the selected BESS power and a fixed `{ANCILLARY_CAPACITY_AVAILABILITY:.0%}` availability assumption.
            - Explicit single-sided energy prices are annualised using the selected BESS power and a simplified `{ANCILLARY_ENERGY_ACTIVATION_SHARE:.0%}` activation-hours assumption.
            - Two-sided balancing or system-price signals, such as GB `system buy` and `system sell`, are stored and shown but are not auto-monetised because dispatch direction and activation volume are still unknown.
            - The revenue stack chart shows `DA Arbitrage` plus each ancillary product as separate colored components.
            - This output is intended for market screening and prioritisation, not as a dispatch-grade settlement model.
            """
        )

    # Check for ancillary data
    stored_ancillary_zone = st.session_state.get("ancillary_zone")
    stored_ancillary_dates = st.session_state.get("ancillary_dates")
    current_ancillary_dates = (str(start_date), str(end_date))
    ancillary_scope_matches = (
        stored_ancillary_zone == primary_zone
        and stored_ancillary_dates == current_ancillary_dates
    )
    ancillary_scope_mismatch = (
        (st.session_state.get("ancillary_df") is not None)
        or bool(st.session_state.get("auto_fetch_results"))
    ) and not ancillary_scope_matches

    manual_anc_df = st.session_state.get("ancillary_df") if ancillary_scope_matches else None
    auto_fetch_results = (
        st.session_state.get("auto_fetch_results", {})
        if ancillary_scope_matches
        else {}
    )
    anc_df = build_ancillary_dataset(manual_anc_df, auto_fetch_results)
    anc_rev = None
    stack = None
    anc_source = None

    if ancillary_scope_mismatch:
        st.info(
            "Ancillary data was loaded for a different zone/window. Re-fetch or "
            "re-upload to include ancillary revenue."
        )

    if anc_df is not None and not anc_df.empty:
        anc_rev = calculate_ancillary_revenue(anc_df, power_mw, duration_hours)
        stack = merge_revenue_stack(revenue, anc_rev, power_mw=power_mw)
        export_revenue.update(stack)
        export_revenue["annual_revenue_eur"] = stack["total_eur"]
        export_revenue["annual_revenue_eur_per_mw"] = stack["total_per_mw"]
        export_revenue["da_only_annual_revenue_eur"] = revenue["annual_revenue_eur"]
        export_revenue["da_only_annual_revenue_eur_per_mw"] = revenue[
            "annual_revenue_eur_per_mw"
        ]
        if manual_anc_df is not None and not manual_anc_df.empty:
            anc_source = "manual upload"
        elif auto_fetch_results:
            anc_source = f"auto-fetch ({len(auto_fetch_results)} dataset(s))"

    r1, r2, r3 = st.columns(3)
    if stack:
        r1.metric(
            "Headline Annual Revenue",
            f"\u20ac{stack['total_eur']:,.0f}",
            help=stack.get("headline_total_mode", "combined screening total"),
        )
        r2.metric("DA Arbitrage", f"\u20ac{stack['da_arbitrage_eur']:,.0f}",
                   delta=f"{stack['da_pct']:.0f}% of gross reference")
        r3.metric(
            "Ancillary Standalone",
            f"\u20ac{stack['standalone_ancillary_eur']:,.0f}",
            delta=f"{stack['ancillary_pct']:.0f}% of gross reference",
        )

        if stack.get("capacity_stack_warning"):
            st.warning(stack["capacity_stack_warning"])
            st.caption(
                f"Gross additive reference, not co-optimized: "
                f"\u20ac{stack['gross_additive_total_eur']:,.0f}/yr."
            )
            # Co-optimization estimate
            cap_eur = stack.get("capacity_ancillary_eur", 0.0)
            if cap_eur > 0:
                hours_per_year = 8766.0
                avg_cap_price = cap_eur / (power_mw * hours_per_year * 0.95)
                co_opt = co_optimize_revenue_split(
                    da_annual_revenue=stack["da_arbitrage_eur"],
                    capacity_price_eur_mw_h=avg_cap_price,
                    power_mw=power_mw,
                )
                with st.expander("Co-optimization estimate", expanded=False):
                    co1, co2, co3 = st.columns(3)
                    co1.metric(
                        "Optimal DA/Capacity Split",
                        f"{(1 - co_opt['optimal_fraction']):.0%} DA / "
                        f"{co_opt['optimal_fraction']:.0%} Capacity",
                    )
                    co2.metric(
                        "Co-optimized Total",
                        f"\u20ac{co_opt['total_revenue']:,.0f}/yr",
                    )
                    uplift = co_opt["total_revenue"] - stack["da_arbitrage_eur"]
                    co3.metric(
                        "Uplift vs DA-only",
                        f"\u20ac{uplift:,.0f}/yr",
                        delta=f"+{uplift / stack['da_arbitrage_eur'] * 100:.0f}%"
                        if stack["da_arbitrage_eur"] > 0 else "",
                    )
                    st.caption(
                        "Heuristic time-partition: committed hours earn capacity price, "
                        "uncommitted hours earn DA arbitrage pro-rata. Not a joint LP."
                    )

        component_rows = [
            {
                "Source": source,
                "Standalone Annual Revenue (EUR)": value,
                "Revenue Type": stack.get("product_revenue_types", {}).get(source, "energy")
                if source != "DA Arbitrage" else "DA",
            }
            for source, value in stack["source_revenues"].items()
            if value > 0
        ]
        if component_rows:
            st.table(pd.DataFrame(component_rows))

        fig_stack = go.Figure()
        palette = px.colors.qualitative.Bold + px.colors.qualitative.Safe
        for idx, row in enumerate(component_rows):
            fig_stack.add_trace(go.Bar(
                name=row["Source"],
                y=["Annual Revenue"],
                x=[row["Standalone Annual Revenue (EUR)"]],
                orientation="h",
                marker_color=palette[idx % len(palette)],
            ))
        fig_stack.update_layout(
            barmode="stack",
            title="Revenue Stack by Product",
            template=chart_template,
            xaxis_title="EUR",
            yaxis_title="",
            legend_title_text="Source",
        )
        report_figures["revenue_bar"] = fig_stack
        st.plotly_chart(fig_stack, width="stretch")
        if anc_source:
            st.caption(f"Ancillary valuation source: {anc_source}")
        if auto_fetch_results and anc_rev["total_ancillary_eur"] == 0:
            st.info(
                "Auto-fetched balancing/system-price datasets are loaded, but the current "
                "model only monetises explicit capacity prices and single-sided energy prices."
            )

        # Imbalance spread opportunity (P4-D)
        if auto_fetch_results:
            imb_data = auto_fetch_results.get("Imbalance prices")
            if imb_data is not None and not imb_data.empty:
                imb_spread = calculate_imbalance_spread(
                    primary_df, imb_data, tz=zone_tz,
                )
                if imb_spread["avg_spread"] > 0:
                    with st.expander("Imbalance Spread Opportunity", expanded=False):
                        im1, im2, im3 = st.columns(3)
                        im1.metric("Avg DA-Imbalance Spread",
                                   f"\u20ac{imb_spread['avg_spread']:.1f}/MWh")
                        im2.metric("P90 Spread",
                                   f"\u20ac{imb_spread['p90']:.1f}/MWh")
                        im3.metric(
                            "Est. Annual Value/MW",
                            f"\u20ac{imb_spread['estimated_annual_value_per_mw']:,.0f}",
                            help="Theoretical maximum — actual capture depends on forecast accuracy and position limits.",
                        )
                        st.caption(
                            "Supplementary revenue opportunity from DA-vs-imbalance price spread. "
                            "Not added to headline total without co-optimization."
                        )
    else:
        dispatch_label = revenue.get("dispatch_method", "greedy")
        r1.metric(
            "Est. Annual Revenue",
            f"\u20ac{revenue['annual_revenue_eur']:,.0f}",
            help=(
                f"At {power_mw} MW, {duration_hours}h, {efficiency*100:.0f}% eff, "
                f"{revenue['cycles_per_day_assumption']:.1f} avg cycle/day, "
                f"{revenue['capture_rate_assumption']:.0%} capture, "
                f"dispatch: {dispatch_label}"
            ),
        )
        r2.metric("Revenue per MW", f"\u20ac{revenue['annual_revenue_eur_per_mw']:,.0f}/MW/yr")
        r3.metric("Avg Daily Revenue", f"\u20ac{revenue['avg_daily_revenue']:,.0f}")
        st.info("Upload or auto-fetch ancillary services data to see the full revenue stack.")

    # CapEx / Payback
    if capex_eur_kwh > 0:
        total_capex = capex_eur_kwh * power_mw * duration_hours * 1000
        annual_rev = (
            stack["total_eur"] if stack
            else revenue["annual_revenue_eur"]
        )
        payback_years = total_capex / annual_rev if annual_rev > 0 else float("inf")
        st.divider()
        p1, p2, p3 = st.columns(3)
        p1.metric("Total CapEx", f"\u20ac{total_capex:,.0f}")
        p2.metric("Annual Revenue", f"\u20ac{annual_rev:,.0f}")
        p3.metric(
            "Simple Payback",
            f"{payback_years:.1f} years" if payback_years < 100 else "N/A",
        )

        # Battery degradation & lifetime
        if "n_cycles" in daily_spreads.columns:
            avg_cycles_day = float(daily_spreads["n_cycles"].mean())
        else:
            avg_cycles_day = float(revenue.get("cycles_per_day_assumption", 1.0))
        annual_cycles = avg_cycles_day * 365.25
        capacity_kwh = power_mw * duration_hours * 1000
        deg_cost = calculate_degradation_cost(
            n_cycles=annual_cycles,
            capex_eur_kwh=capex_eur_kwh,
            capacity_kwh=capacity_kwh,
        )
        lifetime = estimate_battery_lifetime(avg_cycles_per_day=avg_cycles_day)
        net_rev = calculate_net_revenue(
            annual_rev,
            deg_cost["total_degradation_eur"],
        )
        annual_throughput_mwh = calculate_annual_throughput_mwh(
            avg_cycles_day,
            capacity_kwh,
        )
        lcos_eur_mwh = (
            calculate_levelized_cost_of_storage(
                capex_eur_kwh,
                capacity_kwh,
                float(lifetime["effective_life_years"]),
                annual_throughput_mwh,
            )
            if annual_throughput_mwh > 0 else None
        )
        net_payback = (
            total_capex / net_rev["net_revenue_eur"]
            if net_rev["net_revenue_eur"] > 0 else float("inf")
        )

        export_revenue.update({
            "annual_degradation_cost_eur": deg_cost["total_degradation_eur"],
            "degradation_cost_per_cycle_eur": deg_cost["cost_per_cycle_eur"],
            "degradation_cycle_life": deg_cost["cycle_life"],
            "net_revenue_eur": net_rev["net_revenue_eur"],
            "degradation_pct": net_rev["degradation_pct"],
            "effective_life_years": lifetime["effective_life_years"],
            "cycle_limited_years": lifetime["cycle_limited_years"],
            "calendar_life_years": lifetime["calendar_life_years"],
            "lifetime_limiting_factor": lifetime["limiting_factor"],
            "annual_throughput_mwh": annual_throughput_mwh,
            "net_payback_years": net_payback,
        })
        if lcos_eur_mwh is not None:
            export_revenue["lcos_eur_mwh"] = lcos_eur_mwh

        st.divider()
        st.markdown("**Battery Degradation & Lifetime**")
        d1, d2, d3 = st.columns(3)
        d1.metric(
            "Degradation Cost/Year",
            f"\u20ac{deg_cost['total_degradation_eur']:,.0f}",
        )
        d2.metric(
            "Net Revenue/Year",
            f"\u20ac{net_rev['net_revenue_eur']:,.0f}",
            delta=f"-{net_rev['degradation_pct']:.1f}% degradation",
        )
        d3.metric(
            "Effective Lifetime",
            f"{float(lifetime['effective_life_years']):.1f} years",
            help=(
                f"Limited by {lifetime['limiting_factor']} "
                f"({float(lifetime['cycle_limited_years']):.1f}y cycling, "
                f"{float(lifetime['calendar_life_years']):.0f}y calendar)"
            ),
        )

        d4, d5, d6, d7 = st.columns(4)
        d4.metric("Cost per Cycle", f"\u20ac{deg_cost['cost_per_cycle_eur']:,.0f}")
        d5.metric(
            "Net Payback",
            f"{net_payback:.1f} years" if net_payback < 100 else "N/A",
        )
        d6.metric(
            "LCOS",
            f"\u20ac{lcos_eur_mwh:,.1f}/MWh"
            if lcos_eur_mwh is not None else "N/A",
        )
        d7.metric("Avg Cycles/Day", f"{avg_cycles_day:.2f}")
        st.caption(
            f"Degradation uses {avg_cycles_day:.2f} modeled DA full-equivalent "
            "cycles/day. Ancillary activation wear is not modeled in this "
            "screening estimate."
        )

    # Revenue waterfall
    st.divider()
    theoretical_spread = percentiles["mean"]
    eff_loss = theoretical_spread * (1 - efficiency)
    post_eff = theoretical_spread - eff_loss
    capture_loss = post_eff * (1 - capture_rate)
    realized_spread = post_eff - capture_loss
    wf_measures = ["relative", "relative", "total", "relative", "total"]
    wf_labels = [
        "Avg Ordered Spread",
        "Efficiency Loss",
        "Post-Efficiency",
        "Capture Discount",
        "Realized Spread",
    ]
    wf_values = [
        theoretical_spread,
        -eff_loss,
        post_eff,
        -capture_loss,
        realized_spread,
    ]
    fig_wf = go.Figure(go.Waterfall(
        x=wf_labels,
        y=wf_values,
        measure=wf_measures,
        connector={"line": {"color": "rgba(150,150,150,0.4)"}},
        decreasing={"marker": {"color": "#E74C3C"}},
        increasing={"marker": {"color": "#2ECC71"}},
        totals={"marker": {"color": "#2E86C1"}},
        textposition="outside",
        text=[f"\u20ac{v:+.1f}" if m == "relative" else f"\u20ac{v:.1f}"
              for v, m in zip(wf_values, wf_measures, strict=True)],
    ))
    fig_wf.update_layout(
        title="Revenue Attribution Waterfall (EUR/MWh per cycle)",
        template=chart_template,
        yaxis_title="EUR/MWh",
        showlegend=False,
    )
    report_figures["revenue_waterfall"] = fig_wf
    st.plotly_chart(fig_wf, width="stretch")

    # LP dispatch details
    if use_lp_dispatch and "lp_revenue" in daily_spreads.columns:
        st.divider()
        st.markdown("**LP Dispatch Details**")
        lp1, lp2, lp3 = st.columns(3)
        avg_cycles = float(daily_spreads["n_cycles"].mean())
        avg_lp_spread = float(daily_spreads["lp_spread_eur_mwh"].mean())
        greedy_spread = float(daily_spreads["spread"].mean())
        uplift_pct = (
            (avg_lp_spread - greedy_spread * efficiency) / (greedy_spread * efficiency) * 100
            if greedy_spread * efficiency > 0 else 0.0
        )
        lp1.metric("Avg Cycles/Day", f"{avg_cycles:.2f}")
        lp2.metric("LP Spread (EUR/MWh)", f"\u20ac{avg_lp_spread:.1f}")
        lp3.metric(
            "LP vs Greedy Uplift",
            f"{uplift_pct:+.0f}%",
            help="Percentage improvement of LP-optimal over greedy single-cycle (adjusted for efficiency)",
        )

    # Sensitivity table (cached)
    _render_sensitivity_table(primary_df, zone_tz, efficiency, capture_rate, chart_template)

    # Spread distribution
    fig_hist = px.histogram(
        daily_spreads, x="spread", nbins=30,
        title="Ordered Spread Distribution",
        labels={"spread": "Daily Ordered Spread (EUR/MWh)", "count": "Days"},
        template=chart_template,
    )
    fig_hist.add_vline(x=percentiles["p50"], line_dash="dash",
                       annotation_text=f"P50: {percentiles['p50']:.1f}")
    fig_hist.add_vline(x=percentiles["p90"], line_dash="dash", line_color="red",
                       annotation_text=f"P90: {percentiles['p90']:.1f}")
    st.plotly_chart(fig_hist, width="stretch")

    # Monthly revenue seasonality & volatility
    if not monthly_spreads.empty and len(monthly_spreads) >= 2:
        _render_monthly_seasonality(
            daily_spreads, monthly_spreads, duration_hours, chart_template, report_figures,
        )

    # ── Year-over-year backtest (P4-A) ─────────────────────────────────
    yearly = calculate_yearly_revenue_breakdown(
        daily_spreads,
        power_mw=power_mw,
        duration_hours=duration_hours,
        roundtrip_efficiency=efficiency,
        capture_rate=capture_rate,
    )
    if len(yearly) >= 2:
        st.divider()
        st.markdown("**Year-over-Year Revenue Backtest**")
        rev_cov = (
            float(yearly["annual_revenue"].std() / yearly["annual_revenue"].mean())
            if yearly["annual_revenue"].mean() > 0 else 0.0
        )
        y1, y2 = st.columns(2)
        y1.metric("Revenue CoV", f"{rev_cov:.2f}",
                   help="Coefficient of variation across years — lower = more stable")
        y2.metric("Years in Sample", str(len(yearly)))
        fig_yoy = px.bar(
            yearly, x="year", y="revenue_per_mw",
            title="Annual Revenue per MW by Year",
            labels={"year": "Year", "revenue_per_mw": "EUR/MW/yr"},
            template=chart_template,
            text_auto=True,
        )
        fig_yoy.update_traces(texttemplate="\u20ac%{y:,.0f}", textposition="outside")
        st.plotly_chart(fig_yoy, width="stretch")

        # Monthly revenue heatmap (year x month)
        monthly_rev = calculate_monthly_revenue(
            daily_spreads,
            power_mw=power_mw,
            duration_hours=duration_hours,
            roundtrip_efficiency=efficiency,
            capture_rate=capture_rate,
        )
        if len(monthly_rev) > 3:
            pivot = monthly_rev.pivot_table(
                index="year", columns="month", values="monthly_revenue", aggfunc="sum",
            )
            pivot.columns = [f"{m:02d}" for m in pivot.columns]
            fig_seasonal = px.imshow(
                pivot.values,
                x=list(pivot.columns),
                y=[str(y) for y in pivot.index],
                labels={"x": "Month", "y": "Year", "color": "EUR"},
                title="Monthly Revenue Heatmap (EUR)",
                color_continuous_scale="YlOrRd",
                template=chart_template,
                aspect="auto",
            )
            st.plotly_chart(fig_seasonal, width="stretch")

    # ── Risk analysis — Monte Carlo (P4-B) ─────────────────────────────
    with st.expander("Risk Analysis (Monte Carlo)", expanded=False):
        if use_lp_dispatch and "lp_revenue" in daily_spreads.columns:
            daily_rev_series = daily_spreads["lp_revenue"] * capture_rate
        else:
            energy_mwh = power_mw * duration_hours
            daily_rev_series = (
                daily_spreads["spread"] * energy_mwh * efficiency * capture_rate
            )

        mc = bootstrap_annual_revenue(daily_rev_series, n_simulations=5000)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("P10 Revenue", f"\u20ac{mc['p10']:,.0f}")
        mc2.metric("P50 Revenue", f"\u20ac{mc['p50']:,.0f}")
        mc3.metric("P90 Revenue", f"\u20ac{mc['p90']:,.0f}")

        fig_mc = px.histogram(
            x=mc["simulations"], nbins=50,
            title="Bootstrapped Annual Revenue Distribution",
            labels={"x": "Annual Revenue (EUR)", "count": "Simulations"},
            template=chart_template,
        )
        fig_mc.add_vline(x=mc["p10"], line_dash="dash",
                         annotation_text=f"P10: \u20ac{mc['p10']:,.0f}")
        fig_mc.add_vline(x=mc["p50"], line_dash="solid", line_color="green",
                         annotation_text=f"P50: \u20ac{mc['p50']:,.0f}")
        fig_mc.add_vline(x=mc["p90"], line_dash="dash", line_color="red",
                         annotation_text=f"P90: \u20ac{mc['p90']:,.0f}")
        st.plotly_chart(fig_mc, width="stretch")

        # NPV distribution (if CapEx provided)
        if capex_eur_kwh > 0:
            capacity_kwh = power_mw * duration_hours * 1000
            total_capex_mc = capex_eur_kwh * capacity_kwh
            if "n_cycles" in daily_spreads.columns:
                mc_avg_cycles = float(daily_spreads["n_cycles"].mean())
            else:
                mc_avg_cycles = float(revenue.get("cycles_per_day_assumption", 1.0))
            annual_cycles = mc_avg_cycles * 365.25
            mc_deg = calculate_degradation_cost(
                n_cycles=annual_cycles,
                capex_eur_kwh=capex_eur_kwh,
                capacity_kwh=capacity_kwh,
            )
            mc_lifetime = estimate_battery_lifetime(avg_cycles_per_day=mc_avg_cycles)

            npv_dist = calculate_npv_distribution(
                mc["simulations"],
                total_capex=total_capex_mc,
                annual_degradation_cost=mc_deg["total_degradation_eur"],
                effective_life_years=float(mc_lifetime["effective_life_years"]),
            )

            n1, n2, n3, n4 = st.columns(4)
            n1.metric("NPV P10", f"\u20ac{npv_dist['npv_p10']:,.0f}")
            n2.metric("NPV P50", f"\u20ac{npv_dist['npv_p50']:,.0f}")
            n3.metric("NPV P90", f"\u20ac{npv_dist['npv_p90']:,.0f}")
            prob_color = "normal" if npv_dist["prob_positive_npv"] >= 0.5 else "inverse"
            n4.metric("P(NPV>0)", f"{npv_dist['prob_positive_npv']:.0%}",
                      delta_color=prob_color)

            fig_npv = px.histogram(
                x=npv_dist["npv_array"], nbins=50,
                title="NPV Distribution",
                labels={"x": "NPV (EUR)", "count": "Simulations"},
                template=chart_template,
            )
            fig_npv.add_vline(x=0, line_dash="solid", line_color="gray",
                              annotation_text="Break-even")
            st.plotly_chart(fig_npv, width="stretch")

            # Sensitivity tornado
            sens = sensitivity_table(
                base_revenue=mc["p50"],
                total_capex=total_capex_mc,
                effective_life_years=float(mc_lifetime["effective_life_years"]),
                annual_degradation_cost=mc_deg["total_degradation_eur"],
            )
            base_npv = sens.loc[
                (sens["param"] == "revenue") & (sens["value"] == 1.0), "npv"
            ].iloc[0]
            tornado_rows = []
            for param in sens["param"].unique():
                param_data = sens[sens["param"] == param].sort_values("value")
                low_npv = param_data["npv"].iloc[0]
                high_npv = param_data["npv"].iloc[-1]
                tornado_rows.append({
                    "param": param,
                    "low_delta": low_npv - base_npv,
                    "high_delta": high_npv - base_npv,
                })
            tornado_df = pd.DataFrame(tornado_rows)
            tornado_df["swing"] = tornado_df["high_delta"] - tornado_df["low_delta"]
            tornado_df = tornado_df.sort_values("swing", ascending=True)

            fig_tornado = go.Figure()
            fig_tornado.add_trace(go.Bar(
                y=tornado_df["param"],
                x=tornado_df["low_delta"],
                orientation="h",
                name="Downside",
                marker_color="#E74C3C",
            ))
            fig_tornado.add_trace(go.Bar(
                y=tornado_df["param"],
                x=tornado_df["high_delta"],
                orientation="h",
                name="Upside",
                marker_color="#2ECC71",
            ))
            fig_tornado.update_layout(
                title="NPV Sensitivity (vs base case)",
                template=chart_template,
                barmode="overlay",
                xaxis_title="NPV Delta (EUR)",
            )
            st.plotly_chart(fig_tornado, width="stretch")

    return export_revenue


def _render_sensitivity_table(
    primary_df: pd.DataFrame,
    zone_tz: str,
    efficiency: float,
    capture_rate: float,
    chart_template: str,
) -> None:
    """Render the duration sensitivity table."""
    st.markdown("**Duration Sensitivity (per MW)**")
    sens_rows = []
    for dur in [1, 2, 4]:
        spreads_d = calculate_daily_spreads(
            primary_df, tz=zone_tz, duration_hours=dur,
        )
        rev_d = estimate_annual_arbitrage_revenue(
            spreads_d, power_mw=1.0, duration_hours=dur,
            roundtrip_efficiency=efficiency,
            capture_rate=capture_rate,
        )
        sens_rows.append({
            "Duration (h)": dur,
            "Annual Revenue (EUR/MW)": f"\u20ac{rev_d['annual_revenue_eur_per_mw']:,.0f}",
            "Avg Daily (EUR/MW)": f"\u20ac{rev_d['avg_daily_revenue']:,.0f}",
        })
    st.table(pd.DataFrame(sens_rows))


def _render_monthly_seasonality(
    daily_spreads: pd.DataFrame,
    monthly_spreads: pd.DataFrame,
    duration_hours: int,
    chart_template: str,
    report_figures: dict[str, object],
) -> None:
    """Render the monthly revenue seasonality section."""
    st.divider()
    st.markdown("**Monthly Revenue Seasonality & Risk**")

    monthly_rev = monthly_spreads["avg_spread"].values
    spread_cv = float(monthly_rev.std() / monthly_rev.mean()) if monthly_rev.mean() > 0 else 0
    best_month = monthly_spreads.loc[monthly_spreads["avg_spread"].idxmax()]
    worst_month = monthly_spreads.loc[monthly_spreads["avg_spread"].idxmin()]
    zero_spread_days = int((daily_spreads["spread"] <= 0).sum())

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Spread CV", f"{spread_cv:.2f}",
               help="Coefficient of variation — lower = more stable revenue")
    v2.metric("Best Month", f"{best_month['year_month']}",
               delta=f"\u20ac{best_month['avg_spread']:.1f}/MWh")
    v3.metric("Worst Month", f"{worst_month['year_month']}",
               delta=f"\u20ac{worst_month['avg_spread']:.1f}/MWh",
               delta_color="inverse")
    v4.metric("Zero-Spread Days", f"{zero_spread_days}",
               delta=f"{zero_spread_days / len(daily_spreads) * 100:.0f}% of total",
               delta_color="inverse")

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=monthly_spreads["year_month"],
        y=monthly_spreads["avg_spread"],
        name="Avg Spread",
        marker_color="#2E86C1",
    ))
    fig_monthly.add_trace(go.Scatter(
        x=monthly_spreads["year_month"],
        y=monthly_spreads["max_spread"],
        mode="markers+lines",
        name="Max Spread",
        marker=dict(color="#2ECC71", size=6),
        line=dict(dash="dot"),
    ))
    fig_monthly.add_trace(go.Scatter(
        x=monthly_spreads["year_month"],
        y=monthly_spreads["min_spread"],
        mode="markers+lines",
        name="Min Spread",
        marker=dict(color="#E74C3C", size=6),
        line=dict(dash="dot"),
    ))
    fig_monthly.update_layout(
        title=f"Monthly Spread Breakdown ({duration_hours}h windows)",
        template=chart_template,
        xaxis_title="Month",
        yaxis_title="EUR/MWh",
        legend=dict(orientation="h", y=-0.15),
    )
    report_figures["monthly_seasonality"] = fig_monthly
    st.plotly_chart(fig_monthly, width="stretch")
