"""Tab 7: Forward Scenarios — BESS revenue at user-supplied forward curve prices.

Upload-first workflow: the user pastes / uploads a forward CSV (zone, period,
baseload EUR/MWh) from EEX EOD, broker quotes, internal price assumptions, or
Bloomberg/Refinitiv exports. We synthesise hourly forward prices by overlaying
each contract's baseload onto the historical hour-of-week shape for the same
zone, then run the existing daily-spread / MILP-dispatch / annualisation
pipeline. Output is per-contract period revenue plus an annualised EUR/MW/yr
view, with explicit warnings on overlap and shape-source coverage.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import (
    calculate_daily_dispatch,
    calculate_daily_spreads,
)
from src.config import get_zone_timezone
from src.forward_curve import (
    build_forward_synthetic_prices,
    find_overlapping_contracts,
    generate_forward_template_csv,
    list_supported_zones,
    parse_forward_csv,
    summarise_forward_revenue,
)


def render(
    *,
    zone_data: dict[str, pd.DataFrame],
    power_mw: float,
    duration_hours: int,
    efficiency: float,
    capture_rate: float,
    chart_template: str,
) -> None:
    """Render the Forward Scenarios tab."""
    st.subheader("Forward Scenarios — BESS Revenue at Forward Curve Prices")
    st.caption(
        "Upload a forward-curve CSV (zone * delivery period * baseload "
        "EUR/MWh). Each contract's baseload is multiplied by the historical "
        "hour-of-week shape for that zone to synthesise hourly forward "
        "prices, then the standard BESS dispatch + annualisation pipeline "
        "runs on the synthetic series. The result is a *what-would-this-"
        "battery-earn-at-these-prices* screening, not a market forecast."
    )

    with st.expander("How forward scenarios work", expanded=False):
        st.markdown(
            f"""
            **Inputs**: a CSV with `zone`, `delivery_start`, `delivery_end`,
            `price_eur_mwh` columns (plus optional `contract`, `shape`,
            `source`, `as_of`). Sources you might pull from:

            - **EEX EOD** (subscription) — official German / French / NL
              Power futures end-of-day, CSV per delivery period.
            - **Broker / Bloomberg / Refinitiv / Montel** — pull your
              firm's forward curve and reformat to the schema below.
            - **Internal price assumptions** — your IPP's central case,
              high / low sensitivities.

            **Method**: for each contract,
            `hourly_price[t] = forward_base * historical_shape[hour_of_week(t)]`,
            where the historical shape comes from the cached DA series for
            the same zone (the one shown in Market Overview). This
            preserves the intra-day spread pattern that drives BESS
            arbitrage while letting the level move with the forward.

            **Caveats** to read every output through:
            - Real future shape may differ from history (RE penetration
              shifts the spread curve over time).
            - Capture rate is applied identically to forward and historical
              analyses; revisit this assumption when the forward implies a
              very different volatility regime.
            - Currently shows BESS case at {power_mw:.1f} MW / {duration_hours}h /
              {efficiency:.0%} eff / {capture_rate:.0%} capture (set in
              the sidebar).
            """
        )

    c1, c2 = st.columns([3, 1])
    uploaded = c1.file_uploader(
        "Forward-curve CSV", type=["csv"], key="forward_csv_upload",
    )
    template = c2.download_button(
        "Download template",
        data=generate_forward_template_csv(),
        file_name="forward_curve_template.csv",
        mime="text/csv",
    )
    del template

    if uploaded is None:
        st.info("Upload a CSV to compute forward scenarios.")
        return

    try:
        content = uploaded.getvalue().decode("utf-8-sig")
        forward_df = parse_forward_csv(content)
    except UnicodeDecodeError:
        st.error("Parse error: the uploaded file is not valid UTF-8/CSV text.")
        return
    except ValueError as exc:
        st.error(f"Parse error: {exc}")
        return

    st.success(
        f"Parsed {len(forward_df)} forward contract(s) "
        f"across {len(set(forward_df['zone']))} zone(s)."
    )
    with st.expander("Parsed forward curve", expanded=False):
        st.dataframe(forward_df, width="stretch", hide_index=True)

    overlaps = find_overlapping_contracts(forward_df)
    if not overlaps.empty:
        st.warning(
            f"{len(overlaps)} overlapping contract pair(s) detected. "
            "When two contracts cover the same hour on the same zone, "
            "the LATER one in the CSV order wins (lets a Q1 quote "
            "override a broader Cal quote)."
        )
        st.dataframe(overlaps, width="stretch", hide_index=True)

    # Which forward zones have historical reference data?
    fwd_zones = list_supported_zones(forward_df)
    fwd_with_history = [z for z in fwd_zones if z in zone_data and not zone_data[z].empty]
    missing_history = sorted(set(fwd_zones) - set(fwd_with_history))
    if missing_history:
        st.warning(
            "No historical DA reference for: "
            + ", ".join(missing_history)
            + ". Fetch these zones in the sidebar to enable shape recovery."
        )
    if not fwd_with_history:
        st.error(
            "No zone in the forward CSV has historical DA data loaded. "
            "Fetch at least one of the listed zones first."
        )
        return

    all_summaries: list[pd.DataFrame] = []
    all_synth: dict[str, pd.DataFrame] = {}
    for zone in fwd_with_history:
        tz = get_zone_timezone(zone)
        synth = build_forward_synthetic_prices(
            forward_df, zone_data[zone], zone=zone, tz=tz,
        )
        if synth.empty:
            continue
        all_synth[zone] = synth
        # Use MILP dispatch when the sidebar opts in by data volume; for
        # screening we just run the LP path always — it's cheap and the
        # same path the rest of the dashboard uses.
        daily = calculate_daily_dispatch(
            synth[["price_eur_mwh"]],
            tz=tz, duration_hours=duration_hours,
            power_mw=power_mw, efficiency=efficiency,
        )
        if daily.empty:
            daily = calculate_daily_spreads(
                synth[["price_eur_mwh"]], tz=tz, duration_hours=duration_hours,
            )
        summary = summarise_forward_revenue(
            daily, forward_df, zone=zone,
            power_mw=power_mw, duration_hours=duration_hours,
            efficiency=efficiency, capture_rate=capture_rate,
        )
        if summary.empty:
            continue
        summary.insert(0, "zone", zone)
        all_summaries.append(summary)

    if not all_summaries:
        st.warning(
            "Could not build a forward scenario — historical zones loaded "
            "do not overlap the forward delivery windows enough to recover "
            "a usable shape."
        )
        return

    summary_all = pd.concat(all_summaries, ignore_index=True)
    st.markdown("**Per-contract forward revenue**")
    st.dataframe(
        summary_all,
        width="stretch", hide_index=True,
        column_config={
            "forward_base": st.column_config.NumberColumn(
                "Forward base", format="€%.1f/MWh",
            ),
            "avg_daily_spread": st.column_config.NumberColumn(
                "Avg daily spread", format="€%.1f/MWh",
            ),
            "period_revenue_eur": st.column_config.NumberColumn(
                "Period revenue", format="€%,.0f",
            ),
            "annualised_revenue_eur_per_mw": st.column_config.NumberColumn(
                "Annualised €/MW/yr", format="€%,.0f",
            ),
            "days_in_period": "Days",
        },
    )

    # Per-zone totals
    z_totals = (
        summary_all
        .groupby("zone")
        .agg(
            n_contracts=("contract", "size"),
            total_period_revenue_eur=("period_revenue_eur", "sum"),
            avg_annualised_eur_per_mw=("annualised_revenue_eur_per_mw", "mean"),
        )
        .reset_index()
    )
    st.markdown("**Per-zone summary**")
    st.dataframe(
        z_totals,
        width="stretch", hide_index=True,
        column_config={
            "total_period_revenue_eur": st.column_config.NumberColumn(
                "Total period revenue", format="€%,.0f",
            ),
            "avg_annualised_eur_per_mw": st.column_config.NumberColumn(
                "Avg annualised €/MW/yr", format="€%,.0f",
            ),
        },
    )

    # Synthetic price preview chart per zone
    st.markdown("**Synthetic hourly price preview**")
    zone_pick = st.selectbox(
        "Zone", options=list(all_synth.keys()),
        index=0, key="forward_preview_zone",
    )
    synth_pick = all_synth[zone_pick].reset_index()
    fig = px.line(
        synth_pick,
        x="timestamp", y="price_eur_mwh",
        color="contract",
        title=f"Synthetic forward hourly prices — {zone_pick}",
        labels={"price_eur_mwh": "EUR/MWh", "timestamp": ""},
        template=chart_template,
    )
    st.plotly_chart(fig, width="stretch")

    # Download summary
    st.download_button(
        "Download forward summary (CSV)",
        data=summary_all.to_csv(index=False),
        file_name="forward_scenario_summary.csv",
        mime="text/csv",
    )
