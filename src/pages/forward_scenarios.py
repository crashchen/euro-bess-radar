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

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    calculate_daily_dispatch,
    calculate_daily_spreads,
)
from src.config import get_zone_timezone
from src.export import cockpit_tables_to_excel
from src.forward_curve import (
    build_forward_synthetic_prices,
    find_overlapping_contracts,
    generate_forward_template_csv,
    list_supported_zones,
    parse_forward_csv,
    summarise_forward_revenue,
)
from src.trader_benchmark import (
    MODEL_REVENUE_COLUMN,
    QUOTE_REVENUE_COLUMN,
    benchmark_comparability_notes,
    build_forward_model_yearly,
    generate_trader_benchmark_template_csv,
    parse_trader_benchmark_csv,
    reconcile_trader_benchmark,
)

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_BENCHMARK_HARD_CAPTION = (
    "External benchmark reconciliation only: the uploaded annual revenue "
    "curve is user-supplied and is never used as a price curve, contracted "
    "floor, capture rate, solver input, or bankable forecast."
)


def _benchmark_assumptions(
    selected: pd.DataFrame,
    *,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    capture_rate: float,
) -> pd.DataFrame:
    """Self-contained audit rows for the benchmark workbook."""
    row = selected.iloc[0]
    source = row["source"] if pd.notna(row["source"]) else "User upload"
    as_of = (
        str(pd.Timestamp(row["as_of"]).date())
        if pd.notna(row["as_of"])
        else "Not supplied"
    )
    return pd.DataFrame(
        [
            {
                "parameter": "External benchmark source",
                "value": str(source),
                "unit": "",
                "source": "Benchmark CSV",
                "affects": "Provenance only; never changes platform calculations",
            },
            {
                "parameter": "External benchmark as-of",
                "value": as_of,
                "unit": "",
                "source": "Benchmark CSV",
                "affects": "Provenance only",
            },
            {
                "parameter": "External benchmark basis",
                "value": (
                    f"asset={row['asset_type']}; markets={row['market_scope']}; "
                    f"revenue={row['revenue_basis']}"
                ),
                "unit": "",
                "source": "Benchmark CSV",
                "affects": "Comparability warnings only",
            },
            {
                "parameter": "Platform comparison basis",
                "value": (
                    "Forward-synthetic DA-only dispatch revenue after capture "
                    "haircut; before wear, fees, tax, and financing"
                ),
                "unit": "",
                "source": "Forward Scenarios",
                "affects": "Model curve in reconciliation table",
            },
            {
                "parameter": "Platform BESS case",
                "value": (
                    f"{power_mw:g} MW / {duration_hours:g}h / "
                    f"{efficiency:.1%} efficiency"
                ),
                "unit": "",
                "source": "Sidebar",
                "affects": "Platform model curve only",
            },
            {
                "parameter": "Platform capture haircut",
                "value": f"{capture_rate:.1%}",
                "unit": "",
                "source": "Sidebar",
                "affects": "Platform model curve only; not inferred from benchmark",
            },
            {
                "parameter": "Benchmark/model ratio semantics",
                "value": (
                    "Observed quote/model statistic over overlapping years; "
                    "NOT a capture-rate estimate"
                ),
                "unit": "",
                "source": "docs/design/external-trader-benchmark-v1.md",
                "affects": "Interpretation only",
            },
        ]
    )


def _benchmark_display_frame(comparison: pd.DataFrame) -> pd.DataFrame:
    """Convert ratio fractions to percentage points for the Streamlit table."""
    display = comparison.copy()
    display["benchmark_to_model_ratio"] = (
        display["benchmark_to_model_ratio"] * 100.0
    )
    return display


def _render_external_benchmark_section(
    *,
    model_yearly: pd.DataFrame,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    capture_rate: float,
    chart_template: str,
) -> None:
    """Optional external annual revenue-curve reconciliation panel."""
    with st.expander("External trader revenue benchmark", expanded=False):
        st.caption(_BENCHMARK_HARD_CAPTION)
        st.caption(
            "Use this to explain, not hide, the gap between an indicative "
            "trader curve and the platform's forward-DA screening curve. "
            "The join is calendar year + zone; non-overlapping years remain "
            "visible but do not enter the headline comparison."
        )
        c1, c2 = st.columns([3, 1])
        uploaded = c1.file_uploader(
            "External annual revenue benchmark CSV",
            type=["csv"],
            key="external_trader_benchmark_upload",
        )
        c2.download_button(
            "Download benchmark template",
            data=generate_trader_benchmark_template_csv(),
            file_name="external_trader_benchmark_template.csv",
            mime="text/csv",
        )
        if uploaded is None:
            st.info(
                "Upload an annual EUR/MW/yr curve to compare it with the "
                "platform model. No external value is assumed by default."
            )
            return
        try:
            benchmark = parse_trader_benchmark_csv(
                uploaded.getvalue().decode("utf-8-sig")
            )
        except UnicodeDecodeError:
            st.error("Benchmark parse error: file is not valid UTF-8/CSV text.")
            return
        except ValueError as exc:
            st.error(f"Benchmark parse error: {exc}")
            return

        st.success(
            f"Parsed {len(benchmark)} benchmark year(s) across "
            f"{benchmark[['zone', 'scenario']].drop_duplicates().shape[0]} "
            "scenario(s)."
        )
        with st.expander("Parsed external benchmark", expanded=False):
            st.dataframe(benchmark, width="stretch", hide_index=True)

        b1, b2 = st.columns(2)
        zones = sorted(benchmark["zone"].unique().tolist())
        zone = b1.selectbox(
            "Benchmark zone", options=zones, key="external_benchmark_zone"
        )
        scenarios = sorted(
            benchmark.loc[benchmark["zone"] == zone, "scenario"].unique().tolist()
        )
        scenario = b2.selectbox(
            "Benchmark scenario",
            options=scenarios,
            key="external_benchmark_scenario",
        )
        comparison, summary = reconcile_trader_benchmark(
            benchmark, model_yearly, zone=zone, scenario=scenario
        )
        selected = benchmark.loc[
            (benchmark["zone"] == zone) & (benchmark["scenario"] == scenario)
        ]

        notes = benchmark_comparability_notes(
            benchmark,
            zone=zone,
            scenario=scenario,
            model_duration_hours=duration_hours,
        )
        if notes:
            st.warning("Not like-for-like:\n- " + "\n- ".join(notes))
        else:
            st.info(
                "Metadata is aligned to a standalone, DA-only, gross case. "
                "Forecast method, fees, market depth, and risk margin may "
                "still differ; matching labels do not make the quote bankable."
            )

        overlap = int(summary["n_overlap_years"])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Benchmark avg (all years)",
            f"EUR {float(summary['benchmark_average_eur_per_mw_yr']):,.0f}/MW/yr",
        )
        cagr = float(summary["benchmark_endpoint_cagr"])
        k2.metric(
            "Benchmark endpoint CAGR",
            "n/a" if math.isnan(cagr) else f"{cagr:+.1%}/yr",
        )
        if overlap:
            model_avg = float(summary["model_overlap_average_eur_per_mw_yr"])
            gap = float(summary["model_minus_benchmark_average_eur_per_mw_yr"])
            ratio = float(summary["benchmark_to_model_ratio"])
            k3.metric(
                f"Model avg ({overlap} overlap yr)",
                f"EUR {model_avg:,.0f}/MW/yr",
            )
            k4.metric(
                "Benchmark / model ratio",
                "n/a" if math.isnan(ratio) else f"{ratio:.1%}",
                delta=f"model - benchmark: EUR {gap:+,.0f}/MW/yr",
                delta_color="off",
            )
        else:
            k3.metric("Model overlap", "0 years")
            k4.metric("Benchmark / model ratio", "n/a")
            st.warning(
                "The benchmark and current forward-price scenario have no "
                "overlapping calendar years for this zone. The quote curve "
                "is shown for provenance only."
            )

        fig = go.Figure()
        fig.add_scatter(
            name="External benchmark",
            x=comparison["year"],
            y=comparison[QUOTE_REVENUE_COLUMN],
            mode="lines+markers",
            line={"color": "#FF2D95", "width": 3},
        )
        model_points = comparison.dropna(subset=[MODEL_REVENUE_COLUMN])
        if not model_points.empty:
            fig.add_scatter(
                name="Platform forward-DA model",
                x=model_points["year"],
                y=model_points[MODEL_REVENUE_COLUMN],
                mode="lines+markers",
                line={"color": "#00A3FF", "width": 3},
            )
        fig.update_layout(
            title="External benchmark vs platform model",
            xaxis_title="Calendar year",
            yaxis_title="EUR/MW/yr",
            template=chart_template,
            height=360,
            legend={"orientation": "h", "font": {"color": "#cfd8e6"}},
        )
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            _benchmark_display_frame(comparison),
            width="stretch",
            hide_index=True,
            column_config={
                QUOTE_REVENUE_COLUMN: st.column_config.NumberColumn(
                    "External benchmark", format="EUR %,.0f/MW/yr"
                ),
                MODEL_REVENUE_COLUMN: st.column_config.NumberColumn(
                    "Platform model", format="EUR %,.0f/MW/yr"
                ),
                "model_minus_benchmark_eur_per_mw_yr": (
                    st.column_config.NumberColumn(
                        "Model - benchmark", format="EUR %+.0f/MW/yr"
                    )
                ),
                "benchmark_to_model_ratio": st.column_config.NumberColumn(
                    "Benchmark / model", format="%.1f%%"
                ),
                "coverage_pct": st.column_config.NumberColumn(
                    "Model coverage", format="%.1f%%"
                ),
            },
        )
        assumptions = _benchmark_assumptions(
            selected,
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            capture_rate=capture_rate,
        )
        workbook = cockpit_tables_to_excel(
            {
                "External benchmark": selected,
                "Benchmark reconciliation": comparison,
                "Platform annual curve": model_yearly.loc[
                    model_yearly["zone"] == zone
                ],
            },
            assumptions=assumptions,
        )
        st.download_button(
            "Download benchmark reconciliation (Excel)",
            data=workbook,
            file_name="external_trader_benchmark_reconciliation.xlsx",
            mime=_XLSX_MIME,
            key="external_benchmark_download",
        )


def render(
    *,
    zone_data: dict[str, pd.DataFrame],
    power_mw: float,
    duration_hours: float,
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
            normalised per contract window so the synthetic mean equals
            the forward base over the contract period. The historical
            shape comes from the cached DA series for the same zone (the
            one shown in Market Overview). This preserves the intra-day
            spread pattern that drives BESS arbitrage while letting the
            level move with the forward.

            **Caveats** to read every output through:
            - **Shape drift over the horizon**: high-RE penetration is
              steepening the "duck curve" year-on-year. Applying a 2025
              shape to a Cal-2030 contract systematically understates
              the spread you'd actually see in 2030 — treat far-dated
              numbers as a low bound, and consider running a sensitivity
              with an RE-amplified shape if you trust the analysis to
              drive investment sizing.
            - Capture rate is applied identically to forward and historical
              analyses; revisit this assumption when the forward implies a
              very different volatility regime.
            - **Peak / off-peak rows are not yet shape-aware**: the v1
              engine treats every row's price as a baseload. A "peak"
              quote will not be applied to the peak hours specifically;
              upload baseload quotes only until that's added.
            - **Forward is a single number per period**: this engine
              cannot recover intra-period shape from forwards alone.
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
        _render_external_benchmark_section(
            model_yearly=build_forward_model_yearly(
                {},
                power_mw=power_mw,
                duration_hours=duration_hours,
                efficiency=efficiency,
                capture_rate=capture_rate,
            ),
            power_mw=power_mw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            capture_rate=capture_rate,
            chart_template=chart_template,
        )
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
    all_daily: dict[str, pd.DataFrame] = {}
    # Long-horizon contracts (multi-year forwards) make the MILP path
    # sequential and slow — show a progress bar so the user knows the
    # browser hasn't frozen.
    progress = st.progress(0.0, text="Solving forward dispatch...")
    for i, zone in enumerate(fwd_with_history):
        tz = get_zone_timezone(zone)
        synth = build_forward_synthetic_prices(
            forward_df, zone_data[zone], zone=zone, tz=tz,
        )
        if synth.empty:
            progress.progress((i + 1) / len(fwd_with_history), text=f"Skipped {zone} (no synth)")
            continue
        all_synth[zone] = synth
        progress.progress(
            (i + 0.3) / len(fwd_with_history),
            text=f"Solving MILP dispatch for {zone} ({len(synth):,} hours)...",
        )
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
        all_daily[zone] = daily
        summary = summarise_forward_revenue(
            daily, forward_df, synth, zone=zone,
            power_mw=power_mw, duration_hours=duration_hours,
            efficiency=efficiency, capture_rate=capture_rate,
            tz=tz,
        )
        if summary.empty:
            progress.progress((i + 1) / len(fwd_with_history), text=f"No revenue for {zone}")
            continue
        summary.insert(0, "zone", zone)
        all_summaries.append(summary)
        progress.progress((i + 1) / len(fwd_with_history), text=f"Done: {zone}")
    progress.empty()

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

    model_yearly = build_forward_model_yearly(
        all_daily,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        capture_rate=capture_rate,
    )
    _render_external_benchmark_section(
        model_yearly=model_yearly,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        capture_rate=capture_rate,
        chart_template=chart_template,
    )

    # Synthetic price preview chart per zone. For multi-year forwards the
    # hourly point count (>8760 per year) makes Plotly sluggish; downsample
    # to a daily mean when the period exceeds ~3000 hours so the browser
    # stays responsive.
    st.markdown("**Synthetic hourly price preview**")
    zone_pick = st.selectbox(
        "Zone", options=list(all_synth.keys()),
        index=0, key="forward_preview_zone",
    )
    synth_pick_full = all_synth[zone_pick]
    if len(synth_pick_full) > 3000:
        synth_pick = (
            synth_pick_full[["price_eur_mwh", "contract"]]
            .resample("D")
            .agg({"price_eur_mwh": "mean", "contract": "first"})
            .reset_index()
        )
        chart_title = (
            f"Synthetic forward daily-mean prices - {zone_pick} "
            f"(downsampled from {len(synth_pick_full):,} hourly points)"
        )
    else:
        synth_pick = synth_pick_full.reset_index()
        chart_title = f"Synthetic forward hourly prices - {zone_pick}"
    fig = px.line(
        synth_pick,
        x="timestamp", y="price_eur_mwh",
        color="contract",
        title=chart_title,
        labels={"price_eur_mwh": "EUR/MWh", "timestamp": ""},
        template=chart_template,
    )
    st.plotly_chart(fig, width="stretch")

    # Download summary. Neutralise spreadsheet formula injection on any
    # user-supplied string column (the ``contract`` field is the obvious
    # vector — an uploaded `=HYPERLINK("//evil/", "x")` would be executed
    # by Excel / Calc / Sheets when the user opens the export). Include
    # both ``object`` and pandas StringDtype so a future dtype-upgrade
    # cannot silently re-introduce the hole (Gemini-3.1 P1).
    _FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r", "\n", "|")

    def _safe_csv(v):
        if isinstance(v, str) and v:
            stripped = v.lstrip()
            if stripped and stripped[0] in _FORMULA_TRIGGERS:
                return "'" + v
        return v

    safe_summary = summary_all.copy()
    for col in safe_summary.select_dtypes(include=["object", "string"]).columns:
        safe_summary[col] = safe_summary[col].map(_safe_csv)
    st.download_button(
        "Download forward summary (CSV)",
        data=safe_summary.to_csv(index=False),
        file_name="forward_scenario_summary.csv",
        mime="text/csv",
    )
