"""Synthetic layout harness: real market, Project Case and Cockpit renderers.

Only synthetic prices; every cache path points at a temporary directory.
Run from the repository root with PYTHONPATH=. so ``src`` and ``tests`` import.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

import src.data_ingestion as ingestion
import src.export as export

_TMP = Path(tempfile.gettempdir()) / "step3c-layout-cache"
_TMP.mkdir(exist_ok=True)
ingestion.CACHE_DIR = _TMP
ingestion.DB_PATH = _TMP / "bess_pulse.db"
export.CACHE_DIR = _TMP

from src.analytics import (  # noqa: E402
    calculate_daily_spreads,
    calculate_negative_price_hours,
    calculate_spread_percentiles,
    filter_to_complete_local_days,
)
from src.ui_theme import cockpit_chart_template, inject_global_cockpit_theme  # noqa: E402

st.set_page_config(page_title="Step 3C layout harness", layout="wide")
inject_global_cockpit_theme()

_SECTIONS = ["Market Overview", "Project Case", "Cockpit", "Cockpit KPI gallery"]
_requested = st.query_params.get("section", "Market Overview")
section = st.sidebar.radio(
    "Section", _SECTIONS,
    index=_SECTIONS.index(_requested) if _requested in _SECTIONS else 0,
)
power_mw = st.sidebar.number_input("Power (MW)", value=250.0, step=10.0)
duration_hours = st.sidebar.selectbox("Duration (h)", [1, 2, 4], index=2)
st.sidebar.slider("Capture rate", 0.5, 1.0, 0.7)
st.sidebar.number_input("CapEx (EUR/kWh)", value=180.0)

rng = np.random.default_rng(7)
before = pd.date_range("2025-09-18", "2025-10-01", inclusive="left", freq="h", tz="Europe/Berlin")
after = pd.date_range("2025-10-01", "2025-10-08", inclusive="left", freq="15min", tz="Europe/Berlin")
index = before.append(after).tz_convert("UTC").rename("timestamp")
local_hour = index.tz_convert("Europe/Berlin").hour.to_numpy()
prices = 85.0 + 70.0 * np.sin((local_hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 25, len(index))
prices[local_hour == 13] -= 180.0  # a few deep negative middays
da = pd.DataFrame({"price_eur_mwh": prices}, index=index)
template = cockpit_chart_template()

if section == "Market Overview":
    from src.pages.market_overview import render

    daily = calculate_daily_spreads(da, tz="Europe/Berlin", duration_hours=duration_hours)
    render(
        "DE_LU", da, daily, calculate_spread_percentiles(daily),
        calculate_negative_price_hours(filter_to_complete_local_days(da, tz="Europe/Berlin")),
        duration_hours, "Europe/Berlin", template, {},
    )
elif section == "Project Case":
    from src.pages.project_case import _render_outcome, render_project_case_result
    from src.project_case import compute_project_case
    from tests import pc_case_fixtures as fx

    st.subheader("Stress: long negative and large amounts")
    stress = SimpleNamespace(
        available=True, status="ok", message="",
        distribution=SimpleNamespace(
            p10=-12_345_678.0, p50=2_560_914.0, p90=123_456_789.0, prob_positive=0.57,
        ),
    )
    _render_outcome("No-lifecycle-cost screening NPV", stress)
    _render_outcome("Pre-tax unlevered lifecycle cash NPV", stress)
    st.subheader("Real fixture RunResult")
    render_project_case_result(compute_project_case(fx.project_case()))
elif section == "Cockpit KPI gallery":
    import src.pages.simulation_cockpit as cockpit

    st.subheader("Cockpit KPI gallery")
    st.caption("Synthetic presentation fixtures; real production renderers; no solver calls.")
    st.markdown("### Multi-day replay")
    batch = pd.DataFrame({
        "date": ["2026-09-01", "2026-09-02"],
        "total_revenue_eur": [195_980.0, 196_000.0],
        "annualized_eur_per_mw": [286_331.0, 286_351.0],
        "daily_fce": [1.55, 1.75],
    })
    cockpit._render_batch_kpis(
        batch, requested_days=3, excluded_days=1, carry_mode="continuous_horizon",
    )

    st.markdown("### Forecast policy")
    cockpit._render_forecast_policy_kpis({
        "total_da_only_eur": 195_990.0, "total_realised_eur": 286_341.0,
        "total_captured_eur": 90_351.0, "total_ceiling_eur": 345_678.0,
        "total_forecast_error_eur": 59_337.0, "capture_rate": 0.60,
        "total_ceiling_uplift_eur": 149_688.0,
    })
    cockpit._render_forecast_skill({
        "n_points": 12_345, "mae": 123.4, "bias": -123.4,
        "rmse": 234.5, "skill_vs_da": -0.25, "realised_std": 135.7,
    }, template)

    st.markdown("### Reserve forecast gap")
    cockpit._render_reserve_gap_panel({
        "valid_days": 30, "total_realised_eur": 195_990.0,
        "total_global_ceiling_eur": 345_678.0,
        "total_forecast_effect_eur": -23_456.0,
        "total_timing_cost_eur": 173_144.0,
        "total_full_gap_eur": 149_688.0,
    }, "FCR", template)

    st.markdown("### Stochastic attribution and risk")
    policy_value = 286_341.0 * 30.0 / cockpit.DAYS_PER_YEAR
    cockpit._render_stochastic_attribution_panel({
        "valid_days": 30, "total_policy_value_eur": policy_value,
        "rebid_cap_mw": 123.4,
        "total_commitment_value_eur": -123_456.0,
        "total_distribution_value_eur": 123_456.0 + policy_value,
        "risk_block": {
            "n": 12_345, "p10": -195_990.0, "p50": 286_341.0,
            "p90": 345_678.0, "cvar90": -234_567.0,
        },
    }, power_mw=1.0)

    st.markdown("### Cycle frontier")
    frontier_row = {
        name: 0.0 for name in cockpit._FRONTIER_DISPLAY_COLUMNS if name != "best"
    }
    frontier_row.update({
        "cycle_cap": 1.5, "label": "1.5 FEC/day",
        "gross_eur": 345_678.0, "wear_eur": 59_337.0, "net_eur": 286_341.0,
        "gross_eur_per_mw_yr": 345_678.0,
        "wear_eur_per_mw_yr": 59_337.0, "net_eur_per_mw_yr": 286_341.0,
        "avg_efc_per_day": 1.42, "cycle_limited_life_years": 13.5,
        "charge_vwap_eur_mwh": -12.3, "discharge_vwap_eur_mwh": 123.4,
    })
    frontier = pd.DataFrame([frontier_row])
    frontier_summary = {
        "best_cap_label": "1.5 FEC/day", "cost_per_cycle_eur": 12_345.67,
        "wear_eur_per_mwh_discharged": 123.45, "valid_days": 30,
        "excluded_days": 2, "cycle_life": 7_000.0, "capex_eur_kwh": 180.0,
    }
    cockpit._render_frontier_result(frontier, frontier_summary, template)

    st.markdown("### Contracted floor")
    cockpit._render_contracted_floor_result(
        frontier_context={
            "frontier": frontier, "summary": frontier_summary,
            "sweep_dates": ["2026-09-01", "2026-09-30"], "power_mw": 1.0,
            "duration_hours": 4.0, "primary_zone": "DE_LU",
        },
        result={
            "merchant_net_eur": 286_341.0,
            "merchant_net_eur_per_mw_yr": 286_341.0,
            "quoted_floor_eur": 400_000.0, "contract_availability": 0.95,
            "effective_floor_eur": 380_000.0,
            "effective_floor_eur_per_mw_yr": 380_000.0,
            "floor_protected_cashflow_eur": 380_000.0,
            "annual_top_up_eur": 93_659.0, "floor_tenor_years": 10.0,
            "discount_rate": 0.08, "merchant_pv_eur": 1_921_924.0,
            "floor_protected_pv_eur": 2_550_230.0,
            "floor_pv_uplift_eur": 628_306.0,
        },
        chart_template=template,
    )
else:
    import src.pages.simulation_cockpit as cockpit

    ida = pd.DataFrame(
        {"intraday_price_eur_mwh": prices + rng.normal(0, 15, len(index))}, index=index,
    )
    render_kwargs = dict(
        primary_zone="DE_LU", primary_df=da, intraday_df=ida, anc_df=None,
        power_mw=power_mw, duration_hours=duration_hours, efficiency=0.88,
        capture_rate=0.7, capex_eur_kwh=180.0, zone_tz="Europe/Berlin",
        chart_template=template,
    )
    cockpit._STOCHASTIC_N_SCENARIOS = 2
    cockpit.render(**render_kwargs)
