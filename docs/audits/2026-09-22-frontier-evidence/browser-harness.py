"""Isolated synthetic real-panel smoke; no market cache or network reads.

Run from repository root:
PYTHONPATH=. .venv/bin/streamlit run docs/audits/2026-09-22-frontier-evidence/browser-harness.py --server.port 8613
"""
import pandas as pd
import streamlit as st

import src.cycle_frontier as frontier_module
import src.pages.simulation_cockpit as cockpit
from src.simulation import available_local_dates

st.set_page_config(page_title="Frontier identity verification", layout="wide")
st.title("Synthetic Frontier identity verification")
st.caption("Synthetic hourly prices; real production panels and solver. No cache reads or writes.")
st.checkbox("Correct one price without changing shape", key="probe_corrected")
st.checkbox("Use light chart theme", key="probe_light")
st.text_input("Audit-only source marker", value="run-original", key="probe_marker")
if "cycle_frontier_caps" not in st.session_state:
    st.session_state.cycle_frontier_caps = ["1", "uncapped"]

idx = pd.date_range("2025-03-02", periods=48, freq="h", tz="UTC")
day = [10.0] * 6 + [100.0] * 3 + [10.0] * 6 + [100.0] * 3 + [50.0] * 6
prices = pd.DataFrame({"price_eur_mwh": day * 2}, index=idx)
prices.index.name = "timestamp"
if st.session_state.probe_corrected:
    prices.iloc[6, 0] = 900.0
assumptions = pd.DataFrame([{
    "parameter": "Source marker", "value": st.session_state.probe_marker,
    "unit": "", "source": "Synthetic fixture", "affects": "Audit only",
}])

original_compute = cockpit.compute_cycle_cap_frontier
original_solve = frontier_module.solve_daily_lp

def counted_compute(*args, **kwargs):
    st.session_state.probe_compute_calls = st.session_state.get("probe_compute_calls", 0) + 1
    return original_compute(*args, **kwargs)


def counted_solve(*args, **kwargs):
    st.session_state.probe_solver_calls = st.session_state.get("probe_solver_calls", 0) + 1
    return original_solve(*args, **kwargs)


cockpit.compute_cycle_cap_frontier = counted_compute
frontier_module.solve_daily_lp = counted_solve
try:
    theme = "plotly_white" if st.session_state.probe_light else "plotly_dark"
    context = cockpit._render_cycle_frontier_section(
        primary_zone="DE_LU", primary_df=prices,
        dates=available_local_dates(prices, tz="UTC"), zone_tz="UTC",
        power_mw=1.0, duration_hours=1, efficiency=0.9,
        capex_eur_kwh=150.0, chart_template=theme, assumptions=assumptions,
    )
    cockpit._render_contracted_floor_section(
        frontier_context=context, chart_template=theme, assumptions=assumptions,
    )
finally:
    cockpit.compute_cycle_cap_frontier = original_compute
    frontier_module.solve_daily_lp = original_solve

st.write({
    "frontier_compute_calls": st.session_state.get("probe_compute_calls", 0),
    "daily_solver_calls": st.session_state.get("probe_solver_calls", 0),
    "frontier_context_current": context is not None,
})
