"""Local, synthetic layout harness for three production page renderers.

Run from the repository root with PYTHONPATH=. .venv/bin/streamlit run <this file>.
It reads no production cache and changes no market or settlement calculations.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.pages import data_trust, forward_scenarios, renewable_correlation
from src.trader_benchmark import build_forward_model_yearly
from src.ui_theme import cockpit_chart_template, inject_global_cockpit_theme

st.set_page_config(page_title="Synthetic metric layout", layout="wide")
inject_global_cockpit_theme()
st.sidebar.title("Synthetic evidence only")
page = st.sidebar.radio(
    "Production renderer", ["Data Trust", "Renewable Correlation", "Forward benchmark"]
)

index = pd.date_range("2026-01-01", periods=24 * 8, freq="h", tz="UTC")
prices = pd.DataFrame(
    {
        "price_eur_mwh": [35.0 + (i % 24) * 3 for i in range(len(index))],
        "filled": False,
        "imputed": False,
    },
    index=index,
)

if page == "Data Trust":
    # Keep the source/provenance sidecars out of this UI-only harness.
    data_trust.build_coverage_matrix = lambda *args, **kwargs: pd.DataFrame()
    data_trust.build_intraday_source_table = lambda: pd.DataFrame()
    data_trust.build_capacity_source_table = lambda: pd.DataFrame()
    data_trust.build_activation_source_table = lambda: pd.DataFrame()
    data_trust.build_imbalance_source_table = lambda: pd.DataFrame()
    data_trust.render(
        zone_data={"DE_LU": prices},
        zone_timezones={"DE_LU": "Europe/Berlin"},
        primary_zone="DE_LU",
    )
elif page == "Renewable Correlation":
    generation = pd.DataFrame(
        {"renewable_pct": [10.0 + (i % 24) * 3 + (i // 24) for i in range(len(index))]},
        index=index,
    )
    renewable_correlation.load_generation = lambda *args, **kwargs: generation
    renewable_correlation.render(
        "DE_LU", prices, index[0].date(), index[-1].date(),
        1, "Europe/Berlin", cockpit_chart_template(), 0,
    )
else:
    benchmark_csv = (
        "zone,scenario,year,revenue_eur_per_mw_yr,asset_type,market_scope,"
        "revenue_basis,duration_hours,max_efc_per_day,source,as_of\n"
        "DE_LU,Base,2027,123456,standalone,da-only,gross,2,,Synthetic,2026-01-01\n"
        "DE_LU,Base,2028,145678,standalone,da-only,gross,2,,Synthetic,2026-01-01\n"
    )

    class SyntheticUpload:
        def getvalue(self):
            return benchmark_csv.encode()

    DeltaGenerator.file_uploader = lambda *args, **kwargs: SyntheticUpload()
    model_yearly = build_forward_model_yearly(
        {"DE_LU": pd.DataFrame({
            "date": [pd.Timestamp("2027-01-01"), pd.Timestamp("2028-01-01")],
            "spread": [100, 120],
            "lp_revenue": [600, 800],
            "n_cycles": [1, 1],
        })},
        power_mw=1, duration_hours=2, efficiency=.9, capture_rate=.75,
    )
    forward_scenarios._render_external_benchmark_section(
        model_yearly=model_yearly,
        power_mw=1, duration_hours=2, efficiency=.9, capture_rate=.75,
        chart_template=cockpit_chart_template(),
    )
