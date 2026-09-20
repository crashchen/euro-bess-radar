"""Reproduce the three imported-data Cockpit panels using synthetic data only.

Run with the desired checkout on PYTHONPATH; e.g.:
  PYTHONPATH="$PWD" .venv/bin/streamlit run /tmp/euro-bess-step3c-r2/panel_harness.py
The panels and calculations are production functions. Only the two cache reads
are replaced by in-memory fixtures; no production cache is read or written.
"""

from pathlib import Path
from tempfile import gettempdir
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st

from src import config, data_ingestion, export
from src.pages import simulation_cockpit as cockpit
from src.ui_theme import cockpit_chart_template, inject_global_cockpit_theme


def synthetic_frames():
    """Two complete local days, with deliberately wide but computed outputs."""
    local = pd.date_range(
        "2026-01-12", "2026-01-14", freq="15min", inclusive="left",
        tz="Europe/Berlin",
    )
    index = local.tz_convert("UTC").rename("timestamp")
    day = np.repeat([0, 1], 96)
    block = np.asarray(local.hour) // 4
    capacity = pd.DataFrame({
        "capacity_price_eur_mw": 100.0 + day * 123.45 + block * 75.0,
        "product_type": "FCR",
    }, index=index)
    # At the default 1% share, system 25,000 MW becomes the 250 MW cap.
    # 192 quarter hours = 48 h; the explicit targets stress currency width.
    activation = pd.DataFrame({
        "product_type": "aFRR",
        "direction": "up",
        "activation_price_eur_mwh": 123_456_789.0 / (250.0 * 48.0),
        "system_activated_volume_mw": 25_000.0,
    }, index=index)
    direction = np.where(np.arange(len(index)) % 2 == 0, 1.0, -1.0)
    imbalance = pd.DataFrame({
        "imbalance_price_eur_mwh": direction * -98_765_432.0 / (250.0 * 48.0),
        "system_imbalance_volume_mw": direction * 25_000.0,
    }, index=index)
    return capacity, activation, imbalance, sorted(set(local.date))


def main():
    st.set_page_config(
        page_title="Step 3C imported-data panels", layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_cockpit_theme()
    st.sidebar.header("Synthetic layout fixture")
    st.sidebar.caption("Local in-memory data only; no production cache access.")
    power_mw = st.sidebar.number_input(
        "Power (MW)", min_value=1.0, value=250.0, step=10.0,
    )
    st.sidebar.caption("Desktop checks keep this sidebar expanded.")
    st.title("Imported-data Cockpit panels")
    st.caption(
        "Production reserve forecast, activation and imbalance sections. "
        "Open each expander to inspect all eight metrics."
    )
    capacity, activation, imbalance, dates = synthetic_frames()
    scratch = Path(gettempdir()) / "euro-bess-step3c-r2" / "synthetic-cache"
    # Defensive isolation, in addition to substituting both actual cache reads.
    with (
        patch.object(config, "CACHE_DIR", scratch),
        patch.object(config, "DB_PATH", scratch / "synthetic.db"),
        patch.object(data_ingestion, "CACHE_DIR", scratch),
        patch.object(data_ingestion, "DB_PATH", scratch / "synthetic.db"),
        patch.object(export, "CACHE_DIR", scratch),
        patch.object(cockpit, "read_activation_cache", return_value=activation),
        patch.object(cockpit, "read_imbalance_cache", return_value=imbalance),
    ):
        cockpit._render_reserve_forecast_skill_section(
            anc_df=capacity, zone_tz="Europe/Berlin",
            chart_template=cockpit_chart_template(),
        )
        cockpit._render_activation_overlay_section(
            primary_zone="DE_LU", dates=dates, zone_tz="Europe/Berlin",
            power_mw=power_mw,
        )
        cockpit._render_imbalance_overlay_section(
            primary_zone="DE_LU", dates=dates, zone_tz="Europe/Berlin",
            power_mw=power_mw,
        )


if __name__ == "__main__":
    main()
