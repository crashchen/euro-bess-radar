"""Local synthetic Step 3D cockpit presentation; no production cache access.

The test factory stubs the sequential/reserve/triple/stochastic solver results
and capacity-series lookup. It still runs the production bundle/disclosure and
strategy-comparison builders. On render only unrelated KPI, forecast plot/skill
and stochastic-attribution panels are suppressed, because the small fixture
does not contain their inputs. The production capacity caption, strategy chart,
table, and download/export are left intact.
"""

from pathlib import Path
from tempfile import gettempdir

import pytest
import streamlit as st

from src.export import cockpit_tables_to_excel
from src.pages import simulation_cockpit as cockpit
from src.ui_theme import inject_global_cockpit_theme
from tests.test_step3d_cockpit_disclosure import _compute_bundle

st.set_page_config(
    page_title="Step 3D synthetic Cockpit", layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_cockpit_theme()
cockpit._inject_cockpit_css()
st.sidebar.title("Synthetic evidence only")
st.sidebar.caption("No production cache is read or written.")
st.sidebar.caption(
    "Strategy cash is stubbed for this presentation check. This page verifies "
    "the production disclosure, comparison and export, not solver revenue."
)
st.title("DST capacity settlement — Cockpit")
if "step3d_synthetic_bundle" not in st.session_state:
    with pytest.MonkeyPatch.context() as monkeypatch:
        st.session_state["step3d_synthetic_bundle"] = _compute_bundle(monkeypatch)
bundle = st.session_state["step3d_synthetic_bundle"]
with pytest.MonkeyPatch.context() as monkeypatch:
    for name in (
        "_render_forecast_policy_kpis", "_plot_forecast_policy",
        "_render_forecast_skill", "_render_stochastic_attribution_panel",
    ):
        monkeypatch.setattr(cockpit, name, lambda *args, **kwargs: None)
    with st.expander(
        "Forecast-driven IDA policy (vs perfect-foresight ceiling)", expanded=True,
    ):
        cockpit._render_forecast_policy_bundle(bundle, "plotly_dark")

derived = bundle["derived"]
(Path(gettempdir()) / "step3d-cockpit-dst.xlsx").write_bytes(cockpit_tables_to_excel(
    derived["export_tables"], assumptions=derived["export_assumptions"],
))
