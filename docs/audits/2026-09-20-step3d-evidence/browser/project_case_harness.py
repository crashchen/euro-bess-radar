"""Local synthetic Step 3D UI harness; no production cache access."""
from pathlib import Path
from tempfile import gettempdir

import streamlit as st

from src.export import project_case_to_excel
from src.pages.project_case import (
    ProjectCaseRunCache,
    _CACHE_KEY,
    render_project_case_cockpit_mirror,
    render_project_case_result,
)
from src.ui_theme import inject_global_cockpit_theme
from tests.test_step3d_project_case_disclosure import synthetic_reserve_result

st.set_page_config(page_title="Step 3D synthetic Project Case", layout="wide")
inject_global_cockpit_theme()
st.sidebar.title("Synthetic evidence only")
st.sidebar.caption("No production cache is read or written.")
st.title("DST capacity settlement — Project Case")
st.caption("DE_LU · 2026-03-29 · 1 MW · 20 EUR/MW/h · availability 95%")
result = synthetic_reserve_result()
st.session_state[_CACHE_KEY] = ProjectCaseRunCache(
    "synthetic-review", result.input_fingerprint, result
)
render_project_case_result(result)
render_project_case_cockpit_mirror()
output = Path(gettempdir()) / "euro-bess-step3d"
output.mkdir(exist_ok=True)
workbook = project_case_to_excel(result)
(output / "project-case-dst.xlsx").write_bytes(workbook)
st.download_button("Download synthetic Project Case workbook", workbook,
                   file_name="project-case-dst.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
