"""Synthetic Revenue harness for manual check 42 (2026-10-03).

Run from an isolated `git archive 68d3486` tree:
  PYTHONPATH=. streamlit run <this file> --server.address 127.0.0.1 --server.port 8632
Reuses the Step 3D display-test page (`tests.test_step3d_export_disclosure.
_revenue_app`): the production Revenue page with two capacity products
(FCR, aFRR Up) and one energy-only product (mFRR Up) in the session
ancillary frame, the real ancillary aggregation and the real joint solver.
Unrelated subpanels are stubbed by that factory. This file only adds the
production Excel/PDF exporters fed with the page's own export dict, so the
browser downloads the same disclosure the page shows. No cache is touched.
"""

import streamlit as st

from src.export import export_to_bytes, export_to_pdf_bytes
from src.ui_theme import inject_global_cockpit_theme
from tests.test_step3a_display_contract import _export_args
from tests.test_step3d_export_disclosure import _revenue_app

st.set_page_config(page_title="Check 42 synthetic Revenue", layout="wide",
                   initial_sidebar_state="expanded")
inject_global_cockpit_theme()
st.sidebar.title("Synthetic evidence only")
st.sidebar.write("DE_LU 2026-03-29: FCR + aFRR Up capacity, mFRR Up energy-only.")
_revenue_app()

export = st.session_state["test_export"]
import datetime as dt  # noqa: E402

import pandas as pd  # noqa: E402

lo = pd.Timestamp(dt.date(2026, 3, 29), tz="Europe/Berlin")
index = pd.date_range(lo, lo + pd.DateOffset(days=1), freq="15min",
                      inclusive="left").tz_convert("UTC").rename("timestamp")
frame = pd.DataFrame({"price_eur_mwh": 40.0}, index=index)
args = _export_args(frame)
args["revenue_estimate"] = export
st.subheader("Exports from this page's export dict")
st.download_button("Export to Excel", export_to_bytes(**args),
                   file_name="check42_DE_LU_report.xlsx")
st.download_button("Export to PDF", export_to_pdf_bytes(**args),
                   file_name="check42_DE_LU_report.pdf")
