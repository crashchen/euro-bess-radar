"""Synthetic real Revenue page; no cache, only unrelated subpanels stubbed."""
import streamlit as st
from tests.test_step3d_export_disclosure import _revenue_app
from src.ui_theme import inject_global_cockpit_theme
st.set_page_config(page_title='Step 3D synthetic Revenue', layout='wide', initial_sidebar_state='expanded')
inject_global_cockpit_theme()
st.sidebar.title('Synthetic evidence only')
st.sidebar.write('DE_LU spring DST. Real joint solver and capacity aggregation. No production cache.')
_revenue_app()
