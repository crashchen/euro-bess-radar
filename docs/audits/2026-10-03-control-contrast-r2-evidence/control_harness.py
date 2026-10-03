"""Offline controls plus the production Data Trust renderer; no market cache.

RADAR_UI_SOURCE selects a source checkout/archive. Run with Streamlit 1.55:
RADAR_UI_SOURCE=<source> python -m streamlit run <this file> --theme.base light
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.environ['RADAR_UI_SOURCE'])
import pandas as pd
import streamlit as st
from src import config, data_ingestion
from src.pages import data_trust
from src.ui_theme import inject_global_cockpit_theme

# Readers use these module constants; use an empty disposable cache. Do not
# load app.py, .env, provider clients, or the workstation's market database.
if 'fixture_cache' not in st.session_state:
    st.session_state.fixture_cache = tempfile.mkdtemp(prefix='radar-control-fixture-')
config.CACHE_DIR = Path(st.session_state.fixture_cache)
data_ingestion.CACHE_DIR = config.CACHE_DIR
st.set_page_config(layout='wide')
inject_global_cockpit_theme()
st.sidebar.write('Control fixture — no live sources')
st.sidebar.button('Sidebar help button', help='Fixture tooltip', key='side_button')
st.sidebar.file_uploader('Sidebar upload', key='side_upload')
st.sidebar.file_uploader('Sidebar disabled upload', disabled=True, key='side_disabled')
st.title('PR #105 button contrast fixture')
controls, trust = st.tabs(['Controls', 'Data Trust'])
with controls:
    st.download_button('Export Project Revenue Handoff JSON', '{}', help='Fixture tooltip', key='handoff')
    st.button('Enabled help button', help='Fixture tooltip', key='enabled')
    st.button('Disabled help button', help='Fixture tooltip', disabled=True, key='disabled')
    st.file_uploader('Main upload', key='main_upload')
    st.file_uploader('Main disabled upload', disabled=True, key='main_disabled')
    with st.expander('Contract controls', expanded=True):
        st.button('Contract help button', help='Fixture tooltip', key='contract_button')
        st.file_uploader('Source document', key='contract_upload')
        st.file_uploader('Contract disabled upload', disabled=True, key='contract_disabled')
    with st.form('control_form'):
        st.form_submit_button('Submit fixture', help='Fixture tooltip')
with trust:
    prices = pd.DataFrame({'price_eur_mwh': 50.0, 'filled': False, 'imputed': False},
                          index=pd.date_range('2026-06-08', periods=96, freq='15min', tz='UTC'))
    data_trust.render(zone_data={'DE_LU': prices},
                      zone_timezones={'DE_LU': 'Europe/Berlin'}, primary_zone='DE_LU')
