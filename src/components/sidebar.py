"""Sidebar controls and state management for BESS Pulse dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ancillary import (
    ANCILLARY_TEMPLATES,
    generate_template_csv,
    parse_ancillary_csv,
)
from src.ancillary_fetchers import get_available_fetchers, run_auto_fetch
from src.config import ALL_ZONES
from src.data_ingestion import (
    DataSourceAuthError,
    DataSourceNetworkError,
    DataSourceParseError,
    build_zone_query_window,
    fetch_prices,
)

ANCILLARY_STATE_KEYS = (
    "ancillary_df",
    "ancillary_template",
    "auto_fetch_results",
    "ancillary_zone",
    "ancillary_dates",
)


def _clear_stale_ancillary_state() -> None:
    """Remove ancillary data that no longer matches the active sidebar scope."""
    for key in ANCILLARY_STATE_KEYS:
        st.session_state.pop(key, None)


def _store_ancillary_scope(zone: str, start: object, end: object) -> None:
    """Persist the zone/date scope associated with ancillary data in session state."""
    st.session_state["ancillary_zone"] = zone
    st.session_state["ancillary_dates"] = (str(start), str(end))


def _clear_stale_price_state() -> None:
    """Remove fetched price data when the sidebar scope changes."""
    for key in ("zone_data", "selected_zones", "fetched_zone_date_scope", "refresh_token"):
        st.session_state.pop(key, None)


def _parse_and_store_ancillary_upload(
    uploaded_file,
    template_key: str,
    zone: str,
    start: object,
    end: object,
) -> None:
    """Parse one uploaded ancillary CSV and store it under the active scope."""
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig")
        parsed = parse_ancillary_csv(content, template_key)
        st.session_state["ancillary_df"] = parsed
        st.session_state["ancillary_template"] = template_key
        _store_ancillary_scope(zone, start, end)
        st.success(f"{template_key} loaded: {len(parsed)} rows")
    except UnicodeDecodeError:
        st.error("Parse error: the uploaded file is not valid UTF-8/CSV text.")
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Parse error: {exc}")


def _run_and_store_ancillary_fetch(zone: str, start: object, end: object) -> None:
    """Run the configured ancillary auto-fetchers and store successful results."""
    fetchers = get_available_fetchers(zone)
    with st.spinner(f"Fetching from {', '.join(f['source'] for f in fetchers)}..."):
        auto_start, auto_end = build_zone_query_window(zone, start, end)
        results = run_auto_fetch(zone, auto_start, auto_end)
        if results:
            st.session_state["auto_fetch_results"] = results
            _store_ancillary_scope(zone, start, end)
            st.success(f"Fetched {len(results)} dataset(s): " + ", ".join(results.keys()))
        else:
            st.warning("No data returned. Try manual CSV upload instead.")


# ── Data loading (cached) ────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def load_zone_data(
    zone: str,
    start: str,
    end: str,
    force_refresh: bool = False,
    refresh_token: int = 0,
) -> pd.DataFrame:
    """Fetch price data for a zone (cached by Streamlit)."""
    del refresh_token
    api_start, api_end = build_zone_query_window(zone, start, end)
    return fetch_prices(
        zone=zone,
        start=api_start,
        end=api_end,
        use_cache=not force_refresh,
    )


@st.cache_data(show_spinner=False)
def load_generation(zone: str, start: str, end: str, refresh_token: int = 0) -> pd.DataFrame:
    """Fetch generation data for a zone (cached by Streamlit)."""
    from src.data_ingestion import fetch_generation_data

    del refresh_token
    api_start, api_end = build_zone_query_window(zone, start, end)
    return fetch_generation_data(
        zone=zone,
        start=api_start,
        end=api_end,
    )


def _format_data_error(exc: Exception) -> str:
    """Convert internal fetch errors into concise user-facing messages."""
    if isinstance(exc, (DataSourceAuthError, DataSourceNetworkError, DataSourceParseError, ValueError)):
        return str(exc)
    return "Unexpected data fetch failure. Check logs for details."


# ── Sidebar rendering ────────────────────────────────────────────────────────


def render_sidebar() -> dict:
    """Render the sidebar and return all user-selected parameters.

    Returns:
        Dict with keys: selected_zones, start_date, end_date, power_mw,
        duration_hours, efficiency, capture_rate, capex_eur_kwh,
        use_lp_dispatch, force_refresh, chart_template,
        primary_zone_for_fetch, fetch_btn.
    """
    st.sidebar.title("BESS Pulse")
    st.sidebar.caption("European BESS Market Screening")

    # Zone selector
    zone_options = {v: f"{k} ({v})" for k, v in ALL_ZONES.items()}
    selected_zones = st.sidebar.multiselect(
        "Bidding Zones",
        options=list(zone_options.keys()),
        default=["DE_LU"],
        format_func=lambda x: zone_options[x],
    )

    # Date range (inclusive local calendar dates per selected bidding zone)
    col1, col2 = st.sidebar.columns(2)
    default_end = pd.Timestamp.now().normalize()
    default_start = default_end - pd.Timedelta(days=30)
    start_date = col1.date_input("Start", value=default_start.date())
    end_date = col2.date_input("End", value=default_end.date())
    primary_zone_for_fetch = selected_zones[0] if selected_zones else "DE_LU"

    # Scope change detection
    current_zone_date_scope = (tuple(selected_zones), str(start_date), str(end_date))
    previous_zone_date_scope = st.session_state.get("zone_date_scope")
    if previous_zone_date_scope is not None and previous_zone_date_scope != current_zone_date_scope:
        _clear_stale_ancillary_state()
        _clear_stale_price_state()
    st.session_state["zone_date_scope"] = current_zone_date_scope

    # BESS parameters — wrapped in a form so slider/input changes don't trigger
    # a full Streamlit re-run until the user clicks "Apply".
    st.sidebar.subheader("BESS Parameters")
    with st.sidebar.form("bess_params"):
        st.caption("Changes here update the dashboard only after you click Apply.")
        power_mw = st.number_input("Power (MW)", min_value=0.1, value=10.0, step=1.0)
        duration_hours = st.selectbox("Duration (h)", [1, 2, 4], index=0)
        efficiency = st.slider("Efficiency (%)", 85, 95, 88) / 100.0
        capture_rate = st.slider("Capture (%)", 30, 100, 70) / 100.0
        capex_eur_kwh = st.number_input(
            "CapEx (EUR/kWh)",
            min_value=0.0, value=0.0, step=50.0,
            help="Enter installed CapEx to calculate payback period. Leave 0 to skip.",
        )
        use_lp_dispatch = st.checkbox(
            "Multi-cycle LP dispatch",
            value=False,
            help="Use LP optimizer for multi-cycle dispatch instead of greedy single-cycle heuristic.",
        )
        st.form_submit_button("Apply BESS parameters", type="primary")
    force_refresh = st.sidebar.checkbox(
        "Force Refresh",
        value=False,
        help="Bypass Streamlit and local price caches for the next fetch only.",
    )

    # Chart theme
    chart_template = st.sidebar.selectbox(
        "Chart Theme", ["plotly_dark", "plotly_white", "plotly"], index=0,
    )

    # ── Ancillary services upload ────────────────────────────────────────
    with st.sidebar.expander("Ancillary Services Data"):
        anc_template = st.selectbox(
            "Template", list(ANCILLARY_TEMPLATES.keys()),
            format_func=lambda k: f"{k} — {ANCILLARY_TEMPLATES[k]['description'][:40]}",
        )
        anc_file = st.file_uploader("Upload CSV", type=["csv"], key="anc_upload")

        if anc_file is not None and st.button("Parse & Import"):
            _parse_and_store_ancillary_upload(
                anc_file,
                anc_template,
                primary_zone_for_fetch,
                start_date,
                end_date,
            )

        if "ancillary_df" in st.session_state:
            st.caption(f"Loaded: {st.session_state['ancillary_template']} "
                       f"({len(st.session_state['ancillary_df'])} rows)")

        # Template download
        tmpl_key = st.selectbox(
            "Download template",
            list(ANCILLARY_TEMPLATES.keys()),
            key="tmpl_select",
        )
        st.download_button(
            label=f"\U0001f4e5 Download {tmpl_key} template",
            data=generate_template_csv(tmpl_key),
            file_name=f"{tmpl_key}_template.csv",
            mime="text/csv",
            key="tmpl_download",
        )

    # ── Auto-fetch ancillary data ────────────────────────────────────────
    with st.sidebar.expander("Auto-Fetch Ancillary Data"):
        fetchers = get_available_fetchers(primary_zone_for_fetch)

        if fetchers:
            st.markdown("**Available for this zone:**")
            for f in fetchers:
                st.caption(f"\U0001f4e1 {f['name']} ({f['source']})")
            if st.button(
                f"\u26a1 Fetch ancillary data for {primary_zone_for_fetch}",
                key="auto_fetch_ancillary",
            ):
                _run_and_store_ancillary_fetch(primary_zone_for_fetch, start_date, end_date)
        else:
            st.info("No auto-fetch available for this zone. Use CSV upload.")

    fetch_btn = st.sidebar.button("Fetch Data", type="primary", width="stretch")

    return {
        "selected_zones": selected_zones,
        "start_date": start_date,
        "end_date": end_date,
        "power_mw": power_mw,
        "duration_hours": duration_hours,
        "efficiency": efficiency,
        "capture_rate": capture_rate,
        "capex_eur_kwh": capex_eur_kwh,
        "use_lp_dispatch": use_lp_dispatch,
        "force_refresh": force_refresh,
        "chart_template": chart_template,
        "primary_zone_for_fetch": primary_zone_for_fetch,
        "fetch_btn": fetch_btn,
        "current_zone_date_scope": current_zone_date_scope,
    }
