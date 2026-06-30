"""Sidebar controls and state management for BESS Pulse dashboard."""

from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from src.ancillary import (
    ANCILLARY_TEMPLATES,
    generate_activation_import_template_csv,
    generate_capacity_import_template_csv,
    generate_template_csv,
    parse_activation_import_csv,
    parse_ancillary_csv,
    parse_capacity_import_csv,
)
from src.ancillary_fetchers import get_available_fetchers, run_auto_fetch
from src.config import ALL_ZONES
from src.data_ingestion import (
    DataSourceAuthError,
    DataSourceNetworkError,
    DataSourceParseError,
    build_zone_query_window,
    fetch_prices,
    generate_intraday_template_csv,
    parse_intraday_csv,
    persist_activation_frame,
    persist_capacity_frame,
    persist_intraday_frame,
    read_intraday_sources,
)

ANCILLARY_STATE_KEYS = (
    "ancillary_df",
    "ancillary_template",
    "auto_fetch_results",
    "ancillary_zone",
    "ancillary_dates",
)

DURATION_PRESET_HOURS = (1.0, 2.0, 4.0, 6.0, 8.0)

_UNIFIED_CAPACITY_COLUMNS = {
    "timestamp",
    "zone",
    "product",
    "direction",
    "capacity_price_eur_mw_h",
}


def _format_duration_option(hours: float) -> str:
    """Readable duration labels for the sidebar selectbox."""
    return f"{hours:g}"


def _looks_like_unified_capacity_csv(content: str) -> bool:
    """Return True when a CSV header matches the unified capacity schema.

    This catches a common UI mistake: dropping the zone-tagged capacity import
    file into the legacy per-country ancillary uploader. The two uploaders feed
    different persistence paths, so we detect the schema early and point the
    user to the correct box instead of surfacing a template-column traceback.
    """
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        delimiter = next((delim for delim in [",", ";", "\t"] if delim in line), ",")
        columns = {
            col.strip().lower().lstrip("\ufeff")
            for col in line.split(delimiter)
        }
        return _UNIFIED_CAPACITY_COLUMNS.issubset(columns)
    return False


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


def _parse_and_store_intraday_upload(
    uploaded_file,
    default_zone: str,
    default_sequence: int,
) -> None:
    """Parse an uploaded IDA price CSV, persist it, and flag the manual source.

    The parsed prices are written to the same ``ida_prices_{zone}_seq{n}``
    SQLite tables the live fetch uses, so the Revenue and Simulation Cockpit
    tabs pick them up through their existing cache-first read. Provenance is
    recorded durably in the ``ida_price_sources`` sidecar table (by
    ``persist_intraday_frame`` -> ``write_intraday_cache``), so Data Trust can
    label these rows ``Manual CSV`` even after this session ends.
    """
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig")
        parsed = parse_intraday_csv(
            content, default_zone=default_zone, default_sequence=default_sequence,
        )
    except UnicodeDecodeError:
        st.error("Parse error: the uploaded file is not valid UTF-8/CSV text.")
        return
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Parse error: {exc}")
        return

    summaries = persist_intraday_frame(parsed)
    # Drop any cached session frames so the next render rehydrates from the
    # freshly written SQLite rows (cache key is zone/window scoped).
    for key in [k for k in st.session_state if str(k).startswith("intraday_cache::")]:
        st.session_state.pop(key, None)
    total = sum(s["rows"] for s in summaries)
    pairs = ", ".join(f"{s['zone']} IDA{s['sequence']}" for s in summaries)
    st.success(f"Imported {total} IDA rows ({pairs}).")


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
        if _looks_like_unified_capacity_csv(content):
            st.error(
                "This looks like the unified reserve-capacity CSV. Use the "
                "'Unified Reserve Capacity CSV' uploader below, not the "
                "per-country ancillary template uploader."
            )
            return
        parsed = parse_ancillary_csv(content, template_key)
        st.session_state["ancillary_df"] = parsed
        st.session_state["ancillary_template"] = template_key
        _store_ancillary_scope(zone, start, end)
        st.success(f"{template_key} loaded: {len(parsed)} rows")
    except UnicodeDecodeError:
        st.error("Parse error: the uploaded file is not valid UTF-8/CSV text.")
    except (DataSourceParseError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Parse error: {exc}")


def _parse_and_store_capacity_upload(uploaded_file, default_zone: str) -> None:
    """Parse + persist an uploaded unified reserve-capacity CSV.

    Writes to the ``capacity_prices_{zone}`` SQLite tables + the
    ``capacity_price_sources`` provenance sidecar (via ``persist_capacity_frame``
    -> ``write_capacity_cache``), so Data Trust shows it per
    (zone, product, direction) and survives a restart. This is the zone-tagged
    SQLite/provenance path — distinct from the per-country session-ancillary
    uploader, which is left untouched.
    """
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig")
        parsed = parse_capacity_import_csv(content, default_zone=default_zone)
    except UnicodeDecodeError:
        st.error("Parse error: the uploaded file is not valid UTF-8/CSV text.")
        return
    except DataSourceParseError as exc:
        st.error(f"Capacity import error: {exc}")
        return

    if parsed.empty:
        st.warning("No valid capacity rows found in the uploaded file.")
        return
    try:
        summaries = persist_capacity_frame(parsed, source="Manual CSV")
    except (OSError, sqlite3.DatabaseError, ValueError) as exc:
        st.error(f"Capacity import persistence error: {exc}")
        return
    total = sum(s["rows"] for s in summaries)
    streams = ", ".join(
        f"{s['zone']} {s['product']} {s['direction']}" for s in summaries
    )
    st.success(
        f"Imported {total} reserve-capacity rows ({streams}). See Data Trust → "
        "'Reserve-capacity price sources' and the coverage matrix."
    )


def _parse_and_store_activation_upload(uploaded_file, default_zone: str) -> None:
    """Parse + persist an uploaded unified activation-energy CSV.

    Writes to the ``activation_prices_{zone}`` SQLite tables + the
    ``activation_price_sources`` provenance sidecar (via
    ``persist_activation_frame`` -> ``write_activation_cache``). Energy-leg
    parity with the capacity uploader; ``system_activated_volume_mw`` is stored
    system-level (the asset/capture share is a model assumption applied later).
    """
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig")
        parsed = parse_activation_import_csv(content, default_zone=default_zone)
    except UnicodeDecodeError:
        st.error("Parse error: the uploaded file is not valid UTF-8/CSV text.")
        return
    except DataSourceParseError as exc:
        st.error(f"Activation import error: {exc}")
        return

    if parsed.empty:
        st.warning("No valid activation-energy rows found in the uploaded file.")
        return
    try:
        summaries = persist_activation_frame(parsed, source="Manual CSV")
    except (OSError, sqlite3.DatabaseError, ValueError) as exc:
        st.error(f"Activation import persistence error: {exc}")
        return
    total = sum(s["rows"] for s in summaries)
    streams = ", ".join(
        f"{s['zone']} {s['product']} {s['direction']}" for s in summaries
    )
    st.success(
        f"Imported {total} activation-energy rows ({streams}). System-level "
        "volumes stored as-is; capture share is applied later in the model."
    )


def _run_and_store_ancillary_fetch(zone: str, start: object, end: object) -> None:
    """Run the configured ancillary auto-fetchers and store successful results."""
    fetchers = get_available_fetchers(zone)
    with st.spinner(f"Fetching from {', '.join(f['source'] for f in fetchers)}..."):
        auto_start, auto_end = build_zone_query_window(zone, start, end)
        try:
            results = run_auto_fetch(zone, auto_start, auto_end)
        except DataSourceAuthError as exc:
            st.error(f"Auto-fetch auth error: {exc}")
            return
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
        use_lp_dispatch, force_refresh, primary_zone_for_fetch, fetch_btn.
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
        duration_options = [
            _format_duration_option(hours) for hours in DURATION_PRESET_HOURS
        ]
        duration_choice = st.selectbox("Duration (h)", duration_options, index=0)
        duration_hours = float(duration_choice)
        efficiency = st.slider("Efficiency (%)", 85, 95, 88) / 100.0
        capture_rate = st.slider("Capture (%)", 30, 100, 70) / 100.0
        capex_eur_kwh = st.number_input(
            "CapEx (EUR/kWh)",
            min_value=0.0, value=0.0, step=50.0,
            help="Enter installed CapEx to calculate payback period. Leave 0 to skip.",
        )
        use_lp_dispatch = st.checkbox(
            "Multi-cycle MILP dispatch",
            value=False,
            help="Use MILP optimizer for multi-cycle dispatch instead of greedy single-cycle heuristic.",
        )
        st.form_submit_button("Apply BESS parameters", type="primary")
    force_refresh = st.sidebar.checkbox(
        "Force Refresh",
        value=False,
        help="Bypass Streamlit and local price caches for the next fetch only.",
    )

    st.sidebar.caption("Chart theme: Cockpit dark visual system")

    # ── Ancillary services upload ────────────────────────────────────────
    with st.sidebar.expander("Ancillary Services Data"):
        anc_template = st.selectbox(
            "Template", list(ANCILLARY_TEMPLATES.keys()),
            format_func=lambda k: f"{k} — {ANCILLARY_TEMPLATES[k]['description'][:40]}",
        )
        anc_file = st.file_uploader(
            "Upload per-country ancillary CSV", type=["csv"], key="anc_upload",
        )
        st.caption(
            "Use this box only for the selected per-country template above. "
            "For the zone-tagged unified reserve-capacity file, use the "
            "'Unified Reserve Capacity CSV' uploader below."
        )

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
        st.download_button(
            label="\U0001f4e5 Download unified reserve-capacity template",
            data=generate_capacity_import_template_csv(),
            file_name="reserve_capacity_import_template.csv",
            mime="text/csv",
            key="cap_import_tmpl_download",
        )
        st.caption(
            "Unified zone-tagged reserve-capacity format (EUR/MW/h, UTC) to "
            "request from exchanges/TSOs; one provenance/cache path for all "
            "zones. See docs/import-templates.md."
        )
        st.download_button(
            label="\U0001f4e5 Download activation-energy template",
            data=generate_activation_import_template_csv(),
            file_name="activation_energy_import_template.csv",
            mime="text/csv",
            key="act_import_tmpl_download",
        )
        st.caption(
            "Activation-ENERGY leg (EUR/MWh, UTC), separate from the capacity "
            "fee. system_activated_volume_mw is system-level (the asset/capture "
            "share is a model assumption); historical replay only. See "
            "docs/import-templates.md."
        )

        st.markdown("**Unified Reserve Capacity CSV**")
        st.caption(
            "Import the template above. Writes to the SQLite cache + provenance "
            "sidecar (distinct from the per-country uploader); shows up in Data "
            "Trust per (zone, product, direction)."
        )
        cap_file = st.file_uploader(
            "Upload unified capacity CSV", type=["csv"], key="cap_import_upload",
        )
        if cap_file is not None and st.button(
            "Parse & Import capacity", key="cap_import_btn",
        ):
            _parse_and_store_capacity_upload(cap_file, primary_zone_for_fetch)

        st.markdown("**Unified Activation-Energy CSV**")
        st.caption(
            "Energy leg of reserves (aFRR/mFRR, up/down). Writes to the "
            "activation_prices_{zone} cache + provenance sidecar; system-level "
            "volumes stored as-is (capture share is applied later in the model)."
        )
        act_file = st.file_uploader(
            "Upload unified activation CSV", type=["csv"], key="act_import_upload",
        )
        if act_file is not None and st.button(
            "Parse & Import activation", key="act_import_btn",
        ):
            _parse_and_store_activation_upload(act_file, primary_zone_for_fetch)

    # ── Intraday (IDA) price upload ──────────────────────────────────────
    with st.sidebar.expander("Intraday (IDA) Prices"):
        st.caption(
            "Manual fallback for the IDA1 cockpit/uplift panels when ENTSO-E "
            "returns no intraday-auction data. Rows write to the same cache "
            "the live fetch uses."
        )
        ida_seq = st.selectbox(
            "Default IDA round (for rows without a sequence column)",
            options=[1, 2, 3],
            format_func=lambda s: f"IDA{s}",
            key="ida_default_seq",
        )
        ida_file = st.file_uploader("Upload IDA CSV", type=["csv"], key="ida_upload")
        if ida_file is not None and st.button("Parse & Import IDA", key="ida_import"):
            _parse_and_store_intraday_upload(
                ida_file, primary_zone_for_fetch, int(ida_seq),
            )
        manual_sources = {
            key: meta
            for key, meta in read_intraday_sources().items()
            if meta.get("source") == "Manual CSV"
        }
        if manual_sources:
            loaded = ", ".join(
                f"{zone} IDA{seq} ({meta['rows']})"
                for (zone, seq), meta in sorted(manual_sources.items())
            )
            st.caption(f"Manual IDA loaded: {loaded}")
        st.download_button(
            label="\U0001f4e5 Download IDA CSV template",
            data=generate_intraday_template_csv(),
            file_name="intraday_ida_template.csv",
            mime="text/csv",
            key="ida_tmpl_download",
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
        "primary_zone_for_fetch": primary_zone_for_fetch,
        "fetch_btn": fetch_btn,
        "current_zone_date_scope": current_zone_date_scope,
    }
