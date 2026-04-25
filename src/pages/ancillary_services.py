"""Tab 6: Ancillary Services — auto-fetch, manual upload, data preview."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ancillary import ANCILLARY_TEMPLATES
from src.ancillary_fetchers import get_available_fetchers
from src.components.sidebar import _parse_and_store_ancillary_upload, _run_and_store_ancillary_fetch


def render(
    primary_zone: str,
    start_date,
    end_date,
    anc_df: pd.DataFrame | None,
) -> None:
    """Render the Ancillary Services tab."""
    st.subheader(f"Ancillary Services — {primary_zone}")

    anc_col1, anc_col2 = st.columns(2)
    with anc_col1:
        st.markdown("**Auto-Fetch**")
        tab_fetchers = get_available_fetchers(primary_zone)
        if tab_fetchers:
            for f in tab_fetchers:
                st.caption(f"\U0001f4e1 {f['name']} ({f['source']})")
            if st.button(
                f"\u26a1 Fetch ancillary for {primary_zone}",
                key="tab_auto_fetch",
            ):
                _run_and_store_ancillary_fetch(primary_zone, start_date, end_date)
        else:
            st.info("No auto-fetch available for this zone.")

    with anc_col2:
        st.markdown("**Manual CSV Upload**")
        tab_template = st.selectbox(
            "Template", list(ANCILLARY_TEMPLATES.keys()),
            format_func=lambda k: f"{k} — {ANCILLARY_TEMPLATES[k]['description'][:40]}",
            key="tab_anc_template",
        )
        tab_file = st.file_uploader("Upload CSV", type=["csv"], key="tab_anc_upload")
        if tab_file is not None and st.button("Parse & Import", key="tab_anc_parse"):
            _parse_and_store_ancillary_upload(
                tab_file,
                tab_template,
                primary_zone,
                start_date,
                end_date,
            )

    # Show loaded ancillary data summary
    st.divider()
    if anc_df is not None and not anc_df.empty:
        st.markdown(f"**Loaded ancillary data:** {len(anc_df)} rows")
        if "product_type" in anc_df.columns:
            products = anc_df["product_type"].unique()
            st.caption(f"Products: {', '.join(str(p) for p in products)}")
        with st.expander("Preview ancillary data"):
            st.dataframe(anc_df.head(50), hide_index=True)
    else:
        st.info(
            "No ancillary data loaded. Use auto-fetch or manual CSV upload above."
        )
