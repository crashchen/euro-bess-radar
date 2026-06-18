"""Tab 8: Data Trust — source traceability and quality diagnostics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_trust import (
    build_intraday_source_table,
    build_zone_data_quality_table,
)


def render(
    *,
    zone_data: dict[str, pd.DataFrame],
    zone_timezones: dict[str, str] | None = None,
) -> None:
    """Render source and data-quality diagnostics for fetched zones."""
    st.subheader("Data Trust — Source Traceability")
    st.caption(
        "Use this as the audit label for the current run: source, timezone, "
        "covered interval count, unresolved gaps, and short-gap imputations. "
        "Revenue and risk charts should be read more cautiously when coverage "
        "is low or source gaps are concentrated."
    )

    quality = build_zone_data_quality_table(
        zone_data,
        zone_timezones=zone_timezones,
    )
    if quality.empty:
        st.info("Fetch at least one bidding zone to see data trust diagnostics.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fetched Zones", f"{len(quality):,}")
    c2.metric("Avg Coverage", f"{quality['coverage_pct'].mean():.1f}%")
    c3.metric("Source Gap Intervals", f"{int(quality['source_gap_intervals'].sum()):,}")
    c4.metric("Unresolved Missing", f"{int(quality['missing_intervals'].sum()):,}")

    if int(quality["missing_intervals"].sum()) > 0:
        st.warning(
            "One or more zones have unresolved missing intervals. Complete-day "
            "filters exclude affected days from spread/dispatch analytics, but "
            "the raw price chart still shows the underlying gaps."
        )
    elif int(quality["imputed_intervals"].sum()) > 0:
        st.info(
            "Short internal gaps were imputed to keep charts continuous. "
            "The imputed share is shown below and exported as data-quality metadata."
        )

    st.dataframe(
        quality,
        width="stretch",
        hide_index=True,
        column_config={
            "zone": "Zone",
            "source": "Source",
            "timezone": "Timezone",
            "first_timestamp_utc": st.column_config.DatetimeColumn(
                "First UTC", format="YYYY-MM-DD HH:mm",
            ),
            "last_timestamp_utc": st.column_config.DatetimeColumn(
                "Last UTC", format="YYYY-MM-DD HH:mm",
            ),
            "coverage_pct": st.column_config.NumberColumn(
                "Coverage", format="%.2f%%",
            ),
            "source_gap_pct": st.column_config.NumberColumn(
                "Source Gap %", format="%.2f%%",
            ),
            "imputed_pct": st.column_config.NumberColumn(
                "Imputed %", format="%.2f%%",
            ),
            "missing_pct": st.column_config.NumberColumn(
                "Missing %", format="%.2f%%",
            ),
            "max_source_gap_hours": st.column_config.NumberColumn(
                "Max Gap (h)", format="%.2f",
            ),
        },
    )

    intraday_sources = build_intraday_source_table()
    if not intraday_sources.empty:
        st.markdown("**Intraday (IDA) price sources**")
        if (intraday_sources["source"] == "Manual CSV").any():
            st.caption(
                "Provenance of cached intraday (IDA) prices. Rows marked "
                "'Manual CSV' were imported from an upload, not the ENTSO-E "
                "intraday-auction API — the cockpit/uplift numbers that use them "
                "are only as trustworthy as the uploaded file."
            )
        else:
            st.caption("Provenance of cached intraday (IDA) prices.")
        st.dataframe(
            intraday_sources,
            width="stretch",
            hide_index=True,
            column_config={
                "zone": "Zone",
                "sequence": "IDA Round",
                "source": "Source",
                "rows": st.column_config.NumberColumn("Rows", format="%d"),
                "first_timestamp_utc": st.column_config.DatetimeColumn(
                    "First UTC", format="YYYY-MM-DD HH:mm",
                ),
                "last_timestamp_utc": st.column_config.DatetimeColumn(
                    "Last UTC", format="YYYY-MM-DD HH:mm",
                ),
                "imported_at": st.column_config.DatetimeColumn(
                    "Imported (UTC)", format="YYYY-MM-DD HH:mm",
                ),
            },
        )

    st.caption(
        "Licensing note: this repository licenses code under Apache-2.0 only. "
        "Fetched, cached, uploaded, or exported market data remains subject to "
        "the original source terms."
    )
