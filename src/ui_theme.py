"""Shared visual styling for the cockpit-oriented Streamlit UI."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

_COCKPIT_TEMPLATE_NAME = "bess_cockpit_dark"


def cockpit_chart_template() -> str:
    """Return the chart template used by the cockpit visual system."""
    _register_cockpit_plotly_template()
    return _COCKPIT_TEMPLATE_NAME


def apply_cockpit_plot_theme(fig: go.Figure) -> go.Figure:
    """Force a figure into the cockpit visual style.

    Plotly templates are not always enough once figures add rangesliders,
    coloraxes, or page-specific layout overrides. This helper is intentionally
    explicit so every core chart keeps the same dark operations-room feel.
    """
    _register_cockpit_plotly_template()
    fig.update_layout(
        template=_COCKPIT_TEMPLATE_NAME,
        paper_bgcolor="#07090d",
        plot_bgcolor="#0b1118",
        font=dict(color="#dce8f7", family="Aptos, Segoe UI, sans-serif", size=12),
        title=dict(font=dict(color="#edf4ff", size=16)),
        hovermode="x unified",
        margin=dict(l=54, r=34, t=60, b=46),
        legend=dict(
            font=dict(color="#cfd8e6"),
            bgcolor="rgba(7,9,13,0.62)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
        ),
        coloraxis=dict(
            colorbar=dict(
                tickfont=dict(color="#a8b3c5"),
                title_font=dict(color="#cfd8e6"),
                bgcolor="rgba(7,9,13,0.55)",
                bordercolor="rgba(255,255,255,0.10)",
                borderwidth=1,
            ),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.075)",
        zerolinecolor="rgba(255,255,255,0.18)",
        linecolor="rgba(255,255,255,0.18)",
        tickfont=dict(color="#a8b3c5"),
        title_font=dict(color="#a8b3c5"),
        rangeslider=dict(bgcolor="#07090d", bordercolor="rgba(255,255,255,0.22)"),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.075)",
        zerolinecolor="rgba(255,255,255,0.18)",
        linecolor="rgba(255,255,255,0.18)",
        tickfont=dict(color="#a8b3c5"),
        title_font=dict(color="#a8b3c5"),
    )
    return fig


def _register_cockpit_plotly_template() -> None:
    """Register a dark Plotly template shared by all dashboard tabs."""
    if _COCKPIT_TEMPLATE_NAME in pio.templates:
        return
    pio.templates[_COCKPIT_TEMPLATE_NAME] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="#07090d",
            plot_bgcolor="#0b1118",
            font=dict(color="#dce8f7", family="Aptos, Segoe UI, sans-serif", size=12),
            title=dict(font=dict(color="#edf4ff", size=16)),
            colorway=[
                "#ff2d95", "#00a3ff", "#ffc233", "#7fb6ff",
                "#34d399", "#d0d4dc", "#f97316", "#a78bfa",
            ],
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.08)",
                zerolinecolor="rgba(255,255,255,0.18)",
                linecolor="rgba(255,255,255,0.16)",
                tickfont=dict(color="#a8b3c5"),
                title_font=dict(color="#a8b3c5"),
                rangeslider=dict(bgcolor="#07090d", bordercolor="rgba(255,255,255,0.20)"),
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.08)",
                zerolinecolor="rgba(255,255,255,0.18)",
                linecolor="rgba(255,255,255,0.16)",
                tickfont=dict(color="#a8b3c5"),
                title_font=dict(color="#a8b3c5"),
            ),
            legend=dict(font=dict(color="#cfd8e6")),
            margin=dict(l=50, r=30, t=58, b=42),
            hoverlabel=dict(
                bgcolor="#111925",
                bordercolor="rgba(255,255,255,0.16)",
                font=dict(color="#edf4ff"),
            ),
        ),
    )


def inject_global_cockpit_theme() -> None:
    """Inject global CSS to align all tabs with the cockpit visual language."""
    _register_cockpit_plotly_template()
    st.markdown(
        """
        <style>
        :root {
            --bp-bg: #05070b;
            --bp-panel: #0b1118;
            --bp-panel-2: #111925;
            --bp-border: rgba(255,255,255,0.10);
            --bp-text: #e8eef8;
            --bp-muted: #9aa7b8;
            --bp-cyan: #00a3ff;
            --bp-magenta: #ff2d95;
            --bp-magenta-2: #b84dff;
            --bp-warn: #ffc233;
        }

        .stApp {
            color: var(--bp-text);
            background:
                radial-gradient(circle at 12% 0%, rgba(255,45,149,0.12), transparent 28%),
                radial-gradient(circle at 92% 8%, rgba(0,163,255,0.12), transparent 24%),
                linear-gradient(180deg, #070a10 0%, #05070b 45%, #030407 100%);
        }

        [data-testid="stHeader"] {
            background: rgba(5,7,11,0.72);
            backdrop-filter: blur(12px);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(115deg, rgba(0,196,255,0.13) 0%, transparent 34%),
                radial-gradient(circle at 14% 8%, rgba(0,205,255,0.34), transparent 34%),
                radial-gradient(circle at 86% 20%, rgba(53,123,255,0.18), transparent 30%),
                linear-gradient(180deg, #103a61 0%, #102947 34%, #0b1830 68%, #061022 100%);
            border-right: 1px solid rgba(0,205,255,0.24);
            box-shadow:
                inset -1px 0 0 rgba(255,255,255,0.08),
                14px 0 42px rgba(0,10,28,0.46);
        }

        [data-testid="stSidebar"] * {
            color: var(--bp-text);
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--bp-text) !important;
        }

        .block-container {
            padding-top: 3.2rem;
        }

        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: var(--bp-text);
        }

        [data-testid="stCaptionContainer"],
        .stCaption,
        small {
            color: var(--bp-muted) !important;
        }

        button[data-baseweb="tab"] {
            color: var(--bp-muted);
            background: transparent;
            border-radius: 10px 10px 0 0;
            min-height: 2.45rem;
            padding: 0.55rem 0.72rem;
            line-height: 1.1;
        }

        div[data-testid="stTabs"] [role="tablist"] {
            padding-top: 0.35rem;
            overflow-x: auto;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #ffffff;
            background: linear-gradient(180deg, rgba(255,45,149,0.18), rgba(255,45,149,0.04));
            border-bottom: 2px solid var(--bp-magenta);
        }

        [data-testid="stMetric"],
        [data-testid="stExpander"],
        [data-testid="stDataFrame"],
        [data-testid="stAlert"] {
            border-radius: 14px;
            border: 1px solid var(--bp-border);
            background: linear-gradient(180deg, rgba(17,25,37,0.94), rgba(8,12,18,0.94));
            box-shadow: 0 12px 32px rgba(0,0,0,0.24);
        }

        [data-testid="stMetric"] {
            padding: 0.75rem 0.9rem;
        }

        [data-testid="stMetricLabel"] p {
            color: var(--bp-muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.055em;
            font-size: 0.72rem;
        }

        [data-testid="stMetricValue"] {
            color: #ffffff;
            font-weight: 800;
            letter-spacing: -0.035em;
        }

        [data-testid="stPlotlyChart"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
            background: #07090d;
            box-shadow: 0 16px 38px rgba(0,0,0,0.26);
            margin-bottom: 0.9rem;
        }

        [data-testid="stPlotlyChart"] > div {
            background: #07090d !important;
        }

        .stSelectbox > div,
        .stMultiSelect > div,
        .stNumberInput > div,
        .stDateInput > div,
        .stTextInput > div,
        .stSlider > div {
            color: var(--bp-text);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        input,
        textarea {
            color: var(--bp-text) !important;
            background-color: rgba(255,255,255,0.06) !important;
            border-color: rgba(255,255,255,0.12) !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] div[data-baseweb="input"] > div,
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: #eaf3ff !important;
            -webkit-text-fill-color: #eaf3ff !important;
            background:
                linear-gradient(180deg, rgba(32,74,108,0.92), rgba(10,31,54,0.92)) !important;
            border-color: rgba(0,205,255,0.38) !important;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.08),
                0 0 0 1px rgba(0,205,255,0.06),
                0 10px 26px rgba(0,0,0,0.18) !important;
            caret-color: var(--bp-cyan) !important;
        }

        [data-testid="stSidebar"] input:disabled,
        [data-testid="stSidebar"] input[disabled] {
            color: rgba(234,243,255,0.82) !important;
            -webkit-text-fill-color: rgba(234,243,255,0.82) !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] input::placeholder {
            color: rgba(188,204,224,0.62) !important;
            -webkit-text-fill-color: rgba(188,204,224,0.62) !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] {
            border-radius: 12px;
            box-shadow: 0 0 0 1px rgba(0,205,255,0.12),
                        0 12px 26px rgba(0,0,0,0.20);
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] span {
            color: #eaf3ff !important;
        }

        div[data-baseweb="tag"] {
            color: #ffffff !important;
            background:
                linear-gradient(135deg, var(--bp-magenta), var(--bp-magenta-2)) !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            box-shadow: 0 8px 22px rgba(255,45,149,0.22) !important;
        }

        div[data-baseweb="tag"] *,
        div[data-baseweb="tag"] svg,
        div[data-baseweb="tag"] span {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] [role="button"] {
            color: #eaf3ff !important;
            -webkit-text-fill-color: #eaf3ff !important;
        }

        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .stDownloadButton > button,
        [data-testid="stSidebar"] .stFormSubmitButton > button {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background:
                linear-gradient(135deg, rgba(255,45,149,0.96), rgba(184,77,255,0.82)) !important;
            border: 1px solid rgba(255,255,255,0.24) !important;
            box-shadow: 0 10px 24px rgba(255,45,149,0.22) !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover,
        [data-testid="stSidebar"] .stDownloadButton > button:hover,
        [data-testid="stSidebar"] .stFormSubmitButton > button:hover {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border-color: rgba(255,255,255,0.42) !important;
            filter: brightness(1.08);
        }

        [data-testid="stSidebar"] .stButton > button:disabled,
        [data-testid="stSidebar"] .stDownloadButton > button:disabled,
        [data-testid="stSidebar"] .stFormSubmitButton > button:disabled,
        [data-testid="stSidebar"] .stButton > button[disabled],
        [data-testid="stSidebar"] .stDownloadButton > button[disabled],
        [data-testid="stSidebar"] .stFormSubmitButton > button[disabled] {
            color: rgba(234,243,255,0.88) !important;
            -webkit-text-fill-color: rgba(234,243,255,0.88) !important;
            background:
                linear-gradient(135deg, rgba(35,51,72,0.98), rgba(28,38,58,0.98)) !important;
            border-color: rgba(255,255,255,0.26) !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
            color: #eaf3ff !important;
            background: linear-gradient(180deg, rgba(20,59,92,0.95), rgba(8,25,45,0.95)) !important;
            border-color: rgba(0,205,255,0.26) !important;
        }

        [data-testid="stSidebar"] [data-testid="stCheckbox"] label span {
            color: #eaf3ff !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] {
            overflow: hidden !important;
            border: 1px solid rgba(0,205,255,0.22) !important;
            border-radius: 16px !important;
            background:
                linear-gradient(180deg, rgba(14,30,47,0.92), rgba(7,14,25,0.94)) !important;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.08),
                0 12px 30px rgba(0,0,0,0.26) !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] details,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary *,
        [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"],
        [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
            color: #eaf3ff !important;
            -webkit-text-fill-color: #eaf3ff !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            background:
                linear-gradient(180deg, rgba(22,50,76,0.98), rgba(8,18,32,0.98)) !important;
            border-bottom: 1px solid rgba(0,205,255,0.14) !important;
            min-height: 3.15rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
            background:
                linear-gradient(180deg, rgba(31,70,104,0.98), rgba(10,25,45,0.98)) !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] svg {
            color: #eaf3ff !important;
            fill: #eaf3ff !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"],
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            color: #eaf3ff !important;
            -webkit-text-fill-color: #eaf3ff !important;
            background:
                linear-gradient(180deg, rgba(19,57,88,0.92), rgba(8,25,45,0.96)) !important;
            border-color: rgba(0,205,255,0.34) !important;
            border-radius: 14px !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] *,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
            color: #eaf3ff !important;
            -webkit-text-fill-color: #eaf3ff !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] button,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background:
                linear-gradient(135deg, var(--bp-magenta), var(--bp-magenta-2)) !important;
            border: 1px solid rgba(255,255,255,0.24) !important;
            box-shadow: 0 10px 24px rgba(255,45,149,0.22) !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="tooltip"],
        div[data-baseweb="tooltip"] > div,
        div[data-baseweb="popover"] > div,
        div[data-baseweb="popover"] [role="dialog"],
        div[data-baseweb="popover"] [data-baseweb="calendar"],
        div[data-baseweb="popover"] [data-baseweb="menu"],
        div[data-baseweb="popover"] [role="listbox"],
        div[data-baseweb="menu"],
        ul[role="listbox"],
        div[role="listbox"] {
            color: #eaf3ff !important;
            background:
                linear-gradient(180deg, rgba(17,50,80,0.98), rgba(8,22,42,0.98)) !important;
            border-color: rgba(0,205,255,0.30) !important;
            box-shadow: 0 18px 52px rgba(0,0,0,0.46) !important;
        }

        div[data-baseweb="popover"] *,
        div[data-baseweb="tooltip"] *,
        div[data-baseweb="calendar"] *,
        div[data-baseweb="menu"] *,
        ul[role="listbox"] *,
        div[role="listbox"] *,
        [role="listbox"] *,
        [role="option"] * {
            color: #eaf3ff !important;
        }

        div[data-baseweb="menu"] li,
        div[data-baseweb="menu"] li > div,
        ul[role="listbox"] li,
        ul[role="listbox"] li > div,
        div[role="listbox"] [role="option"],
        [role="option"] {
            color: #eaf3ff !important;
            -webkit-text-fill-color: #eaf3ff !important;
            background: rgba(8,22,42,0.98) !important;
        }

        div[data-baseweb="menu"] li:hover,
        div[data-baseweb="menu"] li:hover > div,
        ul[role="listbox"] li:hover,
        ul[role="listbox"] li:hover > div,
        [role="option"]:hover,
        [aria-selected="true"][role="option"] {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background: rgba(0,205,255,0.22) !important;
        }

        div[data-baseweb="calendar"] button {
            background: transparent !important;
            border-radius: 999px !important;
        }

        div[data-baseweb="calendar"] button[aria-selected="true"],
        div[data-baseweb="calendar"] [aria-selected="true"] {
            color: #ffffff !important;
            background: linear-gradient(135deg, var(--bp-magenta), var(--bp-magenta-2)) !important;
        }

        div[data-baseweb="calendar"] {
            overflow: hidden !important;
            color: #eaf3ff !important;
            background:
                linear-gradient(180deg, rgba(17,50,80,0.98), rgba(7,19,38,0.99)) !important;
            border: 1px solid rgba(0,205,255,0.30) !important;
            border-radius: 14px !important;
        }

        div[data-baseweb="calendar"] > div,
        div[data-baseweb="calendar"] > div > div,
        div[data-baseweb="calendar"] [role="row"],
        div[data-baseweb="calendar"] [role="grid"],
        div[data-baseweb="calendar"] [role="gridcell"],
        div[data-baseweb="calendar"] [role="columnheader"] {
            color: #eaf3ff !important;
            -webkit-text-fill-color: #eaf3ff !important;
            background: transparent !important;
        }

        div[data-baseweb="calendar"] [role="grid"] {
            background: #08213a !important;
        }

        div[data-baseweb="calendar"] [role="columnheader"],
        div[data-baseweb="calendar"] [aria-label*="Sunday"],
        div[data-baseweb="calendar"] [aria-label*="Monday"],
        div[data-baseweb="calendar"] [aria-label*="Tuesday"],
        div[data-baseweb="calendar"] [aria-label*="Wednesday"],
        div[data-baseweb="calendar"] [aria-label*="Thursday"],
        div[data-baseweb="calendar"] [aria-label*="Friday"],
        div[data-baseweb="calendar"] [aria-label*="Saturday"] {
            color: rgba(234,243,255,0.74) !important;
            -webkit-text-fill-color: rgba(234,243,255,0.74) !important;
        }

        div[data-baseweb="calendar"] button,
        div[data-baseweb="calendar"] [role="gridcell"] button {
            color: #eaf3ff !important;
            -webkit-text-fill-color: #eaf3ff !important;
            background: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 999px !important;
            box-shadow: none !important;
        }

        div[data-baseweb="calendar"] [role="gridcell"] > div {
            background: transparent !important;
        }

        div[data-baseweb="calendar"] [role="gridcell"] > div:has(button[aria-selected="true"]),
        div[data-baseweb="calendar"] [role="gridcell"] > div:has([aria-selected="true"]) {
            background: rgba(255,45,149,0.16) !important;
            border-radius: 999px !important;
        }

        div[data-baseweb="calendar"] button:hover,
        div[data-baseweb="calendar"] [role="gridcell"]:hover button {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background: rgba(0,205,255,0.18) !important;
            border-color: rgba(0,205,255,0.36) !important;
        }

        div[data-baseweb="calendar"] button[aria-selected="true"],
        div[data-baseweb="calendar"] [role="gridcell"] [aria-selected="true"] {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background: linear-gradient(135deg, var(--bp-magenta), var(--bp-magenta-2)) !important;
            border-color: rgba(255,255,255,0.26) !important;
            box-shadow: 0 0 0 3px rgba(255,45,149,0.18),
                        0 12px 24px rgba(255,45,149,0.26) !important;
        }

        div[data-baseweb="calendar"] button:disabled,
        div[data-baseweb="calendar"] [aria-disabled="true"] {
            color: rgba(234,243,255,0.30) !important;
            -webkit-text-fill-color: rgba(234,243,255,0.30) !important;
            background: transparent !important;
        }

        /*
         * BaseWeb datepicker nests its header and out-of-month fillers in anonymous
         * divs, so role-based selectors miss the pale default surfaces. Keep this
         * override scoped to the calendar popover and reapply the intentional
         * states below it.
         */
        div[data-baseweb="calendar"] div:not([role="grid"]):not([aria-selected="true"]),
        div[data-baseweb="calendar"] span:not([aria-selected="true"]) {
            background-color: transparent !important;
            background-image: none !important;
            color: #eaf3ff !important;
            -webkit-text-fill-color: #eaf3ff !important;
        }

        div[data-baseweb="calendar"] [role="presentation"],
        div[data-baseweb="calendar"] [role="presentation"] > div,
        div[data-baseweb="calendar"] [role="row"],
        div[data-baseweb="calendar"] [role="gridcell"] {
            background-color: transparent !important;
            background-image: none !important;
        }

        div[data-baseweb="calendar"] [role="gridcell"]::after {
            background-color: transparent !important;
            background-image: none !important;
        }

        div[data-baseweb="calendar"] [role="grid"] {
            background:
                linear-gradient(180deg, rgba(9,35,61,0.98), rgba(7,25,47,0.99)) !important;
        }

        div[data-baseweb="calendar"] button,
        div[data-baseweb="calendar"] button *,
        div[data-baseweb="calendar"] [role="gridcell"],
        div[data-baseweb="calendar"] [role="gridcell"] * {
            color: #eaf3ff !important;
            -webkit-text-fill-color: #eaf3ff !important;
        }

        div[data-baseweb="calendar"] [role="gridcell"][aria-label^="Selected"],
        div[data-baseweb="calendar"] button[aria-selected="true"],
        div[data-baseweb="calendar"] [aria-selected="true"] {
            position: relative !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background-color: var(--bp-magenta) !important;
            background-image:
                linear-gradient(135deg, var(--bp-magenta), var(--bp-magenta-2)) !important;
            border-radius: 999px !important;
            box-shadow: 0 0 0 3px rgba(255,45,149,0.18),
                        0 12px 24px rgba(255,45,149,0.26) !important;
        }

        div[data-baseweb="calendar"] [role="gridcell"][aria-label^="Selected"] {
            isolation: isolate !important;
        }

        div[data-baseweb="calendar"] [role="gridcell"][aria-label^="Selected"]::before {
            content: "" !important;
            position: absolute !important;
            inset: 3px !important;
            z-index: 0 !important;
            border-radius: 999px !important;
            background:
                linear-gradient(135deg, var(--bp-magenta), var(--bp-magenta-2)) !important;
            box-shadow: 0 0 0 3px rgba(255,45,149,0.18),
                        0 12px 24px rgba(255,45,149,0.26) !important;
        }

        div[data-baseweb="calendar"] [role="gridcell"][aria-label^="Selected"] * {
            position: relative !important;
            z-index: 1 !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        div[data-baseweb="select"] svg {
            color: var(--bp-muted);
        }

        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {
            color: #ffffff;
            border: 1px solid rgba(255,255,255,0.14);
            background: linear-gradient(135deg, rgba(255,45,149,0.96), rgba(184,77,255,0.78));
            border-radius: 12px;
            font-weight: 750;
            box-shadow: 0 10px 26px rgba(0,0,0,0.24);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {
            border-color: rgba(255,255,255,0.38);
            filter: brightness(1.08);
        }

        hr {
            border-color: rgba(255,255,255,0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
