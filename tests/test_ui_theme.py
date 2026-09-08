"""Regression tests for cockpit visual theme helpers."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import src.ui_theme as ui_theme
from src.pages.simulation_cockpit import (
    _apply_panel_layout,
    _health_metric,
    _kpi_card,
    _plot_batch_summary,
    _plot_forecast_policy,
    _plot_rolling_summary,
)
from src.ui_theme import cockpit_chart_template

_EXPECTED_CADENCE_SPLIT_CAPTION = (
    "This window crosses a market resolution change. The continuous horizon "
    "is split at each cadence change: SoC carries between segments, but "
    "terminal-neutral equality is reapplied at each segment end."
)


def _cadence_panel_app() -> None:
    import pandas as pd

    from src.pages.simulation_cockpit import _render_multi_day_summary

    index = pd.date_range("2025-09-30", periods=48, freq="h", tz="UTC")
    prices = pd.DataFrame({"price_eur_mwh": 50.0}, index=index)
    _render_multi_day_summary(
        primary_df=prices,
        intraday_df=None,
        dates=sorted(set(index.date)),
        mode="DA MILP Replay",
        zone_tz="UTC",
        power_mw=1.0,
        duration_hours=4.0,
        efficiency=1.0,
        capture_rate=1.0,
        capex_eur_kwh=0.0,
        chart_template="plotly_dark",
    )


def _cadence_batch(splits: int) -> pd.DataFrame:
    batch = pd.DataFrame({
        "date": pd.date_range("2025-09-30", periods=2).date,
        "total_revenue_eur": [316.0, 0.0],
        "annualized_eur_per_mw": [115419.0, 0.0],
        "daily_fce": [1.0, 0.0],
    })
    batch.attrs.update({
        "n_cadence_splits": splits,
        "carry_mode": "continuous_horizon",
        "excluded_days": 0,
        "model_available": True,
    })
    return batch


def test_cadence_split_copy_is_verbatim() -> None:
    import src.pages.simulation_cockpit as cockpit

    assert getattr(cockpit, "_CADENCE_SPLIT_CAPTION", None) == _EXPECTED_CADENCE_SPLIT_CAPTION


def test_cadence_split_caption_is_rendered_after_batch_run(monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    import src.pages.simulation_cockpit as cockpit

    monkeypatch.setattr(cockpit, "simulate_replay_batch", lambda *a, **kw: _cadence_batch(1))
    app = AppTest.from_function(_cadence_panel_app).run(timeout=30)
    assert not app.exception
    assert _EXPECTED_CADENCE_SPLIT_CAPTION not in [caption.value for caption in app.caption]
    app.button(key="simulation_batch_run").click().run(timeout=30)
    assert not app.exception
    assert _EXPECTED_CADENCE_SPLIT_CAPTION in [caption.value for caption in app.caption]


def test_single_cadence_batch_has_no_split_caption(monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    import src.pages.simulation_cockpit as cockpit

    monkeypatch.setattr(cockpit, "simulate_replay_batch", lambda *a, **kw: _cadence_batch(0))
    app = AppTest.from_function(_cadence_panel_app).run(timeout=30)
    app.button(key="simulation_batch_run").click().run(timeout=30)
    assert not app.exception
    assert _EXPECTED_CADENCE_SPLIT_CAPTION not in [caption.value for caption in app.caption]


def test_cockpit_chart_template_registers_idempotently() -> None:
    """Repeated access should return the same registered template name."""
    name_1 = cockpit_chart_template()
    template_1 = pio.templates[name_1]

    name_2 = cockpit_chart_template()
    template_2 = pio.templates[name_2]

    assert name_1 == "bess_cockpit_dark"
    assert name_2 == name_1
    assert template_2 is template_1


def test_cockpit_chart_template_does_not_change_plotly_default() -> None:
    """Registering the cockpit template must not mutate Plotly's global default."""
    original_default = pio.templates.default

    cockpit_chart_template()

    assert pio.templates.default == original_default


def test_cockpit_charts_keep_legends_readable_on_dark_background(
    monkeypatch,
) -> None:
    figures = [go.Figure()]
    template = cockpit_chart_template()

    _apply_panel_layout(figures[0], "Title", "EUR", template)

    monkeypatch.setattr(
        st,
        "plotly_chart",
        lambda figure, **_kwargs: figures.append(figure),
    )
    dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
    _plot_forecast_policy(
        pd.DataFrame({
            "date": dates,
            "da_only_eur": [10.0, 20.0],
            "realised_eur": [12.0, 21.0],
            "ceiling_eur": [14.0, 23.0],
        }),
        template,
    )
    batch = pd.DataFrame({
        "date": dates,
        "total_revenue_eur": [100.0, 120.0],
        "daily_fce": [1.0, 1.2],
    })
    _plot_rolling_summary(batch, template)
    _plot_batch_summary(batch, template)

    assert len(figures) == 4
    assert all(
        figure.layout.legend.font.color == "#cfd8e6"
        for figure in figures
    )


def test_kpi_card_escapes_user_visible_strings() -> None:
    html = _kpi_card(
        "<Revenue>",
        "EUR <script>alert(1)</script>",
        "Use <b>safe</b> labels",
        "primary accent-magenta",
    )

    assert "<script>" not in html
    assert "&lt;Revenue&gt;" in html
    assert "EUR &lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Use &lt;b&gt;safe&lt;/b&gt; labels" in html


def test_health_metric_escapes_user_visible_strings() -> None:
    html = _health_metric("SoH <Delta>", "<0.01%")

    assert "&lt;Delta&gt;" in html
    assert "&lt;0.01%" in html


def _injected_theme_css(monkeypatch) -> str:
    import streamlit as st

    injected_css: list[str] = []
    monkeypatch.setattr(
        st, "markdown",
        lambda body, **_kwargs: injected_css.append(str(body)),
    )
    ui_theme.inject_global_cockpit_theme()
    return "\n".join(injected_css)


def test_global_theme_guards_expander_header_contrast(monkeypatch) -> None:
    css = _injected_theme_css(monkeypatch)

    assert '[data-testid="stExpander"] summary' in css
    assert '[data-testid="stExpander"] details[open] > summary' in css
    assert "-webkit-text-fill-color: #eaf3ff" in css


def test_global_theme_guards_number_input_contrast(monkeypatch) -> None:
    css = _injected_theme_css(monkeypatch)

    assert '[data-testid="stNumberInput"] div[data-baseweb="input"] > div' in css
    assert '[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within > div' in css
    assert '[data-testid="stNumberInput"] button' in css


def test_global_theme_guards_sidebar_disabled_button_contrast(monkeypatch) -> None:
    css = _injected_theme_css(monkeypatch)

    assert '[data-testid="stSidebar"] button:disabled' in css
    assert '[data-testid="stSidebar"] [data-testid^="stBaseButton"]:disabled' in css
    assert '[data-testid="stSidebar"] .stButton > button:disabled' in css
    assert '[data-testid="stSidebar"] .stDownloadButton > button:disabled' in css
    assert '[data-testid="stSidebar"] button:disabled *' in css
    assert "background-color: #172033" in css
    assert "-webkit-text-fill-color: #dbeafe" in css


def test_global_theme_styles_base_button_primary_as_brand_gradient(monkeypatch) -> None:
    css = _injected_theme_css(monkeypatch)

    assert '[data-testid="stBaseButton-primary"]' in css
    assert '[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]' in css
    assert "linear-gradient(135deg, rgba(255,45,149,0.96)" in css


def test_global_theme_guards_inline_code_contrast(monkeypatch) -> None:
    """Inline code must not inherit Streamlit's light default background.

    Browser-verified before the fix: computed style was color rgb(234,243,255)
    on background rgb(248,249,251) — roughly 1.05:1, i.e. invisible. The rule
    must own BOTH the background and the foreground; setting only one leaves the
    other free to come from whichever default wins.
    """
    css = _injected_theme_css(monkeypatch)

    assert ".stApp :not(pre) > code" in css
    assert "background: rgba(0,163,255,0.14) !important" in css
    assert "color: var(--bp-text) !important" in css
    assert "-webkit-text-fill-color: var(--bp-text) !important" in css
