"""Regression tests for cockpit visual theme helpers."""

from __future__ import annotations

import plotly.io as pio

from src.pages.simulation_cockpit import _health_metric, _kpi_card
from src.ui_theme import cockpit_chart_template


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
