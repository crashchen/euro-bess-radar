"""SD-B contract pins for Revenue-tab merchant-revenue decay controls."""

from __future__ import annotations

import base64
import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import src.pages.revenue_estimation as revenue_page
from src.pages.revenue_estimation import (
    _REVENUE_DECAY_HARD_CAPTION,
    _build_npv_tornado_frame,
    _normalise_revenue_decay_inputs,
    _terminal_decay_year_and_weight,
)


def _risk_app() -> None:
    import pandas as pd

    from src.pages.revenue_estimation import _render_revenue_risk_analysis

    daily = pd.DataFrame({
        "spread": [20.0, 60.0, 100.0, 40.0],
        "n_cycles": [1.0, 1.0, 1.0, 1.0],
    })
    _render_revenue_risk_analysis(
        daily_spreads=daily,
        revenue={"cycles_per_day_assumption": 1.0},
        capture_rate=0.8,
        use_lp_dispatch=False,
        power_mw=1.0,
        duration_hours=2.0,
        efficiency=0.9,
        capex_eur_kwh=100.0,
        chart_template="plotly_dark",
    )


def _metric_values(app: AppTest, labels: tuple[str, ...]) -> dict[str, str]:
    return {
        metric.label: metric.value
        for metric in app.metric
        if metric.label in labels
    }


def _find_elements(node, type_name: str) -> list:
    found: list = []
    children = getattr(node, "children", None)
    if isinstance(children, dict):
        for child in children.values():
            if getattr(child, "type", "") == type_name:
                found.append(child)
            found.extend(_find_elements(child, type_name))
    return found


def _chart_by_title(app: AppTest, title: str) -> dict:
    charts = _find_elements(app.main, "plotly_chart")
    specs = [json.loads(chart.proto.spec) for chart in charts]
    return next(spec for spec in specs if spec["layout"]["title"]["text"] == title)


def _plotly_values(values: list | dict) -> list[float]:
    if isinstance(values, list):
        return values
    raw = base64.b64decode(values["bdata"])
    return np.frombuffer(raw, dtype=np.dtype(values["dtype"])).tolist()


def test_decay_contract_locked_copy_is_verbatim() -> None:
    assert _REVENUE_DECAY_HARD_CAPTION == (
        "Revenue-trajectory decay: screening assumption on annual merchant "
        "cash flows; does not simulate future hourly prices or re-dispatch. "
        "Battery degradation cost stays flat, so late decayed years can show "
        "negative operating margins rather than an idled asset. Decay begins "
        "after year 1 (the loaded sample's year). User assertion — no build-out "
        "data is fetched."
    )


@pytest.mark.parametrize("missing", [(None, 0.0), (0.0, None)])
def test_decay_percent_boundary_guards_cleared_inputs(
    missing: tuple[float | None, float | None],
) -> None:
    assert _normalise_revenue_decay_inputs(*missing) is None


def test_decay_percent_boundary_and_activity_predicate() -> None:
    assert _normalise_revenue_decay_inputs(10.0, 20.0) == (0.1, 0.2, True)
    assert _normalise_revenue_decay_inputs(10.0, 100.0) == (0.1, 1.0, False)


def test_fractional_terminal_year_uses_residual_year_weight() -> None:
    year, weight = _terminal_decay_year_and_weight(2.5, 0.1, 0.0)
    assert year == 3
    assert weight == pytest.approx(0.81)


def test_tornado_directions_follow_resulting_npv_for_every_axis() -> None:
    sensitivity = pd.DataFrame({
        "param": [
            "revenue", "revenue", "revenue",
            "capex", "capex", "capex",
            "discount_rate", "discount_rate", "discount_rate",
        ],
        "value": [0.7, 1.0, 1.3, 0.8, 1.0, 1.2, 0.06, 0.08, 0.10],
        "npv": [70.0, 100.0, 130.0, 120.0, 100.0, 80.0, 115.0, 100.0, 85.0],
    })
    result = _build_npv_tornado_frame(sensitivity, base_npv=100.0)
    assert (result["downside_delta"] <= result["upside_delta"]).all()
    capex = result.set_index("param").loc["capex"]
    assert capex["downside_delta"] == -20.0
    assert capex["upside_delta"] == 20.0


class TestRevenueDecayAppTest:
    @pytest.fixture()
    def app(self) -> AppTest:
        return AppTest.from_function(_risk_app).run(timeout=30)

    def test_default_has_legacy_npv_and_no_decay_surface(self, app: AppTest) -> None:
        assert not app.exception
        assert _REVENUE_DECAY_HARD_CAPTION not in [c.value for c in app.caption]
        tornado = _chart_by_title(app, "NPV Sensitivity (vs base case)")
        assert "decay" not in tornado["data"][0]["y"]
        assert len(_metric_values(app, ("NPV P10", "NPV P50", "NPV P90"))) == 3

    def test_active_decay_moves_npv_but_not_bootstrap_and_adds_axis(
        self, app: AppTest,
    ) -> None:
        revenue_labels = ("P10 Revenue", "P50 Revenue", "P90 Revenue")
        npv_labels = ("NPV P10", "NPV P50", "NPV P90")
        baseline_revenue = _metric_values(app, revenue_labels)
        baseline_npv = _metric_values(app, npv_labels)

        app.number_input(key="merchant_revenue_decay_pct").set_value(10.0).run(
            timeout=30
        )
        assert not app.exception
        assert _metric_values(app, revenue_labels) == baseline_revenue
        assert _metric_values(app, npv_labels) != baseline_npv
        assert _REVENUE_DECAY_HARD_CAPTION in [c.value for c in app.caption]
        tornado = _chart_by_title(app, "NPV Sensitivity (vs base case)")
        assert "decay" in tornado["data"][0]["y"]

    def test_floor_at_one_is_inactive_and_bit_identical(self, app: AppTest) -> None:
        npv_labels = ("NPV P10", "NPV P50", "NPV P90", "P(NPV>0)")
        baseline_npv = _metric_values(app, npv_labels)
        app.number_input(key="merchant_revenue_decay_floor_pct").set_value(
            100.0
        ).run(timeout=30)
        app.number_input(key="merchant_revenue_decay_pct").set_value(10.0).run(
            timeout=30
        )
        assert not app.exception
        assert _metric_values(app, npv_labels) == baseline_npv
        assert _REVENUE_DECAY_HARD_CAPTION not in [c.value for c in app.caption]

    def test_fractional_life_caption_uses_year_three(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            revenue_page,
            "estimate_battery_lifetime",
            lambda **_: {
                "effective_life_years": 2.5,
                "cycle_limited_years": 2.5,
                "calendar_life_years": 20.0,
                "limiting_factor": "cycling",
            },
        )
        app = AppTest.from_function(_risk_app).run(timeout=30)
        app.number_input(key="merchant_revenue_decay_pct").set_value(10.0).run(
            timeout=30
        )
        assert not app.exception
        assert any(
            "year **3** earns **81.0%**" in caption.value
            for caption in app.caption
        )

    def test_percent_is_divided_before_npv_call(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observed: list[float] = []
        original = revenue_page.calculate_npv_distribution

        def spy(*args, **kwargs):
            observed.append(kwargs["annual_decay_rate"])
            return original(*args, **kwargs)

        monkeypatch.setattr(revenue_page, "calculate_npv_distribution", spy)
        app = AppTest.from_function(_risk_app).run(timeout=30)
        app.number_input(key="merchant_revenue_decay_pct").set_value(10.0).run(
            timeout=30
        )
        assert not app.exception
        assert observed[-1] == pytest.approx(0.10)

    def test_tornado_chart_sorts_capex_by_npv_not_input(self, app: AppTest) -> None:
        tornado = _chart_by_title(app, "NPV Sensitivity (vs base case)")
        downside, upside = tornado["data"]
        capex_position = list(downside["y"]).index("capex")
        downside_x = _plotly_values(downside["x"])
        upside_x = _plotly_values(upside["x"])
        assert downside_x[capex_position] <= 0.0
        assert downside_x[capex_position] <= upside_x[capex_position]

    @pytest.mark.parametrize(
        "key",
        ["merchant_revenue_decay_pct", "merchant_revenue_decay_floor_pct"],
    )
    def test_cleared_input_shows_prompt_instead_of_crashing(self, key: str) -> None:
        app = AppTest.from_function(_risk_app)
        app.session_state[key] = None
        app.run(timeout=30)
        assert not app.exception
        assert any("Enter both" in info.value for info in app.info)
        assert "NPV P50" not in [metric.label for metric in app.metric]
