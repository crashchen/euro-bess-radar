"""Tests for the cockpit cycle-cap frontier panel (F-C).

Covers the pure helpers (cap parsing, default options, export assumptions),
the render() wiring order pin (contract section 4: the expander sits
immediately after the multi-day replay section), and an AppTest headless
smoke that drives the REAL panel — the app-layer surface mocked unit tests
cannot reach.
"""

from __future__ import annotations

import ast
import inspect
from typing import ClassVar

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import src.pages.simulation_cockpit as cockpit
from src.cycle_frontier import DEFAULT_CYCLE_CAPS, UNCAPPED_LABEL
from src.export import cockpit_tables_to_excel
from src.pages.simulation_cockpit import (
    _append_frontier_assumptions,
    _frontier_cap_options,
    _parse_frontier_caps,
)


class TestFrontierCapParsing:
    def test_default_options_mirror_contract_cap_set(self) -> None:
        assert _frontier_cap_options() == [
            "0.5", "1", "1.2", "1.5", "2", "3", UNCAPPED_LABEL,
        ]
        assert len(_frontier_cap_options()) == len(DEFAULT_CYCLE_CAPS)

    def test_parses_numbers_and_uncapped_label(self) -> None:
        caps, invalid = _parse_frontier_caps(["0.5", "1", "2.5", "uncapped"])
        assert caps == [0.5, 1.0, 2.5, None]
        assert invalid == []

    def test_uncapped_label_is_case_insensitive(self) -> None:
        caps, invalid = _parse_frontier_caps(["Uncapped"])
        assert caps == [None]
        assert invalid == []

    def test_invalid_entries_collected_not_raised(self) -> None:
        caps, invalid = _parse_frontier_caps(
            ["banana", "-1", "inf", "nan", "1.2"]
        )
        assert caps == [1.2]
        assert sorted(invalid) == ["-1", "banana", "inf", "nan"]

    def test_blank_and_none_input_yield_empty(self) -> None:
        assert _parse_frontier_caps(None) == ([], [])
        assert _parse_frontier_caps(["", "  "]) == ([], [])


class TestFrontierExportAssumptions:
    _BASE = pd.DataFrame([
        {
            "parameter": "Power", "value": "1", "unit": "MW",
            "source": "Sidebar", "affects": "Everything",
        },
    ])

    _SUMMARY: ClassVar[dict] = {
        "cost_per_cycle_eur": 100.0,
        "wear_eur_per_mwh_discharged": 100.0,
        "cycle_life": 6000.0,
        "valid_days": 2,
    }

    def test_none_assumptions_pass_through(self) -> None:
        assert (
            _append_frontier_assumptions(
                None, summary=self._SUMMARY, capex_eur_kwh=600.0,
            )
            is None
        )

    def test_appends_contract_provenance_rows(self) -> None:
        out = _append_frontier_assumptions(
            self._BASE, summary=self._SUMMARY, capex_eur_kwh=600.0,
        )
        assert len(out) == len(self._BASE) + 7
        params = set(out["parameter"])
        for expected in (
            "Frontier wear model",
            "Frontier cost per cycle",
            "Frontier wear per MWh discharged",
            "Frontier cycle life",
            "Frontier capex basis",
            "Frontier revenue basis",
            "Frontier annualisation",
        ):
            assert expected in params
        cost_row = out[out["parameter"] == "Frontier cost per cycle"].iloc[0]
        assert cost_row["value"] == "100.00"
        assert cost_row["unit"] == "EUR/FEC"
        capex_row = out[out["parameter"] == "Frontier capex basis"].iloc[0]
        assert capex_row["value"] == "600"
        assert "single source of truth" in capex_row["source"]

    def test_export_bytes_roundtrip(self) -> None:
        frontier = pd.DataFrame(
            {
                "cycle_cap": [1.0], "label": ["1 EFC/day"],
                "gross_eur": [10.0], "net_eur": [8.0],
            }
        )
        out = _append_frontier_assumptions(
            self._BASE, summary=self._SUMMARY, capex_eur_kwh=600.0,
        )
        data = cockpit_tables_to_excel(
            {"Cycle-cap frontier": frontier}, assumptions=out,
        )
        assert isinstance(data, bytes) and len(data) > 0


def test_render_places_frontier_after_multi_day_summary() -> None:
    """Contract section 4 wiring pin: the frontier expander renders
    immediately after the multi-day replay section (and before the
    forecast-policy section)."""
    tree = ast.parse(inspect.getsource(cockpit))
    render_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render"
    )
    calls = [
        node.func.id
        for node in ast.walk(render_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    order = (
        "_render_multi_day_summary",
        "_render_cycle_frontier_section",
        "_render_forecast_policy_section",
    )
    for name in order:
        assert name in calls, f"render() no longer calls {name}"
    assert (
        calls.index(order[0]) < calls.index(order[1]) < calls.index(order[2])
    )


def _frontier_app() -> None:
    """Minimal harness driving the REAL frontier panel (AppTest target)."""
    import pandas as pd

    from src.pages.simulation_cockpit import _render_cycle_frontier_section
    from src.simulation import available_local_dates

    day = [10.0] * 6 + [100.0] * 3 + [10.0] * 6 + [100.0] * 3 + [50.0] * 6
    idx = pd.date_range("2026-03-02", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame({"price_eur_mwh": day + day}, index=idx)
    df.index.name = "timestamp"
    _render_cycle_frontier_section(
        primary_df=df,
        dates=available_local_dates(df, tz="UTC"),
        zone_tz="UTC",
        power_mw=1.0,
        duration_hours=1,
        efficiency=0.9,
        capex_eur_kwh=150.0,
        chart_template="plotly_dark",
        assumptions=pd.DataFrame([
            {
                "parameter": "Power", "value": "1", "unit": "MW",
                "source": "Sidebar", "affects": "Everything",
            },
        ]),
    )


class TestFrontierAppTestSmoke:
    """Headless smoke of the real Streamlit panel (no browser needed)."""

    @pytest.fixture()
    def app(self) -> AppTest:
        at = AppTest.from_function(_frontier_app)
        at.run(timeout=30)
        return at

    def test_initial_render_prompts_for_run(self, app: AppTest) -> None:
        assert not app.exception
        assert any("click Run" in info.value for info in app.info)
        # The capex single-source caption is visible before running.
        assert any("inherited from the sidebar" in c.value for c in app.caption)

    def test_run_click_renders_frontier_outputs(self, app: AppTest) -> None:
        app.button(key="cycle_frontier_run").click().run(timeout=120)
        assert not app.exception
        labels = [m.label for m in app.metric]
        assert "Best cap (net of wear)" in labels
        assert "Cost per cycle" in labels
        assert "Valid days" in labels
        # The double-cycle fixture makes the cap bind, so the frontier is
        # not flat and a best cap is reported.
        best = next(
            m for m in app.metric if m.label == "Best cap (net of wear)"
        )
        assert best.value and best.value != "-"
        # Result table + best-cap-rule caption rendered.
        assert len(app.dataframe) >= 1
        assert any("LOWEST finite cap" in c.value for c in app.caption)
