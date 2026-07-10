"""Tests for the cockpit cycle-cap frontier panel (F-C).

Covers the pure helpers (cap parsing, default options, export assumptions,
sweep fingerprint), the render() wiring order pin (contract section 4: the
expander sits immediately after the multi-day replay section), and an
AppTest headless smoke that drives the REAL panel — chart traces, table
content, best-row marker, export button, session-state persistence across
reruns, and fingerprint invalidation — the app-layer surface mocked unit
tests cannot reach.
"""

from __future__ import annotations

import ast
import inspect
import json
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
    _frontier_fingerprint,
    _parse_frontier_caps,
)

# Contract section 4 locked UI copy (literal, including the multiply sign).
_FRONTIER_EXPANDER_TITLE = (
    "Cycle-cap × degradation net-revenue frontier"  # noqa: RUF001
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


class TestFrontierFingerprint:
    _KW: ClassVar[dict] = dict(
        primary_zone="DE_LU",
        cycle_life=6000.0,
        sweep_dates=["2026-03-02", "2026-03-03"],
        zone_tz="UTC",
        power_mw=1.0,
        duration_hours=1.0,
        efficiency=0.9,
        capex_eur_kwh=150.0,
    )

    def test_identical_inputs_match_and_cap_order_is_irrelevant(self) -> None:
        a = _frontier_fingerprint(caps=[1.0, None, 0.5], **self._KW)
        b = _frontier_fingerprint(caps=[None, 0.5, 1.0], **self._KW)
        assert a == b

    def test_uncapped_sentinel_does_not_collide_with_cap_zero(self) -> None:
        a = _frontier_fingerprint(caps=[None], **self._KW)
        b = _frontier_fingerprint(caps=[0.0], **self._KW)
        assert a != b

    @pytest.mark.parametrize(
        "override",
        [
            {"primary_zone": "AT"},
            {"cycle_life": 3000.0},
            {"sweep_dates": ["2026-03-02"]},
            {"capex_eur_kwh": 0.0},
            {"power_mw": 2.0},
        ],
    )
    def test_any_input_change_invalidates(self, override: dict) -> None:
        base = _frontier_fingerprint(caps=[1.0], **self._KW)
        changed = _frontier_fingerprint(caps=[1.0], **{**self._KW, **override})
        assert base != changed


class TestFrontierExportAssumptions:
    _BASE: ClassVar[pd.DataFrame] = pd.DataFrame([
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
    forecast-policy section). Reads TOP-LEVEL statement order in
    render()'s body — ast.walk yields nodes in unspecified order (review
    catch), so the lexical statement sequence is what is pinned."""
    tree = ast.parse(inspect.getsource(cockpit))
    render_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render"
    )
    calls = [
        stmt.value.func.id
        for stmt in render_fn.body
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
    ]
    order = (
        "_render_multi_day_summary",
        "_render_cycle_frontier_section",
        "_render_forecast_policy_section",
    )
    for name in order:
        assert name in calls, f"render() no longer calls {name} at top level"
    idx = [calls.index(name) for name in order]
    assert idx[0] < idx[1] < idx[2]
    # "Immediately after": no other top-level call sits between the
    # multi-day summary and the frontier section.
    assert idx[1] == idx[0] + 1


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
        primary_zone="DE_LU",
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


def _find_elements(node, type_name: str) -> list:
    """Depth-first collect AppTest elements of a given proto type (reaches
    the untyped ones — plotly_chart, download_button — the typed AppTest
    API does not expose)."""
    found: list = []
    children = getattr(node, "children", None)
    if isinstance(children, dict):
        for child in children.values():
            if getattr(child, "type", "") == type_name:
                found.append(child)
            found.extend(_find_elements(child, type_name))
    return found


class TestFrontierAppTestSmoke:
    """Headless smoke of the real Streamlit panel (no browser needed)."""

    @pytest.fixture()
    def app(self) -> AppTest:
        at = AppTest.from_function(_frontier_app)
        at.run(timeout=30)
        return at

    def test_initial_render_prompts_for_run(self, app: AppTest) -> None:
        assert not app.exception
        expander = app.main.children[0]
        assert expander.label == _FRONTIER_EXPANDER_TITLE
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
        best = next(
            m for m in app.metric if m.label == "Best cap (net of wear)"
        )
        assert best.value and best.value != "-"

        # ONE grouped chart with exactly the gross + net traces (the gap
        # IS the wear — a second chart or a dropped trace fails here).
        charts = _find_elements(app.main, "plotly_chart")
        assert len(charts) == 1
        spec = json.loads(charts[0].proto.spec)
        assert [(t.get("name"), t.get("type")) for t in spec["data"]] == [
            ("Gross", "bar"), ("Net of wear", "bar"),
        ]

        # Table: one row per default cap, best row flagged exactly once,
        # and the star sits on the row named by the best-cap metric.
        table = app.dataframe[0].value
        table = table.data if hasattr(table, "data") else table
        assert len(table) == len(DEFAULT_CYCLE_CAPS)
        assert list(table.columns[:2]) == ["Best", "Cap"]
        starred = table[table["Best"] == "★"]
        assert len(starred) == 1
        assert starred["Cap"].iloc[0] == best.value

        # Best-cap-rule caption + Excel export button rendered.
        assert any("LOWEST finite cap" in c.value for c in app.caption)
        downloads = _find_elements(app.main, "download_button")
        assert len(downloads) == 1
        assert "Excel" in downloads[0].proto.label

    def test_results_survive_an_unrelated_rerun(self, app: AppTest) -> None:
        """Review catch (Gemini): st.button is only True on its own click's
        rerun, so without the session-state cache a download-button click
        would collapse the results back to the run prompt."""
        app.button(key="cycle_frontier_run").click().run(timeout=120)
        app.run(timeout=30)  # unrelated rerun: run button back at False
        assert not app.exception
        assert len(app.dataframe) == 1
        assert len(_find_elements(app.main, "download_button")) == 1
        assert "Best cap (net of wear)" in [m.label for m in app.metric]

    def test_changed_inputs_invalidate_cached_results(
        self, app: AppTest
    ) -> None:
        """Fingerprint gate: a knob change after a run prompts a re-run
        instead of rendering a stale table under new labels."""
        app.button(key="cycle_frontier_run").click().run(timeout=120)
        app.number_input(key="cycle_frontier_cycle_life").set_value(
            3000.0
        ).run(timeout=30)
        assert not app.exception
        assert any("Inputs changed" in info.value for info in app.info)
        assert len(app.dataframe) == 0
        assert len(_find_elements(app.main, "download_button")) == 0
