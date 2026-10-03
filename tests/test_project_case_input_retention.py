"""Project Case inputs survive a transient validation failure.

Streamlit discards the state of every widget that a script run does not
render.  The panel used to raise at the first invalid input, so every widget
after it (contract terms, Bootstrap seed and simulations, ...) silently fell
back to its default once the user repaired the input.  These AppTest pins
drive "valid -> invalid -> repaired" and require every entered value to
survive, while the invalid state still cannot run and never shows a result.
"""

from __future__ import annotations

import datetime as dt

import pytest
from streamlit.testing.v1 import AppTest

import src.pages.project_case as project_page
from tests import pc_case_fixtures as fx

_TIMEOUT = 30
_INDICATIVE = next(
    label
    for label, status in project_page._QUOTE_STATUS_CHOICES.items()
    if status.name == "USER_ASSERTED_INDICATIVE_QUOTE"
)
_AS_OF = dt.date(2026, 5, 4)
_SHA = "ab" * 32


def _panel_app() -> None:
    import pandas as pd

    from src.pages.project_case import render_project_case_panel
    from tests import pc_case_fixtures as fx

    idx = pd.date_range("2026-03-10", periods=48, freq="h", tz="Europe/Berlin")
    frame = pd.DataFrame({"price_eur_mwh": [50.0] * len(idx)}, index=idx)
    frame.index.name = "timestamp"
    render_project_case_panel(
        primary_zone="DE_LU",
        primary_df=frame,
        start_date=fx.D1,
        end_date=fx.D2,
        power_mw=10.0,
        duration_hours=2.0,
        efficiency=0.88,
        capture_rate=0.9,
        capex_eur_kwh=100.0,
    )


def _entered_app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """A fully valid contracted panel with every retained input off-default."""
    monkeypatch.setattr(project_page, "emit_da_only", lambda *a, **k: fx.da_only_srr())
    app = AppTest.from_function(_panel_app).run(timeout=_TIMEOUT)
    assert not app.exception
    app.number_input(key="pc_eol_residual").set_value(12_000.0).run(timeout=_TIMEOUT)
    app.number_input(key="pc_decommissioning").set_value(3_000.0).run(timeout=_TIMEOUT)
    app.selectbox(key="pc_contract_mode").set_value(
        project_page._CONTRACT_FLOOR_LABEL
    ).run(timeout=_TIMEOUT)
    app.number_input(key="pc_contract_start_year").set_value(3).run(timeout=_TIMEOUT)
    app.number_input(key="pc_contract_tenor").set_value(2).run(timeout=_TIMEOUT)
    app.number_input(key="pc_contract_flat_rate").set_value(10_000.0).run(
        timeout=_TIMEOUT
    )
    app.selectbox(key="pc_contract_factor_mode").set_value(
        project_page._CONTRACT_FACTOR_FLAT
    ).run(timeout=_TIMEOUT)
    app.number_input(key="pc_contract_flat_factor_pct").set_value(80.0).run(
        timeout=_TIMEOUT
    )
    app.selectbox(key="pc_contract_quote_status").set_value(_INDICATIVE).run(
        timeout=_TIMEOUT
    )
    app.text_input(key="pc_contract_source").set_value("desk indication").run(
        timeout=_TIMEOUT
    )
    app.date_input(key="pc_contract_as_of").set_value(_AS_OF).run(timeout=_TIMEOUT)
    app.text_input(key="pc_contract_source_sha256").set_value(_SHA).run(
        timeout=_TIMEOUT
    )
    app.text_input(key="pc_bootstrap_seed").set_value("7").run(timeout=_TIMEOUT)
    app.number_input(key="pc_bootstrap_simulations").set_value(2_000).run(
        timeout=_TIMEOUT
    )
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    return app


def _assert_downstream_inputs_retained(app: AppTest) -> None:
    """Every value entered by ``_entered_app`` after the lifecycle section."""
    assert app.selectbox(key="pc_contract_mode").value == (
        project_page._CONTRACT_FLOOR_LABEL
    )
    assert app.selectbox(key="pc_contract_factor_mode").value == (
        project_page._CONTRACT_FACTOR_FLAT
    )
    assert app.number_input(key="pc_contract_flat_factor_pct").value == 80.0
    assert app.selectbox(key="pc_contract_quote_status").value == _INDICATIVE
    assert app.text_input(key="pc_contract_source").value == "desk indication"
    assert app.date_input(key="pc_contract_as_of").value == _AS_OF
    assert app.text_input(key="pc_contract_source_sha256").value == _SHA
    assert app.text_input(key="pc_bootstrap_seed").value == "7"
    assert app.number_input(key="pc_bootstrap_simulations").value == 2_000


def _assert_blocked(app: AppTest, fragment: str) -> None:
    """An invalid state shows its reason, has no run button and no result."""
    assert not app.exception
    assert any(fragment in error.value for error in app.error), [
        error.value for error in app.error
    ]
    assert not [button for button in app.button if button.key == "pc_run"]
    assert len(app.metric) == 0


def _assert_runnable(app: AppTest) -> None:
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    assert [button for button in app.button if button.key == "pc_run"]


def test_contract_term_overrun_then_repair_keeps_every_entered_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _entered_app(monkeypatch)
    app.button(key="pc_run").click().run(timeout=_TIMEOUT)
    assert not app.exception
    assert app.metric, "a settled result must render before the overrun"

    app.number_input(key="pc_contract_start_year").set_value(20).run(timeout=_TIMEOUT)
    _assert_blocked(app, "beyond the 20-year project life")
    # The rest of the form stays on screen while the term is invalid.
    assert app.number_input(key="pc_contract_flat_rate").value == 10_000.0
    _assert_downstream_inputs_retained(app)

    app.number_input(key="pc_contract_start_year").set_value(3).run(timeout=_TIMEOUT)
    _assert_runnable(app)
    assert app.number_input(key="pc_contract_tenor").value == 2
    assert app.number_input(key="pc_contract_flat_rate").value == 10_000.0
    _assert_downstream_inputs_retained(app)
    # The invalid state discarded the earlier result; repair needs a fresh run.
    assert len(app.metric) == 0

    preview = next(
        frame.value
        for frame in app.dataframe
        if "effective_whole_project_floor_eur" in frame.value.columns
    )
    assert list(preview["project_year"]) == [3, 4]
    assert list(preview["effective_whole_project_floor_eur"]) == [80_000.0, 80_000.0]


def test_rate_curve_parse_failure_then_repair_keeps_every_entered_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _entered_app(monkeypatch)
    app.selectbox(key="pc_contract_rate_mode").set_value(
        project_page._CONTRACT_CURVE_EXPLICIT
    ).run(timeout=_TIMEOUT)
    app.text_area(key="pc_contract_rate_curve").set_value("10000, abc").run(
        timeout=_TIMEOUT
    )
    _assert_blocked(app, "non-numeric entry 'abc'")
    _assert_downstream_inputs_retained(app)

    app.text_area(key="pc_contract_rate_curve").set_value("10000 12000").run(
        timeout=_TIMEOUT
    )
    _assert_runnable(app)
    _assert_downstream_inputs_retained(app)
    app.button(key="pc_run").click().run(timeout=_TIMEOUT)
    assert not app.exception
    assert app.metric


def test_entitlement_curve_length_failure_keeps_status_source_and_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _entered_app(monkeypatch)
    app.selectbox(key="pc_contract_factor_mode").set_value(
        project_page._CONTRACT_FACTOR_EXPLICIT
    ).run(timeout=_TIMEOUT)
    app.text_area(key="pc_contract_factor_curve").set_value("1.0").run(
        timeout=_TIMEOUT
    )
    _assert_blocked(app, "exactly 2 value(s)")
    assert app.selectbox(key="pc_contract_quote_status").value == _INDICATIVE
    assert app.text_input(key="pc_contract_source").value == "desk indication"
    assert app.date_input(key="pc_contract_as_of").value == _AS_OF
    assert app.text_input(key="pc_contract_source_sha256").value == _SHA
    assert app.text_input(key="pc_bootstrap_seed").value == "7"
    assert app.number_input(key="pc_bootstrap_simulations").value == 2_000


def test_invalid_bootstrap_seed_keeps_the_simulation_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _entered_app(monkeypatch)
    app.text_input(key="pc_bootstrap_seed").set_value("seven").run(timeout=_TIMEOUT)
    _assert_blocked(app, "Bootstrap seed")
    assert app.number_input(key="pc_bootstrap_simulations").value == 2_000

    app.text_input(key="pc_bootstrap_seed").set_value("7").run(timeout=_TIMEOUT)
    _assert_runnable(app)
    _assert_downstream_inputs_retained(app)


def test_pending_augmentation_upload_keeps_later_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _entered_app(monkeypatch)
    app.selectbox(key="pc_capacity_maintenance_basis").set_value(
        "Scheduled nameplate maintenance — CSV"
    ).run(timeout=_TIMEOUT)
    _assert_blocked(app, "Upload an augmentation schedule CSV")
    assert app.number_input(key="pc_eol_residual").value == 12_000.0
    assert app.number_input(key="pc_decommissioning").value == 3_000.0
    _assert_downstream_inputs_retained(app)

    app.selectbox(key="pc_capacity_maintenance_basis").set_value(
        "Unknown — screening NPV only"
    ).run(timeout=_TIMEOUT)
    _assert_runnable(app)
    assert app.number_input(key="pc_eol_residual").value == 12_000.0
    assert app.number_input(key="pc_decommissioning").value == 3_000.0
    _assert_downstream_inputs_retained(app)


def test_pending_multiplier_upload_keeps_contract_and_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _entered_app(monkeypatch)
    app.selectbox(key="pc_projection_kind").set_value("Explicit multiplier CSV").run(
        timeout=_TIMEOUT
    )
    _assert_blocked(app, "Upload an explicit annual multiplier CSV")
    _assert_downstream_inputs_retained(app)

    app.selectbox(key="pc_projection_kind").set_value("Flat real").run(
        timeout=_TIMEOUT
    )
    _assert_runnable(app)
    _assert_downstream_inputs_retained(app)


def test_every_independent_input_error_is_reported_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _entered_app(monkeypatch)
    app.number_input(key="pc_contract_start_year").set_value(20).run(timeout=_TIMEOUT)
    app.text_input(key="pc_bootstrap_seed").set_value("seven").run(timeout=_TIMEOUT)
    _assert_blocked(app, "beyond the 20-year project life")
    assert any("Bootstrap seed" in error.value for error in app.error)
