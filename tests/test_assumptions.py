"""Tests for the consolidated model-assumption audit table."""

from __future__ import annotations

from src.assumptions import ASSUMPTION_COLUMNS, build_assumptions_table


def _base() -> dict:
    return {
        "power_mw": 10.0,
        "duration_hours": 2.0,
        "efficiency": 0.88,
        "capture_rate": 0.70,
        "capex_eur_kwh": 0.0,
        "use_lp_dispatch": False,
    }


def test_core_assumptions_present_with_expected_columns() -> None:
    table = build_assumptions_table(**_base())
    assert list(table.columns) == ASSUMPTION_COLUMNS
    params = set(table["parameter"])
    for expected in (
        "Power", "Duration", "Round-trip efficiency",
        "Capture rate (DA slippage)", "VOM cost", "Annualisation",
        "Continuous-replay interval cap",
    ):
        assert expected in params


def test_values_are_formatted_for_display() -> None:
    table = build_assumptions_table(**_base())
    by_param = dict(zip(table["parameter"], table["value"], strict=True))
    assert by_param["Round-trip efficiency"] == "88%"
    assert by_param["Capture rate (DA slippage)"] == "70%"
    assert by_param["Annualisation"] == "365.25"
    assert by_param["VOM cost"] == "0.5"


def test_dispatch_model_label_reflects_toggle() -> None:
    greedy = build_assumptions_table(**{**_base(), "use_lp_dispatch": False})
    milp = build_assumptions_table(**{**_base(), "use_lp_dispatch": True})
    greedy_val = dict(zip(greedy["parameter"], greedy["value"], strict=True))
    milp_val = dict(zip(milp["parameter"], milp["value"], strict=True))
    assert greedy_val["Dispatch model"] == "Greedy single-cycle"
    assert milp_val["Dispatch model"] == "MILP multi-cycle"


def test_panel_knobs_appended_only_when_provided() -> None:
    without = build_assumptions_table(**_base())
    assert "Rebid deadband" not in set(without["parameter"])
    assert "IDA forecast mode" not in set(without["parameter"])

    with_knobs = build_assumptions_table(
        **_base(),
        rebid_share=0.25,
        deadband_eur_per_mw=2.5,
        forecast_mode="loo",
        forecast_bucket="hour_of_day",
    )
    params = set(with_knobs["parameter"])
    assert "IDA rebid share (screening)" in params
    assert "Rebid deadband" in params
    assert "IDA forecast mode" in params
    assert "IDA forecast bucket" in params
    knob_val = dict(zip(
        with_knobs["parameter"], with_knobs["value"], strict=True,
    ))
    assert knob_val["Rebid deadband"] == "2.5"
    assert knob_val["IDA rebid share (screening)"] == "25%"
