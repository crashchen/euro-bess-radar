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


def test_activation_capture_share_appends_overlay_rows() -> None:
    without = build_assumptions_table(**_base())
    assert "Activation capture share" not in set(without["parameter"])

    with_act = build_assumptions_table(**_base(), activation_capture_share=0.01)
    params = set(with_act["parameter"])
    assert "Activation capture share" in params
    assert "Activation data interpretation" in params
    assert "Activation overlay basis" in params
    by_val = dict(zip(with_act["parameter"], with_act["value"], strict=True))
    assert by_val["Activation capture share"] == "1.00%"
    # The non-additive / no-SoC red-line must be spelled out in the audit.
    basis = with_act[with_act["parameter"] == "Activation overlay basis"].iloc[0]
    assert "not additive" in basis["affects"].lower()
    assert "no soc" in basis["affects"].lower()


# ── Cockpit export capture-row override ─────────────────────────────────────

def test_cockpit_export_overrides_sidebar_capture_row() -> None:
    from src.pages.simulation_cockpit import _cockpit_export_assumptions

    table = build_assumptions_table(**_base())  # capture_rate=0.70 -> "70%"
    out = _cockpit_export_assumptions(
        table, capture_value="100%",
        capture_affects="Cockpit haircut",
    )
    # The sidebar capture row is gone; a cockpit capture row replaces it.
    params = set(out["parameter"])
    assert "Capture rate (DA slippage)" not in params
    assert "Cockpit capture haircut" in params
    row = out[out["parameter"] == "Cockpit capture haircut"].iloc[0]
    assert row["value"] == "100%"
    assert row["source"] == "Cockpit panel"
    # No new row count drift (replaced in place, not appended).
    assert len(out) == len(table)


def test_cockpit_export_appends_when_capture_row_missing() -> None:
    import pandas as pd

    from src.assumptions import ASSUMPTION_COLUMNS
    from src.pages.simulation_cockpit import _cockpit_export_assumptions

    no_capture = pd.DataFrame(
        [["Power", "10", "MW", "Sidebar", "scaling"]],
        columns=ASSUMPTION_COLUMNS,
    )
    out = _cockpit_export_assumptions(
        no_capture, capture_value="not applied", capture_affects="raw",
        capture_label="Capture haircut",
    )
    assert "Capture haircut" in set(out["parameter"])
    assert len(out) == 2  # appended, nothing dropped


def test_cockpit_export_assumptions_none_passthrough() -> None:
    from src.pages.simulation_cockpit import _cockpit_export_assumptions

    assert _cockpit_export_assumptions(None, capture_value="x", capture_affects="y") is None


def test_reserve_assumptions_appended_with_provenance() -> None:
    import pandas as pd

    from src.config import ANCILLARY_CAPACITY_AVAILABILITY
    from src.pages.simulation_cockpit import _append_reserve_assumptions

    base = pd.DataFrame(
        [["Power", "10", "MW", "Sidebar", "scaling"]], columns=ASSUMPTION_COLUMNS,
    )
    out = _append_reserve_assumptions(
        base, product="FCR", capacity_price_eur_mw_h=13.0,
    )
    params = set(out["parameter"])
    assert {"Reserve co-opt product", "Reserve capacity price",
            "Reserve availability haircut", "Reserve activation energy",
            "Reserve additivity with DA"} <= params
    # Audit red-line is spelled out, not implied.
    avail_row = out[out["parameter"] == "Reserve availability haircut"].iloc[0]
    assert avail_row["value"] == f"{ANCILLARY_CAPACITY_AVAILABILITY:.2f}"
    energy_row = out[out["parameter"] == "Reserve activation energy"].iloc[0]
    assert energy_row["value"] == "not modelled"


def test_reserve_assumptions_none_passthrough() -> None:
    from src.pages.simulation_cockpit import _append_reserve_assumptions

    assert _append_reserve_assumptions(
        None, product="FCR", capacity_price_eur_mw_h=1.0,
    ) is None


def test_triple_assumptions_state_ceiling_and_not_forecast_driven() -> None:
    import pandas as pd

    from src.pages.simulation_cockpit import _append_triple_assumptions

    base = pd.DataFrame(
        [["Power", "10", "MW", "Sidebar", "scaling"]], columns=ASSUMPTION_COLUMNS,
    )
    out = _append_triple_assumptions(base)
    row = out[out["parameter"] == "DA+IDA1+reserve row type"].iloc[0]
    assert row["value"] == "perfect-foresight ceiling"
    # Codex's audit requirement: no-activation / not-forecast-driven spelled out.
    assert "no activation energy" in row["affects"]
    assert "NOT forecast-driven" in row["affects"]


def test_triple_assumptions_none_passthrough() -> None:
    from src.pages.simulation_cockpit import _append_triple_assumptions

    assert _append_triple_assumptions(None) is None
