"""Capacity settlement is attached to actual, frozen cockpit result paths."""

from __future__ import annotations

import inspect
from io import BytesIO

import pandas as pd
import pytest
from openpyxl import load_workbook
from streamlit.testing.v1 import AppTest


def _compute_bundle(
    monkeypatch, *, product="FCR", zone="DE_LU", reserve=437.0,
    triple=450.0, realistic=430.0, policy=-3.0, reserve_mode=True,
    assumptions=None, unavailable=False,
):
    from src.pages import simulation_cockpit as cockpit
    from src.strategy_compare import (
        STOCHASTIC_POLICY_VALUE_LABEL,
        STOCHASTIC_POLICY_VALUE_RESERVE_LABEL,
    )

    day = pd.Timestamp("2026-03-29").date()
    per_day = pd.DataFrame({"date": [day], "realised_eur": [400.0]})
    summary = {
        "valid_days": 1, "excluded_days": 0, "total_da_only_eur": 300.0,
        "total_realised_eur": 400.0, "total_ceiling_eur": 460.0,
        "forecast_meta": {
            "forecast_mode": "walk_forward", "n_buckets_filled": 24,
            "n_buckets_requested": 24, "fallback_points": 0, "coverage": 1.0,
        },
    }
    if unavailable:
        per_day = per_day.iloc[:0]
        summary.update(
            valid_days=0, excluded_days=1, model_available=False,
            excluded_days_due_to_solver_failure=1,
        )
    monkeypatch.setattr(
        cockpit, "simulate_sequential_da_id_batch",
        lambda *args, **kwargs: (per_day, summary),
    )
    monkeypatch.setattr(
        cockpit, "_reserve_coopt_total",
        lambda *args, **kwargs: (reserve, f"DA + {product} co-opt (headroom)", 20.0),
    )
    monkeypatch.setattr(cockpit, "capacity_price_series_for_product", lambda *a: None)
    monkeypatch.setattr(cockpit, "_reserve_triple_totals", lambda *a, **kw: {
        "triple_total": triple, "realistic_total": realistic,
        "triple_valid_days": 1, "triple_da_baseline": 300.0,
        "seq_per_day": None, "seq_summary": None,
        "solver_failed_days": 0, "model_available": True,
    })
    monkeypatch.setattr(cockpit, "_compute_stochastic_policy", lambda *a, **kw: {
        "per_day": None, "summary": {"valid_days": 1} if policy is not None else None,
        "reserve_mode": reserve_mode, "policy_value_total": policy,
        "policy_value_valid_days": 1,
        "policy_value_label": (
            STOCHASTIC_POLICY_VALUE_RESERVE_LABEL if reserve_mode
            else STOCHASTIC_POLICY_VALUE_LABEL
        ),
    })
    prices = pd.DataFrame(
        {"price_eur_mwh": [50.0] * 23},
        index=pd.date_range("2026-03-28 23:00", periods=23, freq="h", tz="UTC"),
    )
    context = (
        {"primary_zone": zone}
        if "primary_zone" in inspect.signature(
            cockpit._compute_forecast_policy_bundle
        ).parameters else {}
    )
    return cockpit._compute_forecast_policy_bundle(
        fingerprint="frozen-input", primary_df=prices,
        intraday_df=prices.rename(columns={"price_eur_mwh": "intraday_price_eur_mwh"}),
        capacity_df=None, reserve_product=product, batch_dates=[day],
        zone_tz="Europe/Berlin", power_mw=1.0, duration_hours=2, efficiency=0.88,
        bucket="hour_of_day", forecast_mode="walk_forward", deadband_eur_per_mw=0.0,
        include_stochastic=policy is not None, stochastic_cap_pct=50.0,
        assumptions=assumptions, **context,
    )


def _capacity_rows(bundle):
    table = bundle["derived"]["comparison"]
    return table.loc[~table["capacity_settlement_basis"].str.startswith("Not applicable")]


@pytest.mark.parametrize("zone,product", [("DE_LU", "FCR"), ("FI", "FCR-N"), ("NL", "aFRR")])
def test_capacity_rows_bind_actual_run_zone_product(monkeypatch, zone, product):
    bundle = _compute_bundle(monkeypatch, zone=zone, product=product)
    settlement = bundle["derived"]["capacity_settlement"]
    assert settlement["zone"] == zone
    assert settlement["product"] == product
    assert "physical" in settlement["basis"].lower()
    assert _capacity_rows(bundle)["capacity_settlement_basis"].tolist() == [
        "Physical delivery hours"
    ] * 4
    assert _capacity_rows(bundle)["capacity_settlement_scope"].tolist() == [
        f"{zone} / {product}"
    ] * 4
    assert bundle["derived"]["comparison"].iloc[:3][
        "capacity_settlement_basis"
    ].str.startswith("Not applicable").all()


@pytest.mark.parametrize(
    "reserve,triple,realistic,policy,reserve_mode,expected",
    [
        (437.0, None, None, None, False, 1),
        (None, 450.0, None, None, False, 1),
        (None, None, 430.0, None, False, 1),
        (None, None, None, -3.0, True, 1),
        (None, None, None, -3.0, False, 0),
        (None, None, None, None, False, 0),
        (0.0, 0.0, 0.0, 0.0, True, 4),
        (float("nan"), float("inf"), None, None, False, 0),
    ],
)
def test_only_real_finite_capacity_paths_receive_basis(
    monkeypatch, reserve, triple, realistic, policy, reserve_mode, expected,
):
    bundle = _compute_bundle(
        monkeypatch, reserve=reserve, triple=triple, realistic=realistic,
        policy=policy, reserve_mode=reserve_mode,
    )
    assert len(_capacity_rows(bundle)) == expected
    assert (bundle["derived"]["capacity_settlement"] is None) is (expected == 0)


def test_numeric_comparison_unchanged_including_negative_delta(monkeypatch):
    from src.strategy_compare import STRATEGY_COMPARE_COLUMNS, build_strategy_comparison

    bundle = _compute_bundle(monkeypatch)
    table = bundle["derived"]["comparison"]
    expected = build_strategy_comparison(
        bundle["summary"], power_mw=1.0, reserve_coopt_total=437.0,
        reserve_label=table.iloc[3]["strategy"], triple_joint_total=450.0,
        triple_joint_label=table.iloc[4]["strategy"], realistic_triple_total=430.0,
        realistic_triple_label=table.iloc[5]["strategy"], triple_valid_days=1,
        triple_da_baseline=300.0, policy_value_total=-3.0,
        policy_value_valid_days=1, policy_value_label=table.iloc[6]["strategy"],
    )
    pd.testing.assert_frame_equal(table[STRATEGY_COMPARE_COLUMNS], expected)
    assert table.iloc[-1]["window_revenue_eur"] == -3.0
    assert pd.isna(table.iloc[-1]["uplift_vs_da_pct"])


def test_text_containing_reserve_does_not_classify_a_row():
    from src.pages.simulation_cockpit import _comparison_capacity_disclosure

    table = pd.DataFrame({"strategy": ["DA-only reserve wording", "Actual capacity"]})
    rendered, settlement = _comparison_capacity_disclosure(
        table, zone="DE_LU", product="FCR", reserve_labels=["Actual capacity"],
    )
    assert rendered.iloc[0]["capacity_settlement_basis"].startswith("Not applicable")
    assert settlement["strategies"] == ("Actual capacity",)
    assert list(table.columns) == ["strategy"]  # input remains untouched


@pytest.mark.parametrize("assumptions", [None, pd.DataFrame()])
def test_export_without_caller_assumptions_still_has_bound_basis(monkeypatch, assumptions):
    from src.export import cockpit_tables_to_excel

    bundle = _compute_bundle(monkeypatch, assumptions=assumptions)
    derived = bundle["derived"]
    wb = load_workbook(BytesIO(cockpit_tables_to_excel(
        derived["export_tables"], assumptions=derived["export_assumptions"],
    )))
    rows = list(wb["Assumptions"].values)
    labels = {row[0]: row[1] for row in rows[1:]}
    assert labels["Capacity settlement zone"] == "DE_LU"
    assert labels["Capacity settlement product"] == "FCR"
    assert labels["Capacity settlement basis"] == derived["capacity_settlement"]["basis"]
    exported = list(wb["Strategy comparison"].values)
    assert exported[0][-2:] == ("Capacity basis", "Capacity scope")
    assert exported[4][-2:] == ("Physical delivery hours", "DE_LU / FCR")
    assert wb["Strategy comparison"]["B5"].data_type == "n"
    assert wb["Strategy comparison"]["B5"].value == 437.0


def test_missing_product_is_unavailable_not_a_no_capacity_claim(monkeypatch):
    bundle = _compute_bundle(monkeypatch, product=None)
    assert "not recorded" in bundle["derived"]["capacity_settlement"]["basis"]
    assert len(_capacity_rows(bundle)) == 4


def test_unregistered_zone_is_explicitly_unverified(monkeypatch):
    bundle = _compute_bundle(monkeypatch, zone="UNREGISTERED")
    settlement = bundle["derived"]["capacity_settlement"]
    assert "unverified" in settlement["basis"].lower()
    assert _capacity_rows(bundle)["capacity_settlement_basis"].tolist() == ["Unverified"] * 4
    assert _capacity_rows(bundle)["capacity_settlement_scope"].tolist() == [
        "UNREGISTERED / FCR"
    ] * 4


def test_no_product_no_capacity_paths_remain_not_applicable(monkeypatch):
    bundle = _compute_bundle(
        monkeypatch, product=None, reserve=None, triple=None, realistic=None,
        policy=None, reserve_mode=False,
    )
    derived = bundle["derived"]
    assert derived["capacity_settlement"] is None
    assert derived["export_assumptions"] is None
    for column in ("capacity_settlement_basis", "capacity_settlement_scope"):
        assert derived["comparison"][column].tolist() == ["Not applicable"] * 3


def test_all_solver_failed_preserves_guard_and_never_builds_disclosure(monkeypatch):
    from src.pages import simulation_cockpit as cockpit

    def forbidden(*args, **kwargs):
        raise AssertionError("Disclosure must not run for unavailable model")

    monkeypatch.setattr(cockpit, "screening_capacity_settlement_basis", forbidden, raising=False)
    monkeypatch.setattr(cockpit, "screening_capacity_settlement_label", forbidden, raising=False)
    bundle = _compute_bundle(monkeypatch, unavailable=True)
    assert bundle["derived"] is None
    assert not bundle["summary"]["model_available"]


def _stored_result_app():
    import streamlit as st

    from src.pages.simulation_cockpit import _render_forecast_policy_bundle

    _render_forecast_policy_bundle(st.session_state["stored_bundle"], "plotly_dark")


def test_renderer_and_export_use_frozen_basis_without_resolving(monkeypatch):
    from src.pages import simulation_cockpit as cockpit

    bundle = _compute_bundle(monkeypatch)
    original = bundle["derived"]["capacity_settlement"]["basis"]
    for name in (
        "_render_forecast_policy_kpis", "_plot_forecast_policy",
        "_render_forecast_skill", "_render_stochastic_attribution_panel",
    ):
        monkeypatch.setattr(cockpit, name, lambda *a, **kw: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Rerender must not resolve settlement context or solve")

    monkeypatch.setattr(cockpit, "screening_capacity_settlement_basis", forbidden)
    monkeypatch.setattr(cockpit, "simulate_sequential_da_id_batch", forbidden)
    monkeypatch.setattr(cockpit, "screening_capacity_settlement_label", forbidden)
    app = AppTest.from_function(_stored_result_app)
    app.session_state["stored_bundle"] = bundle
    app.run(timeout=30)
    app.run(timeout=30)
    assert not app.exception
    assert any(f"Capacity settlement basis: {original}" in item.value for item in app.caption)
    assert len(app.dataframe) == 2
    assert app.dataframe[0].value.iloc[3]["capacity_settlement_basis"] == "Physical delivery hours"
    assert app.dataframe[0].value.iloc[3]["capacity_settlement_scope"] == "DE_LU / FCR"


def test_pre_disclosure_session_bundle_is_stale(monkeypatch):
    from src.pages import simulation_cockpit as cockpit
    from tests.test_step3b_batch_result_persistence import _forecast_kwargs

    current = cockpit._forecast_policy_fingerprint(**_forecast_kwargs())
    monkeypatch.setattr(cockpit, "_FORECAST_POLICY_PANEL_ID", "cockpit-forecast-policy/v1")
    assert current != cockpit._forecast_policy_fingerprint(**_forecast_kwargs())
