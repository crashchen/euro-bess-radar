"""PC-C RunResult-driven UI and cache contract tests."""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import src.pages.project_case as project_page
from src.project_case import (
    AssetCase,
    BootstrapCase,
    CapacityMaintenanceBasis,
    CashflowRow,
    CashflowTable,
    LifecycleCase,
    NpvDistribution,
    NpvOutcome,
    Projection,
    ProjectionKind,
    RunResult,
    ValuationCase,
    compute_project_case,
)
from src.project_case.audit import build_reserve_coverage_audit
from src.project_case.enums import (
    BOOTSTRAP_ALGORITHM_V1,
    LIFECYCLE_UNKNOWN_MESSAGE,
    LIFECYCLE_UNKNOWN_STATUS,
)
from tests import pc_case_fixtures as fx


def _table(basis: str, *, lifecycle: bool) -> CashflowTable:
    return CashflowTable(
        basis,
        (
            CashflowRow(
                year=1,
                revenue_eur=101.0,
                opex_eur=11.0 if lifecycle else 0.0,
                augmentation_eur=12.0 if lifecycle else 0.0,
                terminal_eur=13.0 if lifecycle else 0.0,
                net_eur=91.0 if lifecycle else 101.0,
                discount_factor=0.9,
                discounted_net_eur=81.9 if lifecycle else 90.9,
            ),
        ),
    )


def _available_result() -> RunResult:
    return RunResult(
        input_fingerprint="ab" * 32,
        no_lifecycle_cost_screening_npv=NpvOutcome.ok(
            NpvDistribution(-101.0, 202.0, 909.0, 0.61)
        ),
        lifecycle_cash_npv=NpvOutcome.ok(
            NpvDistribution(-303.0, 44.0, 707.0, 0.37)
        ),
        provenance={"red_line_assertions": {"floor_included": False}},
        screening_cashflow_table=_table("screening", lifecycle=False),
        lifecycle_cashflow_table=_table("lifecycle", lifecycle=True),
    )


def _unknown_result() -> RunResult:
    return RunResult(
        input_fingerprint="cd" * 32,
        no_lifecycle_cost_screening_npv=NpvOutcome.ok(
            NpvDistribution(-101.0, 202.0, 909.0, 0.61)
        ),
        lifecycle_cash_npv=NpvOutcome.unavailable(
            LIFECYCLE_UNKNOWN_STATUS,
            LIFECYCLE_UNKNOWN_MESSAGE,
        ),
        provenance={"red_line_assertions": {"floor_included": False}},
        screening_cashflow_table=_table("screening", lifecycle=False),
        lifecycle_cashflow_table=None,
    )


def _available_result_app() -> None:
    from src.pages.project_case import render_project_case_result
    from tests.test_project_case_ui import _available_result

    render_project_case_result(_available_result())


def _unknown_result_app() -> None:
    from src.pages.project_case import render_project_case_result
    from tests.test_project_case_ui import _unknown_result

    render_project_case_result(_unknown_result())


def _partial_result_app() -> None:
    import dataclasses

    from src.pages.project_case import render_project_case_result
    from tests.test_project_case_ui import _available_result

    result = _available_result()
    partial = dataclasses.replace(
        result,
        provenance={
            "red_line_assertions": {"floor_included": False},
            "strategy_run_result": {
                "strategy_kind": "DA_ONLY",
                "adapter_provenance": {"producer_adapter_id": "PC_ADP_DA_ONLY"},
                "coverage_audit": {
                    "observed_dates": ["2026-03-10", "2026-03-11"],
                    "valid_dates": ["2026-03-10"],
                    "missing_dates": ["2026-03-11"],
                    "solver_failed_dates": [],
                },
            },
        },
    )
    render_project_case_result(partial)


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


def _real_panel_app() -> None:
    import datetime as dt

    import pandas as pd

    from src.pages.project_case import render_project_case_panel

    index = pd.date_range(
        "2026-03-10",
        "2026-03-12",
        freq="15min",
        inclusive="left",
        tz="Europe/Berlin",
    )
    quarter = index.hour * 4 + index.minute // 15
    prices = [10.0 if value < 32 else 100.0 for value in quarter]
    frame = pd.DataFrame({"price_eur_mwh": prices}, index=index)
    frame.index.name = "timestamp"
    render_project_case_panel(
        primary_zone="DE_LU",
        primary_df=frame,
        start_date=dt.date(2026, 3, 10),
        end_date=dt.date(2026, 3, 11),
        power_mw=1.0,
        duration_hours=2.0,
        efficiency=0.88,
        capture_rate=0.9,
        capex_eur_kwh=100.0,
    )

def _multi_strategy_panel_app() -> None:
    import pandas as pd

    from src.pages.project_case import render_project_case_panel
    from tests import pc_case_fixtures as fx

    reserve = fx.reserve_series("DE_LU", [(fx.D1, 6), (fx.D2, 6)])
    ancillary = pd.DataFrame(
        {
            "product_type": ["FCR"] * len(reserve),
            "direction": ["symmetric"] * len(reserve),
            "zone": ["DE_LU"] * len(reserve),
            "capacity_price_eur_mw": reserve.to_numpy(),
        },
        index=reserve.index,
    )
    render_project_case_panel(
        primary_zone="DE_LU",
        primary_df=fx.da_frame("DE_LU", [fx.D1, fx.D2]),
        start_date=fx.D1,
        end_date=fx.D2,
        power_mw=10.0,
        duration_hours=2.0,
        efficiency=0.88,
        capture_rate=0.9,
        capex_eur_kwh=100.0,
        intraday_df=fx.ida_frame("DE_LU", [fx.D1, fx.D2]),
        ancillary_df=ancillary,
    )


def _cached_reserve_panel_app() -> None:
    from src import data_ingestion as di
    from src.pages.project_case import render_project_case_panel
    from tests import pc_case_fixtures as fx

    capacity_df = di.read_capacity_cache("DE_LU")
    if capacity_df is not None:
        capacity_df = capacity_df.assign(zone="DE_LU")
    render_project_case_panel(
        primary_zone="DE_LU",
        primary_df=fx.da_frame("DE_LU", [fx.D1, fx.D2]),
        start_date=fx.D1,
        end_date=fx.D2,
        power_mw=10.0,
        duration_hours=2.0,
        efficiency=0.88,
        capture_rate=0.9,
        capex_eur_kwh=100.0,
        capacity_df=capacity_df,
        capacity_sources=di.read_capacity_sources(),
    )


def _metric_map(app: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in app.metric}


def test_locked_project_case_labels_are_literal() -> None:
    assert project_page.SCREENING_NPV_LABEL == "No-lifecycle-cost screening NPV"
    assert project_page.LIFECYCLE_NPV_LABEL == (
        "Pre-tax unlevered lifecycle cash NPV"
    )
    assert project_page.P10_LABEL == "P10 (Downside)"
    assert project_page.P50_LABEL == "P50 (Median)"
    assert project_page.P90_LABEL == "P90 (Upside)"
    assert project_page.PROBABILITY_LABEL == "P(NPV > 0)"


def test_available_result_maps_distinct_sentinels_to_exact_ui_slots() -> None:
    app = AppTest.from_function(_available_result_app).run(timeout=30)
    assert not app.exception
    metrics = _metric_map(app)
    screening = project_page.SCREENING_NPV_LABEL
    lifecycle = project_page.LIFECYCLE_NPV_LABEL
    assert metrics[f"{screening} — P10 (Downside)"] == "€-101"
    assert metrics[f"{screening} — P50 (Median)"] == "€202"
    assert metrics[f"{screening} — P90 (Upside)"] == "€909"
    assert metrics[f"{screening} — P(NPV > 0)"] == "61%"
    assert metrics[f"{lifecycle} — P10 (Downside)"] == "€-303"
    assert metrics[f"{lifecycle} — P50 (Median)"] == "€44"
    assert metrics[f"{lifecycle} — P90 (Upside)"] == "€707"
    assert metrics[f"{lifecycle} — P(NPV > 0)"] == "37%"


def test_unknown_is_screening_only_and_never_rendered_as_zero() -> None:
    app = AppTest.from_function(_unknown_result_app).run(timeout=30)
    assert not app.exception
    labels = set(_metric_map(app))
    assert sum(label.startswith(project_page.SCREENING_NPV_LABEL) for label in labels) == 4
    assert not any(label.startswith(project_page.LIFECYCLE_NPV_LABEL) for label in labels)
    warnings = [warning.value for warning in app.warning]
    assert any(LIFECYCLE_UNKNOWN_STATUS in value for value in warnings)
    assert any(LIFECYCLE_UNKNOWN_MESSAGE in value for value in warnings)
    assert any("not recorded or displayed as EUR 0" in value for value in warnings)
    assert len(app.dataframe) == 1


def test_partial_coverage_is_a_visible_warning_not_only_a_caption() -> None:
    app = AppTest.from_function(_partial_result_app).run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 8
    assert any(
        "audited valid days (1/2)" in warning.value
        and "Missing 1" in warning.value
        for warning in app.warning
    )


def test_cashflow_tables_are_exact_runresult_rows_and_disclose_p50_basis() -> None:
    app = AppTest.from_function(_available_result_app).run(timeout=30)
    assert not app.exception
    assert len(app.dataframe) == 2
    expected_columns = [
        "year",
        "revenue_eur",
        "opex_eur",
        "augmentation_eur",
        "terminal_eur",
        "net_eur",
        "discount_factor",
        "discounted_net_eur",
    ]
    screening = app.dataframe[0].value
    screening = screening.data if hasattr(screening, "data") else screening
    lifecycle = app.dataframe[1].value
    lifecycle = lifecycle.data if hasattr(lifecycle, "data") else lifecycle
    assert list(screening.columns) == expected_columns
    assert list(lifecycle.columns) == expected_columns
    assert screening.iloc[0].to_dict() == _table(
        "screening", lifecycle=False
    ).rows[0].to_payload()
    assert lifecycle.iloc[0].to_dict() == _table(
        "lifecycle", lifecycle=True
    ).rows[0].to_payload()
    assert any(
        "linear P50 annual bootstrap draw" in caption.value
        and "not expected-value tables" in caption.value
        for caption in app.caption
    )


def test_panel_cache_survives_rerun_and_invalidates_before_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"emit": 0, "compute": 0}
    real_compute = compute_project_case

    def fake_emit(*args, **kwargs):
        calls["emit"] += 1
        return fx.da_only_srr()

    def counted_compute(case):
        calls["compute"] += 1
        return real_compute(case)

    monkeypatch.setattr(project_page, "emit_da_only", fake_emit)
    monkeypatch.setattr(project_page, "compute_project_case", counted_compute)
    app = AppTest.from_function(_panel_app).run(timeout=30)
    assert not app.exception
    assert calls == {"emit": 0, "compute": 0}

    app.button(key="pc_run").click().run(timeout=30)
    assert not app.exception
    assert calls == {"emit": 1, "compute": 1}
    assert len(app.metric) == 4  # UNKNOWN: screening only

    app.run(timeout=30)
    assert not app.exception
    assert calls == {"emit": 1, "compute": 1}
    assert len(app.metric) == 4

    app.text_input(key="pc_bootstrap_seed").set_value("1").run(timeout=30)
    assert not app.exception
    assert calls == {"emit": 1, "compute": 1}
    assert len(app.metric) == 0
    assert any("stale result is hidden" in warning.value for warning in app.warning)

    app.button(key="pc_run").click().run(timeout=30)
    assert not app.exception
    assert calls == {"emit": 2, "compute": 2}
    assert len(app.metric) == 4


def test_request_fingerprint_covers_source_data_lifecycle_and_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0]},
        index=pd.date_range("2026-03-10", periods=2, freq="h", tz="UTC"),
    )
    asset = AssetCase(10.0, 2.0, 0.88, 1000.0, 10.0)
    lifecycle = LifecycleCase(
        2,
        CapacityMaintenanceBasis.UNKNOWN,
        None,
        None,
        (),
        0.0,
        0.0,
    )
    projection = Projection(ProjectionKind.FlatRealProjection)
    valuation = ValuationCase(0.08, 2026)
    bootstrap = BootstrapCase(0, 1000, BOOTSTRAP_ALGORITHM_V1)
    kwargs = dict(
        primary_df=frame,
        primary_zone="DE_LU",
        start_date=dt.date(2026, 3, 10),
        end_date=dt.date(2026, 3, 11),
        asset=asset,
        lifecycle=lifecycle,
        projection=projection,
        valuation=valuation,
        bootstrap=bootstrap,
        capture_rate=0.9,
        strategy=project_page._StrategySelection(
            project_page._DA_ONLY_LABEL,
            project_page.ProducerAdapterId.PC_ADP_DA_ONLY,
        ),
    )
    baseline = project_page._request_fingerprint(**kwargs)
    changed_frame = frame.copy()
    changed_frame.iloc[0, 0] = 999.0
    assert project_page._request_fingerprint(
        **{**kwargs, "primary_df": changed_frame}
    ) != baseline
    changed_lifecycle = LifecycleCase(
        3,
        CapacityMaintenanceBasis.UNKNOWN,
        None,
        None,
        (),
        0.0,
        0.0,
    )
    assert project_page._request_fingerprint(
        **{**kwargs, "lifecycle": changed_lifecycle}
    ) != baseline
    monkeypatch.setattr(project_page, "PC_A_CALCULATOR_VERSION", "pc-a-test-next")
    assert project_page._request_fingerprint(**kwargs) != baseline


def test_active_maintenance_basis_renders_both_npv_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(project_page, "emit_da_only", lambda *_a, **_k: fx.da_only_srr())
    app = AppTest.from_function(_panel_app).run(timeout=30)
    app.selectbox(key="pc_capacity_maintenance_basis").set_value(
        "No augmentation required — engineering assertion"
    ).run(timeout=30)
    app.text_input(key="pc_maintenance_source").set_value(
        "engineering capacity memo"
    ).run(timeout=30)
    app.button(key="pc_run").click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 8


def test_all_four_typed_strategies_are_reachable_when_inputs_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"triple": 0}

    def fake_triple(*_args, **_kwargs):
        calls["triple"] += 1
        return fx.da_id_reserve_srr()

    monkeypatch.setattr(project_page, "emit_da_id_reserve", fake_triple)
    app = AppTest.from_function(_multi_strategy_panel_app).run(timeout=30)
    strategy = app.selectbox(key="pc_strategy")
    assert strategy.options == [
        project_page._DA_ONLY_LABEL,
        project_page._DA_ID_LABEL,
        project_page._DA_RESERVE_LABEL,
        project_page._DA_ID_RESERVE_LABEL,
    ]
    strategy.set_value(project_page._DA_ID_RESERVE_LABEL).run(timeout=30)
    assert not any(item.key == "pc_projection_kind" for item in app.selectbox)
    app.button(key="pc_run").click().run(timeout=30)
    assert not app.exception
    assert calls == {"triple": 1}
    assert len(app.metric) == 4


def test_reserve_streams_exclude_wrong_zone_instead_of_relabelling_it() -> None:
    reserve = fx.reserve_series("DE_LU", [(fx.D1, 6)])
    ancillary = pd.DataFrame(
        {
            "product_type": ["FCR"] * len(reserve),
            "direction": ["symmetric"] * len(reserve),
            "zone": ["FI"] * len(reserve),
            "capacity_price_eur_mw": reserve.to_numpy(),
        },
        index=reserve.index,
    )
    streams = project_page._resolve_reserve_streams(
        primary_zone="DE_LU",
        ancillary_df=ancillary,
        capacity_df=None,
        capacity_sources=None,
    )
    assert streams == ()


def test_unbound_or_foreign_capacity_parameter_cannot_be_relabelled() -> None:
    reserve = fx.reserve_series("DE_LU", [(fx.D1, 6)])
    unbound = pd.DataFrame(
        {
            "product_type": ["FCR"] * len(reserve),
            "direction": ["symmetric"] * len(reserve),
            "capacity_price_eur_mw": reserve.to_numpy(),
        },
        index=reserve.index,
    )
    for capacity_df in (unbound, unbound.assign(zone="FI")):
        assert project_page._resolve_reserve_streams(
            primary_zone="DE_LU",
            ancillary_df=None,
            capacity_df=capacity_df,
            capacity_sources=None,
        ) == ()


def test_reserve_streams_keep_same_product_up_and_down_separate() -> None:
    index = pd.date_range("2026-03-10", periods=2, freq="4h", tz="UTC")
    ancillary = pd.DataFrame(
        {
            "product_type": ["aFRR", "aFRR", "aFRR", "aFRR"],
            "direction": ["up", "up", "down", "down"],
            "zone": ["DE_LU"] * 4,
            "capacity_price_eur_mw": [10.0, 11.0, 20.0, 21.0],
        },
        index=index.append(index),
    )
    streams = project_page._resolve_reserve_streams(
        primary_zone="DE_LU",
        ancillary_df=ancillary,
        capacity_df=None,
        capacity_sources=None,
    )
    assert [(stream.product, stream.direction) for stream in streams] == [
        ("aFRR", "down"),
        ("aFRR", "up"),
    ]
    by_direction = {stream.direction: stream.series.tolist() for stream in streams}
    assert by_direction == {"down": [20.0, 21.0], "up": [10.0, 11.0]}


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_reserve_streams_preserve_bad_day_for_pc_a_date_audit(
    bad_value: float,
) -> None:
    reserve = fx.reserve_series("DE_LU", [(fx.D1, 6), (fx.D2, 6)])
    ancillary = pd.DataFrame(
        {
            "product_type": ["FCR"] * len(reserve),
            "direction": ["symmetric"] * len(reserve),
            "zone": ["DE_LU"] * len(reserve),
            "capacity_price_eur_mw": reserve.to_numpy(),
        },
        index=reserve.index,
    )
    ancillary.iloc[0, ancillary.columns.get_loc("capacity_price_eur_mw")] = bad_value

    streams = project_page._resolve_reserve_streams(
        primary_zone="DE_LU",
        ancillary_df=ancillary,
        capacity_df=None,
        capacity_sources=None,
    )

    assert len(streams) == 1
    assert len(streams[0].series) == len(reserve)
    assert pd.isna(streams[0].series.iloc[0]) or not math.isfinite(
        float(streams[0].series.iloc[0])
    )
    audit = build_reserve_coverage_audit(
        streams[0].series,
        zone="DE_LU",
        evaluation_dates=(fx.D1, fx.D2),
    )
    assert audit.covered_dates == frozenset({fx.D2})


def test_reserve_streams_preserve_duplicate_for_pc_a_date_audit() -> None:
    reserve = fx.reserve_series("DE_LU", [(fx.D1, 6), (fx.D2, 6)])
    ancillary = pd.DataFrame(
        {
            "product_type": ["FCR"] * len(reserve),
            "direction": ["symmetric"] * len(reserve),
            "zone": ["DE_LU"] * len(reserve),
            "capacity_price_eur_mw": reserve.to_numpy(),
        },
        index=reserve.index,
    )
    duplicate = ancillary.iloc[[0]].copy()
    ancillary = pd.concat([ancillary, duplicate])

    streams = project_page._resolve_reserve_streams(
        primary_zone="DE_LU",
        ancillary_df=ancillary,
        capacity_df=None,
        capacity_sources=None,
    )

    assert len(streams) == 1
    assert streams[0].series.index.has_duplicates
    audit = build_reserve_coverage_audit(
        streams[0].series,
        zone="DE_LU",
        evaluation_dates=(fx.D1, fx.D2),
    )
    assert audit.covered_dates == frozenset({fx.D2})


def test_unified_capacity_cache_exposes_reserve_strategy_and_source(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import data_ingestion as di

    monkeypatch.setattr(di, "DB_PATH", tmp_path / "bess.db")
    reserve = fx.reserve_series("DE_LU", [(fx.D1, 6), (fx.D2, 6)])
    cached = pd.DataFrame(
        {
            "product_type": ["FCR"] * len(reserve),
            "direction": ["symmetric"] * len(reserve),
            "zone": ["DE_LU"] * len(reserve),
            "capacity_price_eur_mw": reserve.to_numpy(),
        },
        index=reserve.index,
    )
    di.write_capacity_cache(cached, "DE_LU", source=di.CAPACITY_SOURCE_MANUAL)

    app = AppTest.from_function(_cached_reserve_panel_app).run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="pc_strategy").options == [
        project_page._DA_ONLY_LABEL,
        project_page._DA_RESERVE_LABEL,
    ]
    app.selectbox(key="pc_strategy").set_value(
        project_page._DA_RESERVE_LABEL
    ).run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="pc_reserve_product").options == [
        "FCR — symmetric"
    ]
    assert any(
        "Unified capacity cache / Manual CSV" in caption.value
        for caption in app.caption
    )


@pytest.mark.slow
def test_real_public_adapter_reaches_project_case_ui() -> None:
    app = AppTest.from_function(_real_panel_app).run(timeout=30)
    assert not app.exception
    app.button(key="pc_run").click().run(timeout=120)
    assert not app.exception
    metrics = _metric_map(app)
    assert len(metrics) == 4
    assert all(
        label.startswith(project_page.SCREENING_NPV_LABEL) for label in metrics
    )
    assert any("pre-tax unlevered, not bankable" in c.value for c in app.caption)
