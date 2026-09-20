"""Stored reserve settlement basis follows the displayed Project Case result.

The synthetic reserve result is producer-issued by the real public co-optimising
adapter on a spring DST day. Nothing reads or writes a production price cache.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.project_case import (
    AssetCase,
    BootstrapCase,
    CapacityMaintenanceBasis,
    CurrencyBasis,
    CurrencyBasisMode,
    LifecycleCase,
    MarketCase,
    ProjectCase,
    Projection,
    ProjectionKind,
    RunResult,
    ValuationCase,
    compute_project_case,
    emit_reserve_coopt,
    grid,
)
from src.project_case.enums import BOOTSTRAP_ALGORITHM_V1

DAY = dt.date(2026, 3, 29)


def synthetic_reserve_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zero DA prices and six EUR 20/MW/h German capacity product blocks."""
    da = pd.DataFrame(
        {"price_eur_mwh": 0.0},
        index=pd.DatetimeIndex(grid.expected_da_timestamps("DE_LU", DAY)),
    )
    capacity = pd.DataFrame(
        {
            "capacity_price_eur_mw": 20.0,
            "product_type": "FCR",
            "direction": "symmetric",
            "zone": "DE_LU",
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp(block) for block, _ in grid.reserve_blocks("DE_LU", DAY)]
        ),
    )
    return da, capacity


@lru_cache(maxsize=4)
def synthetic_reserve_result(product: str = "FCR [symmetric]") -> RunResult:
    """A real solver-backed RunResult also used by the local review harness."""
    da, capacity = synthetic_reserve_inputs()
    strategy = emit_reserve_coopt(
        da,
        capacity["capacity_price_eur_mw"],
        zone="DE_LU",
        first_delivery_date=DAY,
        last_delivery_date=DAY,
        power_mw=1.0,
        duration_hours=1.0,
        efficiency=0.88,
        currency_basis=CurrencyBasis(
            CurrencyBasisMode.SOURCE_EUR_TREATED_AS_BASE_YEAR_REAL, 2026
        ),
        reserve_product=product,
        reserve_source="Synthetic local Regelleistung-shaped capacity blocks",
        availability=0.95,
    )
    case = ProjectCase(
        AssetCase(1.0, 1.0, 0.88, 100_000.0, 1_000.0),
        LifecycleCase(
            3,
            CapacityMaintenanceBasis.NO_AUGMENTATION_REQUIRED_ASSERTED,
            "Synthetic review fixture",
            "2026-09-20",
            (),
            0.0,
            0.0,
        ),
        MarketCase(strategy, Projection(ProjectionKind.FlatRealProjection)),
        ValuationCase(0.08, 2026),
        BootstrapCase(0, 1000, BOOTSTRAP_ALGORITHM_V1),
    )
    return compute_project_case(case)


def _result_app(product: str, compact: bool) -> None:
    import streamlit as st

    from src.pages.project_case import render_project_case_result
    from tests.test_step3d_project_case_disclosure import synthetic_reserve_result

    # Ambient controls are intentionally unrelated to the stored result.
    st.selectbox("Current zone", ["FI", "DE_LU"])
    st.selectbox("Current reserve product", ["FCR-N", "aFRR down"])
    render_project_case_result(synthetic_reserve_result(product), compact=compact)


def _da_only_app(compact: bool) -> None:
    from src.pages.project_case import render_project_case_result
    from tests.test_project_case_ui import _available_result

    render_project_case_result(_available_result(), compact=compact)


def _mirror_app(populated: bool) -> None:
    import streamlit as st

    from src.pages.project_case import (
        _CACHE_KEY,
        ProjectCaseRunCache,
        render_project_case_cockpit_mirror,
    )
    from tests.test_step3d_project_case_disclosure import synthetic_reserve_result

    if populated:
        result = synthetic_reserve_result()
        st.session_state[_CACHE_KEY] = ProjectCaseRunCache(
            "synthetic request", result.input_fingerprint, result
        )
    render_project_case_cockpit_mirror()


def _panel_and_mirror_app() -> None:
    from src.pages.project_case import (
        render_project_case_cockpit_mirror,
        render_project_case_panel,
    )
    from tests.test_step3d_project_case_disclosure import DAY, synthetic_reserve_inputs

    da, capacity = synthetic_reserve_inputs()
    render_project_case_panel(
        primary_zone="DE_LU",
        primary_df=da,
        start_date=DAY,
        end_date=DAY,
        power_mw=1.0,
        duration_hours=1.0,
        efficiency=0.88,
        capture_rate=0.9,
        capex_eur_kwh=100.0,
        capacity_df=capacity,
    )
    render_project_case_cockpit_mirror()


def _capacity_captions(app: AppTest) -> list[str]:
    return [
        caption.value
        for caption in app.caption
        if "capacity settlement" in caption.value.lower()
    ]


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("product", ["FCR [symmetric]", "aFRR [up]"])
def test_result_discloses_its_stored_nominal_capacity_basis(product, compact):
    app = AppTest.from_function(_result_app, args=(product, compact)).run(timeout=60)
    assert not app.exception
    captions = _capacity_captions(app)
    assert len(captions) == 1
    text = captions[0]
    assert "DE_LU" in text and product in text
    assert "nominal" in text and "4" in text
    assert "DST" in text and "physical" in text
    assert "FCR-N" not in text and "FI" not in text
    before = text
    app.selectbox[0].set_value("DE_LU").run(timeout=60)
    app.selectbox[1].set_value("aFRR down").run(timeout=60)
    assert not app.exception
    assert _capacity_captions(app) == [before]


@pytest.mark.parametrize("compact", [False, True])
def test_da_only_result_has_no_reserve_capacity_disclosure(compact):
    app = AppTest.from_function(_da_only_app, args=(compact,)).run(timeout=60)
    assert not app.exception
    assert not _capacity_captions(app)
    assert len(app.metric) == 8


def test_real_dst_fixture_keeps_nominal_cash_and_provenance():
    result = synthetic_reserve_result()
    strategy = result.provenance["strategy_run_result"]
    assert len(synthetic_reserve_inputs()[0]) == 92  # 23 physical hours
    assert strategy["daily_realised_cash_series"][0][1] == pytest.approx(456.0)
    assert strategy["reserve_product"] == "FCR [symmetric]"
    assert strategy["adapter_provenance"]["expected_grid_profiles"]["reserve"] == (
        "pc-reserve-block-of-day-4h-v1"
    )


@pytest.mark.parametrize("populated", [False, True])
def test_real_cockpit_mirror_uses_only_the_cached_result(populated):
    app = AppTest.from_function(_mirror_app, args=(populated,)).run(timeout=60)
    assert not app.exception
    assert len(_capacity_captions(app)) == int(populated)
    assert len(app.metric) == (8 if populated else 0)
    if not populated:
        assert any("Run Project Case" in item.value for item in app.info)


def test_stale_project_case_hides_basis_and_mirror_without_recomputation(monkeypatch):
    import src.pages.project_case as page

    calls = 0
    real_emit = page.emit_reserve_coopt

    def counted_emit(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_emit(*args, **kwargs)

    monkeypatch.setattr(page, "emit_reserve_coopt", counted_emit)
    app = AppTest.from_function(_panel_and_mirror_app).run(timeout=60)
    app.selectbox(key="pc_strategy").set_value("DA + reserve co-optimised").run(timeout=60)
    assert not app.exception
    assert not _capacity_captions(app)
    app.button(key="pc_run").click().run(timeout=60)
    assert not app.exception
    assert calls == 1
    assert len(_capacity_captions(app)) == 2  # Panel and its read-only mirror.
    app.run(timeout=60)
    assert calls == 1
    assert len(_capacity_captions(app)) == 2
    app.text_input(key="pc_bootstrap_seed").set_value("1").run(timeout=60)
    assert not app.exception
    assert calls == 1
    assert not _capacity_captions(app)
    assert not app.metric
    assert any("stale result is hidden" in item.value for item in app.warning)
    assert any("Run Project Case" in item.value for item in app.info)
