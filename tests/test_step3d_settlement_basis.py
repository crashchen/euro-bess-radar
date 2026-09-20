"""Display-only settlement scope plus unchanged DST cash known answers.

New helper imports stay inside cases so the same file collects on the baseline.
The real-solver controls establish the pre-existing 456/437/475 difference; they
must pass on both versions. All inputs are synthetic and no cache is read.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta

import pandas as pd
import pytest

from src.project_case import (
    AdapterUnavailableError,
    MarketCase,
    Projection,
    ProjectionKind,
    compute_project_case,
    emit_da_id_reserve,
    emit_reserve_coopt,
    grid,
)
from tests import pc_case_fixtures as fx
from tests.test_project_case_adapters import _regelleistung_block_series


@pytest.mark.parametrize(
    "zone, product, expected",
    [("DE_LU", "FCR", "Physical delivery hours"),
     ("FI", "FCR-N", "Physical delivery hours"),
     ("GB", "custom capacity", "Physical delivery hours"),
     ("UNREGISTERED", "FCR", "Unverified"),
     (None, "FCR", "Unverified"),
     ("DE_LU", None, None), ("DE_LU", "", None), ("DE_LU", "   ", None)],
)
def test_compact_table_label_preserves_full_disclosure_scope(zone, product, expected):
    from src.settlement_disclosure import (
        screening_capacity_settlement_basis,
        screening_capacity_settlement_label,
    )

    assert screening_capacity_settlement_label(zone, product) == expected
    full_text = screening_capacity_settlement_basis(zone, product)
    if expected == "Physical delivery hours":
        assert "physical-hour screening" in full_text
    elif expected == "Unverified":
        assert "basis unverified" in full_text
    else:
        assert full_text is None


@pytest.mark.parametrize("product", [None, "", "   "])
def test_screening_without_capacity_product_has_no_capacity_disclosure(product):
    from src.settlement_disclosure import screening_capacity_settlement_basis

    assert screening_capacity_settlement_basis("DE_LU", product) is None


@pytest.mark.parametrize("product", ["FCR", "aFRR Up", "mFRR Down", "custom product"])
def test_screening_describes_actual_selected_product_without_product_whitelist(product):
    from src.settlement_disclosure import screening_capacity_settlement_basis

    text = screening_capacity_settlement_basis("DE_LU", product)
    assert f"DE_LU / {product}" in text
    assert "physical-hour screening" in text
    assert "hours and availability" in text
    assert "no nominal-block DST adjustment" in text
    assert "registered DE_LU Project Case profile" in text
    assert "Energy/SoC use physical time" in text


@pytest.mark.parametrize("zone", ["FI", "FR", "GB"])
def test_other_screening_zones_do_not_inherit_german_product_calendar(zone):
    from src.settlement_disclosure import screening_capacity_settlement_basis

    text = screening_capacity_settlement_basis(zone, "FCR")
    assert f"{zone} / FCR" in text
    assert "physical-hour screening" in text
    assert "six nominal 4h" not in text
    assert "Project Case" not in text


@pytest.mark.parametrize("zone", [None, "", "UNREGISTERED"])
def test_unknown_screening_zone_is_explicitly_unverified(zone):
    from src.settlement_disclosure import screening_capacity_settlement_basis

    text = screening_capacity_settlement_basis(zone, "FCR")
    assert "basis unverified" in text
    assert "physical-hour screening" not in text


def test_project_case_uses_frozen_run_scope_and_preserves_payload_and_fingerprint():
    from src.settlement_disclosure import project_case_capacity_settlement_basis

    case = replace(fx.project_case(), market_case=MarketCase(
        fx.da_id_reserve_srr(), Projection(ProjectionKind.FlatRealProjection)
    ))
    result = compute_project_case(case)
    before = result.to_payload()
    fingerprint = result.input_fingerprint
    text = project_case_capacity_settlement_basis(result)
    assert "DE_LU / aFRR" in text
    assert "nominal 4h capacity blocks" in text
    assert "six nominal 4h capacity blocks per local day, including DST" in text
    assert "Energy and SoC use physical delivery time" in text
    assert result.to_payload() == before
    assert result.input_fingerprint == fingerprint


def test_da_only_project_case_has_no_reserve_settlement_disclosure():
    from src.settlement_disclosure import project_case_capacity_settlement_basis

    assert project_case_capacity_settlement_basis(
        compute_project_case(fx.project_case())
    ) is None


@pytest.mark.parametrize("adapter", ["PC_ADP_RESERVE_COOPT", "PC_ADP_DA_ID_RESERVE"])
@pytest.mark.parametrize("product", ["FCR [symmetric]", "mFRR [down]"])
def test_recorded_reserve_profile_controls_disclosure_not_product_spelling(adapter, product):
    from src.settlement_disclosure import project_case_strategy_capacity_settlement_basis

    strategy = fx.da_id_reserve_srr().to_payload()
    strategy["adapter_provenance"]["producer_adapter_id"] = adapter
    strategy["reserve_product"] = product
    text = project_case_strategy_capacity_settlement_basis(strategy)
    assert f"DE_LU / {product}" in text
    assert "nominal 4h capacity blocks" in text


@pytest.mark.parametrize(
    "changed_field, value",
    [
        ("zone", "FI"),
        ("reserve_product", None),
        ("producer_adapter_id", "PC_ADP_FUTURE_RESERVE"),
        ("expected_grid_registry_version", "pc-market-grid-v2"),
        ("reserve_profile", "pc-reserve-hourly-v2"),
        ("reserve_profile", None),
    ],
)
def test_unknown_recorded_reserve_contract_remains_visible_without_false_claim(changed_field, value):
    from src.settlement_disclosure import project_case_strategy_capacity_settlement_basis

    # A payload projection represents a future presentation input; it does not
    # forge a valid RunResult or bypass the separate schema/export validation.
    strategy = deepcopy(fx.da_id_reserve_srr().to_payload())
    if changed_field in {"zone", "reserve_product"}:
        strategy[changed_field] = value
    elif changed_field == "reserve_profile":
        strategy["adapter_provenance"]["expected_grid_profiles"]["reserve"] = value
    else:
        strategy["adapter_provenance"][changed_field] = value
    text = project_case_strategy_capacity_settlement_basis(strategy)
    assert "basis unverified" in text
    assert "nominal 4h capacity blocks" not in text


@pytest.mark.slow
@pytest.mark.parametrize(
    "target, physical_hours, screening_cash",
    [(date(2025, 3, 29), 24, 456.0), (date(2025, 3, 30), 23, 437.0),
     (date(2025, 10, 26), 25, 475.0)],
)
def test_existing_same_source_coopt_cash_contract_is_preserved(target, physical_hours, screening_cash):
    from src.dispatch import solve_joint_capacity_batch
    from src.pages.simulation_cockpit import _reserve_coopt_total
    from src.time_utils import infer_delivery_interval_hours, interval_hours_vector

    index = pd.DatetimeIndex(grid.expected_da_timestamps("DE_LU", target))
    da = pd.DataFrame({"price_eur_mwh": 0.0}, index=index)
    reserve = _regelleistung_block_series(target)
    assert reserve.tolist() == [20.0] * 6  # Six EUR 80/MW / nominal 4h blocks.
    ancillary = pd.DataFrame({"capacity_price_eur_mw": reserve, "product_type": "FCR"})
    common = dict(power_mw=1.0, duration_hours=1.0, efficiency=0.88)
    project = emit_reserve_coopt(
        da, reserve, zone="DE_LU", first_delivery_date=target,
        last_delivery_date=target, currency_basis=fx.CURRENCY_SOURCE,
        reserve_product="FCR", reserve_source="synthetic Regelleistung",
        availability=0.95, **common,
    )
    cockpit, _, price = _reserve_coopt_total(
        da, "FCR", ancillary, valid_dates={target}, tz="Europe/Berlin", **common,
    )
    # Revenue Estimation calls this same non-overlay batch. Keep its physical
    # capacity component separate from DA cash rather than testing just totals.
    joint = solve_joint_capacity_batch(da, price, tz="Europe/Berlin", **common)
    durations = interval_hours_vector(infer_delivery_interval_hours(index), len(index))
    assert float(durations.sum()) == physical_hours
    assert sum(h for _, h in grid.reserve_blocks("DE_LU", target)) == 24
    assert dict(project.daily_realised_cash_series)[target] == pytest.approx(456.0)
    assert cockpit == pytest.approx(screening_cash)
    assert joint.iloc[0]["joint_capacity_revenue"] == pytest.approx(screening_cash)
    assert joint.iloc[0]["joint_da_revenue"] == pytest.approx(0.0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "target, screening_cash",
    [(date(2026, 3, 28), 456.0), (date(2026, 3, 29), 437.0),
     (date(2025, 10, 26), 475.0)],
)
def test_existing_same_source_sequential_and_ceiling_cash_contract_is_preserved(target, screening_cash):
    from src.simulation import (
        simulate_da_id_reserve_ceiling_batch,
        simulate_sequential_da_id_reserve_batch,
    )

    days = [target - timedelta(days=2), target - timedelta(days=1), target]
    da = pd.DataFrame({"price_eur_mwh": 0.0}, index=pd.DatetimeIndex([
        ts for day in days for ts in grid.expected_da_timestamps("DE_LU", day)
    ]))
    ida = pd.DataFrame({"intraday_price_eur_mwh": 0.0}, index=pd.DatetimeIndex([
        ts for day in days for ts in grid.expected_ida_timestamps("DE_LU", day)
    ]))
    reserve = pd.concat([_regelleistung_block_series(day) for day in days])
    common = dict(power_mw=1.0, duration_hours=1.0, efficiency=0.88, availability=0.95)
    project = emit_da_id_reserve(
        da, ida, reserve, zone="DE_LU", first_delivery_date=target,
        last_delivery_date=target, currency_basis=fx.CURRENCY_SOURCE,
        reserve_product="FCR", reserve_source="synthetic Regelleistung",
        bucket="hour_of_day", **common,
    )
    sequential, _ = simulate_sequential_da_id_reserve_batch(
        da, ida, reserve, dates=[target], tz="Europe/Berlin", bucket="hour_of_day", **common,
    )
    ceiling = simulate_da_id_reserve_ceiling_batch(
        da, ida, reserve, dates=[target], tz="Europe/Berlin", **common,
    )
    assert dict(project.daily_realised_cash_series)[target] == pytest.approx(456.0)
    assert sequential.iloc[0]["realised_eur"] == pytest.approx(screening_cash)
    assert ceiling["total_eur"] == pytest.approx(screening_cash)


def test_finnish_project_case_does_not_gain_a_reserve_profile():
    assert grid.reserve_profile_id("FI") is None
    assert grid.reserve_blocks("FI", date(2026, 3, 29)) is None


@pytest.mark.slow
def test_pre_cutover_spring_triple_retains_mismatched_grid_rejection():
    target = date(2025, 3, 30)
    days = [target - timedelta(days=2), target - timedelta(days=1), target]
    da = fx.da_frame("DE_LU", days)
    ida = fx.ida_frame("DE_LU", days)
    reserve = pd.concat([_regelleistung_block_series(day) for day in days])
    with pytest.raises(AdapterUnavailableError, match="no valid dates"):
        emit_da_id_reserve(
            da, ida, reserve, zone="DE_LU", first_delivery_date=target,
            last_delivery_date=target, currency_basis=fx.CURRENCY_SOURCE,
            reserve_product="FCR", reserve_source="synthetic Regelleistung",
            bucket="hour_of_day", power_mw=1.0, duration_hours=1.0, efficiency=0.88,
        )
