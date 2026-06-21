"""Tests for the dispatch-strategy comparison table."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.pages.simulation_cockpit import (
    _fmt_strategy_bar_label,
    _reserve_coopt_total,
    _slice_to_local_dates,
)
from src.simulation import DAYS_PER_YEAR
from src.strategy_compare import (
    STRATEGY_COMPARE_COLUMNS,
    build_strategy_comparison,
)


def _summary(da: float, realised: float, ceiling: float, valid_days: int) -> dict:
    return {
        "total_da_only_eur": da,
        "total_realised_eur": realised,
        "total_ceiling_eur": ceiling,
        "valid_days": valid_days,
    }


def test_three_strategy_rows_with_da_baseline_zero_uplift() -> None:
    table = build_strategy_comparison(
        _summary(100.0, 130.0, 160.0, valid_days=10), power_mw=2.0,
    )
    assert list(table.columns) == STRATEGY_COMPARE_COLUMNS
    assert list(table["strategy"]) == [
        "DA-only",
        "DA + IDA1 (forecast-driven)",
        "DA + IDA1 (perfect-foresight ceiling)",
    ]
    assert table.iloc[0]["uplift_vs_da_pct"] == 0.0


def test_annualisation_and_uplift_math() -> None:
    table = build_strategy_comparison(
        _summary(100.0, 130.0, 160.0, valid_days=10), power_mw=2.0,
    )
    # annualised EUR/MW = total * 365.25/valid_days / power
    expected_da = 100.0 * DAYS_PER_YEAR / 10 / 2.0
    assert table.iloc[0]["annualized_eur_per_mw"] == pytest.approx(expected_da)
    # forecast-driven uplift vs DA-only: (130-100)/100 = 30%
    assert table.iloc[1]["uplift_vs_da_pct"] == pytest.approx(30.0)
    assert table.iloc[2]["uplift_vs_da_pct"] == pytest.approx(60.0)


def test_empty_when_no_valid_days() -> None:
    table = build_strategy_comparison(
        _summary(0.0, 0.0, 0.0, valid_days=0), power_mw=1.0,
    )
    assert table.empty
    assert list(table.columns) == STRATEGY_COMPARE_COLUMNS


def test_zero_power_yields_nan_annualisation_not_crash() -> None:
    table = build_strategy_comparison(
        _summary(100.0, 130.0, 160.0, valid_days=5), power_mw=0.0,
    )
    assert all(math.isnan(v) for v in table["annualized_eur_per_mw"])
    # Uplift is power-independent and still computed.
    assert table.iloc[1]["uplift_vs_da_pct"] == pytest.approx(30.0)


def test_uplift_nan_when_da_baseline_zero() -> None:
    table = build_strategy_comparison(
        _summary(0.0, 25.0, 40.0, valid_days=5), power_mw=1.0,
    )
    assert math.isnan(table.iloc[1]["uplift_vs_da_pct"])


def test_strategy_bar_label_hides_non_finite_values() -> None:
    assert _fmt_strategy_bar_label(1234.4) == "1,234"
    assert _fmt_strategy_bar_label(float("nan")) == "N/A"
    assert _fmt_strategy_bar_label(float("inf")) == "N/A"


def test_reserve_row_appended_when_total_provided() -> None:
    table = build_strategy_comparison(
        _summary(100.0, 130.0, 160.0, valid_days=10),
        power_mw=2.0,
        reserve_coopt_total=145.0,
        reserve_label="DA + FCR co-opt (headroom)",
    )
    assert len(table) == 4
    assert table.iloc[3]["strategy"] == "DA + FCR co-opt (headroom)"
    # Same annualisation + uplift convention as the other rows.
    assert table.iloc[3]["annualized_eur_per_mw"] == pytest.approx(
        145.0 * DAYS_PER_YEAR / 10 / 2.0,
    )
    assert table.iloc[3]["uplift_vs_da_pct"] == pytest.approx(45.0)


def test_reserve_row_default_label_when_none() -> None:
    table = build_strategy_comparison(
        _summary(100.0, 130.0, 160.0, valid_days=10),
        power_mw=2.0,
        reserve_coopt_total=120.0,
    )
    assert len(table) == 4
    assert table.iloc[3]["strategy"] == "DA + reserve co-opt (headroom)"


def test_reserve_row_absent_when_total_none() -> None:
    table = build_strategy_comparison(
        _summary(100.0, 130.0, 160.0, valid_days=10), power_mw=2.0,
    )
    assert len(table) == 3


def test_reserve_row_dropped_when_total_non_finite() -> None:
    for bad in (float("nan"), float("inf")):
        table = build_strategy_comparison(
            _summary(100.0, 130.0, 160.0, valid_days=10),
            power_mw=2.0,
            reserve_coopt_total=bad,
        )
        assert len(table) == 3


def test_triple_joint_row_appended_with_default_label() -> None:
    table = build_strategy_comparison(
        _summary(100.0, 130.0, 160.0, valid_days=10),
        power_mw=2.0,
        triple_joint_total=175.0,
    )
    assert len(table) == 4  # 3 base + triple (no reserve here)
    assert table.iloc[3]["strategy"] == "DA + IDA1 + reserve (co-opt ceiling)"
    assert table.iloc[3]["uplift_vs_da_pct"] == pytest.approx(75.0)


def test_five_row_layout_with_reserve_and_triple() -> None:
    table = build_strategy_comparison(
        _summary(100.0, 130.0, 160.0, valid_days=10),
        power_mw=2.0,
        reserve_coopt_total=120.0,
        reserve_label="DA + FCR co-opt (headroom)",
        triple_joint_total=175.0,
        triple_joint_label="DA + IDA1 + FCR (co-opt ceiling)",
    )
    assert list(table["strategy"]) == [
        "DA-only",
        "DA + IDA1 (forecast-driven)",
        "DA + IDA1 (perfect-foresight ceiling)",
        "DA + FCR co-opt (headroom)",
        "DA + IDA1 + FCR (co-opt ceiling)",
    ]


def test_triple_joint_row_dropped_when_non_finite() -> None:
    for bad in (float("nan"), float("inf")):
        table = build_strategy_comparison(
            _summary(100.0, 130.0, 160.0, valid_days=10),
            power_mw=2.0,
            triple_joint_total=bad,
        )
        assert len(table) == 3


def _price_frame(days: int = 4) -> pd.DataFrame:
    idx = pd.date_range("2025-06-01 00:00", periods=24 * days, freq="h", tz="UTC")
    prices = 50 + 40 * np.sin(np.arange(24 * days) / 24 * 2 * np.pi)
    return pd.DataFrame({"price_eur_mwh": prices}, index=idx)


def _capacity_anc() -> pd.DataFrame:
    """FCR capacity rows: three inside the June 2-3 window at 10 EUR/MW/h plus a
    sharp out-of-window May print that must NOT leak into the windowed price."""
    idx = pd.to_datetime([
        "2025-05-15T12:00:00Z",   # out of window
        "2025-06-02T10:00:00Z",   # in window (Berlin June 2)
        "2025-06-02T11:00:00Z",   # in window
        "2025-06-03T10:00:00Z",   # in window (Berlin June 3)
    ], utc=True)
    return pd.DataFrame({
        "product_type": ["FCR"] * 4,
        "capacity_price_eur_mw": [1000.0, 10.0, 10.0, 10.0],
        "energy_price_eur_mwh": [np.nan] * 4,
    }, index=idx)


def _berlin_dates(df: pd.DataFrame) -> list:
    return sorted(set(df.index.tz_convert("Europe/Berlin").date))


def test_slice_to_local_dates_filters_by_local_day() -> None:
    df = _price_frame(days=4)
    keep = set(_berlin_dates(df)[1:3])  # two interior complete days
    out = _slice_to_local_dates(df, keep, "Europe/Berlin")
    got = set(pd.DatetimeIndex(out.index).tz_convert("Europe/Berlin").date)
    assert got == keep


def test_reserve_coopt_total_sums_joint_batch_over_valid_dates() -> None:
    df = _price_frame(days=4)
    valid = set(_berlin_dates(df)[1:3])
    total, label, price = _reserve_coopt_total(
        df, "FCR", _capacity_anc(), valid_dates=valid, tz="Europe/Berlin",
        power_mw=1.0, duration_hours=2, efficiency=0.88,
    )
    assert total is not None and total > 0.0
    assert label == "DA + FCR co-opt (headroom)"
    # In-window FCR rows are all 10 EUR/MW/h; the out-of-window 1000 is excluded.
    assert price == pytest.approx(10.0)


def test_reserve_coopt_total_ignores_out_of_window_capacity_price() -> None:
    df = _price_frame(days=4)
    valid = set(_berlin_dates(df)[1:3])  # June 2-3 only
    # _capacity_anc carries an out-of-window May print at 1000 EUR/MW/h.
    _, _, price = _reserve_coopt_total(
        df, "FCR", _capacity_anc(), valid_dates=valid, tz="Europe/Berlin",
        power_mw=1.0, duration_hours=2, efficiency=0.88,
    )
    # Priced only from the in-window 10 EUR/MW/h rows, not pulled toward 1000.
    assert price == pytest.approx(10.0)


def test_reserve_coopt_total_omitted_when_no_in_window_capacity() -> None:
    df = _price_frame(days=4)
    valid = set(_berlin_dates(df)[1:3])
    may_only = pd.DataFrame(
        {
            "product_type": ["FCR"],
            "capacity_price_eur_mw": [1000.0],
            "energy_price_eur_mwh": [np.nan],
        },
        index=pd.to_datetime(["2025-05-15T12:00:00Z"], utc=True),
    )
    # No capacity rows overlap the window -> row omitted, not full-sample priced.
    assert _reserve_coopt_total(
        df, "FCR", may_only, valid_dates=valid, tz="Europe/Berlin",
        power_mw=1.0, duration_hours=2, efficiency=0.88,
    ) == (None, None, None)


def test_reserve_coopt_total_guards_return_none() -> None:
    df = _price_frame(days=4)
    valid = set(_berlin_dates(df)[1:3])
    common = dict(tz="Europe/Berlin", power_mw=1.0, duration_hours=2, efficiency=0.88)
    # No product selected.
    assert _reserve_coopt_total(df, None, _capacity_anc(), valid_dates=valid, **common) == (
        None, None, None,
    )
    # Empty valid-date set.
    assert _reserve_coopt_total(df, "FCR", _capacity_anc(), valid_dates=set(), **common) == (
        None, None, None,
    )
    # Product carries no capacity price.
    energy_only = pd.DataFrame({
        "product_type": ["aFRR Up"],
        "capacity_price_eur_mw": [np.nan],
        "energy_price_eur_mwh": [5.0],
    })
    assert _reserve_coopt_total(
        df, "aFRR Up", energy_only, valid_dates=valid, **common,
    ) == (None, None, None)
