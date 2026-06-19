"""Tests for the dispatch-strategy comparison table."""

from __future__ import annotations

import math

import pytest

from src.pages.simulation_cockpit import _fmt_strategy_bar_label
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
