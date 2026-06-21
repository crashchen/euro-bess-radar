"""Tests for the reserve capacity-price forecast skill diagnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pages.simulation_cockpit import _reserve_history_for_product
from src.reserve_forecast import (
    RESERVE_VALUE_COL,
    SKILL_BY_BLOCK_COLUMNS,
    compute_reserve_forecast_skill,
)


def _history(values_per_hour: list[float], days: int, *, start: str = "2025-02-03"):
    """Build a UTC-indexed capacity-price frame repeating a 24h shape."""
    idx = pd.date_range(start, periods=24 * days, freq="h", tz="UTC")
    series = np.tile(np.asarray(values_per_hour, dtype=float), days)
    df = pd.DataFrame({RESERVE_VALUE_COL: series}, index=idx)
    df.index.name = "timestamp"
    return df


# A single 4h block (block 2 = hours 08-11) priced high, everything else 0.
_BLOCK_SHAPE = [0.0] * 8 + [100.0] * 4 + [0.0] * 12


def test_block_climatology_beats_flat_mean_on_structured_prices() -> None:
    # Three identical block-structured days (tz=None so UTC blocks align).
    skill = compute_reserve_forecast_skill(_history(_BLOCK_SHAPE, 3), tz=None)
    assert skill["n_points"] == 72
    assert skill["n_blocks_requested"] == 6
    # LOO of identical days reproduces each block mean exactly -> ~0 error,
    # so the climatology fully beats the flat sample mean.
    assert skill["mae"] == pytest.approx(0.0, abs=1e-9)
    assert skill["skill_vs_mean"] == pytest.approx(1.0)
    assert list(skill["by_block"].columns) == SKILL_BY_BLOCK_COLUMNS


def test_flat_prices_yield_none_skill() -> None:
    # Perfectly flat -> naive MAE is 0 -> skill undefined (None), not a blowup.
    skill = compute_reserve_forecast_skill(_history([12.0] * 24, 3), tz=None)
    assert skill["mae"] == pytest.approx(0.0, abs=1e-9)
    assert skill["skill_vs_mean"] is None
    assert skill["naive_mean_mae"] == pytest.approx(0.0, abs=1e-9)


def test_unstructured_noise_has_near_zero_skill() -> None:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-02-03", periods=24 * 8, freq="h", tz="UTC")
    df = pd.DataFrame(
        {RESERVE_VALUE_COL: 20.0 + rng.normal(0, 3.0, 24 * 8)}, index=idx,
    )
    skill = compute_reserve_forecast_skill(df, tz=None)
    # No block structure -> block climatology no better than a flat mean.
    assert skill["skill_vs_mean"] is not None
    assert abs(skill["skill_vs_mean"]) < 0.15


def test_walk_forward_drops_first_day() -> None:
    loo = compute_reserve_forecast_skill(_history(_BLOCK_SHAPE, 3), tz=None)
    wf = compute_reserve_forecast_skill(
        _history(_BLOCK_SHAPE, 3), tz=None, forecast_mode="walk_forward",
    )
    # Walk-forward cannot forecast the first day (no prior history).
    assert wf["n_points"] == loo["n_points"] - 24


def test_day_missing_entire_block_does_not_crash() -> None:
    # Sparse reserve data: one day missing all of one 4h block. The
    # pre-aggregated unstack path leaves NaN for that (date, block) cell, which
    # must be filled (not crash on the int cast) and still score other blocks.
    df = _history(_BLOCK_SHAPE, 4)  # 2025-02-03 .. 02-06, tz=None -> UTC blocks
    day2 = pd.DatetimeIndex(df.index).normalize() == pd.Timestamp(
        "2025-02-04", tz="UTC",
    )
    drop = day2 & pd.DatetimeIndex(df.index).hour.isin([8, 9, 10, 11])  # block 2
    df = df[~np.asarray(drop)]
    skill = compute_reserve_forecast_skill(df, tz=None)
    assert skill["n_points"] > 0
    assert skill["skill_vs_mean"] is not None
    # Block 2 is still backed by the other three days.
    assert skill["n_blocks_filled"] == 6


def test_coverage_and_block_count_reported() -> None:
    skill = compute_reserve_forecast_skill(_history(_BLOCK_SHAPE, 3), tz=None)
    assert skill["coverage"] == pytest.approx(1.0)
    assert skill["n_blocks_filled"] == 6
    assert len(skill["by_block"]) == 6


def test_empty_and_missing_column_return_empty_skill() -> None:
    assert compute_reserve_forecast_skill(pd.DataFrame())["n_points"] == 0
    no_col = pd.DataFrame({"other": [1.0]}, index=pd.to_datetime(["2025-02-03"], utc=True))
    assert compute_reserve_forecast_skill(no_col)["n_points"] == 0


def test_invalid_forecast_mode_raises() -> None:
    with pytest.raises(ValueError, match="forecast_mode"):
        compute_reserve_forecast_skill(_history(_BLOCK_SHAPE, 2), forecast_mode="bogus")


def test_reserve_history_for_product_filters_to_one_product() -> None:
    idx = pd.date_range("2025-02-03", periods=3, freq="h", tz="UTC")
    anc = pd.DataFrame(
        {
            "product_type": ["FCR", "aFRR Up", "FCR"],
            RESERVE_VALUE_COL: [10.0, 20.0, 12.0],
            "energy_price_eur_mwh": [np.nan, np.nan, np.nan],
        },
        index=idx,
    )
    hist = _reserve_history_for_product(anc, "FCR")
    assert list(hist.columns) == [RESERVE_VALUE_COL]
    assert list(hist[RESERVE_VALUE_COL]) == [10.0, 12.0]
    assert _reserve_history_for_product(anc, "missing").empty
    assert _reserve_history_for_product(None, "FCR").empty
