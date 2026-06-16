"""Tests for interval-level simulation cockpit outputs."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest

from src.simulation import (
    _count_dispatch_blocks,
    available_local_dates,
    empty_simulation_result,
    simulate_da_milp_replay,
)


def _make_prices() -> pd.DataFrame:
    idx = pd.date_range("2026-03-19", periods=24, freq="h", tz="UTC")
    prices = [20.0] * 8 + [90.0] * 8 + [25.0] * 8
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    return df


def test_empty_or_missing_price_day_returns_safe_result() -> None:
    empty = empty_simulation_result("empty")
    assert empty["summary"]["total_revenue_eur"] == 0.0
    assert empty["timeseries"].empty

    prices = _make_prices()
    prices.loc[prices.index[3], "price_eur_mwh"] = np.nan
    result = simulate_da_milp_replay(
        prices,
        simulation_date=prices.index[0].date(),
        power_mw=1.0,
        duration_hours=2.0,
    )
    assert result["timeseries"].empty
    assert "missing" in result["summary"]["reason"].lower()


def test_available_local_dates_respects_timezone() -> None:
    prices = _make_prices()
    dates = available_local_dates(prices, tz="Europe/Berlin")
    assert dates == [pd.Timestamp("2026-03-19").date(), pd.Timestamp("2026-03-20").date()]


def test_da_replay_preserves_soc_bounds_and_terminal_soc() -> None:
    prices = _make_prices()
    result = simulate_da_milp_replay(
        prices,
        simulation_date=prices.index[0].date(),
        power_mw=1.0,
        duration_hours=2.0,
        efficiency=0.88,
    )
    ts = result["timeseries"]
    capacity = 2.0
    assert not ts.empty
    assert ts["soc_mwh"].min() >= -1e-6
    assert ts["soc_mwh"].max() <= capacity + 1e-6
    assert ts["soc_mwh"].iloc[-1] == pytest.approx(capacity * 0.5, abs=1e-5)


def test_net_dispatch_and_revenue_cumulative_identity() -> None:
    prices = _make_prices()
    result = simulate_da_milp_replay(
        prices,
        simulation_date=prices.index[0].date(),
        power_mw=1.0,
        duration_hours=2.0,
        capture_rate=0.70,
    )
    ts = result["timeseries"]
    assert np.allclose(
        ts["net_dispatch_mw"],
        ts["p_discharge_mw"] - ts["p_charge_mw"],
    )
    assert ts["cumulative_revenue_eur"].iloc[-1] == pytest.approx(
        ts["interval_revenue_eur"].sum(),
    )
    assert result["summary"]["total_revenue_eur"] == pytest.approx(
        ts["interval_revenue_eur"].sum(),
    )


def test_throughput_and_fce_match_dispatch_arrays() -> None:
    prices = _make_prices()
    result = simulate_da_milp_replay(
        prices,
        simulation_date=prices.index[0].date(),
        power_mw=1.0,
        duration_hours=2.0,
    )
    ts = result["timeseries"]
    expected_throughput = float((ts["p_charge_mw"] + ts["p_discharge_mw"]).sum())
    expected_fce = float(ts["p_discharge_mw"].sum() / 2.0)

    assert result["summary"]["physical_throughput_mwh"] == pytest.approx(expected_throughput)
    assert result["summary"]["traded_volume_mwh"] == pytest.approx(expected_throughput)
    assert result["summary"]["daily_fce"] == pytest.approx(expected_fce, abs=1e-4)
    assert result["summary"]["rebalancing_factor"] == pytest.approx(1.0)


def test_trade_count_uses_contiguous_nonzero_blocks() -> None:
    net = np.array([0.0, -1.0, -1.0, 0.0, 2.0, 2.0, -0.5, 0.0, -0.5])
    assert _count_dispatch_blocks(net) == 4


def test_degradation_cost_zero_safe_when_capex_zero() -> None:
    prices = _make_prices()
    result = simulate_da_milp_replay(
        prices,
        simulation_date=prices.index[0].date(),
        power_mw=1.0,
        duration_hours=2.0,
        capex_eur_kwh=0.0,
    )
    assert result["summary"]["degradation_cost_eur"] == 0.0
    assert result["summary"]["soh_delta_pct"] <= 0.0


def test_simulation_page_and_app_parse() -> None:
    for path in ("app.py", "src/pages/simulation_cockpit.py"):
        with open(path, encoding="utf-8") as handle:
            ast.parse(handle.read())
