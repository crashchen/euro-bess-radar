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
    simulate_da_id_replay,
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


def test_da_replay_default_capture_rate_is_raw_lp() -> None:
    prices = _make_prices()
    raw = simulate_da_milp_replay(
        prices,
        simulation_date=prices.index[0].date(),
        power_mw=1.0,
        duration_hours=2.0,
    )
    derated = simulate_da_milp_replay(
        prices,
        simulation_date=prices.index[0].date(),
        power_mw=1.0,
        duration_hours=2.0,
        capture_rate=0.70,
    )
    assert raw["summary"]["total_revenue_eur"] > 0
    assert derated["summary"]["total_revenue_eur"] == pytest.approx(
        raw["summary"]["total_revenue_eur"] * 0.70,
        rel=1e-6,
    )


def _make_da_id_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2026-03-19", periods=24, freq="h", tz="UTC")
    da = [20.0] * 8 + [90.0] * 8 + [25.0] * 8
    ida = [25.0] * 8 + [110.0] * 8 + [22.0] * 8
    da_df = pd.DataFrame({"price_eur_mwh": da}, index=idx)
    ida_df = pd.DataFrame({"intraday_price_eur_mwh": ida}, index=idx)
    da_df.index.name = "timestamp"
    ida_df.index.name = "timestamp"
    return da_df, ida_df


def test_da_id_replay_traded_volume_exceeds_physical_when_rebid() -> None:
    da, ida = _make_da_id_pair()
    result = simulate_da_id_replay(
        da, ida,
        simulation_date=da.index[0].date(),
        power_mw=1.0, duration_hours=2.0,
    )
    summary = result["summary"]
    assert summary["physical_throughput_mwh"] > 0
    assert summary["traded_volume_mwh"] >= summary["physical_throughput_mwh"]
    assert summary["rebalancing_factor"] >= 1.0


def test_da_id_replay_exposes_wholesales_columns() -> None:
    da, ida = _make_da_id_pair()
    ts = simulate_da_id_replay(
        da, ida,
        simulation_date=da.index[0].date(),
        power_mw=1.0, duration_hours=2.0,
    )["timeseries"]
    for col in ("da_position_mw", "ida_position_mw", "rebid_delta_mw",
                "intraday_price_eur_mwh"):
        assert col in ts.columns, f"missing column {col}"
    assert np.allclose(
        ts["rebid_delta_mw"],
        ts["ida_position_mw"] - ts["da_position_mw"],
    )


def test_da_id_replay_full_unwind_reports_inf_rebalancing_factor() -> None:
    """Edge case Codex flagged: DA traded but IDA fully unwound (physical=0).

    Old code returned 0.0 (divide-by-zero guard), which is the OPPOSITE
    operational signal. New code returns float('inf') and the UI formats
    it as 'Inf'.
    """
    import math
    idx = pd.date_range("2026-03-19", periods=2, freq="h", tz="UTC")
    da = pd.DataFrame({"price_eur_mwh": [0.0, 100.0]}, index=idx)
    da.index.name = "timestamp"
    # IDA flat at midpoint -> stage-2 has no profitable trade, unwinds.
    ida = pd.DataFrame({"intraday_price_eur_mwh": [50.0, 50.0]}, index=idx)
    ida.index.name = "timestamp"
    result = simulate_da_id_replay(
        da, ida,
        simulation_date=idx[0].date(),
        power_mw=1.0, duration_hours=1.0, efficiency=1.0,
    )
    summary = result["summary"]
    assert summary["physical_throughput_mwh"] == pytest.approx(0.0, abs=1e-6)
    assert summary["traded_volume_mwh"] > 0
    assert math.isinf(summary["rebalancing_factor"])


def test_da_id_replay_no_ida_data_returns_safe_empty() -> None:
    da, _ = _make_da_id_pair()
    empty_ida = pd.DataFrame(columns=["intraday_price_eur_mwh"])
    result = simulate_da_id_replay(
        da, empty_ida,
        simulation_date=da.index[0].date(),
        power_mw=1.0, duration_hours=2.0,
    )
    assert result["timeseries"].empty
    assert "ida1" in result["summary"]["reason"].lower()
