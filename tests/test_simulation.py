"""Tests for interval-level simulation cockpit outputs."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest

from src.simulation import (
    _count_dispatch_blocks,
    available_local_dates,
    build_dispatch_event_table,
    empty_simulation_result,
    simulate_da_id_replay,
    simulate_da_milp_replay,
    simulate_replay_batch,
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


def test_dispatch_event_table_collapses_contiguous_blocks() -> None:
    prices = _make_prices()
    result = simulate_da_milp_replay(
        prices,
        simulation_date=prices.index[0].date(),
        power_mw=1.0,
        duration_hours=2.0,
    )
    ts = result["timeseries"]
    events = build_dispatch_event_table(ts)

    assert not events.empty
    assert set(events["event_type"]).issubset({"Charge", "Discharge"})
    assert events["event_id"].tolist() == list(range(1, len(events) + 1))
    assert events["energy_mwh"].sum() == pytest.approx(
        result["summary"]["physical_throughput_mwh"],
    )


def test_dispatch_event_table_empty_when_no_dispatch() -> None:
    ts = pd.DataFrame({
        "local_time": pd.date_range("2026-03-19", periods=2, freq="h", tz="UTC"),
        "net_dispatch_mw": [0.0, 0.0],
    })
    events = build_dispatch_event_table(ts)
    assert events.empty
    assert "event_type" in events.columns


def test_replay_batch_excludes_days_with_missing_prices() -> None:
    prices = pd.concat([
        _make_prices(),
        _make_prices().shift(freq="1D"),
    ])
    prices.loc[prices.index[-1], "price_eur_mwh"] = np.nan
    dates = available_local_dates(prices, tz="UTC")

    batch = simulate_replay_batch(
        prices,
        dates=dates,
        tz="UTC",
        power_mw=1.0,
        duration_hours=2.0,
    )

    assert len(batch) == 1
    assert batch.attrs["excluded_days"] == 1
    assert batch["date"].iloc[0] == dates[0]


def test_replay_batch_da_id_requires_intraday_data() -> None:
    prices = _make_prices()
    dates = available_local_dates(prices, tz="UTC")
    batch = simulate_replay_batch(
        prices,
        mode="DA + IDA1 Replay",
        intraday_df=None,
        dates=dates,
        tz="UTC",
        power_mw=1.0,
        duration_hours=2.0,
    )

    assert batch.empty
    assert batch.attrs["excluded_days"] == len(dates)


def _make_overnight_arbitrage_prices() -> pd.DataFrame:
    """Two-day prices where overnight SoC carry is strictly profitable.

    Day 1: low all day (20), then a SHORT cheap window at the end (5).
    Day 2: a SHORT expensive spike at the start (200), then low all day.

    Per-day terminal-neutral dispatch caps each day to its internal
    spread (20 -> 20 -> 5 on day 1, 200 -> 20 -> 20 on day 2). A true
    continuous-horizon solve can charge cheap at end of day 1 and
    discharge into the day 2 morning spike, capturing the 5 -> 200 spread
    that per-day reset cannot see.
    """
    idx = pd.date_range("2026-03-19", periods=48, freq="h", tz="UTC")
    day1 = [20.0] * 22 + [5.0, 5.0]
    day2 = [200.0, 200.0] + [20.0] * 22
    prices = day1 + day2
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    return df


def test_replay_batch_carry_soc_produces_higher_revenue_than_reset() -> None:
    """Continuous-horizon dispatch must capture overnight arbitrage.

    Regression guard for f8cb1d9: the original `carry_soc` implementation
    only changed `soc_init_frac` per day, but solve_daily_lp pins terminal
    SoC == initial SoC every day, so carry_soc=True and carry_soc=False
    produced identical revenue. A correct implementation lets day 1 end
    with cheap charge stored and day 2 begin discharging into the spike.
    """
    prices = _make_overnight_arbitrage_prices()
    dates = available_local_dates(prices, tz="UTC")
    assert len(dates) == 2

    chained = simulate_replay_batch(
        prices, dates=dates, tz="UTC",
        power_mw=1.0, duration_hours=2.0, carry_soc=True,
    )
    reset = simulate_replay_batch(
        prices, dates=dates, tz="UTC",
        power_mw=1.0, duration_hours=2.0, carry_soc=False,
    )

    assert chained.attrs["carry_soc"] is True
    assert reset.attrs["carry_soc"] is False
    chained_revenue = float(chained["total_revenue_eur"].sum())
    reset_revenue = float(reset["total_revenue_eur"].sum())
    assert chained_revenue > reset_revenue + 50.0, (
        f"carry_soc=True ({chained_revenue:.2f}) must beat reset "
        f"({reset_revenue:.2f}) on an overnight-arbitrage price path"
    )
    # Day 1 should end with stored energy (above 50% baseline) so day 2
    # can monetise the spike — the smoking gun for true carry-over.
    assert chained.iloc[0]["soc_end_pct"] > 60.0
    assert chained.iloc[1]["soc_start_pct"] == pytest.approx(
        chained.iloc[0]["soc_end_pct"], abs=1e-6,
    )


def test_replay_batch_reset_mode_still_resets_each_day_to_50_pct() -> None:
    prices = _make_overnight_arbitrage_prices()
    dates = available_local_dates(prices, tz="UTC")
    reset = simulate_replay_batch(
        prices, dates=dates, tz="UTC",
        power_mw=1.0, duration_hours=2.0, carry_soc=False,
    )
    assert all(row == pytest.approx(50.0, abs=1e-6) for row in reset["soc_start_pct"])


def test_replay_batch_carry_soc_excluded_day_does_not_advance_soc() -> None:
    """Empty days must not corrupt the carried SoC for the next valid day."""
    prices = _make_overnight_arbitrage_prices()
    prices.loc[prices.index[24:32], "price_eur_mwh"] = np.nan
    dates = available_local_dates(prices, tz="UTC")

    batch = simulate_replay_batch(
        prices, dates=dates, tz="UTC",
        power_mw=1.0, duration_hours=2.0, carry_soc=True,
    )
    assert len(batch) == 1
    assert batch.attrs["excluded_days"] == 1
