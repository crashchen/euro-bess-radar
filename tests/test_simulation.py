"""Tests for interval-level simulation cockpit outputs."""

from __future__ import annotations

import ast
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

import src.simulation as sim
from src.simulation import (
    _count_dispatch_blocks,
    _group_clean_runs,
    align_reserve_price_to_index,
    available_local_dates,
    build_dispatch_event_table,
    empty_simulation_result,
    simulate_da_id_replay,
    simulate_da_id_reserve_ceiling_batch,
    simulate_da_milp_replay,
    simulate_replay_batch,
    simulate_sequential_da_id_batch,
    simulate_sequential_da_id_reserve_batch,
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


def test_da_replay_summary_exposes_physical_da_vwaps() -> None:
    prices = _make_prices()
    result = simulate_da_milp_replay(
        prices,
        simulation_date=prices.index[0].date(),
        power_mw=1.0,
        duration_hours=2.0,
    )
    ts = result["timeseries"]
    charge_energy = ts["p_charge_mw"].sum()
    discharge_energy = ts["p_discharge_mw"].sum()

    assert result["summary"]["market_vwap_available"] is True
    assert result["summary"]["charge_vwap_eur_mwh"] == pytest.approx(
        (ts["price_eur_mwh"] * ts["p_charge_mw"]).sum() / charge_energy
    )
    assert result["summary"]["discharge_vwap_eur_mwh"] == pytest.approx(
        (ts["price_eur_mwh"] * ts["p_discharge_mw"]).sum() / discharge_energy
    )


def test_da_id_replay_does_not_claim_a_single_market_vwap() -> None:
    da_df, ida_df = _make_da_id_pair()
    result = simulate_da_id_replay(
        da_df,
        ida_df,
        simulation_date=da_df.index[0].date(),
        power_mw=1.0,
        duration_hours=2.0,
    )

    assert result["summary"]["market_vwap_available"] is False
    assert math.isnan(result["summary"]["charge_vwap_eur_mwh"])
    assert math.isnan(result["summary"]["discharge_vwap_eur_mwh"])


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


def test_continuous_horizon_excludes_sparse_day_without_nan() -> None:
    """A 23-interval day with no NaN must not be absorbed into a run.

    Without the UTC-regularity guard the continuous-horizon MILP would
    compress the missing interval and silently distort the time axis.
    """
    idx1 = pd.date_range("2026-03-19 00:00", periods=24, freq="h", tz="UTC")
    idx2 = pd.date_range("2026-03-20 00:00", periods=24, freq="h", tz="UTC").delete(2)
    idx = idx1.append(idx2)
    prices = [20.0] * 22 + [5.0, 5.0] + [200.0, 200.0] + [20.0] * 21
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    dates = available_local_dates(df, tz="UTC")

    batch = simulate_replay_batch(
        df, dates=dates, tz="UTC",
        power_mw=1.0, duration_hours=2.0, carry_soc=True,
    )
    assert len(batch) == 1
    assert batch.attrs["excluded_days"] == 1
    assert batch["n_intervals"].tolist() == [24]


def _make_da_id_overnight_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """DA + IDA pair where overnight SoC carry is strictly profitable."""
    idx = pd.date_range("2026-03-19", periods=48, freq="h", tz="UTC")
    da = [20.0] * 22 + [5.0, 5.0] + [200.0, 200.0] + [20.0] * 22
    ida = [22.0] * 22 + [4.0, 4.0] + [210.0, 210.0] + [21.0] * 22
    da_df = pd.DataFrame({"price_eur_mwh": da}, index=idx)
    ida_df = pd.DataFrame({"intraday_price_eur_mwh": ida}, index=idx)
    da_df.index.name = "timestamp"
    ida_df.index.name = "timestamp"
    return da_df, ida_df


def test_da_id_continuous_horizon_beats_per_day_reset() -> None:
    """DA + IDA1 multi-day replay must use continuous horizon, not fallback.

    Regression guard for the DA+ID carry-over work: the batch must report
    carry_mode="continuous_horizon" and capture overnight arbitrage that
    the per-day terminal-neutral reset cannot.
    """
    da_df, ida_df = _make_da_id_overnight_pair()
    dates = available_local_dates(da_df, tz="UTC")
    chained = simulate_replay_batch(
        da_df, mode="DA + IDA1 Replay", intraday_df=ida_df,
        dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0, carry_soc=True,
    )
    reset = simulate_replay_batch(
        da_df, mode="DA + IDA1 Replay", intraday_df=ida_df,
        dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0, carry_soc=False,
    )
    assert chained.attrs["carry_mode"] == "continuous_horizon"
    assert chained.attrs["da_id_carry_soc_supported"] is True
    assert float(chained["total_revenue_eur"].sum()) > float(
        reset["total_revenue_eur"].sum()
    ) + 50.0
    assert chained.iloc[0]["soc_end_pct"] > 60.0
    assert chained.iloc[1]["soc_start_pct"] == pytest.approx(
        chained.iloc[0]["soc_end_pct"], abs=1e-6,
    )


def test_da_id_continuous_total_dominates_independent_days() -> None:
    """Continuous DA+ID total >= sum of independent single-day solves."""
    da_df, ida_df = _make_da_id_pair()
    da_df = pd.concat([da_df, da_df.shift(freq="1D")])
    ida_df = pd.concat([ida_df, ida_df.shift(freq="1D")])
    dates = available_local_dates(da_df, tz="UTC")
    chained = simulate_replay_batch(
        da_df, mode="DA + IDA1 Replay", intraday_df=ida_df,
        dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0, carry_soc=True,
    )
    independent = sum(
        simulate_da_id_replay(
            da_df, ida_df, simulation_date=d, tz="UTC",
            power_mw=1.0, duration_hours=2.0,
        )["summary"]["total_revenue_eur"]
        for d in dates
    )
    assert float(chained["total_revenue_eur"].sum()) >= independent - 1e-6


def test_continuous_horizon_keeps_dst_spring_forward_day() -> None:
    """A 23-local-hour DST day stays uniform in UTC and must be kept."""
    idx = pd.date_range("2026-03-28 00:00", periods=72, freq="h", tz="UTC")
    prices = list(np.linspace(10.0, 100.0, 72))
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    dates = available_local_dates(df, tz="Europe/Berlin")
    # 2026-03-29 is the Berlin spring-forward day (23 local hours).
    assert pd.Timestamp("2026-03-29").date() in dates

    batch = simulate_replay_batch(
        df, dates=dates, tz="Europe/Berlin",
        power_mw=1.0, duration_hours=2.0, carry_soc=True,
    )
    assert batch.attrs["excluded_days"] == 0
    assert batch.attrs["carry_mode"] == "continuous_horizon"


def test_group_clean_runs_splits_long_run_at_interval_cap(monkeypatch) -> None:
    """A contiguous run longer than the interval cap is chunked.

    Performance guard: without the cap a 90-day 15-min run is one giant
    binary MILP that hangs the dashboard. The split must stay contiguous
    (every day kept, in order, each chunk's day_breaks covering its slice)
    so the continuous solvers can seed the next chunk with the prior end
    SoC (a soft terminal-neutral reset at each boundary).
    """
    monkeypatch.setattr(sim, "MAX_CONTINUOUS_REPLAY_INTERVALS", 48)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    days = [(start + pd.Timedelta(days=i)).date() for i in range(5)]

    def build_day(d: date) -> pd.DataFrame:
        idx = pd.date_range(pd.Timestamp(d, tz="UTC"), periods=24, freq="h")
        return pd.DataFrame({"price_eur_mwh": np.arange(24.0)}, index=idx)

    runs = _group_clean_runs(dates=days, build_day=build_day)
    # 5 * 24 = 120 intervals, cap 48 -> [d1,d2], [d3,d4], [d5].
    assert [len(slice_df) for _, slice_df, _ in runs] == [48, 48, 24]
    assert all(len(slice_df) <= 48 for _, slice_df, _ in runs)
    covered = [d for run_dates, _, _ in runs for d in run_dates]
    assert covered == days  # every day kept exactly once, in order
    for _, slice_df, breaks in runs:
        assert breaks[0][0] == 0
        assert breaks[-1][1] == len(slice_df)


def test_chunked_continuous_run_keeps_within_chunk_carry(monkeypatch) -> None:
    """A forced chunk split must not break within-chunk SoC carry-over.

    cap=48 puts the day1->day2 overnight arbitrage inside the first chunk
    and day3 in a second chunk. The intra-chunk carry must still beat
    per-day reset, and the split must keep all days.
    """
    monkeypatch.setattr(sim, "MAX_CONTINUOUS_REPLAY_INTERVALS", 48)
    idx = pd.date_range("2026-03-19", periods=72, freq="h", tz="UTC")
    day1 = [20.0] * 22 + [5.0, 5.0]
    day2 = [200.0, 200.0] + [20.0] * 22
    day3 = [20.0] * 24
    df = pd.DataFrame({"price_eur_mwh": day1 + day2 + day3}, index=idx)
    df.index.name = "timestamp"
    dates = available_local_dates(df, tz="UTC")
    assert len(dates) == 3

    carry = simulate_replay_batch(
        df, dates=dates, tz="UTC",
        power_mw=1.0, duration_hours=2.0, carry_soc=True,
    )
    reset = simulate_replay_batch(
        df, dates=dates, tz="UTC",
        power_mw=1.0, duration_hours=2.0, carry_soc=False,
    )
    assert carry.attrs["carry_mode"] == "continuous_horizon"
    assert carry.attrs["excluded_days"] == 0
    assert len(carry) == 3
    assert float(carry["total_revenue_eur"].sum()) > float(
        reset["total_revenue_eur"].sum()
    ) + 50.0


def _make_seq_history(days: int = 5, *, anomaly_day: int | None = None):
    """DA + IDA over `days` with a fixed daily shape; one IDA day optionally
    inverted to act as an out-of-climatology anomaly the forecast misses."""
    shape = np.array(
        [10, 10, 10, 10, 10, 12, 20, 45, 60, 50, 30, 25,
         22, 28, 45, 72, 92, 80, 55, 40, 30, 20, 15, 12],
        dtype=float,
    )
    idx = pd.date_range("2026-03-16", periods=24 * days, freq="h", tz="UTC")
    da = np.tile(shape, days)
    ida = np.tile(shape, days).astype(float)
    if anomaly_day is not None:
        s = anomaly_day * 24
        ida[s:s + 24] = shape[::-1]
    da_df = pd.DataFrame({"price_eur_mwh": da}, index=idx)
    ida_df = pd.DataFrame({"intraday_price_eur_mwh": ida}, index=idx)
    da_df.index.name = "timestamp"
    ida_df.index.name = "timestamp"
    return da_df, ida_df


def test_sequential_batch_three_way_ordering_and_identity() -> None:
    da_df, ida_df = _make_seq_history(days=5, anomaly_day=2)
    dates = available_local_dates(da_df, tz="UTC")
    per_day, summary = simulate_sequential_da_id_batch(
        da_df, ida_df, dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0,
    )
    assert summary["valid_days"] == 5
    # Per-day identity: captured + forecast_error == ceiling - da_only.
    lhs = per_day["captured_eur"] + per_day["forecast_error_eur"]
    rhs = per_day["ceiling_eur"] - per_day["da_only_eur"]
    assert np.allclose(lhs, rhs, atol=1e-6)
    # Realised never exceeds the perfect-foresight ceiling.
    assert (per_day["realised_eur"] <= per_day["ceiling_eur"] + 1e-6).all()
    # The anomaly day carries most of the achievable uplift the forecast misses.
    anomaly_row = per_day.iloc[2]
    assert anomaly_row["ceiling_eur"] - anomaly_row["da_only_eur"] > 50.0
    assert anomaly_row["forecast_error_eur"] > 50.0


def test_sequential_batch_capture_rate_none_when_no_opportunity() -> None:
    # DA shape == IDA shape every day -> ceiling == da_only, no rebid value,
    # so capture_rate is undefined (None) rather than a divide-by-zero blowup.
    da_df, ida_df = _make_seq_history(days=4, anomaly_day=None)
    dates = available_local_dates(da_df, tz="UTC")
    _, summary = simulate_sequential_da_id_batch(
        da_df, ida_df, dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0,
    )
    assert summary["total_ceiling_uplift_eur"] == pytest.approx(0.0, abs=1e-6)
    assert summary["capture_rate"] is None


def test_da_id_reserve_ceiling_batch_sums_and_dominates() -> None:
    da_df, ida_df = _make_seq_history(days=5, anomaly_day=2)
    dates = available_local_dates(da_df, tz="UTC")
    triple = simulate_da_id_reserve_ceiling_batch(
        da_df, ida_df, 8.0, dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0,
    )
    assert triple["solved_days"] == 5
    # Triple ceiling >= DA+IDA ceiling over the same window (reserve is optional).
    _, seq = simulate_sequential_da_id_batch(
        da_df, ida_df, dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0,
    )
    assert triple["total_eur"] >= seq["total_ceiling_eur"] - 1e-6


def test_align_reserve_price_maps_block_of_day_and_missing_to_zero() -> None:
    idx = pd.date_range("2025-06-01", periods=24, freq="h", tz="UTC")
    # Block 2 (08-11 UTC) priced high, the rest low.
    res = pd.Series([5.0] * 8 + [20.0] * 4 + [5.0] * 12, index=idx)
    out = align_reserve_price_to_index(res, idx, tz=None)
    assert out[8:12].tolist() == [20.0, 20.0, 20.0, 20.0]
    assert out[0:4].tolist() == [5.0, 5.0, 5.0, 5.0]
    # None / empty -> zeros; a target day with no source price -> 0.
    assert align_reserve_price_to_index(None, idx, None).tolist() == [0.0] * 24
    other_day = pd.date_range("2025-07-01", periods=24, freq="h", tz="UTC")
    assert align_reserve_price_to_index(res, other_day, None).tolist() == [0.0] * 24


def test_align_reserve_price_treats_naive_indices_as_utc_for_tz_alignment() -> None:
    target = pd.date_range("2025-06-01", periods=24, freq="h", tz="UTC")
    # Block-start prices in UTC. 06:00 UTC is the 08:00-12:00 local block in
    # Berlin summer time. A naive source must be treated as UTC, not local wall
    # time, otherwise that high block maps to the wrong local 4h bucket.
    block_starts_naive = pd.DatetimeIndex([
        "2025-06-01 02:00", "2025-06-01 06:00", "2025-06-01 10:00",
        "2025-06-01 14:00", "2025-06-01 18:00", "2025-06-01 22:00",
    ])
    values = [5.0, 20.0, 5.0, 5.0, 5.0, 5.0]
    aware_res = pd.Series(values, index=block_starts_naive.tz_localize("UTC"))
    naive_res = pd.Series(values, index=block_starts_naive)

    expected = align_reserve_price_to_index(aware_res, target, tz="Europe/Berlin")
    out = align_reserve_price_to_index(naive_res, target, tz="Europe/Berlin")
    assert out.tolist() == expected.tolist()
    assert out[6:10].tolist() == [20.0, 20.0, 20.0, 20.0]

    naive_target = pd.date_range("2025-06-01", periods=24, freq="h")
    out_naive_target = align_reserve_price_to_index(
        naive_res, naive_target, tz="Europe/Berlin",
    )
    assert out_naive_target.tolist() == expected.tolist()


def test_ceiling_batch_constant_series_equals_scalar() -> None:
    da_df, ida_df = _make_seq_history(days=3, anomaly_day=1)
    dates = available_local_dates(da_df, tz="UTC")
    scalar = simulate_da_id_reserve_ceiling_batch(
        da_df, ida_df, 8.0, dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0,
    )
    const_series = pd.Series(8.0, index=da_df.index)
    series = simulate_da_id_reserve_ceiling_batch(
        da_df, ida_df, const_series, dates=dates, tz="UTC",
        power_mw=1.0, duration_hours=2.0,
    )
    assert series["solved_days"] == scalar["solved_days"]
    assert series["total_eur"] == pytest.approx(scalar["total_eur"], abs=1e-6)


def test_ceiling_batch_per_interval_series_differs_from_scalar_mean() -> None:
    # A block-varying reserve price should not match its flat scalar mean: the
    # joint MILP concentrates reserve in the high-price block.
    da_df, ida_df = _make_seq_history(days=2, anomaly_day=None)
    dates = available_local_dates(da_df, tz="UTC")
    hour = pd.DatetimeIndex(da_df.index).hour
    varying = pd.Series(np.where((hour >= 8) & (hour < 12), 40.0, 4.0), index=da_df.index)
    mean_price = float(varying.mean())
    per_interval = simulate_da_id_reserve_ceiling_batch(
        da_df, ida_df, varying, dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0,
    )
    flat_mean = simulate_da_id_reserve_ceiling_batch(
        da_df, ida_df, mean_price, dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0,
    )
    assert per_interval["total_eur"] != pytest.approx(flat_mean["total_eur"], abs=1.0)


def _block_reserve_series(da_df: pd.DataFrame) -> pd.Series:
    hour = pd.DatetimeIndex(da_df.index).hour
    return pd.Series(np.where((hour >= 8) & (hour < 12), 18.0, 5.0), index=da_df.index)


def test_sequential_reserve_batch_identity_and_ceiling_consistency() -> None:
    da_df, ida_df = _make_seq_history(days=6, anomaly_day=2)
    res = _block_reserve_series(da_df)
    dates = available_local_dates(da_df, tz="UTC")
    per_day, summary = simulate_sequential_da_id_reserve_batch(
        da_df, ida_df, res, dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0,
    )
    # walk-forward (default) excludes the first day (no prior history).
    assert summary["valid_days"] == 5
    assert summary["excluded_days"] == 1
    # Per-day exact attribution + bound.
    lhs = per_day["forecast_effect_eur"] + per_day["timing_cost_eur"]
    assert np.allclose(lhs, per_day["full_gap_eur"], atol=1e-4)
    assert (per_day["realised_eur"] <= per_day["global_ceiling_eur"] + 1e-6).all()
    # CONSISTENCY (the Increment-4 prerequisite): the batch's global ceiling
    # equals the 9.2a ceiling batch with the SAME per-interval reserve series
    # over the SAME valid days. Otherwise the cockpit's 9.2a row would not
    # match the 9.2b row's global-ceiling reference.
    ceil = simulate_da_id_reserve_ceiling_batch(
        da_df, ida_df, res, dates=list(per_day["date"]), tz="UTC",
        power_mw=1.0, duration_hours=2.0,
    )
    assert summary["total_global_ceiling_eur"] == pytest.approx(
        ceil["total_eur"], abs=1e-6,
    )


def test_sequential_reserve_batch_none_reserve_commits_zero() -> None:
    da_df, ida_df = _make_seq_history(days=4, anomaly_day=None)
    dates = available_local_dates(da_df, tz="UTC")
    per_day, _ = simulate_sequential_da_id_reserve_batch(
        da_df, ida_df, None, dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0,
    )
    # No reserve price -> no reserve committed (safe degrade to DA+ID).
    assert (per_day["avg_reserve_mw"] == 0.0).all()


def test_sequential_reserve_batch_empty_dates_returns_empty() -> None:
    da_df, ida_df = _make_seq_history(days=2)
    per_day, summary = simulate_sequential_da_id_reserve_batch(
        da_df, ida_df, None, dates=[], tz="UTC",
    )
    assert per_day.empty
    assert summary["valid_days"] == 0


def test_da_id_reserve_ceiling_batch_skips_days_without_overlap() -> None:
    da_df, ida_df = _make_seq_history(days=3, anomaly_day=None)
    # IDA only covers the first local day; the other two have no overlap.
    ida_partial = ida_df.iloc[:24]
    dates = available_local_dates(da_df, tz="UTC")
    triple = simulate_da_id_reserve_ceiling_batch(
        da_df, ida_partial, 8.0, dates=dates, tz="UTC",
        power_mw=1.0, duration_hours=2.0,
    )
    assert triple["solved_days"] == 1


def test_da_id_reserve_ceiling_batch_invalid_capacity_price_returns_empty() -> None:
    da_df, ida_df = _make_seq_history(days=2, anomaly_day=None)
    dates = available_local_dates(da_df, tz="UTC")
    for bad_capacity_price in (float("nan"), float("inf")):
        triple = simulate_da_id_reserve_ceiling_batch(
            da_df, ida_df, bad_capacity_price, dates=dates, tz="UTC",
            power_mw=1.0, duration_hours=2.0,
        )
        assert triple == {"total_eur": 0.0, "solved_days": 0}


def test_da_id_reserve_ceiling_batch_skips_irregular_merged_day() -> None:
    da_df, ida_df = _make_seq_history(days=2, anomaly_day=None)
    # Dropping one IDA timestamp creates a non-uniform merged UTC index for
    # day 1. The batch must skip it rather than compressing the time axis.
    ida_sparse = ida_df.drop(ida_df.index[5])
    dates = available_local_dates(da_df, tz="UTC")
    triple = simulate_da_id_reserve_ceiling_batch(
        da_df, ida_sparse, 8.0, dates=dates, tz="UTC",
        power_mw=1.0, duration_hours=2.0,
    )
    assert triple["solved_days"] == 1


def test_sequential_batch_missing_intraday_excludes_all_days() -> None:
    da_df, _ = _make_seq_history(days=3)
    empty_ida = pd.DataFrame(columns=["intraday_price_eur_mwh"])
    empty_ida.index = pd.DatetimeIndex([], name="timestamp")
    dates = available_local_dates(da_df, tz="UTC")
    per_day, summary = simulate_sequential_da_id_batch(
        da_df, empty_ida, dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0,
    )
    assert per_day.empty
    assert summary["excluded_days"] == len(dates)
    assert summary["capture_rate"] is None


def _make_seq_edge_history(days: int = 4):
    """DA and IDA with DIFFERENT shapes so a rebid has real value, and an IDA
    history identical day-to-day so the climatology forecast matches the
    realised print (accurate forecast -> clean positive uplift, no churn)."""
    da_shape = np.array([20.0] * 8 + [90.0] * 8 + [25.0] * 8)
    ida_shape = np.array([95.0] * 8 + [20.0] * 8 + [88.0] * 8)
    idx = pd.date_range("2026-03-16", periods=24 * days, freq="h", tz="UTC")
    da_df = pd.DataFrame({"price_eur_mwh": np.tile(da_shape, days)}, index=idx)
    ida_df = pd.DataFrame(
        {"intraday_price_eur_mwh": np.tile(ida_shape, days)}, index=idx,
    )
    da_df.index.name = "timestamp"
    ida_df.index.name = "timestamp"
    return da_df, ida_df


def test_sequential_batch_default_threshold_rebids_when_edge_exists() -> None:
    # With a real forecast edge (DA != IDA, forecast accurate), the default
    # 0.0 deadband rebids every day and captures genuine uplift.
    da_df, ida_df = _make_seq_edge_history(days=4)
    dates = available_local_dates(da_df, tz="UTC")
    per_day, summary = simulate_sequential_da_id_batch(
        da_df, ida_df, dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0,
    )
    assert summary["n_rebid_days"] == summary["valid_days"]
    assert summary["n_hold_days"] == 0
    assert per_day["rebid"].all()
    assert summary["total_captured_eur"] > 0.0


def test_sequential_batch_deadband_holds_all_days_at_high_threshold() -> None:
    # A per-day hurdle no day can clear holds every day on the DA schedule:
    # rebid counts collapse and realised falls back to the DA-only baseline.
    da_df, ida_df = _make_seq_edge_history(days=4)
    dates = available_local_dates(da_df, tz="UTC")
    per_day, summary = simulate_sequential_da_id_batch(
        da_df, ida_df, dates=dates, tz="UTC", power_mw=1.0, duration_hours=2.0,
        min_rebid_uplift_eur=1e9,
    )
    assert summary["n_rebid_days"] == 0
    assert summary["n_hold_days"] == summary["valid_days"]
    assert (~per_day["rebid"]).all()
    assert summary["total_realised_eur"] == pytest.approx(
        summary["total_da_only_eur"], abs=1e-6,
    )
    assert summary["total_captured_eur"] == pytest.approx(0.0, abs=1e-6)
