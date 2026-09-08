"""Physical-duration known answers for the local-day-internal SDAC cutover."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    calculate_daily_dispatch,
    calculate_dispatch_price_vwaps,
    calculate_negative_price_hours,
)
from src.config import ZONE_TIMEZONES
from src.dispatch import solve_daily_joint_capacity_lp, solve_daily_lp, solve_joint_capacity_batch
from src.project_case.grid import expected_da_timestamps
from src.simulation import (
    build_dispatch_event_table,
    simulate_da_milp_replay,
    simulate_replay_batch,
)

ZONES = ["BG", "EE", "FI", "GR", "LT", "LV", "PT", "RO"]


def market_day(zone, *, flat=False):
    day = date(2025, 9, 30) if zone == "PT" else date(2025, 10, 1)
    index = pd.DatetimeIndex(expected_da_timestamps(zone, day))
    cutover = pd.Timestamp("2025-09-30T22:00Z")
    values = np.full(len(index), 50.0) if flat else np.where(index < cutover, 0.0, 100.0)
    return day, pd.DataFrame({"price_eur_mwh": values}, index=index)


@pytest.mark.parametrize("zone", ZONES)
def test_joint_capacity_preserves_24_physical_hours(zone):
    _, prices = market_day(zone, flat=True)
    result = solve_joint_capacity_batch(prices, 5.0, tz=ZONE_TIMEZONES[zone], efficiency=1.0)
    assert result.attrs["model_available"] is True
    assert len(result) == 1
    assert result.iloc[0]["joint_capacity_revenue"] == pytest.approx(114.0)
    assert result.iloc[0]["joint_da_revenue"] == 0.0


@pytest.mark.parametrize("zone", ZONES)
def test_daily_greedy_and_milp_use_native_delivery_duration(zone):
    _, prices = market_day(zone)
    result = calculate_daily_dispatch(prices, tz=ZONE_TIMEZONES[zone], efficiency=1.0)
    assert len(result) == 1
    assert result.iloc[0]["spread"] == pytest.approx(100.0)
    assert result.iloc[0]["lp_revenue"] == pytest.approx(99.0)
    assert result.iloc[0]["n_cycles"] == pytest.approx(1.0)


@pytest.mark.parametrize("zone", ZONES)
def test_single_replay_retains_mixed_day_energy_vwap_and_event_hours(zone):
    day, prices = market_day(zone)
    result = simulate_da_milp_replay(
        prices, simulation_date=day, tz=ZONE_TIMEZONES[zone],
        efficiency=1.0, soc_init_frac=0.0,
    )
    summary, ts = result["summary"], result["timeseries"]
    assert summary["success"] is True
    assert len(ts) == len(prices)
    assert summary["total_revenue_eur"] == pytest.approx(99.0)
    assert summary["physical_throughput_mwh"] == pytest.approx(2.0)
    assert summary["daily_fce"] == pytest.approx(1.0)
    assert summary["charge_vwap_eur_mwh"] == pytest.approx(0.0)
    assert summary["discharge_vwap_eur_mwh"] == pytest.approx(100.0)
    assert ts["interval_hours"].sum() == pytest.approx(24.0)
    assert ts["soc_mwh"].between(-1e-7, 1.0 + 1e-7).all()
    events = build_dispatch_event_table(ts)
    assert events["energy_mwh"].sum() == pytest.approx(2.0)
    assert events["duration_h"].sum() == pytest.approx(2.0)
    for row in events.itertuples():
        assert (row.end_time - row.start_time).total_seconds() / 3600 == pytest.approx(row.duration_h)


@pytest.mark.parametrize("zone", ZONES)
@pytest.mark.parametrize("carry", [False, True])
def test_batch_replay_retains_mixed_day(zone, carry):
    day, prices = market_day(zone)
    result = simulate_replay_batch(
        prices, tz=ZONE_TIMEZONES[zone], dates=[day], carry_soc=carry,
        efficiency=1.0, soc_init_frac=0.0,
    )
    assert result.attrs["model_available"] is True
    assert result.attrs["valid_days"] == 1
    assert result.attrs["excluded_days_due_to_missing"] == 0
    assert result.iloc[0]["total_revenue_eur"] == pytest.approx(99.0)
    assert result.iloc[0]["physical_throughput_mwh"] == pytest.approx(2.0)


@pytest.mark.parametrize("zone", ZONES)
def test_negative_hours_are_physical_even_on_utc_index(zone):
    _, prices = market_day(zone, flat=True)
    prices["price_eur_mwh"] = -50.0
    assert calculate_negative_price_hours(prices)["negative_hours"] == 24.0


@pytest.mark.parametrize("solver", [solve_daily_lp, solve_daily_joint_capacity_lp])
def test_duration_vector_known_answer_and_soc_balance(solver):
    prices = np.array([0.0, 100.0, 100.0, 100.0, 100.0])
    dt = np.array([1.0, 0.25, 0.25, 0.25, 0.25])
    kwargs = {"capacity_price_eur_mw_h": 0.0} if solver is solve_daily_joint_capacity_lp else {}
    result = solver(prices, dt, efficiency=1.0, soc_init_frac=0.0, **kwargs)
    assert result["success"] is True
    revenue = result["total_revenue_eur"] if kwargs else result["revenue_eur"]
    assert revenue == pytest.approx(99.0)
    assert np.dot(result["p_discharge"], dt) == pytest.approx(1.0)
    assert result["n_cycles"] == pytest.approx(1.0)
    np.testing.assert_allclose(np.diff(result["soc"]), (result["p_charge"] - result["p_discharge"]) * dt)


@pytest.mark.parametrize("solver", [solve_daily_lp, solve_daily_joint_capacity_lp])
@pytest.mark.parametrize("dt", [0.0, -0.25, np.inf, np.nan, [1.0], [[1.0, 1.0]], [1.0, 0.0]])
def test_bad_duration_returns_typed_failure_before_optimisation(solver, dt, monkeypatch):
    import src.dispatch as dispatch

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid duration reached linprog")

    monkeypatch.setattr(dispatch, "linprog", unexpected)
    kwargs = {"capacity_price_eur_mw_h": 5.0} if solver is solve_daily_joint_capacity_lp else {}
    result = solver(np.array([0.0, 100.0]), dt, **kwargs)
    assert result["success"] is False
    assert result["status"] == "invalid_input"
    assert "duration" in result["message"].lower()


def test_duration_vector_cycle_cap_and_canonical_energy():
    dt = np.array([1.0, 0.25, 0.25, 0.25, 0.25])
    result = solve_daily_lp(
        np.array([0.0, 100.0, 100.0, 100.0, 100.0]), dt,
        efficiency=1.0, soc_init_frac=0.0, max_efc_per_day=0.5,
        min_throughput_tiebreak=True,
    )
    assert result["success"] is True
    assert result["tiebreak_applied"] is True
    assert result["revenue_eur"] == pytest.approx(49.5, abs=1e-6)
    assert np.dot(result["p_discharge"], dt) == pytest.approx(0.5)


def test_vwap_weights_each_interval_energy():
    result = calculate_dispatch_price_vwaps(
        np.array([0.0, 100.0]), np.array([1.0, 1.0]), np.array([0.0, 0.0]),
        dt_hours=np.array([1.0, 0.25]),
    )
    assert result["charge_energy_mwh"] == 1.25
    assert result["charge_vwap_eur_mwh"] == 20.0


@pytest.mark.parametrize("mutation", ["first", "last", "cutover", "hole", "duplicate", "offset"])
def test_mixed_day_gaps_are_not_stretched_into_delivery_intervals(mutation):
    day, prices = market_day("FI")
    if mutation == "duplicate":
        prices = pd.concat([prices, prices.iloc[[1]]]).sort_index()
    elif mutation == "offset":
        prices = prices.shift(freq="5min")
    else:
        pos = {"first": 0, "last": -1, "cutover": 1, "hole": 20}[mutation]
        prices = prices.drop(prices.index[pos])
    result = simulate_da_milp_replay(prices, simulation_date=day, tz="Europe/Helsinki")
    assert result["summary"]["success"] is False
    assert result["summary"]["status"] == "invalid_input"


@pytest.mark.parametrize("zone", ["FI", "PT"])
def test_cycle_frontier_keeps_common_mixed_day_and_physical_fec(zone):
    from src.cycle_frontier import compute_cycle_cap_frontier

    day, prices = market_day(zone)
    frame, summary = compute_cycle_cap_frontier(
        prices, dates=[day], tz=ZONE_TIMEZONES[zone], power_mw=1.0,
        duration_hours=4.0, efficiency=1.0, capex_eur_kwh=0.0,
        cycle_caps=[0.125, None],
    )
    assert summary["valid_days"] == 1
    np.testing.assert_allclose(frame["gross_eur"], [49.5, 99.0], atol=1e-6)
    np.testing.assert_allclose(frame["avg_efc_per_day"], [0.125, 0.25], atol=1e-8)
    np.testing.assert_allclose(frame["discharge_vwap_eur_mwh"], [100.0, 100.0])


@pytest.mark.parametrize("zone", ["FI", "PT"])
def test_continuous_replay_slices_duration_vectors_back_to_each_day(zone):
    middle, prices = market_day(zone)
    dates = [(pd.Timestamp(middle) + pd.Timedelta(days=offset)).date() for offset in [-1, 0, 1]]
    frames = [
        prices if day == middle else pd.DataFrame(
            {"price_eur_mwh": 0.0}, index=pd.DatetimeIndex(expected_da_timestamps(zone, day)),
        )
        for day in dates
    ]
    result = simulate_replay_batch(
        pd.concat(frames), dates=dates, tz=ZONE_TIMEZONES[zone],
        efficiency=1.0, soc_init_frac=0.0,
    )
    assert result.attrs["valid_days"] == 3
    assert result["n_intervals"].tolist() == [len(frame) for frame in frames]
    assert result["total_revenue_eur"].sum() == pytest.approx(99.0)
    assert result["physical_throughput_mwh"].sum() == pytest.approx(2.0)


@pytest.mark.parametrize("zone", ["FI", "PT"])
def test_project_case_da_accepts_existing_registered_mixed_day(zone):
    from src.project_case import CurrencyBasis, CurrencyBasisMode, emit_da_only

    day, prices = market_day(zone)
    result = emit_da_only(
        prices, zone=zone, first_delivery_date=day, last_delivery_date=day,
        power_mw=1.0, duration_hours=4.0, efficiency=1.0,
        currency_basis=CurrencyBasis(CurrencyBasisMode.SOURCE_EUR_TREATED_AS_BASE_YEAR_REAL, 2025),
    )
    assert result.coverage_audit.valid_dates == (day,)
    assert dict(result.daily_realised_cash_series)[day] == pytest.approx(99.0)
    assert len(result.fingerprint()) == 64


def test_joint_vector_reserve_average_and_cash_are_duration_weighted():
    result = solve_daily_joint_capacity_lp(
        np.array([0.0, 100.0]), np.array([1.0, 0.25]), 10.0,
        efficiency=1.0, soc_init_frac=0.0,
    )
    assert result["success"] is True
    assert result["da_revenue_eur"] == pytest.approx(24.75)
    assert result["capacity_revenue_eur"] == pytest.approx(7.125)
    assert result["avg_reserve_mw"] == pytest.approx(0.6)


def test_unknown_grid_does_not_fabricate_negative_price_hours():
    prices = pd.DataFrame(
        {"price_eur_mwh": -50.0},
        index=pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T01:00Z", "2026-01-01T03:00Z"]),
    )
    result = calculate_negative_price_hours(prices)
    assert result["negative_intervals"] == 3
    assert np.isnan(result["negative_hours"])
