"""Public-consumer regression tests for the Step 1b delivery-grid guards."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

import src.simulation as sim
from src.analytics import calculate_intraday_uplift

_BATCH_CASES = [
    ("reserve_ceiling", 0.0), ("reserve_ceiling", 5.0),
    ("sequential", 0.0),
    ("sequential_reserve", 0.0), ("sequential_reserve", 5.0),
    ("stochastic", 0.0),
    ("triple", 0.0), ("triple", 5.0),
]


def _market_day(day: str, freq: str, *, ida: bool = False) -> pd.DataFrame:
    start = pd.Timestamp(day, tz="Europe/Berlin")
    index = pd.date_range(start, start + pd.DateOffset(days=1), freq=freq, inclusive="left")
    column = "intraday_price_eur_mwh" if ida else "price_eur_mwh"
    return pd.DataFrame({column: 60.0 if ida else 50.0}, index=index.tz_convert("UTC"))


def _markets(da_freq="h", ida_freq="h"):
    da = pd.concat([_market_day(day, da_freq) for day in ["2025-09-01", "2025-09-02"]])
    ida = pd.concat([
        _market_day(day, ida_freq, ida=True) for day in ["2025-09-01", "2025-09-02"]
    ])
    return da, ida


def _batch(kind, reserve_price, da, ida):
    """Exercise real forecasts, scenarios and solvers through each public API."""
    common = dict(
        dates=[date(2025, 9, 2)], tz="Europe/Berlin",
        power_mw=1.0, duration_hours=1.0, efficiency=1.0,
    )
    reserve = pd.Series(reserve_price, index=da.index) if reserve_price else None
    if kind == "reserve_ceiling":
        return sim.simulate_da_id_reserve_ceiling_batch(da, ida, reserve_price, **common)
    if kind == "sequential":
        result = sim.simulate_sequential_da_id_batch(
            da, ida, forecast_mode="walk_forward", **common,
        )
    elif kind == "sequential_reserve":
        result = sim.simulate_sequential_da_id_reserve_batch(da, ida, reserve, **common)
    elif kind == "stochastic":
        result = sim.simulate_stochastic_da_id_batch(
            da, ida, forecast_mode="walk_forward", n_scenarios=2, seed=7, **common,
        )
    else:
        result = sim.simulate_stochastic_triple_batch(
            da, ida, reserve, n_scenarios=2, seed=7, **common,
        )
    frame, summary = result
    assert len(frame) == summary["valid_days"]
    return summary


@pytest.mark.parametrize("kind,reserve_price", _BATCH_CASES)
@pytest.mark.parametrize("da_freq,ida_freq", [("h", "15min"), ("15min", "h")])
def test_policy_batches_reject_market_grid_loss(kind, reserve_price, da_freq, ida_freq):
    da, ida = _markets(da_freq, ida_freq)
    summary = _batch(kind, reserve_price, da, ida)
    assert summary["valid_days"] == 0
    assert summary["excluded_days_due_to_missing"] == 1
    assert summary["excluded_days_due_to_solver_failure"] == 0
    assert summary["model_available"] is False


@pytest.mark.parametrize("kind,reserve_price", _BATCH_CASES)
@pytest.mark.parametrize("sample", ["complete", "missing", "nan", "quarter_hour"])
def test_policy_batch_same_grid_compatibility(kind, reserve_price, sample):
    """Compatibility controls: these assertions also pass before Step 1b."""
    freq = "15min" if sample == "quarter_hour" else "h"
    da, ida = _markets(freq, freq)
    if sample == "missing":
        ida = ida.drop(ida.index[-12])
    elif sample == "nan":
        ida.iloc[-12, 0] = np.nan
    summary = _batch(kind, reserve_price, da, ida)
    expected_valid = int(sample in {"complete", "quarter_hour"})
    assert summary["valid_days"] == expected_valid
    assert summary["excluded_days_due_to_missing"] == 1 - expected_valid
    assert summary["excluded_days_due_to_solver_failure"] == 0
    assert summary["model_available"] is bool(expected_valid)


@pytest.mark.parametrize("da_freq,ida_freq", [("h", "15min"), ("15min", "h")])
def test_uplift_rejects_incompatible_delivery_cadence(da_freq, ida_freq):
    da, ida = _markets(da_freq, ida_freq)
    result = calculate_intraday_uplift(da, ida, tz="Europe/Berlin")
    assert result["n_periods"] == 0
    assert result["model_available"] is False
    assert result["annual_uplift_per_mw"] == 0.0
    assert "delivery grid" in result["reason"].lower()
    assert sorted([result["da_coverage_pct"], result["ida_coverage_pct"]]) == [25.0, 100.0]


def test_uplift_cannot_hide_one_mismatched_day_inside_a_window():
    da, ida = _markets()
    ida = pd.concat([ida.iloc[:24], _market_day("2025-09-02", "15min", ida=True)])
    result = calculate_intraday_uplift(da, ida, tz="Europe/Berlin")
    assert result["n_periods"] == 0
    assert result["model_available"] is False
    assert result["annual_uplift_per_mw"] == 0.0


@pytest.mark.parametrize("sample", ["duplicate_da", "duplicate_ida", "offset", "singleton"])
def test_uplift_rejects_unverifiable_source_grid(sample):
    da, ida = _markets()
    if sample == "duplicate_da":
        da = pd.concat([da, da.iloc[[-1]]]).sort_index()
    elif sample == "duplicate_ida":
        ida = pd.concat([ida, ida.iloc[[-1]]]).sort_index()
    elif sample == "offset":
        ida = ida.shift(freq="15min")
    else:
        ida = ida.iloc[[24]]
    result = calculate_intraday_uplift(da, ida, tz="Europe/Berlin")
    assert result["n_periods"] == 0
    assert result["model_available"] is False
    assert "delivery grid" in result["reason"].lower()


@pytest.mark.parametrize("day,count", [("2025-03-30", 23), ("2025-10-26", 25)])
def test_uplift_dst_same_grid_compatibility(day, count):
    """Compatibility control: actual UTC hour spacing stays regular across DST."""
    result = calculate_intraday_uplift(
        _market_day(day, "h"), _market_day(day, "h", ida=True), tz="Europe/Berlin",
    )
    assert result["n_periods"] == count
    assert result["coverage_pct"] == 100.0
    assert result["annual_uplift_per_mw"] == pytest.approx(639.19, abs=0.01)


@pytest.mark.parametrize("sample", ["one_missing", "half_window", "sparse_da", "cutover"])
def test_uplift_reports_both_source_coverage_denominators(sample):
    da, ida = _markets()
    if sample == "one_missing":
        ida = ida.drop(ida.index[-12])
        expected = (47, 97.9, 100.0, 47 / 48)
    elif sample == "half_window":
        ida = ida.iloc[:24]
        expected = (24, 50.0, 100.0, 0.5)
    elif sample == "sparse_da":
        da = da.drop(da.index[-12])
        expected = (47, 100.0, 97.9, 47 / 48)
    else:
        da = pd.concat([da.iloc[:24], _market_day("2025-09-02", "15min")])
        ida = pd.concat([ida.iloc[:24], _market_day("2025-09-02", "15min", ida=True)])
        expected = (120, 100.0, 100.0, 1.0)
    result = calculate_intraday_uplift(da, ida, tz="Europe/Berlin")
    count, da_pct, ida_pct, ratio = expected
    assert result["model_available"] is True
    assert result["n_periods"] == count
    assert result["da_coverage_pct"] == pytest.approx(da_pct)
    assert result["ida_coverage_pct"] == pytest.approx(ida_pct)
    assert result["coverage_adjustment_factor"] == pytest.approx(ratio, abs=0.0001)
    assert result["annual_uplift_per_mw"] == pytest.approx(639.1875 * ratio, abs=0.01)


@pytest.mark.parametrize("source", ["da", "ida"])
@pytest.mark.parametrize("position", [0, 12, 23])
def test_uplift_singleton_day_matches_absent_day(source, position):
    """An extra unverified quote must neither veto nor inflate valid days."""
    days = pd.date_range("2025-09-01", periods=5).strftime("%Y-%m-%d")
    da = pd.concat([_market_day(day, "h") for day in days])
    ida = pd.concat([_market_day(day, "h", ida=True) for day in days])
    target = da if source == "da" else ida
    missing_day = target.iloc[48:72]
    sparse = target.drop(missing_day.index)
    lone_quote = missing_day.iloc[[position]].copy()
    lone_quote.iloc[0, 0] = 1_000_000.0
    singleton = pd.concat([sparse, lone_quote]).sort_index()

    def estimate(frame):
        return calculate_intraday_uplift(
            frame if source == "da" else da,
            frame if source == "ida" else ida, tz="Europe/Berlin",
        )

    absent_result = estimate(sparse)
    singleton_result = estimate(singleton)
    for result in [absent_result, singleton_result]:
        assert result["model_available"] is True
        assert result["n_periods"] == 96
        assert result["avg_abs_spread"] == 10.0
        assert result["p90_abs"] == 10.0
        assert result["mean_signed"] == 10.0
        assert result["coverage_adjustment_factor"] == 0.8
        assert result["annual_uplift_per_mw"] == 511.35
    # Do not silently shrink the sparse source's denominator from 97 to 96.
    assert absent_result[f"{source}_coverage_pct"] == 100.0
    assert singleton_result[f"{source}_coverage_pct"] == 99.0
    other = "ida" if source == "da" else "da"
    assert singleton_result[f"{other}_coverage_pct"] == 80.0


@pytest.mark.parametrize("column", ["price_eur_mwh", "intraday_price_eur_mwh"])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_uplift_counts_only_finite_price_pairs(column, value):
    da, ida = _markets()
    target = da if column == "price_eur_mwh" else ida
    target.iloc[-12, 0] = value
    result = calculate_intraday_uplift(da, ida, tz="Europe/Berlin")
    assert result["n_periods"] == 47
    assert result["model_available"] is True
    assert np.isfinite(result["annual_uplift_per_mw"])
    assert result["da_coverage_pct"] == pytest.approx(97.9)
    assert result["ida_coverage_pct"] == pytest.approx(97.9)


def test_single_replay_missing_row_message_leads_with_coverage():
    da, ida = _markets()
    ida = ida.drop(ida.index[-12])
    result = sim.simulate_da_id_replay(da, ida, simulation_date=date(2025, 9, 2), tz="Europe/Berlin")
    assert result["summary"]["status"] == "invalid_input"
    assert result["summary"]["message"].startswith("DA and IDA1 coverage is incomplete")
    assert "DA=24, IDA1=23, merged=23" in result["summary"]["message"]


@pytest.mark.parametrize("mode,zone", [("da_id", "DE_LU"), ("da_id", "FR"), ("reserve", "DE_LU")])
def test_project_case_rejects_native_pre_cutover_grids(mode, zone):
    """Native hourly DA + quarter-hour IDA must not become a fingerprinted result."""
    from src.project_case import (
        CurrencyBasis,
        CurrencyBasisMode,
        emit_da_id,
        emit_da_id_reserve,
    )
    from src.project_case.audit import AdapterUnavailableError

    da, ida = _markets("h", "15min")
    common = dict(
        zone=zone, first_delivery_date=date(2025, 9, 1), last_delivery_date=date(2025, 9, 2),
        power_mw=1.0, duration_hours=1.0, efficiency=1.0,
        currency_basis=CurrencyBasis(CurrencyBasisMode.SOURCE_EUR_TREATED_AS_BASE_YEAR_REAL, 2025),
        bucket="hour_of_day",
    )
    with pytest.raises(AdapterUnavailableError, match="no valid dates"):
        if mode == "da_id":
            emit_da_id(da, ida, min_rebid_uplift_eur=0.0, **common)
        else:
            reserve = pd.Series(5.0, index=da.index[::4])
            emit_da_id_reserve(
                da, ida, reserve, reserve_product="FCR", reserve_source="regelleistung", **common,
            )


def _uplift_app(sample):
    from datetime import date

    import pandas as pd
    import streamlit as st

    from src.pages.revenue_estimation import _render_intraday_uplift_section

    start = pd.Timestamp("2025-09-01", tz="Europe/Berlin")
    da_index = pd.date_range(start, periods=24, freq="h").tz_convert("UTC")
    ida_index = pd.date_range(start, periods=96, freq="15min").tz_convert("UTC")
    if sample == "sparse":
        ida_index = da_index[:-1]
    elif sample == "singleton":
        da_index = pd.date_range(start, periods=48, freq="h").tz_convert("UTC")
        ida_index = da_index[:25]
    da = pd.DataFrame({"price_eur_mwh": 50.0}, index=da_index)
    ida = pd.DataFrame({"intraday_price_eur_mwh": 60.0}, index=ida_index)
    if sample == "singleton":
        ida.iloc[-1, 0] = 1_000_000.0
    end = date(2025, 9, 2 if sample == "singleton" else 1)
    st.session_state[f"intraday_cache::DE_LU::2025-09-01::{end}"] = ida
    _render_intraday_uplift_section(
        primary_zone="DE_LU", primary_df=da, zone_tz="Europe/Berlin",
        start_date=date(2025, 9, 1), end_date=end,
        power_mw=1.0, duration_hours=1.0, efficiency=1.0,
        capture_rate=0.70, chart_template="plotly_dark",
    )


def test_revenue_panel_discloses_incompatible_grid_without_uplift_headline():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_uplift_app, args=("mismatch",)).run(timeout=30)
    assert not app.exception
    assert any("delivery grid" in warning.value.lower() for warning in app.warning)
    assert "Coverage-adjusted uplift" not in [metric.label for metric in app.metric]


def test_revenue_panel_discloses_da_and_ida_coverage_for_sparse_sample():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_uplift_app, args=("sparse",)).run(timeout=30)
    assert not app.exception
    assert "Coverage-adjusted uplift" in [metric.label for metric in app.metric]
    assert any("DA coverage: 95.8%" in caption.value and "IDA coverage: 100.0%" in caption.value
               for caption in app.caption)


def test_revenue_panel_excludes_singleton_from_headline_and_histogram(monkeypatch):
    import plotly.express as px
    from streamlit.testing.v1 import AppTest

    histogram = px.histogram
    plotted = []

    def capture_sample(data_frame, *args, **kwargs):
        plotted.append(data_frame[kwargs["x"]].copy())
        return histogram(data_frame, *args, **kwargs)

    monkeypatch.setattr(px, "histogram", capture_sample)
    app = AppTest.from_function(_uplift_app, args=("singleton",)).run(timeout=30)
    assert not app.exception
    headlines = [metric.value for metric in app.metric if metric.label == "Coverage-adjusted uplift"]
    assert headlines == ["€320"]
    assert len(plotted) == 1
    assert len(plotted[0]) == 24
    assert plotted[0].eq(10.0).all()
    assert any("Sample: 24 periods" in caption.value and "DA coverage: 50.0%" in caption.value
               and "IDA coverage: 96.0%" in caption.value for caption in app.caption)
