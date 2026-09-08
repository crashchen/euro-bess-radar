"""Regression checks for physical rolling windows and forecast baseline disclosure."""

import numpy as np
import pandas as pd
import pytest

from src.ida_forecast import FORECAST_COL, compute_forecast_skill


@pytest.mark.parametrize("sample", ["quarter_da", "missing", "infinite", "duplicate", "none"])
def test_forecast_baseline_reports_its_own_finite_comparison_population(sample):
    index = pd.date_range("2025-09-01", periods=96, freq="15min", tz="UTC")
    forecast = pd.DataFrame({FORECAST_COL: 55.0}, index=index)
    realised = pd.DataFrame({"intraday_price_eur_mwh": 60.0}, index=index)
    da = pd.DataFrame({"price_eur_mwh": 50.0}, index=index[::4])
    count = 24
    if sample == "missing":
        da.iloc[0, 0] = np.nan
        count = 23
    elif sample == "infinite":
        da.iloc[0, 0] = np.inf
        count = 23
    elif sample == "duplicate":
        da = pd.concat([da, da.iloc[[0]]])
        count = 0
    elif sample == "none":
        da = None
        count = 0
    result = compute_forecast_skill(forecast, realised, da_prices=da, tz="Europe/Berlin")
    assert result["n_points"] == 96
    assert result["mae"] == 5.0
    assert result["da_baseline_n_points"] == count
    assert result["da_baseline_coverage_pct"] == pytest.approx(count / 96 * 100)
    assert result["forecast_mae_on_da_overlap"] == (5.0 if count else None)
    assert result["skill_vs_da"] == (0.5 if count else None)


def _skill_app():
    import pandas as pd

    from src.ida_forecast import FORECAST_COL, compute_forecast_skill
    from src.pages.simulation_cockpit import _render_forecast_skill

    index = pd.date_range("2025-09-01", periods=96, freq="15min", tz="UTC")
    forecast = pd.DataFrame({FORECAST_COL: 55.0}, index=index)
    realised = pd.DataFrame({"intraday_price_eur_mwh": 60.0}, index=index)
    da = pd.DataFrame({"price_eur_mwh": 50.0}, index=index[::4])
    _render_forecast_skill(compute_forecast_skill(forecast, realised, da_prices=da), "plotly_dark")


def test_forecast_panel_discloses_da_subset_separately():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_skill_app).run(timeout=30)
    assert not app.exception
    assert any("24/96" in caption.value and "25.0%" in caption.value for caption in app.caption)


def _market_app(sample):
    import numpy as np
    import pandas as pd
    import streamlit as st

    from src.analytics import (
        calculate_daily_spreads,
        calculate_negative_price_hours,
        calculate_spread_percentiles,
    )
    from src.pages.market_overview import render

    if sample == "mixed":
        before = pd.date_range("2025-09-01", "2025-10-01", inclusive="left", freq="h", tz="Europe/Berlin")
        after = pd.date_range("2025-10-01", "2025-10-11", inclusive="left", freq="15min", tz="Europe/Berlin")
        index = before.append(after).tz_convert("UTC")
        values = np.r_[np.full(len(before), 10.0), np.full(len(after), 100.0)]
    else:
        count = 40 * 96 if sample == "quarter" else 40 * 24
        index = pd.date_range("2026-01-01", periods=count, freq="15min" if sample == "quarter" else "h", tz="UTC")
        values = np.r_[np.full(count * 3 // 4, 10.0), np.full(count // 4, 100.0)]
    frame = pd.DataFrame({"price_eur_mwh": values}, index=index.rename("timestamp"))
    daily = calculate_daily_spreads(frame)
    figures = {}
    render("DE_LU", frame, daily, calculate_spread_percentiles(daily),
           calculate_negative_price_hours(frame), 1, "UTC", "plotly_dark", figures)
    st.session_state["ma_values"] = np.asarray(figures["price_ts"].data[1].y, dtype=float)


@pytest.mark.parametrize("sample,warmup", [("hour", 24), ("quarter", 96), ("mixed", 24)])
def test_real_market_chart_uses_30_physical_days_and_24_hours_warmup(sample, warmup):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_market_app, args=(sample,)).run(timeout=30)
    assert not app.exception
    values = app.session_state["ma_values"]
    assert np.isnan(values[:warmup - 1]).all()
    assert values[warmup - 1] == 10.0
    assert values[-1] == pytest.approx(40.0)


def test_rolling_window_clips_a_partial_old_interval_and_ignores_nan():
    from src.analytics import time_weighted_rolling_price_mean

    before = pd.date_range("2025-09-30", periods=24, freq="h", tz="Europe/Berlin")
    after = pd.date_range("2025-10-01", periods=4, freq="15min", tz="Europe/Berlin")
    prices = pd.Series(np.r_[np.full(24, 10.0), np.full(4, 100.0)], index=before.append(after))
    result = time_weighted_rolling_price_mean(prices, window="1D")
    assert result.iloc[24] == pytest.approx((23.75 * 10 + 0.25 * 100) / 24)
    prices.iloc[0] = np.nan
    result = time_weighted_rolling_price_mean(prices, window="1D")
    assert result.iloc[:27].isna().all()  # fewer than 24 finite covered hours
    assert result.iloc[-1] == pytest.approx((23 * 10 + 100) / 24)
