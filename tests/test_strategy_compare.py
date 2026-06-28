"""Tests for the dispatch-strategy comparison table."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ancillary import (
    capacity_price_for_product,
    capacity_price_series_for_product,
    parse_capacity_import_csv,
)
from src.pages.simulation_cockpit import (
    _CAP_SOURCE_CACHE,
    _CAP_SOURCE_SESSION,
    _fmt_strategy_bar_label,
    _reserve_coopt_total,
    _reserve_triple_totals,
    _resolve_capacity_dataset,
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


def test_realistic_triple_row_appended_with_default_label() -> None:
    table = build_strategy_comparison(
        _summary(100.0, 130.0, 160.0, valid_days=10),
        power_mw=2.0,
        realistic_triple_total=150.0,
    )
    assert len(table) == 4  # 3 base + realistic (no reserve/ceiling here)
    assert table.iloc[3]["strategy"] == "DA + IDA1 + reserve (forecast-driven realistic)"
    assert table.iloc[3]["uplift_vs_da_pct"] == pytest.approx(50.0)


def test_realistic_triple_row_dropped_when_non_finite() -> None:
    for bad in (float("nan"), float("inf")):
        table = build_strategy_comparison(
            _summary(100.0, 130.0, 160.0, valid_days=10),
            power_mw=2.0,
            realistic_triple_total=bad,
        )
        assert len(table) == 3


def test_triple_rows_use_their_own_window_denominator_and_baseline() -> None:
    # The 9.2b walk-forward window can span fewer days (8) and carry a different
    # DA baseline (80) than the DA/IDA rows (10 days, baseline 100). The triple
    # ceiling + realistic rows must annualise/uplift on THEIR window, not the
    # DA/IDA one.
    table = build_strategy_comparison(
        _summary(100.0, 130.0, 160.0, valid_days=10),
        power_mw=2.0,
        triple_joint_total=180.0,
        triple_joint_label="DA + IDA1 + FCR (co-opt ceiling)",
        realistic_triple_total=150.0,
        realistic_triple_label="DA + IDA1 + FCR (forecast-driven realistic)",
        triple_valid_days=8,
        triple_da_baseline=80.0,
    )
    assert len(table) == 5  # da, forecast, ceiling, triple(4), realistic(5)
    # Ceiling + realistic annualise over 8 days and uplift vs baseline 80.
    assert table.iloc[3]["annualized_eur_per_mw"] == pytest.approx(
        180.0 * DAYS_PER_YEAR / 8 / 2.0,
    )
    assert table.iloc[3]["uplift_vs_da_pct"] == pytest.approx((180.0 - 80.0) / 80.0 * 100.0)
    assert table.iloc[4]["annualized_eur_per_mw"] == pytest.approx(
        150.0 * DAYS_PER_YEAR / 8 / 2.0,
    )
    assert table.iloc[4]["uplift_vs_da_pct"] == pytest.approx((150.0 - 80.0) / 80.0 * 100.0)
    # The DA/IDA rows still use the 10-day / baseline-100 convention.
    assert table.iloc[0]["annualized_eur_per_mw"] == pytest.approx(
        100.0 * DAYS_PER_YEAR / 10 / 2.0,
    )
    assert table.iloc[1]["uplift_vs_da_pct"] == pytest.approx(30.0)


def test_six_row_layout_reserve_triple_realistic() -> None:
    table = build_strategy_comparison(
        _summary(100.0, 130.0, 160.0, valid_days=10),
        power_mw=2.0,
        reserve_coopt_total=120.0,
        reserve_label="DA + FCR co-opt (headroom)",
        triple_joint_total=175.0,
        triple_joint_label="DA + IDA1 + FCR (co-opt ceiling)",
        realistic_triple_total=150.0,
        realistic_triple_label="DA + IDA1 + FCR (forecast-driven realistic)",
        triple_valid_days=9,
        triple_da_baseline=95.0,
    )
    assert list(table["strategy"]) == [
        "DA-only",
        "DA + IDA1 (forecast-driven)",
        "DA + IDA1 (perfect-foresight ceiling)",
        "DA + FCR co-opt (headroom)",
        "DA + IDA1 + FCR (co-opt ceiling)",
        "DA + IDA1 + FCR (forecast-driven realistic)",
    ]


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


def _ida_frame(days: int = 4) -> pd.DataFrame:
    """IDA series on the same timestamps as ``_price_frame``, perturbed off DA so
    the rebid stage has something to act on."""
    n = 24 * days
    idx = pd.date_range("2025-06-01 00:00", periods=n, freq="h", tz="UTC")
    prices = (
        50 + 40 * np.sin(np.arange(n) / 24 * 2 * np.pi)
        + 6 * np.cos(np.arange(n) / 12 * 2 * np.pi)
    )
    return pd.DataFrame({"intraday_price_eur_mwh": prices}, index=idx)


def _fcr_series_daily(days: int = 4) -> pd.DataFrame:
    """One FCR capacity print per 4h block per day, so the block-of-day reserve
    climatology has history to forecast from."""
    base = pd.Timestamp("2025-06-01 00:00", tz="UTC")
    stamps = [
        base + pd.Timedelta(days=d, hours=h)
        for d in range(days) for h in range(0, 24, 4)
    ]
    return pd.DataFrame(
        {
            "product_type": ["FCR"] * len(stamps),
            "capacity_price_eur_mw": [12.0] * len(stamps),
            "energy_price_eur_mwh": [np.nan] * len(stamps),
        },
        index=pd.DatetimeIndex(stamps),
    )


def test_reserve_triple_totals_ceiling_matches_summary_and_bounds_realistic() -> None:
    da = _price_frame(days=4)
    ida = _ida_frame(days=4)
    series = capacity_price_series_for_product(_fcr_series_daily(days=4), "FCR")
    valid = set(_berlin_dates(da))
    out = _reserve_triple_totals(
        da, ida, series, valid_dates=valid, tz="Europe/Berlin",
        power_mw=1.0, duration_hours=2, efficiency=0.88, bucket="hour_of_day",
    )
    assert out["seq_per_day"] is not None and not out["seq_per_day"].empty
    # The displayed 9.2a ceiling equals the 9.2b batch's global-ceiling ref.
    assert out["triple_total"] == pytest.approx(
        out["seq_summary"]["total_global_ceiling_eur"],
    )
    # Forecast-driven realistic can never beat the perfect-foresight ceiling.
    assert out["realistic_total"] <= out["triple_total"] + 1e-6
    # Both rows share the walk-forward window's denominator + DA baseline.
    assert out["triple_valid_days"] == out["seq_summary"]["valid_days"]
    assert out["triple_da_baseline"] == pytest.approx(
        out["seq_summary"]["total_da_only_eur"],
    )


def test_reserve_triple_totals_none_when_series_missing_or_empty() -> None:
    da = _price_frame(days=4)
    ida = _ida_frame(days=4)
    valid = set(_berlin_dates(da))
    common = dict(
        valid_dates=valid, tz="Europe/Berlin", power_mw=1.0,
        duration_hours=2, efficiency=0.88, bucket="hour_of_day",
    )
    none_out = _reserve_triple_totals(da, ida, None, **common)
    assert none_out["triple_total"] is None
    assert none_out["realistic_total"] is None
    assert none_out["seq_per_day"] is None
    empty_out = _reserve_triple_totals(
        da, ida, pd.Series(dtype=float), **common,
    )
    assert empty_out["realistic_total"] is None


# ── 6c-3: cockpit cache-first capacity consumption ────────────────────────────

def _seed_capacity_cache(monkeypatch, tmp_path, csv_text: str) -> None:
    from src import data_ingestion as di
    monkeypatch.setattr(di, "DB_PATH", tmp_path / "bess.db")
    di.persist_capacity_frame(parse_capacity_import_csv(csv_text))


def test_resolve_capacity_prefers_cache_over_session(tmp_path, monkeypatch) -> None:
    _seed_capacity_cache(
        monkeypatch, tmp_path,
        "timestamp,zone,product,direction,capacity_price_eur_mw_h\n"
        "2026-05-01T00:00:00Z,DE_LU,FCR,symmetric,12.5\n",
    )
    # Session ancillary carries a DIFFERENT product; the cache must win.
    session = pd.DataFrame(
        {"product_type": ["aFRR Up"], "capacity_price_eur_mw": [5.0]},
        index=pd.to_datetime(["2026-05-01T00:00:00Z"], utc=True),
    )
    out, label = _resolve_capacity_dataset("DE_LU", session)
    assert label == _CAP_SOURCE_CACHE
    assert list(out["product_type"]) == ["FCR"]  # cached, not session aFRR


def test_resolve_capacity_falls_back_to_session_when_cache_empty(
    tmp_path, monkeypatch,
) -> None:
    from src import data_ingestion as di
    monkeypatch.setattr(di, "DB_PATH", tmp_path / "bess.db")  # nothing seeded
    session = pd.DataFrame(
        {"product_type": ["FCR"], "capacity_price_eur_mw": [9.0]},
        index=pd.to_datetime(["2026-05-01T00:00:00Z"], utc=True),
    )
    out, label = _resolve_capacity_dataset("DE_LU", session)
    assert label == _CAP_SOURCE_SESSION
    assert out is session


def test_resolve_capacity_empty_zone_falls_back_without_cache_lookup() -> None:
    session = pd.DataFrame(
        {"product_type": ["FCR"], "capacity_price_eur_mw": [9.0]},
        index=pd.to_datetime(["2026-05-01T00:00:00Z"], utc=True),
    )
    out, label = _resolve_capacity_dataset("", session)
    assert label == _CAP_SOURCE_SESSION
    assert out is session


def test_resolve_capacity_none_when_no_cache_and_no_session(
    tmp_path, monkeypatch,
) -> None:
    from src import data_ingestion as di
    monkeypatch.setattr(di, "DB_PATH", tmp_path / "bess.db")
    out, label = _resolve_capacity_dataset("DE_LU", None)
    assert out is None
    assert label == _CAP_SOURCE_SESSION


def test_resolve_capacity_cache_is_zone_scoped(tmp_path, monkeypatch) -> None:
    # Capacity cached for FR must not leak into a DE_LU resolve (zone-scoped read).
    _seed_capacity_cache(
        monkeypatch, tmp_path,
        "timestamp,zone,product,direction,capacity_price_eur_mw_h\n"
        "2026-05-01T00:00:00Z,FR,FCR,symmetric,12.5\n",
    )
    out, label = _resolve_capacity_dataset("DE_LU", None)
    assert out is None and label == _CAP_SOURCE_SESSION


def test_cache_capacity_window_slicing_excludes_out_of_window(
    tmp_path, monkeypatch,
) -> None:
    # Red-line: a cached out-of-window print must not leak into the comparison
    # window's price (slicing happens before pricing).
    _seed_capacity_cache(
        monkeypatch, tmp_path,
        "timestamp,zone,product,direction,capacity_price_eur_mw_h\n"
        "2026-05-15T12:00:00Z,DE_LU,FCR,symmetric,1000.0\n"  # out of window
        "2026-06-02T10:00:00Z,DE_LU,FCR,symmetric,10.0\n"     # Berlin Jun 2
        "2026-06-03T10:00:00Z,DE_LU,FCR,symmetric,10.0\n",    # Berlin Jun 3
    )
    out, _ = _resolve_capacity_dataset("DE_LU", None)
    window = _slice_to_local_dates(out, {date(2026, 6, 2), date(2026, 6, 3)}, "Europe/Berlin")
    assert capacity_price_for_product(window, "FCR") == pytest.approx(10.0)
