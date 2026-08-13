"""Coverage classification from raw prices (contract §4.3, red-lines #17/#21)."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.project_case import grid
from src.project_case.audit import (
    AdapterUnavailableError,
    build_reserve_coverage_audit,
    classify_leg_complete_dates,
    reserve_scalar_price,
)

ZONE = "DE_LU"
DAY = dt.date(2025, 6, 5)  # 60-min DA day, 24 stamps


def _da_series(days):
    idx, vals = [], []
    for d in days:
        for ts in grid.expected_da_timestamps(ZONE, d):
            idx.append(ts)
            vals.append(50.0)
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def test_complete_day_classified():
    s = _da_series([DAY])
    assert classify_leg_complete_dates(s, zone=ZONE, leg="da", evaluation_dates=(DAY,)) == frozenset({DAY})


def test_missing_timestamp_incomplete():
    s = _da_series([DAY]).iloc[:-1]  # drop one expected stamp
    assert classify_leg_complete_dates(s, zone=ZONE, leg="da", evaluation_dates=(DAY,)) == frozenset()


def test_extra_timestamp_incomplete():
    s = _da_series([DAY])
    extra = pd.Series([1.0], index=pd.DatetimeIndex([s.index[0] + pd.Timedelta(minutes=30)]))
    s2 = pd.concat([s, extra])
    assert classify_leg_complete_dates(s2, zone=ZONE, leg="da", evaluation_dates=(DAY,)) == frozenset()


def test_duplicate_timestamp_incomplete():
    s = _da_series([DAY])
    dup = pd.concat([s, s.iloc[[0]]])
    assert classify_leg_complete_dates(dup, zone=ZONE, leg="da", evaluation_dates=(DAY,)) == frozenset()


def test_nan_value_incomplete():
    s = _da_series([DAY]).copy()
    s.iloc[3] = float("nan")
    assert classify_leg_complete_dates(s, zone=ZONE, leg="da", evaluation_dates=(DAY,)) == frozenset()


def test_negative_energy_price_is_valid():
    s = _da_series([DAY]).copy()
    s.iloc[5] = -40.0
    assert classify_leg_complete_dates(s, zone=ZONE, leg="da", evaluation_dates=(DAY,)) == frozenset({DAY})


def test_twelve_row_every_two_hours_day_is_rejected():
    # The exact cadence-inference hole the contract closes (review 32/36): a
    # 12-row 00:00..22:00 every-2-hours day must NOT pass as a complete day.
    full = grid.expected_da_timestamps(ZONE, DAY)
    sparse = full[::2]  # 12 rows, uniform 2h delta -> _is_regular_utc_day would accept
    s = pd.Series([50.0] * len(sparse), index=pd.DatetimeIndex(sparse))
    assert classify_leg_complete_dates(s, zone=ZONE, leg="da", evaluation_dates=(DAY,)) == frozenset()


def test_unsupported_leg_raises_unavailable():
    s = _da_series([DAY])
    with pytest.raises(AdapterUnavailableError):
        classify_leg_complete_dates(s, zone="ES", leg="ida", evaluation_dates=(DAY,))


def test_fi_hourly_reserve_fails_closed_in_four_hour_v1_registry():
    # A complete Fingrid local day is 24 hourly capacity-price rows, not six
    # German blocks. Until the hourly contract/profile lands, it is unsupported
    # rather than silently coerced into the wrong calendar.
    idx = pd.date_range("2025-06-04T21:00:00Z", periods=24, freq="h", tz="UTC")
    prices = pd.Series(10.0, index=idx)
    with pytest.raises(AdapterUnavailableError, match="no reserve calendar"):
        build_reserve_coverage_audit(prices, zone="FI", evaluation_dates=(DAY,))


# --- Reserve coverage --------------------------------------------------------
def _reserve_series(day, n_blocks, price=12.0):
    idx, vals = [], []
    for bid, _ in grid.reserve_blocks(ZONE, day)[:n_blocks]:
        idx.append(pd.Timestamp(bid))
        vals.append(price)
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def test_reserve_full_coverage():
    a = build_reserve_coverage_audit(_reserve_series(DAY, 6), zone=ZONE, evaluation_dates=(DAY,))
    e = a.entries[0]
    assert e.fully_covered
    assert len(e.present_blocks) == 6 and e.missing_blocks == ()


def test_reserve_missing_block_not_covered():
    a = build_reserve_coverage_audit(_reserve_series(DAY, 5), zone=ZONE, evaluation_dates=(DAY,))
    assert not a.entries[0].fully_covered
    assert len(a.entries[0].missing_blocks) == 1


def test_reserve_negative_price_block_missing():
    s = _reserve_series(DAY, 6, price=-1.0)
    a = build_reserve_coverage_audit(s, zone=ZONE, evaluation_dates=(DAY,))
    assert a.entries[0].present_blocks == ()  # negatives unsupported for reserve


def test_reserve_entry_per_observed_date_including_fully_missing():
    d2 = dt.date(2025, 6, 6)
    a = build_reserve_coverage_audit(_reserve_series(DAY, 6), zone=ZONE, evaluation_dates=(DAY, d2))
    assert a.date_set() == frozenset({DAY, d2})
    missing_day = next(e for e in a.entries if e.date == d2)
    assert missing_day.present_blocks == () and len(missing_day.missing_blocks) == 6


def test_reserve_scalar_duration_weighted():
    s = _reserve_series(DAY, 6)
    s.iloc[0] = 24.0
    expected = (24.0 + 12.0 * 5) / 6
    assert reserve_scalar_price(s, zone=ZONE, pricing_dates=(DAY,)) == pytest.approx(expected)


def test_reserve_scalar_raises_on_incomplete_pre_gate():
    with pytest.raises(AdapterUnavailableError):
        reserve_scalar_price(_reserve_series(DAY, 5), zone=ZONE, pricing_dates=(DAY,))


def test_reserve_block_total_conserved_across_dst():
    # Review r3 #2 (ingestion -> alignment -> cash known-answer). The Regelleistung
    # ingestion divides each published block-total by its NOMINAL label hours
    # (always 4, DST-independent). Grid settlement duration is therefore also 4h, so
    # per_hour x duration reconstructs the SAME block-total on a DST day; weighting
    # by an elapsed 3h/5h duration would turn an €80 block into €60 (spring) / €100
    # (fall) — a cash leak. Different block prices make the weighting discriminating.
    spring, fall, normal = dt.date(2025, 3, 30), dt.date(2025, 10, 26), dt.date(2025, 6, 5)
    published_totals = [80.0] + [120.0] * 5  # €80 for the transition block, €120 rest
    expected_scalar = sum(published_totals) / (4.0 * 6)  # nominal-4h weighted mean
    for day in (spring, fall, normal):
        blocks = grid.reserve_blocks(ZONE, day)
        # Ingestion output: per-hour rate = published block-total / nominal 4h.
        per_hour = {
            pd.Timestamp(bid): total / 4.0
            for (bid, _), total in zip(blocks, published_totals, strict=True)
        }
        s = pd.Series(per_hour)
        # Cash side is DST-invariant (would be 28.70 spring / 28.0 fall under 3/5h).
        assert reserve_scalar_price(s, zone=ZONE, pricing_dates=(day,)) == pytest.approx(
            expected_scalar
        )
        # Per-block conservation: per_hour x grid_duration == the published total.
        for (bid, dur), total in zip(blocks, published_totals, strict=True):
            assert per_hour[pd.Timestamp(bid)] * dur == pytest.approx(total)


def test_reserve_extra_non_canonical_row_makes_day_uncovered():
    # 6 clean blocks PLUS an extra non-canonical row -> malformed day -> fully
    # uncovered (no pass-through as "fully covered", red-line #17/#21).
    s = _reserve_series(DAY, 6)
    extra = pd.Series([5.0], index=pd.DatetimeIndex([s.index[0] + pd.Timedelta(hours=1)]))
    audit = build_reserve_coverage_audit(pd.concat([s, extra]), zone=ZONE, evaluation_dates=(DAY,))
    assert not audit.entries[0].fully_covered
    assert audit.entries[0].present_blocks == ()
