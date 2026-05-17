"""Tests for forward-curve parsing and forward-scenario revenue."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import calculate_daily_spreads
from src.forward_curve import (
    build_forward_synthetic_prices,
    find_overlapping_contracts,
    generate_forward_template_csv,
    list_supported_zones,
    parse_forward_csv,
    summarise_forward_revenue,
)

# ── Template + parsing ──────────────────────────────────────────────────────

class TestForwardTemplate:
    def test_template_round_trips(self) -> None:
        csv = generate_forward_template_csv()
        df = parse_forward_csv(csv)
        assert len(df) == 3
        assert set(df["zone"]) == {"DE_LU", "NL", "IT_NORD"}

    def test_required_columns_present(self) -> None:
        csv = generate_forward_template_csv()
        df = parse_forward_csv(csv)
        for col in ("zone", "delivery_start", "delivery_end", "price_eur_mwh"):
            assert col in df.columns


class TestParseForwardCsv:
    def test_missing_required_column_raises(self) -> None:
        csv = "zone,price_eur_mwh\nDE_LU,80\n"
        with pytest.raises(ValueError, match="missing required columns"):
            parse_forward_csv(csv)

    def test_unknown_zone_raises(self) -> None:
        csv = (
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "XX_FAKE,2027-01-01,2027-02-01,80\n"
        )
        with pytest.raises(ValueError, match="unknown zone"):
            parse_forward_csv(csv)

    def test_unparseable_date_raises(self) -> None:
        csv = (
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,not-a-date,2027-02-01,80\n"
        )
        with pytest.raises(ValueError, match="unparseable"):
            parse_forward_csv(csv)

    def test_end_before_start_raises(self) -> None:
        csv = (
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,2027-02-01,2027-01-01,80\n"
        )
        with pytest.raises(ValueError, match="strictly after"):
            parse_forward_csv(csv)

    def test_non_numeric_price_raises(self) -> None:
        csv = (
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,2027-01-01,2027-02-01,not-a-price\n"
        )
        with pytest.raises(ValueError, match="non-numeric"):
            parse_forward_csv(csv)

    def test_comment_lines_skipped(self) -> None:
        csv = (
            "# This is a comment\n"
            "# Another one\n"
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,2027-01-01,2027-02-01,80\n"
        )
        df = parse_forward_csv(csv)
        assert len(df) == 1

    def test_optional_columns_default(self) -> None:
        """Optional columns missing in source CSV should be added as NA / 'base'."""
        csv = (
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,2027-01-01,2027-02-01,80\n"
        )
        df = parse_forward_csv(csv)
        assert df["shape"].iloc[0] == "base"
        assert df["contract"].iloc[0] is None
        assert pd.isna(df["source"].iloc[0])


# ── Overlap detection ───────────────────────────────────────────────────────

class TestFindOverlappingContracts:
    def test_detects_overlap(self) -> None:
        csv = (
            "zone,contract,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,Cal-2027,2027-01-01,2028-01-01,80\n"
            "DE_LU,Q1-2027,2027-01-01,2027-04-01,82\n"
        )
        df = parse_forward_csv(csv)
        overlaps = find_overlapping_contracts(df)
        assert len(overlaps) == 1
        assert overlaps.iloc[0]["zone"] == "DE_LU"

    def test_no_overlap_when_periods_distinct(self) -> None:
        csv = (
            "zone,contract,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,Q1-2027,2027-01-01,2027-04-01,80\n"
            "DE_LU,Q2-2027,2027-04-01,2027-07-01,82\n"
        )
        df = parse_forward_csv(csv)
        assert find_overlapping_contracts(df).empty

    def test_overlap_only_within_zone(self) -> None:
        """Two zones with overlapping periods is NOT an overlap (it's a portfolio)."""
        csv = (
            "zone,contract,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,Cal-2027,2027-01-01,2028-01-01,80\n"
            "NL,Cal-2027,2027-01-01,2028-01-01,90\n"
        )
        df = parse_forward_csv(csv)
        assert find_overlapping_contracts(df).empty


# ── Synthetic prices ────────────────────────────────────────────────────────

@pytest.fixture
def historical_sine() -> pd.DataFrame:
    """30 days of hourly data with deterministic intra-day spread."""
    idx = pd.date_range("2025-01-01", periods=30 * 24, freq="h", tz="UTC")
    hours = np.arange(30 * 24) % 24
    # Peak at hour 18 local, trough at hour 6; amplitude 30 around mean 60.
    prices = 60.0 + 30.0 * np.sin((hours - 6) * np.pi / 12)
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"
    return df


class TestBuildForwardSyntheticPrices:
    def test_synthetic_mean_matches_forward_base(
        self, historical_sine: pd.DataFrame,
    ) -> None:
        """For a long enough period the synthetic hourly mean should equal
        the forward base price exactly (the shape is normalised to mean=1).
        """
        forward = parse_forward_csv(
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,2027-03-01,2027-04-01,120\n"
        )
        synth = build_forward_synthetic_prices(
            forward, historical_sine, zone="DE_LU", tz="Europe/Berlin",
        )
        # Mean over a full week-aligned period should be ~ forward base.
        assert float(synth["price_eur_mwh"].mean()) == pytest.approx(120.0, rel=0.05)

    def test_shape_factor_preserved(self, historical_sine: pd.DataFrame) -> None:
        """Synthetic peak/trough ratio must mirror historical peak/trough."""
        forward = parse_forward_csv(
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,2027-03-01,2027-04-01,100\n"
        )
        synth = build_forward_synthetic_prices(
            forward, historical_sine, zone="DE_LU", tz="Europe/Berlin",
        )
        # Historical peak-to-trough ratio = 90/30 = 3.0
        hist_ratio = historical_sine["price_eur_mwh"].max() / historical_sine["price_eur_mwh"].min()
        synth_ratio = synth["price_eur_mwh"].max() / synth["price_eur_mwh"].min()
        assert synth_ratio == pytest.approx(hist_ratio, rel=0.1)

    def test_zone_not_in_forward_returns_empty(
        self, historical_sine: pd.DataFrame,
    ) -> None:
        forward = parse_forward_csv(
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "NL,2027-01-01,2027-02-01,80\n"
        )
        out = build_forward_synthetic_prices(
            forward, historical_sine, zone="DE_LU",
        )
        assert out.empty

    def test_later_overlapping_contract_wins(
        self, historical_sine: pd.DataFrame,
    ) -> None:
        """When two contracts overlap, the second in CSV order takes precedence
        on the overlap region (lets the user override broader contracts
        with a tighter, more recent quote).
        """
        forward = parse_forward_csv(
            "zone,contract,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,Cal-2027,2027-01-01,2027-02-01,80\n"
            "DE_LU,Jan-2027,2027-01-15,2027-01-22,200\n"
        )
        synth = build_forward_synthetic_prices(
            forward, historical_sine, zone="DE_LU", tz="Europe/Berlin",
        )
        # Jan-2027 range in Berlin = [2027-01-15 00:00, 2027-01-22 00:00).
        # In UTC that's [2027-01-14 23:00, 2027-01-21 23:00).
        jan_lo = pd.Timestamp("2027-01-14 23:00:00", tz="UTC")
        jan_hi = pd.Timestamp("2027-01-21 23:00:00", tz="UTC")
        jan_overlap = synth[(synth.index >= jan_lo) & (synth.index < jan_hi)]
        assert (jan_overlap["forward_base"] == 200.0).all()
        # Outside the Jan window, Cal should win (within the Cal range).
        before_jan = synth[synth.index < jan_lo]
        after_jan = synth[(synth.index >= jan_hi) & (synth.index < pd.Timestamp("2027-02-01 00:00:00", tz="UTC"))]
        assert (before_jan["forward_base"] == 80.0).all()
        assert (after_jan["forward_base"] == 80.0).all()

    def test_revenue_scales_linearly_with_forward_level(
        self, historical_sine: pd.DataFrame,
    ) -> None:
        """Forward revenue should scale linearly with the forward base price
        because spread also scales linearly.
        """
        forward_low = parse_forward_csv(
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,2027-03-01,2027-03-15,60\n"
        )
        forward_high = parse_forward_csv(
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,2027-03-01,2027-03-15,120\n"
        )
        synth_low = build_forward_synthetic_prices(
            forward_low, historical_sine, zone="DE_LU", tz="Europe/Berlin",
        )
        synth_high = build_forward_synthetic_prices(
            forward_high, historical_sine, zone="DE_LU", tz="Europe/Berlin",
        )
        ds_low = calculate_daily_spreads(
            synth_low[["price_eur_mwh"]], tz="Europe/Berlin", duration_hours=2.0,
        )
        ds_high = calculate_daily_spreads(
            synth_high[["price_eur_mwh"]], tz="Europe/Berlin", duration_hours=2.0,
        )
        # 2x level => 2x spread (within solver tolerance).
        assert ds_high["spread"].mean() == pytest.approx(
            2.0 * ds_low["spread"].mean(), rel=0.05,
        )


# ── Revenue summary ─────────────────────────────────────────────────────────

class TestSummariseForwardRevenue:
    def test_per_contract_aggregation(
        self, historical_sine: pd.DataFrame,
    ) -> None:
        forward = parse_forward_csv(
            "zone,contract,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,Mar-2027,2027-03-01,2027-03-15,100\n"
            "DE_LU,Apr-2027,2027-04-01,2027-04-15,90\n"
        )
        synth = build_forward_synthetic_prices(
            forward, historical_sine, zone="DE_LU", tz="Europe/Berlin",
        )
        ds = calculate_daily_spreads(
            synth[["price_eur_mwh"]], tz="Europe/Berlin", duration_hours=2.0,
        )
        summary = summarise_forward_revenue(
            ds, forward, zone="DE_LU", power_mw=1.0, duration_hours=2.0,
        )
        assert len(summary) == 2
        assert set(summary["contract"]) == {"Mar-2027", "Apr-2027"}
        # Both contracts should have positive period revenue.
        assert (summary["period_revenue_eur"] > 0).all()
        # Higher forward base should produce higher per-day revenue.
        mar = summary[summary["contract"] == "Mar-2027"].iloc[0]
        apr = summary[summary["contract"] == "Apr-2027"].iloc[0]
        assert (mar["period_revenue_eur"] / mar["days_in_period"]) > (
            apr["period_revenue_eur"] / apr["days_in_period"]
        )

    def test_empty_daily_returns_empty_summary(self) -> None:
        forward = parse_forward_csv(
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,2027-01-01,2027-02-01,80\n"
        )
        out = summarise_forward_revenue(
            pd.DataFrame(), forward, zone="DE_LU",
            power_mw=1.0, duration_hours=2.0,
        )
        assert out.empty


class TestListSupportedZones:
    def test_returns_sorted_unique(self) -> None:
        csv = (
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "NL,2027-01-01,2027-02-01,80\n"
            "DE_LU,2027-01-01,2027-02-01,80\n"
            "DE_LU,2027-02-01,2027-03-01,82\n"
        )
        df = parse_forward_csv(csv)
        assert list(list_supported_zones(df)) == ["DE_LU", "NL"]
