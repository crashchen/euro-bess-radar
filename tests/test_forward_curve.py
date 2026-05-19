"""Tests for forward-curve parsing and forward-scenario revenue."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    calculate_daily_spreads,
    estimate_annual_arbitrage_revenue,
)
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
            ds, forward, synth, zone="DE_LU",
            power_mw=1.0, duration_hours=2.0, tz="Europe/Berlin",
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

    def test_per_day_revenue_matches_historical_path(
        self, historical_sine: pd.DataFrame,
    ) -> None:
        """Codex P1 / PR5 repro: ``summarise_forward_revenue`` used
        ``efficiency ** 0.5`` while ``estimate_annual_arbitrage_revenue``
        applies the full ``efficiency``. Identical daily spreads therefore
        produced ~11% higher per-day revenue in the Forward tab than the
        Revenue Estimation tab — inconsistent EUR/MW/yr for the same
        parameter set. After the fix the two paths must agree.
        """
        # Single non-overlapping contract so attribution is trivial.
        forward = parse_forward_csv(
            "zone,contract,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,Mar-2027,2027-03-01,2027-03-15,100\n"
        )
        synth = build_forward_synthetic_prices(
            forward, historical_sine, zone="DE_LU", tz="Europe/Berlin",
        )
        ds = calculate_daily_spreads(
            synth[["price_eur_mwh"]], tz="Europe/Berlin", duration_hours=2.0,
        )
        summary = summarise_forward_revenue(
            ds, forward, synth, zone="DE_LU",
            power_mw=1.0, duration_hours=2.0,
            efficiency=0.88, capture_rate=0.70, tz="Europe/Berlin",
        )
        row = summary.iloc[0]
        forward_per_day = row["period_revenue_eur"] / row["days_in_period"]

        # Build the historical-path expected per-day using the SAME spread
        # series feeding the same EUR/MW/yr formula.
        rev = estimate_annual_arbitrage_revenue(
            ds, power_mw=1.0, duration_hours=2.0,
            roundtrip_efficiency=0.88, capture_rate=0.70,
        )
        # estimate_annual_arbitrage_revenue returns annual EUR/MW; convert
        # to per-day at the same convention summarise uses.
        hist_per_day = rev["annual_revenue_eur_per_mw"] / 365.25

        # Tolerance accounts for round(...,2) on both summary and revenue
        # paths. The pre-fix divergence was ~11% (sqrt(0.88) vs 0.88), so
        # any 0.1% threshold catches the regression with room to spare.
        assert forward_per_day == pytest.approx(hist_per_day, rel=1e-3), (
            f"forward={forward_per_day:.4f} historical={hist_per_day:.4f} "
            "(efficiency double-application regression)"
        )

    def test_overlap_days_attributed_to_winner_not_double_counted(
        self, historical_sine: pd.DataFrame,
    ) -> None:
        """Codex P0 repro: Cal-2027 + Jan-2027 overlap. The synthetic series
        only realises one winner per day, but the pre-fix summary summed
        revenue over each contract's full [start, end) window, so Jan days
        were counted in BOTH contract totals.

        After the fix:
        - Each calendar day is attributed to exactly one contract — the
          one that won that day's hours in build_forward_synthetic_prices.
        - sum(days_in_period) over all summary rows must equal the number
          of unique calendar days covered by the synthetic series.
        """
        csv = (
            "zone,contract,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,Cal-2027,2027-01-01,2027-02-01,80\n"
            "DE_LU,Jan-2027,2027-01-01,2027-02-01,200\n"
        )
        forward = parse_forward_csv(csv)
        synth = build_forward_synthetic_prices(
            forward, historical_sine, zone="DE_LU", tz="Europe/Berlin",
        )
        ds = calculate_daily_spreads(
            synth[["price_eur_mwh"]], tz="Europe/Berlin", duration_hours=2.0,
        )
        summary = summarise_forward_revenue(
            ds, forward, synth, zone="DE_LU",
            power_mw=1.0, duration_hours=2.0, tz="Europe/Berlin",
        )

        # Synthetic local-day coverage: count unique local dates.
        synth_local_dates = (
            synth.index.tz_convert("Europe/Berlin").normalize().unique()
        )
        n_unique_days = len(synth_local_dates)

        # All days attributed; none double-counted.
        assert int(summary["days_in_period"].sum()) == n_unique_days

        # Jan-2027 (uploaded second) wins on the overlap, so Cal-2027
        # either gets zero days (full overlap) or no row at all.
        cal_rows = summary[summary["contract"] == "Cal-2027"]
        if not cal_rows.empty:
            assert int(cal_rows["days_in_period"].iloc[0]) == 0

        # Jan-2027 carries the full coverage at its forward base.
        jan_row = summary[summary["contract"] == "Jan-2027"].iloc[0]
        assert int(jan_row["days_in_period"]) == n_unique_days
        assert float(jan_row["forward_base"]) == 200.0

    def test_duplicate_and_blank_labels_do_not_merge(
        self, historical_sine: pd.DataFrame,
    ) -> None:
        """Codex P0 repro: when two rows share a contract label (or both
        are blank), an earlier label-keyed attribution silently merged
        their days. With CSV-row id (_csv_order) as the join key, each
        row gets only the days where it actually won, so total days
        equal unique calendar days.
        """
        # Two non-overlapping rows with BLANK labels — the legitimate
        # case where the user uploaded raw period quotes without naming.
        csv_blank = (
            "zone,contract,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,,2027-01-01,2027-01-05,80\n"
            "DE_LU,,2027-01-05,2027-01-10,100\n"
        )
        forward = parse_forward_csv(csv_blank)
        synth = build_forward_synthetic_prices(
            forward, historical_sine, zone="DE_LU", tz="Europe/Berlin",
        )
        ds = calculate_daily_spreads(
            synth[["price_eur_mwh"]], tz="Europe/Berlin", duration_hours=2.0,
        )
        summary = summarise_forward_revenue(
            ds, forward, synth, zone="DE_LU",
            power_mw=1.0, duration_hours=2.0, tz="Europe/Berlin",
        )
        unique_days = len(
            synth.index.tz_convert("Europe/Berlin").normalize().unique()
        )
        # Pre-fix: each blank-label row collected ALL unique days
        # (sum = 2 * unique_days). Post-fix: each row gets only its own.
        assert int(summary["days_in_period"].sum()) == unique_days
        # And the two forward_base values are still distinguishable.
        assert sorted(summary["forward_base"].tolist()) == [80.0, 100.0]

    def test_partial_overlap_splits_days_per_winner(
        self, historical_sine: pd.DataFrame,
    ) -> None:
        """A Cal contract that overlaps a Jan contract in the first 31 days
        and runs alone Feb-Dec must report Jan's days against Jan and
        Feb-Dec days against Cal — exactly the calendar split.
        """
        csv = (
            "zone,contract,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,Cal-2027,2027-01-01,2027-03-01,80\n"
            "DE_LU,Jan-2027,2027-01-01,2027-02-01,200\n"
        )
        forward = parse_forward_csv(csv)
        synth = build_forward_synthetic_prices(
            forward, historical_sine, zone="DE_LU", tz="Europe/Berlin",
        )
        ds = calculate_daily_spreads(
            synth[["price_eur_mwh"]], tz="Europe/Berlin", duration_hours=2.0,
        )
        summary = summarise_forward_revenue(
            ds, forward, synth, zone="DE_LU",
            power_mw=1.0, duration_hours=2.0, tz="Europe/Berlin",
        )

        jan = summary[summary["contract"] == "Jan-2027"].iloc[0]
        cal = summary[summary["contract"] == "Cal-2027"].iloc[0]
        # Jan-2027 should win all 31 January days (uploaded later -> wins
        # on overlap). Cal-2027 keeps Feb (~28 days, ignoring last-day
        # exclusive boundary slop).
        assert 28 <= int(jan["days_in_period"]) <= 31
        # Cal coverage = total - jan; no double-count.
        synth_local_dates = (
            synth.index.tz_convert("Europe/Berlin").normalize().unique()
        )
        assert int(cal["days_in_period"]) == (
            len(synth_local_dates) - int(jan["days_in_period"])
        )

    def test_empty_daily_returns_empty_summary(self) -> None:
        forward = parse_forward_csv(
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,2027-01-01,2027-02-01,80\n"
        )
        out = summarise_forward_revenue(
            pd.DataFrame(), forward, pd.DataFrame(), zone="DE_LU",
            power_mw=1.0, duration_hours=2.0,
        )
        assert out.empty


class TestReviewFollowUpFixes:
    """Regressions for findings from Codex/Gemini review of 8d275b3."""

    def test_inline_hash_in_source_field_preserved(self) -> None:
        """Codex P2: pd.read_csv(..., comment='#') was truncating field
        values that contained '#' (e.g. 'Desk #1' -> 'Desk '). Strip
        comments only on full lines so cell contents survive.
        """
        csv = (
            "# leading comment\n"
            "zone,delivery_start,delivery_end,price_eur_mwh,source\n"
            "DE_LU,2027-01-01,2027-02-01,80,Desk #1\n"
        )
        df = parse_forward_csv(csv)
        assert df["source"].iloc[0] == "Desk #1"

    def test_non_base_shape_warns_logger(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Gemini P0: peak/offpeak rows must surface a warning so the user
        knows their peak price is being treated as baseload by the v1 engine.
        """
        import logging
        csv = (
            "zone,delivery_start,delivery_end,price_eur_mwh,shape\n"
            "DE_LU,2027-01-01,2027-02-01,150,peak\n"
        )
        with caplog.at_level(logging.WARNING, logger="src.forward_curve"):
            parse_forward_csv(csv)
        assert any("shape != 'base'" in msg for msg in caplog.messages)

    def test_csv_row_order_preserved_as_overlap_priority(
        self, historical_sine: pd.DataFrame,
    ) -> None:
        """Codex P1: parse_forward_csv sorts by (zone, delivery_start);
        the 'later in CSV wins' semantic on overlap must use the
        original row order, not the post-sort position.
        """
        # Put the OVERRIDE row first in CSV order, then the broader Cal row.
        # Without the fix, the post-sort priority assigns the override a
        # smaller priority and the broader Cal wins on the overlap.
        csv = (
            "zone,contract,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,Jan-override,2027-01-15,2027-01-22,200\n"
            "DE_LU,Cal-later,2027-01-01,2027-02-01,80\n"
        )
        forward = parse_forward_csv(csv)
        synth = build_forward_synthetic_prices(
            forward, historical_sine, zone="DE_LU", tz="Europe/Berlin",
        )
        jan_lo = pd.Timestamp("2027-01-14 23:00:00", tz="UTC")
        jan_hi = pd.Timestamp("2027-01-21 23:00:00", tz="UTC")
        jan_window = synth[(synth.index >= jan_lo) & (synth.index < jan_hi)]
        # Cal-later was uploaded AFTER Jan-override, so Cal must win on
        # the overlap region (later in CSV wins).
        assert (jan_window["forward_base"] == 80.0).all()

    def test_short_contract_baseload_preserved_per_window(
        self, historical_sine: pd.DataFrame,
    ) -> None:
        """Codex P1: a 24h contract used to come out with mean != forward
        base because the shape was normalised globally, not over the
        contract window. After fix, mean over the contract window must
        equal the forward base exactly (within float epsilon).
        """
        csv = (
            "zone,contract,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,OneDay,2027-03-15,2027-03-16,100\n"
        )
        forward = parse_forward_csv(csv)
        synth = build_forward_synthetic_prices(
            forward, historical_sine, zone="DE_LU", tz="Europe/Berlin",
        )
        assert float(synth["price_eur_mwh"].mean()) == pytest.approx(100.0, rel=1e-6)

    def test_short_history_warns_about_partial_buckets(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Codex P1: a history shorter than a full week silently flat-fills
        the missing hour-of-week buckets with factor 1.0. Warn the user
        so they know the recovered shape is partial.
        """
        import logging
        # Only 72 hours = 3 days of history.
        idx = pd.date_range("2025-01-01", periods=72, freq="h", tz="UTC")
        short = pd.DataFrame(
            {"price_eur_mwh": [50.0 + (i % 24) * 2.0 for i in range(72)]},
            index=idx,
        )
        forward = parse_forward_csv(
            "zone,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,2027-03-01,2027-03-08,100\n"
        )
        with caplog.at_level(logging.WARNING, logger="src.forward_curve"):
            build_forward_synthetic_prices(
                forward, short, zone="DE_LU", tz="Europe/Berlin",
            )
        assert any("hour-of-week buckets" in msg for msg in caplog.messages)

    def test_nan_lp_day_excluded_from_period_revenue(self) -> None:
        """Codex P2: pandas mean() skips NaN but multiplying by raw n_days
        would scale a partial mean to a full-period total. Drop NaN LP
        days from both the numerator and the denominator.
        """
        from src.forward_curve import summarise_forward_revenue
        # Two days in the period; one has NaN lp_revenue, the other 100.
        daily = pd.DataFrame({
            "date": [pd.Timestamp("2027-03-01"), pd.Timestamp("2027-03-02")],
            "spread": [50.0, 50.0],
            "lp_revenue": [100.0, float("nan")],
            "n_cycles": [1.0, 0.0],
        })
        forward = parse_forward_csv(
            "zone,contract,delivery_start,delivery_end,price_eur_mwh\n"
            "DE_LU,TwoDays,2027-03-01,2027-03-03,100\n"
        )
        # Minimal synthetic frame covering both days under the same
        # contract label so the summary's per-day attribution lands here.
        synth_idx = pd.date_range(
            "2027-03-01 00:00", "2027-03-03 00:00", freq="h",
            tz="Europe/Berlin", inclusive="left",
        ).tz_convert("UTC")
        synth = pd.DataFrame(
            {"price_eur_mwh": 100.0, "contract": "TwoDays",
             "forward_base": 100.0, "shape_factor": 1.0},
            index=synth_idx,
        )
        synth.index.name = "timestamp"
        summary = summarise_forward_revenue(
            daily, forward, synth, zone="DE_LU",
            power_mw=1.0, duration_hours=2.0, tz="Europe/Berlin",
        )
        # 1 valid LP day * mean(100) * capture(0.7) = 70 EUR period revenue,
        # NOT 1 valid day's worth scaled up to 2 days.
        assert summary["days_in_period"].iloc[0] == 1
        assert summary["period_revenue_eur"].iloc[0] == pytest.approx(70.0)


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
