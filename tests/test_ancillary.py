"""Tests for ancillary services module and ancillary fetchers."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ancillary import (
    ANCILLARY_TEMPLATES,
    build_ancillary_dataset,
    calculate_ancillary_revenue,
    generate_template_csv,
    merge_revenue_stack,
    normalize_auto_fetch_dataset,
    parse_ancillary_csv,
)
from src.ancillary_fetchers import get_available_fetchers, run_auto_fetch
from src.config import ANCILLARY_ENERGY_ACTIVATION_SHARE, GBP_EUR_YEARLY, HOURS_PER_YEAR


# ── Template CSV generation ──────────────────────────────────────────────────

class TestTemplateGeneration:
    def test_all_templates_generate(self) -> None:
        for key in ANCILLARY_TEMPLATES:
            csv_str = generate_template_csv(key)
            assert len(csv_str) > 0
            lines = [line for line in csv_str.strip().split("\n") if not line.startswith("#")]
            assert len(lines) == 3  # header + 2 example rows

    def test_template_has_correct_headers(self) -> None:
        for key, tmpl in ANCILLARY_TEMPLATES.items():
            csv_str = generate_template_csv(key)
            header_line = next(
                line for line in csv_str.strip().split("\n") if not line.startswith("#")
            )
            for col in tmpl["expected_columns"]:
                assert col in header_line

    def test_gb_template_mentions_gbp_units(self) -> None:
        csv_str = generate_template_csv("GB_BALANCING")
        assert "GBP/MWh" in csv_str

    def test_fi_fcr_template_uses_zero_indexed_hours(self) -> None:
        csv_str = generate_template_csv("FI_FCR")
        lines = [line for line in csv_str.strip().split("\n") if not line.startswith("#")]
        assert lines[1].split(",")[1] == "0"
        assert lines[2].split(",")[1] == "1"

# ── Parsing ──────────────────────────────────────────────────────────────────

@pytest.fixture
def de_fcr_csv() -> str:
    """Minimal DE_FCR CSV."""
    return (
        "date,product,capacity_price_eur_mw\n"
        "2025-01-01,POS,15.50\n"
        "2025-01-02,POS,16.20\n"
        "2025-01-03,POS,14.80\n"
    )


@pytest.fixture
def gb_balancing_csv() -> str:
    """Minimal GB_BALANCING CSV."""
    return (
        "settlement_date,settlement_period,system_buy_price,system_sell_price\n"
        "2025-01-01,1,55.00,45.00\n"
        "2025-01-01,2,60.00,50.00\n"
    )


class TestParsing:
    def test_de_fcr_parsing(self, de_fcr_csv: str) -> None:
        df = parse_ancillary_csv(de_fcr_csv, "DE_FCR")
        assert len(df) == 3
        assert "capacity_price_eur_mw" in df.columns
        assert df["capacity_price_eur_mw"].iloc[0] == 15.50

    def test_gb_balancing_parsing(self, gb_balancing_csv: str) -> None:
        df = parse_ancillary_csv(gb_balancing_csv, "GB_BALANCING")
        assert len(df) == 2
        assert "energy_price_eur_mwh" in df.columns
        assert "system_buy_price_eur_mwh" in df.columns
        assert "system_sell_price_eur_mwh" in df.columns
        assert df["system_buy_price_eur_mwh"].iloc[0] == pytest.approx(
            55.0 * GBP_EUR_YEARLY[2025]
        )
        assert df["system_sell_price_eur_mwh"].iloc[0] == pytest.approx(
            45.0 * GBP_EUR_YEARLY[2025]
        )
        assert df["energy_price_eur_mwh"].isna().all()

    def test_gb_balancing_parsing_ignores_template_comment_line(self) -> None:
        csv = generate_template_csv("GB_BALANCING")
        df = parse_ancillary_csv(csv, "GB_BALANCING")
        assert len(df) == 2

    def test_normalize_auto_fetch_preserves_system_prices(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="30min", tz="UTC")
        raw = pd.DataFrame({
            "timestamp": idx,
            "system_buy_price_eur": [60.0, 65.0],
            "system_sell_price_eur": [40.0, 45.0],
        })
        df = normalize_auto_fetch_dataset(raw, "System prices")
        assert df["system_buy_price_eur_mwh"].iloc[0] == 60.0
        assert df["system_sell_price_eur_mwh"].iloc[0] == 40.0
        assert df["energy_price_eur_mwh"].isna().all()

    def test_normalize_auto_fetch_preserves_product_dimension(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        raw = pd.DataFrame({
            "timestamp": idx,
            "fcr_n_price": [10.0, 14.0],
            "fcr_d_up_price": [20.0, 22.0],
            "afrr_up_price": [30.0, 34.0],
        })
        df = normalize_auto_fetch_dataset(raw, "FI reserves")
        assert set(df["product_type"]) == {"FCR-N", "FCR-D Up", "aFRR Up"}
        assert len(df) == 6

    def test_column_mapping(self, de_fcr_csv: str) -> None:
        df = parse_ancillary_csv(de_fcr_csv, "DE_FCR")
        expected = {"product_type", "direction", "capacity_price_eur_mw",
                    "energy_price_eur_mwh", "zone"}
        assert expected.issubset(set(df.columns))

    def test_invalid_template_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown template"):
            parse_ancillary_csv("a,b\n1,2", "INVALID")

    def test_semicolon_delimiter(self) -> None:
        csv = "date;product;capacity_price_eur_mw\n2025-01-01;POS;10.00\n"
        df = parse_ancillary_csv(csv, "DE_FCR")
        assert len(df) == 1

    def test_fi_fcr_hour_zero_and_twenty_three_round_trip(self) -> None:
        csv = (
            "date,hour,fcr_n_price,fcr_d_price\n"
            "2025-01-01,0,10.00,20.00\n"
            "2025-01-01,23,11.00,21.00\n"
        )
        df = parse_ancillary_csv(csv, "FI_FCR")

        fcr_n_hours = list(df[df["product_type"] == "FCR-N"].index.hour)
        fcr_d_hours = list(df[df["product_type"] == "FCR-D"].index.hour)

        assert fcr_n_hours == [0, 23]
        assert fcr_d_hours == [0, 23]


# ── Revenue calculation ──────────────────────────────────────────────────────

class TestAncillaryRevenue:
    def test_with_capacity_prices(self, de_fcr_csv: str) -> None:
        df = parse_ancillary_csv(de_fcr_csv, "DE_FCR")
        result = calculate_ancillary_revenue(df, power_mw=10.0, duration_hours=1.0)
        assert result["total_ancillary_eur"] == 1289910.0
        assert result["fcr_annual_eur"] == 1289910.0

    def test_energy_prices_use_eur_units(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame({
            "energy_price_eur_mwh": [100.0, 100.0],
            "capacity_price_eur_mw": [float("nan"), float("nan")],
            "product_type": ["GB_BALANCING", "GB_BALANCING"],
            "direction": ["", ""],
            "zone": ["GB", "GB"],
        }, index=idx)

        result = calculate_ancillary_revenue(df, power_mw=1.0, duration_hours=1.0)
        assert result["mfrr_annual_eur"] == 87600.0

    def test_capacity_assumption_reads_from_config_constant(self, de_fcr_csv: str) -> None:
        df = parse_ancillary_csv(de_fcr_csv, "DE_FCR")

        with patch("src.ancillary.ANCILLARY_CAPACITY_AVAILABILITY", 0.50):
            result = calculate_ancillary_revenue(df, power_mw=1.0, duration_hours=1.0)

        expected = round(df["capacity_price_eur_mw"].mean() * HOURS_PER_YEAR * 0.50, 2)
        assert result["fcr_annual_eur"] == expected

    def test_capacity_prices_use_duration_weighted_mean(self) -> None:
        idx = pd.to_datetime(
            ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z", "2025-01-01T07:00:00Z"],
            utc=True,
        )
        df = pd.DataFrame({
            "capacity_price_eur_mw": [10.0, 20.0, 30.0],
            "energy_price_eur_mwh": [float("nan"), float("nan"), float("nan")],
            "product_type": ["FCR", "FCR", "FCR"],
            "direction": ["", "", ""],
            "zone": ["FI", "FI", "FI"],
        }, index=idx)

        with patch("src.ancillary.ANCILLARY_CAPACITY_AVAILABILITY", 1.0):
            result = calculate_ancillary_revenue(df, power_mw=1.0, duration_hours=1.0)

        assert result["fcr_annual_eur"] == round(26.25 * HOURS_PER_YEAR, 2)

    def test_energy_activation_share_reads_from_config_constant(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame({
            "energy_price_eur_mwh": [100.0, 100.0],
            "capacity_price_eur_mw": [float("nan"), float("nan")],
            "product_type": ["GB_BALANCING", "GB_BALANCING"],
            "direction": ["", ""],
            "zone": ["GB", "GB"],
        }, index=idx)

        with patch("src.ancillary.ANCILLARY_ENERGY_ACTIVATION_SHARE", 0.25):
            result = calculate_ancillary_revenue(df, power_mw=1.0, duration_hours=1.0)

        assert result["mfrr_annual_eur"] == 100.0 * HOURS_PER_YEAR * 0.25

    def test_energy_prices_annualise_using_power_not_duration(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame({
            "energy_price_eur_mwh": [120.0, 120.0],
            "capacity_price_eur_mw": [float("nan"), float("nan")],
            "product_type": ["GB_BALANCING", "GB_BALANCING"],
            "direction": ["", ""],
            "zone": ["GB", "GB"],
        }, index=idx)

        result = calculate_ancillary_revenue(df, power_mw=2.0, duration_hours=4.0)
        expected = 120.0 * 2.0 * HOURS_PER_YEAR * ANCILLARY_ENERGY_ACTIVATION_SHARE
        assert result["mfrr_annual_eur"] == expected

    def test_gb_balancing_requires_explicit_energy_price(self, gb_balancing_csv: str) -> None:
        df = parse_ancillary_csv(gb_balancing_csv, "GB_BALANCING")
        result = calculate_ancillary_revenue(df, power_mw=1.0, duration_hours=1.0)
        assert result["mfrr_annual_eur"] == 0.0

    def test_auto_fetch_capacity_data_flows_into_revenue(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        auto_results = {
            "FCR-N/D prices": pd.DataFrame({
                "timestamp": idx,
                "fcr_n_price": [10.0, 14.0],
                "fcr_d_up_price": [20.0, 22.0],
            }),
            "aFRR prices": pd.DataFrame({
                "timestamp": idx,
                "afrr_up_price": [30.0, 34.0],
            }),
        }

        anc_df = build_ancillary_dataset(auto_fetch_results=auto_results)
        result = calculate_ancillary_revenue(anc_df, power_mw=1.0, duration_hours=1.0)
        assert result["fcr_annual_eur"] == 274626.0
        assert result["afrr_annual_eur"] == 266304.0
        assert result["product_revenues"] == {
            "FCR-D Up": 174762.0,
            "FCR-N": 99864.0,
            "aFRR Up": 266304.0,
        }

    def test_manual_upload_overrides_same_product_only(self, de_fcr_csv: str) -> None:
        manual_df = parse_ancillary_csv(de_fcr_csv, "DE_FCR")
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        auto_results = {
            "FCR auctions": pd.DataFrame({
                "timestamp": idx,
                "capacity_price_eur_mw": [99.0, 100.0],
                "product": ["FCR", "FCR"],
            }),
            "aFRR prices": pd.DataFrame({
                "timestamp": idx,
                "afrr_up_price": [30.0, 34.0],
            }),
        }

        anc_df = build_ancillary_dataset(manual_df=manual_df, auto_fetch_results=auto_results)
        assert set(anc_df["product_type"]) == {"FCR", "aFRR Up"}
        manual_fcr = anc_df[anc_df["product_type"] == "FCR"]
        assert manual_fcr["capacity_price_eur_mw"].iloc[0] == 15.50

    def test_manual_fi_fcr_d_overrides_auto_fcr_d_aliases(self) -> None:
        idx = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
        manual_df = pd.DataFrame(
            {
                "capacity_price_eur_mw": [50.0],
                "energy_price_eur_mwh": [float("nan")],
                "product_type": ["FCR-D"],
                "direction": [""],
                "zone": ["FI"],
            },
            index=idx,
        )
        auto_results = {
            "Fingrid reserves": pd.DataFrame(
                {
                    "timestamp": idx,
                    "fcr_n_price": [10.0],
                    "fcr_d_up_price": [20.0],
                    "fcr_d_down_price": [30.0],
                }
            )
        }

        anc_df = build_ancillary_dataset(manual_df=manual_df, auto_fetch_results=auto_results)

        assert "FCR-D" in set(anc_df["product_type"])
        assert "FCR-N" in set(anc_df["product_type"])
        assert "FCR-D Up" not in set(anc_df["product_type"])
        assert "FCR-D Down" not in set(anc_df["product_type"])

    def test_empty_df_returns_zeros(self) -> None:
        df = pd.DataFrame(columns=["capacity_price_eur_mw", "energy_price_eur_mwh",
                                    "product_type", "direction", "zone"])
        result = calculate_ancillary_revenue(df)
        assert result["total_ancillary_eur"] == 0.0

    def test_keys(self, de_fcr_csv: str) -> None:
        df = parse_ancillary_csv(de_fcr_csv, "DE_FCR")
        result = calculate_ancillary_revenue(df)
        expected = {"fcr_annual_eur", "afrr_annual_eur", "mfrr_annual_eur",
                    "total_ancillary_eur", "total_ancillary_per_mw", "product_revenues"}
        assert set(result.keys()) == expected


# ── Revenue stack merging ────────────────────────────────────────────────────

class TestMergeRevenueStack:
    def test_arithmetic(self) -> None:
        da = {"annual_revenue_eur": 50000.0, "annual_revenue_eur_per_mw": 50000.0}
        anc = {
            "fcr_annual_eur": 20000.0,
            "afrr_annual_eur": 10000.0,
            "mfrr_annual_eur": 5000.0,
            "total_ancillary_eur": 35000.0,
            "product_revenues": {"FCR-N": 20000.0, "aFRR Up": 10000.0, "mFRR": 5000.0},
        }
        result = merge_revenue_stack(da, anc, power_mw=1.0)
        assert result["total_eur"] == 85000.0
        assert result["da_arbitrage_eur"] == 50000.0
        assert abs(result["da_pct"] - 58.8) < 0.1
        assert abs(result["ancillary_pct"] - 41.2) < 0.1
        assert result["source_revenues"]["FCR-N"] == 20000.0

    def test_da_only(self) -> None:
        da = {"annual_revenue_eur": 50000.0, "annual_revenue_eur_per_mw": 50000.0}
        anc = {
            "fcr_annual_eur": 0.0,
            "afrr_annual_eur": 0.0,
            "mfrr_annual_eur": 0.0,
            "total_ancillary_eur": 0.0,
            "product_revenues": {},
        }
        result = merge_revenue_stack(da, anc, power_mw=1.0)
        assert result["total_eur"] == 50000.0
        assert result["da_pct"] == 100.0

    def test_total_per_mw_uses_power_when_da_is_zero(self) -> None:
        da = {"annual_revenue_eur": 0.0, "annual_revenue_eur_per_mw": 0.0}
        anc = {
            "fcr_annual_eur": 12000.0,
            "afrr_annual_eur": 6000.0,
            "mfrr_annual_eur": 2000.0,
            "total_ancillary_eur": 20000.0,
            "product_revenues": {"FCR-N": 12000.0, "aFRR Up": 6000.0, "mFRR": 2000.0},
        }

        result = merge_revenue_stack(da, anc, power_mw=4.0)
        assert result["total_eur"] == 20000.0
        assert result["total_per_mw"] == 5000.0

    def test_total_per_mw_uses_explicit_power_not_da_ratio(self) -> None:
        da = {"annual_revenue_eur": 50000.0, "annual_revenue_eur_per_mw": 12345.0}
        anc = {
            "fcr_annual_eur": 10000.0,
            "afrr_annual_eur": 0.0,
            "mfrr_annual_eur": 0.0,
            "total_ancillary_eur": 10000.0,
            "product_revenues": {"FCR-N": 10000.0},
        }

        result = merge_revenue_stack(da, anc, power_mw=2.0)
        assert result["total_eur"] == 60000.0
        assert result["total_per_mw"] == 30000.0


# ── Auto-fetcher registry ──────────────────────────────────────────────────

class TestGetAvailableFetchers:
    def test_fi_has_fetchers(self) -> None:
        fetchers = get_available_fetchers("FI")
        assert len(fetchers) == 2
        names = [f["name"] for f in fetchers]
        assert "FCR-N/D prices" in names
        assert "aFRR prices" in names

    def test_gb_has_fetchers(self) -> None:
        fetchers = get_available_fetchers("GB")
        assert len(fetchers) == 1
        assert fetchers[0]["source"] == "Elexon"

    def test_de_lu_has_fetchers(self) -> None:
        fetchers = get_available_fetchers("DE_LU")
        assert len(fetchers) == 1
        assert fetchers[0]["source"] == "Regelleistung.net"

    def test_unknown_zone_returns_empty(self) -> None:
        assert get_available_fetchers("XX_UNKNOWN") == []

    def test_entsoe_zones_have_imbalance(self) -> None:
        for zone in ["RO", "SE_3", "IT_SUD"]:
            fetchers = get_available_fetchers(zone)
            assert len(fetchers) >= 1
            assert fetchers[0]["source"] == "ENTSO-E"


class TestRunAutoFetch:
    @patch("src.data_ingestion.fetch_fingrid_fcr_prices")
    @patch("src.data_ingestion.fetch_fingrid_afrr_prices")
    def test_fi_fetch_success(
        self, mock_afrr: MagicMock, mock_fcr: MagicMock,
    ) -> None:
        """Successful FI fetch returns both datasets."""
        idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        mock_fcr.return_value = pd.DataFrame({
            "timestamp": idx,
            "fcr_n_price": [10.0, 11.0, 12.0],
        })
        mock_afrr.return_value = pd.DataFrame({
            "timestamp": idx,
            "afrr_up_price": [5.0, 6.0, 7.0],
        })

        results = run_auto_fetch(
            "FI",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert len(results) == 2

    @patch("src.data_ingestion.fetch_fingrid_fcr_prices")
    @patch("src.data_ingestion.fetch_fingrid_afrr_prices")
    def test_handles_partial_failure(
        self, mock_afrr: MagicMock, mock_fcr: MagicMock,
    ) -> None:
        """One fetcher failing doesn't prevent others from succeeding."""
        idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        mock_fcr.return_value = pd.DataFrame({
            "timestamp": idx,
            "fcr_n_price": [10.0, 11.0, 12.0],
        })
        mock_afrr.side_effect = Exception("API timeout")

        results = run_auto_fetch(
            "FI",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert len(results) == 1  # FCR succeeded, aFRR failed

    def test_unknown_zone_returns_empty(self) -> None:
        results = run_auto_fetch(
            "XX_UNKNOWN",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert results == {}

    @patch("src.data_ingestion.fetch_regelleistung_results")
    def test_de_lu_calls_regelleistung(
        self, mock_regelleistung: MagicMock,
    ) -> None:
        mock_regelleistung.return_value = pd.DataFrame({
            "date": ["2025-01-01"],
            "product": ["FCR"],
            "time_block": ["00:00"],
            "capacity_price_eur_mw": [5.50],
            "direction": ["Symmetric"],
        })
        results = run_auto_fetch(
            "DE_LU",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert mock_regelleistung.call_count == 2  # FCR + aFRR
        assert len(results) >= 1
