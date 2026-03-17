"""Tests for ancillary services module and ancillary fetchers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ancillary import (
    ANCILLARY_TEMPLATES,
    calculate_ancillary_revenue,
    generate_template_csv,
    merge_revenue_stack,
    parse_ancillary_csv,
)
from src.ancillary_fetchers import get_available_fetchers, run_auto_fetch


# ── Template CSV generation ──────────────────────────────────────────────────

class TestTemplateGeneration:
    def test_all_templates_generate(self) -> None:
        for key in ANCILLARY_TEMPLATES:
            csv_str = generate_template_csv(key)
            assert len(csv_str) > 0
            lines = csv_str.strip().split("\n")
            assert len(lines) == 3  # header + 2 example rows

    def test_template_has_correct_headers(self) -> None:
        for key, tmpl in ANCILLARY_TEMPLATES.items():
            csv_str = generate_template_csv(key)
            header_line = csv_str.strip().split("\n")[0]
            for col in tmpl["expected_columns"]:
                assert col in header_line


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
        assert df["system_buy_price_eur_mwh"].iloc[0] == 55.0
        assert df["system_sell_price_eur_mwh"].iloc[0] == 45.0
        assert df["energy_price_eur_mwh"].iloc[0] == 10.0

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

    def test_gb_balancing_uses_bid_ask_spread(self, gb_balancing_csv: str) -> None:
        df = parse_ancillary_csv(gb_balancing_csv, "GB_BALANCING")
        result = calculate_ancillary_revenue(df, power_mw=1.0, duration_hours=1.0)
        assert result["mfrr_annual_eur"] == 8760.0

    def test_empty_df_returns_zeros(self) -> None:
        df = pd.DataFrame(columns=["capacity_price_eur_mw", "energy_price_eur_mwh",
                                    "product_type", "direction", "zone"])
        result = calculate_ancillary_revenue(df)
        assert result["total_ancillary_eur"] == 0.0

    def test_keys(self, de_fcr_csv: str) -> None:
        df = parse_ancillary_csv(de_fcr_csv, "DE_FCR")
        result = calculate_ancillary_revenue(df)
        expected = {"fcr_annual_eur", "afrr_annual_eur", "mfrr_annual_eur",
                    "total_ancillary_eur", "total_ancillary_per_mw"}
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
        }
        result = merge_revenue_stack(da, anc)
        assert result["total_eur"] == 85000.0
        assert result["da_arbitrage_eur"] == 50000.0
        assert abs(result["da_pct"] - 58.8) < 0.1
        assert abs(result["ancillary_pct"] - 41.2) < 0.1

    def test_da_only(self) -> None:
        da = {"annual_revenue_eur": 50000.0, "annual_revenue_eur_per_mw": 50000.0}
        anc = {
            "fcr_annual_eur": 0.0,
            "afrr_annual_eur": 0.0,
            "mfrr_annual_eur": 0.0,
            "total_ancillary_eur": 0.0,
        }
        result = merge_revenue_stack(da, anc)
        assert result["total_eur"] == 50000.0
        assert result["da_pct"] == 100.0


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
