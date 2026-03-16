"""Tests for data_ingestion module. All API calls are mocked."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import ELEXON_ZONES, ENTSOE_ZONES, GBP_TO_EUR, is_elexon_zone
from src.data_ingestion import (
    clean_prices,
    fetch_elexon_prices,
    fetch_elexon_system_prices,
    fetch_entsoe_imbalance_prices,
    fetch_entsoe_prices,
    fetch_fingrid_data,
    fetch_fingrid_fcr_prices,
    fetch_prices,
    read_cache,
    write_cache,
)


# ── Test 1: ENTSO-E schema ──────────────────────────────────────────────────

class TestFetchEntsoePrices:
    @patch("src.data_ingestion._call_entsoe_api")
    @patch("src.data_ingestion.get_api_key", return_value="fake-key")
    def test_returns_correct_schema(
        self, _mock_key: MagicMock, mock_api: MagicMock, mock_entsoe_series: pd.Series,
    ) -> None:
        """Returned DataFrame has correct columns and UTC timestamps."""
        mock_api.return_value = mock_entsoe_series

        start = pd.Timestamp("2025-01-01", tz="UTC")
        end = pd.Timestamp("2025-01-02", tz="UTC")
        df = fetch_entsoe_prices("DE_LU", start, end)

        assert "price_eur_mwh" in df.columns
        assert df.index.name == "timestamp"
        assert df.index.tz is not None  # timezone-aware
        assert str(df.index.tz) == "UTC"
        assert len(df) == 24


# ── Test 2: Elexon schema ───────────────────────────────────────────────────

class TestFetchElexonPrices:
    @patch("src.data_ingestion._call_elexon_api")
    def test_returns_correct_schema(
        self, mock_api: MagicMock, mock_elexon_json: list[dict],
    ) -> None:
        """Returned DataFrame has correct columns, EUR conversion, 30-min res."""
        mock_api.return_value = mock_elexon_json

        start = pd.Timestamp("2025-01-01", tz="UTC")
        end = pd.Timestamp("2025-01-01", tz="UTC")
        df = fetch_elexon_prices(start, end)

        assert "price_eur_mwh" in df.columns
        assert df.index.name == "timestamp"
        assert df.index.tz is not None
        assert len(df) == 48

        # Verify GBP->EUR conversion
        original_gbp = mock_elexon_json[0]["price"]
        expected_eur = original_gbp * GBP_TO_EUR
        assert abs(df["price_eur_mwh"].iloc[0] - expected_eur) < 0.01


# ── Test 3: GB routes to Elexon ─────────────────────────────────────────────

class TestFetchPricesRouting:
    @patch("src.data_ingestion.fetch_elexon_prices")
    @patch("src.data_ingestion.fetch_entsoe_prices")
    def test_gb_routes_to_elexon(
        self, mock_entsoe: MagicMock, mock_elexon: MagicMock,
        mock_price_df: pd.DataFrame,
    ) -> None:
        """fetch_prices(zone='GB') should call Elexon, not ENTSO-E."""
        mock_elexon.return_value = mock_price_df

        df = fetch_prices(
            zone="GB",
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-01-02", tz="UTC"),
            use_cache=False,
        )
        mock_elexon.assert_called_once()
        mock_entsoe.assert_not_called()
        assert len(df) > 0

    # ── Test 4: DE_LU routes to ENTSO-E ──────────────────────────────────────

    @patch("src.data_ingestion.fetch_elexon_prices")
    @patch("src.data_ingestion.fetch_entsoe_prices")
    def test_de_lu_routes_to_entsoe(
        self, mock_entsoe: MagicMock, mock_elexon: MagicMock,
        mock_price_df: pd.DataFrame,
    ) -> None:
        """fetch_prices(zone='DE_LU') should call ENTSO-E, not Elexon."""
        mock_entsoe.return_value = mock_price_df

        df = fetch_prices(
            zone="DE_LU",
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-01-02", tz="UTC"),
            use_cache=False,
        )
        mock_entsoe.assert_called_once()
        mock_elexon.assert_not_called()
        assert len(df) > 0


# ── Test 5: clean_prices fills gaps ──────────────────────────────────────────

class TestCleanPrices:
    def test_fills_gaps(self) -> None:
        """NaN gaps should be forward-filled, no NaN remaining."""
        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"price_eur_mwh": [50.0 + i for i in range(24)]}, index=idx)
        df.index.name = "timestamp"

        # Introduce NaN gaps
        df.loc[df.index[5:8], "price_eur_mwh"] = None

        result = clean_prices(df)
        assert result["price_eur_mwh"].isna().sum() == 0
        assert len(result) == 24

    def test_empty_df_passthrough(self) -> None:
        """Empty DataFrame should pass through without error."""
        df = pd.DataFrame(columns=["price_eur_mwh"])
        result = clean_prices(df)
        assert result.empty


# ── Test 6: cache roundtrip ──────────────────────────────────────────────────

class TestCacheRoundtrip:
    def test_write_and_read(self, tmp_path: Path, mock_price_df: pd.DataFrame) -> None:
        """Data written to cache should be readable back with matching values."""
        zone = "DE_LU"
        start = pd.Timestamp("2025-01-01", tz="UTC")
        end = pd.Timestamp("2025-01-02", tz="UTC")

        # Patch DB_PATH and CACHE_DIR to use tmp_path
        with patch("src.data_ingestion.DB_PATH", tmp_path / "test.db"), \
             patch("src.data_ingestion.CACHE_DIR", tmp_path):
            write_cache(mock_price_df, zone)
            result = read_cache(zone, start, end)

        assert result is not None
        assert len(result) == 24
        pd.testing.assert_series_equal(
            mock_price_df["price_eur_mwh"].reset_index(drop=True),
            result["price_eur_mwh"].reset_index(drop=True),
            check_names=False,
        )


# ── Test 7: invalid zone raises error ────────────────────────────────────────

class TestInvalidZone:
    def test_raises_value_error(self) -> None:
        """Unknown zone code should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown bidding zone"):
            fetch_prices(
                zone="INVALID_ZONE",
                start=pd.Timestamp("2025-01-01", tz="UTC"),
                end=pd.Timestamp("2025-01-02", tz="UTC"),
            )


# ── Test 8: empty API response handled ───────────────────────────────────────

class TestEmptyApiResponse:
    @patch("src.data_ingestion._call_entsoe_api")
    @patch("src.data_ingestion.get_api_key", return_value="fake-key")
    def test_empty_entsoe_returns_empty_df(
        self, _mock_key: MagicMock, mock_api: MagicMock, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Empty ENTSO-E response returns empty DataFrame with warning logged."""
        mock_api.return_value = pd.Series(
            [], dtype=float, index=pd.DatetimeIndex([], tz="Europe/Brussels"),
        )
        with caplog.at_level(logging.WARNING):
            df = fetch_entsoe_prices(
                "DE_LU",
                pd.Timestamp("2025-01-01", tz="UTC"),
                pd.Timestamp("2025-01-02", tz="UTC"),
            )
        assert df.empty
        assert any("empty" in r.message.lower() for r in caplog.records)

    @patch("src.data_ingestion._call_elexon_api")
    def test_empty_elexon_returns_empty_df(
        self, mock_api: MagicMock, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Empty Elexon response returns empty DataFrame with warning logged."""
        mock_api.return_value = []
        with caplog.at_level(logging.WARNING):
            df = fetch_elexon_prices(
                pd.Timestamp("2025-01-01", tz="UTC"),
                pd.Timestamp("2025-01-01", tz="UTC"),
            )
        assert df.empty
        assert any("no data" in r.message.lower() for r in caplog.records)


# ── Test 9: config zone classification ───────────────────────────────────────

class TestConfigZoneClassification:
    def test_gb_is_elexon(self) -> None:
        assert is_elexon_zone("GB") is True

    def test_de_lu_is_not_elexon(self) -> None:
        assert is_elexon_zone("DE_LU") is False

    def test_all_entsoe_zones_are_not_elexon(self) -> None:
        for code in ENTSOE_ZONES.values():
            assert is_elexon_zone(code) is False

    def test_all_elexon_zones_are_elexon(self) -> None:
        for code in ELEXON_ZONES.values():
            assert is_elexon_zone(code) is True


# ── Test 10: Fingrid data fetcher ─────────────────────────────────────────────

class TestFetchFingridData:
    @patch("src.data_ingestion.requests.get")
    def test_returns_correct_schema(self, mock_get: MagicMock) -> None:
        """fetch_fingrid_data returns DataFrame with timestamp and value."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"startTime": "2025-01-01T00:00:00Z", "value": 10.5},
                {"startTime": "2025-01-01T01:00:00Z", "value": 11.2},
                {"startTime": "2025-01-01T02:00:00Z", "value": 9.8},
            ]
        }
        mock_get.return_value = mock_resp

        df = fetch_fingrid_data(
            318,
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert "timestamp" in df.columns
        assert "value" in df.columns
        assert len(df) == 3

    @patch("src.data_ingestion.requests.get")
    def test_empty_response(self, mock_get: MagicMock) -> None:
        """Empty Fingrid response returns empty DataFrame."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        df = fetch_fingrid_data(
            318,
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert df.empty


# ── Test 11: Fingrid FCR prices ───────────────────────────────────────────────

class TestFetchFingridFcrPrices:
    @patch("src.data_ingestion.fetch_fingrid_data")
    def test_returns_correct_columns(self, mock_fetch: MagicMock) -> None:
        """fetch_fingrid_fcr_prices returns correct column schema."""
        idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        mock_fetch.return_value = pd.DataFrame({
            "timestamp": idx,
            "value": [10.0, 11.0, 12.0],
        })

        df = fetch_fingrid_fcr_prices(
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert "timestamp" in df.columns
        # At least one of the FCR price columns should be present
        fcr_cols = {"fcr_n_price", "fcr_d_up_price", "fcr_d_down_price"}
        assert len(fcr_cols.intersection(df.columns)) > 0

    @patch("src.data_ingestion.fetch_fingrid_data")
    def test_empty_response(self, mock_fetch: MagicMock) -> None:
        """Empty Fingrid response returns empty DataFrame."""
        mock_fetch.return_value = pd.DataFrame(columns=["timestamp", "value"])
        df = fetch_fingrid_fcr_prices(
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert df.empty


# ── Test 12: Elexon system prices ─────────────────────────────────────────────

class TestFetchElexonSystemPrices:
    @patch("src.data_ingestion.requests.get")
    def test_returns_correct_schema(self, mock_get: MagicMock) -> None:
        """fetch_elexon_system_prices returns correct columns."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [
            {
                "settlementDate": "2025-01-01",
                "settlementPeriod": 1,
                "systemBuyPrice": 55.0,
                "systemSellPrice": 45.0,
            },
            {
                "settlementDate": "2025-01-01",
                "settlementPeriod": 2,
                "systemBuyPrice": 60.0,
                "systemSellPrice": 50.0,
            },
        ]
        mock_get.return_value = mock_resp

        df = fetch_elexon_system_prices(
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-01", tz="UTC"),
        )
        assert "system_buy_price_gbp" in df.columns
        assert "system_sell_price_gbp" in df.columns
        assert "spread_eur" in df.columns
        assert len(df) == 2

    @patch("src.data_ingestion.requests.get")
    def test_empty_response(self, mock_get: MagicMock) -> None:
        """Empty Elexon response returns empty DataFrame."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        df = fetch_elexon_system_prices(
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-01", tz="UTC"),
        )
        assert df.empty


# ── Test 13: ENTSO-E imbalance prices ─────────────────────────────────────────

class TestFetchEntsoeImbalancePrices:
    @patch("src.data_ingestion.EntsoePandasClient")
    @patch("src.data_ingestion.get_api_key", return_value="fake-key")
    def test_returns_none_on_failure(
        self, _mock_key: MagicMock, mock_client_cls: MagicMock,
    ) -> None:
        """Returns None gracefully when data is unavailable."""
        mock_client = MagicMock()
        mock_client.query_imbalance_prices.side_effect = Exception("No data")
        mock_client_cls.return_value = mock_client

        result = fetch_entsoe_imbalance_prices(
            "RO",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert result is None

    @patch("src.data_ingestion.EntsoePandasClient")
    @patch("src.data_ingestion.get_api_key", return_value="fake-key")
    def test_returns_dataframe_on_success(
        self, _mock_key: MagicMock, mock_client_cls: MagicMock,
    ) -> None:
        """Returns DataFrame with imbalance prices on success."""
        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="Europe/Brussels")
        raw = pd.DataFrame(
            {"long": range(24), "short": range(24, 48)},
            index=idx,
        )
        mock_client = MagicMock()
        mock_client.query_imbalance_prices.return_value = raw
        mock_client_cls.return_value = mock_client

        result = fetch_entsoe_imbalance_prices(
            "RO",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert result is not None
        assert "imbalance_price_long" in result.columns
        assert "imbalance_price_short" in result.columns
        assert len(result) == 24
