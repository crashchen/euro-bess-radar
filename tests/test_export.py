"""Tests for export module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

from src.analytics import (
    calculate_daily_spreads,
    calculate_monthly_spreads,
    calculate_negative_price_hours,
    calculate_spread_percentiles,
    estimate_annual_arbitrage_revenue,
)
from src.export import export_to_bytes, export_to_excel


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    """Build minimal analytics data for export tests."""
    idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    price_df = pd.DataFrame(
        {"price_eur_mwh": [50.0 + (i % 24) * 2.0 for i in range(48)]},
        index=idx,
    )
    price_df.index.name = "timestamp"

    daily = calculate_daily_spreads(price_df)
    monthly = calculate_monthly_spreads(price_df)
    pctls = calculate_spread_percentiles(daily)
    rev = estimate_annual_arbitrage_revenue(daily)
    neg = calculate_negative_price_hours(price_df)

    return price_df, daily, monthly, pctls, rev, neg


class TestExportToExcel:
    def test_creates_valid_xlsx(self, tmp_path: Path, sample_data) -> None:
        price_df, daily, monthly, pctls, rev, neg = sample_data
        out = tmp_path / "test_report.xlsx"

        result_path = export_to_excel(
            zone="DE_LU",
            price_df=price_df,
            daily_spreads=daily,
            monthly_spreads=monthly,
            percentiles=pctls,
            revenue_estimate=rev,
            negative_stats=neg,
            output_path=out,
        )

        assert result_path.exists()
        assert result_path.suffix == ".xlsx"

    def test_contains_expected_sheets(self, tmp_path: Path, sample_data) -> None:
        price_df, daily, monthly, pctls, rev, neg = sample_data
        out = tmp_path / "test_report.xlsx"

        export_to_excel(
            zone="DE_LU",
            price_df=price_df,
            daily_spreads=daily,
            monthly_spreads=monthly,
            percentiles=pctls,
            revenue_estimate=rev,
            negative_stats=neg,
            output_path=out,
        )

        wb = load_workbook(out)
        sheet_names = wb.sheetnames
        assert "Summary" in sheet_names
        assert "Daily Spreads" in sheet_names
        assert "Monthly Summary" in sheet_names
        assert "Hourly Prices" in sheet_names
        assert "Price Heatmap" in sheet_names

    def test_summary_has_zone(self, tmp_path: Path, sample_data) -> None:
        price_df, daily, monthly, pctls, rev, neg = sample_data
        out = tmp_path / "test_report.xlsx"

        export_to_excel(
            zone="FR",
            price_df=price_df,
            daily_spreads=daily,
            monthly_spreads=monthly,
            percentiles=pctls,
            revenue_estimate=rev,
            negative_stats=neg,
            output_path=out,
        )

        wb = load_workbook(out)
        ws = wb["Summary"]
        values = [ws.cell(row=r, column=2).value for r in range(1, 20)]
        assert "FR" in values


class TestExportToBytes:
    def test_returns_bytes(self, sample_data) -> None:
        price_df, daily, monthly, pctls, rev, neg = sample_data
        result = export_to_bytes(
            zone="DE_LU",
            price_df=price_df,
            daily_spreads=daily,
            monthly_spreads=monthly,
            percentiles=pctls,
            revenue_estimate=rev,
            negative_stats=neg,
        )
        assert isinstance(result, bytes)
        assert len(result) > 0
        # xlsx magic bytes: PK (zip)
        assert result[:2] == b"PK"

    def test_with_timezone(self, sample_data) -> None:
        """Export with tz parameter should produce valid xlsx with timezone in summary."""
        price_df, daily, monthly, pctls, rev, neg = sample_data
        result = export_to_bytes(
            zone="DE_LU",
            price_df=price_df,
            daily_spreads=daily,
            monthly_spreads=monthly,
            percentiles=pctls,
            revenue_estimate=rev,
            negative_stats=neg,
            tz="Europe/Berlin",
        )
        assert isinstance(result, bytes)
        assert len(result) > 0

        from io import BytesIO
        wb = load_workbook(BytesIO(result))
        ws = wb["Summary"]
        values = [ws.cell(row=r, column=2).value for r in range(1, 20)]
        assert "Europe/Berlin" in values
