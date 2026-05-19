"""Tests for source traceability and data-quality summaries."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_trust import build_zone_data_quality_table, source_label_for_zone


def test_source_label_distinguishes_elexon_from_entsoe() -> None:
    assert "Elexon" in source_label_for_zone("GB")
    assert "ENTSO-E" in source_label_for_zone("DE_LU")


def test_quality_table_summarises_gaps_and_imputation() -> None:
    idx = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "price_eur_mwh": [50.0, 51.0, 52.0, float("nan")],
            "filled": [False, True, False, True],
            "imputed": [False, True, False, False],
        },
        index=idx,
    )

    out = build_zone_data_quality_table(
        {"DE_LU": df},
        zone_timezones={"DE_LU": "Europe/Berlin"},
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert row["zone"] == "DE_LU"
    assert row["timezone"] == "Europe/Berlin"
    assert row["total_intervals"] == 4
    assert row["valid_intervals"] == 3
    assert row["coverage_pct"] == pytest.approx(75.0)
    assert row["source_gap_intervals"] == 2
    assert row["imputed_intervals"] == 1
    assert row["missing_intervals"] == 1
    assert row["source_gap_pct"] == pytest.approx(50.0)
    assert row["imputed_pct"] == pytest.approx(25.0)
    assert row["missing_pct"] == pytest.approx(25.0)
    assert row["max_source_gap_hours"] == pytest.approx(1.0)


def test_empty_zone_data_returns_empty_table() -> None:
    out = build_zone_data_quality_table({})
    assert out.empty
