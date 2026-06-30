"""Tests for source traceability and data-quality summaries."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_trust import (
    ACTIVATION_SOURCE_COLUMNS,
    CAPACITY_SOURCE_COLUMNS,
    COVERAGE_MATRIX_COLUMNS,
    IMBALANCE_SOURCE_COLUMNS,
    build_activation_source_table,
    build_capacity_source_table,
    build_coverage_matrix,
    build_imbalance_source_table,
    build_intraday_source_table,
    build_zone_data_quality_table,
    source_label_for_zone,
)


def _clean_da_frame(hours: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "price_eur_mwh": [50.0] * hours,
            "filled": [False] * hours,
            "imputed": [False] * hours,
        },
        index=idx,
    )


def _capacity_anc() -> pd.DataFrame:
    idx = pd.to_datetime(["2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z"], utc=True)
    return pd.DataFrame(
        {
            "product_type": ["FCR", "aFRR Up"],
            "capacity_price_eur_mw": [12.0, 8.0],
            "energy_price_eur_mwh": [float("nan"), float("nan")],
        },
        index=idx,
    )


def test_source_label_distinguishes_elexon_from_entsoe() -> None:
    assert "Elexon" in source_label_for_zone("GB")
    assert "ENTSO-E" in source_label_for_zone("DE_LU")


def test_intraday_source_table_labels_sources() -> None:
    sources = {
        ("DE_LU", 1): {
            "source": "Manual CSV",
            "rows": 24,
            "first": pd.Timestamp("2026-01-01", tz="UTC"),
            "last": pd.Timestamp("2026-01-01 23:00", tz="UTC"),
            "imported_at": "2026-06-18T10:00:00+00:00",
        },
        ("NL", 2): {
            "source": "ENTSO-E intraday auction",
            "rows": 5, "first": pd.NaT, "last": pd.NaT, "imported_at": None,
        },
    }
    table = build_intraday_source_table(sources)
    assert len(table) == 2
    # Sorted by (zone, sequence): DE_LU/1 then NL/2.
    assert table.iloc[0]["zone"] == "DE_LU"
    assert table.iloc[0]["source"] == "Manual CSV"
    assert table.iloc[0]["rows"] == 24
    assert table.iloc[1]["source"] == "ENTSO-E intraday auction"


def test_intraday_source_table_empty_when_no_sources() -> None:
    assert build_intraday_source_table({}).empty


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


def test_coverage_matrix_combines_da_ida_and_reserve_streams() -> None:
    matrix = build_coverage_matrix(
        {"DE_LU": _clean_da_frame(24), "FR": _clean_da_frame(12)},
        intraday_sources={
            ("DE_LU", 1): {"source": "Manual CSV", "rows": 96},
        },
        ancillary_df=_capacity_anc(),
        capacity_sources={},  # no persisted unified-import capacity -> session fallback
        activation_sources={},
        imbalance_sources={},
        primary_zone="DE_LU",
    )
    assert list(matrix.columns) == COVERAGE_MATRIX_COLUMNS
    # Sorted by zone: DE_LU then FR.
    assert list(matrix["zone"]) == ["DE_LU", "FR"]
    de = matrix.iloc[0]
    assert de["DA"] == "100% (24/24)"
    assert de["IDA1"] == "Manual CSV (96)"
    assert de["IDA2"] == "—" and de["IDA3"] == "—"
    # With no persisted capacity, the reserve cell falls back to the primary
    # zone's session ancillary products.
    assert de["reserve_capacity"] == "FCR, aFRR Up"
    assert de["imbalance_settlement"] == "—"
    fr = matrix.iloc[1]
    assert fr["DA"] == "100% (12/12)"
    assert fr["IDA1"] == "—"
    assert fr["reserve_capacity"] == "—"


def test_coverage_matrix_shows_persisted_reserve_per_zone() -> None:
    # The 6b unified-import sidecar is zone-tagged, so reserve is shown PER zone
    # (not just the primary) — fixing the primary-zone-only limitation.
    matrix = build_coverage_matrix(
        {"DE_LU": _clean_da_frame(24), "FR": _clean_da_frame(12)},
        intraday_sources={},
        ancillary_df=None,
        capacity_sources={
            ("DE_LU", "FCR", "symmetric"): {"source": "Manual CSV", "rows": 6},
            ("DE_LU", "aFRR", "up"): {"source": "TSO API", "rows": 6},
            ("FR", "aFRR", "down"): {"source": "Manual CSV", "rows": 6},
        },
        activation_sources={},
        imbalance_sources={},
        primary_zone="DE_LU",
    )
    by_zone = dict(zip(matrix["zone"], matrix["reserve_capacity"], strict=True))
    # DE_LU shows its two distinct products (deduped, sorted); FR shows its own.
    assert by_zone["DE_LU"] == "FCR, aFRR"
    assert by_zone["FR"] == "aFRR"


def test_coverage_matrix_persisted_capacity_overrides_session_fallback() -> None:
    # When a zone has persisted capacity, it wins over the session ancillary_df.
    matrix = build_coverage_matrix(
        {"DE_LU": _clean_da_frame(24)},
        intraday_sources={},
        ancillary_df=_capacity_anc(),  # session products for DE_LU
        capacity_sources={("DE_LU", "mFRR", "up"): {"source": "TSO API", "rows": 6}},
        activation_sources={},
        imbalance_sources={},
        primary_zone="DE_LU",
    )
    assert matrix.iloc[0]["reserve_capacity"] == "mFRR"


def test_coverage_matrix_includes_ida_only_cached_zone() -> None:
    # A zone with cached IDA provenance but no DA loaded this run still appears.
    matrix = build_coverage_matrix(
        {},
        intraday_sources={("NL", 1): {"source": "ENTSO-E intraday auction", "rows": 24}},
        ancillary_df=None,
        capacity_sources={},
        activation_sources={},
        imbalance_sources={},
        primary_zone="DE_LU",
    )
    assert list(matrix["zone"]) == ["NL"]
    assert matrix.iloc[0]["DA"] == "—"
    assert matrix.iloc[0]["IDA1"] == "ENTSO-E intraday auction (24)"
    assert matrix.iloc[0]["reserve_capacity"] == "—"


def test_coverage_matrix_empty_when_nothing_loaded() -> None:
    matrix = build_coverage_matrix(
        {}, intraday_sources={}, ancillary_df=None, capacity_sources={},
        activation_sources={}, imbalance_sources={},
    )
    assert matrix.empty
    assert list(matrix.columns) == COVERAGE_MATRIX_COLUMNS


def test_capacity_source_table_from_sources() -> None:
    table = build_capacity_source_table({
        ("DE_LU", "FCR", "symmetric"): {
            "source": "Manual CSV", "rows": 6,
            "first": pd.Timestamp("2026-05-01", tz="UTC"),
            "last": pd.Timestamp("2026-05-01 20:00", tz="UTC"),
            "imported_at": "2026-06-28T10:00:00+00:00",
        },
        ("FR", "aFRR", "up"): {
            "source": "TSO API", "rows": 6,
            "first": pd.NaT, "last": pd.NaT, "imported_at": None,
        },
    })
    assert list(table.columns) == CAPACITY_SOURCE_COLUMNS
    # Sorted by (zone, product, direction): DE_LU/FCR first.
    assert list(table["zone"]) == ["DE_LU", "FR"]
    assert table.iloc[0]["product"] == "FCR"
    assert table.iloc[0]["direction"] == "symmetric"
    assert table.iloc[0]["source"] == "Manual CSV"
    assert table.iloc[0]["rows"] == 6
    assert table.iloc[1]["source"] == "TSO API"


def test_capacity_source_table_tolerates_malformed_metadata() -> None:
    table = build_capacity_source_table({
        ("DE_LU", "FCR", "symmetric"): None,
    })
    assert len(table) == 1
    row = table.iloc[0]
    assert row["zone"] == "DE_LU"
    assert row["source"] == "Manual CSV"
    assert row["rows"] == 0
    assert pd.isna(row["first_timestamp_utc"])


def test_capacity_source_table_empty_when_no_sources() -> None:
    assert build_capacity_source_table({}).empty


def test_coverage_matrix_shows_activation_per_zone() -> None:
    # The activation sidecar is zone-tagged, so the activation column is shown
    # PER zone (deduped products, sorted), like the reserve column.
    matrix = build_coverage_matrix(
        {"DE_LU": _clean_da_frame(24), "FR": _clean_da_frame(12)},
        intraday_sources={},
        ancillary_df=None,
        capacity_sources={},
        activation_sources={
            ("DE_LU", "aFRR", "up"): {"source": "Manual CSV", "rows": 6},
            ("DE_LU", "mFRR", "down"): {"source": "Manual CSV", "rows": 6},
            ("FR", "aFRR", "up"): {"source": "TSO API", "rows": 6},
        },
        imbalance_sources={},
        primary_zone="DE_LU",
    )
    by_zone = dict(zip(matrix["zone"], matrix["activation_energy"], strict=True))
    assert by_zone["DE_LU"] == "aFRR, mFRR"
    assert by_zone["FR"] == "aFRR"
    # Activation is a separate stream from reserve (empty here).
    assert set(matrix["reserve_capacity"]) == {"—"}


def test_coverage_matrix_includes_activation_only_zone() -> None:
    # A zone present only in the activation sidecar still appears in the matrix.
    matrix = build_coverage_matrix(
        {},
        intraday_sources={},
        ancillary_df=None,
        capacity_sources={},
        activation_sources={
            ("DE_LU", "aFRR", "up"): {"source": "Manual CSV", "rows": 6},
        },
        imbalance_sources={},
        primary_zone=None,
    )
    assert list(matrix["zone"]) == ["DE_LU"]
    assert matrix.iloc[0]["activation_energy"] == "aFRR"
    assert matrix.iloc[0]["DA"] == "—"
    assert matrix.iloc[0]["reserve_capacity"] == "—"


def test_activation_source_table_from_sources() -> None:
    table = build_activation_source_table({
        ("DE_LU", "aFRR", "up"): {
            "source": "Manual CSV", "rows": 6,
            "first": pd.Timestamp("2026-05-01", tz="UTC"),
            "last": pd.Timestamp("2026-05-01 20:00", tz="UTC"),
            "imported_at": "2026-06-29T10:00:00+00:00",
        },
        ("FR", "mFRR", "down"): {
            "source": "TSO API", "rows": 6,
            "first": pd.NaT, "last": pd.NaT, "imported_at": None,
        },
    })
    assert list(table.columns) == ACTIVATION_SOURCE_COLUMNS
    # Sorted by (zone, product, direction): DE_LU/aFRR first.
    assert list(table["zone"]) == ["DE_LU", "FR"]
    assert table.iloc[0]["product"] == "aFRR"
    assert table.iloc[0]["direction"] == "up"
    assert table.iloc[0]["source"] == "Manual CSV"
    assert table.iloc[0]["rows"] == 6
    assert table.iloc[1]["source"] == "TSO API"


def test_activation_source_table_tolerates_malformed_metadata() -> None:
    table = build_activation_source_table({
        ("DE_LU", "aFRR", "up"): None,
    })
    assert len(table) == 1
    row = table.iloc[0]
    assert row["zone"] == "DE_LU"
    assert row["source"] == "Manual CSV"
    assert row["rows"] == 0
    assert pd.isna(row["first_timestamp_utc"])


def test_activation_source_table_empty_when_no_sources() -> None:
    assert build_activation_source_table({}).empty


def test_coverage_matrix_shows_imbalance_per_zone() -> None:
    # reBAP / imbalance provenance is keyed by zone only, so the coverage cell
    # is source + row count (not a product list).
    matrix = build_coverage_matrix(
        {"DE_LU": _clean_da_frame(24), "FR": _clean_da_frame(12)},
        intraday_sources={},
        ancillary_df=None,
        capacity_sources={},
        activation_sources={},
        imbalance_sources={
            "DE_LU": {"source": "Manual CSV", "rows": 96},
            "FR": {"source": "TSO API", "rows": 48},
        },
        primary_zone="DE_LU",
    )
    by_zone = dict(zip(matrix["zone"], matrix["imbalance_settlement"], strict=True))
    assert by_zone["DE_LU"] == "Manual CSV (96)"
    assert by_zone["FR"] == "TSO API (48)"
    # Imbalance is a separate stream from reserve/activation (empty here).
    assert set(matrix["reserve_capacity"]) == {"—"}
    assert set(matrix["activation_energy"]) == {"—"}


def test_coverage_matrix_includes_imbalance_only_zone() -> None:
    matrix = build_coverage_matrix(
        {},
        intraday_sources={},
        ancillary_df=None,
        capacity_sources={},
        activation_sources={},
        imbalance_sources={"DE_LU": {"source": "Manual CSV", "rows": 96}},
        primary_zone=None,
    )
    assert list(matrix["zone"]) == ["DE_LU"]
    assert matrix.iloc[0]["DA"] == "—"
    assert matrix.iloc[0]["imbalance_settlement"] == "Manual CSV (96)"


def test_imbalance_source_table_from_sources() -> None:
    table = build_imbalance_source_table({
        "DE_LU": {
            "source": "Manual CSV", "rows": 96,
            "first": pd.Timestamp("2026-05-01", tz="UTC"),
            "last": pd.Timestamp("2026-05-01 23:45", tz="UTC"),
            "imported_at": "2026-06-30T10:00:00+00:00",
        },
        "FR": {
            "source": "TSO API", "rows": 48,
            "first": pd.NaT, "last": pd.NaT, "imported_at": None,
        },
    })
    assert list(table.columns) == IMBALANCE_SOURCE_COLUMNS
    assert list(table["zone"]) == ["DE_LU", "FR"]
    assert table.iloc[0]["source"] == "Manual CSV"
    assert table.iloc[0]["rows"] == 96
    assert table.iloc[1]["source"] == "TSO API"


def test_imbalance_source_table_tolerates_malformed_metadata() -> None:
    table = build_imbalance_source_table({"DE_LU": None})
    assert len(table) == 1
    row = table.iloc[0]
    assert row["zone"] == "DE_LU"
    assert row["source"] == "Manual CSV"
    assert row["rows"] == 0
    assert pd.isna(row["first_timestamp_utc"])


def test_imbalance_source_table_empty_when_no_sources() -> None:
    assert build_imbalance_source_table({}).empty
