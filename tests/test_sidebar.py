"""Tests for sidebar-only helpers."""

from __future__ import annotations

from src.components.sidebar import (
    DURATION_PRESET_HOURS,
    _format_duration_option,
    _looks_like_unified_capacity_csv,
)


def test_sidebar_duration_presets_cover_longer_bess_cases() -> None:
    assert DURATION_PRESET_HOURS == (1.0, 2.0, 4.0, 6.0, 8.0)
    assert [_format_duration_option(h) for h in DURATION_PRESET_HOURS] == [
        "1", "2", "4", "6", "8",
    ]


def test_sidebar_detects_unified_capacity_csv_schema() -> None:
    csv_text = (
        "# comment header\n"
        "timestamp,zone,product,direction,capacity_price_eur_mw_h\n"
        "2026-06-22T22:00:00Z,DE_LU,FCR,symmetric,12.0\n"
    )

    assert _looks_like_unified_capacity_csv(csv_text)


def test_sidebar_does_not_flag_legacy_ancillary_template_as_unified() -> None:
    csv_text = (
        "date,product,capacity_price_eur_mw\n"
        "2026-06-22,NEGPOS_00_04,12.0\n"
    )

    assert not _looks_like_unified_capacity_csv(csv_text)
