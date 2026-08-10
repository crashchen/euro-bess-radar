"""Leg-specific market-grid registry (contract §4.3, red-line #17)."""

from __future__ import annotations

import datetime as dt

import pytest

from src.project_case import grid


def test_da_resolution_regimes():
    # SDAC zone: 60-min before the 15-min delivery day, 15-min from it.
    assert len(grid.expected_da_timestamps("DE_LU", dt.date(2025, 6, 5))) == 24
    assert len(grid.expected_da_timestamps("DE_LU", dt.date(2026, 1, 5))) == 96
    # IE_SEM stays 30-min; CH stays 60-min (outside SDAC rollout).
    assert len(grid.expected_da_timestamps("IE_SEM", dt.date(2026, 1, 5))) == 48
    assert len(grid.expected_da_timestamps("CH", dt.date(2026, 1, 5))) == 24


def test_da_dst_transitions_counts():
    # Spring forward 2025-03-30 is pre-SDAC-transition (60-min) -> 23 local hours.
    assert len(grid.expected_da_timestamps("DE_LU", dt.date(2025, 3, 30))) == 23
    # Fall back 2025-10-26 is post-SDAC-transition (15-min) -> 25h * 4 = 100 stamps,
    # exercising DST composed with the 15-min regime.
    assert len(grid.expected_da_timestamps("DE_LU", dt.date(2025, 10, 26))) == 100
    # CH stays 60-min, so its fall-back day is 25 hourly stamps.
    assert len(grid.expected_da_timestamps("CH", dt.date(2025, 10, 26))) == 25


def test_da_profiles_and_unsupported_gb():
    assert grid.da_profile_id("DE_LU") == grid.DA_PROFILE_SDAC
    assert grid.da_profile_id("IE_SEM") == grid.DA_PROFILE_IE_SEM
    assert grid.da_profile_id("CH") == grid.DA_PROFILE_CH
    assert grid.da_profile_id("GB") is None  # Elexon/GBP: no DA registry entry
    assert grid.expected_da_timestamps("GB", dt.date(2026, 1, 5)) is None


def test_ida_support_and_resolution():
    for z in ("DE_LU", "NL", "BE", "FR", "AT", "IT_NORD"):
        assert grid.ida_profile_id(z) == grid.IDA_PROFILE
        assert len(grid.expected_ida_timestamps(z, dt.date(2026, 1, 5))) == 96
    assert grid.ida_profile_id("ES") is None
    assert grid.expected_ida_timestamps("ES", dt.date(2026, 1, 5)) is None


def test_reserve_blocks_shape_and_ids():
    for z in ("DE_LU", "FI"):
        blocks = grid.reserve_blocks(z, dt.date(2026, 1, 5))
        assert len(blocks) == 6
        assert all(dur == 4.0 for _, dur in blocks)
        assert all(bid.endswith("Z") and "T" in bid for bid, _ in blocks)
    assert grid.reserve_blocks("FR", dt.date(2026, 1, 5)) is None


def test_registry_version_literal():
    assert grid.REGISTRY_VERSION == "pc-market-grid-v1"


def test_profile_id_dispatch_and_unknown_leg():
    assert grid.profile_id("da", "DE_LU") == grid.DA_PROFILE_SDAC
    assert grid.profile_id("ida", "DE_LU") == grid.IDA_PROFILE
    assert grid.profile_id("reserve", "DE_LU") == grid.RESERVE_PROFILE
    with pytest.raises(ValueError):
        grid.profile_id("bogus", "DE_LU")
