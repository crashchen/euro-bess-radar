"""Leg-specific market-grid registry (contract §4.3, red-line #17)."""

from __future__ import annotations

import datetime as dt

import pandas as pd
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


def test_da_cutover_days_are_mixed_resolution():
    # SDAC switches at the SHARED market instant 2025-09-30T22:00Z, not each zone's
    # civil midnight — so a straddling local day is genuinely mixed (review blocker).
    fi = grid.expected_da_timestamps("FI", dt.date(2025, 10, 1))
    assert len(fi) == 93  # 1 hourly (21:00Z) + 92 quarter-hours from 22:00Z
    assert (fi[1] - fi[0]) == pd.Timedelta(hours=1)
    assert (fi[2] - fi[1]) == pd.Timedelta(minutes=15)
    pt = grid.expected_da_timestamps("PT", dt.date(2025, 9, 30))
    assert len(pt) == 27  # 23 hourly + 4 quarter-hours into the cutover
    assert (pt[-1] - pt[-2]) == pd.Timedelta(minutes=15)


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


def test_reserve_blocks_dst_durations_are_actual_elapsed():
    # Wall-clock 4h blocks tile the civil day exactly, so the transition block is
    # 3h (spring) / 5h (fall), not a nominal 4h (review r2 #5).
    spring = grid.reserve_blocks("DE_LU", dt.date(2025, 3, 30))
    assert [dur for _, dur in spring] == [3.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    assert sum(dur for _, dur in spring) == 23.0
    fall = grid.reserve_blocks("DE_LU", dt.date(2025, 10, 26))
    assert [dur for _, dur in fall] == [5.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    assert sum(dur for _, dur in fall) == 25.0


def test_reserve_blocks_agree_with_ingestion_parser_on_dst_days():
    # The registry and the shipped Regelleistung block parser must build the SAME
    # UTC block-start instants, or an imported reserve price is misclassified as
    # wholly missing on DST days (review r2 #5).
    from src.time_utils import parse_regelleistung_time_block_start

    for day in (dt.date(2025, 3, 30), dt.date(2025, 10, 26), dt.date(2025, 6, 1)):
        for hour, (bid, _dur) in zip(
            (0, 4, 8, 12, 16, 20), grid.reserve_blocks("DE_LU", day), strict=True
        ):
            parsed = parse_regelleistung_time_block_start(
                day, f"{hour:02d}:00", timezone="Europe/Berlin"
            )
            assert parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == bid


def test_registry_version_literal():
    assert grid.REGISTRY_VERSION == "pc-market-grid-v1"


def test_profile_id_dispatch_and_unknown_leg():
    assert grid.profile_id("da", "DE_LU") == grid.DA_PROFILE_SDAC
    assert grid.profile_id("ida", "DE_LU") == grid.IDA_PROFILE
    assert grid.profile_id("reserve", "DE_LU") == grid.RESERVE_PROFILE
    with pytest.raises(ValueError):
        grid.profile_id("bogus", "DE_LU")
