"""Tests for the activation-energy replay overlay (Step 3c-2a)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.activation_overlay import compute_activation_overlay


def _frame(rows: list[tuple]) -> pd.DataFrame:
    """rows: list of (timestamp, product, direction, price, system_volume_mw)."""
    idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "product_type": [r[1] for r in rows],
            "direction": [r[2] for r in rows],
            "activation_price_eur_mwh": [r[3] for r in rows],
            "system_activated_volume_mw": [r[4] for r in rows],
        },
        index=idx,
    )


def test_none_or_empty_returns_zero_overlay() -> None:
    out = compute_activation_overlay(None, reserve_mw=10, capture_share=0.5)
    assert out["activation_energy_overlay_eur"] == 0.0
    assert out["by_stream"].empty
    empty = compute_activation_overlay(
        _frame([]), reserve_mw=10, capture_share=0.5,
    )
    assert empty["activation_energy_overlay_eur"] == 0.0


def test_capture_share_zero_returns_zero() -> None:
    f = _frame([
        ("2026-05-01T00:00:00Z", "aFRR", "up", 100.0, 50.0),
        ("2026-05-01T01:00:00Z", "aFRR", "up", 100.0, 50.0),
    ])
    out = compute_activation_overlay(f, reserve_mw=10, capture_share=0.0)
    assert out["activation_energy_overlay_eur"] == 0.0


def test_volume_exceeding_reserve_is_clipped() -> None:
    # capture_share*vol = 0.5*100 = 50 MW, capped at reserve_mw=10; dt=1h, 2 rows.
    f = _frame([
        ("2026-05-01T00:00:00Z", "aFRR", "up", 100.0, 100.0),
        ("2026-05-01T01:00:00Z", "aFRR", "up", 100.0, 100.0),
    ])
    out = compute_activation_overlay(f, reserve_mw=10, capture_share=0.5)
    row = out["by_stream"].iloc[0]
    assert row["activated_mwh"] == pytest.approx(20.0)  # 10 MW * 1h * 2
    assert out["activation_energy_overlay_eur"] == pytest.approx(2000.0)


def test_down_direction_negative_price_stays_negative_no_sign_flip() -> None:
    # Down activation at a negative price -> negative cash flow, NOT flipped back
    # positive. The price is already a cash-flow price; direction is just a label.
    f = _frame([
        ("2026-05-01T00:00:00Z", "aFRR", "down", -20.0, 4.0),
        ("2026-05-01T01:00:00Z", "aFRR", "down", -20.0, 4.0),
    ])
    out = compute_activation_overlay(f, reserve_mw=10, capture_share=1.0)
    # asset_mw = min(10, 1.0*4) = 4; 4 MWh/row * 2 = 8 MWh; * -20 = -160.
    assert out["activation_energy_overlay_eur"] == pytest.approx(-160.0)
    assert out["by_stream"].iloc[0]["direction"] == "down"


def test_down_direction_positive_price_is_positive() -> None:
    f = _frame([
        ("2026-05-01T00:00:00Z", "aFRR", "down", 15.0, 4.0),
        ("2026-05-01T01:00:00Z", "aFRR", "down", 15.0, 4.0),
    ])
    out = compute_activation_overlay(f, reserve_mw=10, capture_share=1.0)
    assert out["activation_energy_overlay_eur"] == pytest.approx(120.0)  # 8 MWh * 15


def test_negative_system_volume_treated_as_zero() -> None:
    f = _frame([
        ("2026-05-01T00:00:00Z", "aFRR", "up", 100.0, -50.0),
        ("2026-05-01T01:00:00Z", "aFRR", "up", 100.0, -50.0),
    ])
    out = compute_activation_overlay(f, reserve_mw=10, capture_share=0.5)
    assert out["activation_energy_overlay_eur"] == 0.0


def test_breakdown_per_product_and_direction() -> None:
    f = _frame([
        ("2026-05-01T00:00:00Z", "aFRR", "up", 100.0, 4.0),
        ("2026-05-01T01:00:00Z", "aFRR", "up", 100.0, 4.0),
        ("2026-05-01T00:00:00Z", "mFRR", "down", 50.0, 4.0),
        ("2026-05-01T01:00:00Z", "mFRR", "down", 50.0, 4.0),
    ])
    out = compute_activation_overlay(f, reserve_mw=10, capture_share=1.0)
    by = out["by_stream"]
    assert set(zip(by["product"], by["direction"], strict=True)) == {
        ("aFRR", "up"), ("mFRR", "down"),
    }
    # aFRR up: 8 MWh * 100 = 800; mFRR down: 8 MWh * 50 = 400; total 1200.
    assert out["activation_energy_overlay_eur"] == pytest.approx(1200.0)


def test_interval_hours_inferred_and_overridable() -> None:
    # 4h block spacing -> inferred dt = 4h.
    f = _frame([
        ("2026-05-01T00:00:00Z", "aFRR", "up", 100.0, 5.0),
        ("2026-05-01T04:00:00Z", "aFRR", "up", 100.0, 5.0),
    ])
    inferred = compute_activation_overlay(f, reserve_mw=10, capture_share=1.0)
    # asset_mw = min(10, 5) = 5; dt=4h; 2 rows -> 5*4*2 = 40 MWh; * 100 = 4000.
    assert inferred["activation_energy_overlay_eur"] == pytest.approx(4000.0)
    # Explicit override wins.
    override = compute_activation_overlay(
        f, reserve_mw=10, capture_share=1.0, interval_hours=1.0,
    )
    assert override["activation_energy_overlay_eur"] == pytest.approx(1000.0)


def test_duplicate_timestamps_do_not_crash_and_infer_gap() -> None:
    # Duplicate timestamps (e.g. an uncleaned feed) must not break interval
    # inference: the zero gap is filtered, the 4h gap still resolves.
    f = _frame([
        ("2026-05-01T00:00:00Z", "aFRR", "up", 100.0, 5.0),
        ("2026-05-01T00:00:00Z", "aFRR", "up", 100.0, 5.0),
        ("2026-05-01T04:00:00Z", "aFRR", "up", 100.0, 5.0),
    ])
    out = compute_activation_overlay(f, reserve_mw=10, capture_share=1.0)
    # dt inferred as 4h; asset_mw = min(10, 5) = 5; 3 rows * 5 MW * 4h = 60 MWh.
    assert out["activation_energy_overlay_eur"] == pytest.approx(6000.0)


def test_total_is_named_overlay_not_revenue() -> None:
    # Red-line: the headline figure must not be labelled strategy/revenue.
    out = compute_activation_overlay(None, reserve_mw=10, capture_share=0.5)
    assert "activation_energy_overlay_eur" in out
    assert not any("revenue" in k for k in out)
