"""Tests for the passive imbalance-settlement replay overlay (Step 4d-1)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.imbalance_overlay import compute_imbalance_overlay


def _frame(rows: list[tuple]) -> pd.DataFrame:
    """rows: list of (timestamp, imbalance_price, system_imbalance_mw)."""
    idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "imbalance_price_eur_mwh": [r[1] for r in rows],
            "system_imbalance_volume_mw": [r[2] for r in rows],
        },
        index=idx,
    )


def test_none_or_empty_returns_zero_overlay() -> None:
    out = compute_imbalance_overlay(None, power_mw=10, capture_share=0.01)
    assert out["imbalance_settlement_overlay_eur"] == 0.0
    assert out["by_system_state"].empty

    empty = compute_imbalance_overlay(
        _frame([]), power_mw=10, capture_share=0.01,
    )
    assert empty["imbalance_settlement_overlay_eur"] == 0.0


def test_capture_share_zero_returns_zero() -> None:
    f = _frame([
        ("2026-05-01T00:00:00Z", 100.0, 1000.0),
        ("2026-05-01T00:15:00Z", 100.0, 1000.0),
    ])
    out = compute_imbalance_overlay(f, power_mw=10, capture_share=0.0)
    assert out["imbalance_settlement_overlay_eur"] == 0.0


def test_positive_system_imbalance_means_short_and_discharge_helps() -> None:
    # German Netztransparenz convention: positive NRV-Saldo = system short.
    # The helping BESS position is positive net dispatch (discharge/injection).
    f = _frame([
        ("2026-05-01T00:00:00Z", 100.0, 1000.0),
        ("2026-05-01T00:15:00Z", 100.0, 1000.0),
    ])
    out = compute_imbalance_overlay(f, power_mw=10, capture_share=0.01)
    row = out["by_system_state"].iloc[0]
    # min(10 MW, 1% * 1000 MW) = 10 MW; 0.25h * 2 rows = 5 MWh.
    assert row["system_state"] == "system_short"
    assert row["asset_imbalance_mwh"] == pytest.approx(5.0)
    assert out["imbalance_settlement_overlay_eur"] == pytest.approx(500.0)


def test_negative_system_imbalance_means_long_and_charge_helps() -> None:
    # Negative system imbalance means long/overcovered; helping is a negative
    # BESS net dispatch (charge/less injection), so a positive price is a cost.
    f = _frame([
        ("2026-05-01T00:00:00Z", 100.0, -1000.0),
        ("2026-05-01T00:15:00Z", 100.0, -1000.0),
    ])
    out = compute_imbalance_overlay(f, power_mw=10, capture_share=0.01)
    row = out["by_system_state"].iloc[0]
    assert row["system_state"] == "system_long"
    assert row["asset_imbalance_mwh"] == pytest.approx(-5.0)
    assert out["imbalance_settlement_overlay_eur"] == pytest.approx(-500.0)


def test_negative_price_kept_as_cashflow_price_no_sign_flip() -> None:
    # When the system is long, helping means negative dispatch. A negative reBAP
    # price then produces positive cash flow via signed_dispatch * signed_price.
    f = _frame([
        ("2026-05-01T00:00:00Z", -50.0, -1000.0),
        ("2026-05-01T00:15:00Z", -50.0, -1000.0),
    ])
    out = compute_imbalance_overlay(f, power_mw=10, capture_share=0.01)
    assert out["imbalance_settlement_overlay_eur"] == pytest.approx(250.0)


def test_system_volume_exceeding_power_is_clipped() -> None:
    f = _frame([
        ("2026-05-01T00:00:00Z", 80.0, 5000.0),
        ("2026-05-01T00:15:00Z", 80.0, 5000.0),
    ])
    out = compute_imbalance_overlay(f, power_mw=7, capture_share=0.5)
    # capture_share*vol = 2500 MW, capped at 7 MW; 2 * 0.25h = 0.5h.
    row = out["by_system_state"].iloc[0]
    assert row["asset_imbalance_mwh"] == pytest.approx(3.5)
    assert out["imbalance_settlement_overlay_eur"] == pytest.approx(280.0)


def test_signed_states_aggregate_separately() -> None:
    f = _frame([
        ("2026-05-01T00:00:00Z", 100.0, 1000.0),
        ("2026-05-01T00:15:00Z", 100.0, -1000.0),
        ("2026-05-01T00:30:00Z", 100.0, 0.0),
    ])
    out = compute_imbalance_overlay(f, power_mw=10, capture_share=0.01)
    by = {
        row.system_state: row
        for row in out["by_system_state"].itertuples(index=False)
    }
    assert by["system_short"].asset_imbalance_mwh == pytest.approx(2.5)
    assert by["system_long"].asset_imbalance_mwh == pytest.approx(-2.5)
    assert by["neutral"].asset_imbalance_mwh == pytest.approx(0.0)
    assert out["imbalance_settlement_overlay_eur"] == pytest.approx(0.0)


def test_interval_hours_inferred_and_overridable() -> None:
    f = _frame([
        ("2026-05-01T00:00:00Z", 100.0, 1000.0),
        ("2026-05-01T01:00:00Z", 100.0, 1000.0),
    ])
    inferred = compute_imbalance_overlay(f, power_mw=10, capture_share=0.01)
    # 1h spacing -> 10 MW * 1h * 2 rows = 20 MWh.
    assert inferred["imbalance_settlement_overlay_eur"] == pytest.approx(2000.0)

    override = compute_imbalance_overlay(
        f, power_mw=10, capture_share=0.01, interval_hours=0.25,
    )
    assert override["imbalance_settlement_overlay_eur"] == pytest.approx(500.0)


def test_duplicate_timestamps_do_not_crash_and_infer_gap() -> None:
    f = _frame([
        ("2026-05-01T00:00:00Z", 100.0, 1000.0),
        ("2026-05-01T00:00:00Z", 100.0, 1000.0),
        ("2026-05-01T00:15:00Z", 100.0, 1000.0),
    ])
    out = compute_imbalance_overlay(f, power_mw=10, capture_share=0.01)
    # The zero duplicate gap is ignored; modal positive gap is 0.25h.
    assert out["imbalance_settlement_overlay_eur"] == pytest.approx(750.0)


def test_missing_required_columns_returns_zero() -> None:
    f = pd.DataFrame(
        {"imbalance_price_eur_mwh": [100.0]},
        index=pd.DatetimeIndex(["2026-05-01T00:00:00Z"], tz="UTC"),
    )
    out = compute_imbalance_overlay(f, power_mw=10, capture_share=0.01)
    assert out["imbalance_settlement_overlay_eur"] == 0.0


def test_total_is_named_overlay_not_revenue() -> None:
    out = compute_imbalance_overlay(None, power_mw=10, capture_share=0.01)
    assert "imbalance_settlement_overlay_eur" in out
    assert "sign_convention" in out
    assert not any("revenue" in k for k in out)
