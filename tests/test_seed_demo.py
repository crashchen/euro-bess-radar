"""Tests for the Phase 9.2b demo-data seed script (scripts/seed_demo_9_2b.py).

These guard the two contracts the script depends on: the synthetic frames must
match what ``write_cache`` / ``write_intraday_cache`` expect, and the reserve
CSV must parse through the real ancillary DE_FCR template into a capacity
product. They do not touch the real cache (DB paths are monkeypatched).
"""

from __future__ import annotations

import sqlite3
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from scripts import seed_demo_9_2b as seed
from src.ancillary import list_capacity_products, parse_ancillary_csv


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_demo_da_frame_matches_write_cache_contract() -> None:
    idx = seed._window(3)
    da = seed.build_demo_da_frame(idx, _rng())
    # write_cache reset_index()es the index to a "timestamp" column and reads
    # price_eur_mwh; a "filled" column would drop synthetic rows on persist.
    assert da.index.name == "timestamp"
    assert list(da.columns) == ["price_eur_mwh"]
    assert "filled" not in da.columns
    assert len(da) == 3 * 24
    assert str(da.index.tz) == "UTC"


def test_demo_ida_frame_has_intraday_column_on_same_index() -> None:
    idx = seed._window(3)
    da = seed.build_demo_da_frame(idx, _rng())
    ida = seed.build_demo_ida_frame(da, _rng())
    assert list(ida.columns) == ["intraday_price_eur_mwh"]
    assert ida.index.equals(da.index)


def test_demo_reserve_csv_parses_to_block_granular_fcr() -> None:
    idx = pd.date_range("2025-06-01", periods=48, freq="h", tz="UTC")
    csv_text = seed.build_demo_reserve_csv(idx, _rng())
    raw = pd.read_csv(StringIO(csv_text))
    # Berlin summer time: a UTC-midnight window starts at 02:00 local, so the
    # first German 4h reserve block begins at 04:00 local rather than 00:00 UTC.
    assert pd.to_datetime(raw["date"]).dt.hour.iloc[0] == 4
    parsed = parse_ancillary_csv(csv_text, template_key="DE_FCR")
    assert list_capacity_products(parsed) == ["FCR"]
    # 6 four-hour blocks per day -> block-granular, not one row per day.
    assert len(parsed) == 2 * 6
    assert parsed["capacity_price_eur_mw"].notna().all()


def test_seed_and_clean_round_trip(tmp_path, monkeypatch) -> None:
    db = tmp_path / "cache" / "demo.db"
    monkeypatch.setattr("src.data_ingestion.DB_PATH", db)
    monkeypatch.setattr("src.data_ingestion.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(seed, "DB_PATH", db)
    monkeypatch.setattr(seed, "SAMPLES_DIR", tmp_path / "samples")
    monkeypatch.setattr(seed, "RESERVE_CSV", tmp_path / "samples" / "reserve.csv")

    seed.seed(days=4, rng_seed=1)

    assert seed.RESERVE_CSV.exists()
    with sqlite3.connect(db) as conn:
        da_rows = conn.execute('SELECT COUNT(*) FROM "da_prices_de_lu"').fetchone()[0]
        ida_rows = conn.execute(
            'SELECT COUNT(*) FROM "ida_prices_de_lu_seq1"',
        ).fetchone()[0]
    assert da_rows == 4 * 24
    assert ida_rows == 4 * 24

    seed.clean()
    assert not seed.RESERVE_CSV.exists()
    with sqlite3.connect(db) as conn:
        remaining = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%de_lu%'",
        ).fetchall()
    assert remaining == []


def test_seed_refuses_to_overwrite_unmarked_real_cache(tmp_path, monkeypatch) -> None:
    db = tmp_path / "cache" / "demo.db"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            'CREATE TABLE "da_prices_de_lu" '
            "(timestamp TEXT PRIMARY KEY, price_eur_mwh REAL, zone TEXT)"
        )
        conn.execute(
            'INSERT INTO "da_prices_de_lu" VALUES '
            "('2026-01-01T00:00:00+00:00', 99.0, 'DE_LU')"
        )

    monkeypatch.setattr("src.data_ingestion.DB_PATH", db)
    monkeypatch.setattr("src.data_ingestion.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(seed, "DB_PATH", db)
    monkeypatch.setattr(seed, "SAMPLES_DIR", tmp_path / "samples")
    monkeypatch.setattr(seed, "RESERVE_CSV", tmp_path / "samples" / "reserve.csv")

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        seed.seed(days=2, rng_seed=1)

    with sqlite3.connect(db) as conn:
        rows = conn.execute('SELECT COUNT(*) FROM "da_prices_de_lu"').fetchone()[0]
        marker_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (seed.DEMO_MARKER_TABLE,),
        ).fetchone()
    assert rows == 1
    assert marker_exists is None
