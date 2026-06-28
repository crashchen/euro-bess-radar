"""Seed a self-contained SYNTHETIC DE_LU dataset for validating the Phase 9.2b
Simulation Cockpit panel end-to-end without any live API.

It writes synthetic day-ahead + IDA1 prices into the local SQLite cache (the
same tables the live fetch uses) and emits a reserve-capacity CSV under
``samples/`` for manual upload. The cockpit's forecast-driven IDA policy panel
(DA-only / forecast-driven / ceiling rows) then runs cache-first, and uploading
the reserve CSV lights up the 9.2b rows (DA+IDA1+reserve ceiling + forecast-
driven realistic) and the forecast-effect gap panel.

The data is SYNTHETIC and is labelled ``Synthetic demo`` in Data Trust. It is
for UI/feature validation only, NOT a market estimate.

Usage::

    python scripts/seed_demo_9_2b.py            # seed ~30 days ending today
    python scripts/seed_demo_9_2b.py --days 45
    python scripts/seed_demo_9_2b.py --clean    # drop the demo cache tables
    python scripts/seed_demo_9_2b.py --force    # explicitly overwrite DE_LU cache
"""

from __future__ import annotations

import argparse
import contextlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow ``python scripts/seed_demo_9_2b.py`` from the repo root (adds the repo
# root to the path so ``import src`` resolves); harmless when imported as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DB_PATH
from src.data_ingestion import write_cache, write_intraday_cache

ZONE = "DE_LU"
SYNTHETIC_SOURCE = "Synthetic demo"
DEMO_MARKER_TABLE = "demo_seed_metadata"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
RESERVE_CSV = SAMPLES_DIR / "de_lu_reserve_capacity_sample.csv"


def _window(days: int) -> pd.DatetimeIndex:
    """Hourly UTC index covering ``days`` full days ending at today 00:00 UTC."""
    end = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.Timedelta(days=days)
    return pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")


def build_demo_da_frame(idx: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic DA prices with a double-peak daily shape + weekend dip."""
    local_idx = idx.tz_convert("Europe/Berlin")
    hour = local_idx.hour.to_numpy()
    daily = (
        60.0
        + 30.0 * np.sin((hour - 8) / 24 * 2 * np.pi)
        + 12.0 * np.sin((hour - 18) / 12 * 2 * np.pi)
    )
    weekend = np.where(local_idx.dayofweek.to_numpy() >= 5, -8.0, 0.0)
    price = daily + weekend + rng.normal(0.0, 4.0, len(idx))
    df = pd.DataFrame({"price_eur_mwh": np.round(price, 2)}, index=idx)
    df.index.name = "timestamp"
    return df


def build_demo_ida_frame(da: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """IDA1 = DA + a mean-reverting intraday adjustment (gives rebid signal)."""
    n = len(da)
    adjustment = rng.normal(0.0, 6.0, n) + 4.0 * np.sin(np.arange(n) / 6 * 2 * np.pi)
    ida = da["price_eur_mwh"].to_numpy() + adjustment
    df = pd.DataFrame({"intraday_price_eur_mwh": np.round(ida, 2)}, index=da.index)
    df.index.name = "timestamp"
    return df


def build_demo_reserve_csv(idx: pd.DatetimeIndex, rng: np.random.Generator) -> str:
    """FCR capacity CSV (DE_FCR template) — one row per 4h block per day.

    Block granularity matters: the 9.2b reserve forecast is block-of-day, and
    ``align_reserve_price_to_index`` maps source rows onto target intervals by
    (local date, 4h block).
    """
    local_idx = idx.tz_convert("Europe/Berlin")
    block_starts = local_idx[local_idx.hour % 4 == 0]
    base = 14.0 + 4.0 * np.sin(np.arange(len(block_starts)) / 6 * 2 * np.pi)
    prices = np.clip(base + rng.normal(0.0, 2.0, len(block_starts)), 2.0, None)
    out = pd.DataFrame({
        "date": [ts.strftime("%Y-%m-%d %H:%M") for ts in block_starts],
        "product": "FCR",
        "capacity_price_eur_mw": np.round(prices, 2),
    })
    return out.to_csv(index=False)


def _demo_tables() -> list[str]:
    return [f"da_prices_{ZONE.lower()}", f"ida_prices_{ZONE.lower()}_seq1"]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _demo_marker_exists(conn: sqlite3.Connection) -> bool:
    with contextlib.suppress(sqlite3.OperationalError):
        row = conn.execute(
            f'SELECT 1 FROM "{DEMO_MARKER_TABLE}" WHERE zone = ?',
            (ZONE,),
        ).fetchone()
        return row is not None
    return False


def _write_demo_marker(conn: sqlite3.Connection) -> None:
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{DEMO_MARKER_TABLE}" '
        "(zone TEXT PRIMARY KEY, source TEXT NOT NULL, seeded_at TEXT NOT NULL)"
    )
    conn.execute(
        f'INSERT OR REPLACE INTO "{DEMO_MARKER_TABLE}" '
        "(zone, source, seeded_at) VALUES (?, ?, ?)",
        (ZONE, SYNTHETIC_SOURCE, pd.Timestamp.now(tz="UTC").isoformat()),
    )


def clean(*, force: bool = False) -> None:
    """Drop the demo cache tables and their DE_LU provenance/metadata rows."""
    if not DB_PATH.exists():
        print(f"No cache DB at {DB_PATH}; nothing to clean.")
        return
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        if not _demo_marker_exists(conn) and not force:
            print(
                f"No {SYNTHETIC_SOURCE!r} marker found in {DB_PATH}; refusing "
                f"to drop possible real {ZONE} cache tables. Re-run with "
                "--force --clean only if you intentionally want to remove them."
            )
            return
        for table in _demo_tables():
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        for table, col in [("cache_metadata", "zone"), ("ida_price_sources", "zone")]:
            # The sidecar/metadata table may not exist yet.
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(f'DELETE FROM "{table}" WHERE {col} = ?', (ZONE,))
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(f'DELETE FROM "{DEMO_MARKER_TABLE}" WHERE zone = ?', (ZONE,))
        conn.commit()
    if RESERVE_CSV.exists():
        RESERVE_CSV.unlink()
    print(f"Cleaned demo tables {_demo_tables()} and {RESERVE_CSV.name}.")


def seed(days: int, rng_seed: int, *, force: bool = False) -> None:
    """Generate and persist the synthetic DA + IDA1 + reserve demo dataset."""
    if DB_PATH.exists():
        with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
            has_marker = _demo_marker_exists(conn)
            existing_tables = [table for table in _demo_tables() if _table_exists(conn, table)]
        if existing_tables:
            if not has_marker and not force:
                raise SystemExit(
                    f"Refusing to overwrite existing {ZONE} cache tables "
                    f"{existing_tables}; no {SYNTHETIC_SOURCE!r} marker was "
                    "found. Back up/clear the cache first, or re-run with "
                    "--force if this overwrite is intentional."
                )
            clean(force=True)

    rng = np.random.default_rng(rng_seed)
    idx = _window(days)
    da = build_demo_da_frame(idx, rng)
    ida = build_demo_ida_frame(da, rng)

    write_cache(da, ZONE)
    write_intraday_cache(ida, ZONE, 1, source=SYNTHETIC_SOURCE)

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    RESERVE_CSV.write_text(build_demo_reserve_csv(idx, rng))
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        _write_demo_marker(conn)
        conn.commit()

    start_local = idx.min().tz_convert("Europe/Berlin").date()
    end_local = idx.max().tz_convert("Europe/Berlin").date()
    print(
        f"Seeded SYNTHETIC {ZONE}: {len(da)} DA + IDA1 hourly rows "
        f"({idx.min().date()} .. {idx.max().date()} UTC).\n"
        f"Reserve capacity CSV: {RESERVE_CSV}\n\n"
        "Next steps:\n"
        "  1. streamlit run app.py\n"
        f"  2. Select zone {ZONE} and a date range within "
        f"{start_local} .. {end_local} (local).\n"
        f"  3. Sidebar > Ancillary upload: choose template DE_FCR and upload "
        f"{RESERVE_CSV.name}.\n"
        "  4. Simulation Cockpit > 'Forecast-driven IDA policy' > Run; pick the "
        "reserve product to see the 9.2b rows + forecast-effect gap panel.\n"
        "  5. Data Trust tab: the coverage matrix shows DA/IDA1/reserve for "
        f"{ZONE} (IDA1 labelled '{SYNTHETIC_SOURCE}').\n\n"
        "Cleanup when done:  python scripts/seed_demo_9_2b.py --clean"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30, help="Days of data to seed.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--clean", action="store_true", help="Drop the demo cache tables and exit.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "Allow overwriting/cleaning existing DE_LU cache tables without a "
            "Synthetic demo marker. This can remove real local cache data."
        ),
    )
    args = parser.parse_args()
    if args.clean:
        clean(force=args.force)
        return
    seed(args.days, rng_seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
