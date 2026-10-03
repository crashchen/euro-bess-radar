"""SYNTHETIC DST fixture for manual check 43 (2026-10-03).

Run from an isolated `git archive 68d3486` tree whose data/cache starts empty:
  PYTHONPATH=. python <this file> <outdir>
It writes zero DE_LU DA prices for an ordinary day and both 2025/2026 DST
transition days into THAT tree's cache (never the working repository cache),
and emits two capacity CSVs for upload through the real sidebar:
  - unified capacity import (cache-first Project Case path), FCR symmetric
  - DE_FCR per-country template (session ancillary, Revenue joint MILP)
Both carry EUR 20/MW/h on six local 4h block starts per day.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src import config
from src.data_ingestion import write_cache
from src.project_case import grid

ZONE = "DE_LU"
TZ = "Europe/Berlin"
DAYS = [
    date(2025, 10, 25), date(2025, 10, 26),                    # autumn: 25 h
    date(2026, 3, 27), date(2026, 3, 28), date(2026, 3, 29),   # spring: 23 h
]
PRICE = 20.0


def _da_frame() -> pd.DataFrame:
    stamps = [ts for day in DAYS for ts in grid.expected_da_timestamps(ZONE, day)]
    index = pd.DatetimeIndex(stamps).tz_convert("UTC").rename("timestamp")
    return pd.DataFrame({"price_eur_mwh": 0.0}, index=index)


def _block_starts() -> list[pd.Timestamp]:
    # Local wall-clock block labels (00/04/.../20), as Regelleistung names
    # them. Adding elapsed hours to an aware midnight would shift the labels
    # by the DST offset change (a first-draft fixture bug, caught by the
    # Project Case reserve-coverage audit).
    return [pd.Timestamp(day) + pd.Timedelta(hours=h) for day in DAYS for h in range(0, 24, 4)]


def main(outdir: Path) -> None:
    assert config.DB_PATH.parent.resolve() == (Path.cwd() / "data" / "cache").resolve()
    assert not config.DB_PATH.exists(), f"refusing to seed a non-empty cache: {config.DB_PATH}"
    da = _da_frame()
    write_cache(da, ZONE)
    outdir.mkdir(parents=True, exist_ok=True)
    starts = _block_starts()
    pd.DataFrame({
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M") for ts in starts],
        "zone": ZONE, "product": "FCR", "direction": "symmetric",
        "capacity_price_eur_mw_h": PRICE, "timezone": TZ,
    }).to_csv(outdir / "SYNTHETIC_dst_unified_capacity.csv", index=False)
    pd.DataFrame({
        "date": [ts.strftime("%Y-%m-%d %H:%M") for ts in starts],
        "product": "FCR", "capacity_price_eur_mw": PRICE,
    }).to_csv(outdir / "SYNTHETIC_dst_DE_FCR.csv", index=False)
    per_day = da.groupby(da.index.tz_convert(TZ).date).size()
    print({str(k): int(v) for k, v in per_day.items()}, "intervals per local day")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
