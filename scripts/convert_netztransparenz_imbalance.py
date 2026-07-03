"""Convert Netztransparenz reBAP + NRV-Saldo CSVs to the unified imbalance CSV.

Thin CLI wrapper over the production converter shared with the live fetch
(``data_ingestion.fetch_netztransparenz_imbalance``), so the manual path and
the live path have ONE set of parsing semantics:

- German comma decimals (``100,55`` / ``1.234,5``) are normalised explicitly.
- Official unavailable tokens (``N.A.`` etc.) drop those intervals instead of
  failing the whole conversion.
- Only timestamps published in BOTH NRV-Saldo and reBAP survive (inner join).
- The 15-minute regularity check runs on each file's raw axis BEFORE the drop.
- Autumn-DST repeated hours are disambiguated via the per-row ``Zeitzone``
  CET/CEST labels.
- ``reBAP unterdeckt`` / ``ueberdeckt`` must match within tolerance on
  published rows (symmetric German reBAP export).

The output is the repository's unified import template:

    timestamp,zone,imbalance_price_eur_mwh,system_imbalance_volume_mw

Sign convention validated for the replay overlay:

- positive NRV-Saldo = system short / undercovered (discharge helps)
- negative NRV-Saldo = system long / overcovered (charge helps)
- reBAP is the published signed cash-flow price; no extra sign flip

Usage:

    python scripts/convert_netztransparenz_imbalance.py \
      --nrv "Netztransparenz/NRV-Saldo qualitaetsgesichert [...].csv" \
      --rebap "Netztransparenz/reBAP unterdeckt [...].csv" \
      --out samples/netztransparenz_imbalance_de_lu_2026-05.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow ``python scripts/convert_netztransparenz_imbalance.py`` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ancillary import parse_imbalance_import_csv
from src.data_ingestion import (
    NETZTRANSPARENZ_IMBALANCE_ZONE,
    convert_netztransparenz_imbalance_exports,
)

DEFAULT_ZONE = NETZTRANSPARENZ_IMBALANCE_ZONE


def _read_csv_text(path: Path) -> str:
    """Read one official export as text (utf-8-sig strips the BOM)."""
    return path.read_text(encoding="utf-8-sig")


def convert_netztransparenz_imbalance(
    *,
    nrv_path: Path,
    rebap_path: Path,
    zone: str = DEFAULT_ZONE,
) -> pd.DataFrame:
    """Convert official Netztransparenz exports to the unified imbalance CSV frame.

    Delegates to ``data_ingestion.convert_netztransparenz_imbalance_exports``
    (the exact converter the live fetcher uses), then reshapes the dedicated
    imbalance frame into the upload-ready CSV column layout.
    """
    frame = convert_netztransparenz_imbalance_exports(
        nrv_csv=_read_csv_text(nrv_path),
        rebap_csv=_read_csv_text(rebap_path),
        zone=zone,
    )
    out = pd.DataFrame({
        "timestamp": frame.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "zone": frame["zone"],
        "imbalance_price_eur_mwh": frame["imbalance_price_eur_mwh"].astype(float),
        "system_imbalance_volume_mw": frame[
            "system_imbalance_volume_mw"
        ].astype(float),
    }).reset_index(drop=True)
    # Validate the output through the production parser before returning.
    parse_imbalance_import_csv(out.to_csv(index=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrv", required=True, type=Path, help="NRV-Saldo CSV path.")
    parser.add_argument("--rebap", required=True, type=Path, help="reBAP CSV path.")
    parser.add_argument("--out", required=True, type=Path, help="Output unified CSV.")
    parser.add_argument("--zone", default=DEFAULT_ZONE, help="Settlement zone code.")
    args = parser.parse_args()

    out = convert_netztransparenz_imbalance(
        nrv_path=args.nrv, rebap_path=args.rebap, zone=args.zone,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, lineterminator="\n")
    print(
        f"Wrote {len(out)} unified imbalance rows to {args.out} "
        f"({out['timestamp'].iloc[0]} .. {out['timestamp'].iloc[-1]} UTC)."
    )
    print(
        "Official N.A. intervals are dropped; only timestamps published in both "
        "NRV-Saldo and reBAP are kept (same semantics as the live fetch)."
    )
    print(
        "Sign convention: positive system_imbalance_volume_mw = system short; "
        "negative = system long. reBAP price is kept as the signed cash-flow price."
    )


if __name__ == "__main__":
    main()
