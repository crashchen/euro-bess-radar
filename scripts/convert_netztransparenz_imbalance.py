"""Convert Netztransparenz reBAP + NRV-Saldo CSVs to the unified imbalance CSV.

This is the manual-first bridge before a live Netztransparenz fetcher: take the
official German CSV exports, validate their 15-minute time axis and sign
semantics, then emit the repository's unified import template:

    timestamp,zone,imbalance_price_eur_mwh,system_imbalance_volume_mw

The converter assumes the German Netztransparenz convention validated for the
replay overlay:

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

DEFAULT_ZONE = "DE_LU"
_TIMEZONE_BY_LABEL = {
    "CET": "Europe/Berlin",
    "CEST": "Europe/Berlin",
}
_REBAP_EQUAL_TOLERANCE_EUR_MWH = 1e-2


def _read_netztransparenz_csv(path: Path) -> pd.DataFrame:
    """Read a semicolon/comma-decimal Netztransparenz CSV."""
    try:
        return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read Netztransparenz CSV {path}: {exc}") from exc


def _timestamp_utc(df: pd.DataFrame, *, source_name: str) -> pd.Series:
    """Build a UTC timestamp from Netztransparenz Datum/von/Zeitzone columns."""
    required = {"Datum", "Zeitzone", "von"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source_name} is missing required column(s): {sorted(missing)}")
    zone_labels = df["Zeitzone"].astype(str).str.strip().str.upper()
    zones = set(zone_labels)
    unknown = zones - set(_TIMEZONE_BY_LABEL)
    if unknown:
        raise ValueError(
            f"{source_name} has unsupported Zeitzone value(s): {sorted(unknown)}",
        )
    # The official exports split date and interval start into local German time.
    local_naive = pd.to_datetime(
        df["Datum"].astype(str).str.strip() + " " + df["von"].astype(str).str.strip(),
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    )
    if local_naive.isna().any():
        bad = int(local_naive.isna().sum())
        raise ValueError(f"{source_name} has {bad} unparseable timestamp row(s)")
    # Autumn DST repeats 02:00-03:00; the official Zeitzone column tells pandas
    # which copy is daylight-saving time (CEST) and which is standard time (CET).
    is_dst = (zone_labels == "CEST").to_numpy()
    return local_naive.dt.tz_localize(
        "Europe/Berlin", ambiguous=is_dst, nonexistent="raise",
    ).dt.tz_convert("UTC")


def _validate_regular_15min(ts: pd.Series, *, source_name: str) -> None:
    ordered = pd.Series(pd.DatetimeIndex(ts).sort_values())
    if ordered.duplicated().any():
        raise ValueError(f"{source_name} contains duplicate timestamps")
    if len(ordered) < 2:
        raise ValueError(f"{source_name} needs at least two timestamps")
    deltas = ordered.diff().dropna().dt.total_seconds()
    bad = deltas[deltas != 900.0]
    if not bad.empty:
        raise ValueError(
            f"{source_name} is not a regular 15-minute series; "
            f"unexpected gap seconds: {sorted(set(bad.astype(int)))[:5]}",
        )


def _rebap_price(rebap: pd.DataFrame) -> pd.Series:
    """Return the validated signed reBAP cash-flow price."""
    cols = set(rebap.columns)
    has_under = "reBAP unterdeckt" in cols
    has_over = "reBAP ueberdeckt" in cols
    if has_under and has_over:
        under = pd.to_numeric(rebap["reBAP unterdeckt"], errors="coerce")
        over = pd.to_numeric(rebap["reBAP ueberdeckt"], errors="coerce")
        if under.isna().any() or over.isna().any():
            raise ValueError("reBAP file contains non-numeric price value(s)")
        diff = float((under - over).abs().max())
        if diff > _REBAP_EQUAL_TOLERANCE_EUR_MWH + 1e-12:
            raise ValueError(
                "reBAP unterdeckt and ueberdeckt columns differ; this converter "
                f"expects the symmetric German reBAP export (max diff {diff:g})",
            )
        return under
    for col in ("reBAP", "rebap", "reBAP Preis", "reBAP price"):
        if col in cols:
            price = pd.to_numeric(rebap[col], errors="coerce")
            if price.isna().any():
                raise ValueError(f"reBAP column {col!r} has non-numeric value(s)")
            return price
    raise ValueError(
        "reBAP file must contain either both 'reBAP unterdeckt' / "
        "'reBAP ueberdeckt' columns or a single reBAP price column",
    )


def convert_netztransparenz_imbalance(
    *,
    nrv_path: Path,
    rebap_path: Path,
    zone: str = DEFAULT_ZONE,
) -> pd.DataFrame:
    """Convert official Netztransparenz exports to the unified imbalance frame."""
    nrv = _read_netztransparenz_csv(nrv_path)
    rebap = _read_netztransparenz_csv(rebap_path)

    if "Deutschland" not in nrv.columns:
        raise ValueError("NRV-Saldo file is missing the 'Deutschland' MW column")
    nrv_ts = _timestamp_utc(nrv, source_name="NRV-Saldo")
    rebap_ts = _timestamp_utc(rebap, source_name="reBAP")
    _validate_regular_15min(nrv_ts, source_name="NRV-Saldo")
    _validate_regular_15min(rebap_ts, source_name="reBAP")

    nrv_work = pd.DataFrame({
        "timestamp": nrv_ts,
        "system_imbalance_volume_mw": pd.to_numeric(
            nrv["Deutschland"], errors="coerce",
        ),
    })
    rebap_work = pd.DataFrame({
        "timestamp": rebap_ts,
        "imbalance_price_eur_mwh": _rebap_price(rebap),
    })
    if nrv_work["system_imbalance_volume_mw"].isna().any():
        raise ValueError("NRV-Saldo file contains non-numeric Deutschland value(s)")

    merged = nrv_work.merge(rebap_work, on="timestamp", how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(
            "NRV-Saldo and reBAP timestamps do not align exactly: "
            f"{counts}",
        )
    merged = merged.sort_values("timestamp")
    out = pd.DataFrame({
        "timestamp": merged["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "zone": zone,
        "imbalance_price_eur_mwh": merged["imbalance_price_eur_mwh"].astype(float),
        "system_imbalance_volume_mw": merged[
            "system_imbalance_volume_mw"
        ].astype(float),
    })
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
        "Sign convention: positive system_imbalance_volume_mw = system short; "
        "negative = system long. reBAP price is kept as the signed cash-flow price."
    )


if __name__ == "__main__":
    main()
