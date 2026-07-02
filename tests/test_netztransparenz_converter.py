"""Tests for the Netztransparenz reBAP/NRV conversion helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.convert_netztransparenz_imbalance import (
    convert_netztransparenz_imbalance,
)
from src.ancillary import parse_imbalance_import_csv


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _nrv_csv(values: list[float], *, start: str = "01.05.2026") -> str:
    times = [("00:00", "00:15"), ("00:15", "00:30"), ("00:30", "00:45")]
    rows = ["Datum;Zeitzone;von;bis;Einheit;Deutschland"]
    for (von, bis), val in zip(times, values, strict=True):
        rows.append(f"{start};CEST;{von};{bis};MW;{str(val).replace('.', ',')}")
    return "\n".join(rows) + "\n"


def _rebap_csv(
    prices: list[float], *, over_prices: list[float] | None = None,
    start: str = "01.05.2026",
) -> str:
    times = [("00:00", "00:15"), ("00:15", "00:30"), ("00:30", "00:45")]
    over_prices = over_prices or prices
    rows = ["Datum;Zeitzone;von;bis;Einheit;reBAP unterdeckt;reBAP ueberdeckt"]
    for (von, bis), under, over in zip(times, prices, over_prices, strict=True):
        under_s = str(under).replace(".", ",")
        over_s = str(over).replace(".", ",")
        rows.append(f"{start};CEST;{von};{bis};EUR/MWh;{under_s};{over_s}")
    return "\n".join(rows) + "\n"


def test_convert_netztransparenz_imbalance_outputs_unified_csv(tmp_path) -> None:
    nrv = _write(tmp_path / "nrv.csv", _nrv_csv([391.596, -37.672, 230.72]))
    rebap = _write(tmp_path / "rebap.csv", _rebap_csv([122.73, 101.08, -18.2]))

    out = convert_netztransparenz_imbalance(nrv_path=nrv, rebap_path=rebap)

    assert list(out.columns) == [
        "timestamp", "zone", "imbalance_price_eur_mwh",
        "system_imbalance_volume_mw",
    ]
    assert out["timestamp"].tolist() == [
        "2026-04-30T22:00:00Z",
        "2026-04-30T22:15:00Z",
        "2026-04-30T22:30:00Z",
    ]
    assert out["zone"].tolist() == ["DE_LU", "DE_LU", "DE_LU"]
    assert out["imbalance_price_eur_mwh"].tolist() == [122.73, 101.08, -18.2]
    assert out["system_imbalance_volume_mw"].tolist() == [391.596, -37.672, 230.72]

    # The generated frame is directly accepted by the production import parser.
    parsed = parse_imbalance_import_csv(out.to_csv(index=False))
    assert len(parsed) == 3
    assert str(parsed.index.tz) == "UTC"


def test_converter_rejects_rebap_under_over_mismatch(tmp_path) -> None:
    nrv = _write(tmp_path / "nrv.csv", _nrv_csv([1.0, 2.0, 3.0]))
    rebap = _write(
        tmp_path / "rebap.csv",
        _rebap_csv([10.0, 11.0, 12.0], over_prices=[10.0, 99.0, 12.0]),
    )

    with pytest.raises(ValueError, match="unterdeckt and ueberdeckt columns differ"):
        convert_netztransparenz_imbalance(nrv_path=nrv, rebap_path=rebap)


def test_converter_rejects_mismatched_timestamps(tmp_path) -> None:
    nrv = _write(tmp_path / "nrv.csv", _nrv_csv([1.0, 2.0, 3.0]))
    rebap = _write(
        tmp_path / "rebap.csv", _rebap_csv([10.0, 11.0, 12.0], start="02.05.2026"),
    )

    with pytest.raises(ValueError, match="do not align exactly"):
        convert_netztransparenz_imbalance(nrv_path=nrv, rebap_path=rebap)


def test_converter_rejects_non_regular_15min_series(tmp_path) -> None:
    nrv = _write(
        tmp_path / "nrv.csv",
        (
            "Datum;Zeitzone;von;bis;Einheit;Deutschland\n"
            "01.05.2026;CEST;00:00;00:15;MW;1,0\n"
            "01.05.2026;CEST;00:15;00:30;MW;2,0\n"
            "01.05.2026;CEST;01:00;01:15;MW;3,0\n"
        ),
    )
    rebap = _write(tmp_path / "rebap.csv", _rebap_csv([10.0, 11.0, 12.0]))

    with pytest.raises(ValueError, match="regular 15-minute series"):
        convert_netztransparenz_imbalance(nrv_path=nrv, rebap_path=rebap)


def test_cli_writes_upload_ready_file(tmp_path, monkeypatch, capsys) -> None:
    from scripts import convert_netztransparenz_imbalance as conv

    nrv = _write(tmp_path / "nrv.csv", _nrv_csv([1.0, -2.0, 3.0]))
    rebap = _write(tmp_path / "rebap.csv", _rebap_csv([10.0, 11.0, 12.0]))
    out = tmp_path / "samples" / "imbalance.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["convert", "--nrv", str(nrv), "--rebap", str(rebap), "--out", str(out)],
    )

    conv.main()

    assert out.exists()
    written = pd.read_csv(out)
    assert len(written) == 3
    assert "Wrote 3 unified imbalance rows" in capsys.readouterr().out
