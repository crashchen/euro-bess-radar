"""Tests for the Netztransparenz reBAP/NRV conversion helper.

The script is a thin wrapper over the production converter shared with the
live fetch (``data_ingestion._convert_netztransparenz_imbalance_exports``),
so these tests pin the shared semantics: official ``N.A.`` intervals drop,
only timestamps published in both NRV-Saldo and reBAP survive (inner join),
and semantic failures raise ``DataSourceParseError``.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

from scripts.convert_netztransparenz_imbalance import (
    convert_netztransparenz_imbalance,
)
from src.ancillary import parse_imbalance_import_csv
from src.data_ingestion import (
    DataSourceParseError,
    _convert_netztransparenz_imbalance_exports,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _de(value: float | str) -> str:
    """Format a cell the German way; pass strings (e.g. ``N.A.``) through."""
    return value if isinstance(value, str) else str(value).replace(".", ",")


def _slots(n: int, *, first_minute: int = 0) -> list[tuple[str, str]]:
    """Return ``n`` consecutive 15-minute (von, bis) local slots."""
    minutes = [first_minute + 15 * i for i in range(n + 1)]
    labels = [f"{m // 60:02d}:{m % 60:02d}" for m in minutes]
    return list(pairwise(labels))


def _nrv_csv(
    values: list[float | str], *, start: str = "01.05.2026", first_minute: int = 0,
) -> str:
    rows = ["Datum;Zeitzone;von;bis;Einheit;Deutschland"]
    for (von, bis), val in zip(
        _slots(len(values), first_minute=first_minute), values, strict=True,
    ):
        rows.append(f"{start};CEST;{von};{bis};MW;{_de(val)}")
    return "\n".join(rows) + "\n"


def _rebap_csv(
    prices: list[float | str],
    *,
    over_prices: list[float | str] | None = None,
    start: str = "01.05.2026",
    first_minute: int = 0,
) -> str:
    over_prices = over_prices or prices
    rows = ["Datum;Zeitzone;von;bis;Einheit;reBAP unterdeckt;reBAP ueberdeckt"]
    for (von, bis), under, over in zip(
        _slots(len(prices), first_minute=first_minute), prices, over_prices,
        strict=True,
    ):
        rows.append(f"{start};CEST;{von};{bis};EUR/MWh;{_de(under)};{_de(over)}")
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


def test_converter_drops_official_na_blocks_and_inner_joins(tmp_path) -> None:
    """Official N.A. intervals drop; only rows published in BOTH files survive."""
    nrv = _write(
        tmp_path / "nrv.csv",
        _nrv_csv([391.596, "N.A.", 230.72, "1.234,5"]),
    )
    rebap = _write(
        tmp_path / "rebap.csv",
        _rebap_csv([122.73, 101.08, "N.A.", -18.2]),
    )

    out = convert_netztransparenz_imbalance(nrv_path=nrv, rebap_path=rebap)

    # 00:15 is N.A. in NRV, 00:30 is N.A. in reBAP -> only 00:00 + 00:45 remain.
    assert out["timestamp"].tolist() == [
        "2026-04-30T22:00:00Z",
        "2026-04-30T22:45:00Z",
    ]
    # German thousands dot + comma decimal parsed on the object-dtype column.
    assert out["system_imbalance_volume_mw"].tolist() == [391.596, 1234.5]
    assert out["imbalance_price_eur_mwh"].tolist() == [122.73, -18.2]
    assert len(parse_imbalance_import_csv(out.to_csv(index=False))) == 2


def test_converter_inner_joins_partially_overlapping_windows(tmp_path) -> None:
    """Different-but-overlapping regular windows keep only the shared axis."""
    nrv = _write(tmp_path / "nrv.csv", _nrv_csv([1.0, 2.0, 3.0, 4.0]))
    rebap = _write(
        tmp_path / "rebap.csv",
        _rebap_csv([10.0, 11.0, 12.0, 13.0], first_minute=30),
    )

    out = convert_netztransparenz_imbalance(nrv_path=nrv, rebap_path=rebap)

    # NRV covers 00:00-01:00, reBAP covers 00:30-01:30 -> overlap 00:30 + 00:45.
    assert out["timestamp"].tolist() == [
        "2026-04-30T22:30:00Z",
        "2026-04-30T22:45:00Z",
    ]
    assert out["system_imbalance_volume_mw"].tolist() == [3.0, 4.0]
    assert out["imbalance_price_eur_mwh"].tolist() == [10.0, 11.0]


def test_converter_matches_production_converter_exactly(tmp_path) -> None:
    """Script output == the live-fetch converter on the same raw CSV text."""
    nrv_text = _nrv_csv([391.596, "N.A.", 230.72])
    rebap_text = _rebap_csv([122.73, 101.08, -18.2])
    nrv = _write(tmp_path / "nrv.csv", nrv_text)
    rebap = _write(tmp_path / "rebap.csv", rebap_text)

    out = convert_netztransparenz_imbalance(nrv_path=nrv, rebap_path=rebap)
    production = _convert_netztransparenz_imbalance_exports(
        nrv_csv=nrv_text, rebap_csv=rebap_text,
    )

    assert len(out) == len(production)
    assert out["timestamp"].tolist() == (
        production.index.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    )
    assert out["zone"].tolist() == production["zone"].tolist()
    assert out["imbalance_price_eur_mwh"].tolist() == (
        production["imbalance_price_eur_mwh"].tolist()
    )
    assert out["system_imbalance_volume_mw"].tolist() == (
        production["system_imbalance_volume_mw"].tolist()
    )


def test_converter_rejects_rebap_under_over_mismatch(tmp_path) -> None:
    nrv = _write(tmp_path / "nrv.csv", _nrv_csv([1.0, 2.0, 3.0]))
    rebap = _write(
        tmp_path / "rebap.csv",
        _rebap_csv([10.0, 11.0, 12.0], over_prices=[10.0, 99.0, 12.0]),
    )

    with pytest.raises(
        DataSourceParseError, match="unterdeckt and ueberdeckt columns differ",
    ):
        convert_netztransparenz_imbalance(nrv_path=nrv, rebap_path=rebap)


def test_converter_rejects_disjoint_timestamps(tmp_path) -> None:
    nrv = _write(tmp_path / "nrv.csv", _nrv_csv([1.0, 2.0, 3.0]))
    rebap = _write(
        tmp_path / "rebap.csv", _rebap_csv([10.0, 11.0, 12.0], start="02.05.2026"),
    )

    with pytest.raises(
        DataSourceParseError, match="no overlapping published timestamps",
    ):
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

    with pytest.raises(DataSourceParseError, match="regular 15-minute series"):
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


def test_converter_handles_autumn_dst_repeat_and_rebap_tolerance(tmp_path) -> None:
    nrv = _write(
        tmp_path / "nrv.csv",
        (
            "Datum;Zeitzone;von;bis;Einheit;Deutschland\n"
            "25.10.2026;cest;02:00;02:15;MW;10,0\n"
            "25.10.2026;CEST;02:15;02:30;MW;15,0\n"
            "25.10.2026;CEST;02:30;02:45;MW;20,0\n"
            "25.10.2026;CEST;02:45;03:00;MW;25,0\n"
            "25.10.2026;CET;02:00;02:15;MW;30,0\n"
            "25.10.2026;CET;02:15;02:30;MW;35,0\n"
        ),
    )
    rebap = _write(
        tmp_path / "rebap.csv",
        (
            "Datum;Zeitzone;von;bis;Einheit;reBAP unterdeckt;reBAP ueberdeckt\n"
            "25.10.2026;cest;02:00;02:15;EUR/MWh;100,0;100,0\n"
            "25.10.2026;CEST;02:15;02:30;EUR/MWh;105,0;105,0\n"
            "25.10.2026;CEST;02:30;02:45;EUR/MWh;110,0;110,0\n"
            "25.10.2026;CEST;02:45;03:00;EUR/MWh;115,0;115,0\n"
            "25.10.2026;CET;02:00;02:15;EUR/MWh;120,0;119,99\n"
            "25.10.2026;CET;02:15;02:30;EUR/MWh;125,0;125,0\n"
        ),
    )

    out = convert_netztransparenz_imbalance(nrv_path=nrv, rebap_path=rebap)

    assert out["timestamp"].tolist() == [
        "2026-10-25T00:00:00Z",
        "2026-10-25T00:15:00Z",
        "2026-10-25T00:30:00Z",
        "2026-10-25T00:45:00Z",
        "2026-10-25T01:00:00Z",
        "2026-10-25T01:15:00Z",
    ]
    assert out["system_imbalance_volume_mw"].tolist() == [
        10.0, 15.0, 20.0, 25.0, 30.0, 35.0,
    ]
    assert out["imbalance_price_eur_mwh"].tolist() == [
        100.0, 105.0, 110.0, 115.0, 120.0, 125.0,
    ]
