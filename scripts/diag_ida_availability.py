"""Diagnostic: is ENTSO-E IDA (intraday auction) price data actually available?

Run this to distinguish three failure modes for the Simulation Cockpit's
"Forecast-driven IDA policy" panel when it shows ``No IDA1 data``:

  1. Auth   — bad/missing ENTSOE_API_KEY (DA control would also fail).
  2. Parse  — entsoe-py only parses one resolution key (``parse_prices(text)['15min']``),
              so a 60-min IDA print would be dropped *after* a successful fetch.
  3. Upstream — ENTSO-E returns an Acknowledgement document with
                "No matching data found" *before* any parsing happens.

Findings as of 2026-06 (entsoe-py 0.7.11, web-api.tp.entsoe.eu):
the IDA query (documentType=A44, contract_MarketAgreement.type=A07,
classificationSequence position=seq) returns "No matching data found"
for ALL six SIDC zones across multiple windows and all three sequences,
while the plain DA control on the same documentType returns a full
publication document. So this is mode (3) — an upstream data gap, NOT
the (2) resolution-forcing bug (the parse path is never reached).

Usage:
    python -m scripts.diag_ida_availability
"""

from __future__ import annotations

import pandas as pd
import requests
from bs4 import BeautifulSoup
from entsoe.mappings import lookup_area

import src.config  # noqa: F401  (triggers .env load so get_api_key works)
from src.data_ingestion import build_zone_query_window, get_api_key

ENTSOE_API_URL = "https://web-api.tp.entsoe.eu/api"
SIDC_ZONES = ["DE_LU", "NL", "FR", "BE", "AT", "IT_NORD"]
WINDOWS = [
    ("2024-09-02", "2024-09-05"),
    ("2025-06-02", "2025-06-05"),
    ("2025-12-01", "2025-12-04"),
    ("2026-03-02", "2026-03-05"),
]


def _classify(zone: str, win: tuple[str, str], sequence: int, key: str) -> str:
    """Return DATA(n)/no-data/HTTP-err for one raw IDA request."""
    area = lookup_area(zone)
    start, end = build_zone_query_window(zone, *win)
    # entsoe-py pads +-1 day; mirror it so results match the library.
    period_start = (start - pd.Timedelta(days=1)).tz_convert("UTC")
    period_end = (end + pd.Timedelta(days=1)).tz_convert("UTC")
    params = {
        "securityToken": key,
        "documentType": "A44",
        "in_Domain": area.code,
        "out_Domain": area.code,
        "contract_MarketAgreement.type": "A07",
        "classificationSequence_AttributeInstanceComponent.position": sequence,
        "periodStart": period_start.strftime("%Y%m%d%H00"),
        "periodEnd": period_end.strftime("%Y%m%d%H00"),
    }
    resp = requests.get(ENTSOE_API_URL, params=params, timeout=60)
    if resp.status_code != 200:
        return f"HTTP-{resp.status_code}"
    soup = BeautifulSoup(resp.text, "html.parser")
    root = soup.find(["acknowledgement_marketdocument", "publication_marketdocument"])
    if root is not None and root.name == "publication_marketdocument":
        return f"DATA({len(soup.find_all('point'))} pts)"
    return "no-data"


def main() -> None:
    key = get_api_key()
    print(f"Endpoint: {ENTSOE_API_URL}")
    print("IDA1 (sequence=1) availability matrix:\n")
    for zone in SIDC_ZONES:
        cells = [f"{w[0]}:{_classify(zone, w, 1, key)}" for w in WINDOWS]
        print(f"  {zone:8s} | " + " | ".join(cells))
    print("\nDE_LU all sequences (2025-06-02):")
    for seq in (1, 2, 3):
        print(f"  seq{seq}: {_classify('DE_LU', ('2025-06-02', '2025-06-05'), seq, key)}")


if __name__ == "__main__":
    main()
