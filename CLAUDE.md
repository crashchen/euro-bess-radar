# eu-bess-pulse

European BESS Market Screening Dashboard — local-first MVP for evaluating day-ahead arbitrage and merchant revenue potential across European bidding zones.

## Tech Stack
- Python 3.11+ / macOS (Apple Silicon)
- Data: entsoe-py (ENTSO-E), requests (Elexon), pandas, numpy
- Storage: SQLite (local cache) + CSV exports
- Dashboard: Streamlit + Plotly
- Export: openpyxl for .xlsx reports
- Environment: python-dotenv for API key management

## Architecture
```
euro-bess-radar/
├── CLAUDE.md
├── .env                  # ENTSOE_API_KEY=xxx (git-ignored)
├── .gitignore
├── requirements.txt
├── data/
│   ├── cache/            # SQLite DB + CSV cache files
│   └── manual/           # Manual CSV uploads (FCR/aFRR data)
├── src/
│   ├── __init__.py
│   ├── config.py         # Constants: bidding zones, API endpoints, DB paths
│   ├── data_ingestion.py # Unified fetching (ENTSO-E + Elexon) + generation data + caching
│   ├── analytics.py      # Spread calc, P50/P90, heatmaps, revenue, renewable correlation
│   ├── ancillary.py      # Ancillary services upload, parsing, revenue stacking
│   ├── ancillary_fetchers.py # Auto-fetch registry for ancillary data by zone
│   └── export.py         # Excel report generation (openpyxl)
├── app.py                # Streamlit dashboard (5 tabs)
└── tests/
    ├── conftest.py       # Shared fixtures
    ├── test_ingestion.py # Data ingestion tests (mocked APIs)
    ├── test_analytics.py # Analytics + renewable correlation tests
    ├── test_ancillary.py # Ancillary services tests
    └── test_export.py    # Excel export tests
```

## Data Sources

### ENTSO-E Transparency Platform
- Requires API key (loaded from .env)
- Covers: EU-27 + Norway (NO_1–NO_5) + Switzerland (CH)
- Python library: entsoe-py (v0.7.11, uses `web-api.tp.entsoe.eu`)
- Returns: Day-ahead prices in EUR/MWh + generation by fuel type
- Resolution: 60min (most zones), 15min (DE_LU since Oct 2025)
- Historical data: available back to at least 2024 via the new API endpoint

### Elexon Insights API (Great Britain)
- NO API key required — fully public REST API
- Base URL: https://data.elexon.co.uk/bmrs/api/v1
- Endpoints:
  - GET /balancing/pricing/market-index — MID prices (GBP/MWh, 30-min)
  - GET /datasets/FUELINST — generation mix by fuel type
- Must convert GBP to EUR for cross-country comparison (configurable static rate)
- Elexon returns 2 data providers per settlement period — zeros are filtered, non-zero prices averaged

## Geographic Scope

Three tiers of zones:

1. **EU member states** (ENTSO-E): DE_LU, FR, NL, BE, AT, PL, IT_NORD, IT_CSUD, IT_SUD, IT_SICI, IT_SARD, ES, PT, RO, DK_1, DK_2, SE_1, SE_2, SE_3, SE_4, FI, CZ, HU, BG, GR, HR, SK, SI, EE, LT, LV, IE_SEM
2. **Non-EU ENTSO-E members** (same entsoe-py API): NO_1, NO_2, NO_3, NO_4, NO_5, CH
3. **Great Britain** (Elexon API): GB

## Ancillary Services (Manual Upload)

Supported templates for CSV upload:
- **DE_FCR** — Germany FCR from regelleistung.net
- **DE_aFRR** — Germany aFRR from regelleistung.net
- **RO_BALANCING** — Romania balancing from Transelectrica
- **FI_FCR** — Finland FCR from Fingrid
- **GB_BALANCING** — GB system prices from Elexon/BMRS

Upload via sidebar in dashboard. Template CSVs downloadable from the UI.

## Coding Conventions
- Type hints on all function signatures
- Docstrings: one-line summary, then Args/Returns in Google style
- Logging via `logging` module, NOT print()
- All bidding zone definitions live in config.py
- pandas timestamps must be timezone-aware (UTC internally, convert for display)
- SQLite table naming: `da_prices_{zone_code}` (lowercase, e.g. `da_prices_de_lu`)
- Error handling: retry with exponential backoff for API calls, max 3 retries
- Functions should be <50 lines. If longer, split.

## Key Domain Knowledge
- Germany (DE_LU) switched to 15-min DA resolution in Oct 2025
- GB data is 30-min settlement periods, GBP/MWh (not EUR)
- Norway and Switzerland are NOT EU members but their TSOs are ENTSO-E members — their data is on the Transparency Platform
- Some ENTSO-E zones have data gaps — always validate returned DataFrame is not empty
- DA prices across ENTSO-E are in EUR/MWh; GB is in GBP/MWh
- Generation data: not all zones have solar/wind/offshore split — handle missing columns gracefully
- Negative wind/solar-price correlation = BESS-friendly market (high RE → low prices = charging, low RE → high prices = discharging)

## Commands
- `pip install -r requirements.txt` — install deps
- `python -m pytest tests/ -v` — run all tests (69 tests)
- `streamlit run app.py` — launch dashboard
- `python -c "from src.data_ingestion import test_elexon_connection; test_elexon_connection()"` — test Elexon
- `python -c "from src.data_ingestion import test_entsoe_connection; test_entsoe_connection()"` — test ENTSO-E
