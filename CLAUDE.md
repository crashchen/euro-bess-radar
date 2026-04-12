# eu-bess-pulse

European BESS Market Screening Dashboard — local-first MVP for evaluating day-ahead arbitrage and merchant revenue potential across European bidding zones.

## Tech Stack
- Python 3.11+ / macOS (Apple Silicon)
- Data: entsoe-py (ENTSO-E), requests (Elexon + Fingrid + Regelleistung), pandas, numpy
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
│   ├── config.py         # Zones, ZONE_TIMEZONES, GBP_EUR_YEARLY, API endpoints
│   ├── data_ingestion.py # Unified fetching (ENTSO-E + Elexon + Fingrid + Regelleistung) + caching
│   ├── analytics.py      # Ordered spreads, P50/P90, heatmaps, revenue, RE correlation
│   ├── ancillary.py      # Ancillary parsing, auto-fetch normalization, revenue stacking
│   ├── ancillary_fetchers.py # Auto-fetch registry per zone (FI, DE_LU, GB, RO, SE_3, IT_SUD)
│   └── export.py         # Excel report generation with timezone + revenue stack support
├── app.py                # Streamlit dashboard (5 tabs, revenue-tab guidance + ancillary help expander)
└── tests/
    ├── conftest.py       # Shared fixtures
    ├── test_ingestion.py # Data ingestion + zone query window + cache validation tests
    ├── test_analytics.py # Ordered spread, duration-aware, sub-hourly, RE correlation tests
    ├── test_ancillary.py # Ancillary parsing, auto-fetch normalization, revenue stack tests
    └── test_export.py    # Excel export + timezone + revenue breakdown tests
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
  - GET /balancing/settlement/system-prices — system buy/sell prices (30-min)
  - GET /datasets/FUELINST — generation mix by fuel type
- GBP→EUR conversion uses year-specific rates via `GBP_EUR_YEARLY` dict in config.py
- Elexon returns 2 data providers per settlement period — zeros are filtered, non-zero prices averaged

### Fingrid Open Data (Finland)
- API key recommended/required for v2 (`x-api-key` header via `FINGRID_API_KEY`)
- Base URL: https://data.fingrid.fi/api
- Dataset paths use `/datasets/{id}/data`
- Dataset IDs: 317 (FCR-N), 318 (FCR-D Up), 283 (FCR-D Down), 52 (aFRR Up), 51 (aFRR Down)
- Returns: hourly capacity prices in EUR/MW

### Regelleistung.net (Germany)
- NO API key required — best-effort scraping
- Auto-fetch is currently disabled because the public page hit by the legacy code is HTML, not a stable JSON API
- Supported path is manual CSV upload (`DE_FCR`, `DE_aFRR`)
- Returns: capacity prices in EUR/MW when supplied via manual uploads

### ENTSO-E Imbalance Prices
- Uses same ENTSO-E API key as DA prices
- Covers: RO, SE_3, IT_SUD (and other ENTSO-E zones)
- Returns: imbalance/balancing prices in EUR/MWh

## Geographic Scope

Three tiers of zones:

1. **EU member states** (ENTSO-E): DE_LU, FR, NL, BE, AT, PL, IT_NORD, IT_CSUD, IT_SUD, IT_SICI, IT_SARD, ES, PT, RO, DK_1, DK_2, SE_1, SE_2, SE_3, SE_4, FI, CZ, HU, BG, GR, HR, SK, SI, EE, LT, LV, IE_SEM
2. **Non-EU ENTSO-E members** (same entsoe-py API): NO_1, NO_2, NO_3, NO_4, NO_5, CH
3. **Great Britain** (Elexon API): GB

## Ancillary Services

### Auto-Fetch (ancillary_fetchers.py)
Zone-specific fetchers run automatically when the zone is selected:
- **FI** — FCR-N/D + aFRR prices via Fingrid
- **DE_LU** — FCR auction results via Regelleistung.net
- **DE_LU** auto-fetch is best-effort only and currently disabled; use manual `DE_FCR` / `DE_aFRR` uploads
- **GB** — System buy/sell prices via Elexon
- **RO, SE_3, IT_SUD** — Imbalance prices via ENTSO-E

Registry uses lazy `getattr()` lookup for testability. `run_auto_fetch()` handles per-fetcher errors gracefully.

### Manual Upload (CSV Templates)
Supported templates:
- **DE_FCR** — Germany FCR from regelleistung.net
- **DE_aFRR** — Germany aFRR from regelleistung.net
- **RO_BALANCING** — Romania balancing from Transelectrica
- **FI_FCR** — Finland FCR from Fingrid
- **GB_BALANCING** — GB system prices from Elexon/BMRS

Upload via sidebar. Template CSVs downloadable from the UI.

### Data Merging
`build_ancillary_dataset()` merges manual + auto-fetch data. Manual uploads override auto-fetch for the same product type only — other auto-fetched products are preserved. `normalize_auto_fetch_dataset()` converts varied fetcher schemas into a standard ancillary format with per-product rows.

### UI Guidance
- Revenue Estimation tab shows an annualisation note based on the currently selected sample window length
- Revenue Estimation tab includes a `How ancillary works` expander explaining product-level stacking, manual-vs-auto precedence, and which ancillary signals are not auto-monetised

## Coding Conventions
- Type hints on all function signatures
- Docstrings: one-line summary, then Args/Returns in Google style
- Logging via `logging` module, NOT print()
- All bidding zone definitions live in config.py
- pandas timestamps: UTC internally; analytics/export accept `tz` parameter for local-time grouping
- `ZONE_TIMEZONES` in config.py maps all 37+ zones to IANA timezones (e.g. DE_LU → Europe/Berlin)
- `build_zone_query_window()` converts inclusive local calendar dates to UTC `[start, end)` query window
- SQLite table naming: `da_prices_{zone_code}` (lowercase, e.g. `da_prices_de_lu`)
- Cache validation: `_expected_cache_interval()` + `reindex` to detect actual missing data points
- GBP→EUR: `GBP_EUR_YEARLY` dict with per-year rates; `_get_gbp_eur_rate_for_year()` falls back to nearest known year
- Error handling: retry with exponential backoff for API calls, max 3 retries
- Functions should be <50 lines. If longer, split.

## Key Domain Knowledge
- **Ordered spreads**: Revenue estimation uses chronology-aware charge-before-discharge windows, not simple max-min. `_find_daily_ordered_trade()` finds best non-overlapping buy/sell pair using rolling averages and backward scan.
- **Duration-aware analysis**: `duration_hours` parameter (1h/2h/4h) controls rolling window size for spread calculation, heatmaps, and zone comparison. Models actual BESS dispatch constraints.
- **Zone-local time**: All analytics group by local calendar day/hour (via `tz` parameter), not UTC. Critical for correct daily spread, heatmap, and export date ranges.
- Germany (DE_LU) switched to 15-min DA resolution in Oct 2025
- GB data is 30-min settlement periods, GBP/MWh (not EUR)
- **Sub-hourly resolution**: `calculate_negative_price_hours()` returns both `negative_hours` (physical hours) and `negative_intervals` (count of data points) to handle 15-min/30-min data correctly
- Norway and Switzerland are NOT EU members but their TSOs are ENTSO-E members — their data is on the Transparency Platform
- Some ENTSO-E zones have data gaps — always validate returned DataFrame is not empty
- DA prices across ENTSO-E are in EUR/MWh; GB is in GBP/MWh
- Generation data: not all zones have solar/wind/offshore split — handle missing columns gracefully
- Negative wind/solar-price correlation = BESS-friendly market (high RE → low prices = charging, low RE → high prices = discharging)
- **Revenue stacking**: `merge_revenue_stack()` combines DA arbitrage + ancillary revenues with `source_revenues` dict for per-product breakdown (e.g. DA Arbitrage, FCR-N, aFRR Up)
- **Annualisation caveat**: DA arbitrage revenue is extrapolated from the user-selected sample window, so short windows (for example winter-only periods) can materially overstate or understate full-year merchant potential

## Commands
- `pip install -r requirements.txt` — install deps
- `python -m pytest tests/ -v` — run all tests (99 tests)
- `streamlit run app.py` — launch dashboard
- `python -c "from src.data_ingestion import test_elexon_connection; test_elexon_connection()"` — test Elexon
- `python -c "from src.data_ingestion import test_entsoe_connection; test_entsoe_connection()"` — test ENTSO-E
