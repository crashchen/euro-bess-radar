# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`euro-bess-radar` (a.k.a. eu-bess-pulse) is a local-first Streamlit dashboard for evaluating day-ahead arbitrage and merchant revenue potential of BESS across European bidding zones. Python 3.11+, pandas/numpy, SQLite cache, Plotly, openpyxl exports.

## Module Interaction (big picture)

`src/config.py` is the single source of truth for zones, timezones, cache paths, and revenue-math constants (`HOURS_PER_YEAR`, `ANCILLARY_CAPACITY_AVAILABILITY=0.95`, `ANCILLARY_ENERGY_ACTIVATION_SHARE=0.10`, `GBP_EUR_YEARLY`). Nothing else should hard-code zones or FX rates.

Data flow inside one dashboard run (`app.py`):
1. `data_ingestion.fetch_da_prices()` dispatches to the right backend (entsoe-py for ENTSO-E zones, Elexon REST for GB), runs `build_zone_query_window()` to convert local calendar dates to a UTC `[start, end)` window, and caches into `da_prices_{zone}` SQLite tables. Cache validity is checked against `_expected_cache_interval()` + `reindex` so gaps are detected as missing points, not accepted as sparse data.
2. `analytics.py` consumes the UTC price frame but groups by the zone's local calendar day via the `tz` parameter. `_find_daily_ordered_trade()` (called from `calculate_ordered_spreads()`) finds the best chronology-aware charge→discharge pair with rolling windows sized by `duration_hours`. Heatmaps, P50/P90, RE correlation, and revenue all share this ordered-spread basis.
3. `ancillary_fetchers.py` exposes a lazy `getattr()` registry (`fetch_{zone}_ancillary`) used by `run_auto_fetch()`. Results flow into `ancillary.normalize_auto_fetch_dataset()` and then `build_ancillary_dataset()`, which merges with any user-uploaded manual CSVs — manual uploads override auto-fetch *only for the same product type*, preserving other auto-fetched products.
4. `merge_revenue_stack()` combines DA arbitrage with per-product ancillary revenue and returns a `source_revenues` dict (e.g. `DA Arbitrage`, `FCR-N`, `aFRR Up`) that both the Revenue tab and `export.py` read from.
5. `export.py` writes Excel reports that must respect the same `tz` parameter and stacked-revenue breakdown as the dashboard.

Tests in `tests/` are heavily mocked (no live API calls) and mirror the module layout 1:1. `conftest.py` holds shared price/ancillary fixtures.

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
- pandas timestamps are UTC internally; every analytics/export entry point takes a `tz` parameter for local-time grouping. Do not group by UTC day — daily spread and heatmap numbers will be wrong across DST boundaries.
- Add new zones in `config.py` only: `ENTSOE_ZONES`/`ELEXON_ZONES` + `ZONE_TIMEZONES`. Never hard-code zone codes elsewhere.
- SQLite table naming is `da_prices_{zone_code}` lowercase (e.g. `da_prices_de_lu`).
- GBP→EUR conversion must use `_get_gbp_eur_rate_for_year()` (nearest-year fallback against `GBP_EUR_YEARLY`), never a single scalar rate — GB history spans multiple FX regimes.
- API calls retry with exponential backoff, max 3 retries.
- Keep functions under ~50 lines; split long pipelines at natural stage boundaries.
- Logging via `logging`, Google-style docstrings, type hints on public functions.

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
- `pip install -r requirements.txt` — install deps (Python 3.11+; use `.venv` on macOS).
- `python -m pytest tests/ -v` — run all tests (130 tests, fully mocked, no network).
- `python -m pytest tests/test_analytics.py::TestOrderedSpreads -v` — run a single class; swap in `::test_name` for a single test.
- `streamlit run app.py` — launch the dashboard.
- `python -c "from src.data_ingestion import test_elexon_connection; test_elexon_connection()"` — smoke-test Elexon (no API key needed).
- `python -c "from src.data_ingestion import test_entsoe_connection; test_entsoe_connection()"` — smoke-test ENTSO-E (needs `ENTSOE_API_KEY` in `.env`).

CI runs `python -m pytest tests/ -v` on every push/PR via `.github/workflows/ci.yml`; keep the mocked-test suite green before pushing. See `CONTRIBUTING.md` for PR expectations and secret-handling rules.
