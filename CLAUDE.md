# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`euro-bess-radar` (a.k.a. eu-bess-pulse) is a local-first Streamlit dashboard for evaluating day-ahead arbitrage and merchant revenue potential of BESS across European bidding zones. Python 3.11+, pandas/numpy, SQLite cache, Plotly, openpyxl exports.

## Module Interaction (big picture)

`src/config.py` is the single source of truth for zones, timezones, cache paths, and revenue-math constants (`HOURS_PER_YEAR`, `ANCILLARY_CAPACITY_AVAILABILITY=0.95`, `ANCILLARY_ENERGY_ACTIVATION_SHARE=0.10`, `GBP_EUR_YEARLY`, `MAX_SHORT_GAP_HOURS=2.0`). Nothing else should hard-code zones, FX rates, or data-quality thresholds.

Data flow inside one dashboard run (`app.py`):
1. `data_ingestion.fetch_prices()` dispatches to the right backend (entsoe-py for ENTSO-E zones, Elexon REST for GB), runs `build_zone_query_window()` to convert local calendar dates to a UTC `[start, end)` window, and caches into `da_prices_{zone}` SQLite tables. Cache validity is checked per requested slice using row-level `fetched_at`, `_expected_cache_interval()` + `reindex`, so stale rows and gaps are detected as missing points instead of accepted as sparse data.
2. `analytics.py` consumes the UTC price frame but groups by the zone's local calendar day via the `tz` parameter. `_find_daily_ordered_trade()` (called from `calculate_daily_spreads()`) finds the best chronology-aware charge→discharge pair with rolling windows sized by `duration_hours`. Heatmaps, P50/P90, RE correlation, and revenue all share this ordered-spread basis.
3. `ancillary_fetchers.py` exposes a lazy `getattr()` registry (`fetch_{zone}_ancillary`) used by `run_auto_fetch()`. Results flow into `ancillary.normalize_auto_fetch_dataset()` and then `build_ancillary_dataset()`, which merges with any user-uploaded manual CSVs — manual uploads override auto-fetch *only for the same product type*, preserving other auto-fetched products.
4. `merge_revenue_stack()` combines DA arbitrage with per-product ancillary revenue and returns a `source_revenues` dict (e.g. `DA Arbitrage`, `FCR-N`, `aFRR Up`) that both the Revenue tab and `export.py` read from. Capacity-reserve ancillary is not treated as fully additive with DA in the headline total; `gross_additive_total_eur` remains available as a non-co-optimized reference, and `dispatch.solve_joint_capacity_batch()` can provide a screening-grade joint MILP power-headroom estimate.
5. `export.py` writes Excel reports that must respect the same `tz` parameter and stacked-revenue breakdown as the dashboard.

Tests in `tests/` are heavily mocked (no live API calls) and mirror the module layout 1:1. `conftest.py` holds shared price/ancillary fixtures.

## Data Sources

### ENTSO-E Transparency Platform
- Requires API key (loaded from .env)
- Covers: EU-27 + Norway (NO_1–NO_5) + Switzerland (CH)
- Python library: entsoe-py (>=0.7.7, uses `web-api.tp.entsoe.eu`; endpoint overridable via `ENTSOE_ENDPOINT_URL` env var)
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
- Elexon returns multiple data providers per settlement period. Legitimate £0/MWh and negative prices must be retained; only recognizable duplicate zero placeholders are dropped before averaging remaining provider rows.

### Fingrid Open Data (Finland)
- API key recommended/required for v2 (`x-api-key` header via `FINGRID_API_KEY`)
- Base URL: https://data.fingrid.fi/api
- Dataset paths use `/datasets/{id}/data`
- Dataset IDs: 317 (FCR-N), 318 (FCR-D Up), 283 (FCR-D Down), 52 (aFRR Up), 51 (aFRR Down)
- Returns: hourly capacity prices in EUR/MW

### Regelleistung.net (Germany)
- NO API key required — public REST API
- Base URL: `https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders/resultsoverview`
- Params: `productTypes` (FCR/aFRR), `market` (CAPACITY), `exportFormat` (xlsx), `date` (YYYY-MM-DD)
- Auto-fetch enabled: downloads daily tender results as xlsx, parsed into standard ancillary format
- Manual CSV upload (`DE_FCR`, `DE_aFRR`) still supported as fallback and overrides auto-fetch for same product type
- Returns: capacity prices in EUR/MW

### Netztransparenz.de (Germany, Phase 3 reference)
- Public German TSO transparency portal for balancing capacity/energy publications.
- Base URL reserved in config as `NETZTRANSPARENZ_BASE_URL`; no fetcher is implemented yet.
- Relevant future datasets: activated aFRR/mFRR, GCC/LFC area balance, reBAP imbalance prices, and reBAP components.
- Not a replacement for the implemented Regelleistung.net tender-result fetcher in this repo. Treat it as a future activation/reBAP diagnostics source until an explicit fetcher is added.

### ENTSO-E Imbalance Prices
- Uses same ENTSO-E API key as DA prices
- Covers: RO, SE_3, Italian bidding zones, and other ENTSO-E zones where imbalance data is available
- Returns: imbalance/balancing prices in EUR/MWh

### ENTSO-E Intraday Auction Prices (IDA1/2/3)
- Uses same ENTSO-E API key as DA prices via `entsoe-py.query_intraday_prices(zone, start, end, sequence)`.
- `INTRADAY_SUPPORTED_ZONES` (data_ingestion.py) lists the SIDC participants that publish auction results — currently `{DE_LU, NL, BE, FR, AT, IT_NORD}`. Other zones return `None` rather than raising.
- Sequence: 1 = IDA1 (15:00 D-1 opening auction), 2 = IDA2 (22:00 D-1), 3 = IDA3 (10:00 day-of-delivery). The dashboard currently uses IDA1 only.
- `entsoe-py` raises `NoMatchingDataError` (a `ValueError` subclass) when a supported zone has no IDA data for the requested window — this is mapped to `None`, not a hard error.
- Auth failures raise `DataSourceAuthError`; network failures raise `DataSourceNetworkError`.

## Geographic Scope

Three tiers of zones:

1. **EU member states** (ENTSO-E): DE_LU, FR, NL, BE, AT, PL, IT_NORD, IT_CNOR, IT_CSUD, IT_SUD, IT_CALA, IT_SICI, IT_SARD, ES, PT, RO, DK_1, DK_2, SE_1, SE_2, SE_3, SE_4, FI, CZ, HU, BG, GR, HR, SK, SI, EE, LT, LV, IE_SEM
2. **Non-EU ENTSO-E members** (same entsoe-py API): NO_1, NO_2, NO_3, NO_4, NO_5, CH
3. **Great Britain** (Elexon API): GB

## Ancillary Services

### Auto-Fetch (ancillary_fetchers.py)
Zone-specific fetchers run automatically when the zone is selected:
- **FI** — FCR-N/D + aFRR prices via Fingrid
- **DE_LU** — FCR/aFRR auction results via Regelleistung.net REST API (downloads daily xlsx)
- **GB** — System buy/sell prices via Elexon
- **RO, SE_3, IT_NORD, IT_CNOR, IT_CSUD, IT_SUD, IT_CALA, IT_SICI, IT_SARD** — Imbalance prices via ENTSO-E

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
`build_ancillary_dataset()` merges manual + auto-fetch data. Manual uploads override auto-fetch for the same product type only — other auto-fetched products are preserved. `normalize_auto_fetch_dataset()` converts varied fetcher schemas into a standard ancillary format with per-product rows. Override matching uses `_normalize_product_key()` (case- and separator-insensitive) so an upstream label flip from `aFRR Up` to `afrr_up` does not silently bypass the override.

### UI Guidance
- Revenue Estimation tab shows an annualisation note based on the currently selected sample window length
- Revenue Estimation tab includes a `How ancillary works` expander explaining product-level stacking, manual-vs-auto precedence, and which ancillary signals are not auto-monetised

### Auth Errors
Fingrid / Regelleistung / Elexon all detect HTTP 401/403 via the shared `_raise_if_auth_failed()` helper and raise `DataSourceAuthError` *before* the `@retry` decorator can swallow it. `run_auto_fetch()` deliberately does not catch `DataSourceAuthError`, so it propagates to `_run_and_store_ancillary_fetch()` in `src/components/sidebar.py` which surfaces it via `st.error` with a hint (e.g. "Set FINGRID_API_KEY in .env."). Other transient fetcher errors (network, parse) are still caught per-fetcher and logged.

### Data Quality
`clean_prices()` short-gap interpolates gaps up to `MAX_SHORT_GAP_HOURS` (config.py) and flags those rows with `imputed=True`. Longer gaps stay NaN with `filled=True / imputed=False`. `_segment_and_reindex_prices()` also detects sparse internal gaps when no `expected_window` is provided (mode-delta heuristic). For zones listed in `_ZONE_RESOLUTION_TRANSITIONS` (declare new ones there when a TSO changes its market time unit) the helper splits the index at each transition boundary and runs the mode-delta heuristic on each side independently — this surfaces sparse gaps inside both the pre- and post-transition segments without upsampling the lower-resolution side. `analytics.filter_to_complete_local_days(df, tz)` mirrors the day-level filter that `calculate_daily_spreads()` uses, and is applied in `app.py` and `src/export.py` so negative-price stats and heatmaps stay consistent with daily-spread / dispatch / revenue analytics across the dashboard, Excel, and PDF outputs. Excel/PDF exports always recompute the negative-price stats from the filtered subset internally; do not rely on the caller-supplied `negative_stats` to be already filtered.

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
- **Revenue stacking**: `merge_revenue_stack()` exposes DA arbitrage + ancillary `source_revenues` for per-product breakdown, but capacity-reserve ancillary is not assumed to be fully additive with DA in the headline total without co-optimization.
- **Joint co-optimization**: `solve_joint_capacity_batch()` models reserve capacity as power headroom competing with DA charge/discharge in a MILP with binary charge/discharge mutual exclusion. It does not model activation energy, bid acceptance, reserve-specific SoC duration, or qualification constraints.
- **Mutual exclusion is binary, not relaxed**: `solve_daily_lp()` and `solve_daily_joint_capacity_lp()` both use a binary mode variable via scipy `integrality`. A pure LP relaxation of `p_charge + p_discharge <= power_mw` is degenerate at negative prices (the solver can earn revenue by simultaneously charging and discharging through round-trip losses). Do not weaken this back to an LP-only constraint.
- **Annualisation caveat**: DA arbitrage revenue is extrapolated from the user-selected sample window, so short windows (for example winter-only periods) can materially overstate or understate full-year merchant potential
- **Portfolio analysis** (`src/portfolio.py`): treats each zone as a daily-revenue-per-MW series and runs Pearson correlation, Sharpe-like mean/std, and a long-only Markowitz frontier via scipy SLSQP. `build_daily_revenue_matrix()` inner-joins on dates so a single missing local day in any zone drops that date from the whole portfolio view — this keeps correlations and frontier weights computed on a single common sample. Annualisation uses the same `365.25`-day convention as `estimate_annual_arbitrage_revenue` and assumes i.i.d. days (`std_annual = std_daily * sqrt(365.25)`); for short or seasonal samples treat the Sharpe number as a relative ranking, not an absolute investment metric.
- **Intraday uplift (P5-A Phase 1)** (`analytics.calculate_intraday_uplift`): joins DA and IDA1 prices on the delivery timestamp and reports `avg|IDA-DA|`, `mean_signed` (positive ⇒ IDA usually prints above DA), and an annualised rebid uplift = `avg|IDA-DA| * rebid_share * duration_hours * cycles_per_day * capture_rate * 365.25`. The default `rebid_share=0.25` is intentionally conservative; EU practitioner ranges are typically 0.10–0.40 depending on DA commitment strategy and SoC management. The model is a single-stage screening estimate, NOT a proper two-stage DA + ID dispatch — it ignores rebid directionality (a BESS short on DA can only buy back in ID) and treats every absolute spread as fully captureable on average. IDA1 fetches are persisted to SQLite (`ida_prices_{zone}_seq{n}` tables) so browser refresh does not re-hit the ENTSO-E API; the Revenue Estimation UI tries the cache first and falls back to a button-gated live fetch. **Phase 2** would solve a two-stage stochastic dispatch problem: stage 1 commits a DA position under an IDA price forecast, stage 2 rebids into IDA against realised prices with explicit rebid-direction feasibility against the existing DA position. Implementation candidates: scenario-based stochastic MILP, or a sequential LP heuristic that re-optimises against the actual IDA print each day.

## Commands
- `pip install -r requirements.txt` — install deps (Python 3.11+; use `.venv` on macOS).
- `python -m pytest tests/ -v` — run all tests (313 passing tests, fully mocked, no network; 2 PDF chart-render tests may skip when local Kaleido is unavailable).
- `python -m pytest tests/test_analytics.py::TestOrderedSpreads -v` — run a single class; swap in `::test_name` for a single test.
- `streamlit run app.py` — launch the dashboard.
- `python -c "from src.data_ingestion import test_elexon_connection; test_elexon_connection()"` — smoke-test Elexon (no API key needed).
- `python -c "from src.data_ingestion import test_entsoe_connection; test_entsoe_connection()"` — smoke-test ENTSO-E (needs `ENTSOE_API_KEY` in `.env`).

CI runs `python -m pytest tests/ -v` on every push/PR via `.github/workflows/ci.yml`; keep the mocked-test suite green before pushing. See `CONTRIBUTING.md` for PR expectations and secret-handling rules.
