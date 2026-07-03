# euro-bess-radar

European BESS Market Screening Dashboard — evaluate battery energy storage merchant revenue potential across European electricity markets.

## Features

- **Day-ahead price analysis** across 37+ European bidding zones (ENTSO-E + Elexon)
- **Revenue estimation** for 1h, 2h, and 4h BESS using chronology-aware charge/discharge windows
- **Inline revenue guidance** with sample-window annualisation notes and ancillary methodology help in the dashboard
- **Renewables and BESS signal analysis** — quantify how renewable share affects prices and ordered spreads
- **Ancillary services integration** — manual CSV upload + auto-fetch for DE, FI, GB, RO, SE_3, and Italian zones with per-product reserve lines, manual-over-auto product overrides, and preserved directional/system price fields; plus unified zone-tagged reserve-capacity, activation-energy, and reBAP/imbalance CSV imports as separate streams, each with SQLite persistence and provenance; German reBAP/NRV imbalance data can also be fetched live from Netztransparenz.de into the same cache
- **Intraday (IDA) price import** — manual CSV fallback for IDA1/2/3 prices when the ENTSO-E intraday-auction API returns no data, persisted to the same cache the live fetch uses and labelled `Manual CSV` in Data Trust
- **Joint MILP co-optimization estimate** for DA arbitrage vs reserve-capacity power headroom
- **Simulation Cockpit** for interval-level BESS dispatch replay, event tables, multi-day summaries with continuous-horizon SoC carry-over, a forecast-driven sequential DA+ID policy (vs perfect-foresight ceiling) with rebid deadband + forecast-skill report, an annualised strategy comparison (with optional DA + reserve-capacity co-optimisation, a cumulative DA + IDA1 + reserve perfect-foresight ceiling, and a Phase 9.2b forecast-driven realistic reserve-first row with a forecast-effect gap panel when ancillary capacity prices are loaded), activation-energy and reBAP/imbalance historical replay overlays (separate, non-additive screening estimates when those streams are imported), and Excel export — plus SoC, revenue, throughput, and battery-health diagnostics
- **Multi-zone comparison** for market screening
- **Data Trust diagnostics** showing source, timezone, coverage, source gaps, and imputation per fetched zone, plus a zone × data-stream **coverage matrix** (DA / IDA1–3 / reserve capacity / activation energy / imbalance settlement), per-(zone, product, direction) provenance tables for imported reserve-capacity and activation-energy prices, and per-zone provenance for imported reBAP/imbalance prices
- **Excel export** with full analytics and sub-hourly negative-price normalization
- **GBP to EUR normalization** for GB history using yearly FX mappings
- **GitHub Actions CI** for syntax validation and mocked unit tests on pushes and PRs

## Data Sources

| Source | Coverage | Auth Required |
|--------|----------|---------------|
| ENTSO-E Transparency Platform | EU-27 + NO + CH | API key (free) |
| Elexon Insights API | Great Britain | None |
| Fingrid Open Data v2 | Finland | API key recommended (`FINGRID_API_KEY`) |
| Regelleistung.net | Germany | REST API (FCR/aFRR tender results) + manual CSV fallback |
| Netztransparenz.de | Germany | Public CSV download handler for DE_LU reBAP + NRV-Saldo; activation datasets remain future/manual |

## Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd euro-bess-radar
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure API keys
echo "ENTSOE_API_KEY=your_key_here" > .env
# Optional: Fingrid Open Data v2 key for Finnish ancillary data
echo "FINGRID_API_KEY=your_key_here" >> .env

# Run tests
python -m pytest tests/ -v

# Launch dashboard
streamlit run app.py
```

## Project Structure

```
euro-bess-radar/
├── app.py                    # Streamlit dashboard
├── src/
│   ├── config.py             # Zones, API endpoints, constants
│   ├── data_ingestion.py     # ENTSO-E, Elexon, Fingrid, Netztransparenz fetchers
│   ├── analytics.py          # Spread, P50/P90, heatmaps, renewables/BESS signals
│   ├── dispatch.py           # MILP dispatch for multi-cycle daily arbitrage
│   ├── simulation.py         # Interval-level dispatch replay + multi-day continuous-horizon carry-over
│   ├── ida_forecast.py       # Hourly climatology IDA forecast for the sequential DA+ID policy
│   ├── reserve_forecast.py   # Block-of-day capacity-price forecast skill (Phase 9.2b prep)
│   ├── da_forecast.py        # Screening DA-price climatology forecast (Phase 9.2b Stage-0 prep)
│   ├── degradation.py        # Throughput-based degradation and lifetime metrics
│   ├── ancillary.py          # Ancillary services parsing & revenue calc
│   ├── ancillary_fetchers.py # Auto-fetch registry per zone
│   ├── activation_overlay.py # Activation-energy replay overlay (screening, non-additive)
│   ├── imbalance_overlay.py  # reBAP/imbalance replay overlay primitive
│   └── export.py             # Excel report generation
├── tests/                    # 705 passing tests, heavily mocked; 2 PDF tests may skip
├── scripts/                  # Maintenance/demo scripts (seed + Netztransparenz converter)
├── samples/                  # Generated demo CSVs from seed_demo_9_2b.py (git-ignored)
├── docs/runbooks/            # Operator runbooks (9.2b + imbalance validation)
├── data/
│   ├── cache/                # SQLite + CSV (git-ignored)
│   └── manual/               # Manual CSV uploads
└── CLAUDE.md                 # Project spec & conventions
```

## Key Markets

Optimized for BESS investment screening in:
- Germany (DE_LU) — 15min resolution, FCR/aFRR auto-fetch via Regelleistung REST API, plus reBAP/NRV imbalance fetch via Netztransparenz
- Finland (FI) — FCR-N/D + aFRR auto-fetch via Fingrid
- Great Britain (GB) — Elexon MID + system prices
- Romania (RO) — DA prices + ENTSO-E imbalance data
- Italy (IT_NORD, IT_CNOR, IT_CSUD, IT_SUD, IT_CALA, IT_SICI, IT_SARD) — DA prices + ENTSO-E imbalance auto-fetch where available
- Sweden (SE_1-4) — Nordic market dynamics
- Norway (NO_1-5) — Hydro-driven price patterns

## Germany Balancing Data Roadmap

Regelleistung.net remains the implemented German auto-fetch source for FCR/aFRR tender-result capacity prices. Netztransparenz.de is now wired for quarter-hourly DE_LU reBAP + NRV-Saldo downloads: the sidebar fetches the official CSV handler, persists rows into the same imbalance cache/provenance path as manual uploads, and feeds the non-additive cockpit imbalance overlay. Activated aFRR/mFRR energy datasets remain manual/future until their schema and replay semantics are validated.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Code and Data Licensing

The source code in this repository is licensed under Apache-2.0.

Fetched, cached, uploaded, or exported market data is not relicensed by this repository. Data from ENTSO-E, Elexon, Fingrid, Regelleistung, and other external providers remains subject to the original source terms, access rules, attribution requirements, and permitted-use restrictions.

## Security

Please report suspected vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup guidance, PR expectations, and secret-handling rules.
