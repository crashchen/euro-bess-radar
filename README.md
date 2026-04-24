# euro-bess-radar

European BESS Market Screening Dashboard — evaluate battery energy storage merchant revenue potential across European electricity markets.

## Features

- **Day-ahead price analysis** across 35+ European bidding zones (ENTSO-E + Elexon)
- **Revenue estimation** for 1h, 2h, and 4h BESS using chronology-aware charge/discharge windows
- **Inline revenue guidance** with sample-window annualisation notes and ancillary methodology help in the dashboard
- **Renewables and BESS signal analysis** — quantify how renewable share affects prices and ordered spreads
- **Ancillary services integration** — manual CSV upload + auto-fetch for DE, FI, GB with per-product reserve lines, manual-over-auto product overrides, and preserved directional/system price fields
- **Multi-zone comparison** for market screening
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
| Netztransparenz.de | Germany | Future/reference source for balancing transparency, activated balancing, and reBAP |

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
│   ├── data_ingestion.py     # ENTSO-E, Elexon, Fingrid fetchers
│   ├── analytics.py          # Spread, P50/P90, heatmaps, renewables/BESS signals
│   ├── ancillary.py          # Ancillary services parsing & revenue calc
│   ├── ancillary_fetchers.py # Auto-fetch registry per zone
│   └── export.py             # Excel report generation
├── tests/                    # 151 unit tests, heavily mocked
├── data/
│   ├── cache/                # SQLite + CSV (git-ignored)
│   └── manual/               # Manual CSV uploads
└── CLAUDE.md                 # Project spec & conventions
```

## Key Markets

Optimized for BESS investment screening in:
- Germany (DE_LU) — 15min resolution, FCR/aFRR auto-fetch via Regelleistung REST API
- Finland (FI) — FCR-N/D + aFRR auto-fetch via Fingrid
- Great Britain (GB) — Elexon MID + system prices
- Romania (RO) — DA prices + ENTSO-E imbalance data
- Italy South (IT_SUD) — DA prices + ENTSO-E imbalance
- Sweden (SE_1-4) — Nordic market dynamics
- Norway (NO_1-5) — Hydro-driven price patterns

## Germany Balancing Data Roadmap

Regelleistung.net remains the implemented German auto-fetch source for FCR/aFRR tender-result capacity prices. Netztransparenz.de is tracked as a Phase 3 reference source for German balancing transparency datasets, especially activated aFRR/mFRR, GCC/LFC area balance, and quarter-hourly reBAP imbalance prices. These datasets are useful for future activation/reBAP diagnostics, but they are not yet wired into the automated revenue model.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Code and Data Licensing

The source code in this repository is licensed under Apache-2.0.

Fetched, cached, uploaded, or exported market data is not relicensed by this repository. Data from ENTSO-E, Elexon, Fingrid, Regelleistung, and other external providers remains subject to the original source terms, access rules, attribution requirements, and permitted-use restrictions.

## Security

Please report suspected vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup guidance, PR expectations, and secret-handling rules.
