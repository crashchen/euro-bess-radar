# euro-bess-radar

European BESS Market Screening Dashboard — evaluate battery energy storage merchant revenue potential across European electricity markets.

## Features

- **Day-ahead price analysis** across 35+ European bidding zones (ENTSO-E + Elexon)
- **Revenue estimation** for 1h, 2h, and 4h BESS using chronology-aware charge/discharge windows
- **Renewable correlation analysis** — quantify how wind/solar drives price spreads
- **Ancillary services integration** — manual CSV upload + auto-fetch for DE, FI, GB with per-product reserve lines, manual-over-auto product overrides, and preserved directional/system price fields
- **Multi-zone comparison** for market screening
- **Excel export** with full analytics and sub-hourly negative-price normalization
- **GBP to EUR normalization** for GB history using yearly FX mappings

## Data Sources

| Source | Coverage | Auth Required |
|--------|----------|---------------|
| ENTSO-E Transparency Platform | EU-27 + NO + CH | API key (free) |
| Elexon Insights API | Great Britain | None |
| Fingrid Open Data | Finland | None |
| Regelleistung.net | Germany | None (best-effort) |

## Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd euro-bess-radar
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure ENTSO-E API key
echo "ENTSOE_API_KEY=your_key_here" > .env

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
│   ├── analytics.py          # Spread, P50/P90, heatmaps, RE correlation
│   ├── ancillary.py          # Ancillary services parsing & revenue calc
│   ├── ancillary_fetchers.py # Auto-fetch registry per zone
│   └── export.py             # Excel report generation
├── tests/                    # 99 unit tests, heavily mocked
├── data/
│   ├── cache/                # SQLite + CSV (git-ignored)
│   └── manual/               # Manual CSV uploads
└── CLAUDE.md                 # Project spec & conventions
```

## Key Markets

Optimized for BESS investment screening in:
- Germany (DE_LU) — 15min resolution, FCR/aFRR auto-fetch
- Finland (FI) — FCR-N/D + aFRR auto-fetch via Fingrid
- Great Britain (GB) — Elexon MID + system prices
- Romania (RO) — DA prices + ENTSO-E imbalance data
- Italy South (IT_SUD) — DA prices + ENTSO-E imbalance
- Sweden (SE_1-4) — Nordic market dynamics
- Norway (NO_1-5) — Hydro-driven price patterns

## License

Private — internal use only.
