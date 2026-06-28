# Import templates (IDA prices + reserve capacity)

This is the spec to hand to an exchange / TSO / aggregator when requesting sample
data. Both formats flow through one provenance / cache / Data Trust / Simulation
Cockpit path. Download the live templates from the dashboard sidebar
(**Ancillary Services Data** and **Intraday (IDA) Prices** expanders).

> Step 2 status: the **reserve-capacity** template + spec ship now (6a); the
> unified-capacity parser + persistence + provenance land in the next increment
> (6b). The **IDA** import path is already live end-to-end.

## 1. IDA prices CSV (live)

One row per delivery interval. Columns (header match is case-insensitive):

| Column | Required | Notes |
|---|---|---|
| `timestamp` | yes | UTC, ISO-8601 (e.g. `2026-05-01T13:00:00Z`). A naive value is assumed UTC. |
| `ida_price_eur_mwh` | yes | Intraday auction price, EUR/MWh. Negatives are valid and kept. |
| `sequence` | no | `1`/`2`/`3` or `IDA1`/`IDA2`/`IDA3`. Falls back to the UI default round only when blank; a non-1/2/3 value is rejected (never silently filed under IDA1). |
| `zone` | no | Bidding-zone code; falls back to the selected zone when blank. |

Rows write to the `ida_prices_{zone}_seq{n}` tables and are labelled by source in
Data Trust (`Manual CSV` vs live).

## 2. Reserve capacity CSV (unified, zone-tagged)

The single format for all zones. Columns:

| Column | Required | Notes |
|---|---|---|
| `timestamp` | yes | **UTC**, ISO-8601. If your export is local market time, convert to UTC **or** add a `timezone` column. |
| `zone` | yes | Bidding-zone code (e.g. `DE_LU`, `FI`, `FR`). |
| `product` | yes | `FCR` \| `aFRR` \| `mFRR` (case- and separator-insensitive on import). |
| `direction` | yes | `up` \| `down` \| `symmetric` (FCR is symmetric; aFRR/mFRR up or down). |
| `capacity_price_eur_mw_h` | yes | **PER-HOUR** capacity price in EUR/MW/h — **not** a 4h-block total. One row per pricing block (e.g. 4h) is fine; give the hourly rate, not the block sum. |
| `timezone` | no | IANA name (e.g. `Europe/Berlin`) if `timestamp` is local; converted to UTC on import. |

### Two semantics we pin (historical pitfalls)
- **Time**: timestamps are UTC unless a `timezone` column says otherwise. German
  FCR/aFRR blocks are defined in local time — convert, or tell us the zone.
- **Unit**: `capacity_price_eur_mw_h` is a per-hour rate. A 4h block that cleared
  at, say, 12.5 EUR/MW/h is `12.50` repeated across the block, not `50`.

## Where to source samples
- **IDA1/2/3 prices** — power exchanges, not ENTSO-E (live IDA there is empty):
  Nord Pool Data Services (Intraday Auctions; Nordic/Baltic + CWE), EPEX SPOT /
  EEX Group Webshop (DE_LU, FR, NL, BE, AT), or an aggregator/trading-desk export.
- **Reserve capacity** — TSO / reserve platforms: `regelleistung.net` (DE FCR/aFRR/
  mFRR tender results), Fingrid Open Data (FI). Other countries: national TSO
  portals (Elia, RTE, Terna, REE, TenneT) — fields/licence/granularity vary.
