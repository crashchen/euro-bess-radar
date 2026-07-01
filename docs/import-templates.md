# Import templates (IDA prices + reserve capacity + activation energy + imbalance)

This is the spec to hand to an exchange / TSO / aggregator when requesting sample
data. Each format flows through one provenance / cache / Data Trust / Simulation
Cockpit path. Download the live templates from the dashboard sidebar
(**Ancillary Services Data** and **Intraday (IDA) Prices** expanders).

> Status: the **IDA**, **reserve-capacity**, and **activation-energy** import
> paths are live end-to-end (upload → parse → SQLite + provenance → Data Trust
> → cockpit where applicable). The **reBAP / imbalance-settlement** import is
> live through Data Trust, with the Step 4d-1 replay calculation layer added as
> a non-additive overlay primitive; cockpit UI wiring lands separately.

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

## 3. Activation energy CSV (unified, zone-tagged)

The **energy** leg of reserves — paid only when the TSO actually calls (activates)
the reserve, distinct from the standing capacity fee. Columns:

| Column | Required | Notes |
|---|---|---|
| `timestamp` | yes | **UTC**, ISO-8601. If your export is local market time, convert to UTC **or** add a `timezone` column. |
| `zone` | yes | Bidding-zone code (e.g. `DE_LU`, `FI`, `FR`). |
| `product` | yes | `aFRR` \| `mFRR` (case- and separator-insensitive on import). FCR has no separately-paid energy leg. |
| `direction` | yes | `up` \| `down`. Energy activation is directional — there is no `symmetric`. |
| `activation_price_eur_mwh` | yes | Energy price paid/charged **when activated**, EUR/MWh. Negatives are valid (down-activation can pay you to absorb). |
| `system_activated_volume_mw` | yes | **SYSTEM-level** activated power in the interval, MW — **not** this asset's output. |
| `timezone` | no | IANA name (e.g. `Europe/Berlin`) if `timestamp` is local; converted to UTC on import. |

### Three red-lines we pin
- **System vs asset**: `system_activated_volume_mw` is the *market/area* activated
  volume the TSO publishes. The fraction *this* battery captures (its
  asset/capture share) is a **model assumption** that lives in the audit panel —
  it must never be pre-mixed into the data file.
- **Separate streams**: activation energy is its own revenue stream, distinct
  from the **capacity fee** (§2) and from **reBAP/imbalance** settlement (a
  different, passive-balancing strategy, a later template). Do not blindly sum
  them — activating *up* spends SoC that DA/IDA could otherwise have sold, so it
  competes for the same energy budget rather than stacking for free.
- **Replay only**: with no forward activation signal this supports **historical
  replay** of what an asset *would have* earned, **not** live dispatch or
  aggregator activation-following.
- **Regular interval series**: each row must be one settlement interval with
  `system_activated_volume_mw` as the *average activated MW over that interval*.
  The cockpit overlay infers the interval length from the timestamp spacing, so
  upload a regular series — a sparse "activation event" feed (only non-zero
  intervals) would understate the interval count and misstate energy.

## 4. reBAP / imbalance settlement CSV (unified, zone-tagged)

The passive imbalance-settlement stream — e.g. German reBAP or a national
imbalance price — is distinct from both the reserve capacity fee and activated
balancing-energy payments. Columns:

| Column | Required | Notes |
|---|---|---|
| `timestamp` | yes | **UTC**, ISO-8601. If your export is local market time, convert to UTC **or** add a `timezone` column. |
| `zone` | yes | Bidding-zone or settlement-area code (e.g. `DE_LU`, `FI`, `FR`). |
| `imbalance_price_eur_mwh` | yes | Published imbalance/reBAP settlement price, EUR/MWh. Negatives are valid and kept. Treat this as a cash-flow price as published; do not sign-flip it by a separate direction field. |
| `system_imbalance_volume_mw` | yes | **SYSTEM/area** imbalance volume in the interval, MW — **not** this asset's imbalance. For the replay overlay, use the German Netztransparenz sign convention: positive = system short / undercovered (discharge helps), negative = system long / overcovered (charge helps). Convert other providers to this convention before using the overlay. |
| `timezone` | no | IANA name (e.g. `Europe/Berlin`) if `timestamp` is local; converted to UTC on import. |

Rows write to per-zone `imbalance_prices_{zone}` tables and a per-zone
`imbalance_price_sources` sidecar. Re-importing the same timestamp keeps the
last row and refreshes provenance. Data Trust shows this as a per-zone
`source (rows)` stream; cockpit replay is intentionally a later step.

### Four red-lines we pin
- **Separate strategy**: imbalance settlement is a passive BRP/portfolio
  settlement stream. It is not the reserve capacity fee (§2) and not the
  activation-energy leg (§3). Do not blindly add it to reserve revenues.
- **System vs asset**: `system_imbalance_volume_mw` is the TSO/system quantity.
  Any asset imbalance volume, portfolio share, or passive-balancing capture
  assumption belongs in the model/audit panel, never pre-mixed into the CSV.
- **System sign convention**: for the German Netztransparenz NRV-Saldo/reBAP
  path, positive system imbalance means system short / undercovered, so positive
  BESS net dispatch (discharge / extra injection) helps; negative system
  imbalance means system long / overcovered, so negative BESS net dispatch
  (charge / less injection) helps. Other sources must be normalised to this
  convention before the replay overlay is interpreted.
- **Cash-flow price**: `imbalance_price_eur_mwh` is the published settlement
  price. It may be negative; the parser/model must not apply an additional
  direction sign flip.
- **Replay only**: this supports historical replay / diagnostics of a passive
  imbalance strategy, not live BRP control or aggregator dispatch.
- **Non-additive overlay**: the Step 4d-1 calculation uses
  `asset_net_dispatch_mw = sign(system_imbalance_volume_mw) *
  min(power_mw, capture_share * abs(system_imbalance_volume_mw))` and
  `cashflow = asset_net_dispatch_mw * dt * imbalance_price_eur_mwh`. It ignores
  SoC/energy sustainability and is therefore an overlay diagnostic, not a
  strategy-ladder revenue stream.

## Where to source samples
- **IDA1/2/3 prices** — power exchanges, not ENTSO-E (live IDA there is empty):
  Nord Pool Data Services (Intraday Auctions; Nordic/Baltic + CWE), EPEX SPOT /
  EEX Group Webshop (DE_LU, FR, NL, BE, AT), or an aggregator/trading-desk export.
- **Reserve capacity** — TSO / reserve platforms: `regelleistung.net` (DE FCR/aFRR/
  mFRR tender results), Fingrid Open Data (FI). Other countries: national TSO
  portals (Elia, RTE, Terna, REE, TenneT) — fields/licence/granularity vary.
- **Activation energy** — TSO balancing-energy publications: `netztransparenz.de`
  (DE activated aFRR/mFRR volumes + prices), the ENTSO-E balancing platform
  reports (PICASSO aFRR / MARI mFRR clearing), or national TSO portals. Take the
  *system/area* activated volume, not a pre-derived per-asset figure.
- **reBAP / imbalance settlement** — `netztransparenz.de` (DE reBAP + system
  imbalance volumes), national TSO imbalance-price publications, BRP settlement
  exports, or aggregator back-office exports. Keep the settlement price and
  system imbalance volume separate from any per-asset imbalance assumption.
