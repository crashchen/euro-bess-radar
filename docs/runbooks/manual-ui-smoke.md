# Manual UI smoke checklist (last-inch browser acceptance)

Every import/fetch path in this repo is covered by mocked tests up to — but
not including — the literal Streamlit widget interaction (file picker, button
click, success/error rendering). This checklist covers exactly that last inch.
Run it after changes to `src/components/sidebar.py` wiring or after a
Streamlit version bump; each item takes well under a minute.

Mocked CI cannot see `app.py`/sidebar-only breakage (a past regression shipped
a `NameError` that crashed every fresh session while CI stayed green), so this
list is the cheap guard for that class of bug.

## Setup

```bash
streamlit run app.py
```

Select `DE_LU` as the zone. For live fetches pick a window with published
data (for activation energy the quality-assured volumes lag ~1 month, so use
e.g. the month before last). `ENTSOE_API_KEY` must be set in `.env` for the
activation fetch; the imbalance fetch is keyless.

Sample files: download each template from the sidebar itself (the templates
double as minimal valid uploads), or reuse `samples/` files where present.

## Checklist — Ancillary Services Data expander

For each uploader: choose the file, click the parse/import button, and expect
a `st.success` with a per-stream row count — not a stack trace. Then open the
**Data Trust** tab and confirm the matching provenance row.

| # | Entry | Action | Expect |
|---|-------|--------|--------|
| 1 | Template downloads (capacity / activation / imbalance / per-country) | Click each download button | A CSV downloads; header matches `docs/import-templates.md` |
| 2 | Unified Reserve Capacity CSV | Upload template (or `samples/unified_capacity_sample.csv`) → **Parse & Import capacity** | Success message with (zone, product, direction) counts; Data Trust reserve source table row `Manual CSV` |
| 3 | Unified Activation-Energy CSV | Upload activation template → **Parse & Import activation** | Success message; Data Trust activation source table row `Manual CSV`, unpriced columns blank |
| 4 | **Fetch Netztransparenz + ENTSO-E activation energy** (DE_LU only) | Click with a ~1-month-old window | Success message `Fetched N rows (…)` that also states the dropped-unpriced count (an explicit "all … carried a published price" when zero); Data Trust activation row `Netztransparenz.de + ENTSO-E 17.1.f` with `Unpriced dropped` / `Unpriced max MW` populated |
| 5 | Same button, too-recent window | Click with e.g. the current week | Friendly error mentioning the ~1 month publication lag — not a stack trace |
| 6 | Same button, missing API key | Unset `ENTSOE_API_KEY`, restart, click | Friendly auth error telling you to set `ENTSOE_API_KEY` in `.env` |
| 7 | Unified reBAP / Imbalance CSV | Upload imbalance template → **Parse & Import imbalance** | Success message; Data Trust imbalance source row `Manual CSV` |
| 8 | **Fetch Netztransparenz reBAP/imbalance** (DE_LU only) | Click with a recent window | Success message with row count; Data Trust imbalance source row `Netztransparenz.de` |
| 9 | Non-DE_LU zone selected | Switch zone to e.g. `FR` | Both live-fetch buttons are replaced by "available for DE_LU only" captions |

## Checklist — other sidebar entries

| # | Entry | Action | Expect |
|---|-------|--------|--------|
| 10 | Intraday (IDA) Prices expander | Upload IDA template CSV | Success message; Data Trust intraday source table shows `Manual CSV` for the (zone, sequence) |
| 11 | Per-country ancillary CSV (e.g. `DE_FCR`) | Upload the country template | Success; Revenue tab ancillary section reflects the product |
| 12 | Auto-Fetch Ancillary Data | Click fetch for DE_LU | Regelleistung results stored or a friendly per-fetcher error — never an unhandled exception |

## Downstream spot-checks (after 3/4/7/8)

- **Data Trust coverage matrix**: the touched zone row shows the stream
  (`activation_energy` / `imbalance_settlement` / reserve) as `source (rows)`
  or a product list — not `—`.
- **Simulation Cockpit**: with activation rows cached and a window overlapping
  them, the activation-energy overlay expander appears; same for the imbalance
  overlay. Both captions state the non-additive replay red-line.
- **imported_at sanity**: re-run one live fetch; in Data Trust only the
  streams that fetch touched get a fresh `Imported (UTC)` — other streams keep
  their previous timestamp (per-stream last-write semantics).

## Scope

This checklist is deliberately manual — automating the Streamlit file-picker
adds a browser-driver dependency for marginal value. Everything below the
widget layer (parsers, persistence, provenance, Data Trust tables, overlays)
is covered by the mocked suite; see `tests/`.
