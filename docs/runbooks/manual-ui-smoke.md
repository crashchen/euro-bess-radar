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

## Checklist — Project Case contract entry (Revenue Estimation, PC-D3)

The AppTest suite drives this panel headless, so the logic below the widget
layer is covered. What it cannot see is what a reader actually sees: whether a
disclosure is legible, whether a fail-closed path visibly withholds the run
button, and whether the post-run text matches the result rather than the live
widgets. Run this after any change to `src/pages/project_case.py`, to the
locked disclosure literals, or to `src/ui_theme.py`.

Setup: load `DE_LU` with a window that yields several weeks of clean days, set
a non-zero CapEx, then scroll to **Project Case — lifecycle valuation**.

| # | Entry | Action | Expect |
|---|-------|--------|--------|
| 13 | Default state | First confirm **Contracted floor settlement (optional)** is *collapsed* on arrival, then expand it and read the contract-mode control | Collapsed by default, so a merchant-only run needs no interaction; on expanding, the mode reads `No contract — merchant-only settlement` and the NPVs are merchant-only |
| 14 | Capacity-maintenance fail-closed | Pick `No augmentation required` and leave the engineering source blank | Error `capacity_maintenance_source must be non-empty after trimming`; **Run Project Case is withheld**, not merely disabled-looking. Restore a non-empty source before continuing — every later item needs a runnable baseline |
| 15 | Product-boundary disclosures | Expand the contract expander | Two captions: the locked "not MACSE / not a complete legal-contract model / not a bankable valuation" sentence, and the cockpit-sibling sentence naming the wear-net comparator as a different product |
| 16 | Quote entry resolves to one curve | Set tenor 10, pick `Escalating real quote`, rate 150000, escalation 2% | Preview table shows contract years 1–10 with rates 150000 / 153000 / 156060 …; `effective_whole_project_floor_eur` equals rate × sidebar MW × entitlement factor |
| 17 | Coverage wording | Read the caption above the preview | "N covered project year(s), X to Y. Years outside the term have no floor at all; that is not a zero floor." — no wording that implies a zero floor |
| 18 | Bound values are not re-enterable | Read the fixed-basis caption | Quote basis / asset scope / settlement frequency shown as fixed literals; base year matches **Base year (real EUR)**; modelled MW matches sidebar power and says it is inherited |
| 19 | Digest null matrix — user scenario | Set quote status `User scenario — no source document` | No uploader is offered; caption states the digest must be absent |
| 20 | Digest null matrix — executed document | Switch to `User-asserted executed source document` | Uploader appears; its caption states the file is sent to the server running the app, that only the digest is recorded, and offers external `shasum -a 256`. **It must not claim local-only hashing** |
| 21 | Inline code is readable | Look at the `shasum -a 256 <file>` span in that caption | Legible against its background. A grey-on-grey blob means the inline-code contrast rule regressed (see `test_ui_theme.py::test_global_theme_guards_inline_code_contrast`) |
| 22 | Digest fail-closed | Type a 64-hex digest in UPPERCASE | Error `source_document_sha256 must be a lowercase 64-character hex digest`; run button withheld. Lowercase it → preview returns |
| 23 | Out-of-life term | Set start year + tenor to exceed project life | Fails closed naming the ending year and the life — never a silent merchant-only run. Restore an in-life term before continuing |
| 24 | Post-run caption flip | With a valid contract restored, click **Run Project Case** | Lifecycle caption reads "Contract settlement IS included: cash is max(merchant, effective floor) per draw and project year, applied before lifecycle costs" |
| 25 | Settlement disclosure | Expand **Contract settlement disclosure** | Field table (basis, algorithm, quote status, source, as-of, digest, modelled MW, quote basis, scope, frequency), the per-year floor table, and the rank-interpolated P50 block stating the P50 path is neither an actual scenario nor a per-year median |
| 26 | Disclosure follows the result, not the widgets | After the run, change a contract term **without** re-running | Result and disclosure disappear behind "Project Case inputs changed. The stale result is hidden; run again." — you must never see a disclosure describing terms the result was not computed from. This item deliberately destroys the result |
| 27 | Cockpit mirror | Undo item 26's edit (or re-enter valid terms), **re-run**, then open **Project Case NPV — read-only Revenue-tab result** in the Simulation Cockpit | Mirrors both NPV blocks, repeats the flipped lifecycle caption, and adds a one-line "Contract settlement applied: …" caption plus the fingerprint. It is a caption, not a nested expander. Without the re-run there is no result to mirror — the mirror correctly shows nothing, which is not a pass |

## Checklist — Simulation Cockpit batch panels (Step 3B)

AppTest counts every solver call across reruns, but it cannot click a download
button. Run this after changes to the multi-day replay or forecast-policy
panels in `src/pages/simulation_cockpit.py`.

Setup: load a zone with several clean days (and IDA1 for items 30–31).

| # | Entry | Action | Expect |
|---|-------|--------|--------|
| 28 | Multi-day replay survives a download | **Run multi-day replay**, then click **Download multi-day replay (Excel)** | The workbook downloads; KPIs, charts and table stay on screen with no solve spinner |
| 29 | Multi-day stale state | Toggle **Continuous SoC across days**, then toggle it back | While changed: "Inputs changed since the last run…" warning, no charts, table or download. Toggled back: the same result returns without a spinner |
| 30 | Forecast policy survives a download | **Run forecast policy** (tick the stochastic option if you have time), then click its Excel download | Strategy table, attribution and download stay; no "Solving…" spinner |
| 31 | Forecast stale state | Change **Rebid deadband**, then click **Run forecast policy** | Stale warning and no download until Run; Run recomputes and shows the new result |

## Checklist — average-price basis and metric readability (Step 3C)

Use synthetic fixtures for reproducible price and layout checks. Keep the
fixture and its reproduction command with the dated review evidence; do not
overwrite a live market cache to create the fixture. Read actual workbook and
PDF outputs as well as the page. A numeric assertion or a font-width estimate
does not establish that an exported row is visibly readable.

For layout checks use actual viewport widths of **1280, 1440 and 390 CSS px**
at 100% browser zoom, beginning with the sidebar expanded. Record viewport,
sidebar state and page/panel with every screenshot. At mobile width Streamlit
may overlay the sidebar on the content: capture that state, then close the
overlay to inspect the main content and label those captures accordingly.
Do not report an obscured card as readable. AppTest remains the behavioral
check; browser screenshots and overflow inspection supply layout evidence.

| # | Entry | Action | Expect |
|---|-------|--------|--------|
| 32 | Shared overall Avg Price | Load 30 hourly days at EUR 10/MWh immediately before the registered SDAC cutover followed by 10 quarter-hour days at EUR 100/MWh; inspect Market Overview, Zone Comparison and the Excel/PDF summaries | Each overall average is EUR 32.50/MWh with 960 of 960 covered delivery hours. The chart's final trailing 720-hour mean is separately EUR 40/MWh. Available Excel average cells remain numeric |
| 33 | Unknown duration stays unavailable | Use a fixture with an internal missing delivery timestamp, a singleton, or a regular unsupported 2-hour cadence; inspect the four consumers and the zone-comparison workbook | Avg Price shows `n/a` and the shared reason; no guessed one-hour duration, raw `nan` text or substituted zero. An unavailable overall average alone must not hide other valid statistics |
| 34 | Finite coverage is explicit | Replace some prices on a verified grid with NaN or infinity, leaving timestamps intact, then repeat with every price non-finite | With finite prices remaining: duration-weighted average over only their hours, with covered/total hours and excluded coverage disclosed. With none: `n/a` and a no-finite-price reason. Missing values are never priced at zero |
| 35 | Market KPI row | At each width inspect Avg Price, the two ordered spreads and negative-price hours | Full amounts and EUR/MWh or hour units are readable without ellipses or overlap; cards wrap according to the main content width, with the coverage/reason caption still visible |
| 36 | Project Case quantiles | At each width inspect both NPV sections and the cockpit mirror, including long positive and negative amounts | Full economic section headings distinguish the two bases; cards visibly say P10 (Downside), P50 (Median), P90 (Upside) and P(NPV > 0). Full monetary values remain readable, rather than abbreviated or hidden behind a tooltip |
| 37 | Cockpit KPI rows | At each width inspect single-day KPI/health cards, multi-day replay, the frontier, forecast skill/policy, reserve gap, stochastic attribution/risk and floor/NPV metrics when their inputs make them available | Labels, amounts, quantiles and their units remain readable; custom grids and Streamlit metric rows wrap within their actual containers, including expanders. No value is lost to clipping or overlap |
| 38 | Layout reruns preserve stored results | With populated multi-day and forecast panels, resize the viewport, change theme, then repeat their download/stale-state checks in items 28–31 | Results still follow the stored run and its assumptions; presentation changes do not trigger a new solver run or expose a stale download |

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
widget layer (parsers, persistence, provenance, Data Trust tables, overlays,
Project Case gating and settlement) is covered by the mocked suite; see
`tests/`.

Items 15–21 and 25–27 exist because an assertion that a string is *present*
cannot tell you it is *readable* or that it describes the right thing. The
2026-08-17 run of this list found exactly that: the `shasum` fallback was
rendering at ~1.05:1 contrast while its AppTest assertion passed.
