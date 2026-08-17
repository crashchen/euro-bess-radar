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
