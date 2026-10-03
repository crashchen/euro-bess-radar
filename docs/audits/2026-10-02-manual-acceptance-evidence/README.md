# Manual UI acceptance — 2026-10-02 run and 2026-10-03 closeout

All 52 items of the [manual UI smoke checklist](../../runbooks/manual-ui-smoke.md)
were executed across two code heads:

- **2026-10-02:** a clean `git archive e28ee9c` (main after #103), in Chrome
  through Claude-in-Chrome. The [original log](LOG-2026-10-02.md) is kept
  verbatim apart from trailing whitespace.
- **2026-10-03:** the items that run left open or failed were completed on a
  clean `git archive 68d3486`. That is main after the ordinary merges of #104
  (`3bbe9f0`), #106 (`3ba2f40`) and #105 (`68d3486`, reviewed head
  `7e97d42`). Main push CI for all three merges succeeded.

Both days were agent-driven browser runs, not a human operator pass. The
10-03 items ran in headless Chromium, not desktop Chrome.

Result: **51 items met their expectation within the scope stated below.
Item 37 has one open layout finding (F7).** The Radar→ESS walkthrough found
two consumer defects in ESS (F5, F6); they belong to ESS. This record certifies
only the named heads, fixtures, widths and sources. It does not cover later
revisions, every parameter combination or every market value.

## Environment

| | 2026-10-02 | 2026-10-03 |
|---|---|---|
| Radar code | `e28ee9c` archive | `68d3486` archive (two isolated trees) |
| Runtime | Python 3.13.9, Streamlit 1.55.0, pandas 2.3.3 (repo `.venv`) | same |
| Browser | Chrome via Claude-in-Chrome. Viewport fixed at 960 CSS px (DPR 2), Light base theme | Playwright 1.61.1, headless Chromium build 1228. Exact viewports 1440/1280/960/390 × 900 at DPR 1. Light and dark base themes via `--theme.base` |
| Cache | Copy of `data/cache` | Width tree: copy of the 10-02 acceptance cache. DST tree: empty cache seeded only by `fixtures/seed_dst_fixture.py` |
| ESS | Archive `9bad91c` (Python 3.11, Streamlit 1.57) | not re-run |

The working repository's `data/cache` was fingerprinted before and after each
day: 26 files, unchanged (`logs/real-cache-*.sha256`). Server logs are kept with
local paths replaced by placeholders.

## Results

PASS means the checklist expectation was observed within the stated scope.

| # | Head | Result | Evidence and scope |
|---|---|---|---|
| 1 | e28ee9c | PASS (served bytes) | All 5 templates were served as attachments with the documented headers. The Chrome save dialog could not be operated on 10-02. A browser save-to-disk path was exercised on 10-03 through other download buttons (items 42/43) |
| 2 | e28ee9c | PASS (wording note) | The summary reports a total, not per-stream counts. Data Trust shows Manual CSV 26/25/25 |
| 3–4 | e28ee9c | PASS | 4 is a live fetch: 3019 rows; 91 unpriced intervals dropped, max 74 MW |
| 5 | e28ee9c → 68d3486 | FAIL → PASS | F1 at e28ee9c. Live recheck on 68d3486, window 2026-09-30..10-03: "Aktivierte aFRR needs at least two timestamps; got 0. The quality-assured series lags roughly one month…" (`outputs/2026-10-03/check05`) |
| 6–12 | e28ee9c | PASS | 8 and 12 are live fetches. 6 used a separate keyless instance (`logs/2026-10-02-streamlit-8613-nokey.log`) |
| 13–22, 24–25 | e28ee9c | PASS | Project Case contract entry and disclosure. 15–21 readability findings F2/F3 are fixed in #105 |
| 23 | e28ee9c | PASS + F4 | Fail-closed text was correct, but restoring the term reset downstream widgets. Fixed in #104 |
| 26–31 | e28ee9c | PASS | Stale/restore behavior. 30 confirms the stochastic delta row is excluded from the chart and kept in the table and xlsx |
| 32–34 | e28ee9c | PASS | Synthetic harness. Notes N3/N4 below |
| 35 | 68d3486 | PASS | Market KPI row: 4 cards, no clipping at 1440/1280/960/390 (light) |
| 36 | 68d3486 | PASS (scoped) | Screening NPV quantiles with 8-digit negative amounts (€-11,411,577 …) fit in the Revenue result and the Cockpit mirror at every width. With the default "Unknown" capacity-maintenance basis, the lifecycle section is unavailable. Large positive amounts were checked on Revenue/Cockpit cards, not in the PC quantiles |
| 37 | 68d3486 | **FAIL (F7)** | 49 Cockpit metric cards and 31 custom KPI/health items. All fit at 1440/1280/960. At 390 the multi-day replay card "Avg Annualized" shows `EUR 131,976/MW…` (`browser/2026-10-03/matrix/light-390/simulation-cockpit-01.png`) |
| 38 | e28ee9c + 68d3486 | PASS | Theme change on 10-02: no solve; xlsx cell-identical. Resize on 10-03 (same page, 1440→1280→960→390): **0 websocket frames sent**, so no rerun or solve. No stale warning; downloads retained (Revenue 1, Forward 4, Cockpit 4) |
| 39 | e28ee9c | PASS | Recorded DE_LU/FCR nominal-block basis in result, mirror and Excel |
| 40 | e28ee9c | PASS (UI part) | The "no global assumptions supplied" branch is not reachable from the UI |
| 41 | e28ee9c | PASS (scoped) | A zone/product change was not separately exercised |
| 42 | 68d3486 | PASS (synthetic) | The live DE_LU UI has no energy-only product (no ESIOS path), so the Step 3D page factory was used: FCR + aFRR Up capacity, mFRR Up energy-only, real joint solver. Page, downloaded XLSX `Summary!A23` and PDF all read "aggregate capacity: FCR, aFRR Up". mFRR appears only as its standalone "mFRR Up Revenue" row |
| 43 | 68d3486 | PASS | Real app on a seeded isolated cache: zero DA price, 1 MW / 1 h, FCR EUR 20/MW/h, availability 0.95. Joint MILP screening capacity is €166,554 / €159,614 / €173,494 per yr, i.e. 456 / 437 / 475 × 365.25 for 2026-03-28 / 03-29 / 2025-10-26. Project Case year-1 settled revenue in the downloaded handoff JSON is 166,440 = 456 × 365 on all three days. Captions fit at 1280 and 390 |
| 44–45 | e28ee9c | PASS | Handoff schema/digest/fingerprint; a stale result cannot export |
| 46 | e28ee9c + ESS 9bad91c | PASS | Preview/apply reconciles; CPI layer required before calculation |
| 47 | e28ee9c + ESS 9bad91c | PASS for the stream; F5, F6 | Applied stream equals Radar settled revenue × CPI. ESS defects below |
| 48 | 68d3486 | PASS (scoped) | 37 Revenue cards with every expander open: headline/DA/ancillary, joint MILP, IDA uplift, CapEx/degradation, revenue and NPV distribution, Project Case. No clipping at any width. The multi-cycle MILP dispatch toggle was left off |
| 49 | 68d3486 | PASS | Forward benchmark cards (2-year overlap), light and dark, 4 widths |
| 50 | 68d3486 | PASS (scoped) | Renewable and Data Trust cards fit at all widths. The 10-02 Data Trust truncation at 960 px is gone (fixed in #105). The DE_LU window had no quality gaps, so gap values were not exercised |
| 51 | 68d3486 | PASS | Benchmark chart in light and dark at 1440/1280/390: whole years, both endpoints labelled. 20-year range ticks are 2027 … 2046 at every width |
| 52 | e28ee9c | PASS | Multi-day, forecast-policy and frontier Assumptions sheets |

## Findings and disposition

| ID | Where | Finding | Disposition |
|---|---|---|---|
| F1 | Radar | Unpublished activation window failed without naming the publication lag | Fixed in #106; live recheck PASS |
| F2/F3 | Radar | `help=` buttons and main-canvas uploaders unreadable in the Light base theme | Fixed in #105, including the reviewed main-control contrast revision ([evidence](../2026-10-03-control-contrast-r2-evidence/README.md)) |
| F4 | Radar | A Project Case input error reset later widgets | Fixed in #104 |
| Layout | Radar | Data Trust cards clipped at 960 px with the sidebar open | Fixed in #105; 10-03 matrix confirms |
| F5 | ESS | Applying a Radar handoff leaves the grid-tariff energy-margin fallback active, so DA arbitrage is counted twice unless "Avg. Arbitrage Spread" is zeroed (IRR 15.78% vs −9.31%) | ESS change, implemented separately; not part of this repository |
| F6 | ESS | No check of handoff modelled MW/MWh against ESS asset size | ESS change, as above |
| F7 | Radar | At 390 px the multi-day replay "Avg Annualized" value is ellipsized | **Open**, small layout fix |
| N1 | Radar | Deadband `number_input` above its max displays the typed value while the run uses the previous value (Streamlit native behavior) | Open; re-review |
| N2 | Radar | With sparse FCR rows, the scalar-mean "DA + FCR co-opt (headroom)" row exceeds the per-interval "co-opt ceiling" | Documented basis difference; re-review |
| N3 | Radar | All-non-finite input: PDF row-based median prints `inf` (xlsx cell empty) | Open; next Radar fix |
| N4 | Radar | No complete local day: spread cards show €0.00/MWh and negative hours 0.0 h | Open; next Radar fix |

The first DST fixture draft built block labels by adding elapsed hours to an
aware midnight, which shifted DST-day labels by one hour. Project Case correctly
refused with "no DA-complete reserve-covered dates" rather than settling. The
archived seed script builds wall-clock labels and records this in a comment.

## Layout

- **Directories.** `drivers/` holds the Playwright drivers; `fixtures/` holds
  all synthetic inputs and harnesses; `outputs/` holds the saved downloads and
  JSON measurements; `browser/` holds the screenshots; `logs/` holds the server
  logs and cache fingerprints.
- **Metric probe.** `outputs/2026-10-03/matrix/light.json` records, per width
  and tab: sidebar state, metric/KPI counts, clipped nodes (`scrollWidth >
  clientWidth + 1` or right edge beyond the viewport), stale-warning count,
  visible downloads and websocket frames sent during the resize.
- **Desktop sidebar.** At 1440/1280/960 the sidebar is expanded.
- **Mobile sidebar.** At 390 the sidebar overlay is closed before inspecting
  the main content.

## Repeat

```sh
git archive 68d3486 | tar -x -C <tree>    # plus a COPY of a cache and .env
cd <dst-tree> && PYTHONPATH=. python <repo>/docs/audits/2026-10-02-manual-acceptance-evidence/fixtures/seed_dst_fixture.py <outdir>
streamlit run app.py --server.address 127.0.0.1 --server.port <port> --theme.base light
NODE_PATH=<dir with playwright> node drivers/drive43.js <port>
NODE_PATH=<dir with playwright> node drivers/width_matrix.js light <port>
NODE_PATH=<dir with playwright> node drivers/width_matrix.js dark <port> forward-only
PYTHONPATH=. streamlit run fixtures/revenue_energy_only_harness.py --server.port <port> && node drivers/drive42.js <port>
```

The drivers write next to themselves. Live fetches (items 4/5/8/12, Fetch
Data and Regelleistung) depend on provider publication at run time. Downloads
were saved through Playwright download events. A desktop Chrome save dialog was
not operated.
