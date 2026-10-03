# Current verification snapshot

Verified **2026-10-03** against main `68d3486`. That head is the ordinary
merge of the three manual-acceptance follow-ups. Each merge's second parent is
the approved PR head, and its first-parent diff equals that PR's patch. The
previous merged-state snapshot is archived
[verbatim](2026-09-28-forward-chart-followup-merged.md); this snapshot is a
documentation update and does not re-run the full suite.

| PR | Change | Approved head | Merge |
|---|---|---|---|
| [#104](https://github.com/crashchen/euro-bess-radar/pull/104) | Project Case keeps every input widget when another section has an error; collected errors block the run | `a74b59e` | `3bbe9f0` |
| [#106](https://github.com/crashchen/euro-bess-radar/pull/106) | An activation-volume response with fewer than two timestamps names the ~1-month publication lag | `f892e77` | `3ba2f40` |
| [#105](https://github.com/crashchen/euro-bess-radar/pull/105) | Readable `help=` buttons and main uploaders in both base themes; darker main controls (≥5.06:1 normal, ≥6.17:1 hover); muted disabled Browse; Data Trust cards wrap | `7e97d42` | `68d3486` |

CC authored #104 and #106; Codex's independent review passed both. CC
authored #105's first increment `d5bf996`. Codex's review requested a
contrast change, which Codex implemented and reviewed as `a32e707`
([handoff](../audits/2026-10-03-control-contrast-r2-handoff.md)). All three
were merged on the user's explicit authorization.

[Manual acceptance record](../audits/2026-10-02-manual-acceptance-evidence/README.md) ·
[#105 revision evidence](../audits/2026-10-03-control-contrast-r2-evidence/README.md) ·
[Remaining work](follow-ups.md).

## Checks and scope

| Check | Result |
|---|---|
| PR-head CI | `test` and `Python 3.11 / Streamlit 1.55.0 compatibility check` succeeded at each approved head (runs 37123714659, 37137170672, 37141644468). |
| Main push CI | Succeeded for each merge commit: `3bbe9f0` (37142094921), `3ba2f40` (37142158888), `68d3486` (37142487313). |
| Full local suite | Codex's review stacked the three first-round patches (#105 at `d5bf996`) on a clean base: **2134 passed / 2 skipped / 29 warnings**, including slow tests. #105 revision code commit `a32e707` alone: **2127 passed / 2 skipped / 29 warnings**, including 35 slow, 296.93s (its handoff). Neither is a re-run of the final merge tree; main push CI is the merged-head check. |
| Browser acceptance | Agent-driven, not a human operator pass. All 52 checklist items executed: on `e28ee9c` in Chrome on 2026-10-02, and on `68d3486` in headless Chromium on 2026-10-03. 51 met their expectation within the stated scope. Item 37 has one open layout finding (F7): at 390 px the multi-day replay "Avg Annualized" value is ellipsized. The F1 fix was re-checked live on `68d3486`. Resizing populated panels sent no websocket frames (no rerun or solve). |
| Radar → ESS | Items 44–47 against ESS `9bad91c`: the applied stream reconciles with Radar settled revenue × CPI. ESS consumer defects F5 (DA arbitrage double count via the grid-tariff fallback) and F6 (no asset-size check) belong to ESS. |
| Real cache | `data/cache` 26/26 file hashes unchanged across the 2026-10-02 run, the #105 revision runs and the 2026-10-03 closeout. |

The acceptance browser checks use the stated synthetic fixtures, isolated
cache copies and live provider responses at run time. They do not certify
later revisions, every parameter combination, arbitrary currency magnitudes
or a desktop Chrome save dialog. Downloads were saved through Playwright
download events.

## Repeat

```sh
.venv/bin/python -m pytest tests/ -m "not slow" -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff 68d3486^1 68d3486 --stat    # #105 patch; repeat for 3ba2f40 and 3bbe9f0
```

The acceptance record lists the browser driver, fixture and server commands.
