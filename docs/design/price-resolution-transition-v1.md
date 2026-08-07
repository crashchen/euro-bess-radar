# Price-resolution transition contract v1

Status: **implemented; CC-approved; merged in PR #76**

Implementation date: 2026-08-07

## Purpose

Day-Ahead market data can change cadence inside one requested or cached
window. Treating the whole window at its finest observed cadence invents
synthetic prices before the cutover; treating it at its coarsest cadence drops
real sub-hourly prices afterwards. Either failure changes completeness,
dispatch opportunity, and annualised revenue without an obvious error.

This contract makes known market-time-unit transitions market-calendar aware,
segmented, and auditable across fetch cleaning, cache validation, and
data-quality reporting.

## Locked market metadata

The Single Day-Ahead Coupling market changed from a 60-minute to a 15-minute
market time unit for **delivery day 2025-10-01**. The first 15-minute contract
period is 00:00–00:15 in the shared CET/CEST SDAC market calendar. For this
delivery date, every participating zone therefore switches at the same
`2025-09-30T22:00:00Z` instant. The boundary is not each country's civil
midnight; ENTSO-E source timestamps for FI, PT, PL, GR, and RO independently
confirm the shared instant.

`src.config.SDAC_15MIN_ZONES` is the explicit coverage registry,
`SDAC_15MIN_DELIVERY_DATE` is its effective market delivery date, and
`SDAC_MARKET_TIMEZONE` defines the CET/CEST calendar used to resolve it. The
current supported set contains 38 SDAC zones. Ireland remains on a 30-minute
grid; Switzerland and Great Britain are outside this rollout and do not
inherit the cutover.

Primary references:

- [EPEX SPOT — 15-minute products in Market Coupling](https://www.epexspot.com/en/new-15-minute-products-market-coupling)
- [Nord Pool — SDAC go-live market-data reminder](https://www.nordpoolgroup.com/en/trading/Operational-Message-List/2025/09/market-data---reminder-for-sdac-15-minute-go-live-20250905084800/)
- [ENTSO-E — Single Day-Ahead Coupling](https://www.entsoe.eu/network_codes/cacm/implementation/sdac/)

Adding a supported ENTSO-E zone must make an explicit registry decision. A
test compares the configured ENTSO-E coverage with the transition registry so
new zones cannot silently acquire or miss the cutover.

## Segmentation invariants

For a requested UTC `[start, end)` window that crosses a known transition:

1. split at the shared SDAC market-calendar transition instant;
2. build the pre-transition expected index at 60 minutes;
3. build the post-transition expected index at 15 minutes;
4. concatenate the segments without duplicate boundary rows;
5. preserve every source timestamp that lies on the declared segment grid and
   surface absent expected timestamps as `filled=True` rows;
6. never upsample the pre-transition segment to 15 minutes and never downsample
   the post-transition segment to 60 minutes.

The same segmentation applies when an explicit expected window is supplied,
when cleaning a source-only mixed index, and when checking SQLite cache
completeness. A row missing immediately before or exactly at the transition
must be surfaced on the correct side rather than hidden by an observed
minimum/maximum anchor.

Windows wholly before or after the transition keep their native expected
cadence. Unknown or excluded zones retain the existing inference/fallback
behavior and do not inherit SDAC metadata.

## Gap and imputation semantics

`MAX_SHORT_GAP_HOURS` is a physical-duration threshold, not a count of rows at
the dominant cadence. Missing runs are measured on the actual timestamp axis,
including the final missing slot at the cadence applicable to that segment.

Consequences:

- three missing hourly slots are a three-hour gap and remain unresolved when
  the limit is two hours;
- three missing quarter-hours are a 45-minute gap and may be interpolated when
  internal;
- edge gaps are never interpolated;
- `max_source_gap_hours` reports the same physical-duration interpretation.

Interpolation may operate on a combined mixed-resolution frame only after the
expected grid is correctly segmented. It must not create timestamps across a
resolution boundary.

## Cache contract

Cache freshness and cache completeness are separate. A fresh cache slice is
accepted only if each transition segment contains its own complete expected
grid. A complete hourly pre-cutover segment plus a gappy quarter-hourly
post-cutover segment is incomplete, and vice versa.

The transition instant is derived from the SDAC CET/CEST delivery calendar
before cache segmentation. WET/WEST and EET/EEST zones therefore use the same
UTC cutover as CET/CEST zones even though the instant maps to a different civil
clock time locally.

## Required regression coverage

- PL and DE_LU crossing windows preserve the exact source grid with and
  without an explicit expected window;
- missing rows immediately before and at the boundary surface correctly;
- pre-only, post-only, and partial crossing windows use the correct cadence;
- cache validation accepts complete mixed grids and rejects gaps on either
  side;
- CET/CEST, WET/WEST, and EET/EEST examples are pinned to the same 22:00 UTC
  market boundary and exact source grid;
- PL spring/fall DST days retain 23 hourly and 100 quarter-hourly intervals;
- Ireland, Switzerland, Great Britain, and unknown zones have no SDAC cutover;
- physical-duration tests distinguish a multi-hour old-grid gap from a short
  quarter-hourly gap.

## Validation snapshot

- `tests/test_ingestion.py`: 188 passed;
- fast suite: 1227 passed, 2 skipped, 13 slow tests deselected;
- full suite: 1240 passed, 2 skipped (1242 collected);
- ruff and `git diff --check`: clean.

## Scope boundary

This contract covers Day-Ahead ingestion, cache completeness, and price-data
quality. It does not alter source prices, infer unregistered historic market
changes, resample downstream IDA or reserve streams, or define ProjectCase.
