# Replay input guards

## Step 1: reviewed behavior

Continuous DA and ordinary DA/IDA replay split clean runs when the daily market
cadence changes. Each segment has one interval duration. SoC carries between
segments, with terminal-neutral equality reapplied at each segment end, as for
the existing 576-interval size cap. No input is resampled.

`simulate_replay_batch` exposes integer `n_cadence_splits`. It counts actual
cadence-driven cuts between adjacent clean runs; gaps, invalid days and
size-only cuts do not increment it. The cockpit shows the split disclosure only
when this count is positive.

Ordinary single-day and continuous DA/IDA replay require an inner join to retain
both source days' row counts before dropping NaNs. The Revenue two-stage model
requires the intersection to retain at least 90% of each source day, preserving
its historical 23/24 hourly IDA tolerance. Rejected days retain existing
missing-data accounting and `invalid_input` failure status.

Eleven solver price-vector boundaries reject all non-finite prices through
their existing typed failure results. IDA, capacity, activation and imbalance
CSV importers drop non-finite required price/volume rows and log row counts.
Negative finite prices and valid infinite rebid limits remain supported.

## Regression evidence

The hourly-majority mixed-cadence fixture now reports EUR 948 / 24 MWh throughput
/ 12 FCE for its quarter-hour day, instead of EUR 3792 / 96 MWh / 48 FCE.
The opposite fixture's hourly day reports EUR 316 / 8 MWh, matching standalone
replay instead of the previous EUR 237 / 6 MWh. All selected days are retained.

The change adds 90 cases: 86 fail against the original implementation and four
are compatibility controls. Coverage includes both bias directions, physical
throughput bounds, grouping and disclosure, two-sided grid mismatches, sparse
IDA, solver infinities, all importers and IDA persistence/readback. Existing
tests were not removed or rewritten. Validation: 1675 passed / 2 existing
opt-in rendering skips; the separate slow suite passed 21/21. Independent
review reproduced the baseline failures and the full-suite result.

## Step 1b: remaining DA/IDA consumers

Reserve ceiling, sequential, sequential reserve, stochastic and stochastic
triple batch paths now share the ordinary continuous replay's two-sided day
join guard. The joined day must retain both original source row counts before
NaN removal. Existing local-time regular-day checks, forecast/scenario
alignment, reserve skip ordering and solver failure accounting remain intact.
A rejected market grid counts as missing data and does not reach the solvers.
The ordinary single-day error now leads with incomplete coverage and retains
the DA, IDA and joined row counts.

The simplified Revenue uplift estimate checks inferred cadence and timestamp
phase on every overlapping local day. It accepts verifiable sparse samples
and matching grids that change cadence between days, including DST days.
Different cadences, duplicate timestamps, off-grid observations or fewer than
two source observations on a compared day make the entire window unavailable.
No partial-window uplift is silently substituted and no source is resampled.
Without delivery-duration metadata, very sparse samples whose cadence cannot
be established are conservatively unavailable.

Only finite price pairs contribute to the estimate and histogram. The result
exposes `model_available` and `reason`, plus `da_coverage_pct` and
`ida_coverage_pct`: finite timestamp overlap divided by each original input's
row count. On usable grids, the lower ratio is the annual uplift adjustment.
On incompatible grids the estimate has zero usable periods and is unavailable;
raw overlap ratios may still be nonzero, so they are not proof of model
availability. The Revenue panel displays the reason instead of an uplift
headline, and shows both source coverages for usable samples.

Coverage remains an interval-count screening measure, not elapsed-time or
delivered-energy coverage. The 24 DA / 96 IDA fixture now reports raw DA/IDA
overlap of 100%/25% with an unavailable estimate, rather than a 100%-covered
annual uplift. Existing same-grid sparse IDA behavior is retained.

Step 1b adds 72 cases: 37 fail on the Step 1 baseline and 35 are compatibility
controls. They exercise all five public batch paths, both mismatch directions,
positive/zero reserve, two generated scenarios, complete hourly/quarter-hour
grids, missing/NaN rows, DST, both uplift denominators, non-finite pairs and
the real Revenue panel. Three new public-adapter rejection cases guard against
turning native pre-cutover mismatched grids into fingerprinted Project Case
results. An additional post-cutover spring DST case preserves the EUR 480
DA/IDA/reserve settlement assertion on compatible native grids.

Three existing integration cases previously expected successful dispatch on
hourly DA and quarter-hour IDA. Their assumptions are updated explicitly:
the two non-UTC success cases use post-cutover native quarter-hour data while
retaining walk-forward coverage and fingerprint assertions; the pre-cutover
spring DA/IDA/reserve case now expects `AdapterUnavailableError`. The existing
DA-only reserve EUR 480 checks on 23/24/25-hour days and the compatible autumn
DA/IDA/reserve EUR 480 check remain. Solver-heavy integration tests remain
part of the full-suite validation.

## Remaining scope after Step 1b

Step 2 covers interval-duration vectors for local-day-internal resolution
changes in BG, EE, FI, GR, LT, LV, PT and RO. The FI joint-capacity fixture still
reports EUR 110.4375 instead of EUR 114. The current regular-day guard remains
unchanged; neither step resolves that case. Forecast skill's DA-baseline
coverage disclosure remains a separate item: preserving the forecast's own
timestamps does not imply complete coverage of its DA comparator. Moving-window calculations,
dependency bounds, other interaction/visual changes and project-note
housekeeping retain their separate planned scope.
