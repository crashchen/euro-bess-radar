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

## Remaining scope after Step 1

Step 1b must extend the grid guard to reserve ceiling, sequential, sequential
reserve, stochastic and stochastic triple batch paths, and correct the
simplified Revenue uplift diagnostic's DA-only coverage denominator. These
consumers can still accept hourly DA with quarter-hour IDA after a lossy join.

Step 2 covers interval-duration vectors for local-day-internal resolution
changes in BG, EE, FI, GR, LT, LV, PT and RO. The FI joint-capacity fixture still
reports EUR 110.4375 instead of EUR 114. The current regular-day guard remains
unchanged; Step 1 does not resolve that case. Moving-window calculations,
dependency bounds, other interaction/visual changes and project-note
housekeeping retain their separate planned scope.
