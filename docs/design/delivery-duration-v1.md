# Native DA delivery durations

This contract extends the [Step 1 replay guards](replay-input-guards.md).
It covers physical duration within a DA day and duration-based price analytics;
it does not introduce a new market calendar or allow mismatched DA/IDA products
to be joined.

## Accepted inputs

`interval_hours_vector` accepts a positive finite scalar or a one-dimensional
vector with exactly one duration per price. Scalars retain their existing
broadcast behavior. Zero, negative, non-finite, multidimensional and
wrong-length inputs fail before optimization, through the solvers' typed
`invalid_input` result. A one-element array is not a scalar broadcast.

`infer_delivery_interval_hours` retains uniform timestamp inference and
recognizes one nonuniform pattern: consecutive hourly delivery starts before
the configured SDAC cutover, followed by quarter-hour starts afterwards.
It validates every observed delta and its phase. The cutover is the existing
`SDAC_15MIN_DELIVERY_DATE` at midnight in `SDAC_MARKET_TIMEZONE`.
An unrecognized change or missing interval is not stretched into a product.
Uniform partial samples retain historical inference; full-day replay also
requires local midnight boundaries, all rows on that local date, and the last
delivery ending at the next local midnight.

The existing Project Case registry supplies the regression fixtures for BG,
EE, FI, GR, LT, LV, PT and RO. FI and the other eastern zones have 93 products
on 2025-10-01 (one hour plus 92 quarters); PT has 27 on 2025-09-30 (23 hours
plus four quarters). Both represent 24 physical hours. Registry profiles and
the Project Case cash specification version are unchanged.

## Physical and economic propagation

`solve_daily_lp` and `solve_daily_joint_capacity_lp` accept the duration
vector. The same duration multiplies objective cash, charge/discharge energy,
SoC increments, terminal balance and the discharge-energy cycle limit.
The canonical second pass minimizes physical discharged energy; its weights
are normalized by their maximum, preserving uniform-grid tie behavior.
Capacity cash and average reserve power are duration weighted.

DA screening batches, single-day replay, continuous DA replay, the cycle
frontier and the existing Project Case DA adapter use this path. Replay
timeseries now include `interval_hours`; reconstructed daily summaries, VWAP,
FEC, average C-rate and event energy/duration use the corresponding slice.
Trade events end at the end of their last actual delivery interval.

Step 1's run partition remains: a change in the *dominant daily cadence*, a
gap, an invalid day or the 576-interval cap can end a segment. Terminal-neutral
equality is applied at each segment end and SoC seeds the following segment.
A valid mixed day can contain a duration vector inside such a segment.
`n_cadence_splits` counts actual cuts between days, not an internal-day change.
This preserves the reviewed horizon policy while correcting interval pricing.

Ordinary, sequential, reserve and stochastic DA/IDA paths retain their strict
grid guards and scalar public contracts. Supporting DA vectors does not enable
mixed products in those strategies. FI Project Case reserve remains
unsupported by its separate reserve-profile contract: fixing FI screening
capacity cash to EUR 114 does not register a new reserve settlement profile.

## Analytics and presentation

### Overall average price

`calculate_average_price` supplies the one window-wide **Avg Price** used by
Market Overview, Zone Comparison (including its workbook), and the Excel/PDF
report summaries. It validates the delivery grid over the entire input window,
then calculates `sum(finite_price * interval_hours) / sum(finite_interval_hours)`.
The accepted native durations are 1 hour, 30 minutes and 15 minutes, including
the registered SDAC hourly-to-quarter-hour transition. A regular 2-hour or daily
sample is not evidence of a supported delivery product. This restriction is
local to this average; it does not narrow the lower-level inference helper's
historical uniform-cadence behavior or change a solver's input contract.

The input must have a real `DatetimeIndex`; a numeric row index is never
silently converted into delivery timestamps. Empty inputs, a lone timestamp,
duplicates, unsorted or unset timestamps, an internal gap, an unknown cadence
change, or no finite price coverage make the average unavailable. Every
consumer shows `n/a` with the shared `avg_price_reason`. Zone Comparison keeps
its table's average-price column numeric for numerical sorting; an unavailable
entry is a missing numeric cell, with the zone's explicit `n/a` and reason in
a caption below the table. Its workbook uses literal `n/a` beside the reason.
There is no one-hour fallback, reordering, de-duplication, resampling or gap filling.

NaN and infinite prices on an otherwise verified grid contribute neither price
value nor covered hours. `average_price_basis` discloses finite covered hours
out of all verified delivery hours represented by the observations, including
the last observed product's duration. That denominator describes this input
window; it is not a claim that the requested market history is complete. A
verified grid with no finite prices has zero covered hours and an unavailable
average. An unverifiable grid has unavailable durations as well. Available
Excel averages remain numeric cells with their existing precision.

For 30 hourly days priced at EUR 10/MWh followed at the registered cutover by
10 quarter-hour days at EUR 100/MWh, the overall average is **EUR 32.50/MWh**;
the old equal-row average was EUR 61.43/MWh. The final trailing 720-hour average
is separately EUR 40/MWh, since it covers only the last 30 physical days.
Uniform 1-hour, 30-minute and 15-minute windows retain their previous averages.
The adjacent **Std Dev (row-based)** in Zone Comparison and its workbook, and
**Median Price (row-based, EUR/MWh)** in the Excel/PDF summaries, retain their
existing equal-row statistics: sample standard deviation and median,
respectively. They are not weighted by delivery duration, so a quarter-hour
price row has the same weight as an hourly row. Their labels distinguish this
basis from Avg Price; this step does not change either calculation.
Renewable-conditioned averages, heatmap groups and forward-contract averages
are separate statistics and are not changed by this contract.

### Other duration-based analytics

On a mixed DA day, the ordered spread calculation compares duration-weighted
buy-before-sell windows containing whole native products and exactly the
requested physical duration. It does not split an hourly order into four
quarter-hour orders. A requested duration with no complete native window is
excluded. Uniform-grid ordered spread behavior is retained. The MILP can
still choose fractional power within a native product.

Negative-price hours sum delivery durations. Unknown irregular grids retain
the observed negative-interval count but report hours as unavailable.
A record whose delivery instant is unset (NaT) is recognised before the
local-day grouping, because grouping silently drops such a key and the record
would otherwise contribute a confident zero to the total.
`calculate_negative_price_hours` carries `negative_hours_reason`, naming either
the first local date whose grid could not be verified or the number of unset
delivery timestamps. `negative_price_hours_reason` is the one shared answer to
whether the figure is available and why it is not; the page, the Excel summary
and the PDF summary all branch on it and show `n/a` plus that reason, never a
raw NaN, a literal `nan` string or a substituted zero. Each surface keeps its
own formatting of an available value, so the Excel cell stays numeric. The
observed interval count, its share of intervals and the average/most negative
prices remain real numbers, because they are counts rather than durations.

`describe_price_index_issue` is the single definition of a usable price index
shared by the trailing mean and the pages that explain its absence.
`time_weighted_rolling_price_mean` still rejects a defective index. The market
page classifies the defect before calling it rather than catching its error,
so an unrelated programming error is never presented as unavailable data.
Repeated delivery timestamps suppress only the trailing mean and are disclosed;
an unsorted index or an unset (NaT) timestamp additionally suppresses the price
chart, because drawing those rows in their given order would misrepresent the
market, and the corresponding PDF export figure is dropped in the same branch.
Rows are never reordered, de-duplicated or gap-filled to make a chart drawable.

The market price chart's 30-Day MA uses the trailing 720 physical hours through
each delivery interval's end. Prices are weighted by the covered portion of
their delivery interval, including clipping an old interval at the window's
left edge. It appears after 24 hours of finite price coverage. NaN/non-finite
prices and days with unverified cadence contribute no covered duration;
missing prices are not imputed. The window is physical time, including DST,
and is no longer a fixed 720-row window. The plotted x coordinate remains the
delivery start; the caption discloses the end-of-interval calculation.

Forecast skill first scores the finite forecast/realised population, then
reports a separate finite DA comparator count and coverage ratio. Skill vs DA
compares both errors on that same subset, with its forecast MAE disclosed.
Duplicate DA timestamps make only the DA comparator unavailable; they cannot
multiply the main forecast population. Timestamp overlap is a price-space
diagnostic and is explicitly not proof of matching delivery products.

### Metric layout

Market Overview, Project Case and the cockpit's KPI rows use
`ui_theme.metric_columns`, a horizontal Streamlit container whose cards wrap
according to their actual available width, including beside the sidebar or
inside an expander. The cockpit's custom KPI and health grids likewise choose
their column count from a minimum card width rather than a viewport breakpoint.
These presentation changes do not alter the values or units being reported.

Project Case retains each full economic section heading and uses the short
card labels `P10 (Downside)`, `P50 (Median)`, `P90 (Upside)` and `P(NPV > 0)`.
Readers must be able to distinguish the quantiles and read the full monetary
amount without relying on a tooltip. Browser acceptance covers 1280, 1440 and
390 CSS-pixel widths and records the sidebar state; AppTest string assertions
alone do not prove that a label, amount or unit is readable. See the
[manual smoke checklist](../runbooks/manual-ui-smoke.md).

## Reserve average power

`solve_daily_joint_capacity_lp` reports `avg_reserve_mw` as a duration-weighted
average, matching its duration-weighted capacity cash. The sequential reserve
and stochastic triple batches report their `avg_reserve_mw`,
`myopic_avg_reserve_mw` and `stochastic_avg_reserve_mw` as unweighted means.
That is not a present-day error: those paths keep a scalar `interval_hours`
public contract, so every interval carries the same duration and the two
averages coincide. It is recorded here because the equality is a premise,
not an invariant of the expression.

Extending any of those paths to a duration vector must first make these
reporting means duration weighted and add a nonuniform known-answer case;
a mixed-duration day would otherwise let dense intervals dominate a reported
average power. This step does not generalise the sequential or stochastic
`dt`, and does not change any settled cash.

## Runtime and validation

Both dependency manifests require Streamlit >=1.55,<2.0. Version 1.55.0 is a
tested UI floor, not a claim that every used API first appeared in that release;
see the versioned [button](https://docs.streamlit.io/1.55.0/develop/api-reference/widgets/st.button)
and [dataframe](https://docs.streamlit.io/1.55.0/develop/api-reference/data/st.dataframe)
documentation. CI runs the full suite on Python 3.13 and a separate real-panel
smoke job on Python 3.11 with Streamlit pinned to 1.55.0. Pull requests to any
base branch trigger CI, including stacked review branches.

The [Step 2 handoff](../audits/2026-09-08-step2-handoff.md) freezes the change,
baseline failures and validation results. German DST reserve cash still
differs between Project Case's wall-clock settlement and the cockpit's
physical-hour screening convention. Step 3B result persistence is merged in
[#90](https://github.com/crashchen/euro-bess-radar/pull/90) as `cf91374`; the
Step 3C average-price and metric-layout changes are under review. Clarifying
the DST settlement distinction remains Step 3D, and full documentation/Vault
housekeeping remains Step 4. See the [audit index](../audits/README.md) for
revision-specific evidence rather than treating this contract as a test log.
