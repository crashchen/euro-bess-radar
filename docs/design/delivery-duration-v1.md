# Native DA delivery durations — Step 2

This contract extends the [Step 1 replay guards](replay-input-guards.md).
It covers physical duration within a DA day; it does not introduce a new
market calendar or allow mismatched DA/IDA products to be joined.

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

On a mixed DA day, the ordered spread calculation compares duration-weighted
buy-before-sell windows containing whole native products and exactly the
requested physical duration. It does not split an hourly order into four
quarter-hour orders. A requested duration with no complete native window is
excluded. Uniform-grid ordered spread behavior is retained. The MILP can
still choose fractional power within a native product.

Negative-price hours sum delivery durations. Unknown irregular grids retain
the observed negative-interval count but report hours as unavailable.

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
physical-hour screening convention. Clarifying that distinction, result
persistence, other visual work and Vault housekeeping remain subsequent steps.
