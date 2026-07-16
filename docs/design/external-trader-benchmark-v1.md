# External trader revenue benchmark v1

Status: **LANDED** (PR #71, squash `d9b0b18`, plus reviewer hardening
`3f03f5a` — unconditional on-screen platform-basis/capture-haircut caption).
This document describes the shipped implementation; it is not a contract,
floor quote, or market forecast.

## 1. Problem and evidence

An external trader supplied an indicative annual revenue curve for several
standalone Sicily BESS configurations (duration and daily-cycle cases). The
curve is useful as a market calibration artifact, but it has no term sheet and
does not disclose enough methodology to be treated as a bankable forecast.

The Enspired material contributes a second, different lesson: a production
optimizer separates forecast, day-ahead allocation, and final dispatch, and can
allocate one physical asset across wholesale, FCR, and aFRR. It also illustrates
co-location with renewable generation and a PPA opportunity cost. Those are
architectural references, not numbers that can be copied into this platform.

The v1 increment therefore adds a transparent reconciliation surface. It does
not reverse-engineer an unknown trader model or pretend that unlike revenue
stacks are comparable.

## 2. Input contract

The user uploads an annual CSV with one row per `(zone, scenario, year)`:

| Column | Required | Contract |
|---|---:|---|
| `zone` | yes | Supported bidding-zone code. |
| `scenario` | yes | Non-blank user label. |
| `year` | yes | Integer calendar year in `[2000, 2200]`. |
| `revenue_eur_per_mw_yr` | yes | Signed, finite annual value normalised by installed MW. |
| `asset_type` | yes | `standalone`, `co-located`, or `unknown`. |
| `market_scope` | yes | `da-only`, `wholesale`, `all-in`, or `unknown`. |
| `revenue_basis` | yes | `gross`, `net-of-wear`, `net-of-fees`, or `unknown`. |
| `duration_hours` | no | Positive finite duration. |
| `max_efc_per_day` | no | Positive finite cycle assumption/cap. |
| `source` | no | User-supplied provenance. |
| `as_of` | no | Parseable source date. |

Basis metadata must be constant within a scenario. Duplicate years are rejected.
Revenue stays signed; a negative external downside case is valid information.
No external curve is bundled with the repository or enabled by default.

## 3. Platform comparison basis

The model curve reuses `analytics.calculate_yearly_revenue_breakdown()` over the
Forward Scenarios daily dispatch output:

- forward-synthetic **DA-only** prices;
- the selected installed MW, duration, and round-trip efficiency;
- the sidebar capture haircut;
- gross dispatch revenue after that capture haircut, before wear, optimizer or
  trader fees, tax, financing, and other revenue streams;
- the existing `365.25` annualisation convention and calendar-year coverage
  diagnostics.

The benchmark joins to this curve only on equal `zone` and calendar `year`.
Benchmark-only years remain visible, but do not enter overlapping-year metrics.
Partial model years carry their coverage and partial-year flags into the table.

## 4. Reconciliation semantics

For overlapping years with positive model revenue:

```text
gap = model_revenue - benchmark_revenue
benchmark_to_model_ratio = benchmark_revenue / model_revenue
```

The ratio is a descriptive statistic, **not** an inferred capture rate. It can
also reflect different market scope, asset type, foresight, cycle limit,
liquidity, fees, risk margin, or forward-price view. No ratio is produced when
model revenue is non-positive.

The panel always warns when metadata is not a standalone, DA-only, gross case,
when duration differs, or when the quote declares a cycle cap that the current
Forward Scenarios path does not impose. Matching metadata removes those specific
warnings; it never upgrades the curve to a bankable forecast.

Locked hard caption:

> External benchmark reconciliation only: the uploaded annual revenue curve is
> user-supplied and is never used as a price curve, contracted floor, capture
> rate, solver input, or bankable forecast.

## 5. UI and export

The optional expander lives in Forward Scenarios. It can show the uploaded curve
as provenance-only before a forward curve exists; once the platform annual curve
has been calculated it adds the overlap comparison. It contains:

- template download and CSV upload;
- zone/scenario selectors;
- benchmark average, endpoint CAGR, overlap model average, and ratio;
- a two-line annual comparison chart and an audit table;
- explicit comparability warnings; and
- a self-contained Excel workbook containing the source rows, reconciliation,
  platform annual curve, and assumptions/provenance.

The sidebar also exposes a `6.7h` duration preset so a disclosed `6.7h` external
case can be selected without relabelling it as `6h` or `8h`. This is a case-
selection convenience only; it does not import or validate the external curve.

The Excel ratio remains a fraction; only the Streamlit table converts it to
percentage points for display.

## 6. Red lines

The external curve must never:

1. become a price curve, capture-rate calibration, contracted floor, solver
   constraint, or automatic revenue adjustment;
2. be silently presented as like-for-like when metadata differs;
3. be mixed with the Enspired co-location examples or copied from screenshots;
4. overwrite platform results or provenance; or
5. be described as a term sheet, guarantee, or bankable forecast.

## 7. Tests

Pins cover template round-trip, controlled metadata, signed revenue, year and
duplicate validation, constant scenario metadata, optional physical fields,
empty/comment-only files, annual-model convention, partial/no overlap, ratio
semantics, all comparability warnings, percentage display, locked caption,
duration-preset availability, export assumptions, and a real workbook round-trip
that keeps the raw ratio as a fraction and preserves source provenance.

## 8. Deferred work

- **Quote-reconciliation waterfall** after a real quote discloses fee/risk-share
  terms; no unknown fee is invented in v1.
- **Forecast vs DA allocation vs final dispatch** audit when comparable trader
  interval data becomes available.
- **Measured capture interval** using walk-forward dispatch rather than a naked
  scalar assumption.
- **Co-location optimizer** with renewable forecast error, shared grid limit,
  curtailment, PPA opportunity cost, and cross-market allocation. The Enspired
  contour and dispatch slides motivate this work but cannot validate its math.
- Italian MSD/MB/PICASSO and capacity/MACSE revenue streams once import-first
  schemas, settlement semantics, and data provenance are locked.
