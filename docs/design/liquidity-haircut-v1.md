# Self-cannibalization / liquidity haircut — v1 design contract

Status: **DRAFT - IN REVIEW.** This document locks the liquidity-realism
layer only after review and merge. It is deliberately a feasible-volume cap
on the existing DA merchant chain, not a price-impact, market-depth, or
bidding-strategy model.

## 1. Positioning

Every current merchant number in the dashboard implicitly assumes the asset
can transact its full power rating in every interval without moving the
market. In thin bidding zones (the r2b workshop deck flagged IT_SICI, BG and
RO, slide 117) that assumption materially overstates screening revenue: a
100 MW asset in a zone that clears a few hundred MWh per delivery hour
cannot place its full rating at the historical price.

v1 answers one narrow question:

> When my asset's tradable MW is material relative to the zone's day-ahead
> market volume, how much of the frontier merchant estimate survives?

It does so with a **participation cap on executable power**: the asset may
transact at most a stated share of the zone's average hourly DA cleared
volume. The capped power runs through the existing dispatch optimiser, so
the haircut inherits all current dispatch physics (efficiency, duration,
cycle caps, wear, VWAPs) instead of inventing a second revenue formula.

The v1 scope is the **DA-only cycle-cap frontier chain** — the audited
merchant baseline that the contracted-floor overlay consumes. It is not a
market-wide toggle: single-day/multi-day replay, IDA policy, reserve,
activation, imbalance, stochastic, portfolio, and zone-comparison views are
all out of scope (section 9). The original r2b framing (a haircut on IDA/IDC
climatology spreads) needs intraday volume data and its own basis; it is
explicitly deferred, not silently folded into v1.

### 1.1 Why a power cap, not a revenue multiplier

A scalar revenue discount `revenue x g(share)` is not screening-safe:
arbitrage revenue is concave in power (the marginal MW earns less than the
average MW), so a multiplier over-corrects in some regimes and
under-corrects in others, and it cannot interact with the cycle cap, which
binds on energy. Capping executable power inside the existing MILP lets the
optimiser re-time the schedule under the constraint — the honest screening
analogue of "you can only show N MW to this market per interval".

## 2. Locked economic convention

Inputs:

- `zone_da_volume_mw`: explicit, user-entered **average single-sided DA
  cleared volume per delivery hour** for the zone, in MWh per hour
  (numerically MW). Single-sided means the auction's cleared buy volume
  (== cleared sell volume), never buy + sell summed — double counting would
  halve every participation share. Positive and finite when the feature is
  enabled. **Normalisation is pinned**: whatever the source publishes
  (per-MTU MWh at 15/30/60 min, daily or yearly totals), the entered value
  is `total MWh over the reference period / hours in that period` —
  equivalently MWh per delivery interval divided by the interval length in
  hours — so the number is an average MW independent of the market time
  unit. The UI help text must state this rule. v1 has **no data source for
  this number**: the repository fetches no traded-volume data (verified),
  so the value is a user assertion with provenance recorded as such.
- `max_participation_share`: scalar in `(0, 1]`, default **0.10**. The
  maximum share of hourly market volume the asset is assumed able to
  transact without materially moving the price. This is a screening
  calibration knob, not a fitted constant; the default is conservative
  practitioner folklore and must be displayed, never buried.
- `power_mw`: the installed sidebar power basis (unchanged meaning).

Definitions:

```text
executable_power_mw     = min(power_mw, max_participation_share x zone_da_volume_mw)
binding                 = executable_power_mw < power_mw
participation_at_full   = power_mw / zone_da_volume_mw      # diagnostic
```

Mechanics — all four locked:

1. `executable_power_mw` is fed to the dispatch solver as a **per-interval
   physical trading cap** through the existing `solve_daily_lp`
   `power_cap_mw` parameter (built for reserve headroom). No new solver
   constraint, no solver change.
2. **Energy capacity stays installed**: `capacity_mwh = power_mw x
   duration_hours` exactly as today. The cap limits market throughput per
   interval, not the size of the tank. A binding cap therefore behaves like
   a longer-duration asset at lower power, which is the correct physics.
3. **Every per-MW denominator stays installed power.** EUR/MW/yr, the
   best-cap net-tolerance conversion (`NET_TOL_EUR_PER_MW_YR x power_mw x
   valid_days / 365.25`), EFC and cycle-cap accounting (both
   capacity-based) are unchanged. A binding cap must show up as a LOWER
   headline EUR/MW/yr; renormalising by executable MW would cancel the
   haircut out of the headline and is prohibited. Because the tolerance
   stays on installed power, the capped run's best-cap/`frontier_flat`
   selection can legitimately differ from a resized-asset run near the
   tolerance boundary — the selector basis is part of this contract, not
   an implementation accident.
4. The cap is a **constant scalar across intervals** in v1 (one average
   volume per zone). Hour-of-day volume shape is v2 realism (section 9).
5. The cap clips **both directions** — charge and discharge bounds alike
   (`power_cap_mw` semantics) — including charging into negative prices: a
   thin market limits how many MW you can buy at a negative print exactly
   as it limits selling.
6. The frontier's **uncapped reference row stays cycle-uncapped only**:
   with liquidity enabled it is still executable-power-capped, like every
   other row, so the co-temporal comparison across cycle caps shares one
   power basis and `net_delta_vs_uncapped` stays meaningful. UI and export
   must say "uncapped = no cycle cap; liquidity cap still applied".

### 2.1 Input validation and units

- With the feature enabled, `zone_da_volume_mw` must be positive and
  finite; `max_participation_share` must lie in `(0, 1]`; NaN/Inf anywhere
  raises with a field-named message (house rule from the contracted-floor
  contract).
- With the feature disabled (no volume entered), the chain is **bit-identical
  to today** — disabled means absent, never "cap at some default volume" and
  never executable power zero.
- The UI must display installed power, executable power, the entered volume,
  the share cap, and whether the cap binds. A user must never have to infer
  that a liquidity derating has been applied — same disclosure standard as
  the contracted-floor availability haircut.

## 3. Baseline, provenance and downstream interaction

The haircut applies to the cycle-cap frontier sweep (`solve_daily_lp` per
clean local day per cap). The frontier's sweep fingerprint gains the
liquidity inputs (enabled flag, volume, share), so:

- changing or toggling the liquidity inputs invalidates the cached frontier
  result exactly like any other frontier knob;
- the contracted-floor overlay inherits invalidation automatically through
  its existing two-layer fingerprint (frontier fingerprint + recomputed
  best-row values) — no contracted-floor code change.

Provenance rows (panel + Excel) must record: user-entered volume (labelled
"User-entered zone DA volume" — there is no fetched source in v1), share
cap, executable power, binding status, and the DA-only basis.

Contracted-floor interaction is deliberate and stated: the floor quote basis
remains **EUR/MW-year of installed power** (contracted-floor contract
section 2.1). A binding liquidity cap lowers the merchant baseline `M` while
the effective floor `F` is unchanged, so the annual top-up rises — a thin
zone makes a quoted floor more valuable. This composition is the screening
insight, not a bug. The floor **formula and calculation module need no
change**, but the floor contract's section 3 requires baseline provenance
to be carried verbatim — so when the inherited frontier baseline was
liquidity-capped, LH-B must extend the floor panel's caption and its
assumption rows with the inherited liquidity fields (volume, share,
executable power, binding). A liquidity-capped merchant baseline silently
exported as an unconstrained one would violate the floor contract.

The capture-rate sidebar haircut and this cap do **not** overlap by
construction in v1: capture models price-capture imperfection
(slippage/forecast error) and is NOT applied in the frontier (raw solver
values, already stated in the frontier export); liquidity models executable
volume. The audit rows must keep the two labels distinct so neither is ever
tuned to stand in for the other.

## 4. Computation API (increment LH-A)

New module `src/liquidity.py`:

```python
def compute_liquidity_cap(
    *,
    power_mw: float,
    zone_da_volume_mw: float,
    max_participation_share: float = 0.10,
) -> dict[str, float | bool]:
```

Returned fields:

- `executable_power_mw`, `binding`;
- `participation_at_full_power` (= power / volume, diagnostic),
  `applied_share_cap`, `zone_da_volume_mw`, `power_mw` (echoes).

`cycle_frontier.compute_cycle_cap_frontier()` gains one optional parameter
`executable_power_mw: float | None = None` (default `None` = today's
behaviour, bit-identical). When set, it must satisfy
`0 < executable_power_mw <= power_mw` (raise otherwise) and is passed to
every `solve_daily_lp` call as `power_cap_mw`; `power_mw` keeps feeding
capacity, EFC accounting, annualisation, and tolerance conversion. The
frontier summary gains `executable_power_mw` (None when off) so downstream
consumers and the export see the applied cap.

`compute_liquidity_cap` is pure: no solver import, no pandas requirement,
no mutation — same discipline as `contracted_floor`.

## 5. Cockpit and export placement (increment LH-B)

The inputs live **inside the cycle-frontier expander** (the cap must enter
the sweep and its fingerprint; a separate panel would allow a stale
mismatch):

- an "Apply liquidity participation cap" checkbox (default off);
- "Zone DA volume (MWh/h)" number input;
- "Max participation share (%)" number input (percent in the UI, divided by
  100 at the boundary — house convention).

When enabled, the panel captions installed vs executable power, the share
cap, and a binding/non-binding statement, and the frontier caption gains
the derating note plus the reference-row clarification "uncapped = no
cycle cap; liquidity cap still applied" (section 2 mechanics point 6).
Outputs (table schema, chart, best-cap rule) are unchanged — the haircut
changes the numbers, not the shape. Excel export appends the section-3
liquidity provenance rows via `_append_frontier_assumptions` when the cap
is on, and the **contracted-floor panel's caption and assumption rows gain
the inherited liquidity fields** when the consumed baseline was capped
(section 3). `build_assumptions_table` is unchanged in v1 (panel-local
knob, panel-local export — same decision as the frontier and
contracted-floor contracts).

A hard caption when enabled: "Liquidity participation cap: screening
feasible-volume derating, not a price-impact or market-depth model;
executable power min(P, s x V) at user-entered volume; DA only."

## 6. Red lines and non-goals

- **Not price impact.** No elasticity, no bid-curve depth, no price
  modification. The historical price series is taken as given; the cap only
  limits how many MW can transact at it. A price-impact model is a v2+
  contract.
- **Installed-power denominators.** EUR/MW/yr and every per-MW output stay
  per installed MW. A binding cap lowers the headline; it must never be
  renormalised away.
- **Installed energy.** `capacity_mwh` never shrinks with the cap.
- **Off means absent.** No volume proxy, no silent default volume, no
  zone-table lookup in v1. Disabled input reproduces today's outputs
  bit-identically.
- **DA-frontier scope only.** No IDA/IDC spread haircut, no reserve,
  activation, imbalance, stochastic, portfolio, or zone-comparison wiring.
- **No double derating.** The capture haircut and the liquidity cap remain
  separately labelled concepts; v1's frontier basis applies capture nowhere,
  and no code path may multiply both into one number without a future
  contract.
- **No market-power/compliance claim.** The share cap is a screening
  feasibility assumption, not a REMIT or market-manipulation threshold.

## 7. Pinned identities and tests

LH-A tests (before any UI):

1. `executable_power_mw=None` (feature off) leaves the frontier output
   bit-identical to a run without the parameter — full-frame equality, not
   spot checks.
2. A non-binding cap (`s x V >= P`) also reproduces the off outputs exactly.
3. **Capacity-preservation cross-check, window-EUR scope only**: with a
   binding cap `e`, each frontier row's **window EUR and raw-FEC fields**
   (`gross_eur`, `wear_eur`, `net_eur`, `avg_efc_per_day`, VWAPs) match a
   plain frontier run at `power_mw = e, duration_hours = P x D / e` (same
   capacity, same power bound) **within solver and min-FEC tie-break
   tolerance** — the feasible sets are mathematically identical, but the
   constraint matrices are not byte-identical (big-M rows scale with
   `power_mw`) and pass 2 accepts `tol_z` objective slack, so the pin is
   per-field `pytest.approx`, not frame equality. The pin deliberately
   does NOT cover `best_cap_label` / `frontier_flat` / uplift-NaN gating:
   those read the net tolerance converted at installed `P`, so the capped
   run and the resized run can legitimately disagree near the tolerance
   boundary (section 2 mechanics point 3).
4. **Selector basis pin**: in a liquidity-capped run, the best-cap
   tolerance conversion still uses installed `power_mw` — construct a
   fixture where an executable-power-based tolerance would flip the
   best-cap choice and assert it does not.
5. On a fixture where the cap truly binds the optimum, gross window EUR is
   strictly below the uncapped-feature run (feasible-set shrinkage), and
   gross EUR/MW/yr falls with the denominator held at installed power. Net
   is reported, not sign-pinned (the optimiser stays wear-blind per the
   frontier contract, so net ordering is not guaranteed in pathology).
6. **Uncapped-row basis**: with liquidity enabled, the cycle-uncapped
   reference row is also executable-power-capped (its gross falls on a
   binding fixture), and `net_delta_vs_uncapped` is computed against that
   liquidity-capped reference.
7. **Both directions**: on a negative-price fixture the charge leg is
   clipped to `e` exactly like the discharge leg.
8. **Joint binding**: a fixture where the liquidity cap and a cycle cap
   bind together — discharged energy respects `min` of both constraints
   and the day stays feasible.
9. `compute_liquidity_cap` known answers: binding and non-binding cases,
   `binding` flag, `participation_at_full_power`, share-cap echo;
   `share = 1` is valid (the asset may be assumed able to absorb the whole
   cleared volume) and non-binding when `V >= P`; a very small positive
   volume yields a small positive executable power (never zero — forced
   near-idle days are a legitimate solve, executable zero is not).
10. Validation raises with field-named messages: volume <= 0 / NaN / Inf,
    share outside `(0, 1]`, share NaN, `executable_power_mw <= 0` or
    `> power_mw` at the frontier boundary.
11. Purity: no solver import in `src/liquidity.py`, no input mutation.
12. Frontier summary carries `executable_power_mw` (None when off).

LH-B AppTest pins: checkbox off ⇒ no liquidity captions and unchanged
fingerprint; enabling or editing volume/share invalidates the cached
frontier result AND (via the existing mechanism) a cached contracted-floor
result; the derating caption, the uncapped-row wording ("uncapped = no
cycle cap; liquidity cap still applied"), and the hard caption render when
enabled; Excel assumptions contain the liquidity rows when on and omit
them when off; the contracted-floor panel's caption and assumption rows
carry the inherited liquidity fields (volume, share, executable power,
binding) when the frontier baseline was capped, with its power basis still
installed power.

## 8. Increment plan after lock

- **LH-A:** `src/liquidity.py` + the `compute_cycle_cap_frontier`
  passthrough + tests in `tests/test_liquidity.py` (plus frontier
  regression additions). No UI.
- **LH-B:** frontier-panel inputs, fingerprint extension, captions, Excel
  provenance rows, and AppTest pins.

Each increment receives the usual implementation review plus an independent
commercial-semantics pass. The design PR itself must be reviewed before
either increment begins.

## 9. Open questions intentionally deferred

Reasons to write a v2 contract rather than silently expanding this one:

1. Hour-of-day / seasonal volume shape (per-interval `power_cap_mw` vector
   is already solver-supported; needs a volume time-series import template
   and provenance).
2. A live zone-volume source: ENTSO-E publishes DA traded volumes; an
   import-first template + fetcher increment would follow the established
   playbook (entsoe-py support to be verified in that increment, not
   assumed here).
3. Price-impact / elasticity / depth-curve modelling, including whether
   large shares should degrade the realised spread rather than the volume.
4. The original r2b target: a liquidity haircut on IDA/IDC climatology
   spreads (intraday volumes are much thinner than DA; needs its own data
   and basis).
5. Zone-comparison and portfolio views (participation shares aggregate
   across assets; a portfolio in one zone shares one volume).
6. Volume proxies (peak load share, installed RES) for zones where the user
   has no volume number — must be clearly labelled if ever added.
