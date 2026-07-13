# Annual revenue decay (spread cannibalization) — v1 design contract

Status: **LOCKED — merged via PR #62 (`8ba738c`).** This document defines the multi-year
revenue-erosion layer for forward-looking screening. It is deliberately a
decay assumption on the annual merchant cash-flow trajectory, not a
price-series transform, a re-dispatch model, or a build-out forecast.

Implementation: **SD-A IN REVIEW — pure calculation layer; SD-B pending.**

## 1. Positioning

Every multi-year number in the dashboard currently extrapolates the year-1
annual revenue flat across the asset life: `calculate_npv_distribution`
states "each simulation uses the same annual revenue across all years
(static assumption)", and `sensitivity_table` prices every row off the same
flat `annuity_pv_factor`. The r2b workshop deck and the Sicily 45MW memo
flag why that overstates late-life value: BESS build-out (the 2028+ MACSE
wave in Italy, storage pipelines across EU zones) compresses the very
spreads the revenue is harvested from — each new battery cannibalizes the
arbitrage pool. A 20-year NPV computed at frozen year-1 spreads is the
single most optimistic assumption left in the screening chain.

v1 answers one narrow question:

> If merchant revenue erodes by a stated percentage each year, what happens
> to the NPV distribution and its break-even probability?

It does so with an **explicit annual decay rate on the merchant revenue
trajectory**, applied inside the existing NPV/sensitivity machinery: year 1
keeps the full annual revenue the dashboard already estimates; year `t`
earns `(1 - d)^(t-1)` of it, optionally floored at a stated share of the
year-1 level (decay-to-equilibrium, not decay-to-zero). The knob is a user
assertion — v1 fetches no build-out pipeline data and fits no decay curve.

### 1.1 Why revenue-space, not price-space

The honest name for this knob is a **revenue**-trajectory decay, even
though the underlying story is spread compression. Mapping a price-space
spread compression to a revenue change is nonlinear: round-trip losses tie
part of the cash flow to the price *level* (charging buys more MWh than
discharging sells), the optimiser re-times the schedule when the shape
flattens, cycle counts and therefore wear fall with thinner spreads, and
level-dependent VOM interactions shift the cycling threshold. None of that
can be captured by scaling a scalar. A price-space decay (flatten each
day's prices toward the daily mean for a target year and re-run dispatch)
is the physically correct model and is deliberately deferred (section 9.2);
v1 decays the annual cash flow directly, which is exactly how practitioner
screening decks quote it ("x% p.a. merchant erosion"). UI copy and export
labels must say **revenue decay**, never claim the price series was
transformed.

## 2. Locked economic convention

Inputs:

- `annual_decay_rate` (`d`): scalar in `[0, 1)`, default **0.0** (= feature
  off). The fraction of the previous year's merchant revenue lost per year.
  Entered in percent in the UI, divided by 100 at the boundary (house
  convention). A screening assertion, not a fitted constant; v1 provides no
  default other than 0 and no zone lookup.
- `decay_floor_share` (`f`): scalar in `[0, 1]`, default **0.0**. Long-run
  floor on the decayed year weight as a share of year-1 revenue —
  `weight_t = max((1 - d)^(t-1), f)`. Motivated by new-entrant economics:
  spreads compress toward an equilibrium where further build-out stops
  paying, they do not go to zero. `f = 0` (default) is pure geometric
  decay; the UI help text recommends a positive equilibrium floor
  (section 5) without ever defaulting one in.

Definitions (year `t` counted from 1):

```text
weight_t                = max((1 - d)^(t-1), f)
decayed_pv_factor(L, r) = sum_{t=1..floor(L)} weight_t / (1 + r)^t
                          + frac(L) * weight_{floor(L)+1} / (1 + r)^(floor(L)+1)
active                  = (d > 0) and (f < 1)
```

**Feature activity is `active`, everywhere.** `d > 0` with `f = 1` makes
every weight 1: the knob is numerically off and must behave as off in the
factor (delegation, mechanics point 3), the NPV/sensitivity paths, the
captions, and the sensitivity decay axis. No surface may caption or gate on
`d > 0` alone.

Mechanics — all locked:

1. **Year 1 is undecayed.** `weight_1 = 1` always. Year 1 is the base
   sample's own year: the annual revenue estimated from the loaded window
   (historical, or forward-synthetic — see section 3). The knob never
   rebases or back-dates; `(1 - d)` first bites in year 2.
2. **Fractional life follows the existing convention.** The definition
   above mirrors `annuity_pv_factor` exactly: integer years discounted
   in full, the fractional residual as a pro-rata cash flow at the end of
   the next year — at that year's decayed weight `weight_{floor(L)+1}`.
3. **Inactive means the legacy code path.** Validation always runs first
   (section 2.1; an invalid `f` raises even when `d = 0`). After
   validation, when NOT `active` (`d == 0.0` or `f == 1.0`) the factor
   implementation must **delegate to `annuity_pv_factor`** — not run a
   loop that happens to sum to the same value — and
   `calculate_npv_distribution` must keep the legacy expression
   `(R - C) * annuity_pv_factor - capex` **verbatim** (not the
   algebraically equal but floating-point-different
   `R * factor - C * factor - capex`), so the inactive knob is
   **bit-identical** to today. Same discipline as the liquidity cap's
   off-means-absent rule.
4. **The wear/degradation cost stays flat.** When `active`,
   `calculate_npv_distribution` computes
   `R * decayed_pv_factor - C * annuity_pv_factor - capex`
   (revenue decays, the constant annual degradation cost `C` does not).
   Rationale: with no re-dispatch, v1 cannot know how much cycling falls as
   spreads compress; holding the cost flat while revenue falls is the
   conservative screening direction. Consequence, deliberately accepted
   and disclosed: once `R * weight_t < C`, late years show **negative
   operating margins** instead of an idled asset. An idle-option clamp
   (`max(R * weight_t - C, 0)` per year) is prohibited in v1 because it
   creates a discontinuity at the off boundary — for draws with `R < C`
   the legacy path legitimately reports a negative net annuity, while any
   clamped path would jump to `-capex` at `d = 0+` — and because a real
   operator idles *partially* (best days only), which needs the
   re-dispatch model (section 9.7). Discounting `(R - C)` by the decayed
   factor would silently decay the cost too and is likewise prohibited.
5. **The decay acts on the trajectory, not on year-1 estimation.** The
   bootstrap distribution (`bootstrap_annual_revenue`, which resamples 365
   daily revenues per draw), the degradation annualisation (the 365.25-day
   cycle convention), capture rate, and every single-year screening number
   are untouched — the knob changes neither convention. It only changes
   how year-1 revenue is projected across years 2..L inside PV formulas.
6. **The decay multiplies the merchant revenue draw, whole.** v1 applies
   one decay to the annual revenue entering the NPV chain (the Revenue
   tab's DA-arbitrage-based Monte-Carlo draws). No per-stream decay split
   (ancillary/reserve streams have different cannibalization physics —
   section 9.5); no interaction with the cockpit strategy rows.

### 2.1 Input validation and units

- `annual_decay_rate` must be finite and in `[0, 1)` (1.0 itself raises);
  `decay_floor_share` must be finite and in `[0, 1]`; NaN/Inf anywhere
  raises with a field-named message (house rule from the contracted-floor
  contract). Validation is unconditional: an out-of-domain `f` raises even
  when `d == 0` would make the factor delegate.
- Percent-vs-fraction: UI fields are percent; conversion divides by 100 at
  the UI boundary. Cleared (None) number inputs must be guarded exactly
  like the LH-B share fix — never reach `float(None)`.
- The UI must display the decay rate, the floor share, and the resulting
  interpretation ("year N earns X% of year-1 revenue", section 5) whenever
  the knob is `active`. A user must never have to infer that late years
  were derated.

## 3. Baseline, provenance and the forward-scenario boundary

The v1 consumer is the **Revenue Estimation tab's Risk Analysis chain**
(`bootstrap_annual_revenue` → `calculate_npv_distribution` →
`sensitivity_table` + tornado). That chain prices whatever price series is
loaded — historical or forward-synthetic — which fixes the boundary with
`forward_curve.py`:

- **Forward curves move the level; the decay knob erodes the trajectory.**
  `build_forward_synthetic_prices` overlays a forward *baseload level* onto
  the historical hour-of-week *shape* (`synthetic = forward_base *
  hist_shape`); a baseload quote carries no shape information, so the
  synthetic series inherits the historical peak/trough ratio at the forward
  level. The decay knob models what the forward overlay structurally
  cannot: multi-year erosion of merchant revenue relative to its own year-1
  value. The two compose without double counting **by construction** —
  level from the market, erosion from the user.
- **Year 1 = the base sample's delivery year.** On a forward-synthetic 2028
  series, year 1 of the decay is 2028; the knob does not add erosion
  between today and the forward's delivery year (the market's level quote
  already prices that path).
- **Red line: forward CSVs are never pre-decayed.** The forward template's
  prices are market quotes; the decay is a scenario assumption that lives
  in the UI knob and the provenance rows, never mixed into the price file
  (same separation discipline as the activation capture share). Help text
  must warn about the one real double-count risk: a user who manually
  enters an already-eroded forward level *and* a decay rate calibrated to
  total erosion counts the level leg twice.

Provenance: whenever the knob is `active`, the NPV/sensitivity outputs must
be accompanied by the decay rate, the floor share, and the year-1 basis
("decay applies from the loaded sample's year"). `build_assumptions_table`
is **not** extended in v1 (panel-local knob, panel-local disclosure — same
decision as the frontier, contracted-floor, and liquidity contracts).

Downstream non-interaction (all deliberate, all disclosed where relevant):

- **Contracted-floor overlay: out of scope in v1, flagship v2.** The floor
  panel's PVs keep the flat-M assumption (`max(M, F)` scalar × flat
  annuity, per the locked contracted-floor contract). Composing a decaying
  `M_t` with a flat floor is exactly where a floor becomes most valuable —
  the top-up grows as merchant erodes, with a crossover year once
  `M_t < F` — but `PV(max(M_t, F))` no longer factors into
  `max(M, F) * annuity`, so it changes the floor contract's locked formula
  and needs its own amendment round (section 9.1). v1 must not silently
  feed a decayed merchant number into the floor panel. Because a user who
  sets a decay in Risk Analysis would otherwise reasonably assume the
  floor panel reflects it, SD-B must add one **unconditional** sentence to
  the contracted-floor panel caption (an authorised amendment to the floor
  contract's §5 copy, extending its literal-copy test — the LH-B label
  amendment pattern), locked here verbatim:

  `Merchant baseline: flat annual revenue across the tenor; any
  revenue-decay assumption from Risk Analysis is NOT reflected here
  (decaying-merchant floor composition is a v2 contract).`

  Unconditional, because a conditional cross-tab warning would couple the
  cockpit panel to Revenue-tab session state (stale-state risk for zero
  disclosure gain — the sentence is true whether or not a decay is set).
- **Cockpit strategy comparison, frontier, replay: untouched.** Those are
  single-period screening views annualised from the sample window; the
  decay is a multi-year projection assumption and appears only where
  multi-year PVs are computed.

## 4. Computation API (increment SD-A)

`src/scenario.py` gains one public function:

```python
def decaying_annuity_pv_factor(
    life_years: float,
    discount_rate: float,
    annual_decay_rate: float = 0.0,
    decay_floor_share: float = 0.0,
) -> float:
```

- Implements the section-2 definition (loop over integer years plus the
  fractional residual is acceptable; no closed form required).
- Validates first (section 2.1), then when not `active` **delegates to
  `annuity_pv_factor(life_years, discount_rate)`** (bit identity,
  mechanics point 3).
- `life_years <= 0` returns 0.0 and `discount_rate == 0` is a valid input,
  both matching the existing function's contract. The discount-rate domain
  is otherwise inherited unchanged from `annuity_pv_factor` (which does
  not validate `r`); the section-7 bound pins apply for `r >= 0` only.
- Pure: no solver import, no pandas requirement, no mutation — same
  discipline as `contracted_floor` and `liquidity`.

`calculate_npv_distribution` and `sensitivity_table` each gain
`annual_decay_rate: float = 0.0, decay_floor_share: float = 0.0`
passthrough parameters:

- `calculate_npv_distribution`: NPV per mechanics points 3-4 (legacy
  expression verbatim when not `active`; decayed revenue factor with flat
  cost factor when `active`).
- `sensitivity_table`: every row's NPV uses the row-effective decayed
  revenue factor and the flat cost factor consistently (the varied
  parameter still varies; the base decay applies to all rows). The decay
  axis is locked as follows:
  - Only when `vary is None` **and** the knob is `active` does the default
    `vary` dict gain a fifth axis `"decay"`; a caller-supplied `vary` dict
    is never modified (it participates only if it explicitly contains a
    `"decay"` key).
  - Decay-axis values are **absolute decimal rates** (like the existing
    `discount_rate` and `lifetime` axes, unlike the multiplier semantics
    of `revenue`/`capex`): `[0.0, d, min(2 * d, (1 + d) / 2)]`. The high
    value is strictly above `d` and strictly below 1 for every
    `d in (0, 1)` — `min(2d, 0.95)` would fall below the base at
    `d > 0.95` and is rejected.
  - For a `"decay"` row, the row's decay rate is `val` and every other
    parameter stays at base; for every non-decay row, the base `d`/`f`
    apply. The floor share `f` is held at base on all rows (no floor
    axis in v1).
- The lifetime axis and the decay interact multiplicatively through the
  factor — no special-casing.

## 5. UI placement (increment SD-B)

The inputs live in the **Revenue Estimation tab's Risk Analysis (Monte
Carlo) expander**, immediately above the NPV metrics they modify:

- "Annual merchant revenue decay (%/yr)" number input, default 0, range
  [0, 99];
- "Decay floor (% of year-1 revenue)" number input, default 0, range
  [0, 100], with help text recommending a positive equilibrium floor
  without defaulting one in ("Recommended > 0 (e.g. 20-30%) to model a
  long-run equilibrium where thin spreads halt further build-out; 0 means
  pure geometric decay toward zero.");
- both percent-entered, divided by 100 at the boundary, None-guarded
  (section 2.1).

When the knob is `active` (`d > 0 and f < 1` after conversion), the panel
captions: the decay rate and floor, the terminal-year residual share
("year N earns X% of year-1 revenue" — `N = floor(L)` for integer `L`,
else the partial residual year `floor(L) + 1`, using `weight_N`), the
year-1 basis sentence, and a hard caption stating the model class, locked
here verbatim:

"Revenue-trajectory decay: screening assumption on annual merchant cash
flows; does not simulate future hourly prices or re-dispatch. Battery
degradation cost stays flat, so late decayed years can show negative
operating margins rather than an idled asset. Decay begins after year 1
(the loaded sample's year). User assertion — no build-out data is fetched."

When not `active` (including `d > 0` with `f = 1`), none of these captions
render and every output equals the no-knob baseline.

Output shape: metric layout, histogram types, table schema, and chart
types are unchanged; when `active`, the sensitivity table and tornado gain
exactly one additional `decay` axis (section 4). The bootstrap revenue
histogram and its P10/P50/P90 metrics are explicitly NOT decayed (year-1
semantics, mechanics point 5); only the NPV metrics, NPV histogram, and
tornado move.

**Tornado direction fix (mandatory, SD-B).** The current tornado assembly
sorts each axis by parameter value and labels the low-parameter end
"Downside" — which mislabels axes that are inversely related to NPV
(`capex`, `discount_rate` today; `decay` would join them). SD-B must
re-assign downside/upside by **resulting NPV** (downside = the axis value
with the lower NPV), fixing the pre-existing inversion rather than
inheriting it. This is a labelling/ordering correction only; NPV values
are unchanged. The fix applies to all axes and is pinned in section 7.

The contracted-floor panel gains the unconditional flat-baseline sentence
(section 3, locked verbatim there) in the same increment.

## 6. Red lines and non-goals

- **Not a price model.** No price-series transform, no re-dispatch, no
  claim that a d% revenue decay equals a d% spread compression (section
  1.1). Labels say "revenue decay".
- **Off means absent.** Default 0 routes through the legacy code paths
  bit-identically (legacy expression preserved verbatim, mechanics
  point 3). No default decay, no zone-calibrated suggestion, no silent
  floor. `active = (d > 0) and (f < 1)` gates every surface.
- **Year 1 undecayed; no rebasing.** The knob projects forward from the
  loaded sample's own year only.
- **Wear cost stays flat; negative late-year margins are reported, not
  clamped.** No idle-option `max(·, 0)` in v1 (mechanics point 4); the
  hard caption discloses the consequence.
- **No contracted-floor wiring.** The floor overlay keeps its locked
  flat-M formula; v1 adds only the unconditional disclosure sentence
  (section 3). Composing decay with the floor is a v2 contract amendment
  (section 9.1).
- **No cross-haircut merging.** Capture rate (intra-year price-capture
  imperfection), the liquidity participation cap (per-interval executable
  volume), and the revenue decay (multi-year trajectory) are three
  separately labelled assumptions; no code path may fold any two into one
  number or one label.
- **No auto-calibration.** v1 fetches no build-out pipeline, no MACSE
  auction results, no fitted decay curves. The rate is a user assertion
  recorded as such.

## 7. Pinned identities and tests

SD-A tests (before any UI):

1. **Bit-identity off, pinned against the legacy expression** (not
   new-call-vs-new-call, which could drift together): for
   `calculate_npv_distribution`, the expected NPV array is constructed
   literally as `net = R_draws - C; expected = net *
   annuity_pv_factor(L, r) - capex` and compared with `np.array_equal`;
   for `sensitivity_table`, the expected frame is built row-by-row from
   the legacy formula and compared with exact `assert_frame_equal`. Cases:
   a call without the new parameters, `(d=0, f=0)`, `(d=0, f=0.4)`, and
   `(d>0, f=1.0)` — all four must match the legacy expression exactly.
   Factor level: `decaying_annuity_pv_factor(L, r, 0.0, 0.0) ==
   annuity_pv_factor(L, r)` exactly across integer, fractional, zero-rate,
   and zero-life cases; likewise for `d > 0, f = 1.0`.
2. **Known answers** (hand-computed):
   `L=3, r=0, d=0.1, f=0` → `1 + 0.9 + 0.81 = 2.71`;
   `L=2.5, r=0, d=0.1, f=0` → `1 + 0.9 + 0.5 * 0.81 = 2.305`;
   `L=4, r=0, d=0.5, f=0.4` → `1 + 0.5 + 0.4 + 0.4 = 2.3` (floor bites in
   year 3);
   one case with `r > 0` and `d > 0` verified against a manual sum.
3. **Monotonicity and bounds** (for `r >= 0`): with `f = 0` and `L > 1`,
   the factor is strictly decreasing in `d`; with `0 < f < 1` it is
   non-increasing in `d`, and a **plateau fixture** where every
   post-year-1 weight sits on the floor must be exactly equal across two
   decay rates (`f=0.4, L=4`: `d=0.7` vs `d=0.8`, both weights
   `[1, 0.4, 0.4, 0.4]`); for all pinned inputs,
   `f * annuity_pv_factor(L, r) <= decayed <= annuity_pv_factor(L, r)`.
4. **Flat-wear pin (mutation-sensitive)**: construct `R`, `C`, `L`, `r`,
   `d` where `R * decayed - C * flat` differs materially from
   `(R - C) * decayed`, and pin the NPV to the former. A factor-swap or
   cost-decaying mutation must fail this test.
5. **Fractional-life decayed residual**: the pro-rata year uses
   `weight_{floor(L)+1}` (the `L=2.5` known answer above uses
   `0.5 * 0.81`, not `0.5 * 0.9`).
6. **Domain raises, validation-first**: `d` outside `[0, 1)` (including
   exactly 1.0), `f` outside `[0, 1]`, NaN/Inf for either — field-named
   messages; an invalid `f` raises even when `d = 0` (delegation never
   bypasses validation).
7. **Sensitivity decay-axis semantics**: the `"decay"` axis appears only
   when `vary is None` AND `active`; its values are the absolute rates
   `[0, d, min(2d, (1+d)/2)]` (checked at a `d` where the two candidates
   differ, and at `d = 0.97` where the high value must still exceed `d`);
   a caller-supplied `vary` dict passes through unmodified; table shape is
   unchanged when not `active` (including `d>0, f=1`); one non-decay row's
   NPV is checked against a manual factor computation at the base `d`.
   Mutations that must fail: treating decay-axis values as multipliers on
   the base rate; applying the base `d` to decay-axis rows.
8. **Bootstrap untouched**: `bootstrap_annual_revenue` has no decay
   parameter; it still resamples 365 daily revenues per draw, and its
   outputs feed the NPV chain undecayed as year-1 draws.
9. **Purity**: no new imports in `src/scenario.py`; inputs not mutated.

SD-B AppTest pins: default renders no decay caption and NPV metrics equal
the no-knob baseline; entering a decay rate renders the hard caption
(pinned verbatim via the literal-copy test pattern) plus the residual
share caption with the correct year index (an `L=2.5`-style fixture must
show the year-3 weight); `d > 0` with floor `100%` renders NO decay
captions and leaves the NPV metrics at baseline (the `active` gate);
the bootstrap revenue P10/P50/P90 metrics stay unchanged while the NPV
metrics move; clearing either number input (None) shows a friendly prompt
instead of raising (the LH-B regression class); percent-to-fraction
conversion verified at the boundary (10% → 0.10); the tornado
downside/upside assignment is NPV-sorted for **every** axis — on a fixture
where high capex lowers NPV, the capex axis's downside delta must be
non-positive (this fails against today's parameter-sorted assembly, so the
pin is mutation-sensitive by construction); the contracted-floor panel
renders the unconditional flat-baseline sentence verbatim (extending the
floor's literal-copy test).

## 8. Increment plan after lock

- **SD-A:** `scenario.decaying_annuity_pv_factor` + the
  `calculate_npv_distribution` / `sensitivity_table` passthroughs + tests
  in `tests/test_scenario.py`. No UI.
- **SD-B:** Revenue-tab inputs, captions, hard-caption literal pin, None
  guards, the tornado direction fix, the contracted-floor disclosure
  sentence + literal-copy test extension, and the AppTest pins.

Each increment receives the usual implementation review plus an independent
commercial-semantics pass. The design PR itself must be reviewed before
either increment begins.

## 9. Open questions intentionally deferred

Reasons to write a v2 contract rather than silently expanding this one:

1. **Contracted-floor composition** (flagship): `PV(max(M_t, F_eff))` with
   a decaying merchant trajectory — the crossover year makes a floor's
   value grow exactly when the market thins. Requires an authorised
   amendment to the floor contract's locked scalar formula and a decision
   on whether the floor itself escalates or decays.
2. **Price-space decay with re-dispatch**: flatten each day's prices toward
   the daily mean for a target year, re-run the dispatch MILP, and let
   cycles/wear/VWAPs respond. The physically correct model; expensive.
3. **Hour-shape flattening in the forward overlay**: decay the normalised
   shape's deviation from 1 rather than the revenue scalar, composing
   level (forward) with shape erosion (assumption) in price space.
4. **Zone-calibrated decay defaults**: build-out pipelines, MACSE/CM
   auction volumes, interconnector schedules — must arrive via the
   import-first playbook with provenance, never as silent defaults.
5. **Per-stream decay**: ancillary/reserve capacity prices cannibalize on
   different physics (saturation of small reserve markets is faster); the
   v1 knob deliberately covers only the DA-arbitrage-based revenue draw.
6. **Uncertainty on the decay rate itself**: Monte-Carlo over `d` (e.g.
   triangular around the asserted rate) rather than a point assertion.
7. **Idle-option operating floor**: clamping each year's operating cash
   flow at `max(R * weight_t - C, 0)` models the operator's option to
   stop cycling when decayed revenue no longer covers wear. Deferred with
   the re-dispatch model (9.2): the clamp is discontinuous at the off
   boundary for draws with `R < C` (the legacy path legitimately reports
   negative net annuities), and annual-granularity idling is too coarse —
   a real operator idles partially, which only a dispatch model can price.
