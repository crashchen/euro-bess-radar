# Annual revenue decay (spread cannibalization) — v1 design contract

Status: **DRAFT — under review.** This document defines the multi-year
revenue-erosion layer for forward-looking screening. It is deliberately a
decay assumption on the annual merchant cash-flow trajectory, not a
price-series transform, a re-dispatch model, or a build-out forecast.

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
  paying, they do not go to zero. `f = 0` (default) is pure geometric decay.

Definitions (year `t` counted from 1):

```text
weight_t                = max((1 - d)^(t-1), f)
decayed_pv_factor(L, r) = sum_{t=1..floor(L)} weight_t / (1 + r)^t
                          + frac(L) * weight_{floor(L)+1} / (1 + r)^(floor(L)+1)
```

Mechanics — all locked:

1. **Year 1 is undecayed.** `weight_1 = 1` always. Year 1 is the base
   sample's own year: the annual revenue estimated from the loaded window
   (historical, or forward-synthetic — see section 3). The knob never
   rebases or back-dates; `(1 - d)` first bites in year 2.
2. **Fractional life follows the existing convention.** The definition
   above mirrors `annuity_pv_factor` exactly: integer years discounted
   in full, the fractional residual as a pro-rata cash flow at the end of
   the next year — at that year's decayed weight.
3. **Off means the legacy code path.** With `d = 0` (or `f >= 1`, which
   makes every weight 1), the implementation must route through the
   existing `annuity_pv_factor` — not a loop that happens to sum to the
   same value — so the disabled feature is **bit-identical** to today,
   including floating-point behaviour. Same discipline as the liquidity
   cap's off-means-absent rule.
4. **The wear/degradation cost stays flat.** In
   `calculate_npv_distribution`, NPV becomes
   `R * decayed_pv_factor - C * annuity_pv_factor - capex`
   (revenue decays, the constant annual degradation cost `C` does not).
   Rationale: with no re-dispatch, v1 cannot know how much cycling falls as
   spreads compress; holding the cost flat while revenue falls is the
   conservative screening direction and must be disclosed. Discounting
   `(R - C)` by the decayed factor would silently decay the cost too and is
   prohibited. When decay is off, the legacy expression
   `(R - C) * annuity_pv_factor` is kept verbatim (bit-identity, point 3).
5. **The decay acts on the trajectory, not on year-1 estimation.** The
   bootstrap distribution (`bootstrap_annual_revenue`), the 365.25-day
   annualisation, capture rate, and every single-year screening number are
   untouched. The knob only changes how year-1 revenue is projected across
   years 2..L inside PV formulas.
6. **The decay multiplies the merchant revenue draw, whole.** v1 applies
   one decay to the annual revenue entering the NPV chain (the Revenue
   tab's DA-arbitrage-based Monte-Carlo draws). No per-stream decay split
   (ancillary/reserve streams have different cannibalization physics —
   section 9.5); no interaction with the cockpit strategy rows.

### 2.1 Input validation and units

- `annual_decay_rate` must be finite and in `[0, 1)`; `decay_floor_share`
  must be finite and in `[0, 1]`; NaN/Inf anywhere raises with a
  field-named message (house rule from the contracted-floor contract).
- Percent-vs-fraction: UI fields are percent; conversion divides by 100 at
  the UI boundary. Cleared (None) number inputs must be guarded exactly
  like the LH-B share fix — never reach `float(None)`.
- The UI must display the decay rate, the floor share, and the resulting
  interpretation ("year 20 earns X% of year 1") whenever the knob is
  active. A user must never have to infer that late years were derated.

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

Provenance: whenever the knob is active, the NPV/sensitivity outputs must
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
  feed a decayed merchant number into the floor panel.
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
- With `annual_decay_rate == 0` or `decay_floor_share >= 1`, it must
  **delegate to `annuity_pv_factor(life_years, discount_rate)`** (bit
  identity, mechanics point 3).
- Validation per section 2.1; `life_years <= 0` returns 0.0 and
  `discount_rate == 0` is a valid input, both matching the existing
  function's contract.
- Pure: no solver import, no pandas requirement, no mutation — same
  discipline as `contracted_floor` and `liquidity`.

`calculate_npv_distribution` and `sensitivity_table` each gain
`annual_decay_rate: float = 0.0, decay_floor_share: float = 0.0`
passthrough parameters:

- `calculate_npv_distribution`: NPV per mechanics point 4 (decayed revenue
  factor, flat cost factor; legacy expression preserved verbatim when off).
- `sensitivity_table`: every row's NPV uses the decayed revenue factor and
  flat cost factor consistently (the varied parameter still varies; the
  decay applies to all rows). When `annual_decay_rate > 0`, the default
  `vary` dict gains a fifth axis `"decay"` with **absolute** values
  `[0.0, d, min(2 * d, 0.95)]` so the tornado shows how much the erosion
  guess matters; when the knob is off the table is unchanged (no decay
  axis, bit-identical NPVs).
- The lifetime axis and the decay interact multiplicatively through the
  factor — no special-casing.

## 5. UI placement (increment SD-B)

The inputs live in the **Revenue Estimation tab's Risk Analysis (Monte
Carlo) expander**, immediately above the NPV metrics they modify:

- "Annual merchant revenue decay (%/yr)" number input, default 0, range
  [0, 99];
- "Decay floor (% of year-1 revenue)" number input, default 0, range
  [0, 100];
- both percent-entered, divided by 100 at the boundary, None-guarded
  (section 2.1).

When the decay is active (`d > 0` after conversion), the panel captions:
the decay rate and floor, the year-`floor(L)` residual share
("year N earns X% of year-1 revenue"), the year-1 basis sentence, and the
flat-wear disclosure. A hard caption states the model class, locked here
verbatim:

"Revenue-trajectory decay: screening assumption on annual merchant revenue,
not a price-series or re-dispatch model; wear cost stays flat; applies from
the loaded sample's year; user assertion — no build-out data is fetched."

Outputs (metric layout, histogram, tornado) are unchanged in shape — the
knob changes the numbers, not the charts. The bootstrap revenue histogram
and its P10/P50/P90 metrics are explicitly NOT decayed (year-1 semantics,
mechanics point 5); only the NPV metrics, NPV histogram, and tornado move.

## 6. Red lines and non-goals

- **Not a price model.** No price-series transform, no re-dispatch, no
  claim that a d% revenue decay equals a d% spread compression (section
  1.1). Labels say "revenue decay".
- **Off means absent.** Default 0 routes through the legacy code paths
  bit-identically. No default decay, no zone-calibrated suggestion, no
  silent floor.
- **Year 1 undecayed; no rebasing.** The knob projects forward from the
  loaded sample's own year only.
- **Wear cost stays flat.** Decaying the degradation cost alongside revenue
  requires a re-dispatch model (v2); v1 is deliberately conservative here.
- **No contracted-floor wiring.** The floor overlay keeps its locked
  flat-M formula; composing decay with the floor is a v2 contract
  amendment (section 9.1).
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

1. **Bit-identity off**: `decaying_annuity_pv_factor(L, r, 0.0, 0.0) ==
   annuity_pv_factor(L, r)` exactly (delegation, not approx), across
   integer, fractional, zero-rate, and zero-life cases; same for
   `f >= 1` with `d > 0`. `calculate_npv_distribution` and
   `sensitivity_table` with decay 0 return bit-identical outputs to a call
   without the new parameters (full-array equality).
2. **Known answers** (hand-computed):
   `L=3, r=0, d=0.1, f=0` → `1 + 0.9 + 0.81 = 2.71`;
   `L=2.5, r=0, d=0.1, f=0` → `1 + 0.9 + 0.5 * 0.81 = 2.305`;
   `L=4, r=0, d=0.5, f=0.4` → `1 + 0.5 + 0.4 + 0.4 = 2.3` (floor bites in
   year 3);
   one case with `r > 0` and `d > 0` verified against a manual sum.
3. **Monotonicity and bounds**: for `L > 1`, `f < 1`, the factor is
   strictly decreasing in `d`; for all inputs,
   `f * annuity_pv_factor(L, r) <= decayed <= annuity_pv_factor(L, r)`.
4. **Flat-wear pin (mutation-sensitive)**: construct `R`, `C`, `L`, `r`,
   `d` where `R * decayed - C * flat` differs materially from
   `(R - C) * decayed`, and pin the NPV to the former. A sign-flip or
   factor-swap mutation must fail this test.
5. **Fractional-life decayed residual**: the pro-rata year uses
   `weight_{floor(L)+1}` (e.g. the `L=2.5` known answer above uses
   `0.5 * 0.81`, not `0.5 * 0.9`).
6. **Domain raises**: `d` outside `[0, 1)` (including 1.0), `f` outside
   `[0, 1]`, NaN/Inf for either — field-named messages.
7. **Sensitivity axis gating**: decay axis present with absolute values
   `[0, d, min(2d, 0.95)]` only when `d > 0`; table shape unchanged when
   off; all rows decayed consistently when on (spot-check one non-decay
   row's NPV against a manual factor computation).
8. **Bootstrap untouched**: `bootstrap_annual_revenue` has no decay
   parameter and its outputs feed the NPV chain undecayed as year-1 draws.
9. **Purity**: no new imports in `src/scenario.py` beyond the existing
   ones; inputs not mutated.

SD-B AppTest pins: default renders no decay caption and NPV metrics equal
the no-knob baseline; entering a decay rate renders the hard caption
(pinned verbatim, extending the literal-copy test pattern), the residual
share caption, and changes the NPV metrics while the bootstrap revenue
P10/P50/P90 metrics stay unchanged; clearing either number input (None)
shows a friendly prompt instead of raising (the LH-B regression class);
percent-to-fraction conversion verified at the boundary (10% → 0.10).

## 8. Increment plan after lock

- **SD-A:** `scenario.decaying_annuity_pv_factor` + the
  `calculate_npv_distribution` / `sensitivity_table` passthroughs + tests
  in `tests/test_scenario.py`. No UI.
- **SD-B:** Revenue-tab inputs, captions, hard-caption literal pin, None
  guards, AppTest pins.

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
