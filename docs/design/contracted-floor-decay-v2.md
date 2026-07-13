# Decaying merchant × contracted floor — v2 composition contract

Status: **LOCKED — merged via PR #65 (`69d3bf6`).** This document composes two
locked, shipped layers — the contracted-floor overlay (`docs/design/contracted-floor-v1.md`,
PR #54/#57/#58) and the annual merchant-revenue decay
(`docs/design/spread-decay-v1.md`, PR #62/#63/#64) — into the per-year
comparison both contracts deferred: a merchant trajectory that erodes while
the contracted floor holds. It is the flagship v2 named in spread-decay §9.1.

Implementation: **FD-A IN REVIEW** — pure calculation API and contract pins;
FD-B cockpit/export wiring remains pending.

## 0. Authorised amendments

This contract amends exactly three locked clauses; everything else in both
parent contracts remains binding:

1. **Contracted-floor v1 §2 (scalar formula):** the sentence "it does not
   forecast escalation, inflation, or merchant-spread decay" is superseded
   for the ACTIVE composition path only. The inactive path keeps the v1
   scalar × annuity formula bit-identically (section 2, mechanics point 5).
2. **Contracted-floor v1 §5 / SD-B caption copy:** the flat-baseline
   disclosure sentence shipped in PR #64 is REPLACED by the trajectory
   sentence locked in section 5, extending the floor's literal-copy test.
3. **Spread-decay v1 §3/§9.1:** the "floor composition is a v2 contract"
   deferral is discharged here. The Revenue tab's own decay knob and
   captions are NOT touched (section 6).

## 1. Positioning

The floor panel currently answers "is a quoted floor worth accepting against
a FLAT merchant baseline?" — and, since SD-B, explicitly discloses that any
revenue-decay assumption is not reflected. That understates a floor's value
in exactly the scenario that motivates buying one: as build-out compresses
spreads, the merchant trajectory erodes while the contractual floor holds,
so the counterparty top-up grows year by year and there is a **crossover
year** after which the floor binds permanently. A flat comparison prices the
floor at its year-1 top-up forever; a decaying comparison shows the floor's
top-up value growing exactly when the market thins. (The protected cash
flow itself still falls with the market — the floor limits the downside, it
does not restore a healthy merchant year; UI copy must not claim
otherwise.)

v1 answers one narrow question:

> Given the frontier's year-1 merchant net, a user-asserted decay, and a
> quoted floor, in which year does the floor start binding, and what are the
> contract-window PVs of the merchant, protected, and top-up cash flows?

Everything remains a screening overlay on the SAME frontier best-row
baseline: no dispatch effect, no re-dispatch under thinner spreads, no
credit/tax/financing, no post-term value.

## 2. Locked economic convention

Inputs (beyond the v1 floor inputs, which are unchanged):

- `annual_decay_rate` (`d`): `[0, 1)`, default **0.0**; `decay_floor_share`
  (`f`): `[0, 1]`, default **0.0**. Same domains, same percent-entry
  convention, and the SAME weight definition as spread-decay §2:
  `weight_t = max((1 - d)^(t-1), f)`, year 1 undecayed. These are the
  panel's OWN inputs (section 3); they are a separate assertion from the
  Revenue tab's knob.
- `floor_escalation_rate` (`g`): `[0, 1]`, default **0.0**. Annual geometric
  escalation of the effective floor: `F_t = F_eff * (1 + g)^(t-1)`.
  Covers fixed-nominal (`g = 0`) and indexed floors without a CPI model; a
  DECAYING floor (`g < 0`) is rejected — a floor that tracks the market
  down is not a floor product (deferred, section 9).
- `merchant_gross_eur_per_mw_yr` (`G`): the SAME frontier best row's
  annualised gross (pre-wear) merchant revenue. Required by the active
  path only, validated always (section 2.1).
- `MAX_FLOOR_TRAJECTORY_YEARS = 100.0`: one shared API/UI safety limit for
  the ACTIVE path's materialised per-year trajectory. It is an execution
  bound, not an economic assumption; the inactive delegate keeps the full
  positive-tenor domain of the shipped v1 scalar calculation (mechanics 7).

Definitions (year `t` counted from 1; `N` = the v1
`merchant_net_eur_per_mw_yr × power`, `G` scaled by power likewise;
`F_eff` = quoted floor × availability × power, as in v1):

```text
weight_t     = max((1 - d)^(t-1), f)                # spread-decay §2, shared
M_t          = N + G * (weight_t - 1)               # merchant trajectory
F_t          = F_eff * (1 + g)^(t-1)                # escalating floor
P_t          = max(M_t, F_t)                        # protected cash flow
T_t          = P_t - M_t                            # top-up, >= 0 by construction
binding_t    = (T_t > 0)                            # strict: F_t > M_t
composition_active = ((d > 0) and (f < 1)) or (g > 0)
```

`binding_t` is STRICT: an exact tie (`F_t == M_t`) pays no top-up and is
not a binding year; `crossover_year` is the first year with `binding_t`
true. Because `F_t − M_t` is non-decreasing (mechanics 3), both the strict
and non-strict sets are suffixes — the strict rule is locked so the flag
always agrees with "the counterparty actually pays".

Mechanics — all locked:

1. **Gross decays, wear stays flat — via the year-1-anchored form.**
   `M_t = N + G·(weight_t − 1)` is algebraically `G·weight_t − W` (wear
   `W = G − N` held flat), the same convention as spread-decay mechanics
   point 4: with no re-dispatch we cannot know how cycling falls, so the
   conservative direction is to erode revenue and hold the wear cost.
   The anchored form is deliberate: the frontier stores
   `net = (gross − wear) · annualize` as ONE float, so recomputing
   `G − W` can differ from the stored net in the last ulp. Anchoring on
   `N` makes `M_1 == N` bit-exact and keeps every year-1 output identical
   to v1. Decaying `N` directly (`N · weight_t`) is PROHIBITED — it decays
   the wear cost too.
2. **Year 1 is undecayed and unescalated.** `M_1 = N`, `F_1 = F_eff`,
   so the v1 annual outputs (merchant, effective floor, protected, top-up)
   keep their exact v1 values and meaning whether or not the composition
   is active. Only the PV outputs and the new trajectory outputs respond
   to the knobs.
3. **Late years may go negative; binding years form a suffix.** With
   `G ≥ 0` and `weight_t` non-increasing, `M_t` is non-increasing (it
   plateaus at `N − G·(1 − f)` once the weight floor bites, which can be
   negative — reported, never clamped, per spread-decay mechanics 4).
   With `g ≥ 0`, `F_t` is non-decreasing. Therefore once `F_t > M_t` the
   floor binds for every later year: the **crossover year** (first binding
   year) is well-defined and the binding set is a suffix of the tenor.
   **Negative-year consequence, inherited and deliberately kept:** in a
   year with `M_t < 0`, the top-up `T_t = F_t − M_t` EXCEEDS the floor
   `F_t` itself. This is the v1 wear-net settlement basis at work — v1 §2
   locked it for the scalar case ("the top-up then covers the full gap
   from that loss to the contractual floor", pinned by v1 test 5) and this
   contract inherits the basis unchanged (section 9 item 1). Clamping the
   top-up basis at zero or capping `T_t` at `F_t` is PROHIBITED: it would
   break the exact per-year identity `P_t = M_t + T_t = max(M_t, F_t)`,
   contradict the shipped v1 pins, and smuggle a revenue-settled contract
   into a net-settled formula. FD-B must instead DISCLOSE the situation
   whenever it occurs (section 5's negative-year caption); a real
   revenue-settled cap is the settlement-basis amendment.
4. **Fractional tenor mirrors the annuity convention.** Integer years
   `1..floor(L)` count in full; the residual `frac(L)` is a pro-rata cash
   flow at the end of year `floor(L)+1`, with `M`, `F`, `P`, `T` all
   evaluated at that year's weight and escalation before pro-rating —
   the same convention as `annuity_pv_factor` and
   `decaying_annuity_pv_factor`.
5. **Inactive means the v1 code path.** Validation always runs first
   (section 2.1). When NOT `composition_active`, the implementation must
   **delegate to `compute_contracted_floor_overlay`** — not run a per-year
   loop that sums to the same value — so all thirteen v1 output values are
   **bit-identical** to today, and the stored net (never a recomputed
   `G − W`) is what the delegate receives. Same off-means-absent
   discipline as SD-A mechanics point 3 and the liquidity cap.
6. **One loop, per-year identities exact, PVs independent.** The active
   path computes `M_t`, `F_t`, `P_t`, `T_t` in one per-year pass with
   `P_t = max(M_t, F_t)` and `T_t = P_t − M_t` holding exactly per year;
   the three PVs are each their own discounted sum over the same pass.
   The v1 identity pin (`protected_pv − merchant_pv == top_up_pv` at
   floating-point tolerance) carries over, as does consistency with the
   SD-A factor: `merchant_pv ≈ N·A(L,r) + G·(DPV(L,r,d,f) − A(L,r))`
   where `A` is `annuity_pv_factor` and `DPV` is
   `decaying_annuity_pv_factor` (test-level pins, not the implementation —
   summation order may differ in the last ulp).
7. **The materialised trajectory has a hard row budget.** The active path
   must reject `floor_tenor_years > MAX_FLOOR_TRAJECTORY_YEARS` with a
   field-named `ValueError` BEFORE allocating or iterating `per_year`.
   Without this guard, the v1 UI's formerly unbounded tenor input turns a
   closed-form scalar calculation into a user-controlled O(tenor) loop and
   can freeze or exhaust the Streamlit process. The inactive path does not
   materialise rows and therefore still delegates for every positive finite
   tenor accepted by v1; the safety bound must not weaken off-path
   bit-identity. FD-B caps the UI at the same exported constant so ordinary
   users cannot submit an over-budget active trajectory.

### 2.1 Input validation and units

- All v1 validations are inherited unchanged. New inputs: `d`, `f` use the
  spread-decay §2.1 domains and field-named messages (`[0, 1)` with 1.0
  raising; `[0, 1]`); `g` must be finite in `[0, 1]`;
  `merchant_gross_eur_per_mw_yr` must be finite, `>= 0`, and
  `>= merchant_net_eur_per_mw_yr` (wear cannot be negative — a violated
  bound is a data error, raised with the field name, not repaired).
- Validation is unconditional: every domain is checked BEFORE the inactive
  delegation, so an invalid `g` or gross raises even at `d = 0`. After those
  validations determine `composition_active`, the trajectory-year safety
  limit is checked only for the active path; it is not a new v1 tenor domain.
- Percent-vs-fraction: all three new UI fields are percent, divided by 100
  at the UI boundary, None-guarded exactly like the LH-B share fix and the
  SD-B inputs (a cleared input shows a prompt, never reaches `float(None)`).
- The panel decay inputs are PANEL-LOCAL. **Injecting** the Revenue tab's
  Risk-Analysis values into this panel (auto-sync, sync button, shared
  widget state) is prohibited — a computed number whose provenance depends
  on hidden cross-tab session history is not auditable, the same
  stale-state rationale that locked the SD-B unconditional caption. A
  display-only, divergence-gated hint that READS those keys without
  changing any computed value is authorised and specified in section 5.
  The two knobs remain separate assertions; help text says so, and the
  assumptions export records the panel's own values.

## 3. Baseline and provenance

- The merchant baseline is unchanged: the CURRENT frontier best row (or the
  liquidity-capped row when that cap is enabled and binding), with the same
  source labels, gating, and staleness rules as CF-B/LH-B. The composition
  adds `gross_eur_per_mw_yr` to what the panel reads from that row.
- The result fingerprint therefore extends with the best row's recomputed
  GROSS (the active path depends on the gross/net split, not only the net)
  and with the three new inputs `(d, f, g)`. Everything else about the
  CF-B two-layer fingerprint is inherited.
- Year-1 basis: the decay projects forward from the frontier window's
  annualised year, exactly as spread-decay §3 projects from the loaded
  sample's year. No rebasing, no erosion before year 1.
- Provenance: whenever `composition_active`, the panel and the Excel sheet
  must show `d`, `f`, `g`, the crossover year (or "never binds in the
  window"), and the number of binding years. The v1 provenance rows are
  inherited.

## 4. Computation API (increment FD-A)

`src/contracted_floor.py` gains one public function (module stays
pandas-free and solver-free):

```python
MAX_FLOOR_TRAJECTORY_YEARS = 100.0


def compute_decaying_contracted_floor_overlay(
    *,
    merchant_net_eur_per_mw_yr: float,
    merchant_gross_eur_per_mw_yr: float,
    power_mw: float,
    quoted_floor_eur_per_mw_yr: float,
    floor_tenor_years: float,
    contract_availability: float = 1.0,
    discount_rate: float = 0.08,
    annual_decay_rate: float = 0.0,
    decay_floor_share: float = 0.0,
    floor_escalation_rate: float = 0.0,
) -> dict:
```

Return contract:

- Always: the thirteen v1 keys (section 4 of the v1 contract) plus
  `merchant_gross_eur_per_mw_yr`, `annual_decay_rate`, `decay_floor_share`,
  `floor_escalation_rate`, `composition_active`, `per_year`,
  `crossover_year`, `n_binding_years`.
- Inactive (`composition_active == False`): the thirteen v1 values come
  from the delegate bit-identically; `per_year == []`,
  `crossover_year is None`, `n_binding_years is None`. `None` here means
  **not evaluated on the inactive path** — NOT "the floor never binds":
  a flat v1 comparison can bind from year 1, which the v1 scalar output
  already reports via `annual_top_up_eur > 0`. A `0` would falsely assert
  a zero-binding evaluation.
- Active: the four annual keys keep their year-1 values (mechanics 2); the
  three PV keys hold the trajectory PVs from the per-year pass;
  `per_year` is a list of dicts
  `{year, year_fraction, weight, discount_factor, merchant_eur, floor_eur,
  protected_eur, top_up_eur, binding}` covering years `1..floor(L)` plus
  the residual year when `frac(L) > 0` (`year_fraction` is 1.0 or
  `frac(L)`; `discount_factor` is `(1 + r)^(-t)`, so each row's PV
  contribution is `year_fraction · amount · discount_factor` and the Excel
  table reconciles to the headline PVs without recomputation);
  `crossover_year` is the first binding year as an `int`, or `None` when
  the floor never binds in the window; `n_binding_years` is a count of
  binding `per_year` ROWS (the pro-rated residual year counts as one row —
  UI copy must say "binds in X of Y evaluated years", never imply elapsed
  binding time for fractional tenors).

## 5. Cockpit and export placement (increment FD-B)

The existing contracted-floor expander gains three number inputs next to
the v1 contract inputs — "Annual merchant revenue decay (%/yr)" `[0, 99]`,
"Minimum merchant share (% of year-1 merchant)" `[0, 100]`, "Floor
escalation (%/yr)" `[0, 100]`, all default 0, all ÷100 at the boundary,
all None-guarded. The second label deliberately does NOT reuse the Revenue
tab's "Decay floor" wording: inside a panel titled "Contracted floor",
"Decay floor" would collide with the contractual floor `F`. "Minimum" is
also economically exact: the share is a lower bound and may not be reached
within the selected contract tenor. It is the same `decay_floor_share`
parameter underneath; the help text states both facts and the equivalence
("same assumption as the Risk Analysis 'Decay floor' input"). The inherited
"Floor term (years)" input gains `max_value=MAX_FLOOR_TRAJECTORY_YEARS`,
imported from the calculation module so UI and API cannot drift. No new
expander and no change to the panel's gating on a current frontier result.

**Divergence hint (display-only, best-effort):** when the Revenue tab's
Risk-Analysis decay session keys exist and their values differ from this
panel's decay inputs, the panel renders one info line showing both sets
("Risk Analysis currently asserts X%/yr, floor Y%; this panel uses its own
inputs."). Reading those keys for DISPLAY is authorised here; **injecting
their values is prohibited** — no auto-sync on activation and no sync
button, because a number that changes with hidden cross-tab session
history has no auditable provenance (the same reasoning that banned the
conditional caption in spread-decay §3, which this display-only,
divergence-gated hint deliberately relaxes). When the keys are absent the
hint renders nothing.

The SD-B flat-baseline sentence is REPLACED by this unconditional sentence,
locked verbatim (literal-copy test updated):

`Merchant trajectory: projected by this panel's own decay and
floor-escalation inputs (flat when both are inactive); the Risk Analysis
revenue-decay assumption is never used in this calculation.`

("Never used in this calculation", not "not read": the section-5
divergence hint READS the Risk-Analysis keys for display, so a "not read
here" caption would be literally false once the hint ships.)

When `composition_active`, the panel additionally renders:

- a caption with the crossover year ("floor first binds in year N" or
  "floor never binds in the contract window"), the binding count phrased
  as "binds in X of Y evaluated years", and the terminal-year `M` and `F`
  values;
- when any `per_year` row has `merchant_eur < 0`, one additional caption,
  locked verbatim:

`Negative merchant years: under the wear-net settlement basis the annual
top-up exceeds the floor itself in those years; a revenue-settled contract
would cap the top-up at the floor (settlement basis is deferred, section 9
of this contract).`
- ONE new per-year trajectory chart (merchant `M_t` and floor `F_t` lines,
  top-up `T_t` bars) — the v1 grouped annual bar chart keeps its year-1
  meaning and is unchanged;
- a hard caption, locked verbatim:

`Decaying-merchant floor composition: screening projection of the
frontier's year-1 merchant net using this panel's decay and escalation
inputs; wear stays flat inside the trajectory, so late merchant years can
be negative rather than idled. No re-dispatch, credit, tax, financing, or
post-term value is modelled. PVs cover the contract window only.`

When not active, neither the trajectory chart nor these captions render,
and every displayed number equals the v1 baseline.

Excel: the `Contracted floor` sheet gains the per-year table and one
assumption row per new input plus the crossover/binding outputs, on top of
the v1 assumption rows. The KPI metric layout, the v1 chart, and the
strategy-chart exclusion guardrail are unchanged. `build_assumptions_table`
remains untouched (panel-local disclosure, as in CF-B/LH-B/SD-B).

## 6. Red lines and non-goals

- **Still a floor, still not additive**: `P_t = max(M_t, F_t)` per year,
  never `M + F`; no strategy-chart row.
- **No dispatch effect, no re-dispatch**: the decay is revenue-space
  (spread-decay §1.1); thinner spreads do not change the frontier cap,
  cycles, or wear inside this panel.
- **Wear stays flat; negative late years are reported, not clamped**; the
  idle-option clamp remains prohibited (spread-decay §9.7 rationale).
- **Off means absent**: `(d = 0, f = *, g = 0)` and `(d > 0, f = 1, g = 0)`
  route through the v1 delegate bit-identically.
- **Panel-local knobs; no Revenue-tab value coupling**: injecting
  Risk-Analysis values (auto-sync or sync button) is prohibited; the only
  authorised cross-tab touch is the section-5 display-only divergence
  hint, and the replacement caption states the independence
  unconditionally.
- **Contract-window only**: no post-term merchant tail, no project NPV,
  no terminal value. The v1 red line stands.
- **No new commercial machinery**: no CPI fetch, no indexation formulae
  beyond geometric `g`, no credit/collateral/tax/financing, no settlement
  waterfall. Quote basis stays EUR/MW-year of installed power.
- **Separately labelled assumptions**: capture rate, liquidity cap,
  revenue decay, and floor escalation are four distinct, individually
  disclosed assumptions; folding any two into one number or label is
  prohibited (extends the spread-decay no-cross-haircut rule).

## 7. Pinned identities and tests

FD-A tests (before any UI):

1. **Bit-identity off, against the v1 function**: for
   `(d=0, f=0, g=0)`, `(d=0, f=0.4, g=0)`, `(d=0.3, f=1.0, g=0)`, and a
   call omitting the three optional decay/escalation parameters entirely
   (`merchant_gross_eur_per_mw_yr` is REQUIRED and still supplied), every
   one of the thirteen v1 keys equals
   `compute_contracted_floor_overlay(...)`'s value exactly (`==`, not
   approx), `per_year == []`, `crossover_year is None`, and
   `n_binding_years is None` — including a fixture whose FLAT floor binds
   from year 1 (`F_eff > M`), where a `0` would be a false "never binds"
   claim.
2. **Known answers** (hand-computed, `r = 0`):
   `N=80, G=100, F_eff=60, d=0.5, f=0, g=0, L=3` →
   `M = [80, 30, 5]`, `P = [80, 60, 60]`, `T = [0, 30, 55]`,
   `merchant_pv = 115`, `protected_pv = 200`, `top_up_pv = 85`,
   `crossover_year = 2`, `n_binding_years = 2`;
   escalation-only `N=80 (M flat), F_eff=75, g=0.1, L=3` →
   `F = [75, 82.5, 90.75]`, `P = [80, 82.5, 90.75]`, `crossover_year = 2`;
   fractional `N=80, G=100, F_eff=75, d=0.1, f=0, g=0, L=2.5` →
   `M = [80, 70, 61]`, `merchant_pv = 180.5`, `protected_pv = 192.5`,
   `top_up_pv = 12` (residual year at weight `0.81` and `frac = 0.5`);
   and the SAME fixture at `r = 0.07` — explicitly FRACTIONAL so a
   residual discounted at year `floor(L)` instead of `floor(L)+1` cannot
   survive: `merchant_pv = 80/1.07 + 70/1.07² + 0.5·61/1.07³`,
   `protected_pv = 80/1.07 + 75/1.07² + 0.5·75/1.07³`,
   `top_up_pv = 5/1.07² + 0.5·14/1.07³`.
3. **Flat-wear pin (mutation-sensitive)**: at `N=80, G=100, d=0.5, f=0`,
   year-2 merchant is `30` (`N + G·(w−1)`), not `40` (`N·w`); a
   net-decaying mutation must fail.
4. **Year-1 anchor**: in every active fixture, `per_year[0]` reproduces
   the four v1 annual values exactly, and the four annual return keys are
   bit-equal between an active call and the matching inactive call.
5. **Plateau**: `f=0.4, L=4, g=0`: `d=0.7` vs `d=0.8` give exactly equal
   `per_year` and PVs (all post-year-1 weights sit on the floor share).
6. **Escalation-only activation**: `d=0, g>0` is active (per-year path,
   non-empty `per_year`), and `g=0` restores the delegate.
7. **Suffix property and tie strictness**: with `g >= 0`, the `binding`
   flags in `per_year` are non-decreasing (once true, true thereafter);
   `crossover_year` equals the first true index and `n_binding_years` the
   count. An EXACT-TIE fixture pins strictness:
   `N=80, G=100, d=0.5, f=0, g=0, F_eff=30, L=3, r=0` gives
   `M = [80, 30, 5]`, `F = 30` — year 2 ties exactly (`T_2 = 0`, NOT
   binding), year 3 binds, so `crossover_year = 3` and
   `n_binding_years = 1`; a non-strict (`>=`) mutation reports year 2 and
   must fail.
8. **Negative years unclamped (mutation-sensitive)**:
   `N=10, G=100, d=0.5, f=0.4, g=0, F_eff=20, L=4, r=0` gives
   `M = [10, -40, -50, -50]` and `T = [10, 60, 70, 70]` (the floor binds
   from year 1, since `F_eff > N`); in every NEGATIVE merchant year the
   top-up EXCEEDS the floor itself (`60, 70 > 20`), `merchant_pv = -130`,
   `protected_pv = 80`, `top_up_pv = 210`, `crossover_year = 1`,
   `n_binding_years = 4`. A top-up clamp (`min(T_t, F_t)`) or a merchant
   clamp (`max(M_t, 0)`) must fail this pin; per-year identity
   `P_t == M_t + T_t` holds on every row.
9. **PV identities**: `protected_pv − merchant_pv ≈ top_up_pv` (tolerance,
   incl. fractional tenor and `r = 0`); `merchant_pv ≈
   N·A(L,r) + G·(DPV(L,r,d,f) − A(L,r))` against the public SD-A factor.
10. **Domain raises, validation-first**: `d`, `f` domains as spread-decay
    §7 pin 6; `g` outside `[0, 1]`, NaN/Inf, `gross < 0`, and
    `gross < net` raise with field names, all even when `d = 0` would make
    the call delegate. The active-path trajectory budget is pinned at both
    sides: `L = MAX_FLOOR_TRAJECTORY_YEARS` succeeds with exactly 100
    `per_year` rows; any larger active tenor raises with
    `floor_tenor_years` in the message before iteration; the same larger
    tenor on an inactive call still delegates and matches v1 exactly.
11. **Scaling and purity**: power scaling by `k` scales all EUR outputs,
    per-year rows included, by `k`; the module gains no new imports and
    mutates no input.

FD-B AppTest pins: default renders the v1 outputs with no trajectory chart
and no composition captions; the replacement trajectory sentence renders
unconditionally and verbatim (literal-copy test swapped, the SD-B flat
sentence removed); entering a decay renders the trajectory chart, the
crossover caption with the correct year and the "binds in X of Y evaluated
years" phrasing, and the hard caption verbatim; the negative-year caption
renders verbatim exactly when a negative merchant year exists (and not
otherwise); `d>0, f=100%, g=0` renders NO composition surface and leaves
every number at the v1 baseline; `g>0` alone activates; the year-1 KPI
metrics are identical between active and inactive while the PV metrics
move; clearing any of the three inputs shows a prompt instead of raising;
percent→fraction verified at the boundary; the divergence hint renders
when injected Risk-Analysis session keys differ, and nothing renders when
they are absent or equal (and the hint never alters any computed value);
fingerprint invalidation on each new input and on a gross change at
constant net; Excel carries the per-year table (with `discount_factor`)
and the new assumption rows. The literal input label is "Minimum merchant
share (% of year-1 merchant)", its help states that the minimum may not be
reached within the contract tenor, and the floor-term widget's maximum is
the imported `MAX_FLOOR_TRAJECTORY_YEARS` constant.

## 8. Increment plan after lock

- **FD-A:** `compute_decaying_contracted_floor_overlay` + pins 1–11 in
  `tests/test_contracted_floor.py`. No UI.
- **FD-B:** panel inputs, caption replacement + literal-copy swap,
  trajectory chart, crossover caption, fingerprint extension, Excel rows,
  AppTest pins.

Each increment receives the usual implementation review plus an independent
commercial-semantics pass. The design PR itself must be reviewed before
either increment begins.

## 9. Open questions intentionally deferred

1. **Settlement basis**: floors against gross revenue, EBITDA, or a defined
   cash waterfall (contracted-floor v1 §9.2) — the composition inherits the
   wear-net basis; changing it is a different amendment.
2. **Market-tracking or capped floors** (`g < 0`, collars, revenue shares):
   different products, different contracts.
3. **CPI-linked indexation with fetched inflation data** — import-first
   playbook if ever needed; `g` stays a user assertion.
4. **Post-term merchant tail / project NPV** — still the v1 red line.
5. **Price-space decay with re-dispatch** (spread-decay §9.2) — the
   physically correct erosion model; expensive, and orthogonal to this
   composition.
6. **Decay-rate uncertainty** (Monte-Carlo over `d`) and per-stream decay —
   spread-decay §9.5/§9.6.
