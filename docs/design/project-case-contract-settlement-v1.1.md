# Project Case v1.1 — annual pre-lifecycle strategy-cash floor settlement

Status: **locked** (PC-D0 design-contract — LOCKED by a dual **APPROVE** on the
candidate `0c282254aa1388d9df1f700cc43a0f98e8b1156b`: Codex **APPROVE** and
Gemini **APPROVE** — same-hash, zero blocking findings under the §13 bar —
alongside CC's independent machine-verification: both canonical ProjectCase
fingerprint golden vectors reproduced byte-for-byte + SHA-256 exact, all §10
arithmetic vectors A–D and their factor mutants recomputed (31/31 checks), and
`test_project_case_fingerprint.py` green at 17/17. This lock changes only the
status/metadata below; the contract body and the JSON golden vectors are
unchanged. PC-D1+D2 (atomic) implementation begins once this docs-only lock PR
merges.)

Candidate date: 2026-08-16

Locked on: 2026-08-16

Review metadata (lock):

```
reviewed_candidate_commit: 0c282254aa1388d9df1f700cc43a0f98e8b1156b
lock_basis: dual_approve
locked_on: 2026-08-16
```

Extends: [`project-case-v1.md`](./project-case-v1.md), especially its §4.5
`ContractCase.settlement_basis` v1.1 hook.

Depends on / must not violate: `economic-semantics-v1.md`,
`contracted-floor-v1.md`, `contracted-floor-decay-v2.md`,
`spread-decay-v1.md`, and `dispatch-failure-contract-v2.md`.

This document does **not** amend the shipped wear-net contracted-floor
comparator. It defines a new, cash-NPV-eligible settlement basis inside
`ProjectCase` and keeps the old comparator as a separate presentation sibling.

---

## 0. Authorised amendments to locked parent contracts

This contract amends only the following Project Case v1 clauses:

1. **Project Case v1 §4.5 and red-lines #5/#23:** “floor comparator-only / never
   in the aggregator or RunResult” remains fully binding for the shipped
   **wear-net** `FloorComparatorResult`. It is superseded only for the new typed
   pre-lifecycle strategy-cash settlement defined here, whose complete inputs
   are owned and fingerprinted by `ContractCase`.
2. **Project Case v1 §4.6/§4.8 payload registry:** ProjectCase v1.1 adds the
   always-present `contract_case` key (§3/§7); RunResult moves to the v1.1
   schema/provenance contract (§6/§8). The v1 StrategyRunResult payload and
   fingerprint remain unchanged. The parent §4.8 exhaustive integer-field list
   is amended for ProjectCase v1.1 by adding
   `contract_start_project_year` (strict integer, bool rejected); rank/index
   integers exist only in RunResult provenance, not in the ProjectCase
   fingerprint payload.
3. **Project Case v1 PC-B representative-table statistic:** merchant-only cases
   retain the shipped `p50_annual_bootstrap_draw_linear` path; non-null contract
   cases use the nonlinear-safe statistic in §6.2.
4. **Contracted-floor-decay-v2 prohibition on a declining floor:** it remains
   binding for that wear-net comparator and its scalar escalation knob. This is
   a different tagged product: it accepts any explicit, user-asserted,
   non-negative *real-EUR* floor curve, including a declining one, and makes no
   claim about why the asserted curve has that shape. It does not amend the old
   comparator's product claim or code.

All other parent-contract clauses remain binding, especially producer-issued
cash eligibility, non-additivity, VOM/MW/capture/availability counted once,
shadow-wear exclusion, lifecycle UNKNOWN fail-closed, and pre-tax unlevered
output naming.

## 1. Purpose and exact product boundary

Project Case v1 deliberately excludes the shipped contracted-floor result from
cash NPV because that result consumes a merchant baseline **after linear shadow
wear**. Feeding it into the aggregator would deduct installed CapEx at year 0 and
then re-import the same CapEx-derived wear proxy through `max(M, F)`.

PC-D adds one narrowly typed settlement basis:

```text
ANNUAL_PRE_LIFECYCLE_STRATEGY_CASH_FLOOR_V1
```

Public type name:

```text
AnnualPreLifecycleStrategyCashFloor
```

UI label:

```text
Annual strategy-cash floor (before lifecycle costs)
```

The deliberately long name prevents a commercial overclaim. The producer cash
is called **gross** only in the economic-semantics sense: it is before shadow
wear, fixed O&M, augmentation/replacement, terminal cash, tax, and financing.
It is already net of charging-energy purchases and embedded dispatch VOM, and
already includes the producer's capture/liquidity/currency treatment and any
reserve-availability factor exactly once. It is therefore not necessarily
legal/accounting “gross revenue” or top-line turnover.

The basis is an annual whole-project cash floor. It is **not** MACSE, a capacity
premium, tolling, a collar, a revenue share, a floor on one sub-stream, or a
legal-term-sheet engine. In particular, it does not model MACSE committed MWh,
fixed premium, positive-margin return/clawback, variable consideration,
availability tests, penalties, collateral, termination, or credit risk. Those
belong to separate future tagged settlement unions.

## 2. Scope and increments

In scope:

- an optional, fingerprinted `contract_case` inside ProjectCase v1.1;
- one annual settlement union over the selected producer-issued strategy cash;
- per-Monte-Carlo-draw, per-project-year `max(M, F)` composition;
- the existing two pre-tax unlevered NPV outcomes after settlement;
- an auditable merchant / effective-floor / top-up / settled-cash decomposition;
- exact representative-table reconciliation to the reported linear P50;
- canonical payload/version migration and golden vectors;
- minimum Excel provenance support before a non-null contract can be exported.

Out of scope:

- dispatch changes, re-optimisation, or strategy switching under the floor;
- partial contracted tranches or a different contracted MW basis;
- monthly, daily, or settlement-period floors;
- fractional-year contract terms;
- contract fees, option premia, upside sharing, collars, caps, or deductibles;
- endogenous default/credit/collateral/termination cash flows;
- tax, debt, DSCR, leverage, or a “bankable” label;
- MACSE or any other jurisdiction-specific capacity contract.

Delivery remains split and dual-reviewed:

1. **PC-D0** — this docs-only contract.
2. **PC-D1 checkpoint** — types, validation, object-specific schema versions,
   canonical payloads, and golden vectors. A non-null contract remains
   compute-unavailable until PC-D2; it must never silently run the merchant-only
   path. **This checkpoint is reviewed but never merged/released independently.**
3. **PC-D2 atomic implementation PR** — on the same branch, add the settlement
   matrix, NPV aggregation, `RunResult`, representative cash-flow tables,
   provenance, merchant-only v1.1 compatibility, and minimum Excel schema
   support. D1+D2 merge together so main never contains a v1.1 ProjectCase that
   the current exporter cannot validate.
4. **PC-D3** — Streamlit inputs, validation preview, full UI/Excel disclosure,
   and coexistence with the separate wear-net comparator.

## 3. Schema ownership

### 3.1 `ProjectCase.contract_case`

ProjectCase v1.1 adds one declared optional key. The key is always present in the
canonical payload:

```text
contract_case: null | ContractCase
```

`null` means merchant-only settlement. Omitting the key is invalid: the locked
canonical profile requires declared optionals to be present-null, and allowing
both omission and null would create two fingerprints for one case.

No-contract compatibility means **numeric parity**, not fingerprint parity:

- with `contract_case = null`, both NPV distributions and every pre-existing
  representative-table numeric column must be bit-identical to PC-B v1 for
  identical inputs; the v1.1 row/payload shape adds the explicitly versioned
  audit columns from §6.3 and is therefore not byte-identical;
- the ProjectCase input fingerprint and the RunResult schema/provenance change
  intentionally because the public input schema has changed; RunResult has no
  independent canonical fingerprint.

### 3.2 `ContractCase` tagged payload

The only non-null v1.1 union member is:

```text
contract_case = {
  settlement_basis:
    "ANNUAL_PRE_LIFECYCLE_STRATEGY_CASH_FLOOR_V1",
  settlement_terms: {
    contract_start_project_year: int,
    floor_rate_real_eur_per_modeled_mw_year_by_contract_year:
      [float, ...],
    floor_entitlement_factor_by_contract_year:
      [float, ...],
    quote_basis: "EUR_PER_MODELED_PROJECT_MW_YEAR",
    settlement_frequency: "ANNUAL_PROJECT_YEAR_END",
    asset_scope: "WHOLE_PROJECT_MODELED_MW",
    currency_basis: {
      mode: "USER_ASSERTED_REAL_BASE_YEAR_EUR_CURVE",
      target_base_year
    },
    quote_status:
      "USER_SCENARIO" |
      "USER_ASSERTED_INDICATIVE_QUOTE" |
      "USER_ASSERTED_EXECUTED_SOURCE_DOCUMENT",
    source: text,
    source_as_of_date: "YYYY-MM-DD",
    source_document_sha256: hex64 | null
  }
}
```

The annual rate array is an explicit **real, base-year-EUR** curve. A flat quote
is represented by repeating its rate. A UI convenience may generate the array
from a flat quote or an escalation assumption, but the resolved explicit curve —
not a hidden escalation knob — is what is validated and fingerprinted. This also
allows a fixed nominal quote converted to real terms to decline without mixing a
nominal curve with a real discount rate.

`floor_entitlement_factor_by_contract_year` is a deterministic contractual
entitlement scenario in `[0, 1]`. It scales the quoted floor before `max`; it is
never multiplied into merchant cash or applied later to the top-up. It is not
AssetCase availability and must never borrow the reserve-capacity availability
already embedded in producer cash. When the quoted rates already include
performance/availability deductions, every factor is `1.0` and the source must
say so.

The two arrays have the same positive length; that length is the integer
contract tenor. If `s = contract_start_project_year` and `L` is project life,
then `1 <= s <= L` and `s + len(curve) - 1 <= L`. Fractional terms are rejected
because ProjectCase cash timing and bootstrap draws are annual/end-of-year.

No independent power, duration, merchant baseline, discount rate, reserve
availability, or wear input exists in ContractCase. Whole-project modelled MW
comes only from `AssetCase.power_mw`; valuation comes only from
`ValuationCase`; merchant cash comes only from `StrategyRunResult`.

Contract currency has one deliberately narrow v1.1 basis. The stored rate curve
is a user assertion that its EUR values are already real in `target_base_year`,
and that year must equal `ValuationCase.base_year`. PC-D performs no inflation,
deflation, FX, or indexation conversion. Any transformation from a nominal or
indexed source quote happens outside the calculator and must be explained in the
fingerprinted `source`; reconstructable raw-to-real conversion is deferred to a
future currency-basis union. This avoids a scalar/year-wise conversion factor
whose direction or application owner could change cash.

`source` is required and non-empty. `source_document_sha256` is required for
both `USER_ASSERTED_*` statuses and must be null for `USER_SCENARIO`.
`quote_status` records only the user's assertion about source maturity; even
`USER_ASSERTED_EXECUTED_SOURCE_DOCUMENT` does not claim that the platform has
reproduced the complete legal contract.

## 4. Locked economic convention

Indices:

- `d` — Monte Carlo annual bootstrap draw;
- `t` — project year, integer `1..L`;
- `A_d` — annual draw from the selected producer's
  `daily_realised_cash_series`;
- `w_t` — existing ProjectCase projection multiplier (`w_1 = 1`, finite and
  non-negative);
- `D_t = (1 + r)^(-t)` — existing real end-of-year discount factor;
- `C` — contract-covered project-year set;
- `q_k` / `a_k` — quoted real floor rate and entitlement factor for contract
  year `k`, both fingerprinted.

Let `s = contract_start_project_year` and `N` be the common positive array
length. The covered set and the only legal year/index mapping are:

```text
C = {s, s+1, ..., s+N-1}
k = t - s + 1                         # one-based contract year
array_index = k - 1 = t - s           # zero-based stored-array index
```

Merchant pre-lifecycle strategy cash:

```text
M[d,t] = A[d] * w[t]
```

Effective whole-project floor:

```text
F[t] = q[t-s] * AssetCase.power_mw * a[t-s]   when t is in C
F[t] = absent                                 when t is outside C
```

Settlement:

```text
S[d,t] = max(M[d,t], F[t])                    when t is in C
S[d,t] = M[d,t]                               when t is outside C

T[d,t] = S[d,t] - M[d,t]                     # top-up; zero outside C
```

An absent floor is **not** a zero floor. Outside the contract term a negative
merchant year remains negative. Inside the term, an explicit zero floor protects
a negative merchant year to zero. Implementations must carry contract coverage
separately from the numeric floor curve so these two cases cannot collapse.

Signed `M` is never clamped. Therefore, inside the term, when `M[d,t] < 0`,
`T[d,t] = F[t] - M[d,t]` can exceed `F[t]`. This is the deliberate consequence
of the locked `max(M, F)` hook. A product that caps the payment at the quoted
floor, floors only positive revenue, or excludes energy-trading losses is a
different tagged settlement basis.

NPV draws:

```text
screening_NPV[d]
  = -installed_capex_eur + sum_t(S[d,t] * D[t])

lifecycle_adjustment
  = sum_t((-fixed_om[t] - augmentation[t] + terminal[t]) * D[t])

lifecycle_NPV[d]
  = screening_NPV[d] + lifecycle_adjustment
```

The floor settles the strategy-cash line **before** fixed O&M, augmentation,
replacement, and terminal cash. It does not protect those costs. The same
deterministic lifecycle adjustment applies to every draw, preserving the locked
per-draw `lifecycle - screening` identity.

The locked capacity-maintenance state matrix is unchanged. With a non-null
contract and `CapacityMaintenanceBasis.UNKNOWN`, settled screening NPV and its
table are available, while lifecycle remains exactly:

```text
available = false
status = "capacity_maintenance_unknown"
message = "Engineering capacity-maintenance basis is unknown."
distribution = null
lifecycle_cashflow_table = null
```

PC-D must not treat unknown lifecycle costs as zero merely because contract
settlement is available. With either active maintenance basis, lifecycle NPV and
its table use the deterministic adjustment above.

The settlement operates on the one selected, internally co-optimised strategy
total. It never floors DA, IDA, reserve, activation, imbalance, or another
sub-stream separately, and it never sums the floor with merchant cash.

## 5. Gross-base and prohibited dependency

No new merchant-revenue plumbing is required. PC-B already bootstraps
`StrategyRunResult.daily_realised_cash_series`, whose typed basis is post-VOM,
post-capture/liquidity/currency treatment, total EUR for the modelled MW, and
pre-shadow-wear/pre-lifecycle-cost. Therefore `A_d * w_t` is the only lawful
settlement reference.

PC-D must not import, call, adapt, or read outputs from either:

```text
compute_contracted_floor_overlay
compute_decaying_contracted_floor_overlay
```

Those functions accept the wear-net cycle-frontier baseline and produce
contract-window comparator PVs, not project NPV cash. A dependency/mutation test
must replace both functions with a raising sentinel while the PC-D calculation
still succeeds. The existing comparator remains available beside ProjectCase,
with its own provenance, but never enters `ContractCase`, `ProjectCase`, or
`RunResult`.

## 6. Distribution and representative cash-flow table

### 6.1 Distribution rule

Settlement runs on the full `draw x project-year` matrix before percentiles.
For each draw, compute the settled cash vector and both NPV draws; then apply the
existing NumPy `method="linear"` P10/P50/P90 and strict `NPV > 0` probability to
the resulting NPV arrays.

The following are prohibited:

- settling the floor on annual-draw P50;
- settling the floor on an existing representative cash-flow table;
- calculating `max(P50(M_t), F_t)`;
- applying one top-up computed from a mean/P50 merchant path to every draw.

`max` is nonlinear, so these shortcuts change P10/P50/P90 and probability.

### 6.2 P50-reconciled representative path

The v1 PC-B table uses annual-draw P50 because merchant-only NPV is a positive
affine function of the one annual draw. That identity does not survive a floor
kink. The statistic is an exact tagged matrix:

```text
contract_case = null
  -> p50_annual_bootstrap_draw_linear
  -> execute the existing PC-B table code path unchanged

contract_case != null
  -> p50_npv_rank_interpolated_cashflow_linear_v1
  -> execute only the algorithm below
```

This branch is semantic, not an optimisation. The null-contract branch must not
be algebraically rewritten to the interpolation form: floating-point operation
order can change pre-existing table values even when the real-number formulas
are equivalent.

For the non-null branch, PC-D preserves exact table-to-headline reconciliation
with the new statistic:

```text
p50_npv_rank_interpolated_cashflow_linear_v1
```

Let annual draws be sorted ascending, with original draw index as the stable
tie-break. For P50 and `n` draws:

```text
h = 0.5 * (n - 1)
i = floor(h)
j = ceil(h)
lambda = h - i
```

This ordering is valid because every `w_t >= 0`, every discount factor is
positive, and `w_1 = 1`: settled screening NPV is a non-decreasing function of
the one annual draw. A fully binding floor can create plateaus but cannot reverse
the order; original-index tie-breaking makes the displayed path deterministic.
The deterministic lifecycle adjustment preserves the same ordering.

Run the actual contract settlement for sorted scenarios `i` and `j`, then use
the same interpolation weight on every displayed yearly component:

```text
merchant_star[t]
  = (1-lambda) * M[i,t] + lambda * M[j,t]

settled_star[t]
  = (1-lambda) * S[i,t] + lambda * S[j,t]

top_up_star[t]
  = settled_star[t] - merchant_star[t]
```

The displayed effective floor remains the deterministic `F[t]` (or null outside
the term). `settled_star` need not equal `max(merchant_star, F)` because it is a
rank-interpolated reconciliation path across two already-settled scenarios. UI
and Excel must disclose that it is neither an actual scenario nor a per-year
median.

Because both NPV functions are linear sums of the already-settled yearly cash
and the lifecycle adjustment is deterministic:

```text
-CapEx + sum(discounted screening representative rows)
  == reported screening P50

-CapEx + sum(discounted lifecycle representative rows)
  == reported lifecycle P50
```

using the exact reconciliation check
`math.isclose(reconciled, reported_p50, rel_tol=1e-10, abs_tol=1e-6)`.
The check and constants are versioned as
`pc-cashflow-p50-reconciliation-v1`; failure is fail-closed for RunResult/export.
Provenance records the statistic literal, reconciliation version, lower/upper
sorted ranks, original draw indices, and interpolation weight.

### 6.3 Cash-flow row schema

Project Case v1.1 uses one replacement `CashflowRowV11` wire shape for both null
and non-null contracts. Its exact key set is:

```text
year
merchant_revenue_eur
effective_contract_floor_eur        # null outside term
contract_top_up_eur
revenue_eur                         # settled strategy cash
opex_eur
augmentation_eur
terminal_eur
net_eur
discount_factor
discounted_net_eur
```

For `contract_case = null`, `effective_contract_floor_eur` is null,
`contract_top_up_eur = 0`, and `merchant_revenue_eur == revenue_eur`. PC-D1/D2
must not emit the old row shape under the v1.1 schema or make any key optional.
`CashflowTable` retains the exact `{basis, rows}` envelope and the basis literal
is exactly `screening` or `lifecycle`; every row uses the v1.1 replacement shape.

## 7. Canonical schema and fingerprint migration

The CBOR encoding rules do not change, so the serialization profile remains:

```text
PC-CBOR-F64-v1
```

The business schema changes. Do not globally replace the current single
`SCHEMA_VERSION`, because that would change every unchanged producer-issued
`StrategyRunResult` fingerprint and its locked PC-A vectors. The production
fingerprint registry accepts exactly these current object/version pairs:

```text
StrategyRunResult -> project-case-v1
ProjectCase       -> project-case-v1.1
```

`RunResult` is not independently canonical-fingerprinted. It carries
`schema_version = "project-case-v1.1"` and the v1.1 ProjectCase input digest.
`ContractCase` likewise has no independent envelope; its bytes are tested only
inside a containing ProjectCase vector.

The typed production API chooses the schema version from the object type;
callers cannot attach an arbitrary version. Diagnostic `encode_value` remains a
low-level encoder and is not a schema validator. If legacy audit verification is
retained, it uses a separately named explicit
`verify_legacy_project_case_v1_fingerprint(payload, expected_digest)` path that
accepts only the old exact v1 ProjectCase key registry; it is never used to
fingerprint a current typed ProjectCase.

ProjectCase v1.1 payload keys are exactly:

```text
asset_case
lifecycle_case
market_case
valuation_case
bootstrap_case
contract_case             # present, null or the exact union map in §3.2
```

`ContractCase` is embedded in the ProjectCase payload; it is not independently
fingerprinted because there is no separate producer/issuer boundary. Source,
as-of, status, currency, curves, factors, start year, and fixed basis literals
all participate directly in `ProjectCase.input_fingerprint`.

The normative null/non-null containing-ProjectCase vectors are stored in
[`project-case-v1.1-fingerprint-vectors.json`](./project-case-v1.1-fingerprint-vectors.json).
For each vector, the encoded object is the exact envelope:

```text
{
  profile: "PC-CBOR-F64-v1",
  object_type: "ProjectCase",
  schema_version: "project-case-v1.1",
  payload: <the vector payload>
}
```

The checked-in file pins the full payload, encoded hex, and SHA-256 digest. It is
part of this design contract, not an implementation-generated fixture. PC-D1
must reproduce it independently and additionally:

- a regression proving the unchanged StrategyRunResult golden vectors and
  digests remain byte-identical;
- a regression proving null vs non-null, source/as-of, curve, entitlement factor,
  and start-year mutations change the ProjectCase fingerprint;
- rejection of omitted `contract_case`, aliases, unknown settlement kinds, and
  optional-key omission inside the tagged union.

## 8. RunResult, provenance, and export

The RunResult v1.1 top-level key set remains exactly:

```text
schema_version
input_fingerprint
no_lifecycle_cost_screening_npv
lifecycle_cash_npv
screening_cashflow_table
lifecycle_cashflow_table
provenance
```

`schema_version` is exactly `project-case-v1.1`; `input_fingerprint` is the
containing ProjectCase v1.1 digest. NPV outcome envelopes and the
available/table state matrix are unchanged; cash-flow rows use §6.3.

The exact provenance top-level key set is:

```text
calculator_version
project_case_input_fingerprint
strategy_run_fingerprint
project_case
strategy_run_result
bootstrap
projection
valuation
capacity_maintenance_basis
contract_settlement
cashflow_table_statistic
cashflow_reconciliation
red_line_assertions
```

`calculator_version` is exactly `pc-d2-v1.1`. All v1 nested maps retain their
exact schemas. `project_case` is the exact v1.1 payload; `strategy_run_result`
and its fingerprint remain v1.

`contract_settlement` is exactly:

```text
{
  basis: null |
    "ANNUAL_PRE_LIFECYCLE_STRATEGY_CASH_FLOOR_V1",
  algorithm_version: null |
    "pc-annual-pre-lifecycle-strategy-cash-floor-v1",
  resolved_floor_by_project_year:
    [{year: int, effective_floor_eur: float}, ...],
  representative_interpolation: null | {
    lower_sorted_rank: int,
    upper_sorted_rank: int,
    lower_original_draw_index: int,
    upper_original_draw_index: int,
    interpolation_weight: float
  }
}
```

For `contract_case = null`, `basis`, `algorithm_version`, and
`representative_interpolation` are null and the resolved-floor array is empty.
For a non-null contract they are non-null; floor records are sorted by year and
their year set equals `C` exactly. Sorted ranks and original indices are
zero-based integers in `[0, n_simulations-1]`, lower <= upper, and weight is in
`[0,1]` (zero when ranks equal).

`cashflow_table_statistic` follows the exact null/non-null matrix in §6.2.
`cashflow_reconciliation` is exactly:

```text
{
  version: "pc-cashflow-p50-reconciliation-v1",
  relative_tolerance: 1e-10,
  absolute_tolerance_eur: 1e-6
}
```

The exact `red_line_assertions` key set is:

```text
cash_npv_includes_shadow_wear: false
vom_rededucted: false
mw_rescaled: false
wear_net_floor_comparator_included: false
contract_settlement_included: bool
contract_settlement_basis: null |
  "ANNUAL_PRE_LIFECYCLE_STRATEGY_CASH_FLOOR_V1"
pre_tax_unlevered: true
tax_included: false
debt_included: false
financing_fees_included: false
```

The last two dynamic fields must agree with `contract_case` and
`contract_settlement` exactly. Unknown/extra/missing provenance or assertion
keys fail validation/export; “such as” extensions are not permitted under v1.1.

The exporter currently hard-gates the v1 ProjectCase/provenance key sets,
calculator version, and `floor_included = false`. Therefore a non-null contract
must remain export-unavailable until PC-D2 updates those exact gates and
recomputes the v1.1 fingerprints. It may not weaken validation to a substring
deny-list or accept arbitrary extra provenance.

The D1 checkpoint is not merged on its own. D2's atomic PR must migrate the
exporter for both `contract_case = null` and non-null before the v1.1 schema
reaches main; existing merchant-only Excel availability must never regress in an
intermediate public state.

The old wear-net comparator remains a separate sheet/panel and must never be
embedded in the ProjectCase assumptions or cash-flow sheets. The two products
need distinct titles and basis disclosures.

Before any non-null contract workbook is downloadable, D2 must visibly export:

- whole-project modelled MW and the fixed quote/asset-scope literals;
- every real rate, entitlement factor, effective `F[t]`, covered year, and term;
- base year and the user-asserted real-EUR curve basis;
- user-asserted quote status, source/as-of, and source-document hash;
- merchant, top-up, and settled representative cash by year;
- the rank-interpolated P50 disclosure and reconciliation result;
- the exact statement:

  `Annual whole-project strategy-cash floor before lifecycle costs; not MACSE,
  not a complete legal-contract model, and not a bankable valuation.`

D3 may improve interaction/presentation but may not defer these minimum D2
human-readable disclosures.

## 9. Validation and fail-closed states

Every numeric input and every derived `F`, `M`, `S`, `T`, cash-flow row, discount
factor, NPV draw, and percentile must be finite. Additional domains:

- floor rates: finite and `>= 0`;
- floor entitlement factors: finite and in `[0, 1]`;
- contract start year: strict integer in `[1, project_life_years]`;
- both arrays: same positive length and entirely inside project life;
- quote basis/frequency/asset scope: exact literals only;
- currency target base year: equal to `ValuationCase.base_year`;
- required source/status/as-of/document-hash null matrix: exact;
- modelled project MW: inherited from AssetCase and never independently supplied.

Unknown settlement kind, unsupported units (including EUR/MWh-year), partial
contracted MW, fractional tenor, missing curve years, extra curve years, NaN/Inf,
or a currency mismatch raises `ProjectCaseValidationError`. There is no silent
fallback to merchant-only, an old comparator, a zero floor, a lesser strategy,
or EUR 0.

PC-D1's deliberate pre-PC-D2 state is also exact and fail-closed:
`compute_project_case()` raises `ProjectCaseValidationError` with the stable
message `Non-null ContractCase requires the PC-D2 settlement calculator.` before
bootstrap or valuation. It emits no `RunResult`. A typed unavailable outcome is
not allowed because screening availability would be misrepresented as a market/
engineering failure. It must never emit a merchant-only result carrying a
non-null contract fingerprint.

## 10. Golden arithmetic vectors

These vectors pin settlement math independently of bootstrap implementation.
Vectors A–C state effective whole-project floor EUR directly for compactness;
Vector D uses the schema-native EUR/modelled-MW-year rate multiplied by
AssetCase MW and the entitlement factor.

### Vector A — kills `max(P50(M), F)`

```text
annual_draws = [0, 100]
projection = [1]
effective_floor = [60]
CapEx = 70
discount_rate = 0
```

```text
merchant draws = [0, 100]
settled draws = [60, 100]
screening NPV draws = [-10, 30]

P10 = -6
P50 = 10
P90 = 26
P(NPV > 0) = 0.5
```

The P50 reconciliation path has `merchant*=50`, `settled*=80`, `top_up*=30`,
and `-70 + 80 = 10`. The prohibited shortcut gives
`max(50, 60) - 70 = -10`.

### Vector B — term, merchant tail, projection, and lifecycle identity

```text
annual_draws = [40, 80, 120]
projection = [1, 0.5, 0.25]
effective_floor = [60, 50, absent]
CapEx = 10
discount_rate = 0
fixed O&M = 5 per year
augmentation net cash cost = 20 in year 2
terminal inflow = 7 in year 3
```

```text
settled matrix:
  A=40  -> [60, 50, 10]
  A=80  -> [80, 50, 20]
  A=120 -> [120, 60, 30]

screening NPV draws = [110, 140, 200]
P10/P50/P90 = 116 / 140 / 188

lifecycle adjustment = -5 + (-5-20) + (-5+7) = -28
lifecycle NPV draws = [82, 112, 172]
P10/P50/P90 = 88 / 112 / 160
```

Every draw satisfies `lifecycle - screening = -28`; both P50 tables reconcile.

### Vector C — absent floor is not zero floor

```text
annual_draw = -20
projection = [1, 1]
effective_floor = [0, absent]
CapEx = 0
discount_rate = 0
```

Correct settled cash is `[0, -20]`, top-up is `[20, 0]`, and NPV is `-20`.
Treating the absent second-year floor as zero incorrectly produces NPV `0`.

### Vector D — delayed start, MW, entitlement factor, and discounting

```text
annual_draw = 5
project life = 4 years
contract_start_project_year = 2
rate curve = [10, 20] EUR/modelled-MW-year
floor entitlement factors = [0.5, 1.0]
AssetCase.power_mw = 2
projection = [1, 1, 1, 1]
CapEx = 0
discount_rate = 0.10
```

The contract covers project years `{2,3}` with zero-based curve indices `{0,1}`:

```text
effective floor by project year = [absent, 10, 40, absent]
merchant cash = [5, 5, 5, 5]
settled cash = [5, 10, 40, 5]
top-up = [0, 5, 35, 0]

NPV = 5/(1.1)^1 + 10/(1.1)^2 + 40/(1.1)^3 + 5/(1.1)^4
    = 46.27757666826035
```

This vector fails if the start-year offset is wrong, MW/factor is missed or
applied twice, entitlement is multiplied into top-up after `max`, or contract
cash is left undiscounted. The three factor mutants produce, respectively:
missing factor -> year-2 cash `20`, NPV `54.542039478177706`; factor twice ->
year-2 cash `5`, NPV `42.14534526330168`; factor after `max` -> year-2 cash
`12.5`, NPV `48.34369237073969`.

## 11. Required test and mutation matrix

At minimum:

1. `contract_case = null` produces bit-identical PC-B economic numbers and
   pre-existing table columns, while the v1.1 row/payload shape and ProjectCase
   fingerprint intentionally differ.
2. Vector A/B/C/D known answers, including linear P10/P50/P90 and strict
   probability.
3. Floor always below merchant gives zero top-up and merchant-only numbers.
4. Negative merchant, explicit zero floor, and absent post-term floor remain
   distinct.
5. Settlement runs per draw/year; replacing it with P50/mean settlement fails.
6. Contract entitlement factor changes only `F`, exactly once; reserve availability
   is never reused.
7. Whole-project MW comes only from AssetCase; a partial tranche or independent
   MW input is unrepresentable.
8. Lifecycle costs occur after settlement; shadow wear remains absent; the
   per-draw lifecycle-minus-screening adjustment is constant.
9. Monkeypatch both old contracted-floor functions to raise; PC-D still works.
10. Source/as-of/status/currency/rate/factor/start mutations change fingerprint;
    StrategyRunResult fingerprint vectors stay unchanged.
11. Representative row fields reconcile arithmetically and their discounted sum
    equals the reported P50; nearest-draw, per-year-P50, and `max(P50,F)` mutants
    fail.
12. Overflow/non-finite derived values fail before a `RunResult` is produced.
13. Export rejects v1/v1.1 calculator, schema, provenance, or fingerprint aliases
    and never admits the wear-net comparator through the new contract keys.
14. UI/input changes invalidate the cached result before rendering or download;
    no stale floor-protected NPV survives a contract edit.
15. A non-null contract with `CapacityMaintenanceBasis.UNKNOWN` produces settled
    screening output but the exact locked lifecycle-unavailable envelope and
    null lifecycle table; no lifecycle cost is assumed zero.

## 12. Red lines

1. `max`, never add: `S = max(M, F)`, not `M + F`.
2. Settle every draw/year before percentiles; never settle a headline or table.
3. One selected producer-issued strategy total; never floor component streams.
4. No shadow wear, degradation proxy, or cycle-frontier net baseline.
5. Never call or consume the shipped wear-net floor overlay.
6. VOM, MW, capture, liquidity, FX, and reserve availability each remain counted
   exactly once at their existing owners.
7. Contract entitlement factors apply only to `F`, never to `M` or `T` after max.
8. Fixed O&M, augmentation, replacement, and terminal cash occur after settlement
   and are not protected by the floor.
9. Outside-term floor is absent, not zero.
10. No silent fallback on invalid/unsupported contract data.
11. No MACSE/tolling/collar/revenue-share/bankable claim.
12. Schema v1.1 changes the ProjectCase input fingerprint and RunResult
    schema/provenance intentionally but does not rewrite unchanged
    StrategyRunResult fingerprints; RunResult has no independent fingerprint.
13. Wear-net comparator remains a separate sibling and is never nested in
    ProjectCase provenance or cash flow.
14. The P50 representative path is disclosed as rank-interpolated, not an actual
    draw or per-year median.

## 13. Candidate review gate

This candidate is lockable only when Codex and Gemini both approve the same hash
under the Project Case §11 blocking bar, and CC independently verifies:

- all arithmetic vectors, including percentile interpolation;
- canonical payload bytes/digests for null and non-null contracts;
- unchanged StrategyRunResult bytes/digests;
- exact code anchors for the producer gross-of-lifecycle cash basis;
- the exporter/version-gate claims;
- the old contracted-floor functions are absent from the PC-D dependency graph.

Any finding that can change a cash number/basis, eligible draw/year, availability
state, public tagged union, canonical payload/fingerprint, or a red-line blocks
the lock. Editorial wording that cannot change those outcomes is non-blocking and
may be resolved in the implementation PR.
