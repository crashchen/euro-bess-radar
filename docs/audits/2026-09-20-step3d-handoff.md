# Step 3D — disclose DST capacity settlement conventions

Step 3C was merged with the user's explicit authorization: PR #91, ordinary
merge `2e4ed73d7357e4d556373f6019198045f8fc8d04`. Step 3D is a separate draft
for independent review. Do not merge it automatically or invoke CC.

Base: `2e4ed73d7357e4d556373f6019198045f8fc8d04`.
Code: `252a9ab9405f3b5fc2bde357d6f3cbf66a94ff26`.
Later commits contain review evidence/documentation only. Remote CI must be
checked against the latest PR head; local test results below are not CI results.

## Problem and final behavior

On a German spring DST day, the existing registered Project Case model pays
six nominal 4h capacity blocks, while the screening model pays 23 physical
hours. The results previously offered no shared, scoped explanation. This
change discloses both conventions and retains their numbers.

- A shared display helper identifies the recorded zone/product/adapter/grid
  profile. Only DE_LU with the current Project Case v1 reserve profile asserts
  the six nominal 4h blocks. Unknown reserve profiles are visibly unverified;
  DA-only and DA/IDA-only adapters have no capacity basis claim. Product names
  come from the result, including qualified names; no new market whitelist.
- Project Case page, its Cockpit mirror and standalone/appended XLSX NPVs sheet
  use the immutable RunResult provenance. Current sidebar inputs cannot relabel
  a saved result. The original Assumptions & Provenance tree, JSON schema and
  fingerprint are unchanged.
- Cockpit capacity-bearing co-opt, triple ceiling, reserve-first and reserve-mode
  stochastic rows carry short Capacity basis / Capacity scope fields. Full text
  and assumptions are frozen with the result. DA/IDA rows say Not applicable;
  zero/negative capacity outputs remain eligible, unavailable rows do not acquire
  false claims. A v2 panel ID invalidates old session bundles missing disclosure.
- Revenue joint MILP identifies the capacity/mixed products that actually supply
  its aggregate price. The page and ordinary Market Report XLSX/PDF share this
  captured text; energy-only products are excluded. Older export callers without
  product metadata explicitly disclose that identity is unavailable.
- Standalone annual ancillary estimates (8760h/year), activation/imbalance energy
  overlays, and DA/IDA single-day/multi-day/frontier results are not assigned the
  joint solver's physical-hour capacity basis. Market Report PDF has no Project
  Case result parameter; it does not pretend to export that result's basis.

These statements describe implemented models, not verified market billing or
legal rules. All physical energy/SoC timing and cash algorithms stay unchanged.
See the [duration contract](../design/delivery-duration-v1.md).

## Frozen source and tests

[Patch](2026-09-20-step3d-evidence/step3d-code.patch), scope
`src/ tests/ .github/workflows/ci.yml`, SHA-256:

`25d88ed3f3e2c2b58a58631b110c17a461026a4067aa3c573715897798f038f5`.

```sh
git diff --binary --full-index 2e4ed73 252a9ab -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-20-step3d-evidence/step3d-code.patch
.venv/bin/python -m pytest tests/test_step3d_*.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
git diff --check 2e4ed73 HEAD
```

All nine historical patches match their original handoff hashes;
[references and actual hashes](2026-09-20-step3d-evidence/contracts/historical-patches.json).
No old patch or handoff was rewritten.

The same four new test files collect 82 cases on clean base and head:

| Test module | Base fail | Base pass | Head pass |
|---|---:|---:|---:|
| settlement_basis | 33 | 8 | 41 |
| project_case_disclosure | 6 | 4 | 10 |
| cockpit_disclosure | 19 | 2 | 21 |
| export_disclosure | 6 | 4 | 10 |
| Total | 64 | 18 | 82 |

There are no base collection errors or fixture/argument TypeErrors. The test
factory adapts only the new primary_zone argument when running against base;
new-symbol imports happen inside the relevant tests. The 18 compatibility
controls cover existing cash answers, numeric export fields, no-result/failed
solver behavior and DA-only paths. [Names](2026-09-20-step3d-evidence/contracts/compatibility-case-names.txt),
[red-test log](2026-09-20-step3d-evidence/contracts/all-baseline-tests.txt),
[failure classification](2026-09-20-step3d-evidence/contracts/all-baseline-review.json).

To repeat the clean baseline:

```sh
repo_root="$PWD"
baseline_dir=$(mktemp -d)
git archive 2e4ed73 | tar -x -C "$baseline_dir"
cp tests/test_step3d_*.py "$baseline_dir/tests/"
(cd "$baseline_dir" && PYTHONPATH=. "$repo_root/.venv/bin/python" -m pytest tests/test_step3d_*.py -q)
```

Local full suite: **2057 passed / 2 skipped**, 2059 collected, including all
35 slow cases, 332.73 seconds, Python 3.13.9 / Streamlit 1.55.0.
The final product source was present throughout; baseline-compatible test
factory plumbing was finalized during that run, then all final 82 cases were
rerun successfully (2.32s). Ruff and diff check pass.
[Full log](2026-09-20-step3d-evidence/logs/full-suite.txt),
[final new-test log](2026-09-20-step3d-evidence/logs/new-tests.txt),
[slow collection](2026-09-20-step3d-evidence/logs/slow-collected.txt).
The two opt-in chart-render tests remain skipped. The separate actual PDF
rendering below does not turn those skips into passes. CI workflow/check names
are unchanged: test plus Python 3.11 / Streamlit 1.55.0 compatibility check,
not a minimum version matrix for every dependency.

## Independent numerical and scope checks

The [same probe](2026-09-20-step3d-evidence/contracts/probe.py) was run on a
clean base archive and current code: the entire JSON is identical, including
unavailable routes. Synthetic Regelleistung block XLSX is parsed by production
code, then fed to real public Project Case adapters and screening solvers.
1 MW × EUR 20/MW/h × 95% availability gives:

| Local day | Physical hours | Project Case | Screening capacity |
|---|---:|---:|---:|
| Ordinary | 24 | 456 | 456 |
| Spring DST | 23 | 456 | 437 |
| Autumn DST | 25 | 456 | 475 |

[Base output](2026-09-20-step3d-evidence/contracts/probe-base.json),
[head output](2026-09-20-step3d-evidence/contracts/probe.json),
[comparison](2026-09-20-step3d-evidence/contracts/probe-comparison.json).
Run: `PYTHONPATH=. .venv/bin/python docs/audits/2026-09-20-step3d-evidence/contracts/probe.py`.

The 2025 spring triple path correctly remains unavailable because native DA
and IDA grids differ. Sequential/triple known answers use valid post-cutover
grids. The short-history stochastic DST probe is unavailable; it is **not**
numerical acceptance for that route. Its disclosure is supported by the
unchanged physical-hour solver call chain and display tests, not invented cash.

[Scope audit](2026-09-20-step3d-evidence/contracts/scope-review.md): 51 of the
55 pre-existing src files are byte-identical, including every one of the 12
Project Case core files, analytics, dispatch, simulation, time utilities,
strategy comparison and assumptions. Four existing presentation/export files
changed, plus one new display helper. No DA/IDA guard, SoC, segment neutrality,
576 limit, FEC, future-information visibility, schema, fingerprint or cash rule
changed. The complete before/after hashes are in the adjacent scope JSON.

## Actual presentation checks and limits

- [Browser evidence](2026-09-20-step3d-evidence/browser/README.md): production
  PC, mirror, Cockpit and Revenue joint renderers at 1280×900 (sidebar expanded)
  and 390×844 (sidebar overlay closed). New captions fit; comparison basis/scope
  can be reached by horizontal scrolling. Actual Cockpit download clicks retain
  the visible result. Browser solver calls were not instrumented.
- [Export evidence](2026-09-20-step3d-evidence/exports/README.md): five actual
  XLSX and two PDFs, complete numeric cell type/format inspection, six workbook
  range renders and four Poppler-rendered PDF pages. Added disclosures are
  readable. XLSX rendering uses Artifact Tool, not a native Excel screenshot.
- True DE aggregate FCR+aFRR Up cash is EUR 655.50/day and 239421.375/year;
  FI FCR-N is EUR 437/day and 159614.25/year, without the German nominal claim.
  PC remains EUR 456 and keeps its DE basis when appended to an FI market report.
  Cockpit presentation fixtures explicitly stub solver values, preserving numeric
  negative delta -3 and missing uplift; they do not prove those solver outputs.
- Existing long Cockpit XLSX strategy labels still clip. Existing Revenue
  desktop reserve-percentage metric clipping and mobile chart density remain.
  These are recorded for Step 4 prioritization; the acceptance claim concerns
  added disclosures, not every amount, product string or chart in the app.
- No live market data or production cache is accessed by the harnesses/probes.
  README/Vault reconciliation, deprecated-warning cleanup, frontier content
  fingerprint improvements and remaining-page layout work are not completed here.
  Manual smoke items 39–43 are a reproducible checklist, not a claim every
  interactive permutation was executed.

## Review request

Independently verify the exact head, frozen patch, clean-baseline failures,
known answers, immutable-result context, row classification, real exports and
CI. Confirm that the disclosure does not change either economic convention.
Return a pass / needs-changes verdict before any merge. Step 4 follows only
after the user-directed review sequence; CC was not invoked automatically.
