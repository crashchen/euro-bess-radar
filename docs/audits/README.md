# Replay review evidence — 2026-09-08

These checked-in records contain synthetic fixtures, frozen source/test patches,
baseline failures and validation logs. Commands run from the repository root
after installing its requirements. No live market data or workstation project
notes are included. Local absolute paths in logs are replaced by `<repo>` or
`<baseline-repo>` and presentation-only trailing whitespace is removed;
assertions and test results are retained. Frozen patches retain their exact
bytes, including the space prefix on blank context lines; each evidence
directory carries a `.gitattributes` so `git diff --check` does not report
those preserved bytes as whitespace errors.

## Review stages

| Stage | Source reference | Record |
|---|---|---|
| Step 1 | `8f9eab7` → `3823acd`; merged in [#86](https://github.com/crashchen/euro-bess-radar/pull/86) as `850343d` | [Specification](2026-09-08-step1-spec.md), [acceptance and evidence](2026-09-08-step1-handoff.md) |
| Step 1b, first review | `3823acd` → `0b5d718`; 1747 passed / 2 skipped locally and independently reproduced by the user's reviewer | [Frozen historical handoff](2026-09-08-step1b-handoff.md) |
| Step 1b, singleton correction | Increment relative to `0b5d718`; merged in [#87](https://github.com/crashchen/euro-bess-radar/pull/87) as `140fed5` | [Current handoff](2026-09-08-step1b-r2-handoff.md), [PR description](2026-09-08-step1b-pr.md) |
| Step 2 | Increment relative to `c46aaf9`, stacked on #87; merged in [#88](https://github.com/crashchen/euro-bess-radar/pull/88) as `95f6d09` | [Duration contract](../design/delivery-duration-v1.md), [handoff and evidence](2026-09-08-step2-handoff.md) |
| Step 3A | Increment relative to `95f6d09`; merged in [#89](https://github.com/crashchen/euro-bess-radar/pull/89) as `e60a74f` | [Handoff and evidence](2026-09-15-step3a-handoff.md) |
| Step 3B | Increment relative to `e60a74f`; merged in [#90](https://github.com/crashchen/euro-bess-radar/pull/90) as `cf91374` | [Handoff and evidence](2026-09-16-step3b-handoff.md) |
| Step 3C | Both increments independently reviewed; merged in [#91](https://github.com/crashchen/euro-bess-radar/pull/91) as `2e4ed73` | [Original frozen handoff](2026-09-16-step3c-handoff.md), [current increment and evidence](2026-09-19-step3c-r2-handoff.md), [duration contract](../design/delivery-duration-v1.md) |
| Step 3D | `2e4ed73` → `252a9ab`; implementation complete, draft review pending | [Handoff and evidence](2026-09-20-step3d-handoff.md), [duration contract](../design/delivery-duration-v1.md) |

The original Step 1b patch is unchanged: its SHA-256 remains
`a5a8f6adb1ca308b8bed9b95242b8c52cc48343bb4a14cbf2e3344278ed40f12`.
The later correction has a separate patch and baseline. Documents labelled
historical retain their original validation counts and limitations; use the
[guard contract](../design/replay-input-guards.md) and
[duration contract](../design/delivery-duration-v1.md) for the current rules.

## Portable checks

```sh
shasum -a 256 docs/audits/2026-09-08-evidence/*source.patch
PYTHONPATH=. .venv/bin/python docs/audits/2026-09-08-evidence/step1-residual-probe.py
PYTHONPATH=. .venv/bin/python docs/audits/2026-09-08-evidence/step1b-r2-probe.py
PYTHONPATH=. .venv/bin/python docs/audits/2026-09-15-step3a-evidence/step3a-probe.py
PYTHONPATH=. .venv/bin/python docs/audits/2026-09-16-step3b-evidence/step3b-probe.py
.venv/bin/python -m pytest tests/test_market_grid_guards.py -q
.venv/bin/python -m pytest tests/ -q
```

The residual probe calls the five public replay paths with real local solvers;
its wrappers only record calls. The singleton probe compares a missing day with
one extreme quote on that day. Both probes use synthetic data and do not fetch
prices. Archived outputs describe their recorded revision, not a new run.
The two opt-in chart-render skips do not establish PDF rendering success.
Remote CI status is attached to each PR's exact head in GitHub Checks.

## Remaining optimization sequence

Steps 1, 1b, 2, 3A, 3B and 3C are merged. Step 2 delivered the bounded native DA
contract, FI physical capacity cash, forecast comparator coverage, the
physical-time moving average and the UI runtime floor, and was externally
reviewed with #87 before #88 merged. Step 3 is split into four separately reviewed increments.

1. Step 3A (merged in #89): show an explained `n/a` wherever a physical
   duration cannot be verified, name the compatibility CI job accurately,
   replace the market page's bare index exception with a visible diagnostic,
   and record the reserve-average-power uniform-grid premise.
2. Step 3B (merged in #90): retain batch replay and forecast results across
   unrelated reruns behind a content fingerprint, marking stale results and
   blocking their downloads.
3. Step 3C (merged in #91): put the overall average price on
   the shared duration-weighted basis across all four consumers, and make key
   monetary values and quantiles readable at desktop and mobile widths.
4. Step 3D (implementation under review): disclose the zone/product-qualified DST capacity
   settlement basis on the pages, the strategy table and the exports,
   retaining both existing sets of numbers.
5. Step 4 (deferred): reconcile README, CLAUDE.md, validation instructions and
   project notes against verified behavior, against one dated verification snapshot.
   The current publication repairs the audit evidence trail; it is not a claim
   that all documentation or Vault housekeeping is done.

External reviewer results here were supplied by the user. This task did not
invoke or send code to CC or another external reviewer.
