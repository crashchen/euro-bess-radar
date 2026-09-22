# September review evidence

These checked-in records contain synthetic fixtures, frozen source/test patches,
baseline failures and validation logs. Commands run from the repository root
after installing its requirements. No live market data or workstation project
notes are included. Local absolute paths in logs are replaced by `<repo>` or
`<baseline-repo>` and presentation-only trailing whitespace is removed;
assertions and test results are retained. Frozen patches retain their exact
bytes, including the space prefix on blank context lines; each evidence
directory carries a `.gitattributes` so `git diff --check` does not report
those preserved bytes as whitespace errors.

The [current verification snapshot](../validation/current.md) is the single
entry point for today's code-bound results. The records below remain historical.

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
| Step 3D | Independently reviewed; merged in [#92](https://github.com/crashchen/euro-bess-radar/pull/92) as `fb72dbf` | [Handoff and evidence](2026-09-20-step3d-handoff.md), [duration contract](../design/delivery-duration-v1.md) |
| Step 4 | Documentation based on `fb72dbf`; merged in [#93](https://github.com/crashchen/euro-bess-radar/pull/93) as `6d986e3` | [Housekeeping handoff](2026-09-20-step4-handoff.md), [current snapshot](../validation/current.md), [local-note sync receipt](2026-09-20-step4-evidence/vault-sync.json) |
| Frontier follow-up | Increment from `6d986e3`; implemented on the review branch, awaiting review | [Result identity and export-snapshot handoff](2026-09-22-frontier-handoff.md), [panel contract](../design/cycle-cap-frontier-v1.md#session-identity-and-export-assumptions--2026-09-22-follow-up) |

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

## Current sequence

Steps 1, 1b, 2, 3A–3D and Step 4 are merged. Step 4 reconciled repository
documentation and the nine project notes against the [dated snapshot](../validation/current.md),
retaining the earlier records unchanged. The frontier follow-up now extends
content identity and freezes run-time export assumptions; it awaits independent
review, with separate revision-bound evidence and unchanged numerical models.

[Remaining work](../validation/follow-ups.md) distinguishes the frontier change
awaiting review from remaining-page layouts, existing export clipping, warnings,
bounded floor-export/retry behavior and unexecuted manual/consumer checks. A reconciled checklist is not an assertion
that every check passed. Private Vault note text and workstation paths stay
outside this repository; the Step 4 handoff records relative synchronization scope.

External reviewer results here were supplied by the user. This task did not
invoke or send code to CC or another external reviewer.
