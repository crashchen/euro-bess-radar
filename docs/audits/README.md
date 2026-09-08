# Replay review evidence — 2026-09-08

These checked-in records contain synthetic fixtures, frozen source/test patches,
baseline failures and validation logs. Commands run from the repository root
after installing its requirements. No live market data or workstation project
notes are included. Local absolute paths in logs are replaced by `<repo>` or
`<baseline-repo>` and presentation-only trailing whitespace is removed;
assertions and test results are retained. Frozen patches retain their exact
bytes, including the space prefix on blank context lines.

## Review stages

| Stage | Source reference | Record |
|---|---|---|
| Step 1 | `8f9eab7` → `3823acd`; merged in [#86](https://github.com/crashchen/euro-bess-radar/pull/86) as `850343d` | [Specification](2026-09-08-step1-spec.md), [acceptance and evidence](2026-09-08-step1-handoff.md) |
| Step 1b, first review | `3823acd` → `0b5d718`; 1747 passed / 2 skipped locally and independently reproduced by the user's reviewer | [Frozen historical handoff](2026-09-08-step1b-handoff.md) |
| Step 1b, singleton correction | Increment relative to `0b5d718`, in [#87](https://github.com/crashchen/euro-bess-radar/pull/87), now targeting main | [Current handoff](2026-09-08-step1b-r2-handoff.md), [PR description](2026-09-08-step1b-pr.md) |
| Step 2 | Increment relative to `c46aaf9`, stacked on #87 | [Duration contract](../design/delivery-duration-v1.md), [handoff and evidence](2026-09-08-step2-handoff.md) |

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

Step 2 is implemented for the bounded native DA contract, FI physical capacity
cash, forecast comparator coverage, physical-time moving average and the UI
runtime floor. It is awaiting the user's combined external review with #87.

1. Step 3: retain replay results across unrelated reruns using the existing
   fingerprint pattern; make monetary KPIs and Project Case quantiles readable
   at desktop and mobile widths; clarify DST capacity settlement conventions.
2. Step 4: reconcile README, CLAUDE.md, validation instructions and project notes
   against verified behavior. The current publication repairs the audit evidence
   trail; it is not a claim that all documentation or Vault housekeeping is done.

External reviewer results here were supplied by the user. This task did not
invoke or send code to CC or another external reviewer.
