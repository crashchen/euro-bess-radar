# Frontier content-identity evidence

Read the [handoff](../2026-09-22-frontier-handoff.md) first. Code base `6d986e3`,
tested code `e11eaca`; documentation commits do not change the tested code.

- `frontier-code.patch`: exact binary/full-index diff for src/tests/CI.
  `.gitattributes` disables whitespace diagnostics only for frozen patch files.
- `baseline-tests.txt`: the same new 43-case file on clean `git archive` base:
  25 failed / 18 passed. `targeted-tests.txt`: 150 passed on the code head.
- `full-suite.txt`, `full-suite.xml`: fresh complete run, including all slow
  cases. Absolute roots and JUnit hostname are normalized, with XML escaping
  preserved; no assertion or outcome is rewritten.
- `test-inventory.json`, `run-results.json`: collection/outcome artifacts from
  the unchanged [read-only plugin](../2026-09-20-step4-evidence/snapshot_plugin.py).
- `verification.json`, `runtime.json`, `warning-summary.json`, `ruff.txt`: exact
  results, compatibility case names, environment and observed warning classes.
- `scope.json`, `cache-check.json`: unchanged core/runtime files, ten historical
  frozen patches, and privately checked real-cache file set/content hashes.
- `browser-harness.py`, `browser-check.json`: real production panels/solvers with
  synthetic data. Browser observations are transcribed tool results; download
  completion and layout/render acceptance are explicitly not certified.

Recreate the full collection/outcome artifacts into a temporary directory:

```sh
STEP4_OUTPUT=$(mktemp -d)
export STEP4_OUTPUT
PYTHONPATH=docs/audits/2026-09-20-step4-evidence:. .venv/bin/python -m pytest tests/ -q -p snapshot_plugin --junitxml="$STEP4_OUTPUT/full-suite.xml"
```

The private Vault before/after copies stay outside this repository. A later
`vault-sync.json` records only relative scope. `manifest.json` records artifact
hashes excluding itself, not a public proof of private note text.
