# Cockpit export provenance evidence

This directory binds the multi-day/forecast-policy export-only correction to
code commit `e39968dd93eb72f8c792c11049035f2210b8c5eb`, based on the #95
merge `cc6b1e24c527d5eec5454ffbdb2cfcc07cabaff0`.

- `code.patch` is the frozen `src/ tests/ .github/workflows/ci.yml` diff. Its
  SHA-256 is `b295b4bf8e648c3e262d375f6cbc92a52001af87c456e7235539185c90b051ae`.
- `baseline-tests.txt` is the new six-case test file on a clean `git archive`
  of `cc6b1e2`: 5 assertion failures / 1 pass, with no collection failure.
- `targeted-tests.txt` is the related non-slow candidate run: 106 passed,
  6 deselected.
- `full-suite.txt` is the candidate full run: 2111 passed / 2 skipped,
  including all 35 slow cases.
- `ruff.txt` records the complete `src/ app.py tests/` lint result.
- `probe.py` constructs synthetic hourly DA, IDA and sidebar assumptions. Run
  it on both revisions to reproduce `baseline-probe.json` and
  `candidate-probe.json`. Their diff changes only panel export-assumption rows:
  the two replay modes keep the same gross revenue, wear cost and valid-day
  count; forecast comparison values and valid days are identical; global
  sidebar rows are identical.

Recreate the baseline without modifying the working checkout:

```sh
baseline_dir=$(mktemp -d /tmp/euro-bess-cockpit-baseline-XXXXXX)
repo_dir=$(pwd)
git archive cc6b1e2 | tar -x -C "$baseline_dir"
cp tests/test_cockpit_export_provenance.py "$baseline_dir/tests/"
(cd "$baseline_dir" && PYTHONPATH=. "$repo_dir/.venv/bin/python" -m pytest tests/test_cockpit_export_provenance.py -q)
```

The probe is read-only and creates no market-data cache. Logs replace workstation paths
with `<repo>` or `<baseline-repo>`; the frozen patch retains its exact bytes.
Tests inspect saved XLSX cells with openpyxl, not a browser download or
native Excel rendering. CI must be checked against the PR's exact head.
