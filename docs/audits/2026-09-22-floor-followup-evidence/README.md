# Frontier/floor export and retry evidence

All fixtures are synthetic; no market API, production cache, Vault file or ESS
consumer was changed. The code baseline is the ordinary #94 merge
`bd4bb888775fb775b61de10dc6715b9673b1401c`; the tested code commit is
`886b2d5` (see the handoff for its full SHA). Later documentation commits do
not change `src/`, `tests/` or CI configuration.
Workstation paths and presentation-only trailing whitespace in logs are
normalized; failure assertions and test outcomes are retained.

The frozen [code patch](code.patch) covers only `src/`, `tests/` and
`.github/workflows/ci.yml`. Its SHA-256 is
`d378f1571698bda7cd767dd221adb5fcf7cf19a100da639b74ee9170c3cc333f`.
The `.gitattributes` entry preserves its binary/full-index bytes without
turning patch-context whitespace into a `git diff --check` warning.

[Baseline log](baseline-tests.txt): copy the new test file onto a clean
`git archive bd4bb88` checkout and run it with the repository virtualenv.
Four assertions fail for the export/retry behaviors; one old-session gate
passes. There are no import or collection errors.

[Targeted log](targeted-tests.txt): five frontier/floor-related suites pass on
the tested code. The AppTest harness calls the real panels and solvers, then
reads the actual generated workbook bytes with openpyxl. It verifies the
frontier and floor assumptions, numerical cell type, immutable floor snapshot,
fresh snapshot after an explicit Run, failed-retry state, and old-session
version gates. This is saved-file inspection, not a browser download or native
Excel render. [Full-suite log](full-suite.txt) records **2105 passed / 2
skipped**, 2107 collected, all 35 slow cases and 29 existing warnings in
308.83 seconds. [Ruff log](ruff.txt) records the lint gate. Remote CI belongs
to the PR's exact head.

Reproduce from the repository root:

```sh
git diff --binary --full-index bd4bb88..886b2d5 -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-22-floor-followup-evidence/code.patch
.venv/bin/python -m pytest tests/test_floor_frontier_followups.py tests/test_frontier_result_identity.py tests/test_simulation_cockpit_contracted_floor.py tests/test_simulation_cockpit_frontier.py tests/test_cycle_frontier.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
```
