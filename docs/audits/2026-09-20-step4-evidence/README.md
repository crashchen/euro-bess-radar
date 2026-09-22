# Step 4 verification evidence

Code baseline: `fb72dbf529e9e06ba1716aa66120b2f4f68fe4de`, 2026-09-20.
The [current snapshot](../../validation/current.md) explains the results and
limits. This directory records a fresh run, not copied Step 3D results.

- `full-suite.txt` / `.xml`: actual full run with all slow cases. Private root
  paths are normalized and the workstation hostname is removed from JUnit.
- `snapshot_plugin.py`: read-only pytest collection/outcome recorder used by
  that run. `test-inventory.json` separates AST definitions, collected callables
  and parameterized cases; `run-results.json` records outcomes and warnings.
- `focused-command.txt`: the executing agent's recorded output for the corrected
  TestDailySpreads command, transcribed from its tool result, not a second run.
- `runtime.json`, `ruff.txt`, `main-ci.json`: actual environment, lint and exact
  merged-head CI observations. `warning-summary.json` groups the observed
  warnings without claiming any dependency fix.
- `runtime-scope.json`: every tracked app/source/test/script/dependency/CI file
  is byte-identical to the base. `historical-patches.json`: ten frozen patches
  retain their hashes. No source/test/CI patch exists for this documentation PR.
- `layout-inventory.json`: static metric call sites on the four remaining pages;
  not a count of simultaneous cards and not visual acceptance.
- `check_docs.py` / `doc-link-check.json`: local Markdown file/heading links and
  the 47 sequential checklist IDs. Run the checker from the repository root;
  `STEP4_OUTPUT` may choose an output directory, otherwise the system temporary
  directory is used. It does not check internet URLs or launch an application.

Full note text and workstation paths are excluded. The [Step 4 handoff](../2026-09-20-step4-handoff.md) and `vault-sync.json`
record the relative Vault synchronization scope; local before/after copies
are kept outside this repository. No ESS files, schemas or digests change.

`manifest.json` freezes the verification artifacts by SHA-256 and excludes itself.
