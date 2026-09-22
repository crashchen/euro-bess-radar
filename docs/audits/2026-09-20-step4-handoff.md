# Step 4: current documentation and Vault housekeeping

## Result and scope

Step 3D [#92](https://github.com/crashchen/euro-bess-radar/pull/92) was ordinarily
merged, with the user's authorization, as
`fb72dbf529e9e06ba1716aa66120b2f4f68fe4de`. Step 4 is a separate documentation
branch for review; it has not been merged. No further behavior work is included.

The [current verification snapshot](../validation/current.md) is the shared
entry point for README, agent guidance and the nine project notes. It binds
results to the merged **code** head, separate from documentation commit
`434beba35125c22e4f7dc0abaff93072d8b5ad4b`, which first published that snapshot.
Later documentation commits add the synchronization receipt and this handoff.

README, CLAUDE, CONTRIBUTING, the audit index, affected model-contract prose and
the manual checklist now describe the implemented Step 1–3D behavior. This
includes duration-weighted averages, row-based median/std, unavailable values,
replay guards, sparse simplified-IDA days, both DST settlement conventions and
persisted multi-day/forecast results. Obsolete current-status passages were
corrected in place; dated handoffs and frozen patches were retained.

Important distinctions preserved:

- Multi-day/forecast can restore the stored result when inputs return to their
  original values. Project Case deletes a stale cache and requires another Run.
- Frontier's older fingerprint uses the first/last selected local days and the
  count of selected days. Same-length data corrections remain a separate fix.
- Continuous replay carries segment-end SoC while enforcing terminal neutrality
  at each segment end. No model, duration or settlement rule changed here.
- The manual procedure now has 47 numbered checks. Added JSON checks 44–47 are
  **unexecuted**, not another passed test suite. The producer/consumer controls,
  real-base-EUR/CPI boundary and already-settled annual cash semantics were
  reconciled against code, without editing or running ESS.

## Fresh verification

The full local run on the merged code returned **2057 passed / 2 skipped**,
2059 collected, including all 35 slow cases, in **308.86 seconds**. Inventory:
56 test files, 1545 AST test-function definitions and 1545 collected callables.
The corrected `TestDailySpreads` selector independently returned 8 passed.
Ruff passed. See [logs and environment](2026-09-20-step4-evidence/README.md).

The 2022 passed / 2 skipped non-slow result is derived from the full run, not
another executed fast command. The two chart-render tests remain opt-in skips.
The run emitted 23 Streamlit attrs warnings and 6 pandas concat FutureWarnings;
the historical NumPy scalar-conversion warning did not recur. These sources
were not changed.

Both jobs passed on the exact merged code head in
[main CI run 35506713370](https://github.com/crashchen/euro-bess-radar/actions/runs/35506713370).
That record is **not** a claim about the later documentation PR's CI; its exact
head status is available in GitHub Checks. The compatibility job pins Python
3.11 and Streamlit 1.55.0 only, not every package's lower bound.

All **123 tracked runtime/source/test/script/dependency/CI files** remain
byte-identical to the base ([file hashes](2026-09-20-step4-evidence/runtime-scope.json)).
All **ten historical frozen patches** retain their hashes
([inventory](2026-09-20-step4-evidence/historical-patches.json)). No product-code
patch or new baseline red suite is needed for a documentation-only change.

## Vault synchronization

Nine local notes were updated and their saved bytes checked:

| Relative note | Reconciliation |
|---|---|
| README.md | Shared verification entry and current reviewer/synchronization ownership |
| 概览.md | Implemented features, architecture and explicit evidence boundaries |
| 路线图.md | Completed audit stages separated from the actual remaining work |
| 迭代记录.md | Dated September entry; earlier implementation/review history retained |
| 激活电量 live fetcher scope.md | Historical scoping distinguished from implemented live fetchers |
| 测试分布.md | 56-file inventory rebuilt from this collection, separating definitions and cases |
| 操作指南.md | Implemented controls, persisted results, stale behavior and manual boundaries |
| 模型语义与可靠性合同.md | Duration, settlement, solver and cache semantics reconciled |
| 导入模板与口径红线.md | Current imports and Radar/ESS cash handoff responsibilities clarified |

[Public synchronization receipt](2026-09-20-step4-evidence/vault-sync.json) records
scope and method. All nine original hashes were checked before writing. Original
frontmatter was preserved except `docs_updated: 2026-09-20`; dated history was
retained. New validation links pin the already-pushed documentation commit
`434beba`, so they do not point at files absent from main while review is pending.

Vault full text, absolute paths, before/after hashes and backups remain local.
No Vault commit/push, Obsidian CLI or GUI operation occurred. The public receipt
cannot independently prove private note content; an authorized local reviewer
can inspect the notes and the local originals/diff.

## Review and remaining boundaries

This work did not launch a browser, fetch live market data or execute the ESS
consumer. Existing synthetic browser/export evidence retains its original
revision and limits. ESS source was read at `e7cdac0`; its matching golden JSON
fixture was compared, but no ESS file or wire schema/digest changed.

The [remaining-work list](../validation/follow-ups.md) records frontier content
fingerprints, remaining-page layout checks, existing Cockpit XLSX name clipping,
manual/consumer acceptance, warning cleanup and bounded future model work. The
layout counts are static metric call sites, not simultaneously visible cards.
The two non-blocking Step 3D observations retain their scope: direct version
assertions could improve diagnostics despite existing v2-drift protection;
explicit-null raw mappings are rejected earlier on validated production paths.

Independent review can start with these checks from the repository root:

```sh
git diff fb72dbf -- src/ tests/ scripts/ app.py requirements.txt pyproject.toml .github/workflows/ci.yml
git diff --check fb72dbf
.venv/bin/python docs/audits/2026-09-20-step4-evidence/check_docs.py
```

The first command should be empty. The Markdown checker verifies local paths,
headings and checklist sequence, not external URLs or application behavior.
[Its recorded result](2026-09-20-step4-evidence/doc-link-check.json) and the
[evidence manifest](2026-09-20-step4-evidence/manifest.json) support file review.
Use the dated snapshot's reproduction commands for an independent test run.
Review private Vault changes locally; no note text is bundled in this PR.

External CC invocation remains the user's responsibility. This branch is kept
as a draft for review and will not be automatically merged.
