# Cockpit strategy-name export evidence

Base: ordinary #96 merge `175c6a7fba90e0b6e8774cc51f6cf03241eb7610`.
Code candidate: `2bb14f7` (the source/test patch in [code.patch](code.patch)).
SHA-256 of the frozen `src/ tests/ .github/workflows/ci.yml` patch:
`cd1cb1ca7abd4f8a0c4a6141a8730da196c7e64bda0d654c20de123242ac97d1`.

The Step 3D synthetic forecast fixture was run through the production
`cockpit_tables_to_excel` exporter on a clean base archive and on the candidate.
The saved [base](baseline.xlsx) and [candidate](candidate.xlsx) workbooks were
then read back with openpyxl and rendered through the existing read-only
Artifact Tool renderer: [before](baseline.png), [after](candidate.png).
The two existing long strategy names clip in the base render and are fully
visible in the candidate. This is a saved-XLSX render, not a Microsoft Excel
screenshot. The fixture uses named solver stubs and proves presentation, not
revenue economics.

Across both workbooks, every populated cell's value, data type and number
format is identical. The only changes are `Strategy comparison` strategy-cell
wrapping and row heights; the numeric columns retain their `#,##0.00` format.
The new test has one assertion failure on the clean base (`wrap_text` is unset)
and passes on the candidate. The related export/disclosure suites give
60 passed / 2 opt-in chart-render skips.

Reproduce from the repository root:

```sh
git diff --binary --full-index 175c6a7 2bb14f7 -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-23-strategy-export-evidence/code.patch
.venv/bin/python -m pytest tests/test_export.py::test_cockpit_strategy_names_are_readable_without_changing_values -q
.venv/bin/python -m pytest tests/test_export.py tests/test_step3d_export_disclosure.py tests/test_step3d_cockpit_disclosure.py tests/test_cockpit_export_provenance.py -q
```

For another synthetic export and render, run the existing
`docs/audits/2026-09-20-step3d-evidence/exports/generate_exports.py` into a
temporary directory, then the adjacent `render_exports.mjs` with the bundled
`ARTIFACT_TOOL_NODE_MODULES`. Neither script reads a production price cache.
