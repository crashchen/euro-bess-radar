# Step 3D production export evidence

These fixtures use synthetic prices only. The production exporters created the
archived XLSX/PDF bytes. No production price cache was read or written.

| Artifact | Fixture and checked result |
| --- | --- |
| `project-case-dst.xlsx` | Real public Project Case reserve co-optimisation and valuation: DE_LU, FCR [symmetric], 2026-03-29, 1 MW, EUR 20/MW/h, 95% availability. The 23-hour physical day retains EUR 456 capacity cash under the recorded six nominal 4h blocks. |
| `fi-report-with-de-project-case.xlsx` | A Finnish market report with that same immutable German Project Case appended. The Project Case disclosure and fingerprint remain German. |
| `de_lu-screening-spring.xlsx` / `.pdf` | Real Revenue Estimation renderer, aggregation and joint solver: FCR EUR 20/MW/h + aFRR Up EUR 10/MW/h, 1 MW, 95% availability, 23 physical hours. Capacity is EUR 655.50/day, annualised and rounded to EUR 239,421.38. An energy-only mFRR input is excluded from the capacity-product identity. |
| `fi-screening-spring.xlsx` / `.pdf` | Real Finnish screening path with FCR-N EUR 20/MW/h, 1 MW and 95% availability: EUR 437/day, EUR 159,614.25/year. Its disclosure contains no German nominal-4h assertion. |
| `cockpit-forecast-stub-no-input-assumptions.xlsx` | Display fixture with explicitly stubbed solver outputs, built by `_compute_bundle` in `test_step3d_cockpit_disclosure.py`. Caller assumptions are `None`. The production export still includes the frozen capacity basis, scope and affected strategies. EUR 437 remains numeric; policy-value delta EUR -3 remains negative and numeric; its unavailable uplift stays blank. This fixture does not independently validate solver economics. |

## What was rendered and inspected

The four PDF pages were rasterised with Poppler `pdftoppm` at 120 dpi and visually
inspected. The six changed spreadsheet ranges were imported and rendered
read-only with `@oai/artifact-tool` at scale 1.5. These are **not Microsoft Excel
screenshots**. ArtifactTool did not edit, recalculate or re-export the workbooks.
The added capacity disclosures, scope columns and complete Assumptions text fit
in these samples. PDF chart layout is outside these focused summary fixtures.

`export-verification.json` records the original XLSX/PDF byte hashes, synthetic
known answers and rendered ranges. `*-cells.json` records each populated cell's
value, type, number format, wrapping and row height. `visual-review.json` records
the inspected cells and visual limits. Paths in the JSON are artifact-relative;
no private checkout/runtime paths are included.

Existing issues remain separately recorded: long Cockpit strategy names in
column A are clipped by the existing width-30/no-wrap style. The new Assumptions
`affects` field shows those full identities. The existing Summary Capacity Stack
Warning is unwrapped; the bounded A:B preview crops its spill into empty columns.
ArtifactTool displays an existing Project Case Boolean `False` as `0`, while the
XLSX retains Boolean type `b`. The existing `_build_table_sheet` and `_auto_column_width` functions are
unchanged from base `2e4ed73`; the new disclosure formatter runs afterward.

## Reproduce without changing these archived files

From the repository root, run the production generator into a new temporary
directory. It needs the repository Python environment and Poppler on `PATH`:

```sh
STEP3D_EXPORT_DIR="$(mktemp -d)"
PYTHONPATH=. .venv/bin/python docs/audits/2026-09-20-step3d-evidence/exports/generate_exports.py --out "$STEP3D_EXPORT_DIR"
```

Omitting `--out` uses an `euro-bess-step3d-exports` directory below the system
temporary directory. Do not point it at this archived evidence directory.

For spreadsheet previews, obtain the bundled Node executable and `node_modules`
directory from `load_workspace_dependencies`. Assign their values to
`STEP3D_NODE` and `ARTIFACT_TOOL_NODE_MODULES`; the renderer resolves the public
ArtifactTool package through that directory and does not require a copied
`node_modules` tree:

```sh
export ARTIFACT_TOOL_NODE_MODULES
"$STEP3D_NODE" docs/audits/2026-09-20-step3d-evidence/exports/render_exports.mjs "$STEP3D_EXPORT_DIR"
```

Regenerated XLSX/PDF bytes can have different hashes because their containers
include generation metadata. Compare the recorded values, types, formats and
rendered disclosures; the stored hashes identify the archived bytes themselves.
