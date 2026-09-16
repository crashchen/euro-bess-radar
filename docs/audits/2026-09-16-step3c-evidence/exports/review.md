# Step 3C actual export verification

Production `export_to_bytes`, `export_to_pdf_bytes` and
`export_comparison_to_bytes` generated the files in this directory. The saved
workbooks were read with openpyxl and independently imported/rendered, without
editing, with Artifact Tool. Poppler rendered every page of all four PDFs, and
all five page images were visually inspected.

| Case | Actual average | Coverage/disclosure | Excel type | PDF |
| --- | --- | --- | --- | --- |
| 30 hourly days at 10 + 10 quarter-hour days at 100 | 32.50 | 960.00 / 960.00 h, 100.0% | numeric, `#,##0.00` | complete, no clipping |
| Same window with one hourly NaN and two quarter-hour infinities | 32.49 | 958.50 / 960.00 h, 99.8%; excluded prices not zero | numeric, `#,##0.00` | disclosure wraps in its column |
| Missing hour in otherwise hourly window | n/a | gap / unregistered cadence-change reason | string | reason complete |
| Three regular 2-hour samples | n/a | unsupported native product reason | string | reason complete |

The partial-finite PDF has two pages: the last two existing negative-price
summary rows continue onto page 2. Every row is readable; no overlap or clipping
was observed. This checks actual summary rendering, not chart-image rendering.

The Summary workbook average rows and their coverage/reason sentences are
readable in all four actual rendered files. Zone Comparison retains numeric
average cells 32.50 and 32.49, numeric percentage ratios 1.0 and 0.9984 displayed
as 100.0% and 99.8%, and string n/a cells beside the unverified reason. Saved
cell types, coordinates, formats and complete strings are in `results.json`.

## Visual finding fixed and re-rendered

The new Zone Comparison unavailable-reason column D has no wrapping. On row 4,
the following numeric Std Dev cell prevents overflow, so the sentence is
visibly clipped after “the delivery grid has a gap or a cadence c…”. Its saved
text is complete, but the unformatted row is not fully readable. The original
render is preserved as `comparison-before-wrap.png`.

With the parent's authorization, `src/export.py` now wraps the reason and
increases only the affected data row height, using the existing export width
and line-height constants. Values in that row align with the start of its
disclosure. The original strings and numeric types remain intact. The final
`comparison-xlsx.png` is a fresh rendering of the saved workbook and shows the
whole reason across three readable lines. The existing comparison test now
also checks saved wrapping, row height, complete reason text and numeric value
types. Test count remains 37 for the Step 3C average file.

Affected average and export tests: **59 passed / 2 skipped**. Ruff and scoped
diff check passed. The two skipped export tests are existing tests; this check
does not turn those skips into a chart-rendering claim.

## Reproduce

From the repository root:

```sh
PYTHONPATH=. .venv/bin/python docs/audits/2026-09-16-step3c-evidence/exports/export_probe.py
```

The script invokes `pdftotext` and `pdftoppm` and emits the actual XLSX/PDF files,
their text extracts, all rendered PDF pages and `results.json`. It uses the
repository's production export functions; it does not imitate their layout.
`render_exports.mjs` imports and renders the saved XLSX files using the bundled
`@oai/artifact-tool` runtime. The rendering check is not a claim of testing the
native Excel application.
