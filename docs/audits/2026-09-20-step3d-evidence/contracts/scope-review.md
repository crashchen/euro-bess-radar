# Independent Step 3D scope review

Base: 2e4ed73d7357e4d556373f6019198045f8fc8d04.
The 55 tracked files under src were compared byte-for-byte against the base.
51 are unchanged. Existing modified files are src/export.py and the Project
Case, Revenue Estimation, and Simulation Cockpit page modules. The new module
src/settlement_disclosure.py contains classification and display strings only.

The 12 src/project_case files are unchanged, including schema, fingerprints,
producer adapters, market grid, audit, and calculator. analytics.py,
simulation.py, dispatch.py, time_utils.py, strategy_compare.py, assumptions.py,
and all forecast/scenario model modules are unchanged. The evidence JSON
records before/working SHA-256 for each tracked source file.

Cockpit's existing fingerprint already includes primary_zone, product, full
price frame contents, selected dates, and the used model knobs. The new zone
argument is passed from that same run context, not resolved during rendering.
Capacity rows are identified by explicit labels supplied to the comparison
builder for finite solver outputs; the stochastic delta is classified only in
reserve mode. Original strategy comparison columns and numeric calculations
are untouched. The two new columns are presentation-only, added after the
comparison builder returns. Full basis and compact label/scope are stored in
the result bundle; export assumptions are assembled once with that result.
The renderer reads derived['capacity_settlement']; it does not re-resolve the
zone or product from later sidebar state. The v2 panel ID invalidates old v1
bundles that do not carry the disclosure, using the existing stale-result gate.
No solver receives a new timing, price, forecast, or settlement argument.

Project Case rendering/export reads the immutable recorded strategy payload
inside the displayed RunResult, including the original zone, product, adapter,
registry and reserve profile. No fields are appended to schema/provenance.
The registered DE_LU nominal contract is not extended to other zones or future
reserve profiles. Physical energy/SoC timing remains untouched.

Revenue Estimation derives the scope from products classified capacity/mixed
in the same ancillary aggregation that supplies the joint capacity price.
Energy-only products are excluded. A full disclosure is attached only to an
available joint result and shared verbatim between page and market export.
Summary/PDF display it only alongside a joint_cooptimized_total_eur; standalone
annual capacity estimates (HOURS_PER_YEAR=8760), activation overlays, and
imbalance overlays are not assigned the joint solver's physical-hour basis.
Legacy joint export callers without captured product identity receive an
explicit identity-unavailable fallback; no specific product is invented.

No scope or snapshot-context blocker found. This read-only code review does
not certify spreadsheet/PDF visual fit, which needs the separately rendered
artifacts, nor constitute external market/legal settlement verification.
