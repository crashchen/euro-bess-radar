# Multi-day and forecast-policy export provenance — review handoff

**Status:** code implemented on a review branch; do not merge before the
user's explicit instruction. Base: #95 ordinary merge
`cc6b1e24c527d5eec5454ffbdb2cfcc07cabaff0`. Code commit:
`e39968dd93eb72f8c792c11049035f2210b8c5eb`.

The sidebar's Data Trust table can say `Dispatch model: Greedy single-cycle`
when its LP switch is off, yet the multi-day replay and forecast-policy panels
still solve with MILP. Their Excel exports used to copy that sidebar row. The
multi-day workbook also copied `CapEx: Payback period only`, though its
`degradation_cost_eur` column uses the supplied CapEx for an ex-post linear
wear calculation.

The candidate updates only saved export-assumption copies. Multi-day exports
name DA-only or two-stage DA+IDA1 MILP according to the Run's replay mode,
record the cockpit capture haircut, and label CapEx as an ex-post wear input
that does not affect dispatch or gross revenue. Forecast-policy exports name
the sequential DA+IDA1 MILP for core rows; existing assumption rows continue
to describe optional reserve, triple and stochastic variants. Global sidebar
assumptions remain untouched. `Dispatch model` and `CapEx` labels are shared
constants in the builder and adapters, with a rename regression. Both panel
version IDs advance so pre-fix session bundles cannot offer stale exports.

The new test file gives **5 failed / 1 passed** on a clean base archive and
**6 passed** on the candidate. The related non-slow suites give **106 passed**;
the full local suite gives **2111 passed / 2 skipped** (35 slow, 29 observed
warnings), and full Ruff passes. A two-revision synthetic probe
finds identical gross revenue, wear cost, forecast comparison numbers, valid
days and global assumptions. The diff is confined to the intended export
rows. See the [verification snapshot](../validation/current.md) and
[evidence directory](2026-09-22-cockpit-export-evidence/README.md).

Frozen code patch SHA-256:
`b295b4bf8e648c3e262d375f6cbc92a52001af87c456e7235539185c90b051ae`.
Reproduce it with:

```sh
git diff --binary --full-index cc6b1e2..e39968d -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-09-22-cockpit-export-evidence/code.patch
```

Review the saved XLSX values and types, the distinction between gross revenue
and ex-post wear, both replay modes, optional forecast strategy wording,
and panel-version invalidation. The source diff should touch no solver,
dispatch, numerical cash, Project Case or CI module. Native Excel rendering,
the download click in a real browser, live-provider inputs and ESS consumption
are outside this evidence. Existing `None` global-assumptions passthrough
remains; normal `app.py` supplies the table.

The user invokes CC separately; do not invoke it automatically. Keep this
handoff and patch revision-bound if a later review requests changes.
