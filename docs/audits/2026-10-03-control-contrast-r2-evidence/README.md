# PR #105 control contrast revision evidence — 2026-10-03

PR base: `e28ee9c4b439c31c519d07cf390bc22b6fbd46d6`.
First reviewed PR head / revision baseline:
`d5bf99640f2a9f8ef9c3f5ed21ae60754f59c438`.
Revision code commit: `a32e707f4793d4c687b5f366f6d59ce6414051ff`.
This is an unmerged candidate awaiting the user's separate CC review.

## Frozen patches

| Patch | Scope | SHA-256 |
|---|---|---|
| [original-code.patch](original-code.patch) | PR base → first reviewed head; original bytes from Codex's first review | `7513c25fc2c92d941cd2b5bb4081aca8fd4dc3948e6300c2f80426d8fd355e33` |
| [revision-code.patch](revision-code.patch) | First head → this code commit, source/tests/CI | `8cdf7e3f57549d959e42fdc55d958acdd70e036d3ad72c0204335a3edb54b177` |
| [aggregate-code.patch](aggregate-code.patch) | PR base → this code commit, source/tests/CI | `42f959440cf38d64af90d47974960c27caab8a2f052fe7c324aa85a77a24fc52` |
| [baseline-red-tests.patch](baseline-red-tests.patch) | Three new cases only, relative to the first head | `68d12a881955c37b72fdb37981a211cb78843c44c115fc73702c69296e913090` |

The original CC output folders and first-review files remain unchanged. These
patches are separate records; none replaces a historical patch.

## Tests

Apply the red-test patch to a clean `d5bf996` archive and run
`tests/test_ui_theme.py tests/test_data_trust.py`: [3 failed / 42 passed](baseline-red.txt),
all assertion failures, no collection errors. The missing scoped normal/hover
surface and missing disabled override fail the three new cases. Existing
42 cases are the compatibility controls. Candidate: [45 passed](focused.txt).
The numeric cases use a channel-wise luminance ceiling above both opaque sRGB
stops, rather than a mean stop color. They check the declared surface's 4.5:1
contract; the browser records separately check cascade, leaf text and states.

[Full local suite](full-suite.txt): **2127 passed / 2 skipped / 29 warnings**
in 296.93s; all **35 slow** cases are included. This is #105's candidate,
not a stack with #104/#106. Warnings are the existing Streamlit DataFrame-attrs
serialization and pandas empty-concat warnings. Ruff and `git diff --check`
passed. Remote CI is revision-bound; consult the exact PR head's Checks.
No old head's green check is treated as evidence for this revision.

## Browser matrix

[control_harness.py](control_harness.py) calls production theme injection and
the production Data Trust renderer. Controls are real Streamlit widgets with
synthetic labels/content: a help-wrapped Handoff download, help buttons in the
main area and an expander, a form submit and enabled/disabled uploaders. It
redirects CACHE_DIR to an empty temporary directory, but does not redirect
DB_PATH: Data Trust source diagnostics can read the existing provenance
database. Importing src.config also executes load_dotenv() against the source
checkout. No provider client is created, no fetch is invoked and no actual
contract is uploaded. The four KPI values come from the synthetic price frame;
the 26/26 cache-hash comparison verifies no write, not absence of reads. The harness does not execute Project Case or
assert a full production-page acceptance pass.

Codex used the in-app browser via CUA, not a shell/headless browser driver.
The viewport was set to W × 900 CSS px, DPR 1. At 390px the sidebar is closed
for main/metric checks, and opened separately for the sidebar probe. At
960/1280/1440 it is expanded. Server base theme is forced light/dark.

The [matrix index](browser/matrix.json) covers **16 configurations**: baseline
and candidate × light/dark × 390/960/1280/1440. Each has a raw JSON, a controls
screenshot and a Data Trust screenshot. Screenshots are viewport captures;
JSON includes visible controls further down the page. Each candidate also
has eight actual hover records, **64 total**: six enabled controls and two
disabled Browse buttons. CUA dragged the real pointer from adjacent blank
space into the target and asserted `matches(':hover')`; no file chooser or
download was triggered. The first-head matrix records normal states only.
An extra [Handoff hover screenshot](browser/candidate-dark-handoff-hover.png)
and [record](browser/handoff-hover-proof.json) retain one visible hover proof.

The read-only [browser probe](browser_probe.js) filters hidden tooltip-button
copies and reads the actual leaf color/fill, background, filter and disabled
state. [analyze_browser.py](analyze_browser.py) checks the saved matrix; its
[summary](browser-summary.json) records conservative contrast lower bounds
**5.0557:1 normal** and **6.1704:1 hover**, both above 4.5:1. It bounds every
sRGB-interpolated position using the maximum of each channel from both stops;
it does not average stops or measure anti-aliased text pixels.

All eight candidate configurations retain distinct gray disabled Browse
surfaces, native disabled flags, no glow/filter, unchanged sidebar style and
uploader instructions, and the same four untruncated Data Trust metrics:
1 fetched zone, 100.0% coverage, 0 source gaps and 0 unresolved missing.

Scope remains the main controls. The sidebar's older brighter palette,
calendar/tag colors, arbitrary amounts, file/error states, live fetches and
whole-app WCAG conformance are outside this revision's acceptance. If a future
Streamlit button uses only `aria-disabled` rather than native `disabled`, the
enabled selector will need a corresponding exclusion.
[cache-scope.json](cache-scope.json) records actual-cache **26/26 hashes equal**,
closed test servers/tabs and restored viewport. Original 8611/8612 services
were left unchanged. File hashes are in [manifest.json](manifest.json).

## Reproduce

```sh
git diff --binary --full-index d5bf996 a32e707 -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
git diff --binary --full-index e28ee9c a32e707 -- src/ tests/ .github/workflows/ci.yml | shasum -a 256
shasum -a 256 docs/audits/2026-10-03-control-contrast-r2-evidence/*.patch
.venv/bin/python -m pytest tests/test_ui_theme.py tests/test_data_trust.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ app.py tests/
.venv/bin/python docs/audits/2026-10-03-control-contrast-r2-evidence/analyze_browser.py
RADAR_UI_SOURCE="$PWD" .venv/bin/python -m streamlit run docs/audits/2026-10-03-control-contrast-r2-evidence/control_harness.py --server.address 127.0.0.1 --server.port 8620 --server.headless true --browser.gatherUsageStats false --theme.base light
```

For the red comparison, archive the first head into an empty temporary
checkout, apply only `baseline-red-tests.patch`, then run the same two test
files using the installed Python with that checkout as cwd/PYTHONPATH.
For a browser re-run, use the harness on each version with both base themes;
repeat the listed viewport/sidebar/pointer states and use the probe function
as a read-only evaluation. A saved JSON analysis is not a new render.
