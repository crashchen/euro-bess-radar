# PR #105 control contrast revision — CC handoff

**2026-10-03 — implemented, awaiting user-invoked CC review; not merged.**

PR: [#105](https://github.com/crashchen/euro-bess-radar/pull/105), still draft.
Base remains `e28ee9c4b439c31c519d07cf390bc22b6fbd46d6`.
Revision baseline is the previously reviewed head
`d5bf99640f2a9f8ef9c3f5ed21ae60754f59c438`.
Code commit is `a32e707f4793d4c687b5f366f6d59ce6414051ff`;
later commits in this revision only add documentation/evidence.
The exact current documentation head and its CI must be read from PR Checks.

## Problem and resulting behavior

The first #105 patch made light-theme help buttons and main uploaders readable
and kept Data Trust cards from clipping at 960px. It also applied the bright
brand gradient to previously dark buttons: 16px Browse text had about
3.1–3.8:1 contrast, below the 4.5:1 normal-text threshold. Disabled Browse
received the same colored/glowing appearance as its enabled counterpart.
This revision closes both findings while keeping the original wrapper and
uploader fixes.

Only `src/ui_theme.py` changes in the production increment: **30 added lines**.
New `stMain` rules provide an opaque darker gradient, white leaf text even in
expanders, explicit darker hover colors with no brightness filter, and a gray
no-glow/no-filter disabled Browse surface. Shared brand variables and the
sidebar/calendar/tag rules are unchanged. The first #105 Data Trust layout
change stays in place. No source data, cache, dispatch, model, money,
Project Case state or export calculation changes.

## Evidence and checks

[Evidence index](2026-10-03-control-contrast-r2-evidence/README.md).

- Clean `d5bf996` baseline plus the new-test patch: **3 failed / 42 passed**;
  candidate: **45 passed**. Three new checks cover normal/hover contrast
  bounds and a separate muted disabled Browse surface.
- Full local suite at the code change: **2127 passed / 2 skipped / 29 warnings**,
  including **35 slow**, 296.93s. Ruff and whitespace checks pass.
- Browser: baseline/candidate × light/dark × 390/960/1280/1440, **16 configs**,
  with **64 actual candidate hover checks**. Main enabled controls have
  conservative contrast lower bounds **5.06:1 normal / 6.17:1 hover**.
  Disabled Browse is distinct and still disabled; sidebar and uploader
  instructions are unchanged; all four production Data Trust metrics retain
  their values and show no clipping.
- Real market cache: **26/26 file hashes unchanged**. All revision servers and
  browser tabs are closed; viewport reset. Original 8611/8612 untouched.
- Code-head remote compatibility check passed at `a32e707` (run
  [37140544596](https://github.com/crashchen/euro-bess-radar/actions/runs/37140544596)).
  Its full job was still running when this handoff was written. Verify both
  jobs on the latest documentation head before proposing merge; old-head CI
  does not approve later bytes.

Revision patch SHA-256:
`8cdf7e3f57549d959e42fdc55d958acdd70e036d3ad72c0204335a3edb54b177`.
Aggregate PR code patch:
`42f959440cf38d64af90d47974960c27caab8a2f052fe7c324aa85a77a24fc52`.
The first reviewed patch is preserved with SHA-256
`7513c25fc2c92d941cd2b5bb4081aca8fd4dc3948e6300c2f80426d8fd355e33`.

## Requested independent review

1. Recompute both source-patch hashes and run the same three new checks on a
   clean first-head archive; keep the 42 compatibility cases green.
2. Read the actual selector specificity: white leaf text must win inside an
   expander, enabled styling must not reach native disabled Browse, and the
   `stMain` boundary must keep sidebar/calendar/tag styles intact.
3. Re-run the synthetic harness in both themes and all four widths. Check
   only visible tooltip-button copies and actual leaf fill; exercise real
   hover and disabled Browse. Verify all four Data Trust card values and
   clipping, plus unchanged uploader instructions/sidebar. Do not use mean
   gradient-stop contrast as the sole test.
4. Run the saved-browser analysis and inspect the actual before/after images.
   Its channel-wise luminance bound is separate from a pixel/render claim.
   Verify precise head CI and the two-file production/test increment.

The browser fixture uses production CSS and the Data Trust renderer, not the
complete real-data application or contract workflow. No file/error-state
acceptance, all-52 manual acceptance or whole-app accessibility claim is made.
The old sidebar gradient remains outside this scoped change. Original CC
acceptance evidence in `outputs/` remains unchanged; the combined #104–#106
acceptance/merge/docs/Vault closeout is still separate. ESS F5/F6 and PDF
`inf`/empty-sample display follow-ups are untouched.

The user invokes CC separately and authorizes each PR merge explicitly.
