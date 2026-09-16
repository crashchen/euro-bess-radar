# Final Step 3C average regression file: baseline evidence

Baseline: `cf913747d4f27e490545f96b35f1f23e7b956cf4`, extracted with git archive
into an isolated directory. Only the final 37-case
`tests/test_step3c_average_price.py` was copied into it, including the saved-file reason-wrap assertion and explicit zone n/a caption. No baseline production code was modified.

Command from that baseline directory: `python -m pytest tests/test_step3c_average_price.py -q`.
Exact output: `final-base-tests.txt`. Result: **31 failed / 6 passed in 1.34s**.

The six passing compatibility controls remain:

- `test_page_keeps_a_uniform_hourly_average`
- `test_excel_summary_keeps_uniform_averages[h]`
- `test_excel_summary_keeps_uniform_averages[30min]`
- `test_excel_summary_keeps_uniform_averages[15min]`
- `test_pdf_summary_keeps_a_uniform_hourly_average`
- `test_zone_comparison_keeps_a_uniform_hourly_average`

Failures are the new calculation API being absent or the prior consumer
behavior: row-weighted 61.43 instead of 32.50, no duration/coverage disclosure,
no unavailable reason, and no reason-aware comparison export. The three new
review boundary cases also fail at the absent calculation API on this base.
The module imports and collects cleanly; there are no collection failures.

Additional diagnostic against isolated CC commit `c703013`, with the same
final test file: **5 failed / 32 passed in 1.33s**, output
`final-c703013-tests.txt`. The five failures are exactly the two unsupported
uniform frequencies, numeric row-index conversion, saved reason wrapping, and
the zone caption lacking an explicit n/a marker.
This isolates the completion fixes relative to CC's checkpoint.

After the corrections, the final file plus existing export tests yields
**59 passed / 2 skipped in 1.63s** (`export-fix-tests.txt`). No test cases were
added for the wrapping fix; the final Step 3C average file still has 37 cases.
