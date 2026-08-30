# Project Revenue Handoff v1

This contract defines the compact producer boundary from a validated Radar
`RunResult` to the ESS financial platform.

Radar exports `screening_cashflow_table.revenue_eur`, never lifecycle
`net_eur`. The exported cash is post-VOM, post-RTE, post-capture and, where
applicable, post-availability and post-contract settlement. It excludes
lifecycle costs, tax, debt and financing fees so ESS can apply those layers
exactly once.

The UTF-8 JSON wire uses schema
`euro_bess_radar.project_revenue_handoff`, version `1`, and carries:

- contiguous relative project years starting at 1;
- merchant, effective floor, top-up, settlement adjustment and settled EUR;
- ProjectCase and StrategyRunResult fingerprints plus source-data SHA-256;
- calculator/statistic/zone/sample-window provenance;
- modeled MW, duration, MWh, RTE and explicit embedded/excluded flags;
- a SHA-256 digest of canonical JSON excluding the digest field itself.

Merchant and settled values are signed. Consumers must not multiply settled
cash by MW, availability or RTE, and must not run floor settlement again.

The public producer is `project_revenue_handoff_to_json`. The Project Case UI
exposes the same bytes as `radar_project_revenue_handoff.json`.
