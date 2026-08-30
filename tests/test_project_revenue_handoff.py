from __future__ import annotations

import dataclasses as dc
import hashlib
import json

from src.project_case import (
    MarketCase,
    compute_project_case,
    project_revenue_handoff_payload,
    project_revenue_handoff_to_json,
)
from src.project_case.schema import _issue_strategy_run_result
from tests import pc_case_fixtures as fx


def test_handoff_exports_screening_settled_revenue_and_audit_flags() -> None:
    result = compute_project_case(fx.project_case(contract=fx.contract_case()))

    payload = project_revenue_handoff_payload(result)

    assert payload["schema"] == "euro_bess_radar.project_revenue_handoff"
    assert payload["version"] == 1
    assert payload["source"]["project_case_input_fingerprint"] == result.input_fingerprint
    assert payload["basis"]["lifecycle_costs_included"] is False
    assert payload["basis"]["tax_included"] is False
    assert payload["basis"]["debt_included"] is False
    assert payload["basis"]["contract_settlement_included"] is True
    assert [row["settled_revenue_eur"] for row in payload["annual_rows"]] == [
        row.revenue_eur for row in result.screening_cashflow_table.rows
    ]
    assert all(
        row["merchant_revenue_eur"] + row["settlement_adjustment_eur"] == row["settled_revenue_eur"]
        for row in payload["annual_rows"]
    )


def test_handoff_digest_covers_every_unsigned_field() -> None:
    payload = json.loads(project_revenue_handoff_to_json(compute_project_case(fx.project_case())))
    digest = payload.pop("handoff_digest_sha256")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    assert digest == hashlib.sha256(encoded).hexdigest()


def test_handoff_preserves_negative_merchant_and_settled_revenue() -> None:
    source = fx.da_only_srr()
    with _issue_strategy_run_result():
        negative_strategy = dc.replace(
            source,
            daily_realised_cash_series=((fx.D1, -100.0), (fx.D2, -50.0)),
        )
    case = fx.project_case(negative_strategy)
    result = compute_project_case(
        dc.replace(
            case,
            market_case=MarketCase(negative_strategy, case.market_case.projection),
        )
    )

    first_row = project_revenue_handoff_payload(result)["annual_rows"][0]

    assert first_row["merchant_revenue_eur"] < 0
    assert first_row["settled_revenue_eur"] < 0
