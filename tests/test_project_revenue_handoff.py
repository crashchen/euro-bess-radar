from __future__ import annotations

import dataclasses as dc
import hashlib
import json
from pathlib import Path

from src.project_case import (
    CapacityMaintenanceBasis,
    MarketCase,
    compute_project_case,
    project_revenue_handoff_payload,
    project_revenue_handoff_to_json,
)
from src.project_case.schema import _issue_strategy_run_result
from tests import pc_case_fixtures as fx

GOLDEN_HANDOFF_PATH = Path(__file__).parent / "fixtures" / "project_revenue_handoff_v1.json"
GOLDEN_HANDOFF_FILE_SHA256 = "afe1ec795bad06d5fba9bc77271e74c28710a84b5b3b68c3b1daf1ef77117f86"


def _negative_two_year_result():
    source = fx.da_only_srr()
    with _issue_strategy_run_result():
        negative_strategy = dc.replace(
            source,
            daily_realised_cash_series=((fx.D1, -100.0), (fx.D2, -50.0)),
        )
    base_case = fx.project_case(negative_strategy)
    lifecycle = dc.replace(
        base_case.lifecycle_case,
        project_life_years=2,
        capacity_maintenance_basis=CapacityMaintenanceBasis.NO_AUGMENTATION_REQUIRED_ASSERTED,
        augmentation_events=(),
    )
    case = dc.replace(
        base_case,
        lifecycle_case=lifecycle,
        market_case=MarketCase(negative_strategy, base_case.market_case.projection),
    )
    return compute_project_case(case)


def test_cross_repo_golden_handoff_fixture_matches_producer_bytes() -> None:
    fixture = GOLDEN_HANDOFF_PATH.read_bytes()

    assert hashlib.sha256(fixture).hexdigest() == GOLDEN_HANDOFF_FILE_SHA256
    assert project_revenue_handoff_to_json(_negative_two_year_result()) + b"\n" == fixture


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
    result = _negative_two_year_result()

    first_row = project_revenue_handoff_payload(result)["annual_rows"][0]

    assert first_row["merchant_revenue_eur"] < 0
    assert first_row["settled_revenue_eur"] < 0
