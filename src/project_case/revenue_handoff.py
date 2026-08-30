"""Compact annual revenue handoff from Radar Project Case to ESS.

The handoff deliberately exports the screening table's settled revenue, not
the lifecycle table's net cash. Radar owns dispatch, capture, VOM, availability,
RTE, and contract settlement; ESS owns lifecycle cost, tax, and financing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from src.project_case.schema import ProjectCaseValidationError, RunResult

PROJECT_REVENUE_HANDOFF_SCHEMA = "euro_bess_radar.project_revenue_handoff"
PROJECT_REVENUE_HANDOFF_VERSION = 1


def project_revenue_handoff_payload(result: RunResult) -> dict[str, Any]:
    """Build the locked Project Revenue Handoff v1 from a typed RunResult."""

    if not isinstance(result, RunResult):
        raise TypeError("result must be a RunResult")
    table = result.screening_cashflow_table
    if table is None or table.basis != "screening":
        raise ProjectCaseValidationError("revenue handoff requires the screening cashflow table")

    provenance = _mapping(result.provenance, "RunResult.provenance")
    strategy = _mapping(provenance.get("strategy_run_result"), "provenance.strategy_run_result")
    project_case = _mapping(provenance.get("project_case"), "provenance.project_case")
    asset = _mapping(project_case.get("asset_case"), "provenance.project_case.asset_case")
    valuation = _mapping(provenance.get("valuation"), "provenance.valuation")
    red_lines = _mapping(provenance.get("red_line_assertions"), "provenance.red_line_assertions")
    cash_basis = _mapping(strategy.get("cash_basis"), "provenance.strategy_run_result.cash_basis")
    capture = _mapping(
        cash_basis.get("capture"), "provenance.strategy_run_result.cash_basis.capture"
    )
    liquidity = _mapping(
        cash_basis.get("liquidity"), "provenance.strategy_run_result.cash_basis.liquidity"
    )
    sample_window = _mapping(
        strategy.get("sample_window"), "provenance.strategy_run_result.sample_window"
    )

    _require_equal(
        valuation.get("currency_convention"), "real_base_year_eur", "valuation.currency_convention"
    )
    _require_equal(valuation.get("cash_timing"), "end_of_year", "valuation.cash_timing")
    _require_equal(cash_basis.get("post_vom"), True, "strategy.cash_basis.post_vom")
    for key, expected in {
        "mw_rescaled": False,
        "tax_included": False,
        "debt_included": False,
        "financing_fees_included": False,
    }.items():
        _require_equal(red_lines.get(key), expected, f"red_line_assertions.{key}")

    power = _positive(asset.get("power_mw"), "asset_case.power_mw")
    duration = _positive(asset.get("duration_hours"), "asset_case.duration_hours")
    rte = _fraction(asset.get("round_trip_efficiency"), "asset_case.round_trip_efficiency")
    annual_rows: list[dict[str, Any]] = []
    for row in table.rows:
        if not math.isclose(row.net_eur, row.revenue_eur, rel_tol=1e-9, abs_tol=0.01):
            raise ProjectCaseValidationError(
                "screening handoff row must not contain lifecycle cash adjustments"
            )
        annual_rows.append(
            {
                "project_year": row.year,
                "merchant_revenue_eur": float(row.merchant_revenue_eur),
                "effective_contract_floor_eur": (
                    None
                    if row.effective_contract_floor_eur is None
                    else float(row.effective_contract_floor_eur)
                ),
                "contract_top_up_eur": float(row.contract_top_up_eur),
                "settlement_adjustment_eur": float(row.revenue_eur - row.merchant_revenue_eur),
                "settled_revenue_eur": float(row.revenue_eur),
            }
        )

    unsigned = {
        "schema": PROJECT_REVENUE_HANDOFF_SCHEMA,
        "version": PROJECT_REVENUE_HANDOFF_VERSION,
        "source": {
            "run_result_schema_version": result.schema_version,
            "project_case_input_fingerprint": result.input_fingerprint,
            "strategy_run_fingerprint": _text(
                provenance.get("strategy_run_fingerprint"),
                "provenance.strategy_run_fingerprint",
            ),
            "source_data_content_hash": _text(
                strategy.get("source_data_content_hash"),
                "strategy_run_result.source_data_content_hash",
            ),
            "calculator_version": _text(
                provenance.get("calculator_version"), "provenance.calculator_version"
            ),
            "strategy_calculator_version": _text(
                strategy.get("calculator_version"),
                "strategy_run_result.calculator_version",
            ),
            "cashflow_table_statistic": _text(
                provenance.get("cashflow_table_statistic"),
                "provenance.cashflow_table_statistic",
            ),
            "strategy_kind": _text(
                strategy.get("strategy_kind"), "strategy_run_result.strategy_kind"
            ),
            "zone": _text(strategy.get("zone"), "strategy_run_result.zone"),
            "sample_start_date": _text(
                sample_window.get("first_delivery_date"), "sample_window.first_delivery_date"
            ),
            "sample_end_date": _text(
                sample_window.get("last_delivery_date"), "sample_window.last_delivery_date"
            ),
        },
        "basis": {
            "currency": "EUR",
            "currency_mode": "real_base_year_eur",
            "real_base_year": _whole_number(valuation.get("base_year"), "valuation.base_year"),
            "cash_timing": "end_of_year",
            "project_year_origin": "relative",
            "modeled_power_mw": power,
            "modeled_duration_hours": duration,
            "modeled_energy_mwh": power * duration,
            "round_trip_efficiency": rte,
            "post_vom": True,
            "rte_embedded": True,
            "availability_embedded": strategy.get("availability") is not None,
            "capture_embedded": _boolean(capture.get("applied"), "cash_basis.capture.applied"),
            "liquidity_embedded": _boolean(
                liquidity.get("applied"), "cash_basis.liquidity.applied"
            ),
            "contract_settlement_included": _boolean(
                red_lines.get("contract_settlement_included"),
                "red_line_assertions.contract_settlement_included",
            ),
            "contract_settlement_basis": red_lines.get("contract_settlement_basis"),
            "lifecycle_costs_included": False,
            "tax_included": False,
            "debt_included": False,
            "financing_fees_included": False,
            "mw_rescaled": False,
        },
        "annual_rows": annual_rows,
    }
    return {
        **unsigned,
        "handoff_digest_sha256": _canonical_json_digest(unsigned),
    }


def project_revenue_handoff_to_json(result: RunResult) -> bytes:
    return json.dumps(
        project_revenue_handoff_payload(result),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectCaseValidationError(f"{field} must be a mapping")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectCaseValidationError(f"{field} must be non-empty text")
    return value.strip()


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ProjectCaseValidationError(f"{field} must be bool")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ProjectCaseValidationError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProjectCaseValidationError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise ProjectCaseValidationError(f"{field} must be finite")
    return number


def _positive(value: Any, field: str) -> float:
    number = _finite(value, field)
    if number <= 0:
        raise ProjectCaseValidationError(f"{field} must be > 0")
    return number


def _fraction(value: Any, field: str) -> float:
    number = _finite(value, field)
    if not 0 < number <= 1:
        raise ProjectCaseValidationError(f"{field} must be in (0, 1]")
    return number


def _whole_number(value: Any, field: str) -> int:
    number = _finite(value, field)
    if number != int(number):
        raise ProjectCaseValidationError(f"{field} must be a whole number")
    return int(number)


def _require_equal(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise ProjectCaseValidationError(f"{field} must be {expected!r}")
