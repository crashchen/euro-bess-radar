"""PC-C Excel presentation tests for typed Project Case RunResult outputs."""

from __future__ import annotations

import base64
import dataclasses as dc
from io import BytesIO

import pandas as pd
import pytest
from openpyxl import load_workbook

from src.export import export_to_bytes, project_case_to_excel
from src.project_case import (
    CapacityMaintenanceBasis,
    LifecycleCase,
    RunResult,
    compute_project_case,
    fingerprint_hex,
)
from tests import pc_case_fixtures as fx


def _known_result():
    return compute_project_case(fx.project_case())


def _unknown_result():
    case = fx.project_case()
    lifecycle = LifecycleCase(
        project_life_years=case.lifecycle_case.project_life_years,
        capacity_maintenance_basis=CapacityMaintenanceBasis.UNKNOWN,
        capacity_maintenance_source=None,
        capacity_maintenance_as_of=None,
        augmentation_events=(),
        eol_residual_value_eur=case.lifecycle_case.eol_residual_value_eur,
        decommissioning_cost_eur=case.lifecycle_case.decommissioning_cost_eur,
    )
    return compute_project_case(dc.replace(case, lifecycle_case=lifecycle))


def _rows_by_label(ws) -> dict[str, tuple[object, object]]:
    return {
        ws.cell(row=row, column=1).value: (
            ws.cell(row=row, column=2).value,
            ws.cell(row=row, column=3).value,
        )
        for row in range(2, ws.max_row + 1)
        if ws.cell(row=row, column=1).value is not None
    }


def _audit_values(ws) -> dict[str, object]:
    return {
        ws.cell(row=row, column=1).value: ws.cell(row=row, column=2).value
        for row in range(2, ws.max_row + 1)
    }


def _result_with_provenance(result: RunResult, provenance: dict) -> RunResult:
    return RunResult(
        input_fingerprint=result.input_fingerprint,
        no_lifecycle_cost_screening_npv=result.no_lifecycle_cost_screening_npv,
        lifecycle_cash_npv=result.lifecycle_cash_npv,
        provenance=provenance,
        screening_cashflow_table=result.screening_cashflow_table,
        lifecycle_cashflow_table=result.lifecycle_cashflow_table,
    )


def test_project_case_to_excel_writes_stable_full_workbook() -> None:
    result = _known_result()
    workbook = load_workbook(BytesIO(project_case_to_excel(result)))

    assert workbook.sheetnames == [
        "Project Case NPVs",
        "Screening Cash Flow",
        "Lifecycle Cash Flow",
        "Assumptions & Provenance",
    ]

    npv_rows = _rows_by_label(workbook["Project Case NPVs"])
    assert npv_rows["Available"] == (True, True)
    assert npv_rows["Status"] == ("ok", "ok")
    assert npv_rows["P10 (Downside, EUR)"] == pytest.approx(
        (
            result.no_lifecycle_cost_screening_npv.distribution.p10,
            result.lifecycle_cash_npv.distribution.p10,
        )
    )
    assert npv_rows["P50 (Median, EUR)"] == pytest.approx(
        (
            result.no_lifecycle_cost_screening_npv.distribution.p50,
            result.lifecycle_cash_npv.distribution.p50,
        )
    )
    assert npv_rows["P90 (Upside, EUR)"] == pytest.approx(
        (
            result.no_lifecycle_cost_screening_npv.distribution.p90,
            result.lifecycle_cash_npv.distribution.p90,
        )
    )
    assert npv_rows["Strategy Kind"] == ("DA_ONLY", None)
    assert npv_rows["Producer Adapter"] == ("PC_ADP_DA_ONLY", None)
    assert npv_rows["Observed Dates"] == (2, None)
    assert npv_rows["Valid Dates"] == (2, None)
    assert npv_rows["Missing Dates"] == (0, None)
    assert npv_rows["Solver-failed Dates"] == (0, None)

    expected_headers = [
        "Project Year",
        "Revenue (EUR)",
        "Fixed O&M (EUR)",
        "Augmentation Net Cost (EUR)",
        "Terminal Cash (EUR)",
        "Net Cash Flow (EUR)",
        "Discount Factor",
        "Discounted Net Cash Flow (EUR)",
    ]
    for sheet_name, table in (
        ("Screening Cash Flow", result.screening_cashflow_table),
        ("Lifecycle Cash Flow", result.lifecycle_cashflow_table),
    ):
        sheet = workbook[sheet_name]
        assert [cell.value for cell in sheet[1]][:2] == [
            "Representative P50 reconciliation",
            "Value (EUR)",
        ]
        reconciled = {
            sheet.cell(row=row, column=1).value: sheet.cell(row=row, column=2).value
            for row in range(2, 7)
        }
        assert reconciled["Installed CapEx at year 0"] == 5_000_000
        expected_p50 = (
            result.no_lifecycle_cost_screening_npv.distribution.p50
            if sheet_name == "Screening Cash Flow"
            else result.lifecycle_cash_npv.distribution.p50
        )
        assert reconciled["Reconciled P50 NPV"] == pytest.approx(expected_p50)
        assert reconciled["Reported P50 NPV"] == pytest.approx(expected_p50)
        assert reconciled["Reconciliation difference"] == pytest.approx(
            0.0, abs=1e-6
        )
        assert [sheet.cell(row=8, column=col).value for col in range(1, 9)] == expected_headers
        assert sheet.max_row == len(table.rows) + 8
        first = table.rows[0]
        assert [sheet.cell(row=9, column=col).value for col in range(1, 9)] == pytest.approx(
            [
                first.year,
                first.revenue_eur,
                first.opex_eur,
                first.augmentation_eur,
                first.terminal_eur,
                first.net_eur,
                first.discount_factor,
                first.discounted_net_eur,
            ]
        )

    audit = _audit_values(workbook["Assumptions & Provenance"])
    assert audit["run_result.input_fingerprint"] == result.input_fingerprint
    assert audit["provenance.red_line_assertions.floor_included"] == "false"
    assert audit["provenance.red_line_assertions.pre_tax_unlevered"] == "true"
    assert audit["provenance.project_case.asset_case.installed_capex_eur"] == "5000000.0"
    assert audit["provenance.project_case.lifecycle_case.augmentation_events[0].year"] == "8"


def test_unknown_lifecycle_is_blank_and_has_no_cashflow_sheet() -> None:
    result = _unknown_result()
    workbook = load_workbook(BytesIO(project_case_to_excel(result)))

    assert workbook.sheetnames == [
        "Project Case NPVs",
        "Screening Cash Flow",
        "Assumptions & Provenance",
    ]
    rows = _rows_by_label(workbook["Project Case NPVs"])
    assert rows["Available"] == (True, False)
    assert rows["Status"][1] == "capacity_maintenance_unknown"
    assert rows["Message"][1] == "Engineering capacity-maintenance basis is unknown."
    for label in (
        "P10 (Downside, EUR)",
        "P50 (Median, EUR)",
        "P90 (Upside, EUR)",
        "P(NPV > 0)",
    ):
        assert rows[label][0] is not None
        assert rows[label][1] is None


def test_project_case_export_is_formula_safe_and_preserves_large_seed_as_text() -> None:
    case = fx.project_case()
    lifecycle = dc.replace(
        case.lifecycle_case,
        capacity_maintenance_source="=HYPERLINK(\"https://invalid\",\"click\")",
    )
    bootstrap = dc.replace(case.bootstrap_case, seed=2**64 - 1)
    result = compute_project_case(
        dc.replace(case, lifecycle_case=lifecycle, bootstrap_case=bootstrap)
    )
    sheet = load_workbook(BytesIO(project_case_to_excel(result)))[
        "Assumptions & Provenance"
    ]
    audit = _audit_values(sheet)
    source_path = "provenance.project_case.lifecycle_case.capacity_maintenance_source"
    assert audit[source_path] == '=HYPERLINK("https://invalid","click")'
    assert audit["provenance.bootstrap.seed"] == str(2**64 - 1)
    for row in range(2, sheet.max_row + 1):
        assert sheet.cell(row=row, column=2).data_type != "f"
        assert sheet.cell(row=row, column=2).quotePrefix is True


def test_project_case_export_rejects_floor_included_or_wrong_type() -> None:
    result = _known_result()
    provenance = result.to_payload()["provenance"]
    provenance["red_line_assertions"]["floor_included"] = True
    unsafe = RunResult(
        input_fingerprint=result.input_fingerprint,
        no_lifecycle_cost_screening_npv=result.no_lifecycle_cost_screening_npv,
        lifecycle_cash_npv=result.lifecycle_cash_npv,
        provenance=provenance,
        screening_cashflow_table=result.screening_cashflow_table,
        lifecycle_cashflow_table=result.lifecycle_cashflow_table,
    )
    with pytest.raises(ValueError, match="floor_included"):
        project_case_to_excel(unsafe)
    with pytest.raises(TypeError, match="RunResult"):
        project_case_to_excel(object())


def test_project_case_export_rejects_mismatched_audit_fingerprint() -> None:
    result = _known_result()
    provenance = result.to_payload()["provenance"]
    provenance["project_case_input_fingerprint"] = "00" * 32
    unsafe = _result_with_provenance(result, provenance)

    with pytest.raises(ValueError, match="fingerprint does not match"):
        project_case_to_excel(unsafe)


def test_project_case_export_recomputes_strategy_fingerprint_from_payload() -> None:
    result = _known_result()
    provenance = result.to_payload()["provenance"]
    provenance["strategy_run_result"]["daily_realised_cash_series"][0][1] += 99.0
    unsafe = _result_with_provenance(result, provenance)

    with pytest.raises(ValueError, match="StrategyRunResult fingerprint"):
        project_case_to_excel(unsafe)


def test_project_case_export_binds_project_case_to_strategy_payload() -> None:
    result = _known_result()
    provenance = result.to_payload()["provenance"]
    strategy = provenance["strategy_run_result"]
    strategy["daily_realised_cash_series"][0][1] += 99.0
    provenance["strategy_run_fingerprint"] = fingerprint_hex(
        "StrategyRunResult", strategy
    )
    unsafe = _result_with_provenance(result, provenance)

    with pytest.raises(ValueError, match="does not link"):
        project_case_to_excel(unsafe)


def test_project_case_export_recomputes_project_case_fingerprint_from_payload() -> None:
    result = _known_result()
    provenance = result.to_payload()["provenance"]
    provenance["project_case"]["asset_case"]["installed_capex_eur"] += 1.0
    unsafe = _result_with_provenance(result, provenance)

    with pytest.raises(ValueError, match="ProjectCase fingerprint"):
        project_case_to_excel(unsafe)


def test_project_case_export_rejects_comparator_smuggling_under_alias() -> None:
    result = _known_result()
    provenance = result.to_payload()["provenance"]
    # The old semantic denylist only searched for the token "floor" and let this
    # equally dangerous contract-cash alias through.
    provenance["external_contract_comparator"] = {
        "protected_cashflow_eur": 123.0
    }
    unsafe = _result_with_provenance(result, provenance)

    with pytest.raises(ValueError, match="exact current PC-B schema"):
        project_case_to_excel(unsafe)


def test_project_case_export_rejects_extra_red_line_assertion() -> None:
    result = _known_result()
    provenance = result.to_payload()["provenance"]
    provenance["red_line_assertions"]["contract_comparator_included"] = False
    unsafe = _result_with_provenance(result, provenance)

    with pytest.raises(ValueError, match=r"red_line_assertions.*exact current PC-B"):
        project_case_to_excel(unsafe)


def test_project_case_export_accepts_locked_floor_fields_only_in_their_schema_slots() -> None:
    # The fixture's ProjectCase projection carries the legitimate
    # decay_floor_share slot, and red lines carry floor_included=False.
    project_case_to_excel(_known_result())


def test_assumption_export_preserves_full_float_precision() -> None:
    case = fx.project_case()
    valuation = dc.replace(case.valuation_case, discount_rate=0.081234567890123)
    result = compute_project_case(dc.replace(case, valuation_case=valuation))
    sheet = load_workbook(BytesIO(project_case_to_excel(result)))[
        "Assumptions & Provenance"
    ]
    audit = _audit_values(sheet)
    assert audit["provenance.project_case.valuation_case.discount_rate"] == (
        "0.081234567890123"
    )


def test_assumption_export_chunks_long_raw_text_losslessly_and_formula_safely() -> None:
    case = fx.project_case()
    raw_source = "=" + "x" * 70_000
    lifecycle = dc.replace(case.lifecycle_case, capacity_maintenance_source=raw_source)
    result = compute_project_case(dc.replace(case, lifecycle_case=lifecycle))
    sheet = load_workbook(BytesIO(project_case_to_excel(result)))[
        "Assumptions & Provenance"
    ]
    path = "provenance.project_case.lifecycle_case.capacity_maintenance_source"
    chunks = []
    for row in range(2, sheet.max_row + 1):
        if sheet.cell(row=row, column=1).value != path:
            continue
        chunks.append(
            (
                sheet.cell(row=row, column=4).value,
                sheet.cell(row=row, column=5).value,
                sheet.cell(row=row, column=2),
            )
        )

    assert [index for index, _count, _cell in chunks] == [1, 2, 3]
    assert {count for _index, count, _cell in chunks} == {3}
    assert "".join(cell.value for _index, _count, cell in chunks) == raw_source
    assert all(len(cell.value) <= 32_767 for _index, _count, cell in chunks)
    assert all(cell.data_type == "s" for _index, _count, cell in chunks)
    assert all(cell.quotePrefix is True for _index, _count, cell in chunks)


def test_assumption_export_reversibly_encodes_xml_illegal_control_characters() -> None:
    case = fx.project_case()
    raw_source = "\x00" + "x" * 70_000 + "\x01"
    lifecycle = dc.replace(
        case.lifecycle_case,
        capacity_maintenance_source=raw_source,
    )
    result = compute_project_case(dc.replace(case, lifecycle_case=lifecycle))
    sheet = load_workbook(BytesIO(project_case_to_excel(result)))[
        "Assumptions & Provenance"
    ]
    path = "provenance.project_case.lifecycle_case.capacity_maintenance_source"
    rows = [
        row
        for row in range(2, sheet.max_row + 1)
        if sheet.cell(row=row, column=1).value == path
    ]

    assert [sheet.cell(row=row, column=4).value for row in rows] == [1, 2, 3]
    assert {sheet.cell(row=row, column=5).value for row in rows} == {3}
    assert {sheet.cell(row=row, column=6).value for row in rows} == {
        "base64-utf8-surrogatepass"
    }
    assert {sheet.cell(row=row, column=7).value for row in rows} == {len(raw_source)}
    expected_bytes = raw_source.encode("utf-8", errors="surrogatepass")
    assert {sheet.cell(row=row, column=8).value for row in rows} == {
        len(expected_bytes)
    }
    encoded = "".join(sheet.cell(row=row, column=2).value for row in rows)
    reconstructed = base64.b64decode(encoded).decode(
        "utf-8", errors="surrogatepass"
    )
    assert reconstructed == raw_source
    assert all(len(sheet.cell(row=row, column=2).value) <= 32_767 for row in rows)


def test_cashflow_export_fails_closed_when_reported_p50_does_not_reconcile() -> None:
    result = _known_result()
    distribution = dc.replace(
        result.no_lifecycle_cost_screening_npv.distribution,
        p50=result.no_lifecycle_cost_screening_npv.distribution.p50 + 1.0,
    )
    screening = dc.replace(
        result.no_lifecycle_cost_screening_npv,
        distribution=distribution,
    )
    inconsistent = dc.replace(
        result,
        no_lifecycle_cost_screening_npv=screening,
    )

    with pytest.raises(ValueError, match="does not reconcile"):
        project_case_to_excel(inconsistent)


def test_existing_report_export_optionally_appends_project_case_sheets() -> None:
    index = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    prices = pd.DataFrame({"price_eur_mwh": range(24)}, index=index)
    prices.index.name = "timestamp"
    daily = pd.DataFrame({"date": ["2026-01-01"], "spread": [23.0]})
    monthly = pd.DataFrame({"month": ["2026-01"], "spread": [23.0]})
    result = _known_result()

    data = export_to_bytes(
        "GB",
        prices,
        daily,
        monthly,
        {"p50": 23.0, "p75": 23.0, "p90": 23.0, "mean": 23.0},
        {
            "annual_revenue_eur_per_mw": 1.0,
            "cycles_per_day_assumption": 1.0,
            "capture_rate_assumption": 1.0,
        },
        {"negative_hours": 0, "negative_intervals": 0, "pct_negative": 0.0},
        tz="UTC",
        project_case_result=result,
    )
    workbook = load_workbook(BytesIO(data))
    assert workbook.sheetnames[-4:] == [
        "Project Case NPVs",
        "Screening Cash Flow",
        "Lifecycle Cash Flow",
        "Assumptions & Provenance",
    ]
