"""PC-D3 contract-entry, validation-preview and disclosure contract tests.

These pin the human-readable surface of the locked v1.1 settlement contract
(``docs/design/project-case-contract-settlement-v1.1.md`` sections 8 and 11):
the panel must be able to build a ``ContractCase``, preview exactly the floor
the calculator settles, invalidate a cached result on any contract edit, and
disclose the settled terms from the ``RunResult`` alone.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
from functools import lru_cache

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import src.export as export_module
import src.pages.project_case as project_page
import src.pages.simulation_cockpit as cockpit_page
from src.project_case import (
    AssetCase,
    BootstrapCase,
    CapacityMaintenanceBasis,
    ContractQuoteStatus,
    LifecycleCase,
    ProjectCaseValidationError,
    Projection,
    ProjectionKind,
    RunResult,
    ValuationCase,
    compute_project_case,
    resolve_effective_contract_floor,
)
from src.project_case.enums import BOOTSTRAP_ALGORITHM_V1, CONTRACT_PRODUCT_DISCLOSURE_V1
from src.project_case.valuation import _resolved_contract_floor
from tests import pc_case_fixtures as fx


@lru_cache(maxsize=1)
def _contracted_result() -> RunResult:
    """A settled v1.1 result whose floor binds in every covered project year.

    The shared fixture's default rates are deliberately tiny, so this one quotes
    a floor far above the fixture's merchant cash: a disclosure test must show a
    real top-up, not a term whose floor never binds.
    """
    return compute_project_case(
        fx.project_case(
            contract=fx.contract_case(
                rates=(50_000.0, 50_000.0),
                entitlement_factors=(1.0, 0.5),
            )
        )
    )


@lru_cache(maxsize=1)
def _merchant_result() -> RunResult:
    return compute_project_case(fx.project_case())


def _contracted_result_app() -> None:
    from src.pages.project_case import render_project_case_result
    from tests.test_project_case_contract_ui import _contracted_result

    render_project_case_result(_contracted_result())


def _contracted_mirror_app() -> None:
    from src.pages.project_case import render_project_case_result
    from tests.test_project_case_contract_ui import _contracted_result

    render_project_case_result(_contracted_result(), compact=True)


def _merchant_result_app() -> None:
    from src.pages.project_case import render_project_case_result
    from tests.test_project_case_contract_ui import _merchant_result

    render_project_case_result(_merchant_result())


def _tampered_result_app() -> None:
    from src.pages.project_case import render_project_case_result
    from tests.test_project_case_contract_ui import _tampered_provenance_result

    render_project_case_result(_tampered_provenance_result())


def _contract_panel_app() -> None:
    import pandas as pd

    from src.pages.project_case import render_project_case_panel
    from tests import pc_case_fixtures as fx

    idx = pd.date_range("2026-03-10", periods=48, freq="h", tz="Europe/Berlin")
    frame = pd.DataFrame({"price_eur_mwh": [50.0] * len(idx)}, index=idx)
    frame.index.name = "timestamp"
    render_project_case_panel(
        primary_zone="DE_LU",
        primary_df=frame,
        start_date=fx.D1,
        end_date=fx.D2,
        power_mw=10.0,
        duration_hours=2.0,
        efficiency=0.88,
        capture_rate=0.9,
        capex_eur_kwh=100.0,
    )


def _tampered_provenance() -> dict[str, object]:
    """Provenance whose red-line assertion contradicts its own ContractCase."""
    provenance = dict(_contracted_result().provenance)
    assertions = dict(provenance["red_line_assertions"])
    assertions["contract_settlement_included"] = False
    provenance["red_line_assertions"] = assertions
    return provenance


def _tampered_provenance_result() -> RunResult:
    """Force the contradiction past the schema to reach the display-layer guard.

    ``RunResult`` rejects this state at construction (pinned separately), so the
    only way to exercise the renderer's defence-in-depth branch is to write the
    field directly on the frozen instance.
    """
    result = compute_project_case(
        fx.project_case(
            contract=fx.contract_case(
                rates=(50_000.0, 50_000.0),
                entitlement_factors=(1.0, 0.5),
            )
        )
    )
    object.__setattr__(result, "provenance", _tampered_provenance())
    return result


def _captions(app: AppTest) -> list[str]:
    return [caption.value for caption in app.caption]


def _table_texts(app: AppTest) -> str:
    return "\n".join(str(frame.value.to_dict()) for frame in app.dataframe)


def _expander_labels(app: AppTest) -> list[str]:
    return [expander.label for expander in app.expander]


# --------------------------------------------------------------------------- #
# Locked literals                                                             #
# --------------------------------------------------------------------------- #
def test_contract_product_disclosure_is_the_locked_literal() -> None:
    assert CONTRACT_PRODUCT_DISCLOSURE_V1 == (
        "Annual whole-project strategy-cash floor before lifecycle costs; not "
        "MACSE, not a complete legal-contract model, and not a bankable valuation."
    )


def test_panel_and_exporter_share_one_disclosure_literal() -> None:
    # Two hand-maintained copies of a locked sentence are a drift hazard; the
    # panel and the workbook must render the same constant.
    assert project_page._CONTRACT_PRODUCT_DISCLOSURE == CONTRACT_PRODUCT_DISCLOSURE_V1
    assert export_module._CONTRACT_DISCLOSURE == CONTRACT_PRODUCT_DISCLOSURE_V1


def test_settlement_basis_labels_are_literal() -> None:
    assert project_page._CONTRACT_NONE_LABEL == "No contract — merchant-only settlement"
    assert project_page._CONTRACT_FLOOR_LABEL == (
        "Annual whole-project strategy-cash floor"
    )
    assert set(project_page._QUOTE_STATUS_CHOICES.values()) == set(ContractQuoteStatus)


# --------------------------------------------------------------------------- #
# Preview cannot drift from the calculator                                     #
# --------------------------------------------------------------------------- #
def test_preview_uses_the_calculators_own_floor_resolution() -> None:
    case = fx.project_case(contract=fx.contract_case())
    terms = case.contract_case.settlement_terms
    covered, floors = resolve_effective_contract_floor(
        terms,
        power_mw=case.asset_case.power_mw,
        project_life_years=case.lifecycle_case.project_life_years,
    )
    calculator_covered, calculator_floors = _resolved_contract_floor(case)
    np.testing.assert_array_equal(covered, calculator_covered)
    np.testing.assert_array_equal(floors, calculator_floors)


def test_preview_rows_equal_the_settled_floor_the_calculator_reports() -> None:
    case = fx.project_case(contract=fx.contract_case())
    preview, covered_years = project_page._contract_floor_preview(
        case.contract_case.settlement_terms,
        power_mw=case.asset_case.power_mw,
        project_life_years=case.lifecycle_case.project_life_years,
    )
    settled = compute_project_case(case).provenance["contract_settlement"]
    resolved = {
        int(item["year"]): float(item["effective_floor_eur"])
        for item in settled["resolved_floor_by_project_year"]
    }
    assert covered_years == len(resolved)
    assert dict(
        zip(
            preview["project_year"].astype(int),
            preview["effective_whole_project_floor_eur"].astype(float),
            strict=True,
        )
    ) == resolved
    # The locked mapping is F = quoted rate x modelled MW x entitlement factor.
    assert resolved[2] == pytest.approx(10.0 * 10.0 * 0.5)
    assert resolved[3] == pytest.approx(20.0 * 10.0 * 1.0)


def test_preview_covers_only_the_contract_term() -> None:
    case = fx.project_case(contract=fx.contract_case(start_year=4, rates=(5.0,),
                                                     entitlement_factors=(1.0,)))
    preview, covered_years = project_page._contract_floor_preview(
        case.contract_case.settlement_terms,
        power_mw=case.asset_case.power_mw,
        project_life_years=case.lifecycle_case.project_life_years,
    )
    assert covered_years == 1
    assert list(preview["project_year"]) == [4]
    assert list(preview["contract_year"]) == [1]


# --------------------------------------------------------------------------- #
# Curve entry helpers                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    ["1, 2, 3", "1 2 3", "1\n2\n3", " 1;2;3 "],
)
def test_curve_parser_accepts_common_separators(text: str) -> None:
    assert project_page._parse_contract_curve(
        text, expected=3, label="floor-rate curve"
    ) == (1.0, 2.0, 3.0)


def test_curve_parser_rejects_wrong_length_instead_of_padding() -> None:
    with pytest.raises(ProjectCaseValidationError, match="exactly 3 value"):
        project_page._parse_contract_curve(
            "1, 2", expected=3, label="floor-rate curve"
        )


def test_curve_parser_rejects_non_numeric_instead_of_dropping_it() -> None:
    with pytest.raises(ProjectCaseValidationError, match="non-numeric"):
        project_page._parse_contract_curve(
            "1, n/a, 3", expected=3, label="floor-rate curve"
        )


def test_uploaded_source_document_outranks_a_typed_digest() -> None:
    payload = b"executed term sheet"
    assert project_page._resolve_source_document_sha256(payload, "ab" * 32) == (
        hashlib.sha256(payload).hexdigest()
    )
    assert project_page._resolve_source_document_sha256(None, " " + "ab" * 32 + " ") == (
        "ab" * 32
    )
    # Case is never rewritten: an uppercase digest must fail the schema check.
    assert project_page._resolve_source_document_sha256(None, "AB" * 32) == "AB" * 32
    assert project_page._resolve_source_document_sha256(None, "  ") is None


# --------------------------------------------------------------------------- #
# Cache invalidation (locked contract section 11 item 14)                      #
# --------------------------------------------------------------------------- #
def _fingerprint_kwargs() -> dict[str, object]:
    import pandas as pd

    frame = pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0]},
        index=pd.date_range("2026-03-10", periods=2, freq="h", tz="UTC"),
    )
    return {
        "primary_df": frame,
        "primary_zone": "DE_LU",
        "start_date": dt.date(2026, 3, 10),
        "end_date": dt.date(2026, 3, 11),
        "asset": AssetCase(10.0, 2.0, 0.88, 1000.0, 10.0),
        "lifecycle": LifecycleCase(
            15, CapacityMaintenanceBasis.UNKNOWN, None, None, (), 0.0, 0.0
        ),
        "projection": Projection(ProjectionKind.FlatRealProjection),
        "valuation": ValuationCase(0.08, 2026),
        "bootstrap": BootstrapCase(0, 1000, BOOTSTRAP_ALGORITHM_V1),
        "capture_rate": 0.9,
        "strategy": project_page._StrategySelection(
            project_page._DA_ONLY_LABEL,
            project_page.ProducerAdapterId.PC_ADP_DA_ONLY,
        ),
        "contract": None,
    }


def test_adding_a_contract_changes_the_request_fingerprint() -> None:
    kwargs = _fingerprint_kwargs()
    merchant = project_page._request_fingerprint(**kwargs)
    contracted = project_page._request_fingerprint(
        **{**kwargs, "contract": fx.contract_case()}
    )
    assert merchant != contracted


@pytest.mark.parametrize(
    "mutation",
    [
        {"start_year": 3},
        {"rates": (11.0, 20.0)},
        {"entitlement_factors": (0.5, 0.9)},
    ],
)
def test_every_contract_term_mutation_changes_the_request_fingerprint(
    mutation: dict[str, object],
) -> None:
    kwargs = _fingerprint_kwargs()
    baseline = project_page._request_fingerprint(
        **{**kwargs, "contract": fx.contract_case()}
    )
    assert project_page._request_fingerprint(
        **{**kwargs, "contract": fx.contract_case(**mutation)}
    ) != baseline


def test_source_status_and_as_of_mutations_change_the_request_fingerprint() -> None:
    kwargs = _fingerprint_kwargs()
    baseline_contract = fx.contract_case()
    baseline = project_page._request_fingerprint(
        **{**kwargs, "contract": baseline_contract}
    )
    terms = baseline_contract.settlement_terms
    for replacement in (
        dataclasses.replace(terms, source="a different desk quote"),
        dataclasses.replace(terms, source_as_of_date="2026-08-17"),
        dataclasses.replace(
            terms,
            quote_status=ContractQuoteStatus.USER_ASSERTED_INDICATIVE_QUOTE,
            source_document_sha256="ab" * 32,
        ),
    ):
        mutated = dataclasses.replace(
            baseline_contract, settlement_terms=replacement
        )
        assert project_page._request_fingerprint(
            **{**kwargs, "contract": mutated}
        ) != baseline


# --------------------------------------------------------------------------- #
# Disclosure surface                                                           #
# --------------------------------------------------------------------------- #
def test_contracted_result_discloses_every_minimum_locked_item() -> None:
    app = AppTest.from_function(_contracted_result_app).run(timeout=30)
    assert not app.exception
    captions = _captions(app)
    assert CONTRACT_PRODUCT_DISCLOSURE_V1 in captions
    assert any("max(merchant, effective floor)" in caption for caption in captions)
    assert any("rank-interpolated" in caption for caption in captions)
    assert any("not a zero floor" in caption for caption in captions)

    result = _contracted_result()
    terms = result.provenance["project_case"]["contract_case"]["settlement_terms"]
    tables = _table_texts(app)
    for expected in (
        result.provenance["contract_settlement"]["algorithm_version"],
        terms["quote_status"],
        terms["source"],
        terms["source_as_of_date"],
        terms["quote_basis"],
        terms["asset_scope"],
        terms["settlement_frequency"],
        terms["currency_basis"]["mode"],
        str(terms["currency_basis"]["target_base_year"]),
        result.provenance["cashflow_reconciliation"]["version"],
    ):
        assert str(expected) in tables
    # Whole-project MW is disclosed from the asset case, never re-entered.
    assert str(result.provenance["project_case"]["asset_case"]["power_mw"]) in tables


def test_absent_source_document_digest_is_shown_as_null_not_a_blank() -> None:
    app = AppTest.from_function(_contracted_result_app).run(timeout=30)
    assert not app.exception
    terms = _contracted_result().provenance["project_case"]["contract_case"][
        "settlement_terms"
    ]
    assert terms["source_document_sha256"] is None
    summary = next(
        frame.value
        for frame in app.dataframe
        if "Source document SHA-256" in list(frame.value.get("field", []))
    )
    row = summary.loc[summary["field"] == "Source document SHA-256", "value"]
    assert list(row) == ["null"]


def test_contract_cashflow_rows_expose_merchant_top_up_and_settled_cash() -> None:
    app = AppTest.from_function(_contracted_result_app).run(timeout=30)
    assert not app.exception
    result = _contracted_result()
    rows = [row.to_payload() for row in result.screening_cashflow_table.rows]
    covered = [row for row in rows if row["effective_contract_floor_eur"] is not None]
    assert covered, "fixture must cover at least one contract year"
    assert any(row["contract_top_up_eur"] > 0.0 for row in covered)
    for row in covered:
        assert row["revenue_eur"] == pytest.approx(
            row["merchant_revenue_eur"] + row["contract_top_up_eur"]
        )
    tables = _table_texts(app)
    assert "contract_top_up_eur" in tables
    assert "effective_contract_floor_eur" in tables


def test_merchant_only_result_shows_no_contract_disclosure() -> None:
    app = AppTest.from_function(_merchant_result_app).run(timeout=30)
    assert not app.exception
    captions = _captions(app)
    assert CONTRACT_PRODUCT_DISCLOSURE_V1 not in captions
    assert any(
        "contracted-floor settlement are excluded" in caption for caption in captions
    )
    assert "Contract settlement disclosure" not in _expander_labels(app)


def test_contracted_result_never_claims_settlement_is_excluded() -> None:
    app = AppTest.from_function(_contracted_result_app).run(timeout=30)
    assert not app.exception
    captions = _captions(app)
    assert not any(
        "contracted-floor settlement are excluded" in caption for caption in captions
    )
    assert any("Contract settlement IS" in caption for caption in captions)


def test_compact_mirror_discloses_that_a_floor_protected_result_is_shown() -> None:
    app = AppTest.from_function(_contracted_mirror_app).run(timeout=30)
    assert not app.exception
    captions = _captions(app)
    assert any(
        "Contract settlement applied" in caption
        and CONTRACT_PRODUCT_DISCLOSURE_V1 in caption
        for caption in captions
    )


def test_schema_rejects_an_assertion_that_contradicts_the_contract() -> None:
    result = _contracted_result()
    with pytest.raises(
        ProjectCaseValidationError, match="contract red-line assertions invalid"
    ):
        dataclasses.replace(result, provenance=_tampered_provenance())


def test_provenance_disagreement_withholds_the_disclosure_instead_of_guessing() -> None:
    app = AppTest.from_function(_tampered_result_app).run(timeout=30)
    assert not app.exception
    assert any(
        "provenance disagrees about whether a contract was settled" in error.value
        for error in app.error
    )
    assert CONTRACT_PRODUCT_DISCLOSURE_V1 not in _captions(app)
    assert "Contract settlement disclosure" not in _expander_labels(app)


def test_cockpit_wear_net_comparator_names_the_sibling_product() -> None:
    caption = cockpit_page._CONTRACTED_FLOOR_SIBLING_CAPTION
    assert "Project Case" in caption
    assert "never enters Project Case cash" in caption
    assert "wear-net" in caption


# --------------------------------------------------------------------------- #
# End-to-end panel run                                                         #
# --------------------------------------------------------------------------- #
def _run_contract_panel(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setattr(project_page, "emit_da_only", lambda *a, **k: fx.da_only_srr())
    app = AppTest.from_function(_contract_panel_app).run(timeout=30)
    assert not app.exception
    app.selectbox(key="pc_contract_mode").set_value(
        project_page._CONTRACT_FLOOR_LABEL
    ).run(timeout=30)
    assert not app.exception
    app.number_input(key="pc_contract_tenor").set_value(2).run(timeout=30)
    app.number_input(key="pc_contract_flat_rate").set_value(10_000.0).run(timeout=30)
    app.text_input(key="pc_contract_source").set_value("desk indication").run(
        timeout=30
    )
    assert not app.exception
    return app


def test_panel_builds_and_settles_a_contract_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _run_contract_panel(monkeypatch)
    # The validated preview is visible before the run, from the same resolution
    # the calculator will settle against.
    preview = next(
        frame.value
        for frame in app.dataframe
        if "effective_whole_project_floor_eur" in frame.value.columns
    )
    assert list(preview["project_year"]) == [1, 2]
    assert list(preview["effective_whole_project_floor_eur"]) == [100_000.0, 100_000.0]

    app.button(key="pc_run").click().run(timeout=30)
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert any(project_page.SCREENING_NPV_LABEL in label for label in metrics)
    assert CONTRACT_PRODUCT_DISCLOSURE_V1 in _captions(app)


def test_panel_contract_run_is_floor_protected_versus_merchant_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(project_page, "emit_da_only", lambda *a, **k: fx.da_only_srr())
    app = AppTest.from_function(_contract_panel_app).run(timeout=30)
    app.button(key="pc_run").click().run(timeout=30)
    assert not app.exception
    merchant_p50 = {metric.label: metric.value for metric in app.metric}[
        f"{project_page.SCREENING_NPV_LABEL} — {project_page.P50_LABEL}"
    ]

    app = _run_contract_panel(monkeypatch)
    app.button(key="pc_run").click().run(timeout=30)
    assert not app.exception
    contracted_p50 = {metric.label: metric.value for metric in app.metric}[
        f"{project_page.SCREENING_NPV_LABEL} — {project_page.P50_LABEL}"
    ]

    def _euro(text: str) -> float:
        return float(text.replace("€", "").replace(",", ""))

    assert _euro(contracted_p50) > _euro(merchant_p50)


def test_contract_edit_invalidates_a_cached_result_before_it_renders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _run_contract_panel(monkeypatch)
    app.button(key="pc_run").click().run(timeout=30)
    assert not app.exception
    assert app.metric, "a settled result must render before the invalidation check"

    app.number_input(key="pc_contract_flat_rate").set_value(20_000.0).run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 0
    assert any("stale result is hidden" in warning.value for warning in app.warning)
    # The input-side product boundary stays visible; what must disappear is the
    # settled result and its RunResult-derived disclosure.
    assert "Contract settlement disclosure" not in _expander_labels(app)


def test_contract_term_outside_project_life_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _run_contract_panel(monkeypatch)
    app.number_input(key="pc_contract_start_year").set_value(20).run(timeout=30)
    app.number_input(key="pc_contract_tenor").set_value(5).run(timeout=30)
    assert not app.exception
    assert any(
        "beyond the 20-year project life" in error.value for error in app.error
    )
    assert len(app.metric) == 0


def test_missing_contract_source_fails_closed_not_back_to_merchant_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(project_page, "emit_da_only", lambda *a, **k: fx.da_only_srr())
    app = AppTest.from_function(_contract_panel_app).run(timeout=30)
    app.selectbox(key="pc_contract_mode").set_value(
        project_page._CONTRACT_FLOOR_LABEL
    ).run(timeout=30)
    assert not app.exception
    # Source is required by the locked schema; an empty one must not quietly
    # degrade into an unprotected merchant-only run.
    assert any("contract source" in error.value for error in app.error)
    assert len(app.metric) == 0
    assert not [
        button for button in app.button if button.key == "pc_run"
    ], "the run button must not be reachable while the contract is invalid"


def test_asserted_quote_status_requires_a_source_document_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _run_contract_panel(monkeypatch)
    status_label = next(
        label
        for label, status in project_page._QUOTE_STATUS_CHOICES.items()
        if status is ContractQuoteStatus.USER_ASSERTED_EXECUTED_SOURCE_DOCUMENT
    )
    app.selectbox(key="pc_contract_quote_status").set_value(status_label).run(timeout=30)
    assert not app.exception
    assert any(
        "source_document_sha256" in error.value for error in app.error
    )
    assert len(app.metric) == 0

    app.text_input(key="pc_contract_source_sha256").set_value("ab" * 32).run(timeout=30)
    assert not app.exception
    assert not app.error
    app.button(key="pc_run").click().run(timeout=30)
    assert not app.exception
    assert CONTRACT_PRODUCT_DISCLOSURE_V1 in _captions(app)


def test_unknown_maintenance_basis_still_settles_screening_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _run_contract_panel(monkeypatch)
    app.button(key="pc_run").click().run(timeout=30)
    assert not app.exception
    # The panel default is the UNKNOWN capacity-maintenance basis: settlement
    # applies to screening cash, while lifecycle stays typed-unavailable and is
    # never shown as EUR 0.
    assert len(app.metric) == 4
    assert any(
        "capacity_maintenance_unknown" in warning.value for warning in app.warning
    )
    assert CONTRACT_PRODUCT_DISCLOSURE_V1 in _captions(app)
