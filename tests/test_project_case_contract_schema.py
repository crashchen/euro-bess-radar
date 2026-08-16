"""Project Case v1.1 ContractCase and replacement cash-flow schema."""

from __future__ import annotations

import dataclasses as dc

import pytest

from src.project_case import (
    CONTRACT_ASSET_SCOPE_V1,
    CONTRACT_QUOTE_BASIS_V1,
    CONTRACT_SETTLEMENT_FREQUENCY_V1,
    AnnualPreLifecycleStrategyCashFloor,
    CapacityMaintenanceBasis,
    CashflowRowV11,
    ContractCase,
    ContractCurrencyBasis,
    ContractCurrencyBasisMode,
    ContractQuoteStatus,
    ContractSettlementBasis,
    LifecycleCase,
    ProjectCaseValidationError,
)
from tests import pc_case_fixtures as fx


def _terms(**changes) -> AnnualPreLifecycleStrategyCashFloor:
    return dc.replace(fx.contract_case().settlement_terms, **changes)


def test_contract_case_payload_is_exact_and_curves_are_owned_tuples():
    rates = [10.0, 20.0]
    factors = [0.5, 1.0]
    terms = AnnualPreLifecycleStrategyCashFloor(
        2,
        rates,
        factors,
        CONTRACT_QUOTE_BASIS_V1,
        CONTRACT_SETTLEMENT_FREQUENCY_V1,
        CONTRACT_ASSET_SCOPE_V1,
        ContractCurrencyBasis(
            ContractCurrencyBasisMode.USER_ASSERTED_REAL_BASE_YEAR_EUR_CURVE,
            2026,
        ),
        ContractQuoteStatus.USER_SCENARIO,
        "scenario",
        "2026-08-16",
        None,
    )
    rates.append(999.0)
    factors[0] = 0.0
    assert terms.floor_rate_real_eur_per_modeled_mw_year_by_contract_year == (
        10.0,
        20.0,
    )
    assert terms.floor_entitlement_factor_by_contract_year == (0.5, 1.0)
    payload = ContractCase(
        ContractSettlementBasis.ANNUAL_PRE_LIFECYCLE_STRATEGY_CASH_FLOOR_V1,
        terms,
    ).to_payload()
    assert set(payload) == {"settlement_basis", "settlement_terms"}
    assert set(payload["settlement_terms"]) == {
        "contract_start_project_year",
        "floor_rate_real_eur_per_modeled_mw_year_by_contract_year",
        "floor_entitlement_factor_by_contract_year",
        "quote_basis",
        "settlement_frequency",
        "asset_scope",
        "currency_basis",
        "quote_status",
        "source",
        "source_as_of_date",
        "source_document_sha256",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"contract_start_project_year": True},
        {"floor_rate_real_eur_per_modeled_mw_year_by_contract_year": ()},
        {"floor_rate_real_eur_per_modeled_mw_year_by_contract_year": (-1.0, 2.0)},
        {"floor_entitlement_factor_by_contract_year": (0.5,)},
        {"floor_entitlement_factor_by_contract_year": (0.5, 1.1)},
        {"quote_basis": "EUR_PER_MWH_YEAR"},
        {"settlement_frequency": "MONTHLY"},
        {"asset_scope": "PARTIAL_PROJECT"},
        {"source_as_of_date": "2026-02-30"},
    ],
)
def test_contract_terms_fail_closed_on_invalid_domains(changes):
    with pytest.raises(ProjectCaseValidationError):
        _terms(**changes)


def test_quote_status_document_hash_null_matrix():
    with pytest.raises(ProjectCaseValidationError):
        _terms(source_document_sha256="ab" * 32)
    with pytest.raises(ProjectCaseValidationError):
        _terms(quote_status=ContractQuoteStatus.USER_ASSERTED_INDICATIVE_QUOTE)
    quoted = _terms(
        quote_status=ContractQuoteStatus.USER_ASSERTED_INDICATIVE_QUOTE,
        source_document_sha256="ab" * 32,
    )
    assert quoted.source_document_sha256 == "ab" * 32


def test_project_case_contract_cross_invariants_and_present_null_payload():
    merchant = fx.project_case()
    assert merchant.to_payload()["contract_case"] is None
    contracted = fx.project_case(contract=fx.contract_case())
    assert contracted.to_payload()["contract_case"] is not None
    assert contracted.input_fingerprint() != merchant.input_fingerprint()

    short_life = LifecycleCase(
        2,
        CapacityMaintenanceBasis.UNKNOWN,
        None,
        None,
        (),
        0.0,
        0.0,
    )
    with pytest.raises(ProjectCaseValidationError, match="inside project_life"):
        dc.replace(contracted, lifecycle_case=short_life)
    with pytest.raises(ProjectCaseValidationError, match="target_base_year"):
        fx.project_case(contract=fx.contract_case(base_year=2025))


def test_cashflow_row_v11_exact_shape_and_reconciliation():
    row = CashflowRowV11(
        year=1,
        merchant_revenue_eur=50.0,
        effective_contract_floor_eur=60.0,
        contract_top_up_eur=30.0,
        revenue_eur=80.0,
        opex_eur=10.0,
        augmentation_eur=5.0,
        terminal_eur=0.0,
        net_eur=65.0,
        discount_factor=0.9,
        discounted_net_eur=58.5,
    )
    assert set(row.to_payload()) == {
        "year",
        "merchant_revenue_eur",
        "effective_contract_floor_eur",
        "contract_top_up_eur",
        "revenue_eur",
        "opex_eur",
        "augmentation_eur",
        "terminal_eur",
        "net_eur",
        "discount_factor",
        "discounted_net_eur",
    }
    with pytest.raises(ProjectCaseValidationError, match="top_up"):
        dc.replace(row, effective_contract_floor_eur=None)
    with pytest.raises(ProjectCaseValidationError, match="reconcile"):
        dc.replace(row, revenue_eur=79.0)
    with pytest.raises(ProjectCaseValidationError, match="reconcile"):
        dc.replace(row, net_eur=64.0)
