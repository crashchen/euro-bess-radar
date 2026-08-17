"""Project Case v1.1: typed inputs, valuation, and annual contract settlement.

This package implements the PC-A increment of ``docs/design/project-case-v1.md``
(the locked contract): the typed case schema, the public ``StrategyKind`` /
``ProducerAdapterId`` enums, the four producer-issued solver adapters that emit a
``StrategyRunResult``, ``validate()``, the leg-specific market-grid registry, and
the deterministic ``PC-CBOR-F64-v1`` canonical-serialisation fingerprint.

The package remains pure (no UI or Streamlit). PC-A owns producer adapters,
audit, and the unchanged v1 ``StrategyRunResult`` fingerprint; PC-B owns the
bootstrap-driven lifecycle valuation; PC-D1+D2 adds the v1.1 ``ContractCase``
schema and per-draw/per-year strategy-cash settlement while preserving the
merchant-only numerical path.
"""

from __future__ import annotations

from src.project_case import grid
from src.project_case.adapters import (
    PC_A_CALCULATOR_VERSION,
    SPECS,
    AdapterSpec,
    emit_da_id,
    emit_da_id_reserve,
    emit_da_only,
    emit_reserve_coopt,
)
from src.project_case.audit import AdapterUnavailableError
from src.project_case.bootstrap import bootstrap_annual_sums
from src.project_case.enums import (
    BOOTSTRAP_ALGORITHM_V1,
    CASHFLOW_RECONCILIATION_ABS_TOL_EUR_V1,
    CASHFLOW_RECONCILIATION_REL_TOL_V1,
    CASHFLOW_RECONCILIATION_VERSION_V1,
    CONTRACT_ASSET_SCOPE_V1,
    CONTRACT_CASHFLOW_TABLE_STATISTIC_V1,
    CONTRACT_QUOTE_BASIS_V1,
    CONTRACT_SETTLEMENT_ALGORITHM_V1,
    CONTRACT_SETTLEMENT_FREQUENCY_V1,
    EXPECTED_GRID_REGISTRY_VERSION,
    NULL_CASHFLOW_TABLE_STATISTIC_V1,
    PC_D2_CALCULATOR_VERSION,
    PROFILE,
    PROJECT_CASE_SCHEMA_VERSION,
    RUN_RESULT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    STRATEGY_RUN_RESULT_SCHEMA_VERSION,
    CapacityMaintenanceBasis,
    ContractCurrencyBasisMode,
    ContractQuoteStatus,
    ContractSettlementBasis,
    CurrencyBasisMode,
    ProducerAdapterId,
    ProjectionKind,
    StrategyKind,
)
from src.project_case.fingerprint import (
    encode_envelope,
    encode_value,
    fingerprint_hex,
    sorted_by_encoding,
)
from src.project_case.schema import (
    AdapterProvenance,
    AnnualPreLifecycleStrategyCashFloor,
    AssetCase,
    AugmentationEvent,
    BootstrapCase,
    CaptureBasis,
    CashBasis,
    CashflowRow,
    CashflowRowV11,
    CashflowTable,
    ContractCase,
    ContractCurrencyBasis,
    CoverageAudit,
    CurrencyBasis,
    ForecastAudit,
    ForecastAudits,
    LifecycleCase,
    LiquidityBasis,
    MarketCase,
    NpvDistribution,
    NpvOutcome,
    ProjectCase,
    ProjectCaseValidationError,
    Projection,
    ReserveCoverageAudit,
    ReserveCoverageEntry,
    RunResult,
    SampleWindow,
    SolverFailureDetail,
    StrategyRunResult,
    ValuationCase,
)
from src.project_case.valuation import (
    PC_B_CALCULATOR_VERSION,
    compute_project_case,
    resolve_effective_contract_floor,
)

__all__ = [
    "BOOTSTRAP_ALGORITHM_V1",
    "CASHFLOW_RECONCILIATION_ABS_TOL_EUR_V1",
    "CASHFLOW_RECONCILIATION_REL_TOL_V1",
    "CASHFLOW_RECONCILIATION_VERSION_V1",
    "CONTRACT_ASSET_SCOPE_V1",
    "CONTRACT_CASHFLOW_TABLE_STATISTIC_V1",
    "CONTRACT_QUOTE_BASIS_V1",
    "CONTRACT_SETTLEMENT_ALGORITHM_V1",
    "CONTRACT_SETTLEMENT_FREQUENCY_V1",
    "EXPECTED_GRID_REGISTRY_VERSION",
    "NULL_CASHFLOW_TABLE_STATISTIC_V1",
    "PC_A_CALCULATOR_VERSION",
    "PC_B_CALCULATOR_VERSION",
    "PC_D2_CALCULATOR_VERSION",
    "PROFILE",
    "PROJECT_CASE_SCHEMA_VERSION",
    "RUN_RESULT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SPECS",
    "STRATEGY_RUN_RESULT_SCHEMA_VERSION",
    "AdapterProvenance",
    "AdapterSpec",
    # adapters / audit / grid / bootstrap
    "AdapterUnavailableError",
    "AnnualPreLifecycleStrategyCashFloor",
    # schema
    "AssetCase",
    "AugmentationEvent",
    "BootstrapCase",
    "CapacityMaintenanceBasis",
    "CaptureBasis",
    "CashBasis",
    "CashflowRow",
    "CashflowRowV11",
    "CashflowTable",
    "ContractCase",
    "ContractCurrencyBasis",
    "ContractCurrencyBasisMode",
    "ContractQuoteStatus",
    "ContractSettlementBasis",
    "CoverageAudit",
    "CurrencyBasis",
    "CurrencyBasisMode",
    "ForecastAudit",
    "ForecastAudits",
    "LifecycleCase",
    "LiquidityBasis",
    "MarketCase",
    "NpvDistribution",
    "NpvOutcome",
    "ProducerAdapterId",
    "ProjectCase",
    "ProjectCaseValidationError",
    "Projection",
    "ProjectionKind",
    "ReserveCoverageAudit",
    "ReserveCoverageEntry",
    "RunResult",
    "SampleWindow",
    "SolverFailureDetail",
    # enums / literals
    "StrategyKind",
    "StrategyRunResult",
    "ValuationCase",
    "bootstrap_annual_sums",
    "compute_project_case",
    "emit_da_id",
    "emit_da_id_reserve",
    "emit_da_only",
    "emit_reserve_coopt",
    "encode_envelope",
    # fingerprint
    "encode_value",
    "fingerprint_hex",
    "grid",
    "resolve_effective_contract_floor",
    "sorted_by_encoding",
]
