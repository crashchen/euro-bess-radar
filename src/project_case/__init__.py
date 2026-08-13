"""Project Case v1: typed inputs/adapters (PC-A) and lifecycle valuation (PC-B).

This package implements the PC-A increment of ``docs/design/project-case-v1.md``
(the locked contract): the typed case schema, the public ``StrategyKind`` /
``ProducerAdapterId`` enums, the four producer-issued solver adapters that emit a
``StrategyRunResult``, ``validate()``, the leg-specific market-grid registry, and
the deterministic ``PC-CBOR-F64-v1`` canonical-serialisation fingerprint.

The package remains pure (no UI or Streamlit). PC-A owns the input schema,
producer adapters, audit and fingerprint; PC-B owns the bootstrap-driven,
year-by-year lifecycle cash-flow and the two typed NPV outcomes.
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
    EXPECTED_GRID_REGISTRY_VERSION,
    PROFILE,
    SCHEMA_VERSION,
    CapacityMaintenanceBasis,
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
    AssetCase,
    AugmentationEvent,
    BootstrapCase,
    CaptureBasis,
    CashBasis,
    CashflowRow,
    CashflowTable,
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
from src.project_case.valuation import PC_B_CALCULATOR_VERSION, compute_project_case

__all__ = [
    "BOOTSTRAP_ALGORITHM_V1",
    "EXPECTED_GRID_REGISTRY_VERSION",
    "PC_A_CALCULATOR_VERSION",
    "PC_B_CALCULATOR_VERSION",
    "PROFILE",
    "SCHEMA_VERSION",
    "SPECS",
    "AdapterProvenance",
    "AdapterSpec",
    # adapters / audit / grid / bootstrap
    "AdapterUnavailableError",
    # schema
    "AssetCase",
    "AugmentationEvent",
    "BootstrapCase",
    "CapacityMaintenanceBasis",
    "CaptureBasis",
    "CashBasis",
    "CashflowRow",
    "CashflowTable",
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
    "sorted_by_encoding",
]
