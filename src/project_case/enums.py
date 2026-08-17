"""Public enums and locked literal constants for Project Case v1 (contract §4).

Enum *values* are the exact member-name text emitted in the ``PC-CBOR-F64-v1``
fingerprint ("enums use their exact member-name text", §4.8), so ``.value`` is the
wire form.
"""

from __future__ import annotations

from enum import StrEnum

from src.project_case.fingerprint import (
    PROFILE,
    PROJECT_CASE_SCHEMA_VERSION,
    RUN_RESULT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    STRATEGY_RUN_RESULT_SCHEMA_VERSION,
)

__all__ = [
    "BOOTSTRAP_ALGORITHM_V1",
    "BUCKET_BLOCK_OF_DAY_4H",
    "BUCKET_HOUR_OF_DAY",
    "BUCKET_HOUR_OF_WEEK",
    "CASHFLOW_RECONCILIATION_ABS_TOL_EUR_V1",
    "CASHFLOW_RECONCILIATION_REL_TOL_V1",
    "CASHFLOW_RECONCILIATION_VERSION_V1",
    "CONTRACT_ASSET_SCOPE_V1",
    "CONTRACT_CASHFLOW_TABLE_STATISTIC_V1",
    "CONTRACT_PRODUCT_DISCLOSURE_V1",
    "CONTRACT_QUOTE_BASIS_V1",
    "CONTRACT_SETTLEMENT_ALGORITHM_V1",
    "CONTRACT_SETTLEMENT_FREQUENCY_V1",
    "DA_ID_BUCKETS",
    "DEFAULT_SIMULATIONS",
    "EXPECTED_GRID_REGISTRY_VERSION",
    "LIFECYCLE_UNKNOWN_MESSAGE",
    "LIFECYCLE_UNKNOWN_STATUS",
    "MAINTENANCE_PROVENANCE_MODE",
    "MAX_BASE_YEAR",
    "MAX_PROJECT_LIFE_YEARS",
    "MAX_SEED",
    "MAX_SIMULATIONS",
    "MIN_BASE_YEAR",
    "MIN_SIMULATIONS",
    "NULL_CASHFLOW_TABLE_STATISTIC_V1",
    "PC_A_CALCULATOR_VERSION",
    "PC_D2_CALCULATOR_VERSION",
    "PROFILE",
    "PROJECT_CASE_SCHEMA_VERSION",
    "RESERVE_PRICE_AGGREGATION_V1",
    "RUN_RESULT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "STRATEGY_RUN_RESULT_SCHEMA_VERSION",
    "WALK_FORWARD",
    "CapacityMaintenanceBasis",
    "ContractCurrencyBasisMode",
    "ContractQuoteStatus",
    "ContractSettlementBasis",
    "CurrencyBasisMode",
    "ProducerAdapterId",
    "ProjectionKind",
    "StrategyKind",
]


PC_A_CALCULATOR_VERSION = "pc-a-v1"


class StrategyKind(StrEnum):
    """Public identity of a cash-eligible strategy (§4.3). Stable public API."""

    DA_ONLY = "DA_ONLY"
    DA_ID_FORECAST = "DA_ID_FORECAST"
    DA_RESERVE_COOPT = "DA_RESERVE_COOPT"
    DA_ID_RESERVE_REALISED = "DA_ID_RESERVE_REALISED"


class ProducerAdapterId(StrEnum):
    """Pinned producer-adapter identity (§5). Eligibility is a property of this."""

    PC_ADP_DA_ONLY = "PC_ADP_DA_ONLY"
    PC_ADP_DA_ID = "PC_ADP_DA_ID"
    PC_ADP_RESERVE_COOPT = "PC_ADP_RESERVE_COOPT"
    PC_ADP_DA_ID_RESERVE = "PC_ADP_DA_ID_RESERVE"


class CapacityMaintenanceBasis(StrEnum):
    """Three-state maintenance basis (§4.2, red-line #4)."""

    SCHEDULED_NAMEPLATE_MAINTENANCE = "SCHEDULED_NAMEPLATE_MAINTENANCE"
    NO_AUGMENTATION_REQUIRED_ASSERTED = "NO_AUGMENTATION_REQUIRED_ASSERTED"
    UNKNOWN = "UNKNOWN"


class ProjectionKind(StrEnum):
    """Projection mode (§4.7, red-line #9)."""

    FlatRealProjection = "FlatRealProjection"
    DAOnlySpreadDecay = "DAOnlySpreadDecay"
    ExplicitAnnualMultiplierCurve = "ExplicitAnnualMultiplierCurve"


class CurrencyBasisMode(StrEnum):
    """Currency-basis mode (§4.3/§4.4, red-line #19)."""

    DEFLATOR_APPLIED = "DEFLATOR_APPLIED"
    SOURCE_EUR_TREATED_AS_BASE_YEAR_REAL = "SOURCE_EUR_TREATED_AS_BASE_YEAR_REAL"


class ContractSettlementBasis(StrEnum):
    """The sole Project Case v1.1 cash-NPV-eligible settlement union member."""

    ANNUAL_PRE_LIFECYCLE_STRATEGY_CASH_FLOOR_V1 = "ANNUAL_PRE_LIFECYCLE_STRATEGY_CASH_FLOOR_V1"


class ContractCurrencyBasisMode(StrEnum):
    """The deliberately narrow v1.1 floor-curve currency assertion."""

    USER_ASSERTED_REAL_BASE_YEAR_EUR_CURVE = "USER_ASSERTED_REAL_BASE_YEAR_EUR_CURVE"


class ContractQuoteStatus(StrEnum):
    """User-asserted maturity of the source supporting the floor curve."""

    USER_SCENARIO = "USER_SCENARIO"
    USER_ASSERTED_INDICATIVE_QUOTE = "USER_ASSERTED_INDICATIVE_QUOTE"
    USER_ASSERTED_EXECUTED_SOURCE_DOCUMENT = "USER_ASSERTED_EXECUTED_SOURCE_DOCUMENT"


# --- Locked literal constants (§4.8, red-line #15/#25) ---

BOOTSTRAP_ALGORITHM_V1 = "pc-bootstrap-pcg64-choice365-linear-v1"
EXPECTED_GRID_REGISTRY_VERSION = "pc-market-grid-v1"
RESERVE_PRICE_AGGREGATION_V1 = "duration_weighted_mean_complete_blocks_v1"
CONTRACT_QUOTE_BASIS_V1 = "EUR_PER_MODELED_PROJECT_MW_YEAR"
CONTRACT_SETTLEMENT_FREQUENCY_V1 = "ANNUAL_PROJECT_YEAR_END"
CONTRACT_ASSET_SCOPE_V1 = "WHOLE_PROJECT_MODELED_MW"
CONTRACT_SETTLEMENT_ALGORITHM_V1 = "pc-annual-pre-lifecycle-strategy-cash-floor-v1"
CONTRACT_CASHFLOW_TABLE_STATISTIC_V1 = "p50_npv_rank_interpolated_cashflow_linear_v1"
CASHFLOW_RECONCILIATION_VERSION_V1 = "pc-cashflow-p50-reconciliation-v1"
CASHFLOW_RECONCILIATION_REL_TOL_V1 = 1e-10
CASHFLOW_RECONCILIATION_ABS_TOL_EUR_V1 = 1e-6
NULL_CASHFLOW_TABLE_STATISTIC_V1 = "p50_annual_bootstrap_draw_linear"
PC_D2_CALCULATOR_VERSION = "pc-d2-v1.1"

# The exact settlement product boundary the locked contract (§8) requires on
# every human-readable surface.  Excel and the Streamlit panel both render this
# one constant so the two disclosures can never drift apart.
CONTRACT_PRODUCT_DISCLOSURE_V1 = (
    "Annual whole-project strategy-cash floor before lifecycle costs; not MACSE, "
    "not a complete legal-contract model, and not a bankable valuation."
)

# The pinned provenance ``mode`` string for the DA-only adapter (§4.8).
MAINTENANCE_PROVENANCE_MODE = "DA MILP Replay"

# The ONLY legal lifecycle-unavailable envelope, returned when the
# capacity-maintenance basis is UNKNOWN (§3, §4.6). Screening stays available;
# the lifecycle NpvOutcome is fixed to exactly this status + message (no other
# unavailable status/message is a valid RunResult state).
LIFECYCLE_UNKNOWN_STATUS = "capacity_maintenance_unknown"
LIFECYCLE_UNKNOWN_MESSAGE = "Engineering capacity-maintenance basis is unknown."

# Forecast-mode gate (§5, red-line #10): cash-eligible only in walk-forward.
WALK_FORWARD = "walk_forward"

BUCKET_HOUR_OF_DAY = "hour_of_day"
BUCKET_HOUR_OF_WEEK = "hour_of_week"
BUCKET_BLOCK_OF_DAY_4H = "block_of_day_4h"
# The DA/IDA leg buckets share one selected value from this set (§4.3).
DA_ID_BUCKETS = frozenset({BUCKET_HOUR_OF_DAY, BUCKET_HOUR_OF_WEEK})

MIN_SIMULATIONS = 1000
MAX_SIMULATIONS = 50000
DEFAULT_SIMULATIONS = 5000

MAX_PROJECT_LIFE_YEARS = 100

MIN_BASE_YEAR = 1900
MAX_BASE_YEAR = 9999

MAX_SEED = 2**64 - 1
