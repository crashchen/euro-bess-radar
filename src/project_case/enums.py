"""Public enums and locked literal constants for Project Case v1 (contract §4).

Enum *values* are the exact member-name text emitted in the ``PC-CBOR-F64-v1``
fingerprint ("enums use their exact member-name text", §4.8), so ``.value`` is the
wire form.
"""

from __future__ import annotations

from enum import StrEnum

from src.project_case.fingerprint import PROFILE, SCHEMA_VERSION

__all__ = [
    "BOOTSTRAP_ALGORITHM_V1",
    "BUCKET_BLOCK_OF_DAY_4H",
    "BUCKET_HOUR_OF_DAY",
    "BUCKET_HOUR_OF_WEEK",
    "DA_ID_BUCKETS",
    "DEFAULT_SIMULATIONS",
    "EXPECTED_GRID_REGISTRY_VERSION",
    "MAINTENANCE_PROVENANCE_MODE",
    "MAX_BASE_YEAR",
    "MAX_PROJECT_LIFE_YEARS",
    "MAX_SEED",
    "MAX_SIMULATIONS",
    "MIN_BASE_YEAR",
    "MIN_SIMULATIONS",
    "PROFILE",
    "RESERVE_PRICE_AGGREGATION_V1",
    "SCHEMA_VERSION",
    "WALK_FORWARD",
    "CapacityMaintenanceBasis",
    "CurrencyBasisMode",
    "ProducerAdapterId",
    "ProjectionKind",
    "StrategyKind",
]


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


# --- Locked literal constants (§4.8, red-line #15/#25) ---

BOOTSTRAP_ALGORITHM_V1 = "pc-bootstrap-pcg64-choice365-linear-v1"
EXPECTED_GRID_REGISTRY_VERSION = "pc-market-grid-v1"
RESERVE_PRICE_AGGREGATION_V1 = "duration_weighted_mean_complete_blocks_v1"

# The pinned provenance ``mode`` string for the DA-only adapter (§4.8).
MAINTENANCE_PROVENANCE_MODE = "DA MILP Replay"

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
