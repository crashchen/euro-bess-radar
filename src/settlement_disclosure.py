"""Display the capacity-settlement convention of an existing model result.

This module describes implemented model contracts, not market billing rules.
It neither changes cash nor adds fields to the fingerprinted Project Case
payload. The pinned profile below must be reviewed when that contract changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from src.config import ZONE_TIMEZONES

if TYPE_CHECKING:
    from src.project_case import RunResult

_NOMINAL_REGISTRY = "pc-market-grid-v1"
_NOMINAL_PROFILE = "pc-reserve-block-of-day-4h-v1"
_RESERVE_ADAPTERS = frozenset({"PC_ADP_RESERVE_COOPT", "PC_ADP_DA_ID_RESERVE"})
_ENERGY_ADAPTERS = frozenset({"PC_ADP_DA_ONLY", "PC_ADP_DA_ID"})


def _screening_capacity_basis_kind(
    zone: str | None, product: str | None,
) -> Literal["physical", "unverified"] | None:
    """One classification shared by compact table labels and full disclosure."""
    if not product or not str(product).strip():
        return None
    return "physical" if zone in ZONE_TIMEZONES else "unverified"


def screening_capacity_settlement_label(
    zone: str | None, product: str | None,
) -> str | None:
    """Compact strategy-table label for the same scope as the full text."""
    kind = _screening_capacity_basis_kind(zone, product)
    if kind is None:
        return None
    return "Physical delivery hours" if kind == "physical" else "Unverified"


def screening_capacity_settlement_basis(
    zone: str | None, product: str | None,
) -> str | None:
    """Describe a selected product's existing physical-hour screening path.

    Call only for a capacity-bearing result, not activation/imbalance overlays
    or an annual standalone fee estimate. A product name does not establish a
    market settlement profile; the disclosure intentionally makes no such claim.
    """
    kind = _screening_capacity_basis_kind(zone, product)
    if kind is None:
        return None
    scope = f"{zone or 'Unspecified zone'} / {product}"
    if kind == "unverified":
        return (
            f"{scope}: capacity settlement basis unverified; the zone is not "
            "registered. No product-specific settlement convention is asserted."
        )
    text = (
        f"{scope}: physical-hour screening; capacity cash uses elapsed delivery "
        "hours and availability, with no nominal-block DST adjustment. "
        "Energy/SoC use physical time."
    )
    if zone == "DE_LU":
        text += (
            " The registered DE_LU Project Case profile uses six nominal 4h blocks/day."
        )
    return text


def project_case_strategy_capacity_settlement_basis(
    strategy: Mapping[str, Any],
) -> str | None:
    """Read settlement scope from the recorded StrategyRunResult payload.

    Unknown reserve profiles remain visible as unverified. Do not infer a new
    profile from the current sidebar zone or from a familiar product label.
    """
    adapter = strategy.get("adapter_provenance", {})
    adapter_id = adapter.get("producer_adapter_id")
    if adapter_id in _ENERGY_ADAPTERS:
        return None
    zone = strategy.get("zone")
    product = strategy.get("reserve_product")
    profile = adapter.get("expected_grid_profiles", {}).get("reserve")
    registry = adapter.get("expected_grid_registry_version")
    scope = f"{zone or 'Unspecified zone'} / {product or 'Unspecified reserve product'}"
    if (
        adapter_id in _RESERVE_ADAPTERS
        and zone == "DE_LU"
        and product
        and registry == _NOMINAL_REGISTRY
        and profile == _NOMINAL_PROFILE
    ):
        return (
            f"{scope}: Project Case uses six nominal 4h capacity blocks per local "
            "day, including DST, with availability applied. Energy and SoC use "
            "physical delivery time; screening capacity uses elapsed hours "
            "without this adjustment."
        )
    return (
        f"{scope}: capacity settlement basis unverified for recorded adapter "
        f"{adapter_id or 'unspecified'}, registry {registry or 'unspecified'}, "
        f"reserve profile {profile or 'unspecified'}. No nominal 4h convention "
        "is asserted."
    )


def project_case_capacity_settlement_basis(result: RunResult) -> str | None:
    """Describe a displayed RunResult using its immutable run provenance."""
    return project_case_strategy_capacity_settlement_basis(
        result.provenance["strategy_run_result"]
    )
