"""Canonical, immutable producer 5-tuple registry (contract §5, §4.8).

Eligibility is a property of *which adapter emits* a StrategyRunResult, so the
pinned ``(ProducerAdapterId, StrategyKind, source function, per-day cash field,
excluded fields)`` tuple is the single source of truth. It lives in its own module
so both ``schema`` (to *validate* a StrategyRunResult's provenance against it) and
``adapters`` (to *drive* the adapters) can import it without a cycle, and so the
table cannot be mutated at runtime (``MappingProxyType``).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from src.project_case.enums import ProducerAdapterId, StrategyKind

__all__ = ["KIND_TO_ADAPTER", "SPECS", "AdapterSpec", "canonical_spec"]


@dataclass(frozen=True)
class AdapterSpec:
    """Pinned producer 5-tuple (§5). ``source_function`` is provenance, not a call."""

    producer_adapter_id: ProducerAdapterId
    strategy_kind: StrategyKind
    source_function: str
    per_day_cash_field: str
    excluded_fields: tuple[str, ...]
    consumes_ida: bool
    consumes_reserve: bool


_SPEC_LIST: tuple[AdapterSpec, ...] = (
    AdapterSpec(
        ProducerAdapterId.PC_ADP_DA_ONLY,
        StrategyKind.DA_ONLY,
        "simulate_replay_batch",
        "total_revenue_eur",
        ("degradation_cost_eur",),
        consumes_ida=False,
        consumes_reserve=False,
    ),
    AdapterSpec(
        ProducerAdapterId.PC_ADP_DA_ID,
        StrategyKind.DA_ID_FORECAST,
        "simulate_sequential_da_id_batch",
        "realised_eur",
        ("ceiling_eur",),
        consumes_ida=True,
        consumes_reserve=False,
    ),
    AdapterSpec(
        ProducerAdapterId.PC_ADP_RESERVE_COOPT,
        StrategyKind.DA_RESERVE_COOPT,
        "solve_joint_capacity_batch",
        "joint_total_revenue",
        (),
        consumes_ida=False,
        consumes_reserve=True,
    ),
    AdapterSpec(
        ProducerAdapterId.PC_ADP_DA_ID_RESERVE,
        StrategyKind.DA_ID_RESERVE_REALISED,
        "simulate_sequential_da_id_reserve_batch",
        "realised_eur",
        ("reserve_first_ceiling_eur", "global_ceiling_eur"),
        consumes_ida=True,
        consumes_reserve=True,
    ),
)

SPECS: MappingProxyType[ProducerAdapterId, AdapterSpec] = MappingProxyType(
    {spec.producer_adapter_id: spec for spec in _SPEC_LIST}
)
KIND_TO_ADAPTER: MappingProxyType[StrategyKind, ProducerAdapterId] = MappingProxyType(
    {spec.strategy_kind: spec.producer_adapter_id for spec in _SPEC_LIST}
)


def canonical_spec(producer_adapter_id: ProducerAdapterId) -> AdapterSpec:
    """Return the pinned 5-tuple for an adapter id (raises on an unknown id)."""
    return SPECS[producer_adapter_id]
