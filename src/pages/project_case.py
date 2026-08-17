"""PC-C Project Case UI orchestration and RunResult presentation.

The page is deliberately split into two layers:

* :func:`render_project_case_panel` is the single input/run owner in the Revenue
  tab.  It builds one selected, data-supported producer-issued
  ``StrategyRunResult``, then a validated ``ProjectCase`` and finally calls the
  current Project Case valuation kernel
  exactly once on an explicit button click.
* :func:`render_project_case_result` is a pure ``RunResult`` consumer.  The
  Revenue tab and Simulation Cockpit both use it, so neither reconstructs NPV
  arithmetic from ambient widgets or a strategy-comparison table.

The legacy wear-net contracted-floor comparator remains separate and never
enters this module's inputs, result cache, or export.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import pandas as pd
import streamlit as st

from src.project_case import (
    BOOTSTRAP_ALGORITHM_V1,
    EXPECTED_GRID_REGISTRY_VERSION,
    PC_A_CALCULATOR_VERSION,
    PC_D2_CALCULATOR_VERSION,
    PROJECT_CASE_SCHEMA_VERSION,
    AnnualPreLifecycleStrategyCashFloor,
    AssetCase,
    BootstrapCase,
    CapacityMaintenanceBasis,
    ContractCase,
    ContractCurrencyBasis,
    ContractCurrencyBasisMode,
    ContractQuoteStatus,
    ContractSettlementBasis,
    CurrencyBasis,
    CurrencyBasisMode,
    LifecycleCase,
    MarketCase,
    NpvOutcome,
    ProducerAdapterId,
    ProjectCase,
    ProjectCaseValidationError,
    Projection,
    ProjectionKind,
    RunResult,
    StrategyRunResult,
    ValuationCase,
    compute_project_case,
    emit_da_id,
    emit_da_id_reserve,
    emit_da_only,
    emit_reserve_coopt,
    encode_value,
    grid,
    resolve_effective_contract_floor,
)
from src.project_case.audit import AdapterUnavailableError
from src.project_case.enums import (
    BUCKET_HOUR_OF_DAY,
    BUCKET_HOUR_OF_WEEK,
    CONTRACT_ASSET_SCOPE_V1,
    CONTRACT_PRODUCT_DISCLOSURE_V1,
    CONTRACT_QUOTE_BASIS_V1,
    CONTRACT_SETTLEMENT_FREQUENCY_V1,
    DEFAULT_SIMULATIONS,
    MAX_PROJECT_LIFE_YEARS,
    MAX_SIMULATIONS,
    MIN_SIMULATIONS,
)
from src.project_case.imports import (
    AUGMENTATION_TEMPLATE_CSV,
    explicit_multiplier_template_csv,
    parse_augmentation_csv,
    parse_explicit_multiplier_csv,
)

SCREENING_NPV_LABEL: Final = "No-lifecycle-cost screening NPV"
LIFECYCLE_NPV_LABEL: Final = "Pre-tax unlevered lifecycle cash NPV"
P10_LABEL: Final = "P10 (Downside)"
P50_LABEL: Final = "P50 (Median)"
P90_LABEL: Final = "P90 (Upside)"
PROBABILITY_LABEL: Final = "P(NPV > 0)"

_CACHE_KEY: Final = "project_case_pc_c_cache"
_P50_TABLE_CAPTION: Final = (
    "Representative cash-flow tables reconcile to the linear P50 NPV. "
    "Merchant-only cases use the linear P50 annual bootstrap draw; contracted "
    "cases use the rank-interpolated settled path. They are neither expected-value "
    "tables nor year-wise medians. Year 0 CapEx is excluded from the rows: "
    "NPV P50 = -CapEx + sum(discounted net cash flow)."
)

_DA_ONLY_LABEL: Final = "DA-only realised"
_DA_ID_LABEL: Final = "DA + IDA1 forecast-driven"
_DA_RESERVE_LABEL: Final = "DA + reserve co-optimised"
_DA_ID_RESERVE_LABEL: Final = "DA + IDA1 + reserve realistic"

# --- PC-D3 contract-entry vocabulary ---------------------------------------- #
_CONTRACT_NONE_LABEL: Final = "No contract — merchant-only settlement"
_CONTRACT_FLOOR_LABEL: Final = "Annual whole-project strategy-cash floor"
_CONTRACT_CURVE_FLAT: Final = "Flat real quote"
_CONTRACT_CURVE_ESCALATING: Final = "Escalating real quote"
_CONTRACT_CURVE_EXPLICIT: Final = "Explicit per-contract-year values"
_CONTRACT_FACTOR_FULL: Final = "100% — quoted rates already net of availability"
_CONTRACT_FACTOR_FLAT: Final = "Flat entitlement factor"
_CONTRACT_FACTOR_EXPLICIT: Final = "Explicit per-contract-year factors"

# The exporter renders the same locked literal; both import it from the schema
# package so a wording drift is impossible (locked contract section 8).
_CONTRACT_PRODUCT_DISCLOSURE: Final = CONTRACT_PRODUCT_DISCLOSURE_V1

# Coexistence disclosure required by PC-D3: the cockpit ships a DIFFERENT,
# wear-net screening comparator under a similar name.
_CONTRACT_SIBLING_DISCLOSURE: Final = (
    "Different product from the cockpit's 'Contracted floor versus merchant cash "
    "flow' panel. That comparator floors a wear-net DA-only EUR/MW/yr screening "
    "baseline and never enters Project Case cash. This settlement floors the "
    "selected producer's gross pre-lifecycle strategy cash for every bootstrap "
    "draw and project year, and it does change the NPVs reported above."
)
_CONTRACT_TOP_UP_DISCLOSURE: Final = (
    "Settled cash is max(merchant, effective floor) per draw and project year — "
    "never merchant plus floor. Merchant cash is not clamped at zero first, so a "
    "loss-making year inside the term can require a top-up larger than the quoted "
    "floor. Capped top-up, partial contracted MW, fees, sharing, penalties, tax, "
    "and debt are separate products that this basis does not model."
)
_CONTRACT_ENTITLEMENT_DISCLOSURE: Final = (
    "The entitlement factor is a deterministic contractual scenario applied only "
    "to the floor, before the max. It is never applied to merchant cash or to the "
    "top-up, and it must not repeat the reserve-capacity availability already "
    "embedded in producer cash."
)
# Honest handling statement for the source-document uploader. Streamlit's
# file_uploader necessarily sends the selected file to the server running the
# app, so this must never promise local-only hashing; what the panel can truly
# promise is that only the digest flows onward into the case.
_CONTRACT_DOCUMENT_HANDLING_DISCLOSURE: Final = (
    "Uploading a source document sends it to the server running this app, which "
    "reads it only to compute the SHA-256. Project Case then records the digest "
    "alone — the document itself never enters the case, the result cache, the "
    "fingerprint, or the export. On a deployment you do not control, hash the "
    "file yourself (for example `shasum -a 256 <file>`) and type the digest "
    "instead of uploading a confidential contract."
)
_CONTRACT_CURRENCY_DISCLOSURE: Final = (
    "Quoted rates are a user assertion that the EUR values are already real in the "
    "valuation base year. No inflation, deflation, FX, or indexation conversion is "
    "applied here; convert a nominal or indexed quote outside the calculator and "
    "record how in the source field."
)

_QUOTE_STATUS_CHOICES: Final = {
    "User scenario — no source document": ContractQuoteStatus.USER_SCENARIO,
    "User-asserted indicative quote": (
        ContractQuoteStatus.USER_ASSERTED_INDICATIVE_QUOTE
    ),
    "User-asserted executed source document": (
        ContractQuoteStatus.USER_ASSERTED_EXECUTED_SOURCE_DOCUMENT
    ),
}


@dataclass(frozen=True)
class _StrategySelection:
    """UI-owned inputs for exactly one public producer adapter."""

    label: str
    adapter_id: ProducerAdapterId
    intraday_df: pd.DataFrame | None = None
    reserve_series: pd.Series | None = None
    reserve_product: str | None = None
    reserve_direction: str | None = None
    reserve_source: str | None = None
    bucket: str | None = None
    min_rebid_uplift_eur: float | None = None
    availability: float | None = None

    def cache_payload(self) -> dict[str, object]:
        return {
            "label": self.label,
            "adapter_id": self.adapter_id.value,
            "reserve_product": self.reserve_product,
            "reserve_direction": self.reserve_direction,
            "reserve_source": self.reserve_source,
            "bucket": self.bucket,
            "min_rebid_uplift_eur": self.min_rebid_uplift_eur,
            "availability": self.availability,
        }


@dataclass(frozen=True)
class _ReserveStream:
    """One zone/product/direction capacity stream with explicit provenance."""

    product: str
    direction: str
    source: str
    series: pd.Series

    @property
    def label(self) -> str:
        return f"{self.product} — {self.direction}"

    @property
    def adapter_product(self) -> str:
        # ProducerAdapterId has one reserve_product field, so direction must be
        # carried in that fingerprinted string rather than silently discarded.
        return f"{self.product} [{self.direction}]"


@dataclass(frozen=True)
class ProjectCaseRunCache:
    """Session-state value tying a result to the canonical ProjectCase digest."""

    request_fingerprint: str
    fingerprint: str
    result: RunResult

    def __post_init__(self) -> None:
        if not isinstance(self.result, RunResult):
            raise TypeError("ProjectCaseRunCache.result must be a RunResult")
        if self.result.input_fingerprint != self.fingerprint:
            raise ValueError("cached result does not match its ProjectCase fingerprint")


def _cashflow_frame(result: RunResult, *, lifecycle: bool) -> pd.DataFrame | None:
    table = (
        result.lifecycle_cashflow_table
        if lifecycle
        else result.screening_cashflow_table
    )
    if table is None:
        return None
    return pd.DataFrame([row.to_payload() for row in table.rows])


def _render_outcome(label: str, outcome: NpvOutcome) -> None:
    st.markdown(f"**{label}**")
    if not outcome.available:
        st.warning(
            f"Unavailable ({outcome.status}): {outcome.message} "
            "This is not recorded or displayed as EUR 0."
        )
        return
    distribution = outcome.distribution
    if distribution is None:  # defensive; the typed schema forbids this state
        st.error("Unavailable: typed NPV distribution is absent; no EUR 0 fallback.")
        return
    columns = st.columns(4)
    columns[0].metric(f"{label} — {P10_LABEL}", f"€{distribution.p10:,.0f}")
    columns[1].metric(f"{label} — {P50_LABEL}", f"€{distribution.p50:,.0f}")
    columns[2].metric(f"{label} — {P90_LABEL}", f"€{distribution.p90:,.0f}")
    columns[3].metric(
        f"{label} — {PROBABILITY_LABEL}", f"{distribution.prob_positive:.0%}"
    )


def _contract_case_payload(result: RunResult) -> Mapping[str, object] | None:
    """Return the settled ContractCase payload, or ``None`` for merchant-only."""
    project_case = result.provenance.get("project_case")
    if not isinstance(project_case, Mapping):
        return None
    contract_case = project_case.get("contract_case")
    return contract_case if isinstance(contract_case, Mapping) else None


def _disclosure_text(value: object) -> str:
    """Render a provenance scalar without turning an absent value into a blank."""
    return "null" if value is None else str(value)


def _contract_floor_disclosure_frame(
    terms: Mapping[str, object],
    settlement: Mapping[str, object],
) -> pd.DataFrame:
    """Rebuild the disclosed floor table from calculator output, not from inputs."""
    rates = terms["floor_rate_real_eur_per_modeled_mw_year_by_contract_year"]
    factors = terms["floor_entitlement_factor_by_contract_year"]
    floor_by_year = {
        int(item["year"]): float(item["effective_floor_eur"])
        for item in settlement["resolved_floor_by_project_year"]
    }
    start = int(terms["contract_start_project_year"])
    return pd.DataFrame(
        [
            {
                "contract_year": offset + 1,
                "project_year": start + offset,
                "floor_rate_real_eur_per_modeled_mw_year": float(rate),
                "floor_entitlement_factor": float(factor),
                "effective_whole_project_floor_eur": floor_by_year[start + offset],
            }
            for offset, (rate, factor) in enumerate(zip(rates, factors, strict=True))
        ]
    )


def _render_contract_settlement_disclosure(
    result: RunResult,
    *,
    compact: bool,
) -> None:
    """Disclose the settled contract from the RunResult alone, never from widgets."""
    contract_case = _contract_case_payload(result)
    assertions = result.provenance.get("red_line_assertions")
    asserted = (
        assertions.get("contract_settlement_included")
        if isinstance(assertions, Mapping)
        else None
    )
    if asserted is not (contract_case is not None):
        st.error(
            "Project Case provenance disagrees about whether a contract was "
            "settled. The settlement disclosure is withheld rather than guessed."
        )
        return
    if contract_case is None:
        return
    if compact:
        st.caption(
            "Contract settlement applied: "
            f"{contract_case['settlement_basis']}. {_CONTRACT_PRODUCT_DISCLOSURE}"
        )
        return

    with st.expander("Contract settlement disclosure", expanded=False):
        st.caption(_CONTRACT_PRODUCT_DISCLOSURE)
        st.caption(_CONTRACT_SIBLING_DISCLOSURE)
        st.caption(_CONTRACT_TOP_UP_DISCLOSURE)
        try:
            terms = contract_case["settlement_terms"]
            settlement = result.provenance["contract_settlement"]
            reconciliation = result.provenance["cashflow_reconciliation"]
            asset_case = result.provenance["project_case"]["asset_case"]
            currency = terms["currency_basis"]
            interpolation = settlement["representative_interpolation"]
            summary = pd.DataFrame(
                [
                    (field, _disclosure_text(value))
                    for field, value in (
                        ("Settlement basis", contract_case["settlement_basis"]),
                        ("Settlement algorithm", settlement["algorithm_version"]),
                        ("Quote status (user assertion)", terms["quote_status"]),
                        ("Source", terms["source"]),
                        ("Source as-of date", terms["source_as_of_date"]),
                        ("Source document SHA-256", terms["source_document_sha256"]),
                        ("Modelled whole-project power (MW)", asset_case["power_mw"]),
                        ("Quote basis", terms["quote_basis"]),
                        ("Asset scope", terms["asset_scope"]),
                        ("Settlement frequency", terms["settlement_frequency"]),
                        (
                            "Contract start project year",
                            terms["contract_start_project_year"],
                        ),
                        (
                            "Contract tenor (years)",
                            len(
                                terms[
                                    "floor_rate_real_eur_per_modeled_mw_year"
                                    "_by_contract_year"
                                ]
                            ),
                        ),
                        ("Currency basis", currency["mode"]),
                        ("Real-EUR base year", currency["target_base_year"]),
                        ("Reconciliation version", reconciliation["version"]),
                    )
                ],
                columns=["field", "value"],
            )
            floors = _contract_floor_disclosure_frame(terms, settlement)
            interpolation_frame = pd.DataFrame(
                [
                    (field, _disclosure_text(value))
                    for field, value in (
                        (
                            "Lower sorted rank (zero-based)",
                            interpolation["lower_sorted_rank"],
                        ),
                        (
                            "Upper sorted rank (zero-based)",
                            interpolation["upper_sorted_rank"],
                        ),
                        (
                            "Lower original draw index (zero-based)",
                            interpolation["lower_original_draw_index"],
                        ),
                        (
                            "Upper original draw index (zero-based)",
                            interpolation["upper_original_draw_index"],
                        ),
                        (
                            "Interpolation weight",
                            interpolation["interpolation_weight"],
                        ),
                    )
                ],
                columns=["field", "value"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            st.error(
                "Project Case contract provenance is incomplete "
                f"({exc}); the settlement disclosure is withheld."
            )
            return
        st.dataframe(summary, width="stretch", hide_index=True)
        st.caption(_CONTRACT_CURRENCY_DISCLOSURE)
        st.caption(
            "Effective whole-project floor per covered project year. Project "
            "years outside the term carry no floor at all, which is not a zero "
            "floor."
        )
        st.dataframe(floors, width="stretch", hide_index=True)
        st.caption(
            "Representative cash-flow basis: the P50 path is rank-interpolated "
            "between two settled draws. It is neither an actual scenario nor a "
            "per-year median, so its settled year cash need not equal "
            "max(median merchant, floor)."
        )
        st.dataframe(interpolation_frame, width="stretch", hide_index=True)


def render_project_case_result(
    result: RunResult,
    *,
    compact: bool = False,
) -> None:
    """Render ProjectCase NPV outputs exclusively from an immutable RunResult."""
    if not isinstance(result, RunResult):
        st.error("Project Case result is invalid; no result is rendered.")
        return
    _render_outcome(SCREENING_NPV_LABEL, result.no_lifecycle_cost_screening_npv)
    _render_outcome(LIFECYCLE_NPV_LABEL, result.lifecycle_cash_npv)
    if _contract_case_payload(result) is None:
        st.caption(
            "Lifecycle output is pre-tax unlevered, not bankable: tax, debt, DSCR, "
            "financing fees, shadow wear, and contracted-floor settlement are excluded."
        )
    else:
        st.caption(
            "Lifecycle output is pre-tax unlevered, not bankable: tax, debt, DSCR, "
            "financing fees, and shadow wear are excluded. Contract settlement IS "
            "included: cash is max(merchant, effective floor) per draw and project "
            "year, applied before lifecycle costs."
        )
    _render_contract_settlement_disclosure(result, compact=compact)
    strategy = result.provenance.get("strategy_run_result")
    if isinstance(strategy, Mapping):
        audit = strategy.get("coverage_audit")
        adapter = strategy.get("adapter_provenance")
        if isinstance(audit, Mapping) and isinstance(adapter, Mapping):
            observed = audit.get("observed_dates", ())
            valid = audit.get("valid_dates", ())
            missing = audit.get("missing_dates", ())
            failed = audit.get("solver_failed_dates", ())
            if all(isinstance(value, (tuple, list)) for value in (
                observed, valid, missing, failed
            )):
                if len(valid) < len(observed):
                    st.warning(
                        "Project Case annualisation uses only the audited valid "
                        f"days ({len(valid)}/{len(observed)}). Missing "
                        f"{len(missing)}; solver-failed {len(failed)}. Review "
                        "sample coverage before relying on the NPV distribution."
                    )
                st.caption(
                    f"Strategy input: {strategy.get('strategy_kind')} via "
                    f"{adapter.get('producer_adapter_id')}; valid "
                    f"{len(valid)}/{len(observed)} days, missing {len(missing)}, "
                    f"solver-failed {len(failed)}."
                )
    if compact:
        st.caption(f"ProjectCase fingerprint: {result.input_fingerprint[:16]}…")
        return

    st.caption(_P50_TABLE_CAPTION)
    screening = _cashflow_frame(result, lifecycle=False)
    if screening is not None:
        with st.expander("Screening representative cash flow", expanded=False):
            st.dataframe(screening, width="stretch", hide_index=True)
    lifecycle = _cashflow_frame(result, lifecycle=True)
    if lifecycle is not None:
        with st.expander("Lifecycle representative cash flow", expanded=False):
            st.dataframe(lifecycle, width="stretch", hide_index=True)


def _select_maintenance_basis() -> CapacityMaintenanceBasis:
    label = st.selectbox(
        "Capacity-maintenance basis",
        options=[
            "Unknown — screening NPV only",
            "No augmentation required — engineering assertion",
            "Scheduled nameplate maintenance — CSV",
        ],
        key="pc_capacity_maintenance_basis",
        help=(
            "Unknown keeps lifecycle NPV unavailable. An active basis must be "
            "supported by an engineering source and as-of date."
        ),
    )
    return {
        "Unknown — screening NPV only": CapacityMaintenanceBasis.UNKNOWN,
        "No augmentation required — engineering assertion": (
            CapacityMaintenanceBasis.NO_AUGMENTATION_REQUIRED_ASSERTED
        ),
        "Scheduled nameplate maintenance — CSV": (
            CapacityMaintenanceBasis.SCHEDULED_NAMEPLATE_MAINTENANCE
        ),
    }[label]


def _projection_inputs(
    project_life_years: int,
    *,
    allow_non_flat: bool,
) -> Projection:
    if not allow_non_flat:
        st.caption(
            "Composite DA/IDA/reserve strategies use FlatRealProjection in v1. "
            "A single decay multiplier cannot be applied across competing streams."
        )
        return Projection(ProjectionKind.FlatRealProjection)
    label = st.selectbox(
        "Annual revenue projection",
        ["Flat real", "DA-only spread decay", "Explicit multiplier CSV"],
        key="pc_projection_kind",
    )
    if label == "Flat real":
        return Projection(ProjectionKind.FlatRealProjection)
    if label == "DA-only spread decay":
        c1, c2 = st.columns(2)
        decay = c1.number_input(
            "Project Case decay (%/year)", 0.0, 99.0, 0.0, 1.0,
            key="pc_decay_pct",
        )
        floor = c2.number_input(
            "Project Case decay floor (% of year 1)", 0.0, 100.0, 0.0, 1.0,
            key="pc_decay_floor_pct",
        )
        return Projection(
            ProjectionKind.DAOnlySpreadDecay,
            annual_decay_rate=float(decay) / 100.0,
            decay_floor_share=float(floor) / 100.0,
        )

    st.download_button(
        "Download multiplier CSV template",
        data=explicit_multiplier_template_csv(project_life_years),
        file_name="project_case_annual_multipliers.csv",
        mime="text/csv",
        key="pc_multiplier_template_download",
    )
    upload = st.file_uploader(
        "Upload annual multiplier CSV",
        type=["csv"],
        key="pc_multiplier_upload",
    )
    source = st.text_input("Projection source", key="pc_projection_source")
    as_of = st.date_input(
        "Projection as-of",
        value=dt.date.today(),
        key="pc_projection_as_of",
    )
    if upload is None:
        raise ProjectCaseValidationError("Upload an explicit annual multiplier CSV")
    multipliers, preview = parse_explicit_multiplier_csv(
        upload.getvalue(), project_life_years
    )
    st.dataframe(preview, width="stretch", hide_index=True)
    return Projection(
        ProjectionKind.ExplicitAnnualMultiplierCurve,
        multipliers=multipliers,
        source=source,
        as_of=as_of.isoformat(),
    )


def _lifecycle_inputs(project_life_years: int) -> LifecycleCase:
    basis = _select_maintenance_basis()
    events = ()
    source = None
    as_of = None
    if basis is not CapacityMaintenanceBasis.UNKNOWN:
        c1, c2 = st.columns(2)
        source = c1.text_input(
            "Capacity-maintenance engineering source",
            key="pc_maintenance_source",
        )
        as_of_date = c2.date_input(
            "Capacity-maintenance as-of",
            value=dt.date.today(),
            key="pc_maintenance_as_of",
        )
        as_of = as_of_date.isoformat()
    if basis is CapacityMaintenanceBasis.SCHEDULED_NAMEPLATE_MAINTENANCE:
        st.download_button(
            "Download augmentation CSV template",
            data=AUGMENTATION_TEMPLATE_CSV,
            file_name="project_case_augmentation_schedule.csv",
            mime="text/csv",
            key="pc_augmentation_template_download",
        )
        upload = st.file_uploader(
            "Upload augmentation/replacement schedule",
            type=["csv"],
            key="pc_augmentation_upload",
        )
        if upload is None:
            raise ProjectCaseValidationError("Upload an augmentation schedule CSV")
        events, preview = parse_augmentation_csv(upload.getvalue())
        if any(event.year > project_life_years for event in events):
            raise ProjectCaseValidationError(
                "augmentation schedule contains a year beyond project life"
            )
        st.caption("Validated augmentation schedule preview")
        st.dataframe(preview, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    eol_residual = c1.number_input(
        "End-of-life residual value (real EUR)",
        min_value=0.0,
        value=0.0,
        step=10000.0,
        key="pc_eol_residual",
    )
    decommissioning = c2.number_input(
        "Decommissioning cost (real EUR)",
        min_value=0.0,
        value=0.0,
        step=10000.0,
        key="pc_decommissioning",
    )
    return LifecycleCase(
        project_life_years=project_life_years,
        capacity_maintenance_basis=basis,
        capacity_maintenance_source=source,
        capacity_maintenance_as_of=as_of,
        augmentation_events=events,
        eol_residual_value_eur=float(eol_residual),
        decommissioning_cost_eur=float(decommissioning),
    )


def _parse_contract_curve(text: str, *, expected: int, label: str) -> tuple[float, ...]:
    """Parse a user-typed per-contract-year curve; fail closed on any surprise."""
    tokens = [token for token in re.split(r"[,;\s]+", str(text).strip()) if token]
    values: list[float] = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ProjectCaseValidationError(
                f"{label} contains a non-numeric entry {token!r}"
            ) from exc
    if len(values) != expected:
        raise ProjectCaseValidationError(
            f"{label} must supply exactly {expected} value(s) for the selected "
            f"tenor; got {len(values)}"
        )
    return tuple(values)


def _contract_rate_curve(tenor: int) -> tuple[float, ...]:
    """Resolve the explicit real floor-rate curve that is fingerprinted."""
    mode = st.selectbox(
        "Floor quote entry",
        [_CONTRACT_CURVE_FLAT, _CONTRACT_CURVE_ESCALATING, _CONTRACT_CURVE_EXPLICIT],
        key="pc_contract_rate_mode",
        help=(
            "Flat and escalating entry are conveniences. The resolved explicit "
            "curve below is what is validated, fingerprinted and settled."
        ),
    )
    if mode == _CONTRACT_CURVE_FLAT:
        rate = st.number_input(
            "Real floor rate (EUR/modelled MW-year)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            key="pc_contract_flat_rate",
        )
        return (float(rate),) * tenor
    if mode == _CONTRACT_CURVE_ESCALATING:
        c1, c2 = st.columns(2)
        base = c1.number_input(
            "Contract-year 1 real floor rate (EUR/modelled MW-year)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            key="pc_contract_escalating_base",
        )
        escalation_pct = c2.number_input(
            "Real escalation (%/contract year)",
            min_value=-99.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            key="pc_contract_escalation_pct",
        )
        growth = 1.0 + float(escalation_pct) / 100.0
        return tuple(float(base) * growth**offset for offset in range(tenor))
    text = st.text_area(
        "Real floor rates by contract year (EUR/modelled MW-year)",
        key="pc_contract_rate_curve",
        help="Comma-, space- or newline-separated, one value per contract year.",
    )
    return _parse_contract_curve(text, expected=tenor, label="floor-rate curve")


def _contract_entitlement_factors(tenor: int) -> tuple[float, ...]:
    """Resolve the explicit entitlement-factor curve that is fingerprinted."""
    mode = st.selectbox(
        "Floor entitlement factors",
        [_CONTRACT_FACTOR_FULL, _CONTRACT_FACTOR_FLAT, _CONTRACT_FACTOR_EXPLICIT],
        key="pc_contract_factor_mode",
    )
    st.caption(_CONTRACT_ENTITLEMENT_DISCLOSURE)
    if mode == _CONTRACT_FACTOR_FULL:
        return (1.0,) * tenor
    if mode == _CONTRACT_FACTOR_FLAT:
        factor_pct = st.number_input(
            "Entitlement factor (% of quoted floor)",
            min_value=0.0,
            max_value=100.0,
            value=100.0,
            step=1.0,
            key="pc_contract_flat_factor_pct",
        )
        return (float(factor_pct) / 100.0,) * tenor
    text = st.text_area(
        "Entitlement factors by contract year (0-1)",
        key="pc_contract_factor_curve",
        help="Comma-, space- or newline-separated, one factor per contract year.",
    )
    return _parse_contract_curve(text, expected=tenor, label="entitlement-factor curve")


def _resolve_source_document_sha256(
    uploaded: bytes | None,
    typed: str,
) -> str | None:
    """Digest an uploaded source document, else use the typed digest verbatim.

    An uploaded document wins so a stale typed digest cannot outrank the document
    actually in hand.  Only surrounding whitespace is stripped from typed input:
    case is left alone so a non-lowercase digest fails the schema check loudly
    instead of being silently rewritten.
    """
    if uploaded is not None:
        return hashlib.sha256(uploaded).hexdigest()
    return str(typed).strip() or None


def _contract_source_document_sha256(status: ContractQuoteStatus) -> str | None:
    """Resolve the source-document digest under the locked null matrix."""
    if status is ContractQuoteStatus.USER_SCENARIO:
        st.caption(
            "A user scenario carries no source-document digest; the locked schema "
            "requires it to be absent."
        )
        return None
    st.caption(_CONTRACT_DOCUMENT_HANDLING_DISCLOSURE)
    upload = st.file_uploader(
        "Source document — sent to this deployment to compute its SHA-256",
        key="pc_contract_source_document",
    )
    typed = st.text_input(
        "Source document SHA-256 (lowercase 64-hex)",
        key="pc_contract_source_sha256",
    )
    digest = _resolve_source_document_sha256(
        None if upload is None else upload.getvalue(),
        typed,
    )
    if upload is not None:
        st.caption(
            f"Digest computed from the uploaded document: {digest}. An uploaded "
            "document takes precedence over the typed digest."
        )
    return digest


def _contract_floor_preview(
    terms: AnnualPreLifecycleStrategyCashFloor,
    *,
    power_mw: float,
    project_life_years: int,
) -> tuple[pd.DataFrame, int]:
    """Preview the resolved floor using the calculator's own resolution."""
    covered, floors = resolve_effective_contract_floor(
        terms,
        power_mw=power_mw,
        project_life_years=project_life_years,
    )
    start = int(terms.contract_start_project_year)
    rates = terms.floor_rate_real_eur_per_modeled_mw_year_by_contract_year
    factors = terms.floor_entitlement_factor_by_contract_year
    rows = [
        {
            "contract_year": offset + 1,
            "project_year": start + offset,
            "floor_rate_real_eur_per_modeled_mw_year": float(rate),
            "floor_entitlement_factor": float(factor),
            "effective_whole_project_floor_eur": float(floors[start - 1 + offset]),
        }
        for offset, (rate, factor) in enumerate(zip(rates, factors, strict=True))
    ]
    return pd.DataFrame(rows), int(covered.sum())


def _contract_inputs(
    *,
    project_life_years: int,
    base_year: int,
    power_mw: float,
) -> ContractCase | None:
    """Build the optional v1.1 ContractCase, or ``None`` for merchant-only cash."""
    with st.expander("Contracted floor settlement (optional)", expanded=False):
        st.caption(_CONTRACT_PRODUCT_DISCLOSURE)
        st.caption(_CONTRACT_SIBLING_DISCLOSURE)
        mode = st.selectbox(
            "Settlement basis",
            [_CONTRACT_NONE_LABEL, _CONTRACT_FLOOR_LABEL],
            key="pc_contract_mode",
        )
        if mode == _CONTRACT_NONE_LABEL:
            st.caption(
                "Merchant-only settlement: the NPVs above are unprotected "
                "strategy cash and no floor is applied."
            )
            return None

        st.caption(_CONTRACT_TOP_UP_DISCLOSURE)
        c1, c2 = st.columns(2)
        # Both bounds are the schema's own domain, never a derived bound: a
        # dynamic max would silently re-clamp a stored widget value when project
        # life changes. An out-of-life term fails closed below instead.
        start_year = int(c1.number_input(
            "Contract start project year",
            min_value=1,
            max_value=MAX_PROJECT_LIFE_YEARS,
            value=1,
            step=1,
            key="pc_contract_start_year",
        ))
        tenor = int(c2.number_input(
            "Contract tenor (years)",
            min_value=1,
            max_value=MAX_PROJECT_LIFE_YEARS,
            value=1,
            step=1,
            key="pc_contract_tenor",
        ))
        final_year = start_year + tenor - 1
        if final_year > project_life_years:
            raise ProjectCaseValidationError(
                f"contract term ends in project year {final_year}, beyond the "
                f"{project_life_years}-year project life"
            )
        rates = _contract_rate_curve(tenor)
        factors = _contract_entitlement_factors(tenor)

        status_label = st.selectbox(
            "Quote status (user assertion)",
            list(_QUOTE_STATUS_CHOICES),
            key="pc_contract_quote_status",
            help=(
                "Records only the asserted maturity of the source. Even an "
                "executed source document does not claim that the platform has "
                "reproduced the complete legal contract."
            ),
        )
        status = _QUOTE_STATUS_CHOICES[status_label]
        c3, c4 = st.columns(2)
        source = c3.text_input("Contract source", key="pc_contract_source")
        as_of = c4.date_input(
            "Source as-of",
            value=dt.date.today(),
            key="pc_contract_as_of",
        )
        source_sha256 = _contract_source_document_sha256(status)
        st.caption(_CONTRACT_CURRENCY_DISCLOSURE)
        st.caption(
            f"Fixed v1.1 basis: quote {CONTRACT_QUOTE_BASIS_V1}; scope "
            f"{CONTRACT_ASSET_SCOPE_V1}; settlement "
            f"{CONTRACT_SETTLEMENT_FREQUENCY_V1}; real-EUR base year {base_year} "
            f"(bound to the valuation base year); modelled whole-project power "
            f"{power_mw:g} MW (inherited from the asset case, never re-entered here)."
        )

        terms = AnnualPreLifecycleStrategyCashFloor(
            contract_start_project_year=start_year,
            floor_rate_real_eur_per_modeled_mw_year_by_contract_year=rates,
            floor_entitlement_factor_by_contract_year=factors,
            quote_basis=CONTRACT_QUOTE_BASIS_V1,
            settlement_frequency=CONTRACT_SETTLEMENT_FREQUENCY_V1,
            asset_scope=CONTRACT_ASSET_SCOPE_V1,
            currency_basis=ContractCurrencyBasis(
                ContractCurrencyBasisMode.USER_ASSERTED_REAL_BASE_YEAR_EUR_CURVE,
                base_year,
            ),
            quote_status=status,
            source=source,
            source_as_of_date=as_of.isoformat(),
            source_document_sha256=source_sha256,
        )
        preview, covered_years = _contract_floor_preview(
            terms,
            power_mw=power_mw,
            project_life_years=project_life_years,
        )
        st.caption(
            f"Validated floor preview — {covered_years} covered project year(s), "
            f"{start_year} to {final_year}. Years outside the term have no floor "
            "at all; that is not a zero floor."
        )
        st.dataframe(preview, width="stretch", hide_index=True)
        return ContractCase(
            settlement_basis=(
                ContractSettlementBasis.ANNUAL_PRE_LIFECYCLE_STRATEGY_CASH_FLOOR_V1
            ),
            settlement_terms=terms,
        )


def _request_fingerprint(
    *,
    primary_df: pd.DataFrame,
    primary_zone: str,
    start_date: dt.date,
    end_date: dt.date,
    asset: AssetCase,
    lifecycle: LifecycleCase,
    projection: Projection,
    valuation: ValuationCase,
    bootstrap: BootstrapCase,
    capture_rate: float,
    strategy: _StrategySelection,
    contract: ContractCase | None,
) -> str:
    """Hash every pre-adapter input so ordinary reruns never re-run the MILP.

    This digest is a PC-C cache key, not the public ProjectCase fingerprint.  The
    public digest can only be produced after the producer adapter issues its
    typed result.  We still bind the cache to that canonical digest below.
    """
    def content_hash(value: pd.DataFrame | pd.Series | None) -> str | None:
        if value is None:
            return None
        hashed = pd.util.hash_pandas_object(
            value, index=True, categorize=True
        ).to_numpy(dtype="uint64", copy=False).tobytes()
        digest = hashlib.sha256()
        digest.update(hashed)
        if isinstance(value, pd.DataFrame):
            digest.update(encode_value({
                "columns": [str(column) for column in value.columns],
                "dtypes": [str(dtype) for dtype in value.dtypes],
            }))
        else:
            digest.update(encode_value({
                "name": None if value.name is None else str(value.name),
                "dtype": str(value.dtype),
            }))
        return digest.hexdigest()

    payload = {
        "implementation_versions": {
            "schema_version": PROJECT_CASE_SCHEMA_VERSION,
            "pc_a_calculator_version": PC_A_CALCULATOR_VERSION,
            "pc_d2_calculator_version": PC_D2_CALCULATOR_VERSION,
            "expected_grid_registry_version": EXPECTED_GRID_REGISTRY_VERSION,
        },
        "zone": primary_zone,
        "first_delivery_date": start_date.isoformat(),
        "last_delivery_date": end_date.isoformat(),
        "da_content_hash": content_hash(primary_df),
        "ida_content_hash": content_hash(strategy.intraday_df),
        "reserve_content_hash": content_hash(strategy.reserve_series),
        "strategy": strategy.cache_payload(),
        "asset_case": asset.to_payload(),
        "lifecycle_case": lifecycle.to_payload(),
        "projection": projection.to_payload(),
        "valuation_case": valuation.to_payload(),
        "bootstrap_case": bootstrap.to_payload(),
        # A contract edit must invalidate the cache before anything renders or
        # downloads, so no floor-protected NPV can survive its own terms.
        "contract_case": None if contract is None else contract.to_payload(),
        "capture_rate": float(capture_rate),
    }
    return hashlib.sha256(encode_value(payload)).hexdigest()


def _result_uses_current_versions(result: RunResult) -> bool:
    if result.schema_version != PROJECT_CASE_SCHEMA_VERSION:
        return False
    provenance = result.provenance
    if provenance.get("calculator_version") != PC_D2_CALCULATOR_VERSION:
        return False
    strategy = provenance.get("strategy_run_result")
    if not isinstance(strategy, Mapping):
        return False
    if strategy.get("calculator_version") != PC_A_CALCULATOR_VERSION:
        return False
    adapter = strategy.get("adapter_provenance")
    return bool(
        isinstance(adapter, Mapping)
        and adapter.get("expected_grid_registry_version")
        == EXPECTED_GRID_REGISTRY_VERSION
    )


def _cached_result(request_fingerprint: str) -> RunResult | None:
    cached = st.session_state.get(_CACHE_KEY)
    if not isinstance(cached, ProjectCaseRunCache):
        return None
    if cached.request_fingerprint != request_fingerprint:
        return None
    if cached.result.input_fingerprint != cached.fingerprint:
        return None
    if not _result_uses_current_versions(cached.result):
        return None
    return cached.result


def current_project_case_result() -> RunResult | None:
    """Return the last typed result for app-level export; never an untyped dict."""
    cached = st.session_state.get(_CACHE_KEY)
    if not isinstance(cached, ProjectCaseRunCache):
        return None
    if cached.result.input_fingerprint != cached.fingerprint:
        return None
    if not _result_uses_current_versions(cached.result):
        return None
    return cached.result


def render_project_case_panel(
    *,
    primary_zone: str,
    primary_df: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    capture_rate: float,
    capex_eur_kwh: float,
    intraday_df: pd.DataFrame | None = None,
    ancillary_df: pd.DataFrame | None = None,
    capacity_df: pd.DataFrame | None = None,
    capacity_sources: Mapping[
        tuple[str, str, str], Mapping[str, object]
    ] | None = None,
) -> RunResult | None:
    """Render the one canonical PC-C input/run surface in the Revenue tab."""
    st.divider()
    st.subheader("Project Case — lifecycle valuation")
    st.caption(
        "This path uses one selected producer-issued realised strategy, never a "
        "comparison-table row. It reports screening and pre-tax unlevered lifecycle NPV."
    )
    try:
        strategy_selection = _strategy_selection(
            primary_zone=primary_zone,
            intraday_df=intraday_df,
            ancillary_df=ancillary_df,
            capacity_df=capacity_df,
            capacity_sources=capacity_sources,
        )
        if strategy_selection.adapter_id is ProducerAdapterId.PC_ADP_DA_ONLY:
            st.caption(
                f"DA-only producer cash applies the sidebar capture rate "
                f"({capture_rate:.0%})."
            )
        else:
            st.caption(
                "This producer emits its internally realised/co-optimised total at "
                "capture rate 100%; the sidebar DA haircut is not applied again."
            )
        c1, c2, c3 = st.columns(3)
        project_life = int(c1.number_input(
            "Project life (years)", 1, 100, 20, 1, key="pc_project_life"
        ))
        discount_pct = c2.number_input(
            "Real discount rate (%)", -99.0, 100.0, 8.0, 0.5,
            key="pc_discount_rate_pct",
        )
        base_year = int(c3.number_input(
            "Base year (real EUR)", 1900, 9999, dt.date.today().year, 1,
            key="pc_base_year",
        ))
        st.caption(
            "Currency basis assertion: loaded settlement EUR are treated as "
            "base-year real EUR; no deflator is applied in this UI path."
        )
        fixed_om = st.number_input(
            "Fixed O&M (real EUR/MW-year)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            key="pc_fixed_om",
            help="Fixed O&M only. Dispatch VOM is already embedded and is not re-deducted.",
        )
        lifecycle = _lifecycle_inputs(project_life)
        projection = _projection_inputs(
            project_life,
            allow_non_flat=(
                strategy_selection.adapter_id is ProducerAdapterId.PC_ADP_DA_ONLY
            ),
        )

        contract = _contract_inputs(
            project_life_years=project_life,
            base_year=base_year,
            power_mw=power_mw,
        )

        c4, c5 = st.columns(2)
        seed = int(c4.text_input("Bootstrap seed", "0", key="pc_bootstrap_seed"))
        simulations = int(c5.number_input(
            "Bootstrap simulations",
            MIN_SIMULATIONS,
            MAX_SIMULATIONS,
            DEFAULT_SIMULATIONS,
            1000,
            key="pc_bootstrap_simulations",
        ))
        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()
        asset = AssetCase.from_capex_per_kwh(
            power_mw=power_mw,
            duration_hours=duration_hours,
            round_trip_efficiency=efficiency,
            capex_eur_per_kwh=capex_eur_kwh,
            fixed_om_eur_per_mw_yr=float(fixed_om),
        )
        valuation = ValuationCase(float(discount_pct) / 100.0, base_year)
        bootstrap = BootstrapCase(seed, simulations, BOOTSTRAP_ALGORITHM_V1)
        request_fingerprint = _request_fingerprint(
            primary_df=primary_df,
            primary_zone=primary_zone,
            start_date=start,
            end_date=end,
            asset=asset,
            lifecycle=lifecycle,
            projection=projection,
            valuation=valuation,
            bootstrap=bootstrap,
            capture_rate=capture_rate,
            strategy=strategy_selection,
            contract=contract,
        )
        currency_basis = CurrencyBasis(
            CurrencyBasisMode.SOURCE_EUR_TREATED_AS_BASE_YEAR_REAL,
            base_year,
        )
    except (
        AdapterUnavailableError,
        ProjectCaseValidationError,
        TypeError,
        ValueError,
    ) as exc:
        st.session_state.pop(_CACHE_KEY, None)
        st.error(f"Project Case input unavailable: {exc}")
        return None

    cached = _cached_result(request_fingerprint)
    stale = st.session_state.get(_CACHE_KEY) is not None and cached is None
    if stale:
        st.session_state.pop(_CACHE_KEY, None)
    run = st.button("Run Project Case", key="pc_run", type="primary")
    if run:
        try:
            with st.spinner("Running audited DA replay and lifecycle valuation…"):
                strategy = _emit_selected_strategy(
                    strategy_selection,
                    primary_df=primary_df,
                    primary_zone=primary_zone,
                    start=start,
                    end=end,
                    power_mw=power_mw,
                    duration_hours=duration_hours,
                    efficiency=efficiency,
                    currency_basis=currency_basis,
                    capture_rate=capture_rate,
                )
                case = ProjectCase(
                    asset_case=asset,
                    lifecycle_case=lifecycle,
                    market_case=MarketCase(strategy, projection),
                    valuation_case=valuation,
                    bootstrap_case=bootstrap,
                    contract_case=contract,
                )
                result = compute_project_case(case)
            st.session_state[_CACHE_KEY] = ProjectCaseRunCache(
                request_fingerprint, case.input_fingerprint(), result
            )
            cached = result
        except (
            AdapterUnavailableError,
            ProjectCaseValidationError,
            TypeError,
            ValueError,
        ) as exc:
            st.session_state.pop(_CACHE_KEY, None)
            st.error(f"Project Case unavailable: {exc}. No EUR 0 fallback was created.")
            return None

    if cached is None:
        if stale:
            st.warning("Project Case inputs changed. The stale result is hidden; run again.")
        else:
            st.info("Run Project Case to calculate the typed NPV outcomes.")
        return None
    render_project_case_result(cached)
    return cached


def render_project_case_cockpit_mirror() -> None:
    """Read-only Cockpit mirror; never rebuilds or recalculates the case."""
    with st.expander("Project Case NPV — read-only Revenue-tab result", expanded=False):
        result = current_project_case_result()
        if result is None:
            st.info("Run Project Case in Revenue Estimation to populate this view.")
            return
        render_project_case_result(result, compact=True)


_CAPACITY_CACHE_SOURCE = "Unified capacity cache"
_CAPACITY_SESSION_SOURCE = (
    "Session ancillary fallback (manual/auto provenance not stream-resolved)"
)
_LEGACY_ZONE_ALIASES: Final = {"DE": "DE_LU"}


def _canonical_reserve_direction(product: str, value: object) -> str | None:
    """Return a closed reserve direction, or ``None`` when it is ambiguous.

    Direction is part of the selected stream identity.  We accept an explicit
    up/down/symmetric value, or a direction embedded unambiguously in a legacy
    product label.  A bare FCR/FCR-N is intrinsically symmetric; a blank aFRR,
    mFRR or FCR-D direction is not guessed.
    """
    raw = "" if pd.isna(value) else str(value).strip().lower()
    aliases = {
        "up": "up",
        "positive": "up",
        "pos": "up",
        "down": "down",
        "negative": "down",
        "neg": "down",
        "symmetric": "symmetric",
        "symmetrical": "symmetric",
        "both": "symmetric",
    }
    explicit = aliases.get(raw)
    normal_product = " ".join(
        str(product).lower().replace("_", " ").replace("/", " ").split()
    )
    tokens = set(normal_product.replace("-", " ").split())
    embedded = None
    if "up" in tokens and "down" not in tokens:
        embedded = "up"
    elif "down" in tokens and "up" not in tokens:
        embedded = "down"
    if explicit is not None and embedded is not None and explicit != embedded:
        return None
    if explicit is not None:
        return explicit
    if embedded is not None:
        return embedded
    if normal_product in {"fcr", "fcr-n", "fcr n"}:
        return "symmetric"
    return None


def _stream_source(
    sources: Mapping[tuple[str, str, str], Mapping[str, object]],
    *,
    zone: str,
    product: str,
    direction: str,
) -> str:
    """Resolve the persisted per-stream source without case-sensitive drift."""
    target = (zone.casefold(), product.casefold(), direction.casefold())
    for key, metadata in sources.items():
        if len(key) != 3:
            continue
        candidate = tuple(str(part).strip().casefold() for part in key)
        if candidate == target:
            label = str(metadata.get("source", "")).strip()
            if label:
                return f"{_CAPACITY_CACHE_SOURCE} / {label}"
    return f"{_CAPACITY_CACHE_SOURCE} / Unknown (pre-provenance cache)"


def _reserve_streams_from_frame(
    frame: pd.DataFrame | None,
    *,
    primary_zone: str,
    cache_scoped: bool,
    capacity_sources: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> tuple[_ReserveStream, ...]:
    """Build non-mixing streams from one explicitly scoped capacity frame."""
    if frame is None or frame.empty:
        return ()
    required = {"product_type", "direction", "capacity_price_eur_mw"}
    if not required.issubset(frame.columns):
        return ()
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        return ()

    work = frame.copy()
    if "zone" not in work.columns:
        # The app stamps the validated cache lookup zone explicitly.  Keeping
        # this requirement here prevents another caller from laundering an
        # arbitrary unzoned frame by passing it through the cache parameter.
        return ()
    zones = work["zone"].fillna("").astype(str).str.strip().map(
        lambda value: _LEGACY_ZONE_ALIASES.get(value, value)
    )
    work = work[zones == primary_zone]
    if work.empty:
        return ()

    # Preserve blank/non-numeric capacity values.  PC-A owns the per-delivery-
    # date data gate and must see every raw row to distinguish a malformed day
    # from a truly absent one without laundering duplicates.
    work = work.copy()
    work["_pc_product"] = (
        work["product_type"].fillna("").astype(str).str.strip()
    )
    work["_pc_direction"] = [
        _canonical_reserve_direction(product, direction)
        for product, direction in zip(
            work["_pc_product"], work["direction"], strict=True
        )
    ]
    work = work[
        work["_pc_product"].ne("") & work["_pc_direction"].notna()
    ]
    if work.empty:
        return ()

    streams: list[_ReserveStream] = []
    for (product, direction), group in work.groupby(
        ["_pc_product", "_pc_direction"], sort=True
    ):
        # Preserve malformed values and duplicate timestamps for the typed
        # PC-A coverage audit.  That audit classifies only the affected local
        # delivery date as missing; rejecting the whole stream here would turn
        # one bad block into an all-window unavailable result and would change
        # the canonical eligible-date universe.
        prices = pd.to_numeric(
            group["capacity_price_eur_mw"], errors="coerce"
        ).sort_index()
        if prices.empty:
            continue
        source = (
            _stream_source(
                capacity_sources,
                zone=primary_zone,
                product=str(product),
                direction=str(direction),
            )
            if cache_scoped
            else _CAPACITY_SESSION_SOURCE
        )
        streams.append(_ReserveStream(
            product=str(product),
            direction=str(direction),
            source=source,
            series=prices.rename("capacity_price_eur_mw"),
        ))
    return tuple(streams)


def _resolve_reserve_streams(
    *,
    primary_zone: str,
    ancillary_df: pd.DataFrame | None,
    capacity_df: pd.DataFrame | None,
    capacity_sources: Mapping[tuple[str, str, str], Mapping[str, object]] | None,
) -> tuple[_ReserveStream, ...]:
    """Resolve cache-first, zone/product/direction-bound reserve streams."""
    sources = capacity_sources or {}
    cached = _reserve_streams_from_frame(
        capacity_df,
        primary_zone=primary_zone,
        cache_scoped=True,
        capacity_sources=sources,
    )
    if cached:
        return cached
    return _reserve_streams_from_frame(
        ancillary_df,
        primary_zone=primary_zone,
        cache_scoped=False,
        capacity_sources={},
    )


def _strategy_selection(
    *,
    primary_zone: str,
    intraday_df: pd.DataFrame | None,
    ancillary_df: pd.DataFrame | None,
    capacity_df: pd.DataFrame | None = None,
    capacity_sources: Mapping[
        tuple[str, str, str], Mapping[str, object]
    ] | None = None,
) -> _StrategySelection:
    """Render a typed producer selector, exposing only data-supported lanes."""
    ida_available = (
        intraday_df is not None
        and not intraday_df.empty
        and grid.ida_profile_id(primary_zone) is not None
    )
    reserve_streams = (
        _resolve_reserve_streams(
            primary_zone=primary_zone,
            ancillary_df=ancillary_df,
            capacity_df=capacity_df,
            capacity_sources=capacity_sources,
        )
        if grid.reserve_profile_id(primary_zone) is not None
        else ()
    )
    reserve_available = bool(reserve_streams)
    options = [_DA_ONLY_LABEL]
    if ida_available:
        options.append(_DA_ID_LABEL)
    if reserve_available:
        options.append(_DA_RESERVE_LABEL)
    if ida_available and reserve_available:
        options.append(_DA_ID_RESERVE_LABEL)
    label = st.selectbox(
        "Cash-NPV dispatch strategy",
        options,
        key="pc_strategy",
        help=(
            "Only producer-issued realised totals are eligible. Ceilings, value "
            "deltas, overlays, benchmarks and gross-additive rows cannot be selected."
        ),
    )
    unavailable: list[str] = []
    if not ida_available:
        unavailable.append("IDA1 (no complete supported IDA dataset loaded)")
    if not reserve_available:
        unavailable.append(
            f"reserve (no zone/direction-qualified capacity stream for {primary_zone})"
        )
    if unavailable:
        st.caption("Unavailable strategy inputs: " + "; ".join(unavailable) + ".")

    if label == _DA_ONLY_LABEL:
        return _StrategySelection(label, ProducerAdapterId.PC_ADP_DA_ONLY)

    bucket = None
    if label in {_DA_ID_LABEL, _DA_ID_RESERVE_LABEL}:
        bucket_label = st.selectbox(
            "Walk-forward forecast bucket",
            ["Hour of week", "Hour of day"],
            key="pc_forecast_bucket",
        )
        bucket = (
            BUCKET_HOUR_OF_WEEK
            if bucket_label == "Hour of week"
            else BUCKET_HOUR_OF_DAY
        )
    if label == _DA_ID_LABEL:
        deadband = st.number_input(
            "Minimum IDA rebid uplift (EUR)",
            min_value=0.0,
            value=0.0,
            step=10.0,
            key="pc_min_rebid_uplift",
        )
        return _StrategySelection(
            label,
            ProducerAdapterId.PC_ADP_DA_ID,
            intraday_df=intraday_df,
            bucket=bucket,
            min_rebid_uplift_eur=float(deadband),
        )

    reserve_by_label = {stream.label: stream for stream in reserve_streams}
    reserve_label = st.selectbox(
        "Reserve capacity product and direction",
        list(reserve_by_label),
        key="pc_reserve_product",
    )
    reserve_stream = reserve_by_label[reserve_label]
    reserve_series = reserve_stream.series
    availability = st.slider(
        "Reserve availability",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.01,
        key="pc_reserve_availability",
    )
    reserve_product = reserve_stream.adapter_product
    reserve_source = reserve_stream.source
    st.caption(f"Reserve capacity source: {reserve_source}.")
    if label == _DA_RESERVE_LABEL:
        return _StrategySelection(
            label,
            ProducerAdapterId.PC_ADP_RESERVE_COOPT,
            reserve_series=reserve_series,
            reserve_product=reserve_product,
            reserve_direction=reserve_stream.direction,
            reserve_source=reserve_source,
            availability=float(availability),
        )
    return _StrategySelection(
        label,
        ProducerAdapterId.PC_ADP_DA_ID_RESERVE,
        intraday_df=intraday_df,
        reserve_series=reserve_series,
        reserve_product=reserve_product,
        reserve_direction=reserve_stream.direction,
        reserve_source=reserve_source,
        bucket=bucket,
        availability=float(availability),
    )


def _emit_selected_strategy(
    selection: _StrategySelection,
    *,
    primary_df: pd.DataFrame,
    primary_zone: str,
    start: dt.date,
    end: dt.date,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    currency_basis: CurrencyBasis,
    capture_rate: float,
) -> StrategyRunResult:
    """Call exactly the public producer named by the typed UI selection."""
    common = {
        "zone": primary_zone,
        "first_delivery_date": start,
        "last_delivery_date": end,
        "power_mw": power_mw,
        "duration_hours": duration_hours,
        "efficiency": efficiency,
        "currency_basis": currency_basis,
    }
    if selection.adapter_id is ProducerAdapterId.PC_ADP_DA_ONLY:
        return emit_da_only(
            primary_df,
            **common,
            capture_rate=capture_rate,
            capture_source=(
                "sidebar_capture_haircut" if capture_rate != 1.0 else "not_applied"
            ),
        )
    if selection.adapter_id is ProducerAdapterId.PC_ADP_DA_ID:
        if selection.intraday_df is None:
            raise AdapterUnavailableError("selected strategy requires IDA1 data")
        return emit_da_id(
            primary_df,
            selection.intraday_df,
            **common,
            bucket=selection.bucket,
            min_rebid_uplift_eur=selection.min_rebid_uplift_eur,
        )
    if selection.adapter_id is ProducerAdapterId.PC_ADP_RESERVE_COOPT:
        if selection.reserve_series is None:
            raise AdapterUnavailableError("selected strategy requires reserve data")
        return emit_reserve_coopt(
            primary_df,
            selection.reserve_series,
            **common,
            reserve_product=selection.reserve_product,
            reserve_source=selection.reserve_source,
            availability=selection.availability,
        )
    if selection.intraday_df is None or selection.reserve_series is None:
        raise AdapterUnavailableError(
            "selected strategy requires both IDA1 and reserve data"
        )
    return emit_da_id_reserve(
        primary_df,
        selection.intraday_df,
        selection.reserve_series,
        **common,
        reserve_product=selection.reserve_product,
        reserve_source=selection.reserve_source,
        bucket=selection.bucket,
        availability=selection.availability,
    )
