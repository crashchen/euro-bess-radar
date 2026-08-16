"""Project Case v1.1 lifecycle cash-flow and NPV calculation.

This module is the pure-calculation increment defined by
``docs/design/project-case-v1.md`` and the locked PC-D settlement extension in
``docs/design/project-case-contract-settlement-v1.1.md``.  It consumes one validated
``ProjectCase`` and returns the immutable ``RunResult`` scaffold shipped by
PC-A.  It deliberately does not import UI, export, contracted-floor, wear, or
dispatch code: the producer-issued daily series is already the post-VOM,
pre-lifecycle strategy-cash input for the whole modelled asset.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from src.project_case.bootstrap import bootstrap_annual_sums
from src.project_case.enums import (
    CASHFLOW_RECONCILIATION_ABS_TOL_EUR_V1,
    CASHFLOW_RECONCILIATION_REL_TOL_V1,
    CASHFLOW_RECONCILIATION_VERSION_V1,
    CONTRACT_CASHFLOW_TABLE_STATISTIC_V1,
    CONTRACT_SETTLEMENT_ALGORITHM_V1,
    LIFECYCLE_UNKNOWN_MESSAGE,
    LIFECYCLE_UNKNOWN_STATUS,
    NULL_CASHFLOW_TABLE_STATISTIC_V1,
    PC_D2_CALCULATOR_VERSION,
    CapacityMaintenanceBasis,
    ProjectionKind,
)
from src.project_case.schema import (
    AnnualPreLifecycleStrategyCashFloor,
    CashflowRow,
    CashflowTable,
    NpvDistribution,
    NpvOutcome,
    ProjectCase,
    ProjectCaseValidationError,
    RunResult,
)

PC_B_CALCULATOR_VERSION: Final = "pc-b-v1"
_PERCENTILE_METHOD: Final = "linear"

__all__ = [
    "PC_B_CALCULATOR_VERSION",
    "PC_D2_CALCULATOR_VERSION",
    "compute_project_case",
]


def _require_finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ProjectCaseValidationError(f"{name} must be finite")
    return result


def _require_finite_array(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ProjectCaseValidationError(f"{name} contains a non-finite value")
    return result


def _projection_multipliers(case: ProjectCase) -> np.ndarray:
    """Resolve the locked projection tagged union to one multiplier per year."""
    life = int(case.lifecycle_case.project_life_years)
    projection = case.market_case.projection
    kind = projection.projection_kind
    if kind is ProjectionKind.FlatRealProjection:
        values = np.ones(life, dtype=np.float64)
    elif kind is ProjectionKind.DAOnlySpreadDecay:
        decay = float(projection.annual_decay_rate)
        floor = float(projection.decay_floor_share)
        values = np.asarray(
            [max((1.0 - decay) ** (year - 1), floor) for year in range(1, life + 1)],
            dtype=np.float64,
        )
    elif kind is ProjectionKind.ExplicitAnnualMultiplierCurve:
        values = np.asarray(projection.multipliers, dtype=np.float64)
    else:  # pragma: no cover - the schema's closed enum rejects this first
        raise ProjectCaseValidationError(f"unsupported projection kind: {kind!r}")

    if values.shape != (life,):
        raise ProjectCaseValidationError(
            "projection multipliers must cover exactly project_life_years"
        )
    _require_finite_array(values, "projection multipliers")
    if np.any(values < 0.0) or values[0] != 1.0:
        raise ProjectCaseValidationError(
            "projection multipliers must be non-negative with year 1 equal to 1.0"
        )
    return values


def _discount_factors(case: ProjectCase) -> np.ndarray:
    """Return explicit end-of-year discount factors for years 1..L."""
    life = int(case.lifecycle_case.project_life_years)
    rate = float(case.valuation_case.discount_rate)
    years = np.arange(1, life + 1, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        factors = np.power(1.0 + rate, -years, dtype=np.float64)
    _require_finite_array(factors, "discount factors")
    if np.any(factors <= 0.0):
        raise ProjectCaseValidationError("discount factors must be strictly positive")
    return factors


def _lifecycle_vectors(case: ProjectCase) -> tuple[float, np.ndarray, np.ndarray]:
    """Return fixed O&M, net event capital, and terminal cash by project year."""
    life = int(case.lifecycle_case.project_life_years)
    asset = case.asset_case
    lifecycle = case.lifecycle_case
    fixed_om = _require_finite_scalar(
        float(asset.fixed_om_eur_per_mw_yr) * float(asset.power_mw),
        "annual fixed O&M",
    )
    augmentation = np.zeros(life, dtype=np.float64)
    try:
        for year in range(1, life + 1):
            augmentation[year - 1] = math.fsum(
                float(event.cost_eur) - float(event.residual_value_eur)
                for event in lifecycle.augmentation_events
                if event.year == year
            )
    except OverflowError as exc:
        raise ProjectCaseValidationError("augmentation cash flow overflowed") from exc
    _require_finite_array(augmentation, "augmentation cash flows")

    terminal = np.zeros(life, dtype=np.float64)
    terminal[-1] = _require_finite_scalar(
        float(lifecycle.eol_residual_value_eur)
        - float(lifecycle.decommissioning_cost_eur),
        "terminal cash flow",
    )
    return fixed_om, augmentation, terminal


def _screening_npv_draws(
    annual_draws: np.ndarray,
    multipliers: np.ndarray,
    discount_factors: np.ndarray,
    installed_capex_eur: float,
) -> np.ndarray:
    """Discount revenue year by year; CapEx is deducted once at year zero."""
    draws = np.full(
        annual_draws.shape,
        -float(installed_capex_eur),
        dtype=np.float64,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        for multiplier, discount_factor in zip(
            multipliers, discount_factors, strict=True
        ):
            draws += annual_draws * multiplier * discount_factor
            _require_finite_array(draws, "screening NPV draws")
    return draws


def _resolved_contract_floor(
    case: ProjectCase,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(covered_mask, effective_floor)`` for the typed annual floor.

    Coverage is carried independently from the numeric vector: a zero inside the
    contract term is an active floor, whereas zero in the unused array slots is
    ignored.  This prevents an absent post-term floor from protecting a negative
    merchant year to zero (PC-D red-line #9).
    """
    contract = case.contract_case
    if contract is None:
        return None
    terms = contract.settlement_terms
    if not isinstance(terms, AnnualPreLifecycleStrategyCashFloor):
        # The schema owns the closed tagged union.  Keep the calculator defensive
        # so a future union member cannot silently inherit these cash semantics.
        raise ProjectCaseValidationError(
            "unsupported ContractCase settlement terms for PC-D2"
        )

    life = int(case.lifecycle_case.project_life_years)
    start = int(terms.contract_start_project_year)
    rates = terms.floor_rate_real_eur_per_modeled_mw_year_by_contract_year
    factors = terms.floor_entitlement_factor_by_contract_year
    covered = np.zeros(life, dtype=np.bool_)
    floors = np.zeros(life, dtype=np.float64)
    power_mw = float(case.asset_case.power_mw)
    with np.errstate(over="ignore", invalid="ignore"):
        for offset, (rate, factor) in enumerate(zip(rates, factors, strict=True)):
            project_index = start - 1 + offset
            # Operation order follows the locked formula q * MW * a.  The
            # entitlement factor is part of F before max, never a top-up haircut.
            value = float(rate) * power_mw
            value *= float(factor)
            floors[project_index] = value
            covered[project_index] = True
    _require_finite_array(floors[covered], "effective contract floors")
    return covered, floors


def _settled_revenue_matrix(
    annual_draws: np.ndarray,
    multipliers: np.ndarray,
    covered: np.ndarray,
    floors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return merchant, settled and top-up matrices for a non-null contract."""
    with np.errstate(over="ignore", invalid="ignore"):
        merchant = annual_draws[:, np.newaxis] * multipliers[np.newaxis, :]
    merchant = _require_finite_array(merchant, "merchant cash matrix")
    settled = merchant.copy()
    if np.any(covered):
        with np.errstate(over="ignore", invalid="ignore"):
            settled[:, covered] = np.maximum(
                merchant[:, covered], floors[np.newaxis, covered]
            )
    settled = _require_finite_array(settled, "settled cash matrix")
    with np.errstate(over="ignore", invalid="ignore"):
        top_up = settled - merchant
    top_up = _require_finite_array(top_up, "contract top-up matrix")
    return merchant, settled, top_up


def _screening_npv_draws_from_settled(
    settled: np.ndarray,
    discount_factors: np.ndarray,
    installed_capex_eur: float,
) -> np.ndarray:
    """Discount an already-settled draw/year matrix one year at a time."""
    if settled.ndim != 2 or settled.shape[1] != discount_factors.size:
        raise ProjectCaseValidationError(
            "settled cash matrix must have one column per project year"
        )
    draws = np.full(
        (settled.shape[0],),
        -float(installed_capex_eur),
        dtype=np.float64,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        for project_index, discount_factor in enumerate(discount_factors):
            draws += settled[:, project_index] * float(discount_factor)
            _require_finite_array(draws, "screening NPV draws")
    return draws


def _rank_interpolated_representative(
    annual_draws: np.ndarray,
    merchant: np.ndarray,
    settled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    """Build the exact nonlinear-safe P50 reconciliation path.

    Screening NPV is non-decreasing in the single annual draw under the locked
    non-negative projection and positive-discount domains, so annual-draw order
    is also NPV order.  Original draw index is the deterministic tie-break.
    """
    draw_count = int(annual_draws.size)
    if draw_count == 0:
        raise ProjectCaseValidationError("annual bootstrap draws must not be empty")
    original_indices = np.arange(draw_count, dtype=np.int64)
    order = np.lexsort((original_indices, annual_draws))
    h = 0.5 * (draw_count - 1)
    lower_rank = math.floor(h)
    upper_rank = math.ceil(h)
    weight = _require_finite_scalar(h - lower_rank, "P50 interpolation weight")
    lower_original = int(order[lower_rank])
    upper_original = int(order[upper_rank])

    with np.errstate(over="ignore", invalid="ignore"):
        merchant_star = (
            (1.0 - weight) * merchant[lower_original]
            + weight * merchant[upper_original]
        )
        settled_star = (
            (1.0 - weight) * settled[lower_original]
            + weight * settled[upper_original]
        )
        top_up_star = settled_star - merchant_star
    merchant_star = _require_finite_array(
        merchant_star, "representative merchant cash"
    )
    settled_star = _require_finite_array(
        settled_star, "representative settled cash"
    )
    top_up_star = _require_finite_array(
        top_up_star, "representative contract top-up"
    )
    return (
        merchant_star,
        settled_star,
        top_up_star,
        {
            "lower_sorted_rank": lower_rank,
            "upper_sorted_rank": upper_rank,
            "lower_original_draw_index": lower_original,
            "upper_original_draw_index": upper_original,
            "interpolation_weight": weight,
        },
    )


def _lifecycle_adjustment(
    *,
    fixed_om_eur: float,
    augmentation_eur: np.ndarray,
    terminal_eur: np.ndarray,
    discount_factors: np.ndarray,
) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        annual_adjustment = -fixed_om_eur - augmentation_eur + terminal_eur
        discounted = annual_adjustment * discount_factors
    _require_finite_array(annual_adjustment, "annual lifecycle adjustment")
    _require_finite_array(discounted, "discounted lifecycle adjustment")
    try:
        return _require_finite_scalar(
            math.fsum(float(value) for value in discounted),
            "lifecycle adjustment present value",
        )
    except OverflowError as exc:
        raise ProjectCaseValidationError(
            "lifecycle adjustment present value overflowed"
        ) from exc


def _summarise_npv_draws(draws: np.ndarray) -> NpvDistribution:
    values = _require_finite_array(draws, "NPV draws")
    if values.ndim != 1 or values.size == 0:
        raise ProjectCaseValidationError("NPV draws must be a non-empty 1-D array")
    percentiles = np.percentile(
        values,
        [10.0, 50.0, 90.0],
        method=_PERCENTILE_METHOD,
    )
    _require_finite_array(percentiles, "NPV percentiles")
    probability = _require_finite_scalar(
        float(np.mean(values > 0.0)),
        "NPV positive probability",
    )
    return NpvDistribution(
        p10=float(percentiles[0]),
        p50=float(percentiles[1]),
        p90=float(percentiles[2]),
        prob_positive=probability,
    )


def _cashflow_table(
    *,
    basis: str,
    annual_revenue_p50: float,
    multipliers: np.ndarray,
    discount_factors: np.ndarray,
    fixed_om_eur: float,
    augmentation_eur: np.ndarray,
    terminal_eur: np.ndarray,
) -> CashflowTable:
    rows: list[CashflowRow] = []
    for index, (multiplier, discount_factor) in enumerate(
        zip(multipliers, discount_factors, strict=True),
        start=1,
    ):
        revenue = _require_finite_scalar(
            annual_revenue_p50 * float(multiplier),
            f"year {index} representative revenue",
        )
        if basis == "screening":
            opex = 0.0
            augmentation = 0.0
            terminal = 0.0
        else:
            opex = fixed_om_eur
            augmentation = float(augmentation_eur[index - 1])
            terminal = float(terminal_eur[index - 1])
        net = _require_finite_scalar(
            revenue - opex - augmentation + terminal,
            f"year {index} representative net cash flow",
        )
        discounted = _require_finite_scalar(
            net * float(discount_factor),
            f"year {index} representative discounted cash flow",
        )
        rows.append(
            CashflowRow(
                year=index,
                merchant_revenue_eur=revenue,
                effective_contract_floor_eur=None,
                contract_top_up_eur=0.0,
                revenue_eur=revenue,
                opex_eur=opex,
                augmentation_eur=augmentation,
                terminal_eur=terminal,
                net_eur=net,
                discount_factor=float(discount_factor),
                discounted_net_eur=discounted,
            )
        )
    return CashflowTable(basis=basis, rows=tuple(rows))


def _contract_cashflow_table(
    *,
    basis: str,
    merchant_revenue: np.ndarray,
    settled_revenue: np.ndarray,
    contract_top_up: np.ndarray,
    covered: np.ndarray,
    floors: np.ndarray,
    discount_factors: np.ndarray,
    fixed_om_eur: float,
    augmentation_eur: np.ndarray,
    terminal_eur: np.ndarray,
) -> CashflowTable:
    """Materialise one rank-interpolated v1.1 cash-flow table."""
    life = discount_factors.size
    arrays = {
        "merchant representative cash": merchant_revenue,
        "settled representative cash": settled_revenue,
        "representative contract top-up": contract_top_up,
        "contract coverage": covered,
        "effective contract floors": floors,
        "augmentation cash flows": augmentation_eur,
        "terminal cash flows": terminal_eur,
    }
    for name, values in arrays.items():
        if np.asarray(values).shape != (life,):
            raise ProjectCaseValidationError(f"{name} must cover project life")

    rows: list[CashflowRow] = []
    for project_index in range(life):
        year = project_index + 1
        merchant = _require_finite_scalar(
            float(merchant_revenue[project_index]),
            f"year {year} representative merchant cash",
        )
        settled = _require_finite_scalar(
            float(settled_revenue[project_index]),
            f"year {year} representative settled cash",
        )
        top_up = _require_finite_scalar(
            float(contract_top_up[project_index]),
            f"year {year} representative contract top-up",
        )
        floor_value = (
            _require_finite_scalar(
                float(floors[project_index]),
                f"year {year} effective contract floor",
            )
            if bool(covered[project_index])
            else None
        )
        if basis == "screening":
            opex = 0.0
            augmentation = 0.0
            terminal = 0.0
        else:
            opex = fixed_om_eur
            augmentation = float(augmentation_eur[project_index])
            terminal = float(terminal_eur[project_index])
        net = _require_finite_scalar(
            settled - opex - augmentation + terminal,
            f"year {year} representative net cash flow",
        )
        discount_factor = float(discount_factors[project_index])
        discounted = _require_finite_scalar(
            net * discount_factor,
            f"year {year} representative discounted cash flow",
        )
        rows.append(
            CashflowRow(
                year=year,
                merchant_revenue_eur=merchant,
                effective_contract_floor_eur=floor_value,
                contract_top_up_eur=top_up,
                revenue_eur=settled,
                opex_eur=opex,
                augmentation_eur=augmentation,
                terminal_eur=terminal,
                net_eur=net,
                discount_factor=discount_factor,
                discounted_net_eur=discounted,
            )
        )
    return CashflowTable(basis=basis, rows=tuple(rows))


def _require_table_reconciliation(
    *,
    table: CashflowTable,
    installed_capex_eur: float,
    reported_p50_eur: float,
) -> None:
    try:
        discounted_cash = math.fsum(
            float(row.discounted_net_eur) for row in table.rows
        )
    except OverflowError as exc:
        raise ProjectCaseValidationError(
            f"{table.basis} cash-flow reconciliation overflowed"
        ) from exc
    reconciled = _require_finite_scalar(
        -float(installed_capex_eur) + discounted_cash,
        f"{table.basis} reconciled P50 NPV",
    )
    if not math.isclose(
        reconciled,
        float(reported_p50_eur),
        rel_tol=CASHFLOW_RECONCILIATION_REL_TOL_V1,
        abs_tol=CASHFLOW_RECONCILIATION_ABS_TOL_EUR_V1,
    ):
        raise ProjectCaseValidationError(
            f"{table.basis} cash-flow table does not reconcile to reported P50"
        )


def _provenance(
    case: ProjectCase,
    *,
    multipliers: np.ndarray,
    covered: np.ndarray | None,
    floors: np.ndarray | None,
    representative_interpolation: dict[str, int | float] | None,
) -> dict[str, object]:
    strategy = case.market_case.strategy_run_result
    bootstrap = case.bootstrap_case
    contract = case.contract_case
    contract_basis = (
        None if contract is None else contract.settlement_basis.value
    )
    if contract is None:
        contract_settlement = {
            "basis": None,
            "algorithm_version": None,
            "resolved_floor_by_project_year": [],
            "representative_interpolation": None,
        }
        statistic = NULL_CASHFLOW_TABLE_STATISTIC_V1
    else:
        if covered is None or floors is None or representative_interpolation is None:
            raise ProjectCaseValidationError(
                "non-null ContractCase requires resolved settlement provenance"
            )
        contract_settlement = {
            "basis": contract_basis,
            "algorithm_version": CONTRACT_SETTLEMENT_ALGORITHM_V1,
            "resolved_floor_by_project_year": [
                {"year": index + 1, "effective_floor_eur": float(floors[index])}
                for index in range(covered.size)
                if bool(covered[index])
            ],
            "representative_interpolation": representative_interpolation,
        }
        statistic = CONTRACT_CASHFLOW_TABLE_STATISTIC_V1
    return {
        "calculator_version": PC_D2_CALCULATOR_VERSION,
        "project_case_input_fingerprint": case.input_fingerprint(),
        "strategy_run_fingerprint": strategy.fingerprint(),
        "project_case": case.to_payload(),
        "strategy_run_result": strategy.to_payload(),
        "bootstrap": {
            "algorithm_version": bootstrap.bootstrap_algorithm_version,
            "seed": bootstrap.seed,
            "n_simulations": bootstrap.n_simulations,
            "days_per_annual_draw": 365,
            "percentile_method": _PERCENTILE_METHOD,
            "prob_positive_rule": "strict_gt_zero_all_draws",
        },
        "projection": {
            "projection_kind": case.market_case.projection.projection_kind.value,
            "resolved_annual_multipliers": [float(value) for value in multipliers],
        },
        "valuation": {
            "discount_rate": float(case.valuation_case.discount_rate),
            "base_year": int(case.valuation_case.base_year),
            "currency_convention": "real_base_year_eur",
            "cash_timing": "end_of_year",
        },
        "capacity_maintenance_basis": (
            case.lifecycle_case.capacity_maintenance_basis.value
        ),
        "contract_settlement": contract_settlement,
        "cashflow_table_statistic": statistic,
        "cashflow_reconciliation": {
            "version": CASHFLOW_RECONCILIATION_VERSION_V1,
            "relative_tolerance": CASHFLOW_RECONCILIATION_REL_TOL_V1,
            "absolute_tolerance_eur": CASHFLOW_RECONCILIATION_ABS_TOL_EUR_V1,
        },
        "red_line_assertions": {
            "cash_npv_includes_shadow_wear": False,
            "vom_rededucted": False,
            "mw_rescaled": False,
            "wear_net_floor_comparator_included": False,
            "contract_settlement_included": contract is not None,
            "contract_settlement_basis": contract_basis,
            "pre_tax_unlevered": True,
            "tax_included": False,
            "debt_included": False,
            "financing_fees_included": False,
        },
    }


def compute_project_case(case: ProjectCase) -> RunResult:
    """Compute the two Project Case v1.1 NPV outcomes and cash-flow tables.

    Invalid inputs or any non-finite derived value raise
    ``ProjectCaseValidationError``.  The only typed partial-availability state is
    an ``UNKNOWN`` capacity-maintenance basis: screening still computes, while
    lifecycle NPV and its table are unavailable by contract.
    """
    if not isinstance(case, ProjectCase):
        raise ProjectCaseValidationError("case must be a ProjectCase")
    case.validate()

    strategy = case.market_case.strategy_run_result
    daily_values = np.asarray(
        [value for _date, value in strategy.daily_realised_cash_series],
        dtype=np.float64,
    )
    bootstrap = case.bootstrap_case
    with np.errstate(over="ignore", invalid="ignore"):
        annual_draws = bootstrap_annual_sums(
            daily_values,
            seed=bootstrap.seed,
            n_simulations=bootstrap.n_simulations,
        )
    annual_draws = _require_finite_array(annual_draws, "annual bootstrap draws")
    if annual_draws.shape != (bootstrap.n_simulations,):
        raise ProjectCaseValidationError(
            "annual bootstrap draws must match BootstrapCase.n_simulations"
        )

    multipliers = _projection_multipliers(case)
    discount_factors = _discount_factors(case)
    lifecycle = case.lifecycle_case
    if lifecycle.capacity_maintenance_basis is CapacityMaintenanceBasis.UNKNOWN:
        # UNKNOWN is a typed screening-only state.  Do not even derive lifecycle
        # cash (which could overflow at extreme but individually finite inputs),
        # because the contract explicitly forbids assuming those costs are zero.
        fixed_om = 0.0
        augmentation = np.zeros(lifecycle.project_life_years, dtype=np.float64)
        terminal = np.zeros(lifecycle.project_life_years, dtype=np.float64)
    else:
        fixed_om, augmentation, terminal = _lifecycle_vectors(case)
    resolved_floor = _resolved_contract_floor(case)
    representative_interpolation: dict[str, int | float] | None = None
    if resolved_floor is None:
        # This is a semantic compatibility branch.  Keep the exact PC-B
        # arithmetic/order rather than algebraically rewriting it as a matrix
        # operation: null-contract economic numbers and legacy table columns are
        # required to remain bit-identical.
        covered = None
        floors = None
        screening_draws = _screening_npv_draws(
            annual_draws,
            multipliers,
            discount_factors,
            case.asset_case.installed_capex_eur,
        )
        annual_revenue_p50 = _require_finite_scalar(
            float(np.percentile(annual_draws, 50.0, method=_PERCENTILE_METHOD)),
            "annual bootstrap P50",
        )
        screening_table = _cashflow_table(
            basis="screening",
            annual_revenue_p50=annual_revenue_p50,
            multipliers=multipliers,
            discount_factors=discount_factors,
            fixed_om_eur=0.0,
            augmentation_eur=np.zeros_like(augmentation),
            terminal_eur=np.zeros_like(terminal),
        )
        representative_merchant = None
        representative_settled = None
        representative_top_up = None
    else:
        covered, floors = resolved_floor
        merchant_matrix, settled_matrix, _top_up_matrix = _settled_revenue_matrix(
            annual_draws,
            multipliers,
            covered,
            floors,
        )
        screening_draws = _screening_npv_draws_from_settled(
            settled_matrix,
            discount_factors,
            case.asset_case.installed_capex_eur,
        )
        (
            representative_merchant,
            representative_settled,
            representative_top_up,
            representative_interpolation,
        ) = _rank_interpolated_representative(
            annual_draws,
            merchant_matrix,
            settled_matrix,
        )
        screening_table = _contract_cashflow_table(
            basis="screening",
            merchant_revenue=representative_merchant,
            settled_revenue=representative_settled,
            contract_top_up=representative_top_up,
            covered=covered,
            floors=floors,
            discount_factors=discount_factors,
            fixed_om_eur=0.0,
            augmentation_eur=np.zeros_like(augmentation),
            terminal_eur=np.zeros_like(terminal),
        )
    screening_distribution = _summarise_npv_draws(screening_draws)
    _require_table_reconciliation(
        table=screening_table,
        installed_capex_eur=case.asset_case.installed_capex_eur,
        reported_p50_eur=screening_distribution.p50,
    )

    if lifecycle.capacity_maintenance_basis is CapacityMaintenanceBasis.UNKNOWN:
        lifecycle_outcome = NpvOutcome.unavailable(
            LIFECYCLE_UNKNOWN_STATUS,
            LIFECYCLE_UNKNOWN_MESSAGE,
        )
        lifecycle_table = None
    else:
        adjustment = _lifecycle_adjustment(
            fixed_om_eur=fixed_om,
            augmentation_eur=augmentation,
            terminal_eur=terminal,
            discount_factors=discount_factors,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            lifecycle_draws = screening_draws + adjustment
        _require_finite_array(lifecycle_draws, "lifecycle NPV draws")
        lifecycle_outcome = NpvOutcome.ok(_summarise_npv_draws(lifecycle_draws))
        if resolved_floor is None:
            lifecycle_table = _cashflow_table(
                basis="lifecycle",
                annual_revenue_p50=annual_revenue_p50,
                multipliers=multipliers,
                discount_factors=discount_factors,
                fixed_om_eur=fixed_om,
                augmentation_eur=augmentation,
                terminal_eur=terminal,
            )
        else:
            lifecycle_table = _contract_cashflow_table(
                basis="lifecycle",
                merchant_revenue=representative_merchant,
                settled_revenue=representative_settled,
                contract_top_up=representative_top_up,
                covered=covered,
                floors=floors,
                discount_factors=discount_factors,
                fixed_om_eur=fixed_om,
                augmentation_eur=augmentation,
                terminal_eur=terminal,
            )
        _require_table_reconciliation(
            table=lifecycle_table,
            installed_capex_eur=case.asset_case.installed_capex_eur,
            reported_p50_eur=lifecycle_outcome.distribution.p50,
        )

    return RunResult(
        input_fingerprint=case.input_fingerprint(),
        no_lifecycle_cost_screening_npv=NpvOutcome.ok(screening_distribution),
        lifecycle_cash_npv=lifecycle_outcome,
        provenance=_provenance(
            case,
            multipliers=multipliers,
            covered=covered,
            floors=floors,
            representative_interpolation=representative_interpolation,
        ),
        screening_cashflow_table=screening_table,
        lifecycle_cashflow_table=lifecycle_table,
    )
