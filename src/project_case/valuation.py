"""Project Case v1 PC-B lifecycle cash-flow and NPV calculation.

This module is the pure-calculation increment defined by
``docs/design/project-case-v1.md``.  It consumes one validated
``ProjectCase`` and returns the immutable ``RunResult`` scaffold shipped by
PC-A.  It deliberately does not import UI, export, contracted-floor, wear, or
dispatch code: the producer-issued daily series is already the gross,
post-VOM cash input for the whole modelled asset.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from src.project_case.bootstrap import bootstrap_annual_sums
from src.project_case.enums import (
    LIFECYCLE_UNKNOWN_MESSAGE,
    LIFECYCLE_UNKNOWN_STATUS,
    CapacityMaintenanceBasis,
    ProjectionKind,
)
from src.project_case.schema import (
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
_CASHFLOW_TABLE_STATISTIC: Final = "p50_annual_bootstrap_draw_linear"

__all__ = ["PC_B_CALCULATOR_VERSION", "compute_project_case"]


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


def _provenance(
    case: ProjectCase,
    *,
    multipliers: np.ndarray,
) -> dict[str, object]:
    strategy = case.market_case.strategy_run_result
    bootstrap = case.bootstrap_case
    return {
        "calculator_version": PC_B_CALCULATOR_VERSION,
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
        "cashflow_table_statistic": _CASHFLOW_TABLE_STATISTIC,
        "red_line_assertions": {
            "cash_npv_includes_shadow_wear": False,
            "vom_rededucted": False,
            "mw_rescaled": False,
            "floor_included": False,
            "pre_tax_unlevered": True,
            "tax_included": False,
            "debt_included": False,
            "financing_fees_included": False,
        },
    }


def compute_project_case(case: ProjectCase) -> RunResult:
    """Compute the two Project Case v1 NPV outcomes and cash-flow tables.

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
    screening_draws = _screening_npv_draws(
        annual_draws,
        multipliers,
        discount_factors,
        case.asset_case.installed_capex_eur,
    )
    screening_distribution = _summarise_npv_draws(screening_draws)

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
        lifecycle_table = _cashflow_table(
            basis="lifecycle",
            annual_revenue_p50=annual_revenue_p50,
            multipliers=multipliers,
            discount_factors=discount_factors,
            fixed_om_eur=fixed_om,
            augmentation_eur=augmentation,
            terminal_eur=terminal,
        )

    return RunResult(
        input_fingerprint=case.input_fingerprint(),
        no_lifecycle_cost_screening_npv=NpvOutcome.ok(screening_distribution),
        lifecycle_cash_npv=lifecycle_outcome,
        provenance=_provenance(case, multipliers=multipliers),
        screening_cashflow_table=screening_table,
        lifecycle_cashflow_table=lifecycle_table,
    )
