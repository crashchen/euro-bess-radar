"""Bootstrap Monte Carlo and sensitivity analysis for BESS revenue screening."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def annuity_pv_factor(life_years: float, discount_rate: float) -> float:
    """Present-value factor for an annuity of 1 per year over a fractional life.

    The integer years use the closed-form annuity formula; any fractional
    residual is treated as a pro-rata cashflow at the end of the next year.
    Avoids the bug where ``int(life)`` floored 14.9 to 14, dropping ~one full
    year of discounted revenue from NPV.
    """
    if life_years <= 0:
        return 0.0
    if discount_rate == 0:
        return float(life_years)
    n_full = math.floor(life_years)
    frac = life_years - n_full
    pv = 0.0
    if n_full > 0:
        pv += (1 - (1 + discount_rate) ** (-n_full)) / discount_rate
    if frac > 0:
        pv += frac / (1 + discount_rate) ** (n_full + 1)
    return pv


def _validate_decay_inputs(
    annual_decay_rate: float,
    decay_floor_share: float,
) -> tuple[float, float, bool]:
    """Validate decay assumptions and return their canonical activity state."""
    try:
        decay = float(annual_decay_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("annual_decay_rate must be finite") from exc
    try:
        floor = float(decay_floor_share)
    except (TypeError, ValueError) as exc:
        raise ValueError("decay_floor_share must be finite") from exc

    if not math.isfinite(decay):
        raise ValueError("annual_decay_rate must be finite")
    if not 0.0 <= decay < 1.0:
        raise ValueError("annual_decay_rate must be in [0, 1)")
    if not math.isfinite(floor):
        raise ValueError("decay_floor_share must be finite")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("decay_floor_share must be in [0, 1]")
    return decay, floor, decay > 0.0 and floor < 1.0


def decaying_annuity_pv_factor(
    life_years: float,
    discount_rate: float,
    annual_decay_rate: float = 0.0,
    decay_floor_share: float = 0.0,
) -> float:
    """PV factor for a merchant-revenue trajectory with annual decay.

    Year one remains at the full year-one revenue level. Later years receive
    ``max((1 - decay) ** (year - 1), floor_share)``. Fractional lives follow
    :func:`annuity_pv_factor`: the residual is a pro-rata cash flow at the end
    of the next year and therefore uses that next year's decay weight.
    """
    decay, floor, active = _validate_decay_inputs(
        annual_decay_rate,
        decay_floor_share,
    )
    return _decaying_annuity_pv_factor_validated(
        life_years,
        discount_rate,
        decay,
        floor,
        active,
    )


def _decaying_annuity_pv_factor_validated(
    life_years: float,
    discount_rate: float,
    decay: float,
    floor: float,
    active: bool,
) -> float:
    """Compute the factor from canonical values returned by the validator."""
    if not active:
        return annuity_pv_factor(life_years, discount_rate)
    if life_years <= 0:
        return 0.0

    n_full = math.floor(life_years)
    frac = life_years - n_full
    pv = 0.0
    for year in range(1, n_full + 1):
        weight = max((1.0 - decay) ** (year - 1), floor)
        pv += weight / (1.0 + discount_rate) ** year
    if frac > 0:
        residual_year = n_full + 1
        weight = max((1.0 - decay) ** (residual_year - 1), floor)
        pv += frac * weight / (1.0 + discount_rate) ** residual_year
    return pv


def bootstrap_annual_revenue(
    daily_revenues: pd.Series | np.ndarray,
    n_simulations: int = 5000,
    seed: int = 42,
) -> dict:
    """Bootstrap annual revenue distribution from daily observations.

    Resamples 365 daily revenues with replacement to build a distribution
    of possible annual outcomes.

    Args:
        daily_revenues: Daily revenue values (EUR).
        n_simulations: Number of bootstrap samples.
        seed: Random seed for reproducibility.

    Returns:
        Dict with p10, p25, p50, p75, p90, mean, std, and raw simulations.
    """
    arr = np.asarray(daily_revenues, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        zeros = np.zeros(n_simulations)
        return {
            "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0,
            "mean": 0.0, "std": 0.0, "simulations": zeros,
        }

    rng = np.random.default_rng(seed)
    # Each simulation: draw 365 days with replacement, sum to annual
    samples = rng.choice(arr, size=(n_simulations, 365), replace=True)
    annual_sums = samples.sum(axis=1)

    return {
        "p10": float(np.percentile(annual_sums, 10)),
        "p25": float(np.percentile(annual_sums, 25)),
        "p50": float(np.percentile(annual_sums, 50)),
        "p75": float(np.percentile(annual_sums, 75)),
        "p90": float(np.percentile(annual_sums, 90)),
        "mean": float(annual_sums.mean()),
        "std": float(annual_sums.std()),
        "simulations": annual_sums,
    }


def calculate_npv_distribution(
    annual_revenue_dist: np.ndarray,
    total_capex: float,
    annual_degradation_cost: float = 0.0,
    effective_life_years: float = 20.0,
    discount_rate: float = 0.08,
    annual_decay_rate: float = 0.0,
    decay_floor_share: float = 0.0,
) -> dict:
    """Calculate NPV distribution from annual revenue simulations.

    By default, each simulation uses the same annual revenue across all years.
    When revenue decay is active, merchant revenue follows the decayed PV
    factor while annual degradation cost remains flat.

    Args:
        annual_revenue_dist: Array of simulated annual revenues (n_simulations,).
        total_capex: Total upfront CapEx in EUR.
        annual_degradation_cost: Annual degradation cost in EUR.
        effective_life_years: Asset effective life in years.
        discount_rate: Discount rate for NPV (e.g. 0.08 = 8%).
        annual_decay_rate: Annual merchant-revenue decay as a decimal fraction.
        decay_floor_share: Minimum annual revenue weight versus year one.

    Returns:
        Dict with npv_p10, npv_p50, npv_p90, prob_positive_npv, npv_array.
    """
    decay, floor, active = _validate_decay_inputs(
        annual_decay_rate,
        decay_floor_share,
    )
    decay_factor = _decaying_annuity_pv_factor_validated(
        float(effective_life_years),
        discount_rate,
        decay,
        floor,
        active,
    )
    if active:
        flat_factor = annuity_pv_factor(
            float(effective_life_years),
            discount_rate,
        )
        npv_array = (
            annual_revenue_dist * decay_factor
            - annual_degradation_cost * flat_factor
            - total_capex
        )
    else:
        pv_factor = decay_factor
        net_annual = annual_revenue_dist - annual_degradation_cost
        npv_array = net_annual * pv_factor - total_capex

    return {
        "npv_p10": float(np.percentile(npv_array, 10)),
        "npv_p50": float(np.percentile(npv_array, 50)),
        "npv_p90": float(np.percentile(npv_array, 90)),
        "prob_positive_npv": float((npv_array > 0).mean()),
        "npv_array": npv_array,
    }


def sensitivity_table(
    base_revenue: float,
    total_capex: float,
    effective_life_years: float = 20.0,
    annual_degradation_cost: float = 0.0,
    discount_rate: float = 0.08,
    vary: dict[str, list[float]] | None = None,
    annual_decay_rate: float = 0.0,
    decay_floor_share: float = 0.0,
) -> pd.DataFrame:
    """One-at-a-time sensitivity analysis on key parameters.

    Args:
        base_revenue: Base-case annual revenue (EUR).
        total_capex: Total upfront CapEx (EUR).
        effective_life_years: Asset life (years).
        annual_degradation_cost: Annual degradation cost (EUR).
        discount_rate: Discount rate for NPV.
        vary: Dict mapping parameter name to sensitivity values. Revenue and
            CapEx values are multipliers; discount rate, lifetime, and decay
            values are absolute. Defaults to standard ranges.
        annual_decay_rate: Base annual merchant-revenue decay as a fraction.
        decay_floor_share: Minimum annual revenue weight versus year one.

    Returns:
        DataFrame with columns: param, label, multiplier, value, npv.
    """
    decay, floor, active = _validate_decay_inputs(
        annual_decay_rate,
        decay_floor_share,
    )
    if vary is None:
        vary = {
            "revenue": [0.7, 1.0, 1.3],
            "capex": [0.8, 1.0, 1.2],
            "discount_rate": [0.06, 0.08, 0.10],
            "lifetime": [15, 20, 25],
        }
        if active:
            vary["decay"] = [
                0.0,
                decay,
                min(2.0 * decay, (1.0 + decay) / 2.0),
            ]

    rows = []

    for param, values in vary.items():
        for val in values:
            rev = base_revenue * val if param == "revenue" else base_revenue
            capex = total_capex * val if param == "capex" else total_capex
            dr = val if param == "discount_rate" else discount_rate
            life = val if param == "lifetime" else effective_life_years
            deg = annual_degradation_cost

            if param == "decay":
                row_decay, row_floor, row_active = _validate_decay_inputs(
                    val,
                    floor,
                )
            else:
                row_decay, row_floor, row_active = decay, floor, active
            row_decay_factor = _decaying_annuity_pv_factor_validated(
                float(life),
                dr,
                row_decay,
                row_floor,
                row_active,
            )
            if row_active:
                flat_factor = annuity_pv_factor(float(life), dr)
                npv = rev * row_decay_factor - deg * flat_factor - capex
            else:
                pv_factor = row_decay_factor
                npv = (rev - deg) * pv_factor - capex

            rows.append({
                "param": param,
                "value": val,
                "npv": round(npv, 2),
            })

    return pd.DataFrame(rows)
