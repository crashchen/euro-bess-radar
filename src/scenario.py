"""Bootstrap Monte Carlo and sensitivity analysis for BESS revenue screening."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
) -> dict:
    """Calculate NPV distribution from annual revenue simulations.

    Each simulation uses the same annual revenue across all years (static
    assumption), discounted at the given rate.

    Args:
        annual_revenue_dist: Array of simulated annual revenues (n_simulations,).
        total_capex: Total upfront CapEx in EUR.
        annual_degradation_cost: Annual degradation cost in EUR.
        effective_life_years: Asset effective life in years.
        discount_rate: Discount rate for NPV (e.g. 0.08 = 8%).

    Returns:
        Dict with npv_p10, npv_p50, npv_p90, prob_positive_npv, npv_array.
    """
    n_years = int(effective_life_years)
    discount_factors = np.array([1 / (1 + discount_rate) ** t for t in range(1, n_years + 1)])
    pv_factor = discount_factors.sum()

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
) -> pd.DataFrame:
    """One-at-a-time sensitivity analysis on key parameters.

    Args:
        base_revenue: Base-case annual revenue (EUR).
        total_capex: Total upfront CapEx (EUR).
        effective_life_years: Asset life (years).
        annual_degradation_cost: Annual degradation cost (EUR).
        discount_rate: Discount rate for NPV.
        vary: Dict mapping parameter name to [low, mid, high] multipliers
            applied to the base value. Defaults to standard ranges.

    Returns:
        DataFrame with columns: param, label, multiplier, value, npv.
    """
    if vary is None:
        vary = {
            "revenue": [0.7, 1.0, 1.3],
            "capex": [0.8, 1.0, 1.2],
            "discount_rate": [0.06, 0.08, 0.10],
            "lifetime": [15, 20, 25],
        }

    rows = []
    labels = {"revenue": "Low/Base/High", "capex": "Low/Base/High",
              "discount_rate": "6%/8%/10%", "lifetime": "15yr/20yr/25yr"}

    for param, values in vary.items():
        for val in values:
            rev = base_revenue * val if param == "revenue" else base_revenue
            capex = total_capex * val if param == "capex" else total_capex
            dr = val if param == "discount_rate" else discount_rate
            life = val if param == "lifetime" else effective_life_years
            deg = annual_degradation_cost

            n_years = int(life)
            pv_factor = sum(1 / (1 + dr) ** t for t in range(1, n_years + 1))
            npv = (rev - deg) * pv_factor - capex

            rows.append({
                "param": param,
                "value": val,
                "npv": round(npv, 2),
            })

    return pd.DataFrame(rows)
