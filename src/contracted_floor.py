"""Contracted-floor cash-flow comparison primitives.

This module is deliberately independent of dispatch and market simulation. It
compares a quoted annual floor with an already-computed merchant baseline.
"""

from __future__ import annotations

import math

from src.scenario import annuity_pv_factor


def _finite_float(name: str, value: float) -> float:
    """Return a finite float or raise a field-specific validation error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def compute_contracted_floor_overlay(
    *,
    merchant_net_eur_per_mw_yr: float,
    power_mw: float,
    quoted_floor_eur_per_mw_yr: float,
    floor_tenor_years: float,
    contract_availability: float = 1.0,
    discount_rate: float = 0.08,
) -> dict[str, float]:
    """Compare wear-net merchant cash flow with a contracted annual floor.

    The floor is protective, not additive: protected cash flow is
    ``max(merchant, effective_floor)``. Returned PV values cover only the
    stated floor tenor and are operating-cash-flow PVs, not project NPVs.
    """
    merchant_per_mw = _finite_float(
        "merchant_net_eur_per_mw_yr", merchant_net_eur_per_mw_yr
    )
    power = _finite_float("power_mw", power_mw)
    quoted_floor_per_mw = _finite_float(
        "quoted_floor_eur_per_mw_yr", quoted_floor_eur_per_mw_yr
    )
    tenor = _finite_float("floor_tenor_years", floor_tenor_years)
    availability = _finite_float("contract_availability", contract_availability)
    rate = _finite_float("discount_rate", discount_rate)

    if power < 0:
        raise ValueError("power_mw must be >= 0")
    if quoted_floor_per_mw < 0:
        raise ValueError("quoted_floor_eur_per_mw_yr must be >= 0")
    if tenor <= 0:
        raise ValueError("floor_tenor_years must be > 0")
    if not 0 <= availability <= 1:
        raise ValueError("contract_availability must be between 0 and 1")
    if rate < 0:
        raise ValueError("discount_rate must be >= 0")

    merchant_eur = merchant_per_mw * power
    quoted_floor_eur = quoted_floor_per_mw * power
    effective_floor_per_mw = quoted_floor_per_mw * availability
    effective_floor_eur = effective_floor_per_mw * power
    protected_eur = max(merchant_eur, effective_floor_eur)
    top_up_eur = max(protected_eur - merchant_eur, 0.0)
    pv_factor = annuity_pv_factor(tenor, rate)

    return {
        "merchant_net_eur": merchant_eur,
        "merchant_net_eur_per_mw_yr": merchant_per_mw,
        "quoted_floor_eur": quoted_floor_eur,
        "effective_floor_eur": effective_floor_eur,
        "effective_floor_eur_per_mw_yr": effective_floor_per_mw,
        "floor_protected_cashflow_eur": protected_eur,
        "annual_top_up_eur": top_up_eur,
        "merchant_pv_eur": merchant_eur * pv_factor,
        "floor_protected_pv_eur": protected_eur * pv_factor,
        "floor_pv_uplift_eur": top_up_eur * pv_factor,
        "floor_tenor_years": tenor,
        "discount_rate": rate,
        "contract_availability": availability,
    }
