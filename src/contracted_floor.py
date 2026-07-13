"""Contracted-floor cash-flow comparison primitives.

This module is deliberately independent of dispatch and market simulation. It
compares a quoted annual floor with an already-computed merchant baseline.
"""

from __future__ import annotations

import math

from src.scenario import annuity_pv_factor

MAX_FLOOR_TRAJECTORY_YEARS = 100.0


def _finite_float(name: str, value: float) -> float:
    """Return a finite float or raise a field-specific validation error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_contracted_floor_inputs(
    *,
    merchant_net_eur_per_mw_yr: float,
    power_mw: float,
    quoted_floor_eur_per_mw_yr: float,
    floor_tenor_years: float,
    contract_availability: float,
    discount_rate: float,
) -> tuple[float, float, float, float, float, float]:
    """Return the canonical values accepted by the shipped floor overlay."""
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
    return merchant_per_mw, power, quoted_floor_per_mw, tenor, availability, rate


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
    merchant_per_mw, power, quoted_floor_per_mw, tenor, availability, rate = (
        _validate_contracted_floor_inputs(
            merchant_net_eur_per_mw_yr=merchant_net_eur_per_mw_yr,
            power_mw=power_mw,
            quoted_floor_eur_per_mw_yr=quoted_floor_eur_per_mw_yr,
            floor_tenor_years=floor_tenor_years,
            contract_availability=contract_availability,
            discount_rate=discount_rate,
        )
    )

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


def compute_decaying_contracted_floor_overlay(
    *,
    merchant_net_eur_per_mw_yr: float,
    merchant_gross_eur_per_mw_yr: float,
    power_mw: float,
    quoted_floor_eur_per_mw_yr: float,
    floor_tenor_years: float,
    contract_availability: float = 1.0,
    discount_rate: float = 0.08,
    annual_decay_rate: float = 0.0,
    decay_floor_share: float = 0.0,
    floor_escalation_rate: float = 0.0,
) -> dict[str, object]:
    """Compose a decaying merchant trajectory with an escalating floor.

    Inactive composition inputs delegate to :func:`compute_contracted_floor_overlay`
    so the shipped scalar outputs remain bit-identical. The active path is a
    contract-window screening projection only; it does not alter dispatch.
    """
    merchant_per_mw, power, quoted_floor, tenor, availability, rate = (
        _validate_contracted_floor_inputs(
            merchant_net_eur_per_mw_yr=merchant_net_eur_per_mw_yr,
            power_mw=power_mw,
            quoted_floor_eur_per_mw_yr=quoted_floor_eur_per_mw_yr,
            floor_tenor_years=floor_tenor_years,
            contract_availability=contract_availability,
            discount_rate=discount_rate,
        )
    )
    gross_per_mw = _finite_float(
        "merchant_gross_eur_per_mw_yr", merchant_gross_eur_per_mw_yr
    )
    decay = _finite_float("annual_decay_rate", annual_decay_rate)
    decay_floor = _finite_float("decay_floor_share", decay_floor_share)
    escalation = _finite_float("floor_escalation_rate", floor_escalation_rate)

    if gross_per_mw < 0:
        raise ValueError("merchant_gross_eur_per_mw_yr must be >= 0")
    if gross_per_mw < merchant_per_mw:
        raise ValueError(
            "merchant_gross_eur_per_mw_yr must be >= "
            "merchant_net_eur_per_mw_yr"
        )
    if not 0.0 <= decay < 1.0:
        raise ValueError("annual_decay_rate must be in [0, 1)")
    if not 0.0 <= decay_floor <= 1.0:
        raise ValueError("decay_floor_share must be in [0, 1]")
    if not 0.0 <= escalation <= 1.0:
        raise ValueError("floor_escalation_rate must be in [0, 1]")

    active = (decay > 0.0 and decay_floor < 1.0) or escalation > 0.0
    if active and tenor > MAX_FLOOR_TRAJECTORY_YEARS:
        raise ValueError(
            "floor_tenor_years must be <= "
            f"{MAX_FLOOR_TRAJECTORY_YEARS:g} when composition is active"
        )

    base = compute_contracted_floor_overlay(
        merchant_net_eur_per_mw_yr=merchant_per_mw,
        power_mw=power,
        quoted_floor_eur_per_mw_yr=quoted_floor,
        floor_tenor_years=tenor,
        contract_availability=availability,
        discount_rate=rate,
    )
    common = {
        "merchant_gross_eur_per_mw_yr": gross_per_mw,
        "annual_decay_rate": decay,
        "decay_floor_share": decay_floor,
        "floor_escalation_rate": escalation,
        "composition_active": active,
    }
    if not active:
        return {
            **base,
            **common,
            "per_year": [],
            "crossover_year": None,
            "n_binding_years": None,
        }

    merchant_year_one = float(base["merchant_net_eur"])
    gross_year_one = gross_per_mw * power
    floor_year_one = float(base["effective_floor_eur"])
    rate = float(base["discount_rate"])
    n_full = math.floor(tenor)
    residual = tenor - n_full
    n_rows = n_full + (1 if residual > 0.0 else 0)
    per_year: list[dict[str, object]] = []
    merchant_pv = 0.0
    protected_pv = 0.0
    top_up_pv = 0.0
    crossover_year: int | None = None
    n_binding_years = 0

    for year in range(1, n_rows + 1):
        year_fraction = residual if year > n_full else 1.0
        weight = max((1.0 - decay) ** (year - 1), decay_floor)
        merchant_eur = merchant_year_one + gross_year_one * (weight - 1.0)
        floor_eur = floor_year_one * (1.0 + escalation) ** (year - 1)
        protected_eur = max(merchant_eur, floor_eur)
        top_up_eur = protected_eur - merchant_eur
        binding = top_up_eur > 0.0
        discount_factor = (1.0 + rate) ** (-year)

        merchant_pv += year_fraction * merchant_eur * discount_factor
        protected_pv += year_fraction * protected_eur * discount_factor
        top_up_pv += year_fraction * top_up_eur * discount_factor
        if binding:
            n_binding_years += 1
            if crossover_year is None:
                crossover_year = year
        per_year.append(
            {
                "year": year,
                "year_fraction": year_fraction,
                "weight": weight,
                "discount_factor": discount_factor,
                "merchant_eur": merchant_eur,
                "floor_eur": floor_eur,
                "protected_eur": protected_eur,
                "top_up_eur": top_up_eur,
                "binding": binding,
            }
        )

    return {
        **base,
        **common,
        "merchant_pv_eur": merchant_pv,
        "floor_protected_pv_eur": protected_pv,
        "floor_pv_uplift_eur": top_up_pv,
        "per_year": per_year,
        "crossover_year": crossover_year,
        "n_binding_years": n_binding_years,
    }
