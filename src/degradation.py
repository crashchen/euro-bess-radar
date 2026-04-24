"""Screening-level battery degradation and lifetime calculations."""

from __future__ import annotations

import math

DAYS_PER_YEAR = 365.25
DEFAULT_CYCLE_LIFE = 6000
DEFAULT_DOD_EXPONENT = 1.3
DEFAULT_CALENDAR_LIFE_YEARS = 20
DEFAULT_EOL_CAPACITY_PCT = 0.80


def _require_non_negative(name: str, value: float) -> None:
    """Raise when a numeric input is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _require_positive(name: str, value: float) -> None:
    """Raise when a numeric input is zero or negative."""
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def calculate_degradation_cost(
    n_cycles: float,
    capex_eur_kwh: float,
    capacity_kwh: float,
    cycle_life: float = DEFAULT_CYCLE_LIFE,
) -> dict[str, float]:
    """Estimate cycling degradation cost from full-equivalent cycles.

    Args:
        n_cycles: Full-equivalent cycles over the analysis period.
        capex_eur_kwh: Installed CapEx basis in EUR/kWh.
        capacity_kwh: Battery energy capacity in kWh.
        cycle_life: Full-equivalent cycle life to end-of-life.

    Returns:
        Dict with cost per full-equivalent cycle, total degradation cost, and
        cycle-life assumption.
    """
    _require_non_negative("n_cycles", n_cycles)
    _require_non_negative("capex_eur_kwh", capex_eur_kwh)
    _require_non_negative("capacity_kwh", capacity_kwh)
    _require_positive("cycle_life", cycle_life)

    cost_per_cycle = capex_eur_kwh * capacity_kwh / cycle_life
    return {
        "cost_per_cycle_eur": cost_per_cycle,
        "total_degradation_eur": cost_per_cycle * n_cycles,
        "cycle_life": cycle_life,
    }


def estimate_battery_lifetime(
    avg_cycles_per_day: float,
    cycle_life: float = DEFAULT_CYCLE_LIFE,
    calendar_life_years: float = DEFAULT_CALENDAR_LIFE_YEARS,
) -> dict[str, float | str]:
    """Estimate whether battery life is limited by cycling or calendar aging.

    Args:
        avg_cycles_per_day: Average full-equivalent cycles per day.
        cycle_life: Full-equivalent cycle life to end-of-life.
        calendar_life_years: Calendar-life cap regardless of cycling.

    Returns:
        Dict with cycle-limited life, calendar life, effective life, and the
        limiting factor.
    """
    _require_non_negative("avg_cycles_per_day", avg_cycles_per_day)
    _require_positive("cycle_life", cycle_life)
    _require_positive("calendar_life_years", calendar_life_years)

    if avg_cycles_per_day == 0:
        cycle_limited_years = math.inf
    else:
        cycle_limited_years = cycle_life / (avg_cycles_per_day * DAYS_PER_YEAR)

    effective_life = min(cycle_limited_years, calendar_life_years)
    limiting_factor = "cycling" if cycle_limited_years < calendar_life_years else "calendar"
    return {
        "cycle_limited_years": cycle_limited_years,
        "calendar_life_years": calendar_life_years,
        "effective_life_years": effective_life,
        "limiting_factor": limiting_factor,
    }


def calculate_net_revenue(
    annual_gross_revenue: float,
    annual_degradation_cost: float,
) -> dict[str, float]:
    """Subtract degradation cost from gross annual revenue.

    Args:
        annual_gross_revenue: Gross/headline annual revenue in EUR.
        annual_degradation_cost: Annual degradation cost in EUR.

    Returns:
        Dict with gross revenue, degradation cost, net revenue, and degradation
        as a percentage of gross revenue.
    """
    _require_non_negative("annual_gross_revenue", annual_gross_revenue)
    _require_non_negative("annual_degradation_cost", annual_degradation_cost)

    degradation_pct = (
        annual_degradation_cost / annual_gross_revenue * 100.0
        if annual_gross_revenue > 0 else 0.0
    )
    return {
        "gross_revenue_eur": annual_gross_revenue,
        "degradation_cost_eur": annual_degradation_cost,
        "net_revenue_eur": annual_gross_revenue - annual_degradation_cost,
        "degradation_pct": degradation_pct,
    }


def calculate_annual_throughput_mwh(
    avg_cycles_per_day: float,
    capacity_kwh: float,
) -> float:
    """Calculate annual charge plus discharge throughput in MWh.

    Args:
        avg_cycles_per_day: Average full-equivalent cycles per day.
        capacity_kwh: Battery energy capacity in kWh.

    Returns:
        Annual throughput in MWh, counting both charge and discharge legs.
    """
    _require_non_negative("avg_cycles_per_day", avg_cycles_per_day)
    _require_non_negative("capacity_kwh", capacity_kwh)
    return avg_cycles_per_day * (capacity_kwh / 1000.0) * 2.0 * DAYS_PER_YEAR


def calculate_levelized_cost_of_storage(
    capex_eur_kwh: float,
    capacity_kwh: float,
    effective_life_years: float,
    annual_throughput_mwh: float,
    opex_eur_kwh_yr: float = 0.0,
) -> float:
    """Calculate a simple lifetime LCOS in EUR/MWh.

    Args:
        capex_eur_kwh: Installed CapEx basis in EUR/kWh.
        capacity_kwh: Battery energy capacity in kWh.
        effective_life_years: Effective asset life in years.
        annual_throughput_mwh: Annual charge plus discharge throughput in MWh.
        opex_eur_kwh_yr: Optional fixed annual OpEx in EUR/kWh-year.

    Returns:
        Levelized cost of storage in EUR/MWh.
    """
    _require_non_negative("capex_eur_kwh", capex_eur_kwh)
    _require_non_negative("capacity_kwh", capacity_kwh)
    _require_non_negative("opex_eur_kwh_yr", opex_eur_kwh_yr)
    _require_positive("effective_life_years", effective_life_years)
    _require_positive("annual_throughput_mwh", annual_throughput_mwh)

    total_capex = capex_eur_kwh * capacity_kwh
    total_opex = opex_eur_kwh_yr * capacity_kwh * effective_life_years
    total_throughput = annual_throughput_mwh * effective_life_years
    return (total_capex + total_opex) / total_throughput
