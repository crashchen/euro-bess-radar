"""Pure DA liquidity participation-cap calculations.

The result is a screening feasible-volume constraint, not a price-impact or
market-depth model. It deliberately has no solver or pandas dependency.
"""

from __future__ import annotations

import math


def _finite_float(name: str, value: float) -> float:
    """Return a finite float or raise a field-specific validation error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def compute_liquidity_cap(
    *,
    power_mw: float,
    zone_da_volume_mw: float,
    max_participation_share: float = 0.10,
) -> dict[str, float | bool]:
    """Calculate the DA executable-power cap from a zone-volume assertion.

    ``zone_da_volume_mw`` is the average single-sided cleared DA volume in
    MWh per hour (numerically MW), independent of the source market time unit.
    The returned cap limits physical charge and discharge power; installed
    power and energy capacity remain unchanged downstream.
    """
    power = _finite_float("power_mw", power_mw)
    volume = _finite_float("zone_da_volume_mw", zone_da_volume_mw)
    share = _finite_float("max_participation_share", max_participation_share)

    if power <= 0:
        raise ValueError("power_mw must be > 0")
    if volume <= 0:
        raise ValueError("zone_da_volume_mw must be > 0")
    if not 0 < share <= 1:
        raise ValueError("max_participation_share must be between 0 and 1")

    executable = min(power, share * volume)
    return {
        "executable_power_mw": executable,
        "binding": executable < power,
        "participation_at_full_power": power / volume,
        "max_participation_share": share,
        "zone_da_volume_mw": volume,
        "power_mw": power,
    }
