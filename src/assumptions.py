"""Consolidated model-assumption audit for screening transparency.

Every screening number on the dashboard rests on a handful of haircuts and
constants (efficiency, capture rate, VOM, the 365.25-day annualisation, the
rebid deadband, …). They are otherwise scattered across the sidebar, config,
and individual panels, which makes "where did this number come from?" hard to
answer. `build_assumptions_table` lays them all out in one auditable place,
surfaced in the Data Trust tab alongside the data-provenance table.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.dispatch import DISPATCH_VOM_COST_EUR_MWH
from src.simulation import DAYS_PER_YEAR

ASSUMPTION_COLUMNS = ["parameter", "value", "unit", "source", "affects"]
# The sidebar DA-slippage capture row; the cockpit export overrides it because
# the cockpit uses its own capture haircut (or none, for the forecast panel).
CAPTURE_PARAM_LABEL = "Capture rate (DA slippage)"


def build_assumptions_table(
    *,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    capture_rate: float,
    capex_eur_kwh: float = 0.0,
    use_lp_dispatch: bool = False,
    rebid_share: float | None = None,
    deadband_eur_per_mw: float | None = None,
    forecast_mode: str | None = None,
    forecast_bucket: str | None = None,
) -> pd.DataFrame:
    """Build one row per model assumption / haircut driving screening output.

    Runtime values (power / duration / efficiency / capture / capex / dispatch
    model) come from the sidebar; the rest are config / math constants. The
    interactive panel knobs (``rebid_share``, ``deadband_eur_per_mw``,
    ``forecast_mode``, ``forecast_bucket``) are appended only when the caller
    supplies them — they are chosen inside the Revenue / Cockpit panels and
    default there.

    Returns:
        DataFrame with columns ``[parameter, value, unit, source, affects]``.
    """
    rows: list[tuple[str, str, str, str, str]] = [
        ("Power", f"{power_mw:g}", "MW", "Sidebar",
         "Scales all dispatch and revenue figures"),
        ("Duration", f"{duration_hours:g}", "h", "Sidebar",
         "Rolling spread window + SoC capacity (MWh = power x duration)"),
        ("Round-trip efficiency", f"{efficiency:.0%}", "", "Sidebar",
         "Charge/discharge losses in every MILP solve"),
        (CAPTURE_PARAM_LABEL, f"{capture_rate:.0%}", "", "Sidebar",
         "Haircut on realised DA arbitrage; NOT applied to the "
         "forecast-policy panel"),
        ("CapEx", f"{capex_eur_kwh:g}", "EUR/kWh", "Sidebar",
         "Payback period only (0 = skipped)"),
        ("Dispatch model",
         "MILP multi-cycle" if use_lp_dispatch else "Greedy single-cycle",
         "", "Sidebar", "Daily spread / revenue basis"),
        ("VOM cost", f"{DISPATCH_VOM_COST_EUR_MWH:g}", "EUR/MWh",
         "config (dispatch)",
         "Per-MWh throughput cost in the MILP objective + cycling decision"),
        ("Annualisation", f"{DAYS_PER_YEAR:g}", "days/yr", "config",
         "Scales sample-window revenue to annual (i.i.d. days; short or "
         "seasonal windows can mislead)"),
        ("Hours per year", f"{config.HOURS_PER_YEAR}", "h", "config",
         "Capacity-based annual conversions"),
        ("Ancillary capacity availability",
         f"{config.ANCILLARY_CAPACITY_AVAILABILITY:.0%}", "", "config",
         "Reserve capacity revenue derate"),
        ("Ancillary energy activation share",
         f"{config.ANCILLARY_ENERGY_ACTIVATION_SHARE:.0%}", "", "config",
         "Assumed activated-energy fraction for reserve products"),
        ("Max short-gap interpolation", f"{config.MAX_SHORT_GAP_HOURS:g}", "h",
         "config",
         "Gaps <= this are imputed; longer gaps drop the local day"),
        ("Continuous-replay interval cap",
         f"{config.MAX_CONTINUOUS_REPLAY_INTERVALS}", "intervals", "config",
         "Chunk size for multi-day SoC carry-over (soft reset at each "
         "boundary)"),
    ]
    if rebid_share is not None:
        rows.append((
            "IDA rebid share (screening)", f"{rebid_share:.0%}", "",
            "Revenue panel",
            "Phase-1 intraday uplift estimate (EU practitioner range "
            "0.10-0.40)",
        ))
    if deadband_eur_per_mw is not None:
        rows.append((
            "Rebid deadband", f"{deadband_eur_per_mw:g}", "EUR/MW/day",
            "Cockpit panel",
            "Min forecast uplift required to rebid in the sequential DA+ID "
            "policy",
        ))
    if forecast_mode is not None:
        rows.append((
            "IDA forecast mode", forecast_mode, "", "Cockpit panel",
            "loo (cross-val, may use future days) / walk_forward / in_sample",
        ))
    if forecast_bucket is not None:
        rows.append((
            "IDA forecast bucket", forecast_bucket, "", "Cockpit panel",
            "hour_of_day (robust) / hour_of_week (weekday-aware)",
        ))
    return pd.DataFrame(rows, columns=ASSUMPTION_COLUMNS)
