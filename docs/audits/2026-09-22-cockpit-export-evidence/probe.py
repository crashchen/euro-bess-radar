"""Synthetic, revision-comparable solver and saved-export provenance probe."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.assumptions import build_assumptions_table
from src.pages.simulation_cockpit import (
    _compute_forecast_policy_bundle,
    _compute_multi_day_bundle,
)


def rows_by_parameter(table: pd.DataFrame | None) -> dict[str, dict]:
    if table is None:
        return {}
    return table.set_index("parameter").to_dict(orient="index")


index = pd.date_range("2025-06-01", periods=96, freq="h", tz="UTC")
hours = np.arange(len(index))
da_values = 50.0 + 40.0 * np.sin(hours / 24 * 2 * np.pi)
da = pd.DataFrame({"price_eur_mwh": da_values}, index=index)
ida = pd.DataFrame({"intraday_price_eur_mwh": da_values + 6.0 * np.cos(hours)}, index=index)
dates = list(pd.DatetimeIndex(index).date[::24])
assumptions = build_assumptions_table(
    power_mw=1.0, duration_hours=2.0, efficiency=0.88,
    capture_rate=0.7, capex_eur_kwh=150.0, use_lp_dispatch=False,
)

multi_day = {}
for mode in ("DA MILP Replay", "DA + IDA1 Replay"):
    result = _compute_multi_day_bundle(
        fingerprint="probe", primary_df=da, intraday_df=ida,
        batch_dates=dates[:2], mode=mode, zone_tz="UTC", carry_soc=True,
        power_mw=1.0, duration_hours=2, efficiency=0.88,
        capture_rate=1.0, capex_eur_kwh=150.0, assumptions=assumptions,
    )
    batch = result["batch"]
    export_rows = rows_by_parameter(result["export_assumptions"])
    multi_day[mode] = {
        "valid_days": len(batch),
        "total_revenue_eur": round(float(batch["total_revenue_eur"].sum()), 9),
        "degradation_cost_eur": round(float(batch["degradation_cost_eur"].sum()), 9),
        "dispatch": export_rows["Dispatch model"],
        "capex": export_rows["CapEx"],
    }

forecast = _compute_forecast_policy_bundle(
    fingerprint="probe", primary_zone="DE_LU", primary_df=da,
    intraday_df=ida, capacity_df=None, reserve_product=None,
    batch_dates=dates, zone_tz="UTC", power_mw=1.0,
    duration_hours=2, efficiency=0.88, bucket="hour_of_day",
    forecast_mode="loo", deadband_eur_per_mw=0.0,
    include_stochastic=False, stochastic_cap_pct=50.0,
    assumptions=assumptions,
)
derived = forecast["derived"]
forecast_rows = rows_by_parameter(derived["export_assumptions"])
comparison = derived["comparison"]
output = {
    "multi_day": multi_day,
    "forecast": {
        "valid_days": int(forecast["summary"]["valid_days"]),
        "comparison_revenue_eur": [
            round(float(value), 9)
            for value in comparison["window_revenue_eur"]
        ],
        "dispatch": forecast_rows["Dispatch model"],
    },
    "global_dispatch": rows_by_parameter(assumptions)["Dispatch model"],
    "global_capex": rows_by_parameter(assumptions)["CapEx"],
}
print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
