"""Deterministic export-copy probe; run with PYTHONPATH=. on base and candidate."""

from __future__ import annotations

import json
from io import BytesIO

import pandas as pd
from openpyxl import load_workbook

from src.assumptions import build_assumptions_table
from src.export import cockpit_tables_to_excel
from src.pages import simulation_cockpit as cockpit

base = build_assumptions_table(
    power_mw=1.0, duration_hours=2.0, efficiency=0.88,
    capture_rate=0.7, capex_eur_kwh=150.0, use_lp_dispatch=False,
)

def rows(frame):
    if frame is None:
        return None
    selected = frame.loc[frame["parameter"].isin(["Dispatch model", "CapEx"])]
    return [dict(zip(selected.columns, values, strict=True)) for values in selected.itertuples(index=False, name=None)]

multi = cockpit._multi_day_export_assumptions(
    base, mode="DA + IDA1 Replay", capture_rate=1.0, capex_eur_kwh=150.0,
)
forecast = cockpit._forecast_policy_export_assumptions(
    base, reserve_total=None, reserve_product=None, reserve_price=None,
    triple={"triple_total": None, "realistic_total": None},
    stochastic={"summary": None},
)
frontier_missing = cockpit._frontier_basis_export_assumptions(
    base.loc[base["parameter"] != "CapEx"].copy(), capex_eur_kwh=150.0,
)
book = load_workbook(BytesIO(cockpit_tables_to_excel(
    {"Strategy comparison": pd.DataFrame({"window_revenue_eur": [123.45]})},
    assumptions=forecast,
)), data_only=True)
print(json.dumps({
    "global": rows(base),
    "multi_day_da_id": rows(multi),
    "forecast": rows(forecast),
    "frontier_missing_global_capex": rows(frontier_missing),
    "none_global_frontier": cockpit._frontier_basis_export_assumptions(None, capex_eur_kwh=150.0),
    "numeric_export_cell": {
        "value": book["Strategy comparison"]["A2"].value,
        "type": book["Strategy comparison"]["A2"].data_type,
        "format": book["Strategy comparison"]["A2"].number_format,
    },
}, indent=2, sort_keys=True))
