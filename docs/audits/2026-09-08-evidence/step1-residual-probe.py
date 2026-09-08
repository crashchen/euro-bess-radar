"""Read-only synthetic residual-grid probe; no network, cache or source writes."""
import inspect
import json
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.simulation as sim
from src.analytics import calculate_intraday_uplift
from src.ida_forecast import build_ida_forecast, compute_forecast_skill
from src.ida_scenarios import build_ida_scenarios

days = [date(2025, 9, 1), date(2025, 9, 2)]
target = [days[-1]]
da_index = pd.date_range("2025-09-01", periods=48, freq="h", tz="Europe/Berlin").tz_convert("UTC")
ida_index = pd.date_range("2025-09-01", periods=192, freq="15min", tz="Europe/Berlin").tz_convert("UTC")
da = pd.DataFrame({"price_eur_mwh": 50.0}, index=da_index)
ida = pd.DataFrame({"intraday_price_eur_mwh": np.tile([20., 100., 20., 20.], 48)}, index=ida_index)
common = dict(dates=target, tz="Europe/Berlin", power_mw=1., duration_hours=1., efficiency=1.)
checks = [
    ("reserve_ceiling", lambda: sim.simulate_da_id_reserve_ceiling_batch(da, ida, 0., **common)),
    ("sequential", lambda: sim.simulate_sequential_da_id_batch(da, ida, forecast_mode="walk_forward", **common)),
    ("sequential_reserve", lambda: sim.simulate_sequential_da_id_reserve_batch(da, ida, None, **common)),
    ("stochastic", lambda: sim.simulate_stochastic_da_id_batch(da, ida, forecast_mode="walk_forward", n_scenarios=1, seed=7, **common)),
    ("stochastic_triple", lambda: sim.simulate_stochastic_triple_batch(da, ida, None, n_scenarios=1, seed=7, **common)),
]
solvers = [
    "solve_daily_da_id_reserve_dispatch", "solve_sequential_da_id_dispatch",
    "solve_sequential_da_id_reserve_dispatch", "solve_myopic_capped_da_id_dispatch",
    "solve_stochastic_da_id_dispatch", "solve_stochastic_triple_dispatch",
]
from contextlib import ExitStack

result = {"target": str(target[0]), "input_target_rows": {"DA": 24, "IDA": 96}, "paths": {}}
for name, call in checks:
    calls = []
    with ExitStack() as stack:
        for solver_name in solvers:
            original = getattr(sim, solver_name)
            def traced(*args, _name=solver_name, _fn=original, **kwargs):
                calls.append({"solver": _name, "n_intervals": len(args[0]), "dt": inspect.signature(_fn).bind(*args, **kwargs).arguments.get("dt")})
                return _fn(*args, **kwargs)
            stack.enter_context(patch.object(sim, solver_name, traced))
        output = call()
    summary = output[1] if isinstance(output, tuple) else output
    record = {key: summary[key] for key in ["valid_days", "excluded_days", "model_available"]}
    record["rows"] = output[0].to_dict(orient="records") if isinstance(output, tuple) else None
    if not isinstance(output, tuple):
        record["total_eur"] = output["total_eur"]
    record["solver_calls"] = calls
    result["paths"][name] = record

forecast, meta = build_ida_forecast(ida, target_dates=target, tz="Europe/Berlin", forecast_mode="walk_forward")
bundles, _ = build_ida_scenarios(ida, target_dates=target, tz="Europe/Berlin", forecast_mode="walk_forward", n_scenarios=1, seed=7)
skill = compute_forecast_skill(forecast, ida, da_prices=da, tz="Europe/Berlin")
result["forecast"] = {"n_rows": len(forecast), "coverage": meta["coverage"], "scenario_target_rows": len(bundles[target[0]]["timestamps"]), "skill_n_points": skill["n_points"], "da_baseline_matched_points": len(forecast.index.intersection(da.index)), "skill_naive_da_mae": skill["naive_da_mae"], "skill_vs_da": skill["skill_vs_da"]}
result["intraday_uplift_diagnostic"] = calculate_intraday_uplift(da, ida, tz="Europe/Berlin")
print(json.dumps(result, indent=2, default=str))
