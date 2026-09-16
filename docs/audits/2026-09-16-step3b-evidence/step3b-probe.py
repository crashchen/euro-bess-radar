"""Step 3B probe: drive both cockpit batch panels and print what persists.

Runs unchanged against the baseline and the Step 3B head. Every
``simulate_*`` / ``solve_*`` / ``compute_*`` entry point imported into the
cockpit module is wrapped with a call counter that forwards to the real
solver. Data is synthetic and built in memory; the SQLite cache is pointed at
a temporary directory so the capacity lookup cannot read or write a real
cache.

Usage (repository root)::

    PYTHONPATH=. .venv/bin/python docs/audits/2026-09-16-step3b-evidence/step3b-probe.py
"""

from __future__ import annotations

import functools
import inspect
import re
import tempfile
from collections import Counter
from pathlib import Path

from streamlit.testing.v1 import AppTest

import src.data_ingestion as ingestion
import src.export as export
import src.pages.simulation_cockpit as cockpit

CALLS: Counter = Counter()


def _wrap_solvers() -> None:
    pattern = re.compile(r"^(simulate|solve|compute)_")
    for name, obj in list(vars(cockpit).items()):
        if (
            pattern.match(name)
            and inspect.isfunction(obj)
            and obj.__module__ != cockpit.__name__
        ):
            def counting(*args, _name=name, _func=obj, **kwargs):
                CALLS[_name] += 1
                return _func(*args, **kwargs)
            setattr(cockpit, name, functools.wraps(obj)(counting))


def _multi_day_app() -> None:
    import numpy as np
    import pandas as pd
    import streamlit as st

    from src.pages.simulation_cockpit import _render_multi_day_summary
    from src.simulation import available_local_dates

    idx = pd.date_range("2025-06-01", periods=72, freq="h", tz="UTC")
    t = np.arange(len(idx))
    da = pd.DataFrame(
        {"price_eur_mwh": 50.0 + 40.0 * np.sin(t / 24 * 2 * np.pi)}, index=idx,
    )
    da.iloc[30, 0] += float(st.session_state.get("_t_da_bump", 0.0))
    _render_multi_day_summary(
        primary_df=da, intraday_df=None,
        dates=available_local_dates(da, tz="UTC"), mode="DA MILP Replay",
        zone_tz="UTC", power_mw=1.0, duration_hours=2, efficiency=0.88,
        capture_rate=1.0, capex_eur_kwh=0.0,
        chart_template=st.session_state.get("_t_template", "plotly_dark"),
    )


def _forecast_app() -> None:
    import numpy as np
    import pandas as pd
    import streamlit as st

    from src.pages.simulation_cockpit import _render_forecast_policy_section
    from src.simulation import available_local_dates

    days = int(st.session_state.get("_t_days", 4))
    idx = pd.date_range("2025-06-01", periods=24 * days, freq="h", tz="UTC")
    t = np.arange(len(idx))
    rng = np.random.default_rng(1)
    da = pd.DataFrame(
        {"price_eur_mwh": 50.0 + 40.0 * np.sin(t / 24 * 2 * np.pi)}, index=idx,
    )
    ida = pd.DataFrame(
        {"intraday_price_eur_mwh": da["price_eur_mwh"].to_numpy()
         + rng.normal(0.0, 15.0, len(idx))},
        index=idx,
    )
    ida.iloc[6, 0] += float(st.session_state.get("_t_history_bump", 0.0))
    anc = None
    if st.session_state.get("_t_with_capacity", True):
        anc = pd.DataFrame({
            "product_type": "FCR",
            "capacity_price_eur_mw": 8.0 + 2.0 * np.cos(t / 24 * 2 * np.pi),
            "energy_price_eur_mwh": np.nan,
        }, index=idx)
        anc.iloc[-6, 1] += float(st.session_state.get("_t_capacity_bump", 0.0))
    _render_forecast_policy_section(
        primary_df=da, intraday_df=ida, anc_df=anc, primary_zone="DE_LU",
        dates=available_local_dates(da, tz="UTC"), zone_tz="UTC",
        power_mw=1.0, duration_hours=2, efficiency=0.88,
        chart_template=st.session_state.get("_t_template", "plotly_dark"),
    )


def _state(app: AppTest) -> str:
    def count(node, kind):
        children = getattr(node, "children", None)
        total = 0
        if isinstance(children, dict):
            for child in children.values():
                total += getattr(child, "type", "") == kind
                total += count(child, kind)
        return total

    stale = any("Inputs changed" in w.value for w in app.warning)
    solves = dict(sorted((k, v) for k, v in CALLS.items() if v))
    return (
        f"exception={bool(app.exception)} tables={len(app.dataframe)} "
        f"charts={count(app.main, 'plotly_chart')} "
        f"downloads={count(app.main, 'download_button')} stale={stale} "
        f"solver_calls={solves}"
    )


def _step(label: str, app: AppTest) -> None:
    print(f"  {label:<46} {_state(app)}")


def _probe_multi_day() -> None:
    print("[multi-day replay]")
    CALLS.clear()
    app = AppTest.from_function(_multi_day_app)
    app.run(timeout=60)
    _step("initial render", app)
    app.button(key="simulation_batch_run").click().run(timeout=120)
    _step("click Run", app)
    app.run(timeout=60)
    _step("unrelated rerun (= download click rerun)", app)
    app.session_state["_t_template"] = "plotly_white"
    app.run(timeout=60)
    _step("chart theme change", app)
    app.session_state["_t_da_bump"] = 150.0
    app.run(timeout=60)
    _step("same-length DA correction", app)
    app.button(key="simulation_batch_run").click().run(timeout=120)
    _step("click Run again", app)


def _probe_forecast() -> None:
    print("[forecast policy: DA+IDA1 + FCR reserve + stochastic, S=2]")
    CALLS.clear()
    app = AppTest.from_function(_forecast_app)
    app.run(timeout=60)
    app.checkbox(key="forecast_policy_stochastic").check().run(timeout=60)
    _step("initial render (stochastic ticked)", app)
    app.button(key="forecast_policy_run").click().run(timeout=300)
    _step("click Run", app)
    app.run(timeout=60)
    _step("unrelated rerun (= download click rerun)", app)
    app.session_state["_t_template"] = "plotly_white"
    app.run(timeout=60)
    _step("chart theme change", app)
    app.number_input(key="forecast_policy_deadband").set_value(1.0).run(timeout=60)
    _step("deadband 0 -> 1", app)
    app.number_input(key="forecast_policy_deadband").set_value(0.0).run(timeout=60)
    _step("deadband back to 0", app)
    app.session_state["_t_capacity_bump"] = 30.0
    app.run(timeout=60)
    _step("same-length reserve price correction", app)
    app.button(key="forecast_policy_run").click().run(timeout=300)
    _step("click Run again", app)

    print("[forecast policy: 9 loaded days, 7 scored, IDA history correction]")
    CALLS.clear()
    app = AppTest.from_function(_forecast_app)
    app.session_state["_t_days"] = 9
    app.session_state["_t_with_capacity"] = False
    app.run(timeout=60)
    app.selectbox(key="forecast_policy_sample").set_value("7 latest days")
    app.run(timeout=60)
    app.button(key="forecast_policy_run").click().run(timeout=300)
    _step("click Run", app)
    app.session_state["_t_history_bump"] = 250.0
    app.run(timeout=60)
    _step("IDA correction on unscored day 2025-06-01", app)


def main() -> None:
    _wrap_solvers()
    cockpit._STOCHASTIC_N_SCENARIOS = 2
    with tempfile.TemporaryDirectory() as tmp:
        ingestion.CACHE_DIR = Path(tmp)
        ingestion.DB_PATH = Path(tmp) / "bess_pulse.db"
        export.CACHE_DIR = Path(tmp)
        _probe_multi_day()
        _probe_forecast()


if __name__ == "__main__":
    main()
