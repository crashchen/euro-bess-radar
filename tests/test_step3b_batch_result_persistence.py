"""Step 3B: cockpit batch results persist behind a content fingerprint.

The multi-day replay and forecast-policy panels are button-run. Their result
must survive unrelated reruns (including the rerun a download click causes and
a chart-theme change) without re-solving, become visibly stale — with the
export blocked — as soon as any input the solvers read changes (including a
same-length price correction, a training-history correction outside the scored
window, and a reserve price correction), and update only when Run is clicked
again. Failure and missing-data disclosures persist under the same rules.

Solver calls are counted by wrapping EVERY ``simulate_*`` / ``solve_*`` /
``compute_*`` entry point imported into the cockpit module, so a derived
strategy (reserve, triple, stochastic) re-solved on a rerun is caught too.

New names are resolved inside each test so that, on the pre-3B baseline, each
test fails on its own rather than the whole module failing to import.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import src.pages.simulation_cockpit as cockpit

_SOLVER_NAME = re.compile(r"^(simulate|solve|compute)_")
_MULTI_DAY_KEY = "simulation_batch_result"
_FORECAST_KEY = "forecast_policy_result"
_STALE_MARKER = "Inputs changed since the last run"


# ── shared fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def solver_calls(monkeypatch) -> Counter:
    """Count every solver entry point the cockpit module can reach."""
    calls: Counter = Counter()

    def counting(name, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return func(*args, **kwargs)
        return wrapper

    wrapped = 0
    for name, obj in list(vars(cockpit).items()):
        if (
            _SOLVER_NAME.match(name)
            and inspect.isfunction(obj)
            and obj.__module__ != cockpit.__name__
        ):
            monkeypatch.setattr(cockpit, name, counting(name, obj))
            wrapped += 1
    assert wrapped >= 8, "solver entry points were not found to wrap"
    return calls


@pytest.fixture()
def downloads(monkeypatch) -> list:
    """Record what each rendered download button would export."""
    seen: list = []
    original = cockpit._cockpit_download_button

    def spy(tables, assumptions, **kwargs):
        seen.append({"tables": tables, "assumptions": assumptions, **kwargs})
        original(tables, assumptions, **kwargs)

    monkeypatch.setattr(cockpit, "_cockpit_download_button", spy)
    return seen


@pytest.fixture()
def two_scenarios(monkeypatch) -> None:
    """Keep the real stochastic solve small; it is part of the fingerprint."""
    monkeypatch.setattr(cockpit, "_STOCHASTIC_N_SCENARIOS", 2)


def _find_elements(node, type_name: str) -> list:
    found: list = []
    children = getattr(node, "children", None)
    if isinstance(children, dict):
        for child in children.values():
            if getattr(child, "type", "") == type_name:
                found.append(child)
            found.extend(_find_elements(child, type_name))
    return found


def _chart_specs(app: AppTest) -> list[str]:
    return [chart.proto.spec for chart in _find_elements(app.main, "plotly_chart")]


def _is_stale(app: AppTest) -> bool:
    return any(_STALE_MARKER in warning.value for warning in app.warning)


def _shows_result(app: AppTest) -> bool:
    return (
        len(app.dataframe) > 0
        and len(_find_elements(app.main, "download_button")) == 1
    )


def _assert_hidden_and_blocked(app: AppTest) -> None:
    assert not app.exception
    assert _is_stale(app)
    assert len(app.dataframe) == 0
    assert _find_elements(app.main, "download_button") == []
    assert _chart_specs(app) == []


def _metric(app: AppTest, label: str) -> str:
    return next(metric.value for metric in app.metric if metric.label == label)


def _solves(calls: Counter) -> Counter:
    return Counter({name: count for name, count in calls.items() if count})


# ── pure identity helpers ────────────────────────────────────────────────


def _content_fingerprint():
    return importlib.import_module("src.content_fingerprint")


def _prices(days: int = 3, *, tz: str = "UTC") -> pd.DataFrame:
    idx = pd.date_range("2025-06-01", periods=24 * days, freq="h", tz="UTC")
    t = np.arange(len(idx))
    frame = pd.DataFrame(
        {"price_eur_mwh": 50.0 + 40.0 * np.sin(t / 24 * 2 * np.pi)}, index=idx,
    )
    frame.index = frame.index.tz_convert(tz)
    frame.index.name = "timestamp"
    return frame


class TestFrameContentHash:
    def test_equal_content_gives_equal_digest(self) -> None:
        cf = _content_fingerprint()
        assert cf.frame_content_hash(_prices()) == cf.frame_content_hash(_prices())

    def test_same_length_value_correction_changes_digest(self) -> None:
        cf = _content_fingerprint()
        corrected = _prices()
        corrected.iloc[30, 0] += 0.01
        assert len(corrected) == len(_prices())
        assert cf.frame_content_hash(corrected) != cf.frame_content_hash(_prices())

    def test_index_timezone_is_part_of_identity(self) -> None:
        cf = _content_fingerprint()
        berlin = _prices(tz="Europe/Berlin")
        assert (berlin.index == _prices().index).all()
        assert cf.frame_content_hash(berlin) != cf.frame_content_hash(_prices())

    def test_shifted_index_with_same_values_changes_digest(self) -> None:
        cf = _content_fingerprint()
        shifted = _prices()
        shifted.index = shifted.index + pd.Timedelta(hours=1)
        assert cf.frame_content_hash(shifted) != cf.frame_content_hash(_prices())

    def test_column_name_and_dtype_are_part_of_identity(self) -> None:
        cf = _content_fingerprint()
        renamed = _prices().rename(columns={"price_eur_mwh": "other"})
        assert cf.frame_content_hash(renamed) != cf.frame_content_hash(_prices())
        ints = pd.DataFrame({"v": [1, 2]})
        floats = pd.DataFrame({"v": [1.0, 2.0]})
        assert cf.frame_content_hash(ints) != cf.frame_content_hash(floats)

    def test_none_is_distinct_from_an_empty_frame(self) -> None:
        cf = _content_fingerprint()
        assert cf.frame_content_hash(None) is None
        assert cf.frame_content_hash(pd.DataFrame()) is not None

    def test_payload_digest_ignores_key_order(self) -> None:
        cf = _content_fingerprint()
        assert cf.payload_digest({"a": 1, "b": [1.5, None]}) == cf.payload_digest(
            {"b": [1.5, None], "a": 1}
        )


def _multi_day_kwargs(**overrides) -> dict:
    ida = _prices().rename(columns={"price_eur_mwh": "intraday_price_eur_mwh"})
    kwargs = dict(
        primary_zone="DE_LU", primary_df=_prices(), intraday_df=ida,
        batch_dates=["2025-06-01", "2025-06-02"], mode="DA MILP Replay",
        zone_tz="UTC", carry_soc=True, power_mw=1.0, duration_hours=2,
        efficiency=0.88, capture_rate=1.0, capex_eur_kwh=0.0,
    )
    kwargs.update(overrides)
    return kwargs


class TestMultiDayFingerprint:
    def test_chart_theme_is_not_an_identity_input(self) -> None:
        params = inspect.signature(cockpit._multi_day_fingerprint).parameters
        assert "chart_template" not in params
        assert "assumptions" not in params

    @pytest.mark.parametrize(
        "override",
        [
            {"mode": "DA + IDA1 Replay"},
            {"batch_dates": ["2025-06-02"]},
            {"zone_tz": "Europe/Berlin"},
            {"carry_soc": False},
            {"power_mw": 2.0},
            {"duration_hours": 4},
            {"efficiency": 0.9},
            {"capture_rate": 0.9},
            {"capex_eur_kwh": 150.0},
            {"primary_zone": "NL"},
        ],
    )
    def test_each_solver_input_changes_identity(self, override: dict) -> None:
        base = cockpit._multi_day_fingerprint(**_multi_day_kwargs())
        assert cockpit._multi_day_fingerprint(**_multi_day_kwargs(**override)) != base

    def test_ida_content_counts_only_in_da_id_mode(self) -> None:
        corrected = _multi_day_kwargs()["intraday_df"].copy()
        corrected.iloc[5, 0] += 10.0
        da_only = cockpit._multi_day_fingerprint(**_multi_day_kwargs())
        assert cockpit._multi_day_fingerprint(
            **_multi_day_kwargs(intraday_df=corrected)
        ) == da_only
        da_id = cockpit._multi_day_fingerprint(
            **_multi_day_kwargs(mode="DA + IDA1 Replay")
        )
        assert cockpit._multi_day_fingerprint(
            **_multi_day_kwargs(mode="DA + IDA1 Replay", intraday_df=corrected)
        ) != da_id

    def test_model_constant_change_invalidates(self, monkeypatch) -> None:
        base = cockpit._multi_day_fingerprint(**_multi_day_kwargs())
        monkeypatch.setattr(cockpit, "DISPATCH_VOM_COST_EUR_MWH", 1.5)
        assert cockpit._multi_day_fingerprint(**_multi_day_kwargs()) != base


def _forecast_kwargs(**overrides) -> dict:
    ida = _prices().rename(columns={"price_eur_mwh": "intraday_price_eur_mwh"})
    capacity = pd.DataFrame(
        {"product_type": "FCR", "capacity_price_eur_mw": 8.0}, index=_prices().index,
    )
    kwargs = dict(
        primary_zone="DE_LU", primary_df=_prices(), intraday_df=ida,
        capacity_df=capacity, capacity_source="Session ancillary fallback",
        reserve_product="FCR", batch_dates=["2025-06-01", "2025-06-02"],
        zone_tz="UTC", power_mw=1.0, duration_hours=2, efficiency=0.88,
        bucket="hour_of_day", forecast_mode="loo", deadband_eur_per_mw=0.0,
        include_stochastic=True, stochastic_cap_pct=50.0,
    )
    kwargs.update(overrides)
    return kwargs


class TestForecastPolicyFingerprint:
    def test_chart_theme_is_not_an_identity_input(self) -> None:
        params = inspect.signature(cockpit._forecast_policy_fingerprint).parameters
        assert "chart_template" not in params
        assert "assumptions" not in params

    @pytest.mark.parametrize(
        "override",
        [
            {"batch_dates": ["2025-06-02"]},
            {"zone_tz": "Europe/Berlin"},
            {"power_mw": 2.0},
            {"duration_hours": 4},
            {"efficiency": 0.9},
            {"bucket": "hour_of_week"},
            {"forecast_mode": "walk_forward"},
            {"deadband_eur_per_mw": 1.0},
            {"reserve_product": "aFRR"},
            {"capacity_source": "Cached unified import"},
            {"include_stochastic": False},
            {"stochastic_cap_pct": 60.0},
            {"primary_zone": "NL"},
        ],
    )
    def test_each_solver_input_changes_identity(self, override: dict) -> None:
        base = cockpit._forecast_policy_fingerprint(**_forecast_kwargs())
        assert (
            cockpit._forecast_policy_fingerprint(**_forecast_kwargs(**override))
            != base
        )

    @pytest.mark.parametrize("frame", ["primary_df", "intraday_df", "capacity_df"])
    def test_same_length_content_correction_changes_identity(self, frame) -> None:
        corrected = _forecast_kwargs()[frame].copy()
        column = corrected.columns[-1]
        corrected.iloc[0, corrected.columns.get_loc(column)] += 1.0
        base = cockpit._forecast_policy_fingerprint(**_forecast_kwargs())
        assert (
            cockpit._forecast_policy_fingerprint(**_forecast_kwargs(**{frame: corrected}))
            != base
        )

    def test_training_history_outside_scored_dates_is_identity(self) -> None:
        corrected = _forecast_kwargs()["intraday_df"].copy()
        corrected.iloc[-1, 0] += 1.0  # 2025-06-03: loaded, not a scored date
        base = cockpit._forecast_policy_fingerprint(**_forecast_kwargs())
        assert (
            cockpit._forecast_policy_fingerprint(**_forecast_kwargs(intraday_df=corrected))
            != base
        )

    def test_unused_reserve_and_cap_inputs_do_not_invalidate(self) -> None:
        no_reserve = _forecast_kwargs(reserve_product=None, include_stochastic=False)
        base = cockpit._forecast_policy_fingerprint(**no_reserve)
        other_capacity = no_reserve["capacity_df"].assign(capacity_price_eur_mw=99.0)
        assert cockpit._forecast_policy_fingerprint(**{
            **no_reserve, "capacity_df": other_capacity,
            "capacity_source": "Cached unified import", "stochastic_cap_pct": 90.0,
        }) == base

    def test_stochastic_scenario_constants_invalidate(self, monkeypatch) -> None:
        base = cockpit._forecast_policy_fingerprint(**_forecast_kwargs())
        monkeypatch.setattr(cockpit, "_STOCHASTIC_SEED", 1)
        seeded = cockpit._forecast_policy_fingerprint(**_forecast_kwargs())
        monkeypatch.setattr(cockpit, "_STOCHASTIC_N_SCENARIOS", 3)
        assert len({base, seeded,
                    cockpit._forecast_policy_fingerprint(**_forecast_kwargs())}) == 3


# ── multi-day replay panel (AppTest, real DA replay solver) ─────────────


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
    ida = pd.DataFrame(
        {"intraday_price_eur_mwh": da["price_eur_mwh"].to_numpy() + 6.0 * np.cos(t)},
        index=idx,
    )
    ida.iloc[40, 0] += float(st.session_state.get("_t_ida_bump", 0.0))
    assumptions = pd.DataFrame([{
        "parameter": "Audit marker",
        "value": st.session_state.get("_t_assumption_value", "run-time"),
        "unit": "", "source": "Test", "affects": "Nothing",
    }])
    _render_multi_day_summary(
        primary_df=da,
        intraday_df=ida,
        dates=available_local_dates(da, tz="UTC"),
        mode=st.session_state.get("_t_mode", "DA MILP Replay"),
        zone_tz="UTC",
        power_mw=float(st.session_state.get("_t_power", 1.0)),
        duration_hours=2,
        efficiency=0.88,
        capture_rate=1.0,
        capex_eur_kwh=0.0,
        chart_template=st.session_state.get("_t_template", "plotly_dark"),
        assumptions=assumptions,
    )


@pytest.fixture()
def multi_day(solver_calls, downloads) -> AppTest:
    app = AppTest.from_function(_multi_day_app)
    app.run(timeout=30)
    return app


class TestMultiDayPanel:
    def test_before_any_run_the_panel_prompts_and_does_not_solve(
        self, multi_day, solver_calls,
    ) -> None:
        assert not multi_day.exception
        assert any("click Run" in info.value for info in multi_day.info)
        assert _solves(solver_calls) == Counter()

    def test_run_renders_the_result(self, multi_day, solver_calls) -> None:
        multi_day.button(key="simulation_batch_run").click().run(timeout=60)
        assert not multi_day.exception
        assert _solves(solver_calls) == Counter({"simulate_replay_batch": 1})
        assert _shows_result(multi_day)
        assert "Valid Days" in [metric.label for metric in multi_day.metric]

    def test_unrelated_rerun_theme_change_and_download_do_not_resolve(
        self, multi_day, solver_calls, downloads,
    ) -> None:
        multi_day.button(key="simulation_batch_run").click().run(timeout=60)
        first_specs = _chart_specs(multi_day)
        first_table = multi_day.dataframe[0].value

        multi_day.run(timeout=30)  # unrelated rerun
        assert _shows_result(multi_day)

        # A download click reruns the script with no input change; AppTest
        # cannot click download buttons, so the same rerun is issued directly.
        multi_day.run(timeout=30)
        assert _shows_result(multi_day)

        multi_day.session_state["_t_template"] = "plotly_white"
        multi_day.run(timeout=30)
        assert not multi_day.exception
        assert _shows_result(multi_day)
        assert not _is_stale(multi_day)
        assert _chart_specs(multi_day) != first_specs  # re-rendered, new theme
        pd.testing.assert_frame_equal(multi_day.dataframe[0].value, first_table)

        assert _solves(solver_calls) == Counter({"simulate_replay_batch": 1})
        assert len(downloads) == 4
        bundle = multi_day.session_state[_MULTI_DAY_KEY]
        for record in downloads:
            assert record["tables"]["Multi-day Replay"] is bundle["batch"]
            assert record["assumptions"] is bundle["export_assumptions"]

    def test_export_keeps_the_run_time_assumptions(
        self, multi_day, solver_calls, downloads,
    ) -> None:
        multi_day.button(key="simulation_batch_run").click().run(timeout=60)
        multi_day.session_state["_t_assumption_value"] = "changed-after-run"
        multi_day.run(timeout=30)
        assert _shows_result(multi_day)
        exported = downloads[-1]["assumptions"]
        marker = exported.loc[exported["parameter"] == "Audit marker", "value"]
        assert marker.tolist() == ["run-time"]
        assert _solves(solver_calls) == Counter({"simulate_replay_batch": 1})

    def test_same_length_price_correction_marks_stale_until_rerun(
        self, multi_day, solver_calls, downloads,
    ) -> None:
        multi_day.button(key="simulation_batch_run").click().run(timeout=60)
        before = downloads[-1]["tables"]["Multi-day Replay"]

        multi_day.session_state["_t_da_bump"] = 150.0
        multi_day.run(timeout=30)
        _assert_hidden_and_blocked(multi_day)
        multi_day.run(timeout=30)  # stays stale on further reruns
        _assert_hidden_and_blocked(multi_day)
        assert _solves(solver_calls) == Counter({"simulate_replay_batch": 1})

        multi_day.button(key="simulation_batch_run").click().run(timeout=60)
        assert not multi_day.exception
        assert _shows_result(multi_day) and not _is_stale(multi_day)
        assert _solves(solver_calls) == Counter({"simulate_replay_batch": 2})
        after = downloads[-1]["tables"]["Multi-day Replay"]
        assert after["total_revenue_eur"].sum() != pytest.approx(
            before["total_revenue_eur"].sum()
        )

    def test_sample_change_that_resolves_to_the_same_dates_keeps_the_result(
        self, multi_day, solver_calls,
    ) -> None:
        multi_day.button(key="simulation_batch_run").click().run(timeout=60)
        # Three days are loaded, so 7 latest / all loaded days are the same dates.
        for sample in ("7 latest days", "All loaded days"):
            multi_day.selectbox(key="simulation_batch_sample").set_value(sample)
            multi_day.run(timeout=30)
            assert _shows_result(multi_day) and not _is_stale(multi_day), sample
        assert _solves(solver_calls) == Counter({"simulate_replay_batch": 1})

    @pytest.mark.parametrize(
        ("kind", "key", "changed", "original"),
        [
            ("checkbox", "simulation_batch_carry_soc", False, True),
            ("state", "_t_power", 2.0, 1.0),
            ("state", "_t_mode", "DA + IDA1 Replay", "DA MILP Replay"),
        ],
    )
    def test_parameter_switch_marks_stale_and_revert_restores(
        self, multi_day, solver_calls, kind, key, changed, original,
    ) -> None:
        multi_day.button(key="simulation_batch_run").click().run(timeout=60)
        self._set(multi_day, kind, key, changed)
        _assert_hidden_and_blocked(multi_day)
        self._set(multi_day, kind, key, original)
        assert not multi_day.exception
        assert _shows_result(multi_day) and not _is_stale(multi_day)
        assert _solves(solver_calls) == Counter({"simulate_replay_batch": 1})

    def test_ida_correction_is_ignored_by_da_only_but_not_by_da_id_replay(
        self, multi_day, solver_calls,
    ) -> None:
        multi_day.button(key="simulation_batch_run").click().run(timeout=60)
        multi_day.session_state["_t_ida_bump"] = 40.0
        multi_day.run(timeout=30)
        assert _shows_result(multi_day) and not _is_stale(multi_day)

        multi_day.session_state["_t_mode"] = "DA + IDA1 Replay"
        multi_day.run(timeout=30)
        multi_day.button(key="simulation_batch_run").click().run(timeout=60)
        assert _shows_result(multi_day)
        multi_day.session_state["_t_ida_bump"] = 0.0
        multi_day.run(timeout=30)
        _assert_hidden_and_blocked(multi_day)
        assert _solves(solver_calls) == Counter({"simulate_replay_batch": 2})

    @staticmethod
    def _set(app: AppTest, kind: str, key: str, value) -> None:
        if kind == "checkbox":
            app.checkbox(key=key).set_value(value).run(timeout=30)
        else:
            app.session_state[key] = value
            app.run(timeout=30)


class TestMultiDayDisclosures:
    @staticmethod
    def _failed_batch(*, solver_failures: int) -> pd.DataFrame:
        batch = pd.DataFrame(columns=["date", "total_revenue_eur"])
        batch.attrs.update({
            "excluded_days": 3,
            "excluded_days_due_to_solver_failure": solver_failures,
            "model_available": False,
            "valid_days": 0,
        })
        return batch

    def _app(self, monkeypatch, solver_failures: int, calls: list) -> AppTest:
        def fake(*args, **kwargs):
            calls.append(1)
            return self._failed_batch(solver_failures=solver_failures)

        monkeypatch.setattr(cockpit, "simulate_replay_batch", fake)
        app = AppTest.from_function(_multi_day_app)
        app.run(timeout=30)
        app.button(key="simulation_batch_run").click().run(timeout=30)
        return app

    @pytest.mark.parametrize(
        ("solver_failures", "expected"),
        [(3, "Replay model unavailable"), (0, "No valid replay days")],
    )
    def test_failure_and_missing_notices_are_shown_on_run(
        self, monkeypatch, solver_failures, expected,
    ) -> None:
        """Compatibility control: unchanged disclosure contract on the run."""
        calls: list = []
        app = self._app(monkeypatch, solver_failures, calls)
        assert not app.exception
        assert any(expected in n.value for n in [*app.error, *app.warning])
        assert _find_elements(app.main, "download_button") == []
        assert len(calls) == 1

    @pytest.mark.parametrize(
        ("solver_failures", "expected"),
        [(3, "Replay model unavailable"), (0, "No valid replay days")],
    )
    def test_failure_and_missing_notices_survive_reruns_without_resolving(
        self, monkeypatch, solver_failures, expected,
    ) -> None:
        calls: list = []
        app = self._app(monkeypatch, solver_failures, calls)
        app.run(timeout=30)
        assert not app.exception
        assert any(expected in n.value for n in [*app.error, *app.warning])
        assert not _is_stale(app)
        assert _find_elements(app.main, "download_button") == []
        assert len(calls) == 1


# ── forecast-policy panel (AppTest, real solvers) ───────────────────────


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
    da.iloc[-18, 0] += float(st.session_state.get("_t_da_bump", 0.0))
    ida = pd.DataFrame(
        {"intraday_price_eur_mwh": da["price_eur_mwh"].to_numpy()
         + rng.normal(0.0, 15.0, len(idx))},
        index=idx,
    )
    # Hour 6 of the FIRST loaded day: training history for later targets.
    ida.iloc[6, 0] += float(st.session_state.get("_t_history_bump", 0.0))
    anc = None
    if st.session_state.get("_t_with_capacity", True):
        fcr = pd.DataFrame({
            "product_type": "FCR",
            "capacity_price_eur_mw": 8.0 + 2.0 * np.cos(t / 24 * 2 * np.pi),
            "energy_price_eur_mwh": np.nan,
        }, index=idx)
        fcr.iloc[-6, 1] += float(st.session_state.get("_t_capacity_bump", 0.0))
        afrr = fcr.assign(product_type="aFRR", capacity_price_eur_mw=5.0)
        anc = pd.concat([fcr, afrr])
    assumptions = pd.DataFrame([{
        "parameter": "Audit marker",
        "value": st.session_state.get("_t_assumption_value", "run-time"),
        "unit": "", "source": "Test", "affects": "Nothing",
    }])
    _render_forecast_policy_section(
        primary_df=da,
        intraday_df=ida,
        anc_df=anc,
        primary_zone="DE_LU",
        dates=available_local_dates(da, tz="UTC"),
        zone_tz="UTC",
        power_mw=float(st.session_state.get("_t_power", 1.0)),
        duration_hours=2,
        efficiency=0.88,
        chart_template=st.session_state.get("_t_template", "plotly_dark"),
        assumptions=assumptions,
    )


_FULL_FORECAST_SOLVES = Counter({
    "simulate_sequential_da_id_batch": 1,
    "solve_joint_capacity_batch": 1,
    "simulate_sequential_da_id_reserve_batch": 1,
    "simulate_stochastic_triple_batch": 1,
})


def _forecast_app_with_stochastic() -> AppTest:
    app = AppTest.from_function(_forecast_app)
    app.run(timeout=30)
    app.checkbox(key="forecast_policy_stochastic").check().run(timeout=30)
    return app


def _run_forecast(app: AppTest) -> None:
    app.button(key="forecast_policy_run").click().run(timeout=180)
    assert not app.exception


def _times(counter: Counter, n: int) -> Counter:
    return Counter({name: count * n for name, count in counter.items()})


@pytest.mark.slow
class TestForecastPolicyPanel:
    def test_run_renders_every_strategy_row_and_the_export(
        self, solver_calls, downloads, two_scenarios,
    ) -> None:
        app = _forecast_app_with_stochastic()
        assert _solves(solver_calls) == Counter()
        _run_forecast(app)
        assert _solves(solver_calls) == _FULL_FORECAST_SOLVES
        assert _shows_result(app)
        assert len(downloads[-1]["tables"]["Strategy comparison"]) == 7
        assert list(downloads[-1]["tables"]) == [
            "Strategy comparison", "Sequential DA+ID",
            "Sequential DA+ID+reserve", "Stochastic policy (per day)",
        ]

    def test_unrelated_rerun_theme_change_and_download_do_not_resolve_any_solver(
        self, solver_calls, downloads, two_scenarios,
    ) -> None:
        app = _forecast_app_with_stochastic()
        _run_forecast(app)
        first_specs = _chart_specs(app)
        app.run(timeout=30)  # unrelated rerun
        app.run(timeout=30)  # the rerun a download click issues
        app.session_state["_t_template"] = "plotly_white"
        app.session_state["_t_assumption_value"] = "changed-after-run"
        app.run(timeout=30)
        assert not app.exception
        assert _shows_result(app) and not _is_stale(app)
        assert _chart_specs(app) != first_specs
        assert _solves(solver_calls) == _FULL_FORECAST_SOLVES

        bundle = app.session_state[_FORECAST_KEY]
        assert len(downloads) == 4
        for record in downloads:
            assert record["tables"] is bundle["derived"]["export_tables"]
            assert record["assumptions"] is bundle["derived"]["export_assumptions"]
        exported = downloads[-1]["assumptions"]
        marker = exported.loc[exported["parameter"] == "Audit marker", "value"]
        assert marker.tolist() == ["run-time"]

    @pytest.mark.parametrize(
        ("state_key", "value"),
        [("_t_da_bump", 120.0), ("_t_capacity_bump", 30.0)],
    )
    def test_same_length_price_and_reserve_corrections_mark_stale(
        self, solver_calls, downloads, two_scenarios, state_key, value,
    ) -> None:
        app = _forecast_app_with_stochastic()
        _run_forecast(app)
        before = downloads[-1]["tables"]["Strategy comparison"]

        app.session_state[state_key] = value
        app.run(timeout=30)
        _assert_hidden_and_blocked(app)
        assert _solves(solver_calls) == _FULL_FORECAST_SOLVES

        _run_forecast(app)
        assert _shows_result(app) and not _is_stale(app)
        assert _solves(solver_calls) == _times(_FULL_FORECAST_SOLVES, 2)
        after = downloads[-1]["tables"]["Strategy comparison"]
        assert not after["window_revenue_eur"].equals(before["window_revenue_eur"])

    def test_training_history_correction_outside_the_scored_window_marks_stale(
        self, solver_calls, downloads,
    ) -> None:
        app = AppTest.from_function(_forecast_app)
        app.session_state["_t_days"] = 9
        app.session_state["_t_with_capacity"] = False
        app.run(timeout=30)
        app.selectbox(key="forecast_policy_sample").set_value("7 latest days")
        app.run(timeout=30)
        _run_forecast(app)
        scored = set(downloads[-1]["tables"]["Sequential DA+ID"]["date"].astype(str))
        assert "2025-06-01" not in scored  # the corrected day is history only
        assert _solves(solver_calls) == Counter({"simulate_sequential_da_id_batch": 1})
        mae_before = _metric(app, "MAE")

        app.session_state["_t_history_bump"] = 250.0
        app.run(timeout=30)
        _assert_hidden_and_blocked(app)

        _run_forecast(app)
        assert _shows_result(app)
        assert _solves(solver_calls) == Counter({"simulate_sequential_da_id_batch": 2})
        assert _metric(app, "MAE") != mae_before

    def test_parameter_switches_mark_stale_and_reverting_restores(
        self, solver_calls, two_scenarios,
    ) -> None:
        app = _forecast_app_with_stochastic()
        _run_forecast(app)

        # 7 vs 30 latest days resolve to the same four loaded dates: not stale.
        app.selectbox(key="forecast_policy_sample").set_value("7 latest days")
        app.run(timeout=30)
        assert _shows_result(app) and not _is_stale(app)

        switches = [
            ("selectbox", "forecast_policy_bucket", "Hour-of-week", "Hour-of-day"),
            ("selectbox", "forecast_policy_mode", "Walk-forward", "LOO cross-validation"),
            ("number_input", "forecast_policy_deadband", 1.0, 0.0),
            ("selectbox", "forecast_policy_reserve_product", "aFRR", "FCR"),
            ("slider", "forecast_policy_stochastic_cap", 60, 50),
            ("state", "_t_power", 2.0, 1.0),
        ]
        for kind, key, changed, original in switches:
            self._set(app, kind, key, changed)
            assert _is_stale(app), key
            _assert_hidden_and_blocked(app)
            self._set(app, kind, key, original)
            assert not app.exception, key
            assert _shows_result(app) and not _is_stale(app), key

        app.checkbox(key="forecast_policy_stochastic").uncheck().run(timeout=30)
        _assert_hidden_and_blocked(app)
        app.checkbox(key="forecast_policy_stochastic").check().run(timeout=30)
        assert _shows_result(app) and not _is_stale(app)
        assert _solves(solver_calls) == _FULL_FORECAST_SOLVES

    @staticmethod
    def _set(app: AppTest, kind: str, key: str, value) -> None:
        if kind == "state":
            app.session_state[key] = value
            app.run(timeout=30)
            return
        getattr(app, kind)(key=key).set_value(value).run(timeout=30)


_FORECAST_DISCLOSURES = pytest.mark.parametrize(
    ("solver_failures", "expected"),
    [(4, "Forecast-policy model unavailable"), (0, "No valid forecast-policy days")],
)


class TestForecastPolicyDisclosures:
    @staticmethod
    def _app(monkeypatch, solver_failures: int) -> AppTest:
        summary = {
            "excluded_days": 4,
            "excluded_days_due_to_solver_failure": solver_failures,
            "model_available": False,
            "valid_days": 0,
        }
        monkeypatch.setattr(
            cockpit, "simulate_sequential_da_id_batch",
            lambda *a, **k: (pd.DataFrame(columns=["date"]), dict(summary)),
        )
        app = AppTest.from_function(_forecast_app)
        app.run(timeout=30)
        app.checkbox(key="forecast_policy_stochastic").check().run(timeout=30)
        _run_forecast(app)
        return app

    @_FORECAST_DISCLOSURES
    def test_failure_and_missing_notices_are_shown_on_run(
        self, monkeypatch, solver_calls, solver_failures, expected,
    ) -> None:
        """Compatibility control: unchanged disclosure contract on the run."""
        app = self._app(monkeypatch, solver_failures)
        assert any(expected in n.value for n in [*app.error, *app.warning])
        assert _find_elements(app.main, "download_button") == []
        # The mocked sequential batch replaced the counted wrapper; no derived
        # reserve / triple / stochastic solver runs for an unavailable result.
        assert _solves(solver_calls) == Counter()

    @_FORECAST_DISCLOSURES
    def test_failure_and_missing_notices_survive_reruns_without_resolving(
        self, monkeypatch, solver_calls, solver_failures, expected,
    ) -> None:
        app = self._app(monkeypatch, solver_failures)
        app.run(timeout=30)
        assert not app.exception
        assert any(expected in n.value for n in [*app.error, *app.warning])
        assert not _is_stale(app)
        assert _find_elements(app.main, "download_button") == []
        assert _solves(solver_calls) == Counter()


class TestReserveSolverFailureNotice:
    """A notice emitted while solving is stored, not lost on the next rerun."""

    _MARKER = "DA + reserve strategy row omitted"

    @staticmethod
    def _app(monkeypatch) -> AppTest:
        def failed_joint(*args, **kwargs):
            frame = pd.DataFrame(columns=["date", "joint_total_revenue"])
            frame.attrs["excluded_days_due_to_solver_failure"] = 2
            return frame

        monkeypatch.setattr(cockpit, "solve_joint_capacity_batch", failed_joint)
        monkeypatch.setattr(
            cockpit, "_reserve_triple_totals",
            lambda *a, **k: {
                "triple_total": None, "realistic_total": None,
                "seq_per_day": None, "seq_summary": None,
                "triple_valid_days": None, "triple_da_baseline": None,
                "solver_failed_days": 0, "model_available": None,
            },
        )
        app = AppTest.from_function(_forecast_app)
        app.run(timeout=30)
        _run_forecast(app)
        return app

    def test_notice_is_shown_on_run(self, monkeypatch, solver_calls) -> None:
        """Compatibility control."""
        app = self._app(monkeypatch)
        assert sum(self._MARKER in w.value for w in app.warning) == 1
        assert _shows_result(app)
        assert _solves(solver_calls) == Counter({"simulate_sequential_da_id_batch": 1})

    def test_notice_survives_a_rerun_without_resolving(
        self, monkeypatch, solver_calls,
    ) -> None:
        app = self._app(monkeypatch)
        app.run(timeout=30)
        assert not app.exception
        assert sum(self._MARKER in w.value for w in app.warning) == 1
        assert _shows_result(app)
        assert _solves(solver_calls) == Counter({"simulate_sequential_da_id_batch": 1})


def test_panel_json_specs_are_real_plotly_figures() -> None:
    """Guard for the theme assertions: chart specs are JSON figure payloads."""
    app = AppTest.from_function(_multi_day_app)
    app.run(timeout=30)
    app.button(key="simulation_batch_run").click().run(timeout=60)
    specs = _chart_specs(app)
    assert specs
    assert all("data" in json.loads(spec) for spec in specs)
