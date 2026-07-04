"""Tests for the stochastic-MILP IDA scenario generator (Increment A)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ida_forecast import IDA_VALUE_COL
from src.ida_scenarios import SCENARIO_BASE_COL, build_ida_scenarios


def _history(
    n_days: int, *, start: str = "2026-05-01", freq: str = "h",
    day_shape: np.ndarray | None = None, noise_seed: int = 0,
) -> pd.DataFrame:
    """Build a UTC-indexed IDA history of ``n_days`` days at ``freq``.

    Each day is an optional deterministic ``day_shape`` (length = intervals per
    day) plus small per-day RNG noise, so error paths differ across days.
    """
    idx = pd.date_range(f"{start} 00:00", periods=n_days * _per_day(freq),
                        freq=freq, tz="UTC")
    per_day = _per_day(freq)
    if day_shape is None:
        day_shape = 50.0 + 10.0 * np.sin(np.arange(per_day) / per_day * 2 * np.pi)
    rng = np.random.default_rng(noise_seed)
    vals = np.tile(day_shape, n_days) + rng.normal(0, 3, size=n_days * per_day)
    return pd.DataFrame({IDA_VALUE_COL: vals}, index=idx)


def _per_day(freq: str) -> int:
    return {"h": 24, "15min": 96}[freq]


class TestBuildIdaScenarios:
    def test_mean_centred_scenarios_average_to_base_forecast(self) -> None:
        # §3 / §8-1 load-bearing identity: with mean-centring the scenario mean
        # per interval equals the base forecast exactly.
        hist = _history(20)
        out, meta = build_ida_scenarios(
            hist, target_dates=[date(2026, 5, 10)], n_scenarios=8, seed=1,
        )
        bundle = out[date(2026, 5, 10)]
        base = bundle["base_forecast"].to_numpy()
        scen_mean = bundle["scenarios"].mean(axis=0)
        np.testing.assert_allclose(scen_mean, base, atol=1e-9)
        assert meta["mean_centered"] is True

    def test_uncentred_scenarios_do_not_generally_average_to_base(self) -> None:
        hist = _history(20)
        out, _ = build_ida_scenarios(
            hist, target_dates=[date(2026, 5, 10)], n_scenarios=8, seed=1,
            mean_centered=False,
        )
        bundle = out[date(2026, 5, 10)]
        base = bundle["base_forecast"].to_numpy()
        scen_mean = bundle["scenarios"].mean(axis=0)
        assert not np.allclose(scen_mean, base, atol=1e-6)

    def test_scenario_shape_weights_and_timestamps(self) -> None:
        hist = _history(15)
        out, _ = build_ida_scenarios(
            hist, target_dates=[date(2026, 5, 8)], n_scenarios=6, seed=3,
        )
        bundle = out[date(2026, 5, 8)]
        assert bundle["scenarios"].shape == (6, 24)
        assert bundle["weights"].shape == (6,)
        np.testing.assert_allclose(bundle["weights"].sum(), 1.0)
        np.testing.assert_allclose(bundle["weights"], 1 / 6)
        assert len(bundle["timestamps"]) == 24
        assert str(bundle["timestamps"].tz) == "UTC"
        assert list(bundle["base_forecast"].index) == list(bundle["timestamps"])
        assert bundle["base_forecast"].name == SCENARIO_BASE_COL

    def test_resolution_partitioning_excludes_other_market_time_unit(self) -> None:
        # A 15-min day's error pool must not draw from 60-min history days.
        hourly = _history(10, start="2026-05-01", freq="h")
        quarter = _history(3, start="2026-05-11", freq="15min")
        hist = pd.concat([hourly, quarter]).sort_index()
        out, _ = build_ida_scenarios(
            hist, target_dates=[date(2026, 5, 12)], n_scenarios=2, seed=0,
        )
        bundle = out[date(2026, 5, 12)]
        # Target is 15-min (96 intervals); pool is only the OTHER two 15-min
        # days, never the ten hourly days.
        assert bundle["scenarios"].shape[1] == 96
        assert bundle["pool_size"] == 2

    def test_same_count_different_grid_day_is_excluded(self) -> None:
        # Codex catch: interval COUNT alone is unsafe. A sparse day with the
        # same count as the target but a different (hour, minute) grid must NOT
        # enter the pool (positional residual-add would misalign). Only a day
        # sharing the exact grid contributes.
        hist = _history(12, noise_seed=1)
        target = date(2026, 5, 10)
        matching = date(2026, 5, 3)   # will share the target's gap
        mismatched = date(2026, 5, 6)  # same count, different gap
        idx_dates = hist.index.date
        hours = hist.index.hour
        drop = (
            ((idx_dates == target) & (hours == 5))       # target drops hour 5
            | ((idx_dates == matching) & (hours == 5))   # same gap -> shares grid
            | ((idx_dates == mismatched) & (hours == 8))  # count 23, other grid
        )
        hist = hist[~drop]
        out, _ = build_ida_scenarios(
            hist, target_dates=[target], n_scenarios=2, seed=0,
        )
        bundle = out[target]
        assert bundle["scenarios"].shape[1] == 23
        # Only the same-grid day pools; the equal-count mismatched day is out.
        assert bundle["pool_size"] == 1

    def test_bundle_and_metadata_report_base_coverage(self) -> None:
        # §3: coverage/fallback reporting in the build_ida_forecast style.
        hist = _history(15)
        out, meta = build_ida_scenarios(
            hist, target_dates=[date(2026, 5, 8)], n_scenarios=4, seed=0,
        )
        bundle = out[date(2026, 5, 8)]
        # A dense multi-day history fully backs every hour-of-day bucket.
        assert bundle["base_coverage"] == 1.0
        assert bundle["base_fallback_points"] == 0
        assert meta["min_base_coverage"] == 1.0
        assert meta["total_fallback_points"] == 0

    def test_loo_does_not_leak_target_day_into_base(self) -> None:
        # Under LOO the base forecast excludes the target day, so a target-day
        # spike must not raise its own forecast.
        hist = _history(12, noise_seed=0)
        target = date(2026, 5, 6)
        spiked = hist.copy()
        mask = spiked.index.date == target
        spiked.loc[mask, IDA_VALUE_COL] += 1000.0
        base_clean = build_ida_scenarios(
            hist, target_dates=[target], n_scenarios=4, seed=0,
        )[0][target]["base_forecast"].to_numpy()
        base_spiked = build_ida_scenarios(
            spiked, target_dates=[target], n_scenarios=4, seed=0,
        )[0][target]["base_forecast"].to_numpy()
        np.testing.assert_allclose(base_clean, base_spiked, atol=1e-9)

    def test_walk_forward_uses_only_prior_days(self) -> None:
        # The first day has no prior history and is dropped.
        hist = _history(5)
        out, meta = build_ida_scenarios(
            hist, target_dates=[date(2026, 5, 1), date(2026, 5, 3)],
            n_scenarios=2, seed=0, forecast_mode="walk_forward",
        )
        assert date(2026, 5, 1) not in out  # no prior history
        assert date(2026, 5, 3) in out
        assert meta["forecast_mode"] == "walk_forward"

    def test_seed_is_reproducible(self) -> None:
        hist = _history(20)
        kwargs = dict(target_dates=[date(2026, 5, 10)], n_scenarios=8, seed=42)
        a = build_ida_scenarios(hist, **kwargs)[0][date(2026, 5, 10)]["scenarios"]
        b = build_ida_scenarios(hist, **kwargs)[0][date(2026, 5, 10)]["scenarios"]
        np.testing.assert_array_equal(a, b)

    def test_small_pool_samples_with_replacement_and_flags_it(self) -> None:
        # Fewer distinct same-shape error days than S -> replacement + flag.
        hist = _history(3)  # LOO on any target leaves 2 pool days
        out, meta = build_ida_scenarios(
            hist, target_dates=[date(2026, 5, 2)], n_scenarios=10, seed=0,
        )
        bundle = out[date(2026, 5, 2)]
        assert bundle["pool_size"] == 2
        assert bundle["sampled_with_replacement"] is True
        assert bundle["scenarios"].shape == (10, 24)
        assert meta["days_with_replacement"] == 1
        assert meta["min_pool_size"] == 2

    def test_large_pool_samples_without_replacement(self) -> None:
        hist = _history(30)
        out, _ = build_ida_scenarios(
            hist, target_dates=[date(2026, 5, 15)], n_scenarios=5, seed=0,
        )
        assert out[date(2026, 5, 15)]["sampled_with_replacement"] is False

    def test_empty_history_returns_empty(self) -> None:
        out, meta = build_ida_scenarios(
            pd.DataFrame(columns=[IDA_VALUE_COL]),
            target_dates=[date(2026, 5, 1)], n_scenarios=4,
        )
        assert out == {}
        assert meta["n_days_generated"] == 0
        assert meta["mode"] == "error_resample"

    def test_no_target_dates_returns_empty(self) -> None:
        out, _ = build_ida_scenarios(_history(5), target_dates=[], n_scenarios=4)
        assert out == {}

    def test_invalid_n_scenarios_raises(self) -> None:
        with pytest.raises(ValueError, match="n_scenarios"):
            build_ida_scenarios(
                _history(5), target_dates=[date(2026, 5, 2)], n_scenarios=0,
            )

    def test_invalid_forecast_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="forecast_mode"):
            build_ida_scenarios(
                _history(5), target_dates=[date(2026, 5, 2)], n_scenarios=2,
                forecast_mode="bogus",
            )

    def test_metadata_records_generation_knobs(self) -> None:
        hist = _history(10)
        _, meta = build_ida_scenarios(
            hist, target_dates=[date(2026, 5, 5), date(2026, 5, 6)],
            n_scenarios=4, seed=7,
        )
        assert meta["seed"] == 7
        assert meta["n_scenarios"] == 4
        assert meta["n_target_days"] == 2
        assert meta["n_days_generated"] == 2
        assert meta["bucket"] == "hour_of_day"
