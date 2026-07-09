"""Tests for the stochastic DA+ID batch + 3-policy comparison (Increment C1)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.stochastic_dispatch as stochastic_dispatch
from src.simulation import (
    simulate_sequential_da_id_batch,
    simulate_stochastic_da_id_batch,
)


def _history(days: int = 10, seed: int = 0, ida_sigma: float = 8.0):
    """UTC hourly DA + IDA history; IDA = DA + spread + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-05-01", periods=days * 24, freq="h", tz="UTC")
    shape = 50 + 20 * np.sin(np.arange(24) / 24 * 2 * np.pi)
    da = np.tile(shape, days) + rng.normal(0, 3, days * 24)
    ida = da + 5 + rng.normal(0, ida_sigma, days * 24)
    da_df = pd.DataFrame({"price_eur_mwh": da}, index=idx)
    ida_df = pd.DataFrame({"intraday_price_eur_mwh": ida}, index=idx)
    return da_df, ida_df


_KW = dict(power_mw=1.0, duration_hours=2.0)


def _force_canonical_tiebreak_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force only the canonical pass-2 solve to time out."""
    original_linprog = stochastic_dispatch.linprog

    def linprog_with_forced_pass2_timeout(c, *args, **kwargs):
        options = kwargs.get("options") or {}
        if "time_limit" in options:
            return SimpleNamespace(
                success=False,
                status=1,
                message="forced canonical tie-break timeout",
            )
        return original_linprog(c, *args, **kwargs)

    monkeypatch.setattr(stochastic_dispatch, "linprog", linprog_with_forced_pass2_timeout)


class TestStochasticBatch:
    def test_batch_aggregates_and_deltas(self) -> None:
        da_df, ida_df = _history()
        per_day, summ = simulate_stochastic_da_id_batch(
            da_df, ida_df, n_scenarios=8, seed=1, **_KW,
        )
        assert summ["valid_days"] == 10
        assert summ["excluded_days"] == 0
        assert summ["n_tiebreak_fallback_days"] == 0
        assert per_day["tiebreak_stable"].all()
        # Headline: robust policy value = stochastic - myopic realised.
        np.testing.assert_allclose(
            summ["total_policy_value_eur"],
            summ["total_stochastic_realised_eur"]
            - summ["total_myopic_realised_eur"], atol=1e-6,
        )
        np.testing.assert_allclose(
            summ["total_policy_value_eur"], per_day["policy_value_eur"].sum(),
            atol=1e-6,
        )
        # Window deltas equal the sum of per-day deltas.
        np.testing.assert_allclose(
            summ["total_commitment_value_eur"],
            per_day["commitment_value_eur"].sum(), atol=1e-6,
        )
        np.testing.assert_allclose(
            summ["total_distribution_value_eur"],
            per_day["distribution_value_eur"].sum(), atol=1e-6,
        )
        # The diagnostic split is an arithmetic decomposition of the headline
        # (commitment + distribution == policy_value). Since the v2 canonical
        # Stage-1 selector the split is tie-stable (see the dispatch-level
        # test_distribution_value_is_zero_at_infinite_cap pin); this test holds
        # the identity regardless.
        np.testing.assert_allclose(
            per_day["commitment_value_eur"] + per_day["distribution_value_eur"],
            per_day["policy_value_eur"], atol=1e-6,
        )
        np.testing.assert_allclose(
            per_day["commitment_value_eur"],
            per_day["coopt_realised_eur"] - per_day["myopic_realised_eur"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            per_day["distribution_value_eur"],
            per_day["stochastic_realised_eur"] - per_day["coopt_realised_eur"],
            atol=1e-6,
        )

    def test_tiebreak_fallback_days_counted_and_headline_unchanged(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        da_df, ida_df = _history(days=3, seed=23)
        kw = dict(n_scenarios=3, seed=17, **_KW)
        normal_per_day, normal = simulate_stochastic_da_id_batch(da_df, ida_df, **kw)
        assert normal["n_tiebreak_fallback_days"] == 0
        assert normal_per_day["tiebreak_stable"].all()

        _force_canonical_tiebreak_fallback(monkeypatch)
        fallback_per_day, fallback = simulate_stochastic_da_id_batch(
            da_df, ida_df, **kw,
        )
        assert fallback["valid_days"] == normal["valid_days"] > 0
        assert fallback["n_tiebreak_fallback_days"] == fallback["valid_days"]
        assert not fallback_per_day["tiebreak_stable"].any()
        # Headline equality is NOT guaranteed in general: fallback keeps the
        # pass-1 Stage-1 schedule, which is equal-OBJECTIVE but not necessarily
        # equal-settlement to the canonical one at realised != base (the C1
        # multi-optimum mechanism). It holds here because this fixture's Stage-1
        # optimum is non-degenerate (pass-1 == canonical). The guaranteed
        # invariant — fallback never changes the objective — is pinned at the
        # solver level in test_stochastic_dispatch. If a solver upgrade breaks
        # this line, relax it rather than treating fallback as a money bug.
        np.testing.assert_allclose(
            fallback["total_policy_value_eur"],
            normal["total_policy_value_eur"],
            atol=1e-6,
        )

    def test_stochastic_realised_at_most_coopt_ceiling(self) -> None:
        da_df, ida_df = _history(seed=2)
        _, summ = simulate_stochastic_da_id_batch(
            da_df, ida_df, n_scenarios=8, seed=3, **_KW,
        )
        assert (
            summ["total_stochastic_realised_eur"]
            <= summ["total_coopt_ceiling_eur"] + 1e-6
        )

    def test_infinite_cap_myopic_ties_sequential_row(self) -> None:
        # Scope §5/§8 regression anchor: at rebid_cap = inf with no reserve, the
        # capped-myopic policy equals the existing 9.2b sequential row.
        da_df, ida_df = _history(seed=4)
        _, stoch = simulate_stochastic_da_id_batch(
            da_df, ida_df, n_scenarios=8, seed=5, rebid_cap_mw=np.inf, **_KW,
        )
        _, seq = simulate_sequential_da_id_batch(da_df, ida_df, **_KW)
        np.testing.assert_allclose(
            stoch["total_myopic_realised_eur"], seq["total_realised_eur"],
            atol=1e-2,
        )
        np.testing.assert_allclose(
            stoch["total_da_only_eur"], seq["total_da_only_eur"], atol=1e-2,
        )

    def test_risk_block_structure(self) -> None:
        da_df, ida_df = _history(seed=6)
        _, summ = simulate_stochastic_da_id_batch(
            da_df, ida_df, n_scenarios=8, seed=7, **_KW,
        )
        rb = summ["risk_block"]
        assert rb["n"] == 10 * 8  # pooled per-(day, scenario)
        assert rb["p10"] <= rb["p50"] <= rb["p90"]
        assert rb["cvar90"] <= rb["p10"] + 1e-9  # downside tail mean

    def test_reserve_capacity_flows_into_totals(self) -> None:
        da_df, ida_df = _history(seed=8)
        no_res = simulate_stochastic_da_id_batch(
            da_df, ida_df, n_scenarios=6, seed=9, **_KW,
        )[1]
        with_res = simulate_stochastic_da_id_batch(
            da_df, ida_df, n_scenarios=6, seed=9, reserve_mw=0.3,
            reserve_price_eur_mw_h=10.0, rebid_cap_mw=1.0, **_KW,
        )[1]
        # Reserve capacity income lifts the reserve-aware totals above the
        # no-reserve run (energy schedules differ under headroom, but the
        # capacity fee is a clear positive shift on the realised totals).
        assert (
            with_res["total_stochastic_realised_eur"]
            > no_res["total_stochastic_realised_eur"]
        )
        assert (
            with_res["total_stochastic_realised_eur"]
            <= with_res["total_coopt_ceiling_eur"] + 1e-6
        )

    def test_local_timezone_does_not_exclude_all_days(self) -> None:
        # Regression (Gemini catch): the merged day index is in the zone's local
        # tz but scenario bundles are UTC; without aligning the lookup to UTC,
        # get_indexer mismatches and every day is silently excluded on any real
        # (tz-aware) zone.
        da_df, ida_df = _history(seed=14)
        _, summ = simulate_stochastic_da_id_batch(
            da_df, ida_df, tz="Europe/Berlin", n_scenarios=6, seed=1, **_KW,
        )
        assert summ["valid_days"] >= 8  # boundary partial days may drop

    def test_window_reserve_series_is_aligned_per_day(self) -> None:
        # A window-indexed reserve Series (the loaded-capacity standard) is
        # aligned per day by (local date, 4h block), so real block-of-day
        # reserve prices/requirements flow through the batch's day loop.
        da_df, ida_df = _history(seed=15)
        rprice = pd.Series(12.0, index=da_df.index)
        rmw = pd.Series(0.3, index=da_df.index)
        no_res = simulate_stochastic_da_id_batch(
            da_df, ida_df, n_scenarios=6, seed=2, **_KW,
        )[1]
        with_res = simulate_stochastic_da_id_batch(
            da_df, ida_df, n_scenarios=6, seed=2, reserve_mw=rmw,
            reserve_price_eur_mw_h=rprice, rebid_cap_mw=1.0, **_KW,
        )[1]
        assert with_res["valid_days"] == no_res["valid_days"] >= 1
        # Capacity income lifts the reserve-aware total above the no-reserve run.
        assert (
            with_res["total_stochastic_realised_eur"]
            > no_res["total_stochastic_realised_eur"]
        )

    def test_seed_reproducible(self) -> None:
        da_df, ida_df = _history(seed=10)
        kw = dict(n_scenarios=8, seed=11, **_KW)
        a = simulate_stochastic_da_id_batch(da_df, ida_df, **kw)[1]
        b = simulate_stochastic_da_id_batch(da_df, ida_df, **kw)[1]
        assert (
            a["total_stochastic_realised_eur"] == b["total_stochastic_realised_eur"]
        )

    def test_empty_and_no_dates(self) -> None:
        da_df, ida_df = _history(seed=12)
        empty = simulate_stochastic_da_id_batch(
            pd.DataFrame(columns=["price_eur_mwh"]),
            pd.DataFrame(columns=["intraday_price_eur_mwh"]), n_scenarios=4,
        )[1]
        assert empty["valid_days"] == 0
        assert empty["risk_block"]["n"] == 0
        # A requested date outside the data is excluded (no bundle / no day).
        from datetime import date
        per_day, summ = simulate_stochastic_da_id_batch(
            da_df, ida_df, dates=[date(2020, 1, 1)], n_scenarios=4,
        )
        assert per_day.empty
        assert summ["valid_days"] == 0
        assert summ["excluded_days"] == 1

    def test_negative_rebid_cap_raises(self) -> None:
        da_df, ida_df = _history(seed=13)
        with pytest.raises(ValueError, match="rebid_cap_mw must be >= 0"):
            simulate_stochastic_da_id_batch(
                da_df, ida_df, n_scenarios=4, rebid_cap_mw=-1.0, **_KW,
            )


def _reserve_series(index: pd.DatetimeIndex, *, skip_first_days: int = 0) -> pd.Series:
    """4h-block alternating reserve price series (12/4 EUR/MW/h).

    ``skip_first_days`` drops the leading days of reserve data so the
    walk-forward reserve forecast has no history for the first valid day —
    producing genuine zero-reserve (Stage-0 skip) days.
    """
    vals = np.where((np.arange(index.size) // 4) % 2 == 0, 12.0, 4.0).astype(float)
    series = pd.Series(vals, index=index)
    if skip_first_days:
        series = series[series.index >= index[skip_first_days * 24]]
    return series


class TestStochasticTripleBatch:
    """Increment V2-C: reserve-mode three-policy batch (v2 contract §3).

    Pins §6-1.2 (constrained collapse), §6-6 (anchor incl. zero-reserve day),
    §6-8 (no deadband knob — inert by construction), §6-9 (adverse-geometry
    headline) + summary/risk block + both tie-break fallback counters.
    """

    def test_batch_aggregates_and_headline(self) -> None:
        da_df, ida_df = _history(days=6)
        from src.simulation import simulate_stochastic_triple_batch

        per_day, summ = simulate_stochastic_triple_batch(
            da_df, ida_df, _reserve_series(da_df.index),
            n_scenarios=4, seed=1, rebid_cap_mw=0.8, **_KW,
        )
        # Walk-forward loses the first day (no bundle / no DA forecast).
        assert summ["valid_days"] == 5
        assert summ["excluded_days"] == 1
        assert summ["forecast_mode"] == "walk_forward"
        assert summ["n_stage0_skip_days"] == 0
        # Headline identity + diagnostic split decomposition.
        np.testing.assert_allclose(
            summ["total_policy_value_v2_eur"],
            summ["total_stochastic_realised_eur"]
            - summ["total_myopic_realised_eur"], atol=1e-6,
        )
        np.testing.assert_allclose(
            per_day["commitment_value_eur"] + per_day["distribution_value_eur"],
            per_day["policy_value_v2_eur"], atol=1e-6,
        )
        # §6-2 threading: per-day realised never exceeds the endogenous
        # ceiling; reserve is actually committed (capacity in the money).
        assert (
            per_day["stochastic_realised_eur"]
            <= per_day["coopt_ceiling_v2_eur"] + 1e-6
        ).all()
        assert (per_day["stochastic_avg_reserve_mw"] > 0).any()
        rb = summ["risk_block"]
        assert rb["n"] == summ["valid_days"] * 4
        assert rb["p10"] <= rb["p50"] <= rb["p90"]

    def test_constrained_collapse_matches_v1_batch(self) -> None:
        # §6-1.2: with NO reserve prices anywhere, every arm's Stage 0 skips
        # (r* == 0) and the v2 batch equals the v1 batch element-wise WHEN the
        # v1 comparator runs under the reserve-mode conventions — walk-forward,
        # deadband 0, same seed/S — including valid-day equality (the skip must
        # not exclude days the comparator keeps) and scenario-RNG equality (the
        # bundle is built once; Stage 0 consumes no randomness).
        da_df, ida_df = _history(days=6, seed=21)
        from src.simulation import simulate_stochastic_triple_batch

        v2_per_day, v2 = simulate_stochastic_triple_batch(
            da_df, ida_df, None, n_scenarios=4, seed=3, rebid_cap_mw=0.8, **_KW,
        )
        v1_per_day, v1 = simulate_stochastic_da_id_batch(
            da_df, ida_df, n_scenarios=4, seed=3, rebid_cap_mw=0.8,
            forecast_mode="walk_forward", min_rebid_uplift_eur=0.0, **_KW,
        )
        assert v2["valid_days"] == v1["valid_days"] > 0
        assert v2["n_stage0_skip_days"] == v2["valid_days"]
        assert v2_per_day["stage0_skipped"].all()
        np.testing.assert_array_equal(
            v2_per_day["date"].to_numpy(), v1_per_day["date"].to_numpy(),
        )
        for v2_col, v1_col in [
            ("da_only_eur", "da_only_eur"),
            ("myopic_realised_eur", "myopic_realised_eur"),
            ("coopt_realised_eur", "coopt_realised_eur"),
            ("stochastic_realised_eur", "stochastic_realised_eur"),
            ("policy_value_v2_eur", "policy_value_eur"),
        ]:
            np.testing.assert_allclose(
                v2_per_day[v2_col].to_numpy(), v1_per_day[v1_col].to_numpy(),
                atol=1e-6, err_msg=v2_col,
            )

    def test_anchor_ties_9_2b_reserve_batch_with_zero_reserve_day(self) -> None:
        # §6-6: at rebid_cap = inf (deadband inert) the cap-feasible myopic
        # baseline ties simulate_sequential_da_id_reserve_batch's realised
        # total EXACTLY, day by day — including zero-reserve days (reserve data
        # starts two days late, so the first valid days have no reserve
        # forecast: v2 skips, 9.2b safe-degrades to r = 0 — the case that
        # exposed the deadband asymmetry). Fixture optima are non-degenerate
        # (shaped DA + alternating block prices), so the canonical Stage-0
        # selector picks the same unique r as 9.2b's raw joint LP.
        da_df, ida_df = _history(days=8, seed=0)
        from src.simulation import (
            simulate_sequential_da_id_reserve_batch,
            simulate_stochastic_triple_batch,
        )

        rp = _reserve_series(da_df.index, skip_first_days=2)
        v2_per_day, _ = simulate_stochastic_triple_batch(
            da_df, ida_df, rp, n_scenarios=4, seed=0, rebid_cap_mw=np.inf, **_KW,
        )
        b92_per_day, _ = simulate_sequential_da_id_reserve_batch(
            da_df, ida_df, rp,
            dates=sorted({ts.date() for ts in da_df.index}), **_KW,
        )
        # Symmetric date-set equality (Codex audit NIT): the two batches must
        # keep EXACTLY the same days, not merely overlap.
        assert len(v2_per_day) > 0
        assert set(v2_per_day["date"]) == set(b92_per_day["date"])
        merged = v2_per_day.merge(b92_per_day, on="date")
        assert len(merged) == len(v2_per_day) == len(b92_per_day)
        assert (merged["myopic_avg_reserve_mw"] < 1e-9).any()  # zero-reserve day
        assert (merged["myopic_avg_reserve_mw"] > 0.5).any()   # live-reserve day
        np.testing.assert_allclose(
            merged["myopic_realised_eur"].to_numpy(),
            merged["realised_eur"].to_numpy(), atol=1e-4,
        )

    def test_adverse_geometry_headline_is_negative(self) -> None:
        # §6-9: the headline can LOSE and is not clamped. Flat DA makes the
        # myopic arm fill reserve at the full fee; the IDA history's huge
        # dispersion makes the scenario-aware Stage 0 hold back headroom for a
        # rebid opportunity that the realised path (== DA, no rebid value)
        # never delivers — so the stochastic arm forgoes fee income for
        # nothing and policy_value_v2 < 0.
        days = 5
        idx = pd.date_range("2026-06-01", periods=days * 24, freq="h", tz="UTC")
        da = np.full(days * 24, 50.0)
        ida = np.full(days * 24, 50.0)
        pattern = np.where(np.arange(24) < 12, -40.0, 40.0)
        for d in range(days - 1):  # dispersion history; final day realised == DA
            ida[d * 24:(d + 1) * 24] = 50.0 + (1 if d % 2 == 0 else -1) * pattern
        da_df = pd.DataFrame({"price_eur_mwh": da}, index=idx)
        ida_df = pd.DataFrame({"intraday_price_eur_mwh": ida}, index=idx)
        from src.simulation import simulate_stochastic_triple_batch

        per_day, summ = simulate_stochastic_triple_batch(
            da_df, ida_df, pd.Series(15.0, index=idx),
            n_scenarios=4, seed=0, rebid_cap_mw=np.inf, **_KW,
        )
        assert summ["valid_days"] > 0
        assert per_day["policy_value_v2_eur"].iloc[-1] < -1.0
        assert summ["total_policy_value_v2_eur"] < -1.0

    def test_no_deadband_or_forecast_mode_knobs(self) -> None:
        # §2.3 + §3 (the §6-8 batch half): walk-forward and the inert deadband
        # are enforced by CONSTRUCTION — the batch deliberately exposes neither
        # a forecast_mode nor a min_rebid_uplift_eur parameter, so no caller
        # can leak LOO history into the reserve gate or re-arm the deadband.
        # (The v1 path's deadband semantics are untouched — its own tests.)
        import inspect

        from src.simulation import simulate_stochastic_triple_batch

        params = inspect.signature(simulate_stochastic_triple_batch).parameters
        assert "forecast_mode" not in params
        assert "min_rebid_uplift_eur" not in params

    def test_stage0_fallback_days_counted(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Both tie-break fallback counters ride the #43 pattern. Forcing every
        # time-limited (canonical) solve to fail marks each valid day unstable
        # on BOTH stages — Stage-0 ties hit the HEADLINE (§2.2), so
        # n_stage0_fallback_days is the count the cockpit warning attaches to.
        da_df, ida_df = _history(days=4, seed=31)
        from src.simulation import simulate_stochastic_triple_batch

        kw = dict(n_scenarios=3, seed=5, rebid_cap_mw=0.8, **_KW)
        rp = _reserve_series(da_df.index)
        _, normal = simulate_stochastic_triple_batch(da_df, ida_df, rp, **kw)
        assert normal["n_stage0_fallback_days"] == 0
        assert normal["n_tiebreak_fallback_days"] == 0

        _force_canonical_tiebreak_fallback(monkeypatch)
        per_day, fallback = simulate_stochastic_triple_batch(
            da_df, ida_df, rp, **kw,
        )
        assert fallback["valid_days"] == normal["valid_days"] > 0
        assert fallback["n_stage0_fallback_days"] == fallback["valid_days"]
        assert not per_day["stage0_tiebreak_stable"].any()
        # The Stage-1 counter rides the same forced-timeout pattern (Codex
        # audit NIT: assert BOTH counters, not just Stage-0).
        assert fallback["n_tiebreak_fallback_days"] == fallback["valid_days"]
        assert not per_day["tiebreak_stable"].any()

    def test_seed_reproducible(self) -> None:
        da_df, ida_df = _history(days=4, seed=32)
        from src.simulation import simulate_stochastic_triple_batch

        kw = dict(n_scenarios=4, seed=7, rebid_cap_mw=0.8, **_KW)
        rp = _reserve_series(da_df.index)
        a = simulate_stochastic_triple_batch(da_df, ida_df, rp, **kw)[1]
        b = simulate_stochastic_triple_batch(da_df, ida_df, rp, **kw)[1]
        assert (
            a["total_stochastic_realised_eur"] == b["total_stochastic_realised_eur"]
        )
        assert a["total_policy_value_v2_eur"] == b["total_policy_value_v2_eur"]

    def test_negative_rebid_cap_raises(self) -> None:
        da_df, ida_df = _history(days=3, seed=33)
        from src.simulation import simulate_stochastic_triple_batch

        with pytest.raises(ValueError, match="rebid_cap_mw must be >= 0"):
            simulate_stochastic_triple_batch(
                da_df, ida_df, _reserve_series(da_df.index),
                n_scenarios=3, rebid_cap_mw=-1.0, **_KW,
            )

    def test_empty_inputs(self) -> None:
        from src.simulation import simulate_stochastic_triple_batch

        per_day, summ = simulate_stochastic_triple_batch(
            pd.DataFrame(columns=["price_eur_mwh"]),
            pd.DataFrame(columns=["intraday_price_eur_mwh"]),
            None, n_scenarios=3,
        )
        assert per_day.empty
        assert summ["valid_days"] == 0
        assert summ["risk_block"]["n"] == 0
