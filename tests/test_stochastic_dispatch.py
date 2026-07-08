"""Tests for the stochastic DA-commitment MILP core (Increment B1).

These pin the load-bearing identities the four-round scope review locked
(docs/design/stochastic-milp-v1.md): the decoupling theorem, the IDA==DA
collapse, the same-cap co-opt ceiling construction, the rebid-cap coupling,
and the reserve feasibility domain.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import src.stochastic_dispatch as stochastic_dispatch
from src.dispatch import solve_daily_joint_capacity_lp, solve_daily_lp
from src.stochastic_dispatch import (
    solve_myopic_capped_da_id_dispatch,
    solve_stochastic_da_commitment,
    solve_stochastic_da_id_dispatch,
    solve_stochastic_reserve_commitment,
    solve_stochastic_triple_dispatch,
    stochastic_coopt_ceiling,
    stochastic_coopt_ceiling_v2,
)


def _count_canonical_passes(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Count linprog calls that carry a time_limit option (canonical passes).

    Pass-1 solves never set ``time_limit``; every canonical tie-break pass
    (Stage-0 2a/2b and the Stage-1 #41 pass) does — so the counter observes
    exactly the selector activity of whatever runs while it is installed.
    """
    original_linprog = stochastic_dispatch.linprog
    calls = {"n": 0}

    def counting(c, *args, **kwargs):
        if "time_limit" in (kwargs.get("options") or {}):
            calls["n"] += 1
        return original_linprog(c, *args, **kwargs)

    monkeypatch.setattr(stochastic_dispatch, "linprog", counting)
    return calls


def _mean_centred_scenarios(base: np.ndarray, s: int, sigma: float, seed: int):
    """S scenarios = base + mean-centred Gaussian error paths (Increment A shape)."""
    rng = np.random.default_rng(seed)
    errs = rng.normal(0, sigma, size=(s, base.size))
    errs -= errs.mean(axis=0, keepdims=True)
    return base[None, :] + errs, np.full(s, 1.0 / s)


def _da_shape(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 50 + 25 * np.sin(np.arange(n) / n * 2 * np.pi) + rng.normal(0, 4, n)


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


class TestStochasticCommitment:
    def test_default_path_reports_canonical_tiebreak_applied(self) -> None:
        da = _da_shape(8)
        scen, w = _mean_centred_scenarios(da + 5, 3, 4.0, seed=31)
        r = solve_stochastic_da_commitment(
            da, scen, w, dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        assert r["success"]
        assert r["canonical_tiebreak_applied"] is True

    def test_forced_tiebreak_fallback_preserves_objective(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        da = _da_shape(8)
        base = da + 5
        scen, w = _mean_centred_scenarios(base, 3, 4.0, seed=32)
        realised = base + np.linspace(-2.0, 2.0, da.size)
        normal = solve_stochastic_da_commitment(
            da, scen, w, dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        assert normal["canonical_tiebreak_applied"] is True

        _force_canonical_tiebreak_fallback(monkeypatch)
        fallback = solve_stochastic_da_commitment(
            da, scen, w, dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        assert fallback["success"]
        assert fallback["canonical_tiebreak_applied"] is False
        np.testing.assert_allclose(
            fallback["expected_total_eur"], normal["expected_total_eur"], atol=1e-6,
        )

        dispatch = solve_stochastic_da_id_dispatch(
            da, scen, w, base, realised, dt=1.0, power_mw=1.0,
            duration_hours=2.0,
        )
        assert dispatch["success"]
        assert dispatch["canonical_tiebreak_applied"] is False

    def test_canonicalize_false_preserves_objective(self) -> None:
        # v2 §4 plumbing: the canonical Stage-1 pass is objective-preserving,
        # so skipping it (the ceilings' objective-only path) must return the
        # same expected total; canonical_tiebreak_applied is False and carries
        # no fallback meaning on that path.
        da = _da_shape(10)
        scen, w = _mean_centred_scenarios(da + 5, 3, 6.0, seed=41)
        kw = dict(dt=1.0, power_mw=1.0, duration_hours=2.0)
        with_pass = solve_stochastic_da_commitment(da, scen, w, **kw)
        without = solve_stochastic_da_commitment(
            da, scen, w, canonicalize=False, **kw,
        )
        assert without["success"]
        assert without["canonical_tiebreak_applied"] is False
        np.testing.assert_allclose(
            without["expected_total_eur"], with_pass["expected_total_eur"],
            atol=1e-6,
        )

    def test_expected_total_matches_weighted_scenario_totals(self) -> None:
        da = _da_shape(12)
        scen, w = _mean_centred_scenarios(da + 5, 4, 8.0, seed=1)
        r = solve_stochastic_da_commitment(
            da, scen, w, dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        assert r["success"]
        # The objective must equal the weighted mean of the per-scenario totals.
        np.testing.assert_allclose(
            r["expected_total_eur"], w @ r["scenario_total_eur"], atol=1e-6,
        )

    def test_stage1_financial_direction_known_answer(self) -> None:
        # Codex catch: the identity tests compare solver calls to each other, so
        # a flipped Stage-1 objective sign could pass them. This is an ABSOLUTE
        # ground-truth pin. With DA = [10, 90] and a flat mean IDA of 50, the
        # rational Stage-1 charges the cheap hour and discharges the dear one:
        #   financial = da_net·(DA - 50) = da_net[0]·(-40) + da_net[1]·(+40)
        # maximised by da_net[0] < 0 (charge low) and da_net[1] > 0 (discharge
        # high). A reversed sign would do the opposite and give a NEGATIVE
        # financial value; the correct solver's optimum is strictly positive
        # (doing nothing is always feasible at financial 0).
        da = np.array([10.0, 90.0])
        base = np.array([50.0, 50.0])  # single scenario == flat mean IDA
        r = solve_stochastic_da_commitment(
            da, base[None, :], np.array([1.0]), dt=1.0, power_mw=1.0,
            duration_hours=1.0, rebid_cap_mw=np.inf,
        )
        da_net = r["da_p_discharge"] - r["da_p_charge"]
        assert da_net[0] < -1e-6   # charge in the cheap hour
        assert da_net[1] > 1e-6    # discharge in the dear hour
        stage1_financial = float((da_net * (da - base)).sum())
        assert stage1_financial > 1e-6

    def test_decoupling_theorem_at_infinite_cap(self) -> None:
        # Scope §8-1: at rebid_cap = inf the pure-financial DA accounting
        # decouples the stages, so the stochastic Stage-1 commitment equals the
        # deterministic co-opt against the scenario-mean (= base) path. Pinned
        # via the Stage-1 financial objective value (robust to MILP multi-optima).
        da = _da_shape(12)
        base = da + 5
        scen, w = _mean_centred_scenarios(base, 5, 9.0, seed=2)
        mean_ida = w @ scen  # == base because scenarios are mean-centred
        np.testing.assert_allclose(mean_ida, base, atol=1e-9)

        r = solve_stochastic_da_commitment(
            da, scen, w, dt=1.0, power_mw=1.0, duration_hours=2.0,
            rebid_cap_mw=np.inf,
        )
        r_base = solve_stochastic_da_commitment(
            da, base[None, :], np.array([1.0]), dt=1.0, power_mw=1.0,
            duration_hours=2.0, rebid_cap_mw=np.inf,
        )
        fin = lambda res: float(  # noqa: E731 - local helper
            ((res["da_p_discharge"] - res["da_p_charge"]) * (da - base)).sum()
        )
        np.testing.assert_allclose(fin(r), fin(r_base), atol=1e-6)

    def test_canonical_stage1_selects_same_schedule_at_infinite_cap(self) -> None:
        # v2 canonical selector (docs/design §8): the decoupling theorem gives the
        # stochastic (S=N) and co-opt (S=1, base) Stage-1 the same optimal SET at
        # rebid_cap = inf, but v1 grabbed arbitrary (different) members, so the
        # commitment/distribution split carried tie noise. The min-throughput
        # lexicographic tie-break makes both solves pick the SAME least-churn
        # Stage-1 — so da_net is element-wise equal, not just equal in objective.
        da = _da_shape(12)
        base = da + 5
        scen, w = _mean_centred_scenarios(base, 6, 9.0, seed=7)
        r = solve_stochastic_da_commitment(
            da, scen, w, dt=1.0, power_mw=1.0, duration_hours=2.0,
            rebid_cap_mw=np.inf,
        )
        r_base = solve_stochastic_da_commitment(
            da, base[None, :], np.array([1.0]), dt=1.0, power_mw=1.0,
            duration_hours=2.0, rebid_cap_mw=np.inf,
        )
        net = r["da_p_discharge"] - r["da_p_charge"]
        net_base = r_base["da_p_discharge"] - r_base["da_p_charge"]
        np.testing.assert_allclose(net, net_base, atol=1e-6)

    def test_ida_equals_da_collapses_to_da_only(self) -> None:
        # Scope §8-3 (expected-level, no-reserve): with every scenario and the
        # realised path == DA, the stochastic total collapses to the DA-only
        # arbitrage. Pinned at rebid_cap = inf (stages decoupled).
        da = _da_shape(12)
        scen = np.tile(da, (3, 1))
        w = np.full(3, 1 / 3)
        r = solve_stochastic_da_commitment(
            da, scen, w, dt=1.0, power_mw=1.0, duration_hours=2.0,
            rebid_cap_mw=np.inf,
        )
        da_only = solve_daily_lp(da, dt=1.0, power_mw=1.0, duration_hours=2.0)
        np.testing.assert_allclose(
            r["expected_total_eur"], da_only["revenue_eur"], atol=1e-6,
        )

    def test_coopt_ceiling_equals_da_only_when_realised_is_da(self) -> None:
        da = _da_shape(12)
        ceiling = stochastic_coopt_ceiling(
            da, da, dt=1.0, power_mw=1.0, duration_hours=2.0, rebid_cap_mw=np.inf,
        )
        da_only = solve_daily_lp(da, dt=1.0, power_mw=1.0, duration_hours=2.0)
        np.testing.assert_allclose(ceiling, da_only["revenue_eur"], atol=1e-6)

    def test_rebid_cap_coupling_is_enforced(self) -> None:
        # A finite cap must bound |stage2_net - da_net| at every interval and
        # scenario, and (on a divergent case) reduce the expected total vs the
        # uncapped decoupled solution — proving the coupling actually binds.
        da = _da_shape(16, seed=3)
        # Scenarios pull the opposite way from DA so the uncapped s2 diverges.
        scen, w = _mean_centred_scenarios(120 - da, 4, 6.0, seed=4)
        capped = solve_stochastic_da_commitment(
            da, scen, w, dt=1.0, power_mw=1.0, duration_hours=2.0,
            rebid_cap_mw=0.2,
        )
        da_net = capped["da_p_discharge"] - capped["da_p_charge"]
        s2_net = capped["scenario_p_discharge"] - capped["scenario_p_charge"]
        assert np.all(np.abs(s2_net - da_net[None, :]) <= 0.2 + 1e-6)

        uncapped = solve_stochastic_da_commitment(
            da, scen, w, dt=1.0, power_mw=1.0, duration_hours=2.0,
            rebid_cap_mw=np.inf,
        )
        assert capped["expected_total_eur"] <= uncapped["expected_total_eur"] + 1e-6
        assert capped["expected_total_eur"] < uncapped["expected_total_eur"]

    def test_reserve_headroom_is_respected(self) -> None:
        da = _da_shape(16, seed=5)
        scen, w = _mean_centred_scenarios(da + 4, 4, 7.0, seed=6)
        reserve = np.where((np.arange(16) // 4) % 2 == 0, 0.3, 0.0)
        r = solve_stochastic_da_commitment(
            da, scen, w, dt=1.0, power_mw=1.0, duration_hours=2.0,
            reserve_mw=reserve, rebid_cap_mw=1.0,
        )
        used = r["scenario_p_charge"] + r["scenario_p_discharge"]
        assert np.all(used <= (1.0 - reserve)[None, :] + 1e-6)

    def test_cap_below_max_reserve_raises(self) -> None:
        # Scope §5 feasibility domain: rebid_cap must be >= max reserve.
        da = _da_shape(8)
        scen, w = _mean_centred_scenarios(da, 2, 3.0, seed=7)
        with pytest.raises(ValueError, match="max reserve"):
            solve_stochastic_da_commitment(
                da, scen, w, dt=1.0, power_mw=1.0, duration_hours=2.0,
                reserve_mw=0.5, rebid_cap_mw=0.2,
            )

    def test_default_cap_is_power_and_satisfies_reserve_domain(self) -> None:
        # Default rebid_cap = power_mw always satisfies cap >= reserve.
        da = _da_shape(8)
        scen, w = _mean_centred_scenarios(da, 2, 3.0, seed=8)
        r = solve_stochastic_da_commitment(
            da, scen, w, dt=1.0, power_mw=1.0, duration_hours=2.0,
            reserve_mw=0.4,
        )
        assert r["success"]
        assert r["rebid_cap_mw"] == 1.0

    def test_degenerate_inputs_return_gracefully(self) -> None:
        w = np.array([1.0])
        # empty
        assert not solve_stochastic_da_commitment(
            np.array([]), np.empty((1, 0)), w, dt=1.0,
        )["success"]
        # NaN in DA
        da = _da_shape(6)
        da[2] = np.nan
        scen = da[None, :].copy()
        assert not solve_stochastic_da_commitment(da, scen, w, dt=1.0)["success"]
        # weights not summing to 1
        da2 = _da_shape(6)
        assert not solve_stochastic_da_commitment(
            da2, np.tile(da2, (2, 1)), np.array([0.3, 0.3]), dt=1.0,
        )["success"]
        # negative weights (sum to 1 but break expected-value semantics)
        assert not solve_stochastic_da_commitment(
            da2, np.tile(da2, (2, 1)), np.array([1.2, -0.2]), dt=1.0,
        )["success"]

    def test_worst_case_15min_day_solves_within_budget(self) -> None:
        # Scope §6: a 15-min day at S=10 (~1056 binaries) must stay well under
        # the ~10s single-day budget. Loose bound guards a scaling regression
        # without flaking on a slow CI box.
        n = 96
        da = _da_shape(n, seed=9)
        scen, w = _mean_centred_scenarios(da + 3, 10, 10.0, seed=10)
        r = solve_stochastic_da_commitment(
            da, scen, w, dt=0.25, power_mw=1.0, duration_hours=2.0,
        )
        assert r["success"]
        assert r["solve_seconds"] > 0.0
        assert r["solve_seconds"] < 30.0


class TestStochasticExecution:
    """Increment B2: forecast-driven realised execution + decomposition."""

    @staticmethod
    def _case(n=12, s=6, seed=0, spread=5.0):
        rng = np.random.default_rng(seed)
        da = 50 + 20 * np.sin(np.arange(n) / n * 2 * np.pi)
        base = da + spread
        errs = rng.normal(0, 8, size=(s, n))
        errs -= errs.mean(axis=0, keepdims=True)
        scen = base[None, :] + errs
        w = np.full(s, 1.0 / s)
        realised = base + rng.normal(0, 6, n)
        return da, scen, w, base, realised

    def test_distribution_value_is_zero_at_infinite_cap(self) -> None:
        # v2 canonical selector: the batch's distribution_value is
        # stochastic_realised - coopt_realised, where coopt is the S=1 base-
        # forecast dispatch and stochastic is the S=N one. At rebid_cap = inf the
        # decoupling theorem + canonical tie-break make both commit the SAME
        # Stage-1, and they execute against the same base forecast and settle at
        # the same realised path, so the realised split must vanish (v1 left it
        # tie-noisy). This is the property that makes the split trustworthy.
        for seed in range(4):
            da, scen, w, base, realised = self._case(seed=seed)
            common = dict(
                dt=1.0, power_mw=1.0, duration_hours=2.0, rebid_cap_mw=np.inf,
            )
            coopt = solve_stochastic_da_id_dispatch(
                da, base[None, :], np.array([1.0]), base, realised, **common,
            )
            stoch = solve_stochastic_da_id_dispatch(
                da, scen, w, base, realised, **common,
            )
            assert coopt["success"] and stoch["success"]
            np.testing.assert_allclose(
                stoch["realised_total_eur"], coopt["realised_total_eur"], atol=1e-4,
            )

    def test_realised_at_most_coopt_ceiling(self) -> None:
        # The key by-construction pin (§2.4/§8-4): the executed (Stage-1
        # stochastic commit, Stage-2 forecast-optimal capped) schedule is
        # feasible for the same-cap perfect-foresight co-opt, so realised
        # cannot exceed it. Checked WITHOUT relying on the clamp.
        for seed in range(6):
            da, scen, w, base, realised = self._case(seed=seed)
            r = solve_stochastic_da_id_dispatch(
                da, scen, w, base, realised, dt=1.0, power_mw=1.0,
                duration_hours=2.0,
            )
            assert r["success"]
            assert r["realised_total_eur"] <= r["coopt_ceiling_eur"] + 1e-6

    def test_decomposition_identities(self) -> None:
        da, scen, w, base, realised = self._case(seed=1)
        r = solve_stochastic_da_id_dispatch(
            da, scen, w, base, realised, dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        np.testing.assert_allclose(
            r["captured_uplift_eur"],
            r["realised_total_eur"] - r["da_only_revenue_eur"], atol=1e-6,
        )
        np.testing.assert_allclose(
            r["forecast_error_cost_eur"],
            r["coopt_ceiling_eur"] - r["realised_total_eur"], atol=1e-6,
        )
        assert r["forecast_error_cost_eur"] >= -1e-9

    def test_ida_equals_da_realised_collapses_to_da_only(self) -> None:
        # Scope §8-3 (no-reserve, default execution): scenarios/forecast/
        # realised all == DA -> realised == DA-only arbitrage.
        n = 12
        da = 50 + 20 * np.sin(np.arange(n) / n * 2 * np.pi)
        scen = np.tile(da, (6, 1))
        w = np.full(6, 1 / 6)
        r = solve_stochastic_da_id_dispatch(
            da, scen, w, da, da, dt=1.0, power_mw=1.0, duration_hours=2.0,
            rebid_cap_mw=np.inf,
        )
        da_only = solve_daily_lp(da, dt=1.0, power_mw=1.0, duration_hours=2.0)
        np.testing.assert_allclose(
            r["realised_total_eur"], da_only["revenue_eur"], atol=1e-4,
        )

    def test_deadband_hold_settles_to_stochastic_hold(self) -> None:
        # A huge hurdle holds the committed schedule on a no-reserve day, and
        # realised then equals stochastic_hold (NOT da_only in general, §5).
        da, scen, w, base, realised = self._case(seed=2)
        r = solve_stochastic_da_id_dispatch(
            da, scen, w, base, realised, dt=1.0, power_mw=1.0, duration_hours=2.0,
            min_rebid_uplift_eur=1e9,
        )
        assert r["rebid"] is False
        np.testing.assert_allclose(
            r["realised_total_eur"], r["stochastic_hold_eur"], atol=1e-6,
        )

    def test_reserve_day_always_rebids(self) -> None:
        # On a reserve day the deadband is inert (holding a full-power Stage-1
        # can violate headroom), so it re-dispatches even at a huge hurdle.
        da, scen, w, base, realised = self._case(n=16, seed=3)
        reserve = np.where((np.arange(16) // 4) % 2 == 0, 0.3, 0.0)
        r = solve_stochastic_da_id_dispatch(
            da, scen, w, base, realised, dt=1.0, power_mw=1.0, duration_hours=2.0,
            reserve_mw=reserve, rebid_cap_mw=1.0, min_rebid_uplift_eur=1e9,
        )
        assert r["rebid"] is True
        assert r["realised_total_eur"] <= r["coopt_ceiling_eur"] + 1e-6

    def test_captured_uplift_not_clamped_when_forecast_misleads(self) -> None:
        # captured_uplift may be negative when the forecast-driven rebid loses
        # against DA-only; it must NOT be clamped to zero. Deterministic
        # construction (solver-multi-optimum independent): flat DA so
        # da_only == 0, and a forecast [0,100,0,100] that the realised path
        # [100,0,100,0] exactly contradicts, so the rebid buys high / sells low
        # and settles deeply negative (~-299, far from the zero boundary). inf
        # cap lets the bad rebid execute fully.
        da = np.array([50.0, 50.0, 50.0, 50.0])
        base = np.array([0.0, 100.0, 0.0, 100.0])
        realised = np.array([100.0, 0.0, 100.0, 0.0])
        r = solve_stochastic_da_id_dispatch(
            da, base[None, :], np.array([1.0]), base, realised, dt=1.0,
            power_mw=1.0, duration_hours=1.0, rebid_cap_mw=np.inf,
        )
        assert r["captured_uplift_eur"] < 0
        np.testing.assert_allclose(
            r["captured_uplift_eur"],
            r["realised_total_eur"] - r["da_only_revenue_eur"], atol=1e-6,
        )

    def test_coopt_ceiling_can_exceed_legacy_ceiling(self) -> None:
        # Scope §2.4: the co-opt Stage-1 optimises the DA-vs-IDA spread, so the
        # same-cap co-opt ceiling can legitimately beat the legacy DA+ID ceiling
        # (whose Stage-1 is myopic DA-only). Not a violation — information.
        da, scen, w, base, realised = self._case(seed=4)
        r = solve_stochastic_da_id_dispatch(
            da, scen, w, base, realised, dt=1.0, power_mw=1.0, duration_hours=2.0,
            rebid_cap_mw=np.inf,
        )
        assert r["coopt_ceiling_eur"] >= r["legacy_ceiling_eur"] - 1e-6

    def test_reserve_capacity_settlement(self) -> None:
        # Scope §2.3: with a reserve price, the committed reserve earns the same
        # capacity fee as the 9.2b dispatch, added as a constant to the
        # reserve-aware totals + coopt ceiling (da_only / legacy stay
        # no-reserve). realised <= coopt still holds with capacity.
        da, scen, w, base, realised = self._case(n=16, seed=8)
        reserve = np.where((np.arange(16) // 4) % 2 == 0, 0.3, 0.0)
        rprice = np.full(16, 12.0)
        base_kw = dict(
            dt=1.0, power_mw=1.0, duration_hours=2.0, reserve_mw=reserve,
            rebid_cap_mw=1.0,
        )
        no_price = solve_stochastic_da_id_dispatch(da, scen, w, base, realised, **base_kw)
        with_price = solve_stochastic_da_id_dispatch(
            da, scen, w, base, realised, reserve_price_eur_mw_h=rprice, **base_kw,
        )
        expected_cap = float((rprice * 0.95 * reserve * 1.0).sum())
        assert no_price["capacity_revenue_eur"] == 0.0
        np.testing.assert_allclose(
            with_price["capacity_revenue_eur"], expected_cap, atol=1e-6,
        )
        # Constant offset on the reserve-aware totals + ceiling + hold.
        for key in ("realised_total_eur", "coopt_ceiling_eur", "stochastic_hold_eur"):
            np.testing.assert_allclose(
                with_price[key], no_price[key] + expected_cap, atol=1e-4,
            )
        # da_only / legacy are the no-reserve baselines, unchanged.
        assert with_price["da_only_revenue_eur"] == no_price["da_only_revenue_eur"]
        assert with_price["legacy_ceiling_eur"] == no_price["legacy_ceiling_eur"]
        assert with_price["realised_total_eur"] <= with_price["coopt_ceiling_eur"] + 1e-6
        # The constant cancels in forecast_error_cost = coopt - realised.
        np.testing.assert_allclose(
            with_price["forecast_error_cost_eur"],
            no_price["forecast_error_cost_eur"], atol=1e-4,
        )

    def test_negative_reserve_price_is_floored_to_zero(self) -> None:
        # 9.2b convention: capacity prices are non-negative, so a stray negative
        # floors to 0 rather than subtracting revenue.
        da, scen, w, base, realised = self._case(n=16, seed=9)
        reserve = np.full(16, 0.3)
        base_kw = dict(
            dt=1.0, power_mw=1.0, duration_hours=2.0, reserve_mw=reserve,
            rebid_cap_mw=1.0,
        )
        neg = solve_stochastic_da_id_dispatch(
            da, scen, w, base, realised,
            reserve_price_eur_mw_h=np.full(16, -12.0), **base_kw,
        )
        zero = solve_stochastic_da_id_dispatch(
            da, scen, w, base, realised,
            reserve_price_eur_mw_h=np.zeros(16), **base_kw,
        )
        assert neg["capacity_revenue_eur"] == 0.0
        np.testing.assert_allclose(
            neg["realised_total_eur"], zero["realised_total_eur"], atol=1e-6,
        )

    def test_negative_rebid_cap_raises(self) -> None:
        da, scen, w, base, realised = self._case(seed=5)
        with pytest.raises(ValueError, match="rebid_cap_mw must be >= 0"):
            solve_stochastic_da_id_dispatch(
                da, scen, w, base, realised, dt=1.0, rebid_cap_mw=-0.5,
            )

    def test_expected_total_passed_through_from_commitment(self) -> None:
        da, scen, w, base, realised = self._case(seed=6)
        commit = solve_stochastic_da_commitment(
            da, scen, w, dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        r = solve_stochastic_da_id_dispatch(
            da, scen, w, base, realised, dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        np.testing.assert_allclose(
            r["expected_total_eur"], commit["expected_total_eur"], atol=1e-6,
        )

    def test_myopic_capped_reports_tiebreak_not_applicable(self) -> None:
        da, _, _, base, realised = self._case(seed=10)
        r = solve_myopic_capped_da_id_dispatch(
            da, base, realised, dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        assert r["success"]
        assert r["canonical_tiebreak_applied"] is None

    def test_degenerate_inputs_return_gracefully(self) -> None:
        da, scen, w, base, realised = self._case(seed=7)
        # length mismatch on realised
        assert not solve_stochastic_da_id_dispatch(
            da, scen, w, base, realised[:-1], dt=1.0,
        )["success"]
        # NaN in forecast
        bad = base.copy()
        bad[0] = np.nan
        assert not solve_stochastic_da_id_dispatch(
            da, scen, w, bad, realised, dt=1.0,
        )["success"]
        # invalid commitment (weights don't sum to 1) propagates
        assert not solve_stochastic_da_id_dispatch(
            da, scen, np.full(6, 0.1), base, realised, dt=1.0,
        )["success"]


def _force_stage0_pass_failure(
    monkeypatch: pytest.MonkeyPatch, fail_on_calls: set[int],
) -> None:
    """Force specific canonical Stage-0 passes to time out.

    Only the canonical passes carry a ``time_limit`` option (pass 1 does not),
    so counting those calls addresses pass 2a (1st) and pass 2b (2nd).
    """
    original_linprog = stochastic_dispatch.linprog
    calls = {"n": 0}

    def patched(c, *args, **kwargs):
        options = kwargs.get("options") or {}
        if "time_limit" in options:
            calls["n"] += 1
            if calls["n"] in fail_on_calls:
                return SimpleNamespace(
                    success=False, status=1, message="forced Stage-0 pass timeout",
                )
        return original_linprog(c, *args, **kwargs)

    monkeypatch.setattr(stochastic_dispatch, "linprog", patched)


class TestStochasticReserveCommitment:
    """Increment V2-A: Stage-0 reserve commitment + canonical selector.

    Pins §6-4/5/7/10 of the locked v2 contract
    (docs/design/stochastic-milp-v2-reserve.md). Fixtures keep |z*| well below
    ~1e3 so the contract's fixed 1e-6 degradation backstop stays strictly wider
    than the 1e-9·(1+|z*|) objective-bound cushion (above that scale a pass-2
    solution sitting at the allowed bound can trip the backstop — a benign
    fallback, but core fixtures must have ZERO Stage-0 fallback per §2.2).
    """

    @staticmethod
    def _dispersed_case(n: int = 8, amp: float = 40.0):
        """Flat DA forecast + two mean-centred, high-dispersion IDA scenarios.

        The myopic joint LP sees zero DA arbitrage (flat prices, VOM kills any
        churn), so it commits full reserve at any positive fee; the scenario
        set carries a large intra-day spread that makes physical headroom
        valuable — the §1.1 anti-decoupling geometry. Also maximally degenerate
        in WHICH interval carries reserve (flat premium), exercising §6-10.
        """
        da_fc = np.full(n, 50.0)
        pattern = np.where(np.arange(n) < n // 2, -1.0, 1.0) * amp
        scenarios = np.stack([da_fc + pattern, da_fc - pattern])
        weights = np.full(2, 0.5)
        return da_fc, scenarios, weights

    def test_zero_price_skip_on_missing_forecast(self) -> None:
        da = _da_shape(8)
        scen = np.tile(da, (2, 1))
        r = solve_stochastic_reserve_commitment(
            da, scen, np.full(2, 0.5), 1.0, reserve_price_forecast_eur_mw_h=None,
        )
        assert r["success"] and r["skipped"]
        np.testing.assert_array_equal(r["reserve_mw"], np.zeros(8))
        assert r["stage0_tiebreak_stable"] is True  # a skip is not a fallback
        assert np.isnan(r["expected_objective_eur"])

    def test_zero_price_skip_on_nonpositive_forecast(self) -> None:
        # Negative prices floor to 0 (the shared capacity-price sanitisation),
        # so an everywhere-nonpositive forecast is the zero-price skip: at zero
        # fee any r > 0 is only weakly dominated and a solver tie must not
        # decide whether headroom gets consumed (§2).
        da = _da_shape(8)
        scen = np.tile(da, (2, 1))
        for price in (0.0, -5.0, np.full(8, -2.0)):
            r = solve_stochastic_reserve_commitment(
                da, scen, np.full(2, 0.5), 1.0,
                reserve_price_forecast_eur_mw_h=price,
            )
            assert r["skipped"]
            np.testing.assert_array_equal(r["reserve_mw"], np.zeros(8))

    def test_skip_decided_before_scenario_validation(self) -> None:
        # §2 skip-ordering: the skip must not require any input the v1 path
        # does not need, so a no-reserve-forecast day skips even when the
        # scenario bundle would fail validation.
        da = _da_shape(8)
        r = solve_stochastic_reserve_commitment(
            da, np.full((2, 8), np.nan), np.full(2, 0.5), 1.0,
            reserve_price_forecast_eur_mw_h=None,
        )
        assert r["success"] and r["skipped"]

    def test_pure_fee_day_commits_full_reserve(self) -> None:
        # Scenarios == flat DA forecast -> zero energy value, so a positive fee
        # fills reserve to the bound in every interval (unique optimum).
        da_fc = np.full(8, 50.0)
        scen = np.tile(da_fc, (2, 1))
        r = solve_stochastic_reserve_commitment(
            da_fc, scen, np.full(2, 0.5), 1.0,
            reserve_price_forecast_eur_mw_h=10.0,
            power_mw=1.0, duration_hours=2.0,
        )
        assert r["success"] and not r["skipped"]
        np.testing.assert_allclose(r["reserve_mw"], np.ones(8), atol=1e-6)
        assert r["stage0_tiebreak_stable"] is True

    def test_collapse_objective_matches_joint_lp(self) -> None:
        # §6-5 (objective half): with every scenario == the DA forecast there
        # is no rebid opportunity, so the Stage-0 optimum equals the myopic
        # joint LP's total. KNOWN-ANSWER anchor against the INDEPENDENT
        # dispatch.py implementation (the B1 lesson: identity tests must not
        # be self-referential). Holds at any cap >= power (the r-bound
        # tightening is vacuous there); pinned at inf and at the default.
        da = _da_shape(8, seed=11)
        rprice = np.array([30.0, 30.0, 5.0, 5.0, 30.0, 30.0, 5.0, 5.0])
        joint = solve_daily_joint_capacity_lp(
            da, 1.0, rprice, power_mw=1.0, duration_hours=2.0,
        )
        for cap in (np.inf, None):
            r = solve_stochastic_reserve_commitment(
                da, np.tile(da, (3, 1)), np.full(3, 1 / 3), 1.0,
                reserve_price_forecast_eur_mw_h=rprice,
                power_mw=1.0, duration_hours=2.0, rebid_cap_mw=cap,
            )
            assert r["success"] and r["stage0_tiebreak_stable"]
            np.testing.assert_allclose(
                r["expected_objective_eur"], joint["total_revenue_eur"], atol=1e-5,
            )

    def test_collapse_elementwise_r_matches_s1(self) -> None:
        # §6-5 (element-wise half) + §6-10(b): on the collapse fixture the
        # S=N-identical and S=1 problems have IDENTICAL objectives, so the
        # canonical selector must pick the same r* element-wise. (The
        # cross-check against a selector-governed cap-constrained myopic
        # joint LP arm lands with V2-C, which owns that arm.)
        da = _da_shape(8, seed=11)
        rprice = np.array([30.0, 30.0, 5.0, 5.0, 30.0, 30.0, 5.0, 5.0])
        kw = dict(
            reserve_price_forecast_eur_mw_h=rprice, power_mw=1.0,
            duration_hours=2.0, rebid_cap_mw=np.inf,
        )
        r1 = solve_stochastic_reserve_commitment(
            da, da[None, :], np.array([1.0]), 1.0, **kw,
        )
        r3 = solve_stochastic_reserve_commitment(
            da, np.tile(da, (3, 1)), np.full(3, 1 / 3), 1.0, **kw,
        )
        assert r1["stage0_tiebreak_stable"] and r3["stage0_tiebreak_stable"]
        np.testing.assert_allclose(r1["reserve_mw"], r3["reserve_mw"], atol=1e-8)

    def test_identical_scenarios_match_s1_on_degenerate_fixture(self) -> None:
        # §6-10(a): an S=N run whose scenarios are all IDENTICAL to the base
        # forecast selects the same r* element-wise as the S=1 run — identical
        # objectives, one optimal set, one canonical member. NOT a claim about
        # genuinely different scenario sets (reserve never decouples, §1.1).
        da_fc, scenarios, _ = self._dispersed_case()
        base = scenarios.mean(axis=0)  # == da_fc (mean-centred)
        kw = dict(
            reserve_price_forecast_eur_mw_h=15.0, power_mw=1.0,
            duration_hours=2.0, rebid_cap_mw=np.inf,
        )
        r1 = solve_stochastic_reserve_commitment(
            da_fc, base[None, :], np.array([1.0]), 1.0, **kw,
        )
        r5 = solve_stochastic_reserve_commitment(
            da_fc, np.tile(base, (5, 1)), np.full(5, 0.2), 1.0, **kw,
        )
        assert r1["stage0_tiebreak_stable"] and r5["stage0_tiebreak_stable"]
        np.testing.assert_allclose(r1["reserve_mw"], r5["reserve_mw"], atol=1e-8)

    def test_anti_decoupling_expectation(self) -> None:
        # §6-4 (in-model expectation test, §1.1): at rebid_cap = inf the
        # scenario-aware Stage 0 commits a DIFFERENT r* than the myopic joint
        # LP and achieves a strictly higher Stage-0 EXPECTED objective — the
        # mathematical reason v2 exists (reserve consumes PHYSICAL headroom,
        # so there is no decoupling analog). The expected objective at the
        # myopic r is evaluated through the exogenous-reserve B1 solver plus
        # the fee constant (a cross-builder evaluation, not a re-run of the
        # Stage-0 form).
        da_fc, scenarios, weights = self._dispersed_case()
        rprice = 15.0
        stoch = solve_stochastic_reserve_commitment(
            da_fc, scenarios, weights, 1.0,
            reserve_price_forecast_eur_mw_h=rprice,
            power_mw=1.0, duration_hours=2.0, rebid_cap_mw=np.inf,
        )
        myopic = solve_daily_joint_capacity_lp(
            da_fc, 1.0, rprice, power_mw=1.0, duration_hours=2.0,
        )
        # Flat DA + VOM -> the myopic arm sees zero arbitrage and fills reserve.
        np.testing.assert_allclose(myopic["reserve_mw"], np.ones(8), atol=1e-6)
        assert np.abs(stoch["reserve_mw"] - myopic["reserve_mw"]).max() > 0.1
        fee_myopic = float((rprice * 0.95 * myopic["reserve_mw"]).sum())
        b1_at_myopic_r = solve_stochastic_da_commitment(
            da_fc, scenarios, weights, 1.0, power_mw=1.0, duration_hours=2.0,
            rebid_cap_mw=np.inf, reserve_mw=myopic["reserve_mw"],
        )
        expected_at_myopic_r = fee_myopic + b1_at_myopic_r["expected_total_eur"]
        assert stoch["expected_objective_eur"] > expected_at_myopic_r + 10.0

    def test_stage1_prime_financial_known_answer(self) -> None:
        # Codex math-audit catch (PR #46): every other Stage-0 fixture has
        # da_forecast == mean_ida, so a sign flip in the reduced Stage-1' term
        # could slip through — and a symmetric fixture cannot catch it even at
        # the objective level (the reversed position earns the mirrored value).
        # ABSOLUTE hand-computed ground truth with the symmetry broken by
        # soc_init: at soc_init_frac=0.9 the correct charge-cheap-then-
        # discharge-dear plan is capped by the tiny SoC headroom
        # (ch0 = 0.1/sqrt(eff)), while the flipped plan could discharge 0.9
        # first — so the values differ by ~9x and the pin is one-sided.
        #   stage1' = 40·ch0·(1+eff);  fee = price·0.95·2·dt;  scenarios flat
        #   (Stage-2 idle under VOM), cap=inf (no coupling), r* = [1, 1].
        eff = 0.88
        ch0 = 0.1 / math.sqrt(eff)
        expected = 40.0 * ch0 * (1.0 + eff) + 10.0 * 0.95 * 2.0
        r = solve_stochastic_reserve_commitment(
            np.array([10.0, 90.0]), np.array([[50.0, 50.0]]), np.array([1.0]),
            1.0, reserve_price_forecast_eur_mw_h=10.0, power_mw=1.0,
            duration_hours=1.0, efficiency=eff, soc_init_frac=0.9,
            rebid_cap_mw=np.inf,
        )
        assert r["success"]
        np.testing.assert_allclose(r["reserve_mw"], np.ones(2), atol=1e-6)
        np.testing.assert_allclose(
            r["expected_objective_eur"], expected, atol=1e-4,
        )

    def test_objective_at_r_star_matches_fee_plus_b1_finite_cap(self) -> None:
        # Codex math-audit catch (PR #46): the infinite-cap cross-builder
        # identity never exercises the Stage-0 rebid-coupling rows. Same
        # identity at a FINITE cap with a nonzero DA-vs-base premium, so the
        # coupling block and the Stage-1' term are both live in the Stage-0
        # matrix being cross-checked against B1's independent assembly.
        n = 8
        da_fc = 50.0 + np.linspace(-10.0, 10.0, n)
        base = da_fc + 5.0
        pattern = np.where(np.arange(n) < n // 2, -1.0, 1.0) * 40.0
        scenarios = np.stack([base + pattern, base - pattern])
        weights = np.full(2, 0.5)
        rprice = 15.0
        cap = 0.6
        stoch = solve_stochastic_reserve_commitment(
            da_fc, scenarios, weights, 1.0,
            reserve_price_forecast_eur_mw_h=rprice,
            power_mw=1.0, duration_hours=2.0, rebid_cap_mw=cap,
        )
        assert stoch["success"]
        assert stoch["reserve_mw"].max() <= cap + 1e-9
        fee = float((rprice * 0.95 * stoch["reserve_mw"]).sum())
        b1 = solve_stochastic_da_commitment(
            da_fc, scenarios, weights, 1.0, power_mw=1.0, duration_hours=2.0,
            rebid_cap_mw=cap, reserve_mw=stoch["reserve_mw"],
        )
        np.testing.assert_allclose(
            stoch["expected_objective_eur"], fee + b1["expected_total_eur"],
            atol=1e-4,
        )

    def test_objective_at_r_star_matches_fee_plus_b1(self) -> None:
        # Cross-builder consistency: the Stage-0 extensive form at a FIXED r is
        # exactly the exogenous-reserve B1 problem plus the fee constant (the
        # additive headroom row collapses to B1's power-cap bounds when r is
        # constant), so the reported diagnostic objective must equal
        # fee(r*) + B1(reserve_mw=r*). Guards the sparse assembly end-to-end.
        da_fc, scenarios, weights = self._dispersed_case()
        rprice = 15.0
        stoch = solve_stochastic_reserve_commitment(
            da_fc, scenarios, weights, 1.0,
            reserve_price_forecast_eur_mw_h=rprice,
            power_mw=1.0, duration_hours=2.0, rebid_cap_mw=np.inf,
        )
        fee = float((rprice * 0.95 * stoch["reserve_mw"]).sum())
        b1 = solve_stochastic_da_commitment(
            da_fc, scenarios, weights, 1.0, power_mw=1.0, duration_hours=2.0,
            rebid_cap_mw=np.inf, reserve_mw=stoch["reserve_mw"],
        )
        np.testing.assert_allclose(
            stoch["expected_objective_eur"], fee + b1["expected_total_eur"],
            atol=1e-4,
        )

    def test_domain_r_never_exceeds_min_power_cap(self) -> None:
        # §6-7: r* <= min(power, rebid_cap) BY CONSTRUCTION (endogenous
        # bounds), even under a fee that dominates every energy trade.
        da_fc, scenarios, weights = self._dispersed_case()
        for cap, expected_ub in ((0.4, 0.4), (1.0, 1.0), (np.inf, 1.0)):
            r = solve_stochastic_reserve_commitment(
                da_fc, scenarios, weights, 1.0,
                reserve_price_forecast_eur_mw_h=50.0,
                power_mw=1.0, duration_hours=2.0, rebid_cap_mw=cap,
            )
            assert r["success"]
            assert r["reserve_mw"].max() <= expected_ub + 1e-9
            assert r["reserve_mw"].min() >= 0.0

    def test_negative_rebid_cap_raises(self) -> None:
        da_fc, scenarios, weights = self._dispersed_case()
        with pytest.raises(ValueError, match="rebid_cap_mw must be >= 0"):
            solve_stochastic_reserve_commitment(
                da_fc, scenarios, weights, 1.0,
                reserve_price_forecast_eur_mw_h=10.0, rebid_cap_mw=-0.5,
            )

    def test_reserve_price_shape_mismatch_raises(self) -> None:
        da_fc, scenarios, weights = self._dispersed_case(n=8)
        with pytest.raises(ValueError, match="scalar or length 8"):
            solve_stochastic_reserve_commitment(
                da_fc, scenarios, weights, 1.0,
                reserve_price_forecast_eur_mw_h=np.full(5, 10.0),
            )

    def test_repeated_solve_is_deterministic(self) -> None:
        # §6-10(e): fixed-scenario repeated-solve determinism on a degenerate
        # fixture (flat capacity premium over interchangeable intervals) —
        # the SAME Stage-0 problem solved twice yields the identical r*
        # element-wise. NOT seed-independence: different scenario sets are
        # different objectives the selector cannot and must not reconcile.
        da_fc, scenarios, weights = self._dispersed_case()
        kw = dict(
            reserve_price_forecast_eur_mw_h=15.0, power_mw=1.0,
            duration_hours=2.0, rebid_cap_mw=np.inf,
        )
        first = solve_stochastic_reserve_commitment(
            da_fc, scenarios, weights, 1.0, **kw,
        )
        second = solve_stochastic_reserve_commitment(
            da_fc, scenarios, weights, 1.0, **kw,
        )
        assert first["stage0_tiebreak_stable"] and second["stage0_tiebreak_stable"]
        np.testing.assert_array_equal(first["reserve_mw"], second["reserve_mw"])

    def test_forced_fallback_flags_unstable_without_changing_objective(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # §6-10(c)+(d): the tie-break never changes z*, and a canonical-pass
        # failure falls back to the complete pass-1 solution with
        # stage0_tiebreak_stable = False.
        da_fc, scenarios, weights = self._dispersed_case()
        kw = dict(
            reserve_price_forecast_eur_mw_h=15.0, power_mw=1.0,
            duration_hours=2.0, rebid_cap_mw=np.inf,
        )
        normal = solve_stochastic_reserve_commitment(
            da_fc, scenarios, weights, 1.0, **kw,
        )
        assert normal["stage0_tiebreak_stable"] is True

        _force_stage0_pass_failure(monkeypatch, {1})  # pass 2a times out
        fallback = solve_stochastic_reserve_commitment(
            da_fc, scenarios, weights, 1.0, **kw,
        )
        assert fallback["success"]
        assert fallback["stage0_tiebreak_stable"] is False
        np.testing.assert_allclose(
            fallback["expected_objective_eur"], normal["expected_objective_eur"],
            atol=1e-6,
        )

    def test_fallback_is_all_or_nothing(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # §2.2: a 2a-only result is NOT accepted. Failing only pass 2b must
        # return the SAME (pass-1) reserve vector as failing pass 2a outright
        # — no partially-canonicalised vectors.
        da_fc, scenarios, weights = self._dispersed_case()
        kw = dict(
            reserve_price_forecast_eur_mw_h=15.0, power_mw=1.0,
            duration_hours=2.0, rebid_cap_mw=np.inf,
        )
        _force_stage0_pass_failure(monkeypatch, {1})
        fail_2a = solve_stochastic_reserve_commitment(
            da_fc, scenarios, weights, 1.0, **kw,
        )
        monkeypatch.undo()
        _force_stage0_pass_failure(monkeypatch, {2})
        fail_2b = solve_stochastic_reserve_commitment(
            da_fc, scenarios, weights, 1.0, **kw,
        )
        assert fail_2a["stage0_tiebreak_stable"] is False
        assert fail_2b["stage0_tiebreak_stable"] is False
        np.testing.assert_array_equal(fail_2b["reserve_mw"], fail_2a["reserve_mw"])

    def test_degenerate_inputs_return_gracefully(self) -> None:
        da_fc, scenarios, weights = self._dispersed_case()
        price = 10.0
        # empty
        assert not solve_stochastic_reserve_commitment(
            np.array([]), np.empty((1, 0)), np.array([1.0]), 1.0,
            reserve_price_forecast_eur_mw_h=price,
        )["success"]
        # NaN in the DA forecast
        bad_da = da_fc.copy()
        bad_da[3] = np.nan
        assert not solve_stochastic_reserve_commitment(
            bad_da, scenarios, weights, 1.0,
            reserve_price_forecast_eur_mw_h=price,
        )["success"]
        # NaN in a scenario
        bad_scen = scenarios.copy()
        bad_scen[0, 0] = np.nan
        assert not solve_stochastic_reserve_commitment(
            da_fc, bad_scen, weights, 1.0,
            reserve_price_forecast_eur_mw_h=price,
        )["success"]
        # weights not summing to 1
        assert not solve_stochastic_reserve_commitment(
            da_fc, scenarios, np.array([0.3, 0.3]), 1.0,
            reserve_price_forecast_eur_mw_h=price,
        )["success"]

    def test_worst_case_15min_day_solves_within_budget(self) -> None:
        # §8 Q2: pass-1 (~4s locally) + the contract-exact free-binary
        # canonical passes under the shared 8s deadline, which THIS size can
        # legitimately exhaust (worst case ~12s total, honest fallback). The
        # loose bound guards a scaling regression without flaking on a slow CI
        # box; a deadline-driven fallback degrades only tie-stability, never
        # completion — so `stable` is deliberately not asserted.
        n = 96
        rng = np.random.default_rng(12)
        da = 50 + 25 * np.sin(np.arange(n) / n * 2 * np.pi) + rng.normal(0, 4, n)
        base = da + 3
        errs = rng.normal(0, 10, size=(10, n))
        errs -= errs.mean(axis=0, keepdims=True)
        scen = base[None, :] + errs
        rprice = np.where((np.arange(n) // 16) % 2 == 0, 12.0, 4.0)
        r = solve_stochastic_reserve_commitment(
            da, scen, np.full(10, 0.1), 0.25,
            reserve_price_forecast_eur_mw_h=rprice,
            power_mw=1.0, duration_hours=2.0, rebid_cap_mw=0.5,
        )
        assert r["success"]
        assert isinstance(r["stage0_tiebreak_stable"], bool)
        assert r["solve_seconds"] < 30.0


class TestStochasticTripleDispatch:
    """Increment V2-B: triple day wrapper + endogenous-reserve co-opt ceiling.

    Pins §6-2/§6-3 of the locked v2 contract plus the wrapper conventions
    (deadband inert, commit-under-forecast/settle-at-realised, skip-ordering,
    selector-disabled ceilings per §4).
    """

    @staticmethod
    def _case(n: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        da = 50 + 20 * np.sin(np.arange(n) / n * 2 * np.pi) + rng.normal(0, 4, n)
        da_fc = da + rng.normal(0, 3, n)   # imperfect walk-forward DA forecast
        base = da + 5
        errs = rng.normal(0, 8, size=(4, n))
        errs -= errs.mean(axis=0, keepdims=True)
        scen = base[None, :] + errs
        w = np.full(4, 0.25)
        realised = base + rng.normal(0, 6, n)
        return da, da_fc, scen, w, base, realised

    _RP_FC = np.array([30.0, 30.0, 5.0, 5.0, 30.0, 30.0, 5.0, 5.0])
    _RP_REAL = _RP_FC * 0.8   # realised fee != forecast fee (settle-at-realised)

    def _triple(self, seed: int = 0, **overrides) -> dict:
        da, da_fc, scen, w, base, realised = self._case(seed=seed)
        kw = dict(
            da_forecast=da_fc,
            reserve_price_forecast_eur_mw_h=self._RP_FC,
            reserve_price_realised_eur_mw_h=self._RP_REAL,
            power_mw=1.0, duration_hours=2.0, rebid_cap_mw=0.8,
        )
        kw.update(overrides)
        return solve_stochastic_triple_dispatch(
            da, scen, w, base, realised, 1.0, **kw,
        )

    def test_realised_at_most_coopt_ceiling_v2(self) -> None:
        # §6-2: every executed (r*, Stage-1, Stage-2) triple is feasible for
        # the endogenous-reserve perfect-foresight problem, so realised can
        # never exceed coopt_ceiling_v2. RAW inequality — deliberately not via
        # the clamped forecast_error_cost_v2_eur.
        for seed in range(6):
            t = self._triple(seed=seed)
            assert t["success"]
            assert (
                t["realised_total_eur"] <= t["coopt_ceiling_v2_eur"] + 1e-6
            ), f"seed {seed}"

    def test_ceiling_v2_dominates_v1_and_fixed_myopic(self) -> None:
        # §6-3: endogenous-r optimisation dominates any fixed feasible r —
        # >= the v1 co-opt ceiling (r = 0; STRICT here because the fee is
        # positive) and >= the fixed-r_myopic ceiling (r from the myopic joint
        # LP, feasible at cap = power where the §3 tightening is vacuous).
        da, _, _, _, _, realised = self._case(seed=1)
        kw = dict(power_mw=1.0, duration_hours=2.0, rebid_cap_mw=1.0)
        ceil_v2 = stochastic_coopt_ceiling_v2(
            da, realised, 1.0,
            reserve_price_realised_eur_mw_h=self._RP_REAL, **kw,
        )
        v1_r0 = stochastic_coopt_ceiling(da, realised, 1.0, **kw)
        assert ceil_v2 > v1_r0 + 1.0  # strict: reserve fee is real income here
        joint = solve_daily_joint_capacity_lp(
            da, 1.0, self._RP_REAL, power_mw=1.0, duration_hours=2.0,
        )
        r_myopic = joint["reserve_mw"]
        fixed_myopic = float(
            (self._RP_REAL * 0.95 * r_myopic).sum()
        ) + stochastic_coopt_ceiling(
            da, realised, 1.0, reserve_mw=r_myopic, **kw,
        )
        assert ceil_v2 >= fixed_myopic - 1e-6

    def test_ceiling_v2_equals_v1_at_zero_reserve_price(self) -> None:
        # §6-3 boundary: at a zero realised fee, r = 0 is weakly optimal and
        # the v2 ceiling collapses to the v1 energy-only co-opt ceiling.
        da, _, _, _, _, realised = self._case(seed=2)
        kw = dict(power_mw=1.0, duration_hours=2.0, rebid_cap_mw=1.0)
        ceil_v2 = stochastic_coopt_ceiling_v2(
            da, realised, 1.0, reserve_price_realised_eur_mw_h=0.0, **kw,
        )
        v1_r0 = stochastic_coopt_ceiling(da, realised, 1.0, **kw)
        np.testing.assert_allclose(ceil_v2, v1_r0, atol=1e-4)

    def test_collapse_to_v1_execution_when_no_reserve_prices(self) -> None:
        # No reserve forecast -> Stage-0 skip (r* = 0) and the wrapper settles
        # exactly like the v1 B2 dispatch run under the reserve-mode
        # conventions (always_rebid, no capacity). Element-level comparator
        # for the batch-level §6-1.2 constrained-collapse pin (V2-C).
        da, da_fc, scen, w, base, realised = self._case(seed=3)
        t = solve_stochastic_triple_dispatch(
            da, scen, w, base, realised, 1.0, da_forecast=da_fc,
            reserve_price_forecast_eur_mw_h=None,
            reserve_price_realised_eur_mw_h=None,
            power_mw=1.0, duration_hours=2.0, rebid_cap_mw=0.8,
        )
        b2 = solve_stochastic_da_id_dispatch(
            da, scen, w, base, realised, 1.0, power_mw=1.0, duration_hours=2.0,
            rebid_cap_mw=0.8, always_rebid=True,
        )
        assert t["success"] and t["stage0_skipped"]
        np.testing.assert_array_equal(t["reserve_mw"], np.zeros(8))
        assert np.isnan(t["stage0_expected_objective_eur"])
        for key in (
            "realised_total_eur", "stochastic_hold_eur", "da_only_revenue_eur",
            "capacity_revenue_eur", "expected_total_eur",
        ):
            np.testing.assert_allclose(t[key], b2[key], atol=1e-6)

    def test_capacity_settles_at_realised_price(self) -> None:
        # Commit under forecast, settle at realised (9.2b convention, §2):
        # a high forecast fee commits r* > 0, but a zero realised fee pays 0.
        t = self._triple(
            seed=4, reserve_price_forecast_eur_mw_h=20.0,
            reserve_price_realised_eur_mw_h=0.0,
        )
        assert t["reserve_mw"].max() > 0.1
        assert t["capacity_revenue_eur"] == 0.0
        # Converse: no forecast -> skip -> r* = 0 earns nothing even at a
        # positive realised fee (there is nothing committed to settle).
        t2 = self._triple(
            seed=4, reserve_price_forecast_eur_mw_h=None,
            reserve_price_realised_eur_mw_h=20.0,
        )
        assert t2["stage0_skipped"]
        assert t2["capacity_revenue_eur"] == 0.0

    def test_skip_ordering_survives_bad_stage0_inputs(self) -> None:
        # §2 skip-ordering through the wrapper: a day the v1 path can run is
        # never excluded by a Stage-0-only input (here an all-NaN DA forecast)
        # when there is no reserve forecast to act on.
        da, _, scen, w, base, realised = self._case(seed=5)
        t = solve_stochastic_triple_dispatch(
            da, scen, w, base, realised, 1.0,
            da_forecast=np.full(8, np.nan),
            reserve_price_forecast_eur_mw_h=None,
            power_mw=1.0, duration_hours=2.0, rebid_cap_mw=0.8,
        )
        assert t["success"] and t["stage0_skipped"]
        # ...but with a live reserve forecast the bad Stage-0 input is fatal.
        t2 = solve_stochastic_triple_dispatch(
            da, scen, w, base, realised, 1.0,
            da_forecast=np.full(8, np.nan),
            reserve_price_forecast_eur_mw_h=10.0,
            power_mw=1.0, duration_hours=2.0, rebid_cap_mw=0.8,
        )
        assert not t2["success"]

    def test_reserve_respects_cap_through_execution(self) -> None:
        # §6-7 threading: r* <= min(power, cap) by construction, so the B2
        # execution layer's cap >= max reserve domain check can never raise
        # from inside the wrapper.
        t = self._triple(seed=0, rebid_cap_mw=0.5)
        assert t["success"]
        assert t["reserve_mw"].max() <= 0.5 + 1e-9
        assert t["realised_total_eur"] <= t["coopt_ceiling_v2_eur"] + 1e-6

    def test_forecast_error_cost_v2_is_clamped_gap(self) -> None:
        t = self._triple(seed=1)
        np.testing.assert_allclose(
            t["forecast_error_cost_v2_eur"],
            max(t["coopt_ceiling_v2_eur"] - t["realised_total_eur"], 0.0),
            atol=1e-6,
        )

    def test_stage0_diagnostic_passthrough(self) -> None:
        t = self._triple(seed=2)
        assert np.isfinite(t["stage0_expected_objective_eur"])
        assert t["stage0_tiebreak_stable"] is True

    def test_negative_rebid_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="rebid_cap_mw must be >= 0"):
            self._triple(seed=0, rebid_cap_mw=-0.5)

    def test_degenerate_inputs_return_gracefully(self) -> None:
        da, da_fc, scen, w, base, realised = self._case(seed=6)
        # weights not summing to 1 -> Stage-0 fails -> empty triple with keys
        bad = solve_stochastic_triple_dispatch(
            da, scen, np.full(4, 0.1), base, realised, 1.0, da_forecast=da_fc,
            reserve_price_forecast_eur_mw_h=10.0,
        )
        assert not bad["success"]
        assert "coopt_ceiling_v2_eur" in bad and "stage0_skipped" in bad
        # realised-IDA length mismatch -> execution fails -> empty triple
        bad2 = solve_stochastic_triple_dispatch(
            da, scen, w, base, realised[:-1], 1.0, da_forecast=da_fc,
            reserve_price_forecast_eur_mw_h=10.0,
        )
        assert not bad2["success"]

    def test_ceiling_v2_is_selector_disabled(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # §4: the ceiling solve is objective-only — neither the Stage-0 nor
        # the Stage-1 canonical selector may run (canonical passes are
        # objective-preserving, so running one is pure waste). Also covers the
        # r4 note that v1's stochastic_coopt_ceiling must stop wasting a
        # canonical pass.
        da, _, _, _, _, realised = self._case(seed=0)
        calls = _count_canonical_passes(monkeypatch)
        stochastic_coopt_ceiling_v2(
            da, realised, 1.0, reserve_price_realised_eur_mw_h=self._RP_REAL,
            power_mw=1.0, duration_hours=2.0,
        )
        assert calls["n"] == 0
        stochastic_coopt_ceiling(
            da, realised, 1.0, power_mw=1.0, duration_hours=2.0,
        )
        assert calls["n"] == 0
