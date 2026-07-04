"""Tests for the stochastic DA-commitment MILP core (Increment B1).

These pin the load-bearing identities the four-round scope review locked
(docs/design/stochastic-milp-v1.md): the decoupling theorem, the IDA==DA
collapse, the same-cap co-opt ceiling construction, the rebid-cap coupling,
and the reserve feasibility domain.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.dispatch import solve_daily_lp
from src.stochastic_dispatch import (
    solve_stochastic_da_commitment,
    solve_stochastic_da_id_dispatch,
    stochastic_coopt_ceiling,
)


def _mean_centred_scenarios(base: np.ndarray, s: int, sigma: float, seed: int):
    """S scenarios = base + mean-centred Gaussian error paths (Increment A shape)."""
    rng = np.random.default_rng(seed)
    errs = rng.normal(0, sigma, size=(s, base.size))
    errs -= errs.mean(axis=0, keepdims=True)
    return base[None, :] + errs, np.full(s, 1.0 / s)


def _da_shape(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 50 + 25 * np.sin(np.arange(n) / n * 2 * np.pi) + rng.normal(0, 4, n)


class TestStochasticCommitment:
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
        # against DA-only; it must NOT be clamped to zero.
        da, scen, w, base, realised = self._case(seed=0)
        r = solve_stochastic_da_id_dispatch(
            da, scen, w, base, realised, dt=1.0, power_mw=1.0, duration_hours=2.0,
        )
        assert r["captured_uplift_eur"] < 0  # this synthetic day misleads

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
