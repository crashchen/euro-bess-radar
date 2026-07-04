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
