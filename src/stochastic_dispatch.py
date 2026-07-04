"""Extensive-form stochastic DA commitment MILP (stochastic MILP Increment B1).

The stochastic upgrade (``docs/design/stochastic-milp-v1.md``) commits the
Stage-1 DA schedule against a *distribution* of IDA price paths instead of a
single point forecast. This module is the **commitment solver core** (B1): the
two-stage extensive-form MILP that produces the Stage-1 DA schedule and the
per-scenario Stage-2 physical schedules, plus the same-cap perfect-foresight
co-opt ceiling built by reusing this solver. The forecast-driven *realised*
execution layer (deadband, ``da_only``/``captured``/``forecast_error``
decomposition) is Increment B2.

Two stages, one MILP (§2 of the contract):

- **Stage 1** (shared across scenarios, non-anticipative): a full-power DA
  schedule ``da_charge/da_discharge`` with a binary mode per interval. It is
  SoC-feasible and terminal-neutral on its own — a committed physical position
  settled financially at DA and marked to market at each scenario's IDA.
- **Stage 2** (per scenario ``s``): a physical schedule at reserve-capped power,
  its own binary mutual exclusion, SoC window, and terminal-neutral equality.

The **load-bearing coupling** (§2.1) is a per-interval rebid volume cap
``|stage2_net - da_net| ≤ rebid_cap_mw``. Without it the pure-financial DA
accounting decouples the stages and the extensive form degenerates to a
deterministic equivalent against the scenario-mean path — pinned as the
``rebid_cap = ∞`` identity (the decoupling theorem). Binary mutual exclusion
stays in BOTH stages per scenario (negative-price degeneracy red-line); scale
is managed by the scenario cap, never by relaxing integrality.

Objective: ``max Σ_s w_s · total_s`` where
``total_s = da_gross - implicit_mtm_s + stage2_value_s`` and

- ``da_gross = Σ_t (da_dis - da_ch)·DA·dt`` (financial, no VOM),
- ``implicit_mtm_s = Σ_t (da_dis - da_ch)·IDA_s·dt``,
- ``stage2_value_s = Σ_t [(s2_dis - s2_ch)·IDA_s - VOM·(s2_dis + s2_ch)]·dt``.

Because ``Σ_s w_s = 1`` and the scenarios are mean-centred to the base forecast
(Increment A), the Stage-1 objective reduces to ``da_net·(DA - base)`` — the
exact deterministic co-optimisation the decoupling theorem predicts.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from src.dispatch import DISPATCH_VOM_COST_EUR_MWH

logger = logging.getLogger(__name__)


def _empty_commitment_result(n: int, s: int) -> dict:
    """Degenerate return for empty / invalid inputs (mirrors dispatch.py)."""
    zeros_n = np.zeros(max(n, 0))
    return {
        "success": False,
        "expected_total_eur": 0.0,
        "da_p_charge": zeros_n, "da_p_discharge": zeros_n,
        "da_soc": np.zeros(max(n, 0) + 1),
        "scenario_p_charge": np.zeros((max(s, 0), max(n, 0))),
        "scenario_p_discharge": np.zeros((max(s, 0), max(n, 0))),
        "scenario_total_eur": np.zeros(max(s, 0)),
        "reserve_mw": zeros_n,
        "rebid_cap_mw": float("inf"),
        "solve_seconds": 0.0,
    }


def _coerce_reserve(reserve_mw: float | np.ndarray | None, n: int, power_mw: float) -> np.ndarray:
    """Return a non-negative per-interval reserve vector clipped to [0, power]."""
    if reserve_mw is None:
        return np.zeros(n)
    arr = np.asarray(reserve_mw, dtype=float).ravel()
    if arr.size == 1:
        arr = np.full(n, float(arr[0]))
    elif arr.size != n:
        raise ValueError(f"reserve_mw must be scalar or length {n}, got {arr.size}")
    return np.clip(np.where(np.isfinite(arr), arr, 0.0), 0.0, power_mw)


def _append_battery_block(
    trip: list, rhs: list, row0: int, base: int, n: int, dt: float,
    sqrt_eff: float, cap_mwh: float, soc_init: float, power_cap: np.ndarray,
) -> int:
    """Append mutual-exclusion + SoC-window inequality rows for one battery block.

    ``trip`` collects ``(row, col, value)`` triplets for the shared A_ub matrix.
    The block's variables are ``charge[base:base+n]``, ``discharge[base+n:base+2n]``,
    ``mode b[base+2n:base+3n]``. ``power_cap`` is the per-interval MW cap (the
    big-M in the mutex rows and the SoC dynamics both use it). Returns the next
    free A_ub row index. The caller adds the terminal-neutral equality separately.
    """
    ch, dis, bb = base, base + n, base + 2 * n
    row = row0
    for t in range(n):
        cap_t = power_cap[t]
        trip.append((row, ch + t, 1.0))            # ch - cap*b <= 0
        trip.append((row, bb + t, -cap_t))
        rhs.append(0.0)
        row += 1
        trip.append((row, dis + t, 1.0))           # dis + cap*b <= cap
        trip.append((row, bb + t, cap_t))
        rhs.append(cap_t)
        row += 1
    for t in range(1, n + 1):
        for i in range(t):
            trip.append((row, ch + i, sqrt_eff * dt))       # SoC upper
            trip.append((row, dis + i, -dt / sqrt_eff))
        rhs.append(cap_mwh - soc_init)
        row += 1
        for i in range(t):
            trip.append((row, ch + i, -sqrt_eff * dt))      # SoC lower
            trip.append((row, dis + i, dt / sqrt_eff))
        rhs.append(soc_init)
        row += 1
    return row


def solve_stochastic_da_commitment(
    da_prices: np.ndarray,
    scenarios: np.ndarray,
    weights: np.ndarray,
    dt: float,
    *,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    soc_init_frac: float = 0.5,
    rebid_cap_mw: float | None = None,
    reserve_mw: float | np.ndarray | None = None,
) -> dict:
    """Solve the two-stage extensive-form DA-commitment MILP for one day.

    Args:
        da_prices: ``(N,)`` DA prices (EUR/MWh).
        scenarios: ``(S, N)`` IDA price scenario paths (from
            ``ida_scenarios.build_ida_scenarios``).
        weights: ``(S,)`` scenario weights (summing to 1).
        dt: Interval duration in hours.
        power_mw, duration_hours, efficiency, soc_init_frac: BESS parameters.
        rebid_cap_mw: Per-interval rebid volume cap ``|stage2_net - da_net|``.
            ``None`` defaults to ``power_mw``. ``inf`` (or ``>= 2*power_mw``)
            removes the coupling, decoupling the stages (decoupling theorem).
        reserve_mw: Per-interval reserve headroom the Stage-2 physical schedule
            must leave free. Requires ``rebid_cap_mw >= max(reserve_mw)`` (the
            feasibility domain, §5) else raises ``ValueError``.

    Returns:
        Dict with ``expected_total_eur`` (the objective), Stage-1 arrays
        (``da_p_charge/da_p_discharge/da_soc``), per-scenario Stage-2 arrays
        (``scenario_p_charge/scenario_p_discharge`` of shape ``(S, N)``),
        ``scenario_total_eur`` (``total_s`` per scenario at its own IDA path),
        ``reserve_mw``, ``rebid_cap_mw``, ``solve_seconds``, and ``success``.
    """
    da_prices = np.asarray(da_prices, dtype=float).ravel()
    scenarios = np.atleast_2d(np.asarray(scenarios, dtype=float))
    weights = np.asarray(weights, dtype=float).ravel()
    n = da_prices.size
    s = scenarios.shape[0]
    if (
        n == 0 or scenarios.shape[1] != n or weights.size != s
        or np.isnan(da_prices).any() or np.isnan(scenarios).any()
        or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-9)
    ):
        return _empty_commitment_result(n, s)

    reserve = _coerce_reserve(reserve_mw, n, power_mw)
    cap = power_mw if rebid_cap_mw is None else float(rebid_cap_mw)
    if reserve.max(initial=0.0) > cap + 1e-9:
        raise ValueError(
            "rebid_cap_mw must be >= max reserve on reserve days "
            f"(cap={cap}, max_reserve={reserve.max():.4f}); below this domain the "
            "capped myopic baseline can have no feasible recourse (scope §5)."
        )

    return _solve_and_unpack(
        da_prices, scenarios, weights, dt, n, s, power_mw, duration_hours,
        efficiency, soc_init_frac, cap, reserve,
    )


def _solve_and_unpack(
    da_prices, scenarios, weights, dt, n, s, power_mw, duration_hours,
    efficiency, soc_init_frac, cap, reserve,
) -> dict:
    """Assemble the sparse MILP, solve it, and unpack the schedules + cash."""
    capacity_mwh = power_mw * duration_hours
    soc_init = soc_init_frac * capacity_mwh
    sqrt_eff = math.sqrt(efficiency)
    block = 3 * n
    n_vars = block * (1 + s)
    da_cap = np.full(n, power_mw)          # Stage-1 uses full power (financial)
    s2_cap = np.clip(power_mw - reserve, 0.0, power_mw)
    couple = np.isfinite(cap) and cap < 2 * power_mw - 1e-12

    c = np.zeros(n_vars)
    mean_ida = weights @ scenarios         # (N,) — == base forecast (mean-centred)
    c[:n] = (da_prices - mean_ida) * dt            # da_charge
    c[n:2 * n] = -(da_prices - mean_ida) * dt      # da_discharge
    for j in range(s):
        o = block * (1 + j)
        c[o:o + n] = weights[j] * (scenarios[j] + DISPATCH_VOM_COST_EUR_MWH) * dt
        c[o + n:o + 2 * n] = -weights[j] * (scenarios[j] - DISPATCH_VOM_COST_EUR_MWH) * dt

    trip: list[tuple[int, int, float]] = []
    rhs: list[float] = []
    row = _append_battery_block(
        trip, rhs, 0, 0, n, dt, sqrt_eff, capacity_mwh, soc_init, da_cap,
    )
    for j in range(s):
        row = _append_battery_block(
            trip, rhs, row, block * (1 + j), n, dt, sqrt_eff,
            capacity_mwh, soc_init, s2_cap,
        )
        if couple:
            row = _append_rebid_coupling(trip, rhs, row, block * (1 + j), n, cap)

    ub_r, ub_c, ub_v = zip(*trip, strict=True)
    a_ub = csr_matrix((ub_v, (ub_r, ub_c)), shape=(len(rhs), n_vars))
    a_eq, b_eq = _terminal_equalities(n, s, block, dt, sqrt_eff)
    bounds = _variable_bounds(n, s, block, da_cap, s2_cap)
    integrality = _binary_mask(n, s, block)

    t0 = time.perf_counter()
    result = linprog(
        c, A_ub=a_ub, b_ub=np.array(rhs), A_eq=a_eq, b_eq=b_eq,
        bounds=bounds, integrality=integrality, method="highs",
    )
    solve_seconds = time.perf_counter() - t0
    if not result.success:
        logger.warning("stochastic commitment MILP failed: %s", result.message)
        out = _empty_commitment_result(n, s)
        out["solve_seconds"] = solve_seconds
        return out

    return _unpack_solution(
        result.x, da_prices, scenarios, weights, dt, n, s, block, sqrt_eff,
        soc_init, cap, reserve, solve_seconds,
    )


def _append_rebid_coupling(
    trip: list, rhs: list, row0: int, base: int, n: int, cap: float,
) -> int:
    """Append ``|stage2_net - da_net| <= cap`` as two inequality rows per interval."""
    ch, dis = base, base + n
    row = row0
    for t in range(n):
        # (s2_dis - s2_ch) - (da_dis - da_ch) <= cap
        trip.append((row, dis + t, 1.0))
        trip.append((row, ch + t, -1.0))
        trip.append((row, n + t, -1.0))
        trip.append((row, t, 1.0))
        rhs.append(cap)
        row += 1
        # (da_dis - da_ch) - (s2_dis - s2_ch) <= cap
        trip.append((row, n + t, 1.0))
        trip.append((row, t, -1.0))
        trip.append((row, dis + t, -1.0))
        trip.append((row, ch + t, 1.0))
        rhs.append(cap)
        row += 1
    return row


def _terminal_equalities(n, s, block, dt, sqrt_eff):
    """Terminal-neutral equality (final SoC == initial) for every battery block."""
    trip: list[tuple[int, int, float]] = []
    for j in range(1 + s):
        base = block * j
        for i in range(n):
            trip.append((j, base + i, sqrt_eff * dt))
            trip.append((j, base + n + i, -dt / sqrt_eff))
    eq_r, eq_c, eq_v = zip(*trip, strict=True)
    a_eq = csr_matrix((eq_v, (eq_r, eq_c)), shape=(1 + s, block * (1 + s)))
    return a_eq, np.zeros(1 + s)


def _variable_bounds(n, s, block, da_cap, s2_cap):
    """Per-variable bounds: charge/discharge in [0, cap], mode b in [0, 1]."""
    bounds = (
        [(0.0, float(c)) for c in da_cap] * 2 + [(0.0, 1.0)] * n
    )
    for _ in range(s):
        bounds += [(0.0, float(c)) for c in s2_cap] * 2 + [(0.0, 1.0)] * n
    return bounds


def _binary_mask(n, s, block):
    """Integrality mask: 1 for the mode variables, 0 for charge/discharge."""
    integrality = np.zeros(block * (1 + s))
    for j in range(1 + s):
        integrality[block * j + 2 * n:block * j + 3 * n] = 1
    return integrality


def _reconstruct_soc(charge, discharge, dt, sqrt_eff, soc_init):
    """SoC trajectory from charge/discharge flows (same recursion as dispatch.py)."""
    soc = np.zeros(len(charge) + 1)
    soc[0] = soc_init
    for t in range(len(charge)):
        soc[t + 1] = soc[t] + (charge[t] * sqrt_eff - discharge[t] / sqrt_eff) * dt
    return soc


def _unpack_solution(
    x, da_prices, scenarios, weights, dt, n, s, block, sqrt_eff, soc_init,
    cap, reserve, solve_seconds,
) -> dict:
    """Slice the solution vector into schedules and recompute per-scenario cash."""
    da_ch = x[:n]
    da_dis = x[n:2 * n]
    da_net = da_dis - da_ch
    da_gross = float((da_net * da_prices * dt).sum())

    s2_ch = np.zeros((s, n))
    s2_dis = np.zeros((s, n))
    scenario_total = np.zeros(s)
    vom = DISPATCH_VOM_COST_EUR_MWH
    for j in range(s):
        o = block * (1 + j)
        s2_ch[j] = x[o:o + n]
        s2_dis[j] = x[o + n:o + 2 * n]
        s2_net = s2_dis[j] - s2_ch[j]
        implicit_mtm = float((da_net * scenarios[j] * dt).sum())
        stage2_value = float(
            (s2_net * scenarios[j] - vom * (s2_dis[j] + s2_ch[j])).sum() * dt
        )
        scenario_total[j] = da_gross - implicit_mtm + stage2_value

    return {
        "success": True,
        "expected_total_eur": round(float(weights @ scenario_total), 6),
        "da_p_charge": da_ch, "da_p_discharge": da_dis,
        "da_soc": _reconstruct_soc(da_ch, da_dis, dt, sqrt_eff, soc_init),
        "scenario_p_charge": s2_ch, "scenario_p_discharge": s2_dis,
        "scenario_total_eur": scenario_total,
        "reserve_mw": reserve,
        "rebid_cap_mw": cap,
        "solve_seconds": round(solve_seconds, 4),
    }


def stochastic_coopt_ceiling(
    da_prices: np.ndarray,
    realised_ida: np.ndarray,
    dt: float,
    *,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    efficiency: float = 0.88,
    soc_init_frac: float = 0.5,
    rebid_cap_mw: float | None = None,
    reserve_mw: float | np.ndarray | None = None,
) -> float:
    """Same-cap perfect-foresight co-opt ceiling (§2.4), the binding upper bound.

    Reuses :func:`solve_stochastic_da_commitment` with a SINGLE scenario equal to
    the realised IDA path, so Stage 1 and Stage 2 both optimise against the
    realised prices under the SAME ``rebid_cap_mw`` and reserve headroom. Every
    executed ``(Stage-1, Stage-2)`` pair of a capped policy is feasible for this
    problem, so it upper-bounds the realised value by construction (the legacy
    9.2a ceiling, whose Stage 1 is myopic DA-only, is NOT this bound).
    """
    realised = np.asarray(realised_ida, dtype=float).ravel()
    res = solve_stochastic_da_commitment(
        da_prices, realised[None, :], np.array([1.0]), dt,
        power_mw=power_mw, duration_hours=duration_hours, efficiency=efficiency,
        soc_init_frac=soc_init_frac, rebid_cap_mw=rebid_cap_mw, reserve_mw=reserve_mw,
    )
    return res["expected_total_eur"]
