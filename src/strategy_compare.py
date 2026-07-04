"""Side-by-side dispatch-strategy comparison for the screening dashboard.

The sequential DA+ID batch already computes three benchmarks for one window
(DA-only / forecast-driven realised / perfect-foresight ceiling). This module
reframes those totals as an *investment* comparison — annualised EUR/MW/yr and
uplift over the DA-only baseline — so a user can answer "is participating in
IDA worth it, and how much is left on the table?" at a glance.

An optional fourth row reframes a DA + reserve-capacity co-optimisation total
(from ``dispatch.solve_joint_capacity_batch``) the same way. That row is a
*different revenue stream* (reserve capacity headroom, not intraday rebidding)
and is a headroom-aware capacity estimate — it does NOT model activation
energy and is NOT additive with DA, because the joint MILP makes reserve
headroom compete with DA charge/discharge for the same power budget.

A fifth row is the cumulative DA + IDA + reserve perfect-foresight ceiling
(Phase 9.2a). An optional sixth row is the *forecast-driven realistic*
triple-joint (Phase 9.2b): the reserve-first sequential policy's window total.
The ceiling and realistic rows are scored over the 9.2b walk-forward window, so
they take their own ``triple_valid_days`` denominator and ``triple_da_baseline``
uplift baseline (which may differ from the DA/IDA rows' window when walk-forward
drops the earliest day) — keeping ``realistic <= ceiling`` and matching the
cockpit's forecast-effect gap panel exactly.

An optional seventh row is the **stochastic policy value** (stochastic MILP
Increment C2): the robust ``policy_value = stochastic_realised -
capped_myopic_realised`` from ``simulation.simulate_stochastic_da_id_batch``.
Unlike the six rows above it is a value DELTA (the realised lift of the
scenario-aware commitment over the deterministic myopic baseline at a COMMON
rebid cap), NOT a strategy revenue total — so it carries no DA-baselined uplift
(``uplift_vs_da_pct`` is ``NaN``) and the cockpit excludes it from the "revenue
by strategy" bar chart. Its tie-sensitive commitment/distribution split is a
diagnostic that belongs in the cockpit attribution subpanel, never as its own
investment row.
"""

from __future__ import annotations

import math

import pandas as pd

from src.simulation import DAYS_PER_YEAR

STRATEGY_COMPARE_COLUMNS = [
    "strategy", "window_revenue_eur", "annualized_eur_per_mw", "uplift_vs_da_pct",
]

_DA_ONLY = "DA-only"
_FORECAST = "DA + IDA1 (forecast-driven)"
_CEILING = "DA + IDA1 (perfect-foresight ceiling)"
_RESERVE_DEFAULT = "DA + reserve co-opt (headroom)"
_TRIPLE_DEFAULT = "DA + IDA1 + reserve (co-opt ceiling)"
_REALISTIC_DEFAULT = "DA + IDA1 + reserve (forecast-driven realistic)"
# Public: the cockpit (Increment D) imports this to identify the value-delta row
# and exclude it from the totals "revenue by strategy" bar chart.
STOCHASTIC_POLICY_VALUE_LABEL = "Stochastic policy value (vs capped myopic)"


def build_strategy_comparison(
    summary: dict, *, power_mw: float,
    reserve_coopt_total: float | None = None,
    reserve_label: str | None = None,
    triple_joint_total: float | None = None,
    triple_joint_label: str | None = None,
    realistic_triple_total: float | None = None,
    realistic_triple_label: str | None = None,
    triple_valid_days: int | None = None,
    triple_da_baseline: float | None = None,
    policy_value_total: float | None = None,
    policy_value_valid_days: int | None = None,
    policy_value_label: str | None = None,
) -> pd.DataFrame:
    """Reframe a sequential DA+ID batch summary as a strategy comparison.

    Args:
        summary: Output of ``simulation.simulate_sequential_da_id_batch``
            (uses ``total_da_only_eur`` / ``total_realised_eur`` /
            ``total_ceiling_eur`` / ``valid_days``).
        power_mw: BESS power rating for the per-MW annualisation.
        reserve_coopt_total: Optional window total (EUR) for a DA + reserve
            capacity co-optimisation, summed from
            ``dispatch.solve_joint_capacity_batch``. When provided and finite,
            adds a fourth row annualised/uplifted on the same window basis.
            This is a headroom-aware capacity estimate (no activation energy,
            not additive with DA — the joint MILP already trades reserve
            headroom against DA dispatch), and a *different stream* from the
            DA+IDA rows, not a cumulative ladder.
        reserve_label: Display label for the reserve row.
        triple_joint_total: Optional window total (EUR) for the cumulative
            DA + IDA + reserve perfect-foresight ceiling
            (``simulation.simulate_da_id_reserve_ceiling_batch`` or the 9.2b
            batch's ``total_global_ceiling_eur``). When provided and finite,
            adds a fifth row. Unlike the reserve row this IS a cumulative ladder
            step (it stacks IDA rebid and reserve headroom on DA in one
            co-optimised solve); same red-line — capacity headroom, no
            activation energy. A perfect-foresight upper bound.
        triple_joint_label: Display label for the triple-joint ceiling row.
        realistic_triple_total: Optional window total (EUR) for the
            *forecast-driven realistic* triple-joint (Phase 9.2b reserve-first
            sequential policy, ``simulate_sequential_da_id_reserve_batch``'s
            ``total_realised_eur``). When provided and finite, adds a sixth row.
            This is NOT a ceiling — it is the walk-forward forecast-driven
            policy, so it sits below the perfect-foresight ceiling by the
            forecast + commitment-timing gap.
        realistic_triple_label: Display label for the realistic triple row.
        triple_valid_days: Annualisation denominator for the triple ceiling and
            realistic rows. The 9.2b walk-forward window may exclude the
            earliest day, so these two rows are scored over their own day count;
            defaults to ``summary["valid_days"]`` (DA/IDA window) when omitted.
        triple_da_baseline: DA-only baseline (EUR) for the triple ceiling and
            realistic rows' uplift%, matching their walk-forward window;
            defaults to ``summary["total_da_only_eur"]`` when omitted.
        policy_value_total: Optional window total (EUR) for the stochastic
            policy value — ``simulation.simulate_stochastic_da_id_batch``'s
            ``total_policy_value_eur`` (``stochastic_realised -
            capped_myopic_realised`` at a common rebid cap). When provided and
            finite, adds a seventh row. This is a value DELTA, not a strategy
            revenue total, so its ``uplift_vs_da_pct`` is ``NaN`` (no DA
            baseline) and the cockpit excludes it from the revenue bar chart. A
            negative value is kept, not dropped (the scenario-aware commitment
            can realise below the myopic baseline on an adverse path).
        policy_value_valid_days: Annualisation denominator for the policy-value
            row — the stochastic batch's own ``valid_days``. Defaults to the
            triple window (``triple_valid_days`` when supplied, else
            ``summary["valid_days"]``) so the policy value aligns with the 9.2b
            reserve rows.
        policy_value_label: Display label for the policy-value row (defaults to
            ``STOCHASTIC_POLICY_VALUE_LABEL``).

    Returns:
        One row per strategy with columns ``[strategy, window_revenue_eur,
        annualized_eur_per_mw, uplift_vs_da_pct]``. Annualisation uses the
        same 365.25-day i.i.d. convention as the rest of the dashboard.
        Empty when the window has no valid days.
    """
    valid_days = int(summary.get("valid_days", 0) or 0)
    if valid_days <= 0:
        return pd.DataFrame(columns=STRATEGY_COMPARE_COLUMNS)

    da = float(summary.get("total_da_only_eur", 0.0))
    realised = float(summary.get("total_realised_eur", 0.0))
    ceiling = float(summary.get("total_ceiling_eur", 0.0))

    def annualized_per_mw(total: float, days: int = valid_days) -> float:
        if power_mw <= 0 or days <= 0:
            return float("nan")
        return total * DAYS_PER_YEAR / days / power_mw

    def uplift_pct(total: float, baseline: float = da) -> float:
        # Undefined when the DA-only baseline is ~0 (no meaningful denominator).
        if abs(baseline) < 1e-9:
            return float("nan")
        return (total - baseline) / baseline * 100.0

    rows = [
        (_DA_ONLY, da, annualized_per_mw(da), 0.0),
        (_FORECAST, realised, annualized_per_mw(realised), uplift_pct(realised)),
        (_CEILING, ceiling, annualized_per_mw(ceiling), uplift_pct(ceiling)),
    ]
    if reserve_coopt_total is not None and math.isfinite(float(reserve_coopt_total)):
        rc = float(reserve_coopt_total)
        rows.append(
            (reserve_label or _RESERVE_DEFAULT, rc,
             annualized_per_mw(rc), uplift_pct(rc)),
        )
    # The triple ceiling + forecast-driven realistic rows may be scored over a
    # different (9.2b walk-forward) window than the DA/IDA rows, so they take
    # their own denominator and DA baseline when supplied.
    t_days = int(triple_valid_days) if triple_valid_days else valid_days
    t_base = float(triple_da_baseline) if triple_da_baseline is not None else da
    if triple_joint_total is not None and math.isfinite(float(triple_joint_total)):
        tj = float(triple_joint_total)
        rows.append(
            (triple_joint_label or _TRIPLE_DEFAULT, tj,
             annualized_per_mw(tj, t_days), uplift_pct(tj, t_base)),
        )
    if realistic_triple_total is not None and math.isfinite(float(realistic_triple_total)):
        rt = float(realistic_triple_total)
        rows.append(
            (realistic_triple_label or _REALISTIC_DEFAULT, rt,
             annualized_per_mw(rt, t_days), uplift_pct(rt, t_base)),
        )
    # Seventh row: the stochastic policy value — a value DELTA (stochastic minus
    # capped-myopic realised at a common cap), not a strategy revenue total. It
    # has no DA-baselined uplift (NaN), annualises over the stochastic batch's
    # own valid days (defaulting to the triple window for 9.2b alignment), and a
    # negative value is preserved rather than dropped.
    if policy_value_total is not None and math.isfinite(float(policy_value_total)):
        pv = float(policy_value_total)
        pv_days = int(policy_value_valid_days) if policy_value_valid_days else t_days
        rows.append(
            (policy_value_label or STOCHASTIC_POLICY_VALUE_LABEL, pv,
             annualized_per_mw(pv, pv_days), float("nan")),
        )
    return pd.DataFrame(rows, columns=STRATEGY_COMPARE_COLUMNS)
