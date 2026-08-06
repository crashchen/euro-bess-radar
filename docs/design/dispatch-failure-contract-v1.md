# Dispatch failure contract v1

Status: **implemented**

Implementation date: 2026-08-06

## Problem

The daily solvers historically returned zero-valued schedules on failure. Two
batch callers appended those zeros without checking solve state, making a model
failure indistinguishable from a valid zero-revenue market day.

## Daily result contract

`solve_daily_lp()` and `solve_daily_joint_capacity_lp()` always return:

- `success`: boolean;
- `status`: `optimal`, `invalid_input`, or `solver_failed`;
- `message`: diagnostic text, empty only for an optimal result.

Revenue and schedule values on a failed daily result exist only for safe API
shape compatibility. The two v1 batch consumers below must never aggregate
them.

## Batch result contract

`solve_dispatch_batch()` and `solve_joint_capacity_batch()`:

- exclude missing-price days and solver-failed days from revenue rows;
- never convert a solver failure into a €0 observation;
- expose `observed_days`, `valid_days`,
  `excluded_days_due_to_missing`,
  `excluded_days_due_to_solver_failure`, `solver_failure_details`, and
  `model_available` in `DataFrame.attrs`;
- return a typed empty frame with `model_available=False` when no valid day
  remains.

`analytics.calculate_daily_dispatch()` preserves this metadata through its
greedy/LP join, and annualisation counts non-null MILP revenue rows only. The
Market Overview and Revenue Estimation pages display solver exclusions; an
all-failure result is explicitly described as unavailable.

## Scope boundary

This v1 increment covers `solve_dispatch_batch()` and
`solve_joint_capacity_batch()`, the two paths identified in the first audit.
Composite DA+ID/reserve, continuous replay, sequential, and stochastic callers
were subsequently closed by
[`dispatch-failure-contract-v2.md`](dispatch-failure-contract-v2.md). Read v1
as the ordinary/joint batch foundation, not as the current whole-system scope.

## Required regression coverage

- a genuine flat-price optimum remains `success=True`, revenue €0;
- malformed daily input reports `invalid_input`;
- partial batch failure excludes the failed date and retains valid dates;
- all-day failure returns no revenue rows and `model_available=False`;
- ordinary and joint-capacity batch paths both carry failure metadata.
