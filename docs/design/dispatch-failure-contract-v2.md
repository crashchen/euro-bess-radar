# Dispatch failure contract v2

Status: **implemented**

Implementation date: 2026-08-06

## Purpose

Version 2 extends the v1 daily-solver contract through every composite path
used by the Revenue Estimation and Simulation Cockpit pages. A failed nested
solve must never become a plausible zero-revenue observation, a partial
multi-stage result, or a successful stochastic policy value.

## Daily and composite result contract

The following public solvers always return `success`, `status`, and `message`
in addition to their backwards-compatible numeric and schedule keys:

- `solve_daily_da_id_dispatch()`;
- `solve_daily_da_id_reserve_dispatch()`;
- `solve_sequential_da_id_dispatch()`;
- `solve_sequential_da_id_reserve_dispatch()`;
- stochastic commitment, capped execution, and triple-policy wrappers.

`status` is `optimal`, `invalid_input`, or `solver_failed`. An optimal result
has `success=True` and an empty message. A valid flat-price optimum may have
zero revenue and remains successful. Invalid-input and solver-failed results
retain zero-shaped values only for API-shape compatibility; callers must not
aggregate them.

Every composite solver checks each nested solve before consuming its numeric
or schedule output. A nested failure is returned with a stage-qualified
diagnostic, so the caller can distinguish (for example) a DA commitment
failure from an IDA execution or reserve-ceiling failure. No later stage runs
after a required earlier stage has failed.

Public scalar ceiling helpers keep their historic float return type for
compatibility. Composite callers use an internal typed result and therefore do
not interpret a failed ceiling as a valid `0.0`.

## Batch and replay contract

Composite, sequential, replay, and stochastic batches expose:

- `observed_days`;
- `valid_days`;
- `excluded_days` (missing plus solver-failed days, retained for compatibility);
- `excluded_days_due_to_missing`;
- `excluded_days_due_to_solver_failure`;
- `solver_failure_details` (`date`, `status`, and stage-qualified `message`);
- `model_available` (`True` only when at least one valid result remains).

DataFrame-returning APIs store the fields in `DataFrame.attrs`; tuple/dict
batch APIs include them in their summary. A solver-failed day is excluded from
all revenue totals, annualisation denominators, risk pools, and common-policy
windows. A valid optimal zero-revenue day remains in all of those calculations.

For stochastic policy comparisons, all arms share one valid-day window. If any
required arm fails, that date is excluded from every arm and from the pooled
scenario distribution. Partial failures are disclosed; an all-failure window
is explicitly unavailable rather than displayed as a zero-value strategy.

## Continuous replay semantics

A continuous run is accepted only when its local days are complete, NaN-free,
and uniformly spaced in UTC. DST transition days remain valid because their
23/25 local-hour shape is complete in the configured timezone. A day missing a
leading, trailing, or interior interval is a missing-data exclusion.

If a continuous run solve fails, every date in that run is recorded as a
solver-failed exclusion. The failed run contributes no rows and its synthetic
zero schedule cannot advance SoC. A later clean run resumes from the last
successfully solved SoC.

## UI contract

Revenue Estimation and Simulation Cockpit consumers:

- warn when some days were excluded by solver failure;
- use an explicit model-unavailable error when no valid result remains;
- never fall back from an all-solver-failed realistic path to a ceiling in a
  way that hides the failure;
- omit unavailable strategy rows and stochastic attribution panels.

## Scope boundary

This increment closes the deferred v1 paths: composite DA+ID/reserve,
forecast-sequential batches, continuous replay, and stochastic wrappers. It
does not introduce solver retries, manufacture fallback schedules, model
reserve activation energy, or define the future ProjectCase / Investment Case
schema.

## Required regression coverage

- nested failures stop composite execution and preserve stage diagnostics;
- solver-failed dates are excluded while valid €0 dates remain;
- all-failure batches report `model_available=False`;
- continuous-run failures exclude the entire run without advancing SoC;
- incomplete leading/trailing/interior intervals are rejected, while DST days
  remain valid;
- stochastic three-arm comparisons use a common valid-day set and exclude
  failed dates from risk statistics;
- UI helpers do not replace an all-failure result with a valid-looking ceiling.
