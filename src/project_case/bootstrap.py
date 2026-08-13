"""The single accepted bootstrap algorithm ``pc-bootstrap-pcg64-choice365-linear-v1``.

PC-A pins the algorithm literal + a golden annual-output/percentile vector so any
RNG / sampling / percentile change is caught (§4.8, red-line #25). PC-A does **not**
build the lifecycle cash-flow or the NPV distributions (that is PC-B); this module
exposes only the deterministic annual-sum generator its golden vector locks.

The literal means: daily values ordered by ``valid_dates``; NumPy
``Generator(PCG64(seed))``; one ``choice(values, size=(n_simulations, 365),
replace=True)``; row-wise float64 sum; and ``percentile(..., method="linear")``.
This is exactly ``scenario.bootstrap_annual_revenue``'s sampler (``default_rng`` is
``Generator(PCG64(seed))``), minus that helper's non-finite drop / €0 fallback —
PC's series is already all-finite and non-empty (fail-closed, §4.6).
"""

from __future__ import annotations

import numpy as np

from src.project_case.enums import BOOTSTRAP_ALGORITHM_V1

__all__ = ["BOOTSTRAP_ALGORITHM_V1", "bootstrap_annual_sums"]

_DAYS_PER_YEAR = 365


def bootstrap_annual_sums(
    daily_values: np.ndarray,
    *,
    seed: int,
    n_simulations: int,
) -> np.ndarray:
    """Return the ``(n_simulations,)`` array of bootstrapped annual sums.

    ``daily_values`` must already be finite and ordered by ``valid_dates``.
    """
    values = np.asarray(daily_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("daily_values must be a non-empty 1-D array")
    if not np.isfinite(values).all():
        raise ValueError("daily_values must be all finite (fail-closed)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative int")
    if isinstance(n_simulations, bool) or not isinstance(n_simulations, int) or n_simulations <= 0:
        raise ValueError("n_simulations must be a positive int")
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = rng.choice(values, size=(n_simulations, _DAYS_PER_YEAR), replace=True)
    return samples.sum(axis=1, dtype=np.float64)
