"""Independent Step 3C average probes; run with either checkout on PYTHONPATH."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.analytics import calculate_average_price


def frame(index, values):
    return pd.DataFrame({"price_eur_mwh": values}, index=index)


rng = np.random.default_rng(761193)
cutover = pd.Timestamp("2025-09-30T22:00:00Z")
worst_error = 0.0
for trial in range(100):
    before_count = int(rng.integers(2, 200))
    after_count = int(rng.integers(2, 400))
    before = pd.date_range(end=cutover, periods=before_count + 1, freq="h")[:-1]
    after = pd.date_range(cutover, periods=after_count, freq="15min")
    values = rng.normal(20, 200, before_count + after_count)
    bad = rng.choice(len(values), size=len(values) // 6, replace=False)
    values[bad] = rng.choice([np.nan, np.inf, -np.inf], len(bad))
    sample = frame(before.append(after), values)
    original = sample.copy(deep=True)
    answer = calculate_average_price(sample)
    expanded = np.r_[np.repeat(values[:before_count], 4), values[before_count:]]
    finite = expanded[np.isfinite(expanded)]
    expected = float(finite.mean())
    error = abs(answer["avg_price_eur_mwh"] - expected)
    worst_error = max(worst_error, error)
    assert error < 1e-10
    assert answer["covered_hours"] == len(finite) / 4
    assert answer["delivery_hours"] == len(expanded) / 4
    pd.testing.assert_frame_equal(sample, original)

before = pd.date_range(cutover - pd.Timedelta(hours=6), periods=6, freq="h")
after = pd.date_range(cutover, periods=12, freq="15min")
index = before.append(after)
gaps = {}
for position in (2, 5, 6, 9):
    sample = frame(index.delete(position), np.full(len(index) - 1, 42.0))
    answer = calculate_average_price(sample)
    assert np.isnan(answer["avg_price_eur_mwh"])
    gaps[str(position)] = answer["avg_price_reason"]

defects = {
    "2h_uniform": frame(pd.date_range("2026-01-01", periods=3, freq="2h", tz="UTC"), [10., 100., 10.]),
    "daily_uniform": frame(pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"), [10., 100., 10.]),
    "range_index": frame(pd.RangeIndex(3), [10., 100., 10.]),
    "nat_only": frame(pd.DatetimeIndex([pd.NaT, pd.NaT]), [-5., 10.]),
    "duplicate": frame(pd.DatetimeIndex(["2026-01-01T00:00Z", "2026-01-01T00:00Z"]), [-5., 10.]),
    "all_nonfinite": frame(pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"), [np.nan, np.inf, -np.inf]),
    "missing_price_column": pd.DataFrame({"other": [1, 2]}),
}
results = {name: calculate_average_price(sample) for name, sample in defects.items()}
assert all(np.isnan(results[name]["avg_price_eur_mwh"]) for name in ("nat_only", "duplicate", "all_nonfinite", "missing_price_column"))
print(json.dumps({"random_native_vs_uniform_trials": 100, "worst_average_error": worst_error, "cutover_gap_rejection": gaps, "defects": results}, indent=2))
