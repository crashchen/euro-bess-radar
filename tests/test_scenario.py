"""Tests for Monte Carlo scenario analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scenario import (
    bootstrap_annual_revenue,
    calculate_npv_distribution,
    sensitivity_table,
)


class TestBootstrapAnnualRevenue:
    def test_output_shape(self) -> None:
        daily = pd.Series([100.0] * 30)
        result = bootstrap_annual_revenue(daily, n_simulations=1000)
        assert result["simulations"].shape == (1000,)

    def test_percentile_ordering(self) -> None:
        rng = np.random.default_rng(0)
        daily = pd.Series(rng.normal(200, 50, 365))
        result = bootstrap_annual_revenue(daily)
        assert result["p10"] <= result["p25"] <= result["p50"]
        assert result["p50"] <= result["p75"] <= result["p90"]

    def test_constant_daily_revenue(self) -> None:
        daily = pd.Series([100.0] * 365)
        result = bootstrap_annual_revenue(daily, n_simulations=100)
        assert result["p50"] == pytest.approx(365 * 100, rel=1e-6)
        assert result["std"] == pytest.approx(0.0, abs=1e-6)

    def test_empty_input(self) -> None:
        result = bootstrap_annual_revenue(pd.Series([], dtype=float))
        assert result["mean"] == 0.0
        assert result["p50"] == 0.0

    def test_nan_values_ignored(self) -> None:
        daily = pd.Series([100.0, np.nan, 100.0] * 50)
        result = bootstrap_annual_revenue(daily, n_simulations=100)
        assert result["p50"] == pytest.approx(365 * 100, rel=0.05)

    def test_reproducible_with_seed(self) -> None:
        daily = pd.Series(np.random.default_rng(0).normal(200, 50, 365))
        r1 = bootstrap_annual_revenue(daily, seed=42)
        r2 = bootstrap_annual_revenue(daily, seed=42)
        assert r1["p50"] == r2["p50"]


class TestNpvDistribution:
    def test_positive_npv_with_good_revenue(self) -> None:
        # Revenue = 100k/yr, capex = 500k, life = 20yr, rate = 8%
        # PV factor ~9.82 → NPV ~9.82*100k - 500k > 0
        annual_rev = np.full(1000, 100000.0)
        result = calculate_npv_distribution(
            annual_rev, total_capex=500000.0,
            effective_life_years=20, discount_rate=0.08,
        )
        assert result["npv_p50"] > 0
        assert result["prob_positive_npv"] == 1.0

    def test_negative_npv_with_low_revenue(self) -> None:
        annual_rev = np.full(1000, 10000.0)
        result = calculate_npv_distribution(
            annual_rev, total_capex=500000.0,
            effective_life_years=10, discount_rate=0.10,
        )
        assert result["npv_p50"] < 0
        assert result["prob_positive_npv"] == 0.0

    def test_degradation_reduces_npv(self) -> None:
        annual_rev = np.full(100, 100000.0)
        r_no_deg = calculate_npv_distribution(annual_rev, total_capex=500000.0)
        r_deg = calculate_npv_distribution(
            annual_rev, total_capex=500000.0, annual_degradation_cost=20000.0,
        )
        assert r_deg["npv_p50"] < r_no_deg["npv_p50"]

    def test_zero_capex_equals_pv_of_revenue(self) -> None:
        annual_rev = np.full(100, 50000.0)
        result = calculate_npv_distribution(
            annual_rev, total_capex=0.0,
            effective_life_years=10, discount_rate=0.08,
        )
        assert result["npv_p50"] > 0
        assert result["prob_positive_npv"] == 1.0


class TestSensitivityTable:
    def test_expected_shape(self) -> None:
        result = sensitivity_table(
            base_revenue=100000.0, total_capex=500000.0,
        )
        # 4 params x 3 values each = 12 rows
        assert len(result) == 12
        assert set(result.columns) == {"param", "value", "npv"}

    def test_higher_revenue_higher_npv(self) -> None:
        result = sensitivity_table(
            base_revenue=100000.0, total_capex=500000.0,
        )
        rev_rows = result[result["param"] == "revenue"].sort_values("value")
        npvs = rev_rows["npv"].tolist()
        assert npvs[0] < npvs[1] < npvs[2]

    def test_higher_capex_lower_npv(self) -> None:
        result = sensitivity_table(
            base_revenue=100000.0, total_capex=500000.0,
        )
        capex_rows = result[result["param"] == "capex"].sort_values("value")
        npvs = capex_rows["npv"].tolist()
        assert npvs[0] > npvs[1] > npvs[2]

    def test_custom_vary(self) -> None:
        result = sensitivity_table(
            base_revenue=100000.0,
            total_capex=500000.0,
            vary={"revenue": [0.8, 1.0, 1.2]},
        )
        assert len(result) == 3
        assert list(result["param"].unique()) == ["revenue"]
