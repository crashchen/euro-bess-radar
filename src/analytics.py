"""Analytics engine for BESS market screening metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_local(df: pd.DataFrame, tz: str | None) -> pd.DataFrame:
    """Return a copy with index converted to local time, or as-is if tz is None."""
    if tz is None:
        return df
    out = df.copy()
    out.index = out.index.tz_convert(tz)
    return out


def _infer_interval_hours(index: pd.DatetimeIndex) -> float:
    """Infer the dominant sampling interval in hours from a DatetimeIndex."""
    if len(index) < 2:
        return 1.0

    deltas = index.to_series().diff().dropna().dt.total_seconds() / 3600.0
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return 1.0

    mode = deltas.mode()
    if not mode.empty:
        return float(mode.iloc[0])
    return float(deltas.median())


def _window_length(index: pd.DatetimeIndex, duration_hours: float) -> int:
    """Convert BESS duration in hours to an integer number of intervals."""
    interval_hours = _infer_interval_hours(index)
    return max(int(round(duration_hours / interval_hours)), 1)


def _calculate_daily_ordered_spread(
    prices: pd.Series,
    duration_hours: float,
) -> dict[str, float | int]:
    """Find the best non-overlapping charge/discharge window within one day."""
    if prices.empty:
        return {
            "daily_min": 0.0,
            "daily_max": 0.0,
            "spread": 0.0,
            "max_hour": 0,
            "min_hour": 0,
        }

    window = _window_length(prices.index, duration_hours)
    if len(prices) < window * 2:
        baseline = float(prices.iloc[:window].mean())
        start_hour = int(prices.index[0].hour)
        return {
            "daily_min": baseline,
            "daily_max": baseline,
            "spread": 0.0,
            "max_hour": start_hour,
            "min_hour": start_hour,
        }

    window_values = np.array([
        float(prices.iloc[i:i + window].mean())
        for i in range(len(prices) - window + 1)
    ])
    window_starts = prices.index[: len(window_values)]
    future_best_values = np.empty(len(window_values))
    future_best_indices = np.empty(len(window_values), dtype=int)

    best_future_value = -np.inf
    best_future_idx = 0
    for i in range(len(window_values) - 1, -1, -1):
        if window_values[i] >= best_future_value:
            best_future_value = window_values[i]
            best_future_idx = i
        future_best_values[i] = best_future_value
        future_best_indices[i] = best_future_idx

    best_spread = 0.0
    best_buy_idx = int(np.argmin(window_values))
    best_sell_idx = best_buy_idx

    for buy_idx, buy_value in enumerate(window_values):
        sell_start = buy_idx + window
        if sell_start >= len(window_values):
            continue

        sell_idx = future_best_indices[sell_start]
        spread = future_best_values[sell_start] - buy_value
        if spread > best_spread:
            best_spread = float(spread)
            best_buy_idx = buy_idx
            best_sell_idx = int(sell_idx)

    buy_value = float(window_values[best_buy_idx])
    if best_spread <= 0:
        return {
            "daily_min": buy_value,
            "daily_max": buy_value,
            "spread": 0.0,
            "max_hour": int(window_starts[best_buy_idx].hour),
            "min_hour": int(window_starts[best_buy_idx].hour),
        }

    sell_value = float(window_values[best_sell_idx])
    return {
        "daily_min": buy_value,
        "daily_max": sell_value,
        "spread": round(sell_value - buy_value, 10),
        "max_hour": int(window_starts[best_sell_idx].hour),
        "min_hour": int(window_starts[best_buy_idx].hour),
    }


def calculate_daily_spreads(
    df: pd.DataFrame,
    tz: str | None = None,
    duration_hours: float = 1.0,
) -> pd.DataFrame:
    """Calculate the best daily ordered spread for arbitrage potential.

    Args:
        df: DataFrame with DatetimeIndex 'timestamp' and column 'price_eur_mwh'.
        tz: IANA timezone for local-time day/hour grouping. None = use index as-is.
        duration_hours: Charge/discharge window length for spread calculation.

    Returns:
        DataFrame with columns:
        [date, daily_min, daily_max, spread, max_hour, min_hour], where
        daily_min/max are the average buy/sell window prices and the spread
        respects charge-before-discharge ordering.
    """
    local = _to_local(df, tz)
    prices = local["price_eur_mwh"]
    daily = prices.groupby(prices.index.date)

    records = []
    for date, group in daily:
        metrics = _calculate_daily_ordered_spread(group.sort_index(), duration_hours)
        metrics["date"] = date
        records.append(metrics)

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return pd.DataFrame(
            columns=["date", "daily_min", "daily_max", "spread", "max_hour", "min_hour"],
        )

    result = result[["date", "daily_min", "daily_max", "spread", "max_hour", "min_hour"]]
    result.index.name = "date"
    return result


def calculate_monthly_spreads(
    df: pd.DataFrame,
    tz: str | None = None,
    duration_hours: float = 1.0,
) -> pd.DataFrame:
    """Aggregate daily spreads to monthly averages.

    Args:
        df: Raw price DataFrame (not daily spreads).
        tz: IANA timezone for local-time grouping. None = use index as-is.
        duration_hours: Charge/discharge window length for spread calculation.

    Returns:
        DataFrame with columns:
        [year_month, avg_spread, median_spread, max_spread, min_spread,
         avg_daily_max, avg_daily_min].
    """
    daily = calculate_daily_spreads(df, tz=tz, duration_hours=duration_hours)
    daily["year_month"] = pd.to_datetime(daily["date"]).dt.to_period("M").astype(str)

    monthly = daily.groupby("year_month").agg(
        avg_spread=("spread", "mean"),
        median_spread=("spread", "median"),
        max_spread=("spread", "max"),
        min_spread=("spread", "min"),
        avg_daily_max=("daily_max", "mean"),
        avg_daily_min=("daily_min", "mean"),
    ).reset_index()
    return monthly


def calculate_spread_percentiles(
    daily_spreads: pd.DataFrame,
) -> dict[str, float]:
    """Calculate P50, P75, P90 of daily spreads.

    Args:
        daily_spreads: DataFrame with 'spread' column from calculate_daily_spreads.

    Returns:
        Dict with keys: p50, p75, p90, mean, std.
    """
    s = daily_spreads["spread"]
    return {
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "mean": float(s.mean()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
    }


def build_price_heatmap(
    df: pd.DataFrame,
    tz: str | None = None,
) -> pd.DataFrame:
    """Build hour-of-day vs year-month average price matrix.

    Args:
        df: Price DataFrame with DatetimeIndex.
        tz: IANA timezone for local-time hour grouping. None = use index as-is.

    Returns:
        DataFrame with rows=hours 0-23, columns=year-month strings.
    """
    local = _to_local(df, tz)
    tmp = local[["price_eur_mwh"]].copy()
    tmp["hour"] = tmp.index.hour
    tmp["year_month"] = tmp.index.tz_localize(None).to_period("M").astype(str)

    pivot = tmp.pivot_table(
        values="price_eur_mwh", index="hour", columns="year_month", aggfunc="mean",
    )
    return pivot.reindex(range(24))


def build_spread_heatmap(
    df: pd.DataFrame,
    tz: str | None = None,
) -> pd.DataFrame:
    """Build hour-of-day vs year-month spread contribution matrix.

    For each hour-month cell: average(price - daily_mean_price).
    Positive = sell window, negative = buy window.

    Args:
        df: Price DataFrame with DatetimeIndex.
        tz: IANA timezone for local-time grouping. None = use index as-is.

    Returns:
        DataFrame with rows=hours 0-23, columns=year-month strings.
    """
    local = _to_local(df, tz)
    tmp = local[["price_eur_mwh"]].copy()
    daily_mean = tmp["price_eur_mwh"].groupby(tmp.index.date).transform("mean")
    tmp["deviation"] = tmp["price_eur_mwh"] - daily_mean
    tmp["hour"] = tmp.index.hour
    tmp["year_month"] = tmp.index.tz_localize(None).to_period("M").astype(str)

    pivot = tmp.pivot_table(
        values="deviation", index="hour", columns="year_month", aggfunc="mean",
    )
    return pivot.reindex(range(24))


def estimate_annual_arbitrage_revenue(
    daily_spreads: pd.DataFrame,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    roundtrip_efficiency: float = 0.88,
    cycles_per_day: float = 1.5,
) -> dict[str, float]:
    """Estimate annualised BESS arbitrage revenue from daily spreads.

    Args:
        daily_spreads: DataFrame with 'spread' column for the relevant
            charge/discharge duration.
        power_mw: BESS power rating in MW.
        duration_hours: BESS duration in hours.
        roundtrip_efficiency: Round-trip efficiency (0-1).
        cycles_per_day: Average cycles per day.

    Returns:
        Dict with annual_revenue_eur, annual_revenue_eur_per_mw,
        avg_daily_revenue, capture_rate_assumption.
    """
    capture_rate = 0.70
    avg_spread = float(daily_spreads["spread"].mean())
    energy_mwh = power_mw * duration_hours

    daily_revenue = (
        avg_spread * energy_mwh * roundtrip_efficiency * capture_rate * cycles_per_day
    )
    days_in_year = 365.25
    annual_revenue = daily_revenue * days_in_year

    return {
        "annual_revenue_eur": round(annual_revenue, 2),
        "annual_revenue_eur_per_mw": round(annual_revenue / power_mw, 2),
        "avg_daily_revenue": round(daily_revenue, 2),
        "capture_rate_assumption": capture_rate,
    }


def calculate_negative_price_hours(df: pd.DataFrame) -> dict[str, float]:
    """Count and analyse negative price occurrences.

    Args:
        df: Price DataFrame with 'price_eur_mwh' column.

    Returns:
        Dict with negative_hours, negative_intervals, pct_negative,
        avg_negative_price, most_negative_price.
    """
    prices = df["price_eur_mwh"]
    negative = prices[prices < 0]

    total = len(prices)
    neg_count = len(negative)
    interval_hours = _infer_interval_hours(df.index)
    negative_hours = round(neg_count * interval_hours, 2)

    return {
        "negative_hours": negative_hours,
        "negative_intervals": int(neg_count),
        "total_negative_hours": negative_hours,
        "pct_negative": round(100.0 * neg_count / total, 2) if total > 0 else 0.0,
        "avg_negative_price": round(float(negative.mean()), 2) if neg_count > 0 else 0.0,
        "most_negative_price": round(float(negative.min()), 2) if neg_count > 0 else 0.0,
    }


# ── Renewable correlation ─────────────────────────────────────────────────────

def analyze_price_renewable_correlation(
    price_df: pd.DataFrame,
    generation_df: pd.DataFrame,
) -> dict:
    """Analyse correlation between renewable output and prices.

    Args:
        price_df: Price DataFrame with 'price_eur_mwh' column.
        generation_df: Generation DataFrame with 'renewable_pct' column.

    Returns:
        Dict with correlation coefficients and quartile analysis.
    """
    merged = price_df[["price_eur_mwh"]].join(generation_df, how="inner")
    if merged.empty or len(merged) < 10:
        return {
            "correlation_wind_price": 0.0,
            "correlation_solar_price": 0.0,
            "correlation_renewable_price": 0.0,
            "avg_price_high_renewable": 0.0,
            "avg_price_low_renewable": 0.0,
            "price_spread_by_renewable_quartile": {},
            "negative_price_renewable_pct": 0.0,
        }

    price = merged["price_eur_mwh"]
    result: dict = {}

    for col, key in [
        ("wind_onshore_mw", "correlation_wind_price"),
        ("solar_mw", "correlation_solar_price"),
        ("renewable_pct", "correlation_renewable_price"),
    ]:
        if col in merged.columns and merged[col].std() > 0:
            result[key] = round(float(price.corr(merged[col])), 4)
        else:
            result[key] = 0.0

    re_pct = merged["renewable_pct"]
    q75 = re_pct.quantile(0.75)
    q25 = re_pct.quantile(0.25)

    result["avg_price_high_renewable"] = round(
        float(price[re_pct >= q75].mean()), 2) if (re_pct >= q75).any() else 0.0
    result["avg_price_low_renewable"] = round(
        float(price[re_pct <= q25].mean()), 2) if (re_pct <= q25).any() else 0.0

    quartile_labels = pd.qcut(re_pct, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    result["price_spread_by_renewable_quartile"] = {
        str(q): round(float(price[quartile_labels == q].mean()), 2)
        for q in quartile_labels.dropna().unique()
    }

    neg_mask = price < 0
    if neg_mask.any():
        result["negative_price_renewable_pct"] = round(float(re_pct[neg_mask].mean()), 2)
    else:
        result["negative_price_renewable_pct"] = 0.0

    return result


def build_renewable_price_scatter(
    price_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    tz: str | None = None,
) -> pd.DataFrame:
    """Prepare data for renewable output vs price scatter plot.

    Args:
        price_df: Price DataFrame.
        generation_df: Generation DataFrame.
        tz: IANA timezone for local-time hour/month extraction.

    Returns:
        DataFrame with [price_eur_mwh, renewable_pct, hour, month].
    """
    merged = price_df[["price_eur_mwh"]].join(generation_df[["renewable_pct"]], how="inner")
    if merged.empty:
        return merged

    local_idx = merged.index.tz_convert(tz) if tz else merged.index
    merged["hour"] = local_idx.hour
    merged["month"] = local_idx.month
    return merged


# ── Zone comparison ──────────────────────────────────────────────────────────

def compare_zones(
    zone_data: dict[str, pd.DataFrame],
    zone_timezones: dict[str, str] | None = None,
    duration_hours: float = 1.0,
) -> pd.DataFrame:
    """Compare key metrics across multiple zones.

    Args:
        zone_data: Dict mapping zone_code -> price DataFrame.
        zone_timezones: Optional dict mapping zone_code -> IANA timezone.
        duration_hours: Charge/discharge window length for spread calculation.

    Returns:
        Summary DataFrame with one row per zone.
    """
    rows = []
    for zone, df in zone_data.items():
        if df.empty or "price_eur_mwh" not in df.columns:
            continue

        tz = zone_timezones.get(zone) if zone_timezones else None
        daily = calculate_daily_spreads(df, tz=tz, duration_hours=duration_hours)
        pctls = calculate_spread_percentiles(daily)
        neg = calculate_negative_price_hours(df)
        rev = estimate_annual_arbitrage_revenue(daily, duration_hours=duration_hours)

        rows.append({
            "zone": zone,
            "avg_price": round(float(df["price_eur_mwh"].mean()), 2),
            "std_price": round(float(df["price_eur_mwh"].std()), 2),
            "avg_spread": round(pctls["mean"], 2),
            "p50_spread": round(pctls["p50"], 2),
            "p90_spread": round(pctls["p90"], 2),
            "negative_pct": neg["pct_negative"],
            "estimated_annual_revenue_per_mw": rev["annual_revenue_eur_per_mw"],
        })

    return pd.DataFrame(rows)
