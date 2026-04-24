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


def _find_daily_ordered_trade(
    prices: pd.Series,
    duration_hours: float,
) -> dict[str, float | int]:
    """Find the best non-overlapping charge/discharge trade within one day."""
    if prices.empty:
        return {
            "buy_value": 0.0,
            "sell_value": 0.0,
            "spread": 0.0,
            "buy_start_idx": 0,
            "sell_start_idx": 0,
            "window": 1,
        }

    window = _window_length(prices.index, duration_hours)
    if len(prices) < window * 2:
        baseline = float(prices.iloc[:window].mean())
        return {
            "buy_value": baseline,
            "sell_value": baseline,
            "spread": 0.0,
            "buy_start_idx": 0,
            "sell_start_idx": 0,
            "window": window,
        }

    # `rolling()` labels windows by their end timestamps, but the values still
    # arrive in start-position order 0..n-window after dropping the initial NaNs.
    window_values = prices.rolling(
        window=window,
        min_periods=window,
    ).mean().dropna().to_numpy()
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
            "buy_value": buy_value,
            "sell_value": buy_value,
            "spread": 0.0,
            "buy_start_idx": best_buy_idx,
            "sell_start_idx": best_buy_idx,
            "window": window,
        }

    sell_value = float(window_values[best_sell_idx])
    return {
        "buy_value": buy_value,
        "sell_value": sell_value,
        "spread": round(sell_value - buy_value, 10),
        "buy_start_idx": best_buy_idx,
        "sell_start_idx": best_sell_idx,
        "window": window,
    }


def _calculate_daily_ordered_spread(
    prices: pd.Series,
    duration_hours: float,
) -> dict[str, float | int]:
    """Summarise the best non-overlapping charge/discharge window within one day."""
    trade = _find_daily_ordered_trade(prices, duration_hours)
    return {
        "daily_min": float(trade["buy_value"]),
        "daily_max": float(trade["sell_value"]),
        "spread": float(trade["spread"]),
        "max_hour": int(prices.index[int(trade["sell_start_idx"])].hour),
        "min_hour": int(prices.index[int(trade["buy_start_idx"])].hour),
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


def calculate_daily_dispatch(
    df: pd.DataFrame,
    tz: str | None = None,
    duration_hours: float = 1.0,
    power_mw: float = 1.0,
    efficiency: float = 0.88,
) -> pd.DataFrame:
    """Compute daily spreads with both greedy and LP dispatch results.

    Returns a superset of :func:`calculate_daily_spreads` with additional
    LP columns (``lp_revenue``, ``n_cycles``, ``lp_spread_eur_mwh``).

    Args:
        df: Price DataFrame with DatetimeIndex and 'price_eur_mwh'.
        tz: IANA timezone for local-day grouping.
        duration_hours: BESS duration in hours.
        power_mw: BESS power rating in MW.
        efficiency: Round-trip efficiency (0–1).
    """
    from src.dispatch import solve_dispatch_batch

    greedy = calculate_daily_spreads(df, tz=tz, duration_hours=duration_hours)
    lp = solve_dispatch_batch(
        df,
        power_mw=power_mw,
        duration_hours=duration_hours,
        efficiency=efficiency,
        tz=tz,
        soc_init_frac=0.0,
    )

    if greedy.empty or lp.empty:
        for col in ("lp_revenue", "n_cycles", "lp_spread_eur_mwh"):
            greedy[col] = pd.Series(dtype=float)
        return greedy

    # greedy has 'date' as both a column and index.name — drop the index name
    # to avoid pandas ambiguity, then join LP results via the 'date' column.
    greedy = greedy.copy()
    greedy.index.name = None
    lp_indexed = lp.set_index("date")
    merged = greedy.join(lp_indexed, on="date", how="left")
    merged.index.name = "date"
    return merged


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
    return calculate_monthly_spreads_from_daily(daily)


def calculate_monthly_spreads_from_daily(daily_spreads: pd.DataFrame) -> pd.DataFrame:
    """Aggregate already-computed daily spreads to monthly averages.

    This avoids recomputing ordered daily trades when the dashboard already has
    a duration-aware `daily_spreads` frame for the active zone.
    """
    if daily_spreads.empty:
        return pd.DataFrame(
            columns=[
                "year_month", "avg_spread", "median_spread", "max_spread",
                "min_spread", "avg_daily_max", "avg_daily_min",
            ]
        )

    daily = daily_spreads.copy()
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
    duration_hours: float = 1.0,
) -> pd.DataFrame:
    """Build hour-of-day vs year-month ordered-spread signal matrix.

    For each day, mark the selected charge window as negative and the selected
    discharge window as positive using the day's ordered spread magnitude.
    This is a selected-window signal for visualising which local hours are
    repeatedly chosen by the ordered-spread model; it is not hourly revenue
    attribution. Positive = selected discharge window, negative = selected
    charge window.

    Args:
        df: Price DataFrame with DatetimeIndex.
        tz: IANA timezone for local-time grouping. None = use index as-is.
        duration_hours: Charge/discharge window length used for selection.

    Returns:
        DataFrame with rows=hours 0-23, columns=year-month strings.
    """
    local = _to_local(df, tz)
    prices = local["price_eur_mwh"]
    signals = []

    for _, group in prices.groupby(prices.index.date):
        group = group.sort_index()
        signal = pd.Series(0.0, index=group.index, name="signal")
        trade = _find_daily_ordered_trade(group, duration_hours)
        spread = float(trade["spread"])
        if spread > 0:
            window = int(trade["window"])
            buy_start = int(trade["buy_start_idx"])
            sell_start = int(trade["sell_start_idx"])
            signal.iloc[buy_start:buy_start + window] = -spread
            signal.iloc[sell_start:sell_start + window] = spread
        signals.append(signal)

    if not signals:
        return pd.DataFrame(index=range(24))

    tmp = pd.DataFrame({"signal": pd.concat(signals).sort_index()})
    tmp["hour"] = tmp.index.hour
    tmp["year_month"] = tmp.index.tz_localize(None).to_period("M").astype(str)

    pivot = tmp.pivot_table(
        values="signal", index="hour", columns="year_month", aggfunc="mean",
    )
    return pivot.reindex(range(24)).fillna(0.0)


def estimate_annual_arbitrage_revenue(
    daily_spreads: pd.DataFrame,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
    roundtrip_efficiency: float = 0.88,
    capture_rate: float = 0.70,
    cycles_per_day: float = 1.0,
) -> dict[str, float]:
    """Estimate annualised BESS arbitrage revenue from daily spreads.

    Auto-detects LP dispatch columns (``lp_revenue``) when present and uses
    the LP-optimal daily revenue directly.  Falls back to greedy heuristic
    otherwise, preserving full backward compatibility.

    Args:
        daily_spreads: DataFrame with 'spread' column (greedy) and optionally
            'lp_revenue'/'n_cycles' columns from LP dispatch.
        power_mw: BESS power rating in MW.
        duration_hours: BESS duration in hours.
        roundtrip_efficiency: Round-trip efficiency (0-1).
        capture_rate: Share of theoretical spread assumed to be captured (0-1).
        cycles_per_day: Average cycles per day (greedy path only).

    Returns:
        Dict with annual_revenue_eur, annual_revenue_eur_per_mw,
        avg_daily_revenue, capture_rate_assumption, cycles_per_day_assumption,
        dispatch_method.
    """
    days_in_year = 365.25

    if "lp_revenue" in daily_spreads.columns:
        avg_daily_revenue = float(daily_spreads["lp_revenue"].mean()) * capture_rate
        annual_revenue = avg_daily_revenue * days_in_year
        avg_cycles = float(daily_spreads["n_cycles"].mean()) if "n_cycles" in daily_spreads.columns else 1.0
        return {
            "annual_revenue_eur": round(annual_revenue, 2),
            "annual_revenue_eur_per_mw": round(annual_revenue / power_mw, 2),
            "avg_daily_revenue": round(avg_daily_revenue, 2),
            "capture_rate_assumption": capture_rate,
            "cycles_per_day_assumption": round(avg_cycles, 2),
            "dispatch_method": "lp",
        }

    avg_spread = float(daily_spreads["spread"].mean())
    energy_mwh = power_mw * duration_hours

    daily_revenue = (
        avg_spread * energy_mwh * roundtrip_efficiency * capture_rate * cycles_per_day
    )
    annual_revenue = daily_revenue * days_in_year

    return {
        "annual_revenue_eur": round(annual_revenue, 2),
        "annual_revenue_eur_per_mw": round(annual_revenue / power_mw, 2),
        "avg_daily_revenue": round(daily_revenue, 2),
        "capture_rate_assumption": capture_rate,
        "cycles_per_day_assumption": cycles_per_day,
        "dispatch_method": "greedy",
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


def _quartile_averages(
    driver: pd.Series,
    metric: pd.Series,
) -> dict[str, float]:
    """Return Q1..Q4 metric averages against the supplied driver series."""
    valid = pd.DataFrame({
        "driver": driver,
        "metric": metric,
    }).dropna()
    if len(valid) < 4 or valid["driver"].nunique() < 4:
        return {}

    quartiles = pd.qcut(
        valid["driver"],
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )
    if quartiles.dropna().nunique() < 4:
        return {}

    return {
        quartile: round(
            float(valid.loc[quartiles == quartile, "metric"].mean()),
            2,
        )
        for quartile in ["Q1", "Q2", "Q3", "Q4"]
    }


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


def build_daily_renewable_spread_view(
    price_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    tz: str | None = None,
    duration_hours: float = 1.0,
) -> pd.DataFrame:
    """Join local-day renewable share with daily ordered spreads."""
    if generation_df.empty:
        return pd.DataFrame(columns=["date", "spread", "daily_avg_renewable_pct"])

    local_generation = _to_local(generation_df, tz)
    if "renewable_pct" not in local_generation.columns:
        return pd.DataFrame(columns=["date", "spread", "daily_avg_renewable_pct"])

    daily_re = (
        local_generation[["renewable_pct"]]
        .groupby(local_generation.index.date)
        .mean()
        .reset_index()
        .rename(
            columns={
                "index": "date",
                "renewable_pct": "daily_avg_renewable_pct",
            }
        )
    )
    if "date" not in daily_re.columns:
        daily_re = daily_re.rename(columns={daily_re.columns[0]: "date"})

    daily_spreads = calculate_daily_spreads(
        price_df,
        tz=tz,
        duration_hours=duration_hours,
    )
    if daily_spreads.empty:
        return pd.DataFrame(columns=["date", "spread", "daily_avg_renewable_pct"])

    merged = (
        daily_spreads.rename_axis(None)
        .reset_index(drop=True)
        .merge(daily_re.rename_axis(None), on="date", how="inner")
    )
    return merged


def analyze_renewable_bess_signal(
    price_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    tz: str | None = None,
    duration_hours: float = 1.0,
) -> dict:
    """Analyse whether renewables compress prices and widen capturable spreads."""
    hourly = build_renewable_price_scatter(price_df, generation_df, tz=tz)
    daily = build_daily_renewable_spread_view(
        price_df,
        generation_df,
        tz=tz,
        duration_hours=duration_hours,
    )

    result: dict = {
        "correlation_renewable_price": None,
        "avg_price_high_renewable": None,
        "avg_price_low_renewable": None,
        "avg_spread_high_renewable_day": None,
        "avg_spread_low_renewable_day": None,
        "spread_uplift_high_vs_low_renewable": None,
        "price_by_renewable_quartile": {},
        "spread_by_renewable_quartile": {},
        "hourly_points": int(len(hourly)),
        "daily_points": int(len(daily)),
    }

    if len(hourly) >= 10 and hourly["renewable_pct"].nunique() > 1:
        price = hourly["price_eur_mwh"]
        re_pct = hourly["renewable_pct"]
        result["correlation_renewable_price"] = round(
            float(price.corr(re_pct)),
            4,
        )

        q75 = re_pct.quantile(0.75)
        q25 = re_pct.quantile(0.25)
        high_prices = price[re_pct >= q75]
        low_prices = price[re_pct <= q25]
        if not high_prices.empty:
            result["avg_price_high_renewable"] = round(float(high_prices.mean()), 2)
        if not low_prices.empty:
            result["avg_price_low_renewable"] = round(float(low_prices.mean()), 2)

        result["price_by_renewable_quartile"] = _quartile_averages(re_pct, price)

    if len(daily) >= 4 and daily["daily_avg_renewable_pct"].nunique() > 1:
        spread = daily["spread"]
        daily_re = daily["daily_avg_renewable_pct"]
        q75 = daily_re.quantile(0.75)
        q25 = daily_re.quantile(0.25)
        high_spreads = spread[daily_re >= q75]
        low_spreads = spread[daily_re <= q25]
        if not high_spreads.empty:
            result["avg_spread_high_renewable_day"] = round(
                float(high_spreads.mean()),
                2,
            )
        if not low_spreads.empty:
            result["avg_spread_low_renewable_day"] = round(
                float(low_spreads.mean()),
                2,
            )
        if (
            result["avg_spread_high_renewable_day"] is not None
            and result["avg_spread_low_renewable_day"] is not None
        ):
            result["spread_uplift_high_vs_low_renewable"] = round(
                result["avg_spread_high_renewable_day"]
                - result["avg_spread_low_renewable_day"],
                2,
            )

        result["spread_by_renewable_quartile"] = _quartile_averages(daily_re, spread)

    return result


# ── Zone comparison ──────────────────────────────────────────────────────────

def compare_zones(
    zone_data: dict[str, pd.DataFrame],
    zone_timezones: dict[str, str] | None = None,
    duration_hours: float = 1.0,
    capture_rate: float = 0.70,
    roundtrip_efficiency: float = 0.88,
) -> pd.DataFrame:
    """Compare key metrics across multiple zones.

    Args:
        zone_data: Dict mapping zone_code -> price DataFrame.
        zone_timezones: Optional dict mapping zone_code -> IANA timezone.
        duration_hours: Charge/discharge window length for spread calculation.
        capture_rate: Share of theoretical spread assumed to be captured (0-1).
        roundtrip_efficiency: Round-trip efficiency of the BESS (0-1).

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
        rev = estimate_annual_arbitrage_revenue(
            daily,
            duration_hours=duration_hours,
            capture_rate=capture_rate,
            roundtrip_efficiency=roundtrip_efficiency,
        )

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
