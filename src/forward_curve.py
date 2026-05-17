"""Forward power curve parsing and BESS forward-scenario revenue.

Two-stage pipeline:
1. ``parse_forward_csv`` reads a tidy CSV of forward contracts
   (zone, delivery_start, delivery_end, price_eur_mwh) such as those
   exported from EEX EOD, brokers, Bloomberg/Refinitiv, or an internal
   price-assumption sheet.
2. ``build_forward_synthetic_prices`` overlays each contract's baseload
   price onto a reference historical hourly shape for the same zone, so
   the existing daily-spread / MILP-dispatch / NPV machinery can run
   forward-looking without further changes.

The "synthetic hourly = forward_base * (historical_hourly /
historical_period_mean)" formula deliberately preserves the historical
intra-day shape (peak-vs-trough ratio) while letting the level move with
the forward curve. This is the standard analyst convention because
forward curves quote a single number per delivery period (base or
optionally peak/offpeak); shape recovery from forwards alone is
impossible without an additional model.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.config import ALL_ZONES

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "zone",
    "delivery_start",
    "delivery_end",
    "price_eur_mwh",
)
OPTIONAL_COLUMNS = ("contract", "shape", "source", "as_of")


def generate_forward_template_csv() -> str:
    """Minimal CSV with header + example rows for the upload UI."""
    rows = [
        "# Forward power curve template.",
        "# zone           = bidding zone code matching src/config.py.",
        "# contract       = optional label (e.g. Cal-2027, Q1-2027, Mar-2027).",
        "# delivery_start = ISO date, inclusive (YYYY-MM-DD).",
        "# delivery_end   = ISO date, EXCLUSIVE (YYYY-MM-DD).",
        "# price_eur_mwh  = forward baseload price for the period (EUR/MWh).",
        "# shape          = optional: 'base' (default) | 'peak' | 'offpeak'.",
        "# source / as_of = optional provenance fields (free text).",
        "zone,contract,delivery_start,delivery_end,price_eur_mwh,shape,source,as_of",
        "DE_LU,Cal-2027,2027-01-01,2028-01-01,82.5,base,EEX EOD,2026-05-15",
        "NL,Q1-2027,2027-01-01,2027-04-01,91.0,base,EEX EOD,2026-05-15",
        "IT_NORD,Mar-2027,2027-03-01,2027-04-01,104.2,base,Internal,2026-05-15",
    ]
    return "\n".join(rows) + "\n"


def parse_forward_csv(csv_content: str | Path) -> pd.DataFrame:
    """Parse a forward-curve CSV into a normalised DataFrame.

    Args:
        csv_content: CSV string or file path. ``#``-prefixed lines are
            treated as comments and skipped.

    Returns:
        DataFrame with columns
        ``[zone, contract, delivery_start, delivery_end, price_eur_mwh,
        shape, source, as_of]`` — optional columns are added with NaN
        when absent. Dates are tz-naive ``datetime64[ns]`` (the actual
        timezone is the zone's local time, applied later when the
        synthetic price series is built).

    Raises:
        ValueError: when a required column is missing, when zone codes
            are not in ``ALL_ZONES``, when prices are not finite, or
            when delivery_start >= delivery_end on any row.
    """
    if isinstance(csv_content, Path):
        text = csv_content.read_text(encoding="utf-8-sig")
    else:
        text = csv_content
    df = pd.read_csv(io.StringIO(text), comment="#")
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Forward CSV missing required columns: {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}."
        )
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Zone validation against the project's single-source-of-truth zone list.
    known_zones = set(ALL_ZONES.values())
    df["zone"] = df["zone"].astype(str).str.strip()
    bad_zones = sorted(set(df["zone"]) - known_zones)
    if bad_zones:
        raise ValueError(
            f"Forward CSV contains unknown zone(s): {bad_zones}. "
            f"Use one of: {sorted(known_zones)}."
        )

    df["delivery_start"] = pd.to_datetime(df["delivery_start"], errors="coerce")
    df["delivery_end"] = pd.to_datetime(df["delivery_end"], errors="coerce")
    if df[["delivery_start", "delivery_end"]].isna().any().any():
        raise ValueError(
            "Forward CSV has unparseable delivery_start / delivery_end values."
        )
    if (df["delivery_end"] <= df["delivery_start"]).any():
        raise ValueError(
            "delivery_end must be strictly after delivery_start on every row."
        )

    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")
    if df["price_eur_mwh"].isna().any():
        raise ValueError("Forward CSV has non-numeric price_eur_mwh values.")

    df["shape"] = df["shape"].fillna("base").astype(str).str.strip().str.lower()
    df["contract"] = df["contract"].astype(str).where(df["contract"].notna(), other=None)

    return df[[
        "zone", "contract",
        "delivery_start", "delivery_end",
        "price_eur_mwh", "shape", "source", "as_of",
    ]].sort_values(["zone", "delivery_start"]).reset_index(drop=True)


def find_overlapping_contracts(forward_df: pd.DataFrame) -> pd.DataFrame:
    """Return pairs of contracts that overlap on the same zone.

    Useful for warning the user that, e.g., a Cal-2027 contract and a
    Q1-2027 contract both cover Jan-Mar 2027 — applying both naively
    would double-count revenue. The dashboard surfaces this; the math
    does not auto-resolve it.
    """
    rows: list[dict] = []
    for zone, group in forward_df.groupby("zone"):
        ordered = group.sort_values("delivery_start").reset_index(drop=True)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered.iloc[i], ordered.iloc[j]
                if a["delivery_end"] > b["delivery_start"]:
                    rows.append({
                        "zone": zone,
                        "contract_a": a.get("contract") or f"#{i}",
                        "contract_b": b.get("contract") or f"#{j}",
                        "overlap_start": max(a["delivery_start"], b["delivery_start"]),
                        "overlap_end": min(a["delivery_end"], b["delivery_end"]),
                    })
    return pd.DataFrame(rows)


def _build_normalised_shape(
    historical_df: pd.DataFrame, tz: str | None,
) -> pd.Series:
    """Return a Series of hour-of-week shape factors normalised to mean=1.

    Hour-of-week (168 buckets) is the lowest-friction shape that captures
    both weekday/weekend and intraday patterns. Days with NaN prices in
    the period drop out of the average so missing data does not bias the
    shape.
    """
    local = historical_df.copy()
    if tz is not None:
        local.index = local.index.tz_convert(tz)
    series = local["price_eur_mwh"].dropna()
    if series.empty:
        raise ValueError(
            "Historical reference series has no usable prices for shape recovery."
        )
    # hour-of-week = weekday * 24 + hour
    bucket = series.index.weekday * 24 + series.index.hour
    by_bucket = series.groupby(bucket).mean()
    overall = float(by_bucket.mean())
    if overall <= 0:
        # Degenerate shape (all-negative or zero historical mean). Fall back to
        # a flat shape so callers don't divide by zero downstream — the level
        # term still carries the forward signal.
        return pd.Series(1.0, index=range(168))
    return (by_bucket / overall).reindex(range(168), fill_value=1.0)


def build_forward_synthetic_prices(
    forward_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    *,
    zone: str,
    tz: str | None = None,
) -> pd.DataFrame:
    """Generate hourly synthetic forward prices for one zone.

    For each forward contract on the zone, expands the contract period
    into hourly timestamps and applies the formula
    ``price[h] = forward_base * historical_shape[hour_of_week(h)]``.

    Args:
        forward_df: Parsed forward-curve DataFrame.
        historical_df: Historical hourly DA frame with DatetimeIndex
            (UTC) and ``price_eur_mwh`` column — typically the same
            ``primary_df`` shown elsewhere in the dashboard.
        zone: Bidding zone to filter forward_df on.
        tz: IANA timezone for the synthetic hour-of-week mapping; usually
            the zone's local time. Falls back to UTC when None.

    Returns:
        DataFrame indexed by UTC hourly timestamp with columns
        ``[price_eur_mwh, contract, forward_base, shape_factor]``. Rows
        are sorted and de-duplicated; if two contracts overlap, the
        later one in the original CSV order wins on the overlap.
    """
    zone_forwards = forward_df[forward_df["zone"] == zone].copy()
    if zone_forwards.empty:
        return pd.DataFrame(
            columns=["price_eur_mwh", "contract", "forward_base", "shape_factor"],
        )

    shape = _build_normalised_shape(historical_df, tz)
    target_tz = tz or "UTC"

    frames: list[pd.DataFrame] = []
    for priority, (_, row) in enumerate(zone_forwards.iterrows()):
        start = pd.Timestamp(row["delivery_start"]).tz_localize(target_tz)
        end = pd.Timestamp(row["delivery_end"]).tz_localize(target_tz)
        if end <= start:
            continue
        idx = pd.date_range(start=start, end=end, freq="h", inclusive="left")
        if len(idx) == 0:
            continue
        local_bucket = idx.weekday * 24 + idx.hour
        factors = shape.reindex(local_bucket).to_numpy()
        prices = float(row["price_eur_mwh"]) * factors
        frame = pd.DataFrame({
            "price_eur_mwh": prices,
            "contract": row.get("contract") or "",
            "forward_base": float(row["price_eur_mwh"]),
            "shape_factor": factors,
            "_priority": priority,
        }, index=idx.tz_convert("UTC"))
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=["price_eur_mwh", "contract", "forward_base", "shape_factor"],
        )

    # Keep the later contract on overlap: sort by (timestamp, priority) so
    # the highest-priority row at each timestamp ends up last, then dedup
    # with keep="last". Explicit priority key avoids the sort_index
    # stability assumption that broke in pandas 2.x for tz-aware indexes.
    out = pd.concat(frames)
    out.index.name = "timestamp"
    out = out.reset_index().sort_values(["timestamp", "_priority"])
    out = out.drop_duplicates(subset="timestamp", keep="last")
    out = out.set_index("timestamp").drop(columns="_priority")
    return out


def summarise_forward_revenue(
    daily_spreads: pd.DataFrame,
    forward_df: pd.DataFrame,
    *,
    zone: str,
    power_mw: float,
    duration_hours: float,
    efficiency: float = 0.88,
    capture_rate: float = 0.70,
) -> pd.DataFrame:
    """Aggregate forward dispatch results to per-contract revenue.

    Joins the per-day daily-spread / LP-revenue output back to the
    forward contract whose delivery period contains that day, and
    annualises within the contract using the same EUR/MW/yr convention
    used by ``estimate_annual_arbitrage_revenue``.

    Args:
        daily_spreads: Output of ``calculate_daily_spreads`` or
            ``calculate_daily_dispatch`` on the synthetic forward prices.
        forward_df: The original parsed forward DataFrame.
        zone: Zone code to filter forward contracts on.
        power_mw, duration_hours, efficiency, capture_rate: BESS params
            for revenue scaling.

    Returns:
        DataFrame with one row per forward contract containing columns
        ``[contract, delivery_start, delivery_end, days_in_period,
        forward_base, avg_daily_spread, period_revenue_eur,
        annualised_revenue_eur_per_mw]``.
    """
    if daily_spreads.empty:
        return pd.DataFrame(
            columns=[
                "contract", "delivery_start", "delivery_end", "days_in_period",
                "forward_base", "avg_daily_spread", "period_revenue_eur",
                "annualised_revenue_eur_per_mw",
            ],
        )

    zone_forwards = forward_df[forward_df["zone"] == zone].reset_index(drop=True)
    daily = daily_spreads.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    energy_mwh = power_mw * duration_hours
    rows: list[dict] = []
    for _, row in zone_forwards.iterrows():
        start = pd.Timestamp(row["delivery_start"])
        end = pd.Timestamp(row["delivery_end"])
        mask = (daily["date"] >= start) & (daily["date"] < end)
        period = daily.loc[mask]
        n_days = len(period)
        if n_days == 0:
            continue
        # LP revenue path wins when available; otherwise greedy spread * energy.
        if "lp_revenue" in period.columns and period["lp_revenue"].notna().any():
            per_day = period["lp_revenue"].mean() * capture_rate
        else:
            per_day = (
                period["spread"].mean()
                * energy_mwh
                * (efficiency ** 0.5)
                * capture_rate
            )
        period_revenue = per_day * n_days
        annualised_per_mw = per_day * 365.25 / power_mw if power_mw > 0 else 0.0
        rows.append({
            "contract": row.get("contract") or "",
            "delivery_start": start,
            "delivery_end": end,
            "days_in_period": n_days,
            "forward_base": float(row["price_eur_mwh"]),
            "avg_daily_spread": float(period["spread"].mean()) if "spread" in period.columns else float("nan"),
            "period_revenue_eur": round(period_revenue, 2),
            "annualised_revenue_eur_per_mw": round(annualised_per_mw, 2),
        })

    return pd.DataFrame(rows)


def list_supported_zones(forward_df: pd.DataFrame) -> Iterable[str]:
    """Convenience: zones present in the uploaded forward curve."""
    return sorted(forward_df["zone"].unique().tolist())
