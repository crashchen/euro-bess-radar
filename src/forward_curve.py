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
    # Strip only FULL-line comments (optional leading whitespace + '#').
    # pandas' built-in `comment="#"` would otherwise truncate any field
    # value containing '#' (e.g. a `source` cell "Desk #1" -> "Desk ").
    no_comments = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    df = pd.read_csv(io.StringIO(no_comments))
    df.columns = [c.strip().lower() for c in df.columns]
    # Preserve user-supplied row order before any further sorting — used by
    # build_forward_synthetic_prices to honour the "later-in-CSV wins on
    # overlap" contract.
    df["_csv_order"] = range(len(df))

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

    # peak/offpeak rows are stored but the v1 synthetic-price path only
    # supports baseload. Surface a warning so users don't think a "peak"
    # price was applied to the peak hours specifically.
    non_base = df[df["shape"] != "base"]
    if not non_base.empty:
        logger.warning(
            "Forward CSV contains %d row(s) with shape != 'base'. "
            "v1 forward-scenario engine treats every row as baseload; "
            "peak/offpeak shaping is not yet implemented and these rows "
            "will be applied at their listed price as if they were base.",
            len(non_base),
        )

    return df[[
        "zone", "contract",
        "delivery_start", "delivery_end",
        "price_eur_mwh", "shape", "source", "as_of",
        "_csv_order",
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
    covered = int(by_bucket.shape[0])
    if covered < 168:
        # A short history (e.g. 3 days = 72 buckets) silently filled the
        # missing buckets with factor=1.0 in earlier versions. That made
        # the synthetic curve under-spread for buckets the history never
        # saw. Warn so the user knows the shape is partial.
        logger.warning(
            "Historical reference covers only %d/168 hour-of-week buckets; "
            "missing buckets fall back to factor 1.0 (flat). Fetch a "
            "longer history (>=1 week) for a fully-recovered shape.",
            covered,
        )
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
    for _, row in zone_forwards.iterrows():
        start = pd.Timestamp(row["delivery_start"]).tz_localize(target_tz)
        end = pd.Timestamp(row["delivery_end"]).tz_localize(target_tz)
        if end <= start:
            continue
        idx = pd.date_range(start=start, end=end, freq="h", inclusive="left")
        if len(idx) == 0:
            continue
        local_bucket = idx.weekday * 24 + idx.hour
        raw_factors = shape.reindex(local_bucket).to_numpy()
        # Per-contract renormalisation: the formula's invariant is
        # mean(hourly_price) == forward_base over the CONTRACT window.
        # The global mean-1 normalisation alone only guarantees that
        # invariant when the window spans full weeks; a 24h or weekend
        # contract otherwise comes in above or below its quoted base.
        window_mean = float(raw_factors.mean())
        factors = raw_factors / window_mean if window_mean > 0 else raw_factors
        prices = float(row["price_eur_mwh"]) * factors
        # Preserve the user's CSV row order as overlap priority so the
        # later-in-CSV-wins contract holds even after parse_forward_csv
        # has sorted rows by (zone, delivery_start).
        priority = int(row.get("_csv_order", 0))
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
    # ``_priority`` is preserved on the output so downstream code
    # (summarise_forward_revenue) can break ties consistently with this
    # function's dedup rule when a local day is split between two
    # contracts at exactly equal hours.
    out = out.set_index("timestamp")
    return out


_SUMMARY_COLUMNS = [
    "contract", "delivery_start", "delivery_end", "days_in_period",
    "forward_base", "avg_daily_spread", "period_revenue_eur",
    "annualised_revenue_eur_per_mw",
]


def summarise_forward_revenue(
    daily_spreads: pd.DataFrame,
    forward_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    *,
    zone: str,
    power_mw: float,
    duration_hours: float,
    efficiency: float = 0.88,
    capture_rate: float = 0.70,
    tz: str | None = None,
) -> pd.DataFrame:
    """Aggregate forward dispatch results to per-contract revenue.

    Attributes each local-tz delivery day to the contract that actually
    won that day's hours in :func:`build_forward_synthetic_prices` (i.e.
    the overlap-priority winner), then aggregates per contract. This
    differs from a naive ``[delivery_start, delivery_end)`` mask, which
    double-counts days when two contracts overlap (e.g. a Cal-2027 and a
    Jan-2027 quote both report the same Jan days in their period totals
    even though the synthetic price series only realised the Jan winner).

    Args:
        daily_spreads: Output of ``calculate_daily_spreads`` or
            ``calculate_daily_dispatch`` on the synthetic forward prices.
        forward_df: The original parsed forward DataFrame.
        synthetic_df: Output of ``build_forward_synthetic_prices`` for
            the same zone — required so each day is attributed to the
            single winning contract on overlap, matching the dispatch
            actually run on the synthetic series.
        zone: Zone code to filter forward contracts on.
        power_mw, duration_hours, efficiency, capture_rate: BESS params
            for revenue scaling.
        tz: IANA timezone for resolving synthetic timestamps to local
            calendar dates. Must match the tz used to compute
            ``daily_spreads`` for the dates to align.

    Returns:
        DataFrame with one row per forward contract THAT CLAIMED AT
        LEAST ONE DAY in the synthetic series, containing columns
        ``[contract, delivery_start, delivery_end, days_in_period,
        forward_base, avg_daily_spread, period_revenue_eur,
        annualised_revenue_eur_per_mw]``. The sum of ``days_in_period``
        over the returned rows equals the number of unique calendar
        days covered by the synthetic series — i.e. overlap days are
        attributed to exactly one contract.
    """
    if daily_spreads.empty or synthetic_df is None or synthetic_df.empty:
        return pd.DataFrame(columns=_SUMMARY_COLUMNS)

    # Use the CSV row order (_csv_order) as the stable row id. ``contract``
    # is a user-facing label that may be blank or repeat across rows —
    # joining on it would silently merge distinct contracts, so we key
    # the attribution on ``_csv_order`` and only carry the label for
    # display.
    if "_csv_order" not in forward_df.columns:
        forward_df = forward_df.copy()
        forward_df["_csv_order"] = range(len(forward_df))
    zone_forwards = forward_df[forward_df["zone"] == zone].copy()
    if zone_forwards.empty:
        return pd.DataFrame(columns=_SUMMARY_COLUMNS)

    # Map every local-tz date in the synthetic series to the contract ROW
    # that actually held it. The synthetic series is already overlap-
    # resolved (one winner per timestamp), so a per-date majority vote
    # attributes the day to whichever row owned most of its hours. On
    # exact-hour ties (rare, but possible when two contracts split a day
    # evenly), the higher ``_priority`` (later in CSV) wins — same rule
    # build_forward_synthetic_prices applies at dedup.
    synth_local_idx = synthetic_df.index
    if tz is not None and synth_local_idx.tz is not None:
        synth_local_idx = synth_local_idx.tz_convert(tz)
    if "_priority" in synthetic_df.columns:
        priority_per_row = synthetic_df["_priority"].to_numpy()
    else:
        # Legacy path: synthetic frame without _priority. Fall back to
        # contract label as the (unstable) row id. This branch is only
        # reached by test/external callers that construct a synth by hand.
        priority_per_row = pd.Categorical(synthetic_df["contract"]).codes

    synth_view = pd.DataFrame({
        "_row": priority_per_row,
    }, index=pd.DatetimeIndex(synth_local_idx).date)

    def _attribute_day(group: pd.DataFrame) -> int:
        counts = group["_row"].value_counts()
        top = counts.iloc[0]
        tied = counts[counts == top].index
        if len(tied) == 1:
            return int(tied[0])
        # Among tied rows, pick the one with the highest priority value
        # (== latest CSV order, matching the dedup rule).
        return int(max(tied))

    row_per_date = synth_view.groupby(level=0).apply(_attribute_day)

    daily = daily_spreads.copy()
    daily["_date"] = pd.to_datetime(daily["date"]).dt.date
    daily["_row"] = daily["_date"].map(row_per_date)
    # Days that fall inside the union of declared delivery windows but
    # missed the synthetic mapping are a quiet signal that something
    # upstream (tz mismatch, partial history coverage) excluded them.
    # Surface a warning instead of silently zeroing them.
    unmapped = daily.loc[daily["_row"].isna(), "_date"]
    if not unmapped.empty:
        bounds = zone_forwards.assign(
            _start=pd.to_datetime(zone_forwards["delivery_start"]).dt.date,
            _end=pd.to_datetime(zone_forwards["delivery_end"]).dt.date,
        )
        in_bounds = unmapped.apply(
            lambda d: bool(
                ((bounds["_start"] <= d) & (d < bounds["_end"])).any()
            )
        )
        n_in_bounds = int(in_bounds.sum())
        if n_in_bounds:
            logger.warning(
                "Forward summary dropped %d day(s) inside a declared "
                "delivery window for %s — likely a tz mismatch between "
                "daily_spreads and the synthetic frame.",
                n_in_bounds, zone,
            )
    daily = daily.dropna(subset=["_row"])
    if daily.empty:
        return pd.DataFrame(columns=_SUMMARY_COLUMNS)
    daily["_row"] = daily["_row"].astype(int)

    energy_mwh = power_mw * duration_hours
    rows: list[dict] = []
    for _, row in zone_forwards.iterrows():
        row_id = int(row["_csv_order"])
        contract_label = row.get("contract") or ""
        period = daily.loc[daily["_row"] == row_id]
        if period.empty:
            continue
        # LP revenue path wins when available. Drop NaN LP days from
        # BOTH the mean and the day count so a single NaN doesn't
        # inflate the period revenue (mean ignores NaN but multiplying
        # by raw n_days would still scale a partial-coverage mean to
        # a full-period total).
        use_lp = (
            "lp_revenue" in period.columns
            and period["lp_revenue"].notna().any()
        )
        if use_lp:
            lp_clean = period["lp_revenue"].dropna()
            per_day = float(lp_clean.mean()) * capture_rate
            valid_days = len(lp_clean)
        else:
            spread_clean = (
                period["spread"].dropna()
                if "spread" in period.columns else pd.Series(dtype=float)
            )
            per_day = (
                float(spread_clean.mean())
                * energy_mwh
                * (efficiency ** 0.5)
                * capture_rate
            ) if not spread_clean.empty else 0.0
            valid_days = len(spread_clean)
        if valid_days == 0:
            continue
        period_revenue = per_day * valid_days
        annualised_per_mw = per_day * 365.25 / power_mw if power_mw > 0 else 0.0
        rows.append({
            "contract": contract_label,
            "delivery_start": pd.Timestamp(row["delivery_start"]),
            "delivery_end": pd.Timestamp(row["delivery_end"]),
            "days_in_period": valid_days,
            "forward_base": float(row["price_eur_mwh"]),
            "avg_daily_spread": (
                float(period["spread"].mean())
                if "spread" in period.columns else float("nan")
            ),
            "period_revenue_eur": round(period_revenue, 2),
            "annualised_revenue_eur_per_mw": round(annualised_per_mw, 2),
        })

    return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)


def list_supported_zones(forward_df: pd.DataFrame) -> Iterable[str]:
    """Convenience: zones present in the uploaded forward curve."""
    return sorted(forward_df["zone"].unique().tolist())
