"""Ancillary services data handling: upload, parse, and integrate."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

import pandas as pd

from src.config import (
    ANCILLARY_CAPACITY_AVAILABILITY,
    ANCILLARY_ENERGY_ACTIVATION_SHARE,
    HOURS_PER_YEAR,
)

logger = logging.getLogger(__name__)

_STANDARD_COLUMNS = [
    "capacity_price_eur_mw",
    "energy_price_eur_mwh",
    "energy_price_up_eur_mwh",
    "energy_price_down_eur_mwh",
    "system_buy_price_eur_mwh",
    "system_sell_price_eur_mwh",
    "product_type",
    "direction",
    "zone",
]
PRODUCT_ALIASES: dict[str, set[str]] = {
    "FCR-D": {"FCR-D Up", "FCR-D Down"},
}

# ── Templates ────────────────────────────────────────────────────────────────

ANCILLARY_TEMPLATES: dict[str, dict] = {
    "DE_FCR": {
        "description": "Germany FCR auction results from regelleistung.net",
        "source_url": "https://www.regelleistung.net/apps/datacenter/tendering-files/",
        "expected_columns": ["date", "product", "capacity_price_eur_mw"],
        "resolution": "4h blocks",
    },
    "DE_aFRR": {
        "description": "Germany aFRR capacity auction results",
        "source_url": "https://www.regelleistung.net/apps/datacenter/tendering-files/",
        "expected_columns": ["date", "product", "direction", "capacity_price_eur_mw"],
        "resolution": "4h blocks",
    },
    "RO_BALANCING": {
        "description": "Romania Transelectrica balancing market daily reports",
        "source_url": "https://www.transelectrica.ro/en/web/tel/echilibrare-si-sts",
        "expected_columns": ["date", "hour", "marginal_price_up", "marginal_price_down"],
        "resolution": "hourly or 15min",
    },
    "FI_FCR": {
        "description": "Finland Fingrid reserve market data",
        "source_url": "https://data.fingrid.fi/en/datasets",
        "expected_columns": ["date", "hour", "fcr_n_price", "fcr_d_price"],
        "resolution": "hourly",
    },
    "GB_BALANCING": {
        "description": "GB system prices and NIV from Elexon (manual upload prices in GBP/MWh)",
        "source_url": "https://bmrs.elexon.co.uk/",
        "expected_columns": [
            "settlement_date", "settlement_period",
            "system_buy_price", "system_sell_price",
        ],
        "resolution": "30min settlement periods",
    },
}


def generate_template_csv(template_key: str) -> str:
    """Generate a minimal CSV string with correct headers and example rows.

    Args:
        template_key: Key from ANCILLARY_TEMPLATES.

    Returns:
        CSV content as a string.
    """
    tmpl = ANCILLARY_TEMPLATES[template_key]
    cols = tmpl["expected_columns"]

    buf = io.StringIO()
    if template_key == "GB_BALANCING":
        buf.write(
            "# system_buy_price and system_sell_price must be uploaded in GBP/MWh; "
            "year-specific GBP->EUR conversion is applied automatically.\n"
        )
    writer = csv.writer(buf)
    writer.writerow(cols)

    # Write 2 example rows
    for i in range(2):
        row = []
        for col in cols:
            if "date" in col:
                row.append(f"2025-01-{1 + i:02d}")
            elif "hour" in col or "period" in col:
                if template_key == "FI_FCR" and col == "hour":
                    row.append(str(i))
                else:
                    row.append(str(i + 1))
            elif "price" in col:
                row.append(f"{10.0 + i * 5:.2f}")
            elif "product" in col:
                row.append("POS" if template_key.endswith("FCR") else "4h_block_1")
            elif "direction" in col:
                row.append("UP" if i == 0 else "DOWN")
            else:
                row.append("")
        writer.writerow(row)

    return buf.getvalue()


# ── Parsing ──────────────────────────────────────────────────────────────────

def _detect_delimiter(content: str) -> str:
    """Auto-detect CSV delimiter."""
    first_line = next(
        (
            line
            for line in content.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
        "",
    )
    for delim in [",", ";", "\t"]:
        if delim in first_line:
            return delim
    return ","


def _empty_ancillary_frame() -> pd.DataFrame:
    """Return an empty standardised ancillary frame with a timestamp index."""
    return pd.DataFrame(columns=_STANDARD_COLUMNS).rename_axis("timestamp")


def _parse_date_hour_index(
    dates,
    hours,
    *,
    template_key: str | None = None,
) -> pd.DatetimeIndex:
    """Convert date/hour columns to a UTC timestamp index."""
    hour_strings = pd.Series(hours).astype(str).str.strip()
    if template_key == "FI_FCR":
        numeric_hours = pd.to_numeric(hour_strings, errors="coerce")
        invalid = numeric_hours[
            numeric_hours.isna()
            | (numeric_hours < 0)
            | (numeric_hours > 23)
            | (numeric_hours % 1 != 0)
        ]
        if not invalid.empty:
            raise ValueError("FI_FCR hour values must be integers between 0 and 23.")
        hour_strings = numeric_hours.astype(int).astype(str)

    return pd.to_datetime(
        pd.Series(dates).astype(str) + " " + hour_strings.str.zfill(2) + ":00",
        utc=True,
    )


def _initialise_output(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create a standardised ancillary frame for the provided index."""
    out = pd.DataFrame(index=index)
    out["capacity_price_eur_mw"] = float("nan")
    out["energy_price_eur_mwh"] = float("nan")
    out["energy_price_up_eur_mwh"] = float("nan")
    out["energy_price_down_eur_mwh"] = float("nan")
    out["system_buy_price_eur_mwh"] = float("nan")
    out["system_sell_price_eur_mwh"] = float("nan")
    out["product_type"] = ""
    out["direction"] = ""
    out["zone"] = ""
    out.index.name = "timestamp"
    return out


def _coerce_numeric_array(values) -> list[float] | float:
    """Convert scalar/array-like values to numeric data suitable for assignment."""
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").to_list()
    return values


def _coerce_text_array(
    values,
    index: pd.DatetimeIndex,
    *,
    default: str = "",
) -> list[str]:
    """Broadcast text-like values to match the provided index length."""
    if isinstance(values, pd.Series):
        series = values.reset_index(drop=True)
    elif isinstance(values, pd.Index):
        series = pd.Series(values)
    elif isinstance(values, (list, tuple)):
        series = pd.Series(list(values))
    else:
        value = default if values is None or pd.isna(values) else str(values)
        return [value] * len(index)

    if len(series) == 0:
        return [default] * len(index)
    if len(series) == 1 and len(index) != 1:
        value = default if pd.isna(series.iloc[0]) else str(series.iloc[0])
        return [value] * len(index)
    if len(series) != len(index):
        raise ValueError(
            f"Cannot align ancillary text column of length {len(series)} "
            f"to index of length {len(index)}"
        )
    return series.fillna(default).astype(str).to_list()


def _canonical_product_label(
    base: str | None,
    direction: str | None = None,
    template_hint: str | None = None,
) -> str:
    """Map source-specific product labels into stable dashboard product names."""
    tokens = " ".join(
        part for part in [template_hint or "", base or ""] if part
    ).upper()

    if "FCR_N" in tokens or "FCR-N" in tokens:
        label = "FCR-N"
    elif "FCR_D" in tokens or "FCR-D" in tokens:
        label = "FCR-D"
    elif "AFRR" in tokens or "AFR" in tokens:
        label = "aFRR"
    elif "MFRR" in tokens:
        label = "mFRR"
    elif "IMBALANCE" in tokens:
        label = "Imbalance"
    elif "BALANCING" in tokens:
        label = "Balancing"
    elif "FCR" in tokens or "POS" in tokens:
        label = "FCR"
    else:
        label = str(base or template_hint or "Unknown")

    dir_upper = str(direction or "").strip().upper()
    if dir_upper in {"UP", "LONG"} and not label.upper().endswith(" UP"):
        return f"{label} Up"
    if dir_upper in {"DOWN", "SHORT"} and not label.upper().endswith(" DOWN"):
        return f"{label} Down"
    return label


def _service_bucket(product: str) -> str:
    """Map a product label to the output revenue bucket."""
    product_upper = str(product).upper()
    if "AFRR" in product_upper or "AFR" in product_upper:
        return "afrr_annual_eur"
    if "MFRR" in product_upper or "BALANC" in product_upper or "IMBALANCE" in product_upper:
        return "mfrr_annual_eur"
    return "fcr_annual_eur"


def _timestamp_index_from_frame(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Best-effort conversion of a raw ancillary frame into a timestamp index."""
    if isinstance(df.index, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(df.index)
        return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], utc=True)

    if "date" in df.columns:
        if "hour" in df.columns:
            return _parse_date_hour_index(df["date"], df["hour"])
        return pd.to_datetime(df["date"], utc=True)

    if "settlement_date" in df.columns and "settlement_period" in df.columns:
        hour_min = ((df["settlement_period"].astype(int) - 1) * 30).apply(
            lambda m: f"{m // 60:02d}:{m % 60:02d}"
        )
        return pd.to_datetime(
            df["settlement_date"].astype(str) + " " + hour_min,
            utc=True,
        )

    return pd.DatetimeIndex([], tz="UTC", name="timestamp")


def _build_standard_frame(
    index: pd.DatetimeIndex,
    product_type,
    zone,
    *,
    direction="",
    capacity=None,
    energy=None,
    energy_up=None,
    energy_down=None,
    system_buy=None,
    system_sell=None,
) -> pd.DataFrame:
    """Build a standard ancillary frame for one product stream."""
    out = _initialise_output(index)
    out["product_type"] = _coerce_text_array(product_type, index, default="Unknown")
    out["direction"] = _coerce_text_array(direction, index, default="")
    out["zone"] = _coerce_text_array(zone, index, default="")

    if capacity is not None:
        out["capacity_price_eur_mw"] = _coerce_numeric_array(capacity)
    if energy is not None:
        out["energy_price_eur_mwh"] = _coerce_numeric_array(energy)
    if energy_up is not None:
        out["energy_price_up_eur_mwh"] = _coerce_numeric_array(energy_up)
    if energy_down is not None:
        out["energy_price_down_eur_mwh"] = _coerce_numeric_array(energy_down)
    if system_buy is not None:
        out["system_buy_price_eur_mwh"] = _coerce_numeric_array(system_buy)
    if system_sell is not None:
        out["system_sell_price_eur_mwh"] = _coerce_numeric_array(system_sell)
    return out


def normalize_auto_fetch_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Map an auto-fetched ancillary dataset into the standard ancillary schema."""
    if df.empty:
        return _empty_ancillary_frame()

    raw = df.copy()
    idx = _timestamp_index_from_frame(raw)
    if idx.empty:
        return _empty_ancillary_frame()

    zone = raw["zone"] if "zone" in raw.columns else ""
    frames: list[pd.DataFrame] = []

    if "capacity_price_eur_mw" in raw.columns:
        labels = _coerce_text_array(
            raw["product"] if "product" in raw.columns else dataset_name,
            idx,
            default=dataset_name,
        )
        directions = _coerce_text_array(
            raw["direction"] if "direction" in raw.columns else "",
            idx,
            default="",
        )
        canonical = [
            _canonical_product_label(label, direction, dataset_name)
            for label, direction in zip(labels, directions)
        ]
        frames.append(_build_standard_frame(
            idx,
            canonical,
            zone,
            direction=directions,
            capacity=raw["capacity_price_eur_mw"],
        ))

    product_columns = {
        "fcr_n_price": ("FCR-N", ""),
        "fcr_d_price": ("FCR-D", ""),
        "fcr_d_up_price": ("FCR-D Up", "Up"),
        "fcr_d_down_price": ("FCR-D Down", "Down"),
        "afrr_up_price": ("aFRR Up", "Up"),
        "afrr_down_price": ("aFRR Down", "Down"),
    }
    for col, (product_label, direction) in product_columns.items():
        if col in raw.columns:
            frames.append(_build_standard_frame(
                idx,
                product_label,
                zone,
                direction=direction,
                capacity=raw[col],
            ))

    if "energy_price_eur_mwh" in raw.columns:
        frames.append(_build_standard_frame(
            idx,
            _canonical_product_label(dataset_name),
            zone,
            energy=raw["energy_price_eur_mwh"],
        ))
    elif any(
        col in raw.columns
        for col in [
            "system_buy_price_eur", "system_sell_price_eur",
            "imbalance_price_long", "imbalance_price_short",
        ]
    ):
        frames.append(_build_standard_frame(
            idx,
            _canonical_product_label(dataset_name),
            zone,
            energy_up=raw["imbalance_price_long"] if "imbalance_price_long" in raw.columns else None,
            energy_down=raw["imbalance_price_short"] if "imbalance_price_short" in raw.columns else None,
            system_buy=raw["system_buy_price_eur"] if "system_buy_price_eur" in raw.columns else None,
            system_sell=raw["system_sell_price_eur"] if "system_sell_price_eur" in raw.columns else None,
        ))

    if not frames:
        return _empty_ancillary_frame()

    combined = pd.concat(frames).sort_index()
    combined.index.name = "timestamp"
    return combined


def build_ancillary_dataset(
    manual_df: pd.DataFrame | None = None,
    auto_fetch_results: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Resolve the ancillary dataset used for valuation.

    Manual uploads override same-name auto-fetched products, but all other
    auto-fetched products are retained.
    """
    frames = []
    for dataset_name, df in (auto_fetch_results or {}).items():
        normalised = normalize_auto_fetch_dataset(df, dataset_name)
        if not normalised.empty:
            frames.append(normalised)

    combined = pd.concat(frames).sort_index() if frames else _empty_ancillary_frame()
    if manual_df is not None and not manual_df.empty:
        manual = manual_df.sort_index()
        manual_products = set(
            manual["product_type"].dropna().astype(str).str.strip()
        )
        expanded_manual_products = manual_products | set().union(
            *(PRODUCT_ALIASES.get(product, set()) for product in manual_products)
        )
        if not combined.empty and expanded_manual_products:
            combined = combined[
                ~combined["product_type"].astype(str).str.strip().isin(expanded_manual_products)
            ]
        combined = pd.concat([combined, manual]).sort_index()

    combined.index.name = "timestamp"
    return combined


def parse_ancillary_csv(
    csv_content: str | Path,
    template_key: str,
) -> pd.DataFrame:
    """Parse an uploaded ancillary services CSV using the appropriate template.

    Args:
        csv_content: CSV content as string or path to file.
        template_key: Key from ANCILLARY_TEMPLATES.

    Returns:
        DataFrame with standardised columns:
        [timestamp, product_type, direction, capacity_price_eur_mw,
         energy_price_eur_mwh, zone] plus preserved directional/system-price
         columns where available.
    """
    if template_key not in ANCILLARY_TEMPLATES:
        raise ValueError(f"Unknown template: {template_key}")

    if isinstance(csv_content, Path):
        text = csv_content.read_text(encoding="utf-8-sig")
    else:
        text = csv_content

    delim = _detect_delimiter(text)
    df = pd.read_csv(io.StringIO(text), sep=delim, comment="#")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Parse timestamp
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    if "date" in df.columns:
        if "hour" in df.columns:
            idx = _parse_date_hour_index(df["date"], df["hour"], template_key=template_key)
        else:
            idx = pd.to_datetime(df["date"], utc=True)
    elif "settlement_date" in df.columns:
        # GB: 30-min periods, period 1 = 00:00
        df["hour_min"] = ((df["settlement_period"].astype(int) - 1) * 30).apply(
            lambda m: f"{m // 60:02d}:{m % 60:02d}"
        )
        idx = pd.to_datetime(
            df["settlement_date"].astype(str) + " " + df["hour_min"], utc=True,
        )

    zone = template_key.split("_")[0]
    frames: list[pd.DataFrame] = []

    if "capacity_price_eur_mw" in df.columns:
        product_series = _coerce_text_array(
            df["product"] if "product" in df.columns else template_key,
            idx,
            default=template_key,
        )
        direction_series = _coerce_text_array(
            df["direction"] if "direction" in df.columns else "",
            idx,
            default="",
        )
        canonical = [
            _canonical_product_label(product, direction, template_key)
            for product, direction in zip(product_series, direction_series)
        ]
        frames.append(_build_standard_frame(
            idx,
            canonical,
            zone,
            direction=direction_series,
            capacity=df["capacity_price_eur_mw"],
        ))
    else:
        if "fcr_n_price" in df.columns:
            frames.append(_build_standard_frame(idx, "FCR-N", zone, capacity=df["fcr_n_price"]))
        if "fcr_d_price" in df.columns:
            frames.append(_build_standard_frame(idx, "FCR-D", zone, capacity=df["fcr_d_price"]))

    if "energy_price_eur_mwh" in df.columns:
        frames.append(_build_standard_frame(
            idx,
            _canonical_product_label(template_key),
            zone,
            energy=df["energy_price_eur_mwh"],
        ))
    elif any(
        col in df.columns
        for col in ["marginal_price_up", "marginal_price_down", "system_buy_price", "system_sell_price"]
    ):
        system_buy = df["system_buy_price"] if "system_buy_price" in df.columns else None
        system_sell = df["system_sell_price"] if "system_sell_price" in df.columns else None
        if template_key == "GB_BALANCING":
            from src.data_ingestion import _convert_gbp_series_to_eur

            if system_buy is not None:
                system_buy = _convert_gbp_series_to_eur(
                    pd.to_numeric(system_buy, errors="coerce"),
                    idx,
                )
            if system_sell is not None:
                system_sell = _convert_gbp_series_to_eur(
                    pd.to_numeric(system_sell, errors="coerce"),
                    idx,
                )
        frames.append(_build_standard_frame(
            idx,
            _canonical_product_label(template_key),
            zone,
            energy_up=df["marginal_price_up"] if "marginal_price_up" in df.columns else None,
            energy_down=df["marginal_price_down"] if "marginal_price_down" in df.columns else None,
            system_buy=system_buy,
            system_sell=system_sell,
        ))

    if not frames:
        return _empty_ancillary_frame()

    out = pd.concat(frames).sort_index()
    out.index.name = "timestamp"
    logger.info("Parsed %d rows from %s template", len(out), template_key)
    return out


# ── Revenue estimation ───────────────────────────────────────────────────────

def _infer_capacity_duration_hours(
    cap_prices: pd.Series,
    product: str,
) -> pd.Series | None:
    """Infer per-row capacity durations in hours from timestamp spacing."""
    if len(cap_prices) < 2 or not isinstance(cap_prices.index, pd.DatetimeIndex):
        return None

    ordered = cap_prices.sort_index()
    deltas = ordered.index.to_series().diff()
    positive = deltas.dropna()
    if positive.empty or (positive <= pd.Timedelta(0)).any():
        logger.debug(
            "Falling back to unweighted capacity mean for %s: irregular timestamp spacing",
            product,
        )
        return None

    durations = deltas.ffill().bfill()
    if durations.isna().any() or (durations <= pd.Timedelta(0)).any():
        logger.debug(
            "Falling back to unweighted capacity mean for %s: could not infer durations",
            product,
        )
        return None

    return durations.dt.total_seconds() / 3600.0


def _capacity_price_mean(cap_prices: pd.Series, product: str) -> float:
    """Return an inferred-duration-weighted capacity mean where possible."""
    weights = _infer_capacity_duration_hours(cap_prices, product)
    if weights is None:
        return float(cap_prices.mean())

    ordered = cap_prices.sort_index()
    weighted_total = float((ordered * weights).sum())
    total_duration = float(weights.sum())
    if total_duration <= 0:
        logger.debug(
            "Falling back to unweighted capacity mean for %s: non-positive duration sum",
            product,
        )
        return float(ordered.mean())
    return weighted_total / total_duration


def calculate_ancillary_revenue(
    ancillary_df: pd.DataFrame,
    power_mw: float = 1.0,
    duration_hours: float = 1.0,
) -> dict[str, float]:
    """Estimate annual ancillary service revenue from uploaded data.

    Args:
        ancillary_df: Parsed ancillary DataFrame.
        power_mw: BESS power in MW.
        duration_hours: BESS duration in hours.

    Returns:
        Dict with per-service and total annual revenue estimates.
    """
    availability = ANCILLARY_CAPACITY_AVAILABILITY
    result: dict[str, float] = {
        "fcr_annual_eur": 0.0,
        "afrr_annual_eur": 0.0,
        "mfrr_annual_eur": 0.0,
        "total_ancillary_eur": 0.0,
        "total_ancillary_per_mw": 0.0,
        "product_revenues": {},
    }

    if ancillary_df.empty:
        return result

    product_revenues: dict[str, float] = {}
    grouped = ancillary_df.groupby(
        ancillary_df["product_type"].fillna("UNKNOWN").astype(str).str.strip()
    )
    for product, group in grouped:
        annual_revenue = 0.0

        cap_prices = group["capacity_price_eur_mw"].dropna()
        if not cap_prices.empty:
            avg_cap = _capacity_price_mean(cap_prices, product)
            annual_revenue += avg_cap * power_mw * HOURS_PER_YEAR * availability

        energy_prices = group["energy_price_eur_mwh"].dropna()
        if not energy_prices.empty:
            avg_energy = float(energy_prices.mean())
            # Annualise explicit single-sided energy prices with the configured
            # screening assumption for activated hours. Two-sided balancing
            # signals are preserved separately and are not auto-monetised here.
            activation_hours = HOURS_PER_YEAR * ANCILLARY_ENERGY_ACTIVATION_SHARE
            annual_revenue += avg_energy * power_mw * activation_hours

        if annual_revenue <= 0:
            continue

        annual_revenue = round(annual_revenue, 2)
        product_revenues[product] = annual_revenue
        bucket = _service_bucket(product)
        result[bucket] += annual_revenue

    result["total_ancillary_eur"] = round(
        result["fcr_annual_eur"] + result["afrr_annual_eur"] + result["mfrr_annual_eur"], 2
    )
    result["total_ancillary_per_mw"] = round(result["total_ancillary_eur"] / power_mw, 2)
    result["product_revenues"] = dict(sorted(product_revenues.items()))
    return result


def merge_revenue_stack(
    da_revenue: dict,
    ancillary_revenue: dict,
    power_mw: float = 1.0,
) -> dict:
    """Combine DA arbitrage and ancillary service revenues into total stack.

    Args:
        da_revenue: Dict from estimate_annual_arbitrage_revenue().
        ancillary_revenue: Dict from calculate_ancillary_revenue().
        power_mw: Reference BESS power rating in MW used for per-MW
            normalisation of the combined revenue stack.

    Returns:
        Combined revenue stack dict.
    """
    da_eur = da_revenue.get("annual_revenue_eur", 0.0)
    fcr = ancillary_revenue.get("fcr_annual_eur", 0.0)
    afrr = ancillary_revenue.get("afrr_annual_eur", 0.0)
    mfrr = ancillary_revenue.get("mfrr_annual_eur", 0.0)
    product_revenues = ancillary_revenue.get("product_revenues", {})
    total = da_eur + fcr + afrr + mfrr

    source_revenues = {"DA Arbitrage": round(da_eur, 2)}
    for product, value in product_revenues.items():
        source_revenues[product] = round(float(value), 2)

    return {
        "da_arbitrage_eur": round(da_eur, 2),
        "fcr_eur": round(fcr, 2),
        "afrr_eur": round(afrr, 2),
        "mfrr_eur": round(mfrr, 2),
        "product_revenues": product_revenues,
        "source_revenues": source_revenues,
        "total_eur": round(total, 2),
        "total_per_mw": round(total / power_mw, 2) if power_mw > 0 else 0.0,
        "da_pct": round(100.0 * da_eur / total, 1) if total > 0 else 0.0,
        "ancillary_pct": round(100.0 * (total - da_eur) / total, 1) if total > 0 else 0.0,
    }
