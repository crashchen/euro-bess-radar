"""Ancillary services data handling: upload, parse, and integrate."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

import pandas as pd

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
        "description": "GB system prices and NIV from Elexon",
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
    writer = csv.writer(buf)
    writer.writerow(cols)

    # Write 2 example rows
    for i in range(2):
        row = []
        for col in cols:
            if "date" in col:
                row.append(f"2025-01-{1 + i:02d}")
            elif "hour" in col or "period" in col:
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
    for delim in [",", ";", "\t"]:
        if delim in content.split("\n")[0]:
            return delim
    return ","


def _empty_ancillary_frame() -> pd.DataFrame:
    """Return an empty standardised ancillary frame with a timestamp index."""
    return pd.DataFrame(columns=_STANDARD_COLUMNS).rename_axis("timestamp")


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
            return pd.to_datetime(
                df["date"].astype(str) + " " + df["hour"].astype(str).str.zfill(2) + ":00",
                utc=True,
            )
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

    out = _initialise_output(idx)

    capacity_cols = [
        "capacity_price_eur_mw",
        "fcr_n_price",
        "fcr_d_up_price",
        "fcr_d_down_price",
        "afrr_up_price",
        "afrr_down_price",
        "fcr_d_price",
    ]
    capacity_series = [
        pd.to_numeric(raw[col], errors="coerce")
        for col in capacity_cols
        if col in raw.columns
    ]
    if capacity_series:
        out["capacity_price_eur_mw"] = (
            pd.concat(capacity_series, axis=1).mean(axis=1).to_numpy()
        )

    if "energy_price_eur_mwh" in raw.columns:
        out["energy_price_eur_mwh"] = pd.to_numeric(
            raw["energy_price_eur_mwh"], errors="coerce",
        ).to_numpy()
    if "system_buy_price_eur" in raw.columns:
        out["system_buy_price_eur_mwh"] = pd.to_numeric(
            raw["system_buy_price_eur"], errors="coerce",
        ).to_numpy()
    if "system_sell_price_eur" in raw.columns:
        out["system_sell_price_eur_mwh"] = pd.to_numeric(
            raw["system_sell_price_eur"], errors="coerce",
        ).to_numpy()
    if "imbalance_price_long" in raw.columns:
        out["energy_price_up_eur_mwh"] = pd.to_numeric(
            raw["imbalance_price_long"], errors="coerce",
        ).to_numpy()
    if "imbalance_price_short" in raw.columns:
        out["energy_price_down_eur_mwh"] = pd.to_numeric(
            raw["imbalance_price_short"], errors="coerce",
        ).to_numpy()

    product_type = raw.get("product", dataset_name)
    direction = raw.get("direction", "")
    zone = raw.get("zone", dataset_name.split("_")[0])

    out["product_type"] = (
        product_type.to_numpy() if isinstance(product_type, pd.Series) else product_type
    )
    out["direction"] = direction.to_numpy() if isinstance(direction, pd.Series) else direction
    out["zone"] = zone.to_numpy() if isinstance(zone, pd.Series) else zone

    return out.sort_index()


def build_ancillary_dataset(
    manual_df: pd.DataFrame | None = None,
    auto_fetch_results: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Resolve the ancillary dataset used for valuation.

    Manual uploads take precedence. If no manual upload is present, auto-fetched
    datasets are normalised and concatenated into a single standardised frame.
    """
    if manual_df is not None and not manual_df.empty:
        return manual_df.sort_index()

    frames = []
    for dataset_name, df in (auto_fetch_results or {}).items():
        normalised = normalize_auto_fetch_dataset(df, dataset_name)
        if not normalised.empty:
            frames.append(normalised)

    if not frames:
        return _empty_ancillary_frame()

    combined = pd.concat(frames).sort_index()
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
    df = pd.read_csv(io.StringIO(text), sep=delim)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Parse timestamp
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    if "date" in df.columns:
        if "hour" in df.columns:
            idx = pd.to_datetime(
                df["date"].astype(str) + " " + df["hour"].astype(str).str.zfill(2) + ":00",
                utc=True,
            )
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

    out = _initialise_output(idx)

    if "capacity_price_eur_mw" in df.columns:
        out["capacity_price_eur_mw"] = pd.to_numeric(
            df["capacity_price_eur_mw"], errors="coerce",
        ).to_numpy()
    else:
        fcr_prices = []
        for col in ["fcr_n_price", "fcr_d_price"]:
            if col in df.columns:
                fcr_prices.append(pd.to_numeric(df[col], errors="coerce"))
        if fcr_prices:
            out["capacity_price_eur_mw"] = (
                pd.concat(fcr_prices, axis=1).mean(axis=1).to_numpy()
            )

    if "energy_price_eur_mwh" in df.columns:
        out["energy_price_eur_mwh"] = pd.to_numeric(
            df["energy_price_eur_mwh"], errors="coerce",
        ).to_numpy()

    if "marginal_price_up" in df.columns:
        out["energy_price_up_eur_mwh"] = pd.to_numeric(
            df["marginal_price_up"], errors="coerce",
        ).to_numpy()
    if "marginal_price_down" in df.columns:
        out["energy_price_down_eur_mwh"] = pd.to_numeric(
            df["marginal_price_down"], errors="coerce",
        ).to_numpy()
    if "system_buy_price" in df.columns:
        out["system_buy_price_eur_mwh"] = pd.to_numeric(
            df["system_buy_price"], errors="coerce",
        ).to_numpy()
    if "system_sell_price" in df.columns:
        out["system_sell_price_eur_mwh"] = pd.to_numeric(
            df["system_sell_price"], errors="coerce",
        ).to_numpy()

    product_type = df.get("product", template_key)
    direction = df.get("direction", "")
    out["product_type"] = (
        product_type.to_numpy() if isinstance(product_type, pd.Series) else product_type
    )
    out["direction"] = direction.to_numpy() if isinstance(direction, pd.Series) else direction
    out["zone"] = template_key.split("_")[0]

    logger.info("Parsed %d rows from %s template", len(out), template_key)
    return out


# ── Revenue estimation ───────────────────────────────────────────────────────

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
    availability = 0.95
    result: dict[str, float] = {
        "fcr_annual_eur": 0.0,
        "afrr_annual_eur": 0.0,
        "mfrr_annual_eur": 0.0,
        "total_ancillary_eur": 0.0,
        "total_ancillary_per_mw": 0.0,
    }

    if ancillary_df.empty:
        return result

    cap_rows = ancillary_df[ancillary_df["capacity_price_eur_mw"].notna()].copy()
    if not cap_rows.empty:
        for product, group in cap_rows.groupby(cap_rows["product_type"].fillna("UNKNOWN")):
            avg_cap = float(group["capacity_price_eur_mw"].mean())
            hours_per_year = 8760
            cap_revenue = avg_cap * power_mw * hours_per_year * availability
            bucket = _service_bucket(str(product))
            result[bucket] += round(cap_revenue, 2)

    energy_rows = ancillary_df[ancillary_df["energy_price_eur_mwh"].notna()].copy()
    if not energy_rows.empty:
        for product, group in energy_rows.groupby(energy_rows["product_type"].fillna("UNKNOWN")):
            avg_energy = float(group["energy_price_eur_mwh"].mean())
            energy_mwh = power_mw * duration_hours
            # Assume balancing activations ~10% of hours only for explicit
            # single-price energy services. Two-sided balancing signals are preserved
            # separately and are not auto-monetised here.
            activation_hours = 8760 * 0.10
            energy_revenue = avg_energy * energy_mwh * activation_hours
            bucket = _service_bucket(str(product))
            result[bucket] += round(energy_revenue, 2)

    result["total_ancillary_eur"] = round(
        result["fcr_annual_eur"] + result["afrr_annual_eur"] + result["mfrr_annual_eur"], 2
    )
    result["total_ancillary_per_mw"] = round(result["total_ancillary_eur"] / power_mw, 2)
    return result


def merge_revenue_stack(
    da_revenue: dict,
    ancillary_revenue: dict,
) -> dict:
    """Combine DA arbitrage and ancillary service revenues into total stack.

    Args:
        da_revenue: Dict from estimate_annual_arbitrage_revenue().
        ancillary_revenue: Dict from calculate_ancillary_revenue().

    Returns:
        Combined revenue stack dict.
    """
    da_eur = da_revenue.get("annual_revenue_eur", 0.0)
    fcr = ancillary_revenue.get("fcr_annual_eur", 0.0)
    afrr = ancillary_revenue.get("afrr_annual_eur", 0.0)
    mfrr = ancillary_revenue.get("mfrr_annual_eur", 0.0)
    total = da_eur + fcr + afrr + mfrr

    return {
        "da_arbitrage_eur": round(da_eur, 2),
        "fcr_eur": round(fcr, 2),
        "afrr_eur": round(afrr, 2),
        "mfrr_eur": round(mfrr, 2),
        "total_eur": round(total, 2),
        "total_per_mw": round(
            total * da_revenue.get("annual_revenue_eur_per_mw", 0.0) / da_eur
            if da_eur > 0 else total, 2
        ),
        "da_pct": round(100.0 * da_eur / total, 1) if total > 0 else 0.0,
        "ancillary_pct": round(100.0 * (total - da_eur) / total, 1) if total > 0 else 0.0,
    }
