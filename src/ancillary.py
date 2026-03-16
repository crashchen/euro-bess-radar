"""Ancillary services data handling: upload, parse, and integrate."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

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
         energy_price_eur_mwh, zone].
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

    # Build standardised output
    out = pd.DataFrame()

    # Parse timestamp
    if "date" in df.columns:
        if "hour" in df.columns:
            out["timestamp"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["hour"].astype(str).str.zfill(2) + ":00",
                utc=True,
            )
        else:
            out["timestamp"] = pd.to_datetime(df["date"], utc=True)
    elif "settlement_date" in df.columns:
        # GB: 30-min periods, period 1 = 00:00
        df["hour_min"] = ((df["settlement_period"].astype(int) - 1) * 30).apply(
            lambda m: f"{m // 60:02d}:{m % 60:02d}"
        )
        out["timestamp"] = pd.to_datetime(
            df["settlement_date"].astype(str) + " " + df["hour_min"], utc=True,
        )
    else:
        out["timestamp"] = pd.NaT

    # Map price columns
    price_cols = {
        "capacity_price_eur_mw": "capacity_price_eur_mw",
        "fcr_n_price": "capacity_price_eur_mw",
        "fcr_d_price": "capacity_price_eur_mw",
        "system_buy_price": "energy_price_eur_mwh",
        "system_sell_price": "energy_price_eur_mwh",
        "marginal_price_up": "energy_price_eur_mwh",
        "marginal_price_down": "energy_price_eur_mwh",
    }
    out["capacity_price_eur_mw"] = float("nan")
    out["energy_price_eur_mwh"] = float("nan")

    for src_col, dst_col in price_cols.items():
        if src_col in df.columns:
            out[dst_col] = pd.to_numeric(df[src_col], errors="coerce")

    out["product_type"] = df.get("product", template_key)
    out["direction"] = df.get("direction", "")
    out["zone"] = template_key.split("_")[0]

    out = out.set_index("timestamp").sort_index()
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

    cap_prices = ancillary_df["capacity_price_eur_mw"].dropna()
    if not cap_prices.empty:
        avg_cap = float(cap_prices.mean())
        hours_per_year = 8760
        cap_revenue = avg_cap * power_mw * hours_per_year * availability / 1000

        product = str(ancillary_df.get("product_type", pd.Series([""])).iloc[0]).upper()
        if "FCR" in product:
            result["fcr_annual_eur"] = round(cap_revenue, 2)
        elif "AFRR" in product or "AFR" in product:
            result["afrr_annual_eur"] = round(cap_revenue, 2)
        else:
            result["fcr_annual_eur"] = round(cap_revenue, 2)

    energy_prices = ancillary_df["energy_price_eur_mwh"].dropna()
    if not energy_prices.empty:
        avg_energy = float(energy_prices.mean())
        energy_mwh = power_mw * duration_hours
        # Assume balancing activations ~10% of hours
        activation_hours = 8760 * 0.10
        energy_revenue = avg_energy * energy_mwh * activation_hours / 1000
        result["mfrr_annual_eur"] = round(energy_revenue, 2)

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
