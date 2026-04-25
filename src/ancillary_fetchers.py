"""Automated fetchers for ancillary/balancing market data by zone."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

import src.data_ingestion as _ingestion

logger = logging.getLogger(__name__)

# Registry of available auto-fetchers per zone
AUTO_FETCHERS: dict[str, list[dict[str, Any]]] = {
    "FI": [
        {
            "name": "FCR-N/D prices",
            "fetcher": "fetch_fingrid_fcr_prices",
            "source": "Fingrid",
        },
        {
            "name": "aFRR prices",
            "fetcher": "fetch_fingrid_afrr_prices",
            "source": "Fingrid",
        },
    ],
    "DE_LU": [
        {
            "name": "FCR/aFRR auctions",
            "fetcher": "fetch_regelleistung_results",
            "source": "Regelleistung.net",
        },
    ],
    "GB": [
        {
            "name": "System prices",
            "fetcher": "fetch_elexon_system_prices",
            "source": "Elexon",
        },
    ],
    "RO": [
        {
            "name": "Imbalance prices",
            "fetcher": "fetch_entsoe_imbalance_prices",
            "source": "ENTSO-E",
        },
    ],
    "SE_3": [
        {
            "name": "Imbalance prices",
            "fetcher": "fetch_entsoe_imbalance_prices",
            "source": "ENTSO-E",
        },
    ],
    "IT_SUD": [
        {
            "name": "Imbalance prices",
            "fetcher": "fetch_entsoe_imbalance_prices",
            "source": "ENTSO-E",
        },
    ],
}

def _resolve_fetcher(func_name: str) -> Callable[..., Any] | None:
    """Resolve a fetcher function by name from data_ingestion at call time."""
    return getattr(_ingestion, func_name, None)


def get_available_fetchers(zone: str) -> list[dict[str, Any]]:
    """Return list of available auto-fetchers for a zone."""
    return AUTO_FETCHERS.get(zone, [])


def run_auto_fetch(
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Run all available auto-fetchers for a zone.

    Args:
        zone: Bidding zone code.
        start: Start timestamp (UTC).
        end: End timestamp (UTC).

    Returns:
        Dict mapping data_name -> DataFrame.
        Catches errors per fetcher and continues with others.
    """
    fetchers = get_available_fetchers(zone)
    results: dict[str, pd.DataFrame] = {}

    for entry in fetchers:
        name = entry["name"]
        func_name = entry["fetcher"]
        func = _resolve_fetcher(func_name)

        if func is None:
            logger.warning("No function found for fetcher '%s'", func_name)
            continue

        logger.info("Running auto-fetch: %s for %s via %s", name, zone, entry["source"])
        try:
            if func_name == "fetch_regelleistung_results":
                if not getattr(_ingestion, "REGELLEISTUNG_AUTO_FETCH_ENABLED", True):
                    logger.info(
                        "Regelleistung auto-fetch disabled for %s; use manual DE_FCR/DE_aFRR uploads",
                        zone,
                    )
                    continue
                # Regelleistung needs product arg; try both FCR and aFRR
                for product in ["FCR", "aFRR"]:
                    result = func(product, start, end)
                    if result is not None and not result.empty:
                        results[f"{name}_{product}"] = result
            elif func_name == "fetch_entsoe_imbalance_prices":
                result = func(zone, start, end)
                if result is not None and not result.empty:
                    results[name] = result
            else:
                result = func(start, end)
                if result is not None and not result.empty:
                    results[name] = result
        except Exception as exc:
            logger.warning("Auto-fetch '%s' for %s failed: %s", name, zone, exc)

    logger.info(
        "Auto-fetch for %s complete: %d/%d succeeded",
        zone, len(results), len(fetchers),
    )
    return results
