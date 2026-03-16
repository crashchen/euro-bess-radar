"""Project configuration: bidding zones, API endpoints, paths."""

from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "cache" / "bess_pulse.db"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
MANUAL_DIR = PROJECT_ROOT / "data" / "manual"

# --- API Config ---
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY", "")

ELEXON_BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
ELEXON_MARKET_INDEX_ENDPOINT = f"{ELEXON_BASE_URL}/balancing/pricing/market-index"

# --- FX Rate (static default, updatable) ---
GBP_TO_EUR = 1.17  # approximate, update as needed

# --- Default Settings ---
DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_QUERY_TIMEZONE = "Europe/Brussels"

# --- Bidding Zones ---
# Tier 1: EU member states (ENTSO-E API)
# Tier 2: Non-EU ENTSO-E members (same API, same treatment)
# Tier 3: Great Britain (Elexon API, separate fetcher)

ENTSOE_ZONES: dict[str, str] = {
    # Central Western Europe
    "Germany/Luxembourg": "DE_LU",
    "France": "FR",
    "Netherlands": "NL",
    "Belgium": "BE",
    "Austria": "AT",
    # Nordics
    "Denmark 1": "DK_1",
    "Denmark 2": "DK_2",
    "Sweden 1": "SE_1",
    "Sweden 2": "SE_2",
    "Sweden 3": "SE_3",
    "Sweden 4": "SE_4",
    "Finland": "FI",
    "Norway 1": "NO_1",
    "Norway 2": "NO_2",
    "Norway 3": "NO_3",
    "Norway 4": "NO_4",
    "Norway 5": "NO_5",
    # Southern Europe
    "Spain": "ES",
    "Portugal": "PT",
    "Italy North": "IT_NORD",
    "Italy Centre-South": "IT_CSUD",
    "Italy South": "IT_SUD",
    "Italy Sicily": "IT_SICI",
    "Italy Sardinia": "IT_SARD",
    "Greece": "GR",
    # Central & Eastern Europe
    "Poland": "PL",
    "Czech Republic": "CZ",
    "Hungary": "HU",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "Croatia": "HR",
    "Romania": "RO",
    "Bulgaria": "BG",
    # Baltics
    "Estonia": "EE",
    "Latvia": "LV",
    "Lithuania": "LT",
    # Other
    "Ireland (SEM)": "IE_SEM",
    "Switzerland": "CH",
}

ELEXON_ZONES: dict[str, str] = {
    "Great Britain": "GB",
}

# Combined for UI display
ALL_ZONES: dict[str, str] = {**ENTSOE_ZONES, **ELEXON_ZONES}


def get_api_key() -> str:
    """Load and validate ENTSO-E API key from environment."""
    key = ENTSOE_API_KEY
    if not key:
        raise EnvironmentError(
            "ENTSOE_API_KEY not found. Add it to .env file in project root."
        )
    return key


def is_elexon_zone(zone_code: str) -> bool:
    """Check if a zone code should use the Elexon API."""
    return zone_code in ELEXON_ZONES.values()


def get_zone_display_name(zone_code: str) -> str:
    """Reverse lookup: zone code -> display name."""
    for name, code in ALL_ZONES.items():
        if code == zone_code:
            return name
    return zone_code
