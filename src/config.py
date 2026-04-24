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
ENTSOE_ENDPOINT_URL = os.getenv("ENTSOE_ENDPOINT_URL", "")

ELEXON_BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
ELEXON_MARKET_INDEX_ENDPOINT = f"{ELEXON_BASE_URL}/balancing/pricing/market-index"
FINGRID_BASE_URL = "https://data.fingrid.fi/api"
REGELLEISTUNG_API_URL = (
    "https://www.regelleistung.net/apps/cpp-publisher/api/v1"
    "/download/tenders/resultsoverview"
)
NETZTRANSPARENZ_BASE_URL = "https://www.netztransparenz.de"

# --- FX Rates (approximate annual averages, update as needed) ---
GBP_EUR_YEARLY: dict[int, float] = {
    2023: 1.15,
    2024: 1.17,
    2025: 1.17,
    2026: 1.18,
}

# --- Default Settings ---
DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_QUERY_TIMEZONE = "Europe/Brussels"
PRICE_CACHE_TTL_HOURS = 24
HOURS_PER_YEAR = 8760
ANCILLARY_CAPACITY_AVAILABILITY = 0.95
ANCILLARY_ENERGY_ACTIVATION_SHARE = 0.10

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

# --- Zone Timezones ---
# Each bidding zone's local IANA timezone.
# DA prices are defined in local time; analytics must group by local day/hour.
ZONE_TIMEZONES: dict[str, str] = {
    # CET/CEST
    "DE_LU": "Europe/Berlin",
    "FR": "Europe/Paris",
    "NL": "Europe/Amsterdam",
    "BE": "Europe/Brussels",
    "AT": "Europe/Vienna",
    "DK_1": "Europe/Copenhagen",
    "DK_2": "Europe/Copenhagen",
    "SE_1": "Europe/Stockholm",
    "SE_2": "Europe/Stockholm",
    "SE_3": "Europe/Stockholm",
    "SE_4": "Europe/Stockholm",
    "NO_1": "Europe/Oslo",
    "NO_2": "Europe/Oslo",
    "NO_3": "Europe/Oslo",
    "NO_4": "Europe/Oslo",
    "NO_5": "Europe/Oslo",
    "ES": "Europe/Madrid",
    "IT_NORD": "Europe/Rome",
    "IT_CSUD": "Europe/Rome",
    "IT_SUD": "Europe/Rome",
    "IT_SICI": "Europe/Rome",
    "IT_SARD": "Europe/Rome",
    "PL": "Europe/Warsaw",
    "CZ": "Europe/Prague",
    "HU": "Europe/Budapest",
    "SK": "Europe/Bratislava",
    "SI": "Europe/Ljubljana",
    "HR": "Europe/Zagreb",
    "CH": "Europe/Zurich",
    # WET/WEST
    "PT": "Europe/Lisbon",
    "IE_SEM": "Europe/Dublin",
    # GMT/BST
    "GB": "Europe/London",
    # EET/EEST
    "FI": "Europe/Helsinki",
    "RO": "Europe/Bucharest",
    "BG": "Europe/Sofia",
    "GR": "Europe/Athens",
    "EE": "Europe/Tallinn",
    "LV": "Europe/Riga",
    "LT": "Europe/Vilnius",
}


def get_api_key() -> str:
    """Load and validate ENTSO-E API key from environment."""
    key = ENTSOE_API_KEY
    if not key:
        raise EnvironmentError(
            "ENTSOE_API_KEY not found. Add it to .env file in project root."
        )
    return key


def get_fingrid_api_key() -> str:
    """Load optional Fingrid API key from environment."""
    return os.getenv("FINGRID_API_KEY", "")


def is_elexon_zone(zone_code: str) -> bool:
    """Check if a zone code should use the Elexon API."""
    return zone_code in ELEXON_ZONES.values()


def get_zone_timezone(zone_code: str) -> str:
    """Return IANA timezone for a bidding zone. Falls back to UTC."""
    return ZONE_TIMEZONES.get(zone_code, "UTC")


def get_zone_display_name(zone_code: str) -> str:
    """Reverse lookup: zone code -> display name."""
    for name, code in ALL_ZONES.items():
        if code == zone_code:
            return name
    return zone_code
