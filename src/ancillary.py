"""Ancillary services data handling: upload, parse, and integrate."""

from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path

import pandas as pd

from src.config import (
    ANCILLARY_CAPACITY_AVAILABILITY,
    ANCILLARY_ENERGY_ACTIVATION_SHARE,
    HOURS_PER_YEAR,
)
from src.data_ingestion import DataSourceParseError, validate_import_zone
from src.time_utils import (
    gb_settlement_period_to_utc,
    parse_regelleistung_time_block_start,
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


def _normalize_product_key(value: object) -> str:
    """Lowercase + collapse separators so 'aFRR Up', 'afrr_up', 'AFRR-UP'
    all match. Used by the manual-vs-auto override logic to avoid silent
    double-counting if an upstream label format flips.
    """
    text = str(value).strip().lower()
    for sep in ("_", "-", "/"):
        text = text.replace(sep, " ")
    return " ".join(text.split())

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
    "IT_BALANCING": {
        "description": (
            "Italy MSD (Mercato Servizi di Dispacciamento) accepted-bid prices "
            "from Terna's Transparency Report. Used when richer-than-imbalance "
            "data is needed; ENTSO-E imbalance via auto-fetch still covers all "
            "7 IT zones for free. The Terna live API requires OAuth credentials "
            "(developer.terna.it) — defer to manual CSV upload for now."
        ),
        "source_url": "https://www.terna.it/en/electric-system/transparency-report/download-center",
        "expected_columns": [
            "date", "hour",
            "marginal_price_up", "marginal_price_down",
        ],
        "resolution": "hourly",
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


# Unified, zone-tagged reserve-capacity import format (the Step-2 "import-first"
# target, distinct from the per-country ANCILLARY_TEMPLATES). The parser lands in
# a follow-up increment; this template + docs/import-templates.md is the spec to
# hand to an exchange / TSO / aggregator when requesting sample data.
CAPACITY_IMPORT_COLUMNS = (
    "timestamp", "zone", "product", "direction", "capacity_price_eur_mw_h",
)
CAPACITY_IMPORT_PRODUCTS = ("FCR", "aFRR", "mFRR")
CAPACITY_IMPORT_DIRECTIONS = ("up", "down", "symmetric")


def generate_capacity_import_template_csv() -> str:
    """Return the unified reserve-capacity import template (EUR/MW/h).

    Units and time semantics are pinned in the header because they are the two
    historically error-prone fields: timestamps must be UTC (or carry a
    ``timezone`` column), and the price is a PER-HOUR rate, never a 4h-block
    total. ``product``/``direction`` are enumerated. See
    ``docs/import-templates.md`` for the full spec to send to a data provider.
    """
    buf = io.StringIO()
    buf.write(
        "# Unified reserve-capacity import template (one format for all zones).\n"
        "# timestamp: UTC, ISO-8601 (e.g. 2026-05-01T00:00:00Z). If your export is\n"
        "#   in local market time, convert to UTC OR add a 'timezone' column with an\n"
        "#   IANA name (e.g. Europe/Berlin) and it is converted to UTC on import.\n"
        "# zone: bidding-zone code (e.g. DE_LU, FI, FR).\n"
        "# product: FCR | aFRR | mFRR (case- and separator-insensitive on import).\n"
        "# direction: up | down | symmetric (FCR is symmetric; aFRR/mFRR up or down).\n"
        "# capacity_price_eur_mw_h: PER-HOUR capacity price in EUR/MW/h -- NOT a 4h\n"
        "#   block total. One row per pricing block (e.g. 4h) is fine; give the\n"
        "#   hourly rate, not the block sum.\n"
    )
    writer = csv.writer(buf)
    writer.writerow(CAPACITY_IMPORT_COLUMNS)
    writer.writerows([
        ["2026-05-01T00:00:00Z", "DE_LU", "FCR", "symmetric", "12.50"],
        ["2026-05-01T04:00:00Z", "DE_LU", "FCR", "symmetric", "15.00"],
        ["2026-05-01T00:00:00Z", "DE_LU", "aFRR", "up", "8.20"],
        ["2026-05-01T00:00:00Z", "DE_LU", "aFRR", "down", "6.10"],
    ])
    return buf.getvalue()


# Unified, zone-tagged activation-ENERGY import format (Step 3a). This is the
# energy leg of reserves and complements the capacity-fee import above. Three
# red-lines are pinned in the template header and docs/import-templates.md:
#   1. ``system_activated_volume_mw`` is SYSTEM-level activated power, NOT this
#      asset's output -- the asset/capture share is a model assumption that lives
#      in the audit panel, never pre-mixed into the data file.
#   2. Activation energy is a SEPARATE stream from the capacity fee and from
#      reBAP/imbalance settlement, and is NOT free additive revenue (it spends
#      SoC that DA/IDA could otherwise have sold).
#   3. With no forward activation signal it supports HISTORICAL REPLAY ONLY --
#      never live dispatch or aggregator following.
# The parser/persistence land in a follow-up increment (3b); this template is the
# stable data contract to hand to a TSO / exchange / aggregator.
ACTIVATION_IMPORT_COLUMNS = (
    "timestamp", "zone", "product", "direction",
    "activation_price_eur_mwh", "system_activated_volume_mw",
)
ACTIVATION_IMPORT_PRODUCTS = ("aFRR", "mFRR")
ACTIVATION_IMPORT_DIRECTIONS = ("up", "down")


def generate_activation_import_template_csv() -> str:
    """Return the unified activation-energy import template (EUR/MWh).

    This is the ENERGY leg of reserves, distinct from the capacity-fee import.
    The header pins the red-lines: ``system_activated_volume_mw`` is the
    SYSTEM-level activated power (the asset/capture share is a model assumption,
    not a CSV field); activation energy is a separate stream from the capacity
    fee and from reBAP/imbalance and is not free additive revenue (it spends SoC
    DA/IDA could have sold); and it is for historical replay only, not live
    dispatch. ``product`` is aFRR/mFRR (FCR has no separately-paid energy leg)
    and ``direction`` is up/down (energy activation is directional). See
    ``docs/import-templates.md`` for the full spec to send to a data provider.
    """
    buf = io.StringIO()
    buf.write(
        "# Unified activation-ENERGY import template (the energy leg of reserves).\n"
        "# Separate stream from the capacity fee AND from reBAP/imbalance -- do not\n"
        "#   sum them blindly; activation energy spends SoC DA/IDA could have sold.\n"
        "# Historical replay only (no forward activation signal); NOT live dispatch.\n"
        "# timestamp: UTC, ISO-8601 (e.g. 2026-05-01T00:00:00Z). If local market\n"
        "#   time, convert to UTC OR add a 'timezone' column (IANA, e.g.\n"
        "#   Europe/Berlin) and it is converted to UTC on import.\n"
        "# zone: bidding-zone code (e.g. DE_LU, FI, FR).\n"
        "# product: aFRR | mFRR (FCR has no separately-paid energy leg).\n"
        "# direction: up | down (energy activation is directional; no symmetric).\n"
        "# activation_price_eur_mwh: energy price paid/charged WHEN activated, EUR/MWh.\n"
        "# system_activated_volume_mw: SYSTEM-level activated power in the interval,\n"
        "#   MW -- NOT this asset's output. The asset/capture share is a model\n"
        "#   assumption (audit panel), never pre-mixed into this file.\n"
    )
    # lineterminator="\n" so the csv.writer rows match the hand-written "\n"
    # comment lines above -- a uniform-newline file to hand to a data provider.
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(ACTIVATION_IMPORT_COLUMNS)
    writer.writerows([
        ["2026-05-01T00:00:00Z", "DE_LU", "aFRR", "up", "85.40", "320"],
        ["2026-05-01T00:00:00Z", "DE_LU", "aFRR", "down", "12.10", "210"],
        ["2026-05-01T00:00:00Z", "DE_LU", "mFRR", "up", "120.00", "150"],
        ["2026-05-01T00:00:00Z", "DE_LU", "mFRR", "down", "5.00", "90"],
    ])
    return buf.getvalue()


# Unified, zone-tagged reBAP / imbalance-settlement import format (Step 4a).
# This is a PASSIVE imbalance-settlement stream, separate from reserve capacity
# and activation energy. Parser/persistence land later; this template is the
# stable data contract to request from TSOs / BRPs / aggregators.
IMBALANCE_IMPORT_COLUMNS = (
    "timestamp", "zone", "imbalance_price_eur_mwh", "system_imbalance_volume_mw",
)


def generate_imbalance_import_template_csv() -> str:
    """Return the unified reBAP / imbalance-settlement import template.

    The header pins the red-lines before any parser/model exists:
    ``imbalance_price_eur_mwh`` is the published settlement cash-flow price
    (negative values are valid and are not direction-flipped);
    ``system_imbalance_volume_mw`` is a system/area quantity, not this asset's
    imbalance; and this stream is separate from reserve capacity and activation
    energy. It supports historical replay only, not live balancing dispatch.
    """
    buf = io.StringIO()
    buf.write(
        "# Unified reBAP / imbalance-settlement import template.\n"
        "# Separate stream from reserve capacity fees and activation energy -- do not\n"
        "#   sum blindly; imbalance settlement is a passive BRP/portfolio cashflow.\n"
        "# Historical replay only; NOT live dispatch or aggregator balancing control.\n"
        "# timestamp: UTC, ISO-8601 (e.g. 2026-05-01T00:00:00Z). If local market\n"
        "#   time, convert to UTC OR add a 'timezone' column (IANA, e.g.\n"
        "#   Europe/Berlin) and it is converted to UTC on import.\n"
        "# zone: bidding-zone / settlement-area code (e.g. DE_LU, FI, FR).\n"
        "# imbalance_price_eur_mwh: published imbalance/reBAP settlement price,\n"
        "#   EUR/MWh. This is already a cash-flow price; negatives are valid and no\n"
        "#   direction sign flip is applied later.\n"
        "# system_imbalance_volume_mw: SYSTEM/area imbalance volume in the interval,\n"
        "#   MW -- NOT this asset's imbalance. The asset imbalance/capture share is\n"
        "#   a model assumption (audit panel), never pre-mixed into this file.\n"
    )
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(IMBALANCE_IMPORT_COLUMNS)
    writer.writerows([
        ["2026-05-01T00:00:00Z", "DE_LU", "42.50", "850"],
        ["2026-05-01T00:15:00Z", "DE_LU", "-18.20", "-420"],
        ["2026-05-01T00:30:00Z", "DE_LU", "65.00", "1200"],
        ["2026-05-01T00:45:00Z", "DE_LU", "8.10", "210"],
    ])
    return buf.getvalue()


# Unified reBAP / imbalance-settlement parser (Step 4b). It returns a dedicated
# imbalance frame because passive imbalance settlement is a separate stream from
# reserve capacity and activation energy. ``system_imbalance_volume_mw`` is kept
# SYSTEM/area-level exactly as imported; any asset/capture share is a later
# replay-model assumption, never pre-mixed into the CSV or parser.
_IMBALANCE_FRAME_COLUMNS = [
    "zone", "imbalance_price_eur_mwh", "system_imbalance_volume_mw",
]


def _empty_imbalance_frame() -> pd.DataFrame:
    """Empty imbalance frame with a UTC DatetimeIndex and standard columns."""
    out = pd.DataFrame(columns=_IMBALANCE_FRAME_COLUMNS)
    out.index = pd.DatetimeIndex([], name="timestamp", tz="UTC")
    return out


def parse_imbalance_import_csv(
    content: str, *, default_zone: str | None = None,
) -> pd.DataFrame:
    """Parse the unified reBAP / imbalance CSV into an imbalance frame.

    Schema (case-insensitive headers): ``timestamp, zone,
    imbalance_price_eur_mwh, system_imbalance_volume_mw`` plus an optional
    ``timezone`` column. Timestamps are UTC unless a per-row IANA ``timezone``
    is given. ``imbalance_price_eur_mwh`` is treated as a published cash-flow
    settlement price, so positive/negative signs are preserved and no direction
    sign flip is applied. ``system_imbalance_volume_mw`` is kept SYSTEM/area
    level exactly as imported. Rows with an unparseable timestamp, price, or
    volume are dropped.

    Returns a frame indexed by UTC ``timestamp`` with columns ``zone``,
    ``imbalance_price_eur_mwh``, and ``system_imbalance_volume_mw``. Raises
    ``DataSourceParseError`` on missing required columns or unsafe zones.
    """
    delimiter = _detect_delimiter(content)
    try:
        raw = pd.read_csv(
            io.StringIO(content), sep=delimiter, comment="#",
            dtype=str, keep_default_na=False,
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataSourceParseError(
            f"Could not parse imbalance import CSV: {exc}"
        ) from exc
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    required = {
        "timestamp", "zone",
        "imbalance_price_eur_mwh", "system_imbalance_volume_mw",
    }
    missing = required - set(raw.columns)
    if missing:
        raise DataSourceParseError(
            "Imbalance import CSV missing columns: " + ", ".join(sorted(missing))
        )

    price = pd.to_numeric(raw["imbalance_price_eur_mwh"], errors="coerce")
    volume = pd.to_numeric(raw["system_imbalance_volume_mw"], errors="coerce")
    tz_col = raw["timezone"] if "timezone" in raw.columns else None
    index = _import_utc_index(raw["timestamp"], tz_col)
    valid = price.notna().to_numpy() & volume.notna().to_numpy() & ~pd.isna(index)
    raw, price, volume, index = raw[valid], price[valid], volume[valid], index[valid]
    if raw.empty:
        return _empty_imbalance_frame()

    out = pd.DataFrame(
        {
            "zone": [_resolve_import_zone_cell(z, default_zone) for z in raw["zone"]],
            "imbalance_price_eur_mwh": price.to_numpy(),
            "system_imbalance_volume_mw": volume.to_numpy(),
        },
        index=pd.DatetimeIndex(index),
    )
    out.index.name = "timestamp"
    return out


_CAPACITY_PRODUCT_CANON = {"fcr": "FCR", "afrr": "aFRR", "mfrr": "mFRR"}
_CAPACITY_DIRECTION_CANON = {
    "up": "up", "down": "down", "symmetric": "symmetric", "sym": "symmetric",
}


def _canonical_capacity_product(value: object) -> str:
    """Canonicalise a product label to FCR/aFRR/mFRR (case/separator tolerant)."""
    key = _normalize_product_key(value).replace(" ", "")
    canon = _CAPACITY_PRODUCT_CANON.get(key)
    if canon is None:
        raise DataSourceParseError(
            f"Unknown reserve product {value!r}; expected one of FCR / aFRR / mFRR."
        )
    return canon


def _canonical_capacity_direction(value: object) -> str:
    """Canonicalise a direction to up/down/symmetric; blank defaults symmetric."""
    key = _normalize_product_key(value).replace(" ", "")
    if not key:
        return "symmetric"
    canon = _CAPACITY_DIRECTION_CANON.get(key)
    if canon is None:
        raise DataSourceParseError(
            f"Unknown reserve direction {value!r}; expected up / down / symmetric."
        )
    return canon


def _import_utc_index(timestamps, tz_col) -> pd.DatetimeIndex:
    """UTC index from timestamps, honouring an optional per-row IANA timezone.

    Shared by the unified capacity and activation-energy importers. Without a
    ``timezone`` column (or for blank cells) a value is assumed UTC — the
    project-internal convention. Unparseable timestamps / unknown zones become
    ``NaT`` so the caller drops them.
    """
    ts = pd.Series(timestamps).astype(str).str.strip()
    if tz_col is None:
        return pd.DatetimeIndex(pd.to_datetime(ts, utc=True, errors="coerce"))
    tzs = pd.Series(tz_col).astype(str).str.strip()
    out: list = []
    for stamp_str, tz_name in zip(ts, tzs, strict=True):
        tz = tz_name if tz_name and tz_name.lower() != "nan" else "UTC"
        try:
            stamp = pd.Timestamp(stamp_str)
            if pd.isna(stamp):
                out.append(pd.NaT)
                continue
            stamp = stamp.tz_localize(tz) if stamp.tzinfo is None else stamp
            out.append(stamp.tz_convert("UTC"))
        except (ValueError, TypeError, KeyError):
            out.append(pd.NaT)  # bad timestamp or unknown IANA zone
    return pd.DatetimeIndex(out)


def _resolve_import_zone_cell(value: object, default_zone: str | None) -> str:
    """Resolve a CSV zone cell to a stripped, table-name-safe zone.

    Shared by both unified importers. A blank cell falls back to ``default_zone``
    (or ``""`` when none — preserved so the persist layer can skip a zone-less
    row). A non-empty value MUST be table-name safe (``validate_import_zone``) or
    this raises ``DataSourceParseError``, so a malformed/malicious zone is
    rejected at parse time before it can reach a SQLite table name.
    """
    zone_val = str(value).strip() or (default_zone or "")
    if zone_val:
        validate_import_zone(zone_val)
    return zone_val


def parse_capacity_import_csv(
    content: str, *, default_zone: str | None = None,
) -> pd.DataFrame:
    """Parse the unified reserve-capacity import CSV into the standard frame.

    Schema (case-insensitive headers): ``timestamp, zone, product, direction,
    capacity_price_eur_mw_h`` plus an optional ``timezone`` column. Timestamps
    are UTC unless a per-row IANA ``timezone`` is given. ``product`` is
    canonicalised to FCR/aFRR/mFRR and ``direction`` to up/down/symmetric
    (case/separator tolerant); an unrecognised value RAISES rather than being
    silently mis-filed (same strictness as the IDA importer). The per-hour
    ``capacity_price_eur_mw_h`` maps to the established frame column
    ``capacity_price_eur_mw`` (same unit), and ``product_type``/``direction``
    are kept faithful to the CSV (direction is NOT folded into the product
    label here). Rows with an unparseable timestamp or price are dropped.

    Returns a standard ancillary frame (timestamp-indexed, UTC) ready to merge
    via ``build_ancillary_dataset``. Raises ``DataSourceParseError`` on missing
    required columns or an unknown product/direction.
    """
    delimiter = _detect_delimiter(content)
    # keep_default_na=False so a blank cell is "" (not NaN -> "nan"), keeping the
    # blank-direction -> symmetric and blank-zone -> default_zone fallbacks honest.
    try:
        raw = pd.read_csv(
            io.StringIO(content), sep=delimiter, comment="#",
            dtype=str, keep_default_na=False,
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataSourceParseError(
            f"Could not parse capacity import CSV: {exc}"
        ) from exc
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    required = {"timestamp", "zone", "product", "direction", "capacity_price_eur_mw_h"}
    missing = required - set(raw.columns)
    if missing:
        raise DataSourceParseError(
            "Capacity import CSV missing columns: " + ", ".join(sorted(missing))
        )

    price = pd.to_numeric(raw["capacity_price_eur_mw_h"], errors="coerce")
    tz_col = raw["timezone"] if "timezone" in raw.columns else None
    index = _import_utc_index(raw["timestamp"], tz_col)
    valid = price.notna().to_numpy() & ~pd.isna(index)
    raw, price, index = raw[valid], price[valid], index[valid]
    if raw.empty:
        return _empty_ancillary_frame()

    out = _initialise_output(pd.DatetimeIndex(index))
    out["product_type"] = [_canonical_capacity_product(p) for p in raw["product"]]
    out["direction"] = [_canonical_capacity_direction(d) for d in raw["direction"]]
    out["zone"] = [_resolve_import_zone_cell(z, default_zone) for z in raw["zone"]]
    out["capacity_price_eur_mw"] = price.to_numpy()
    return out


# Unified activation-ENERGY import parser (Step 3b). It returns a DEDICATED
# activation frame, not the standard ancillary frame, because it carries
# energy-leg columns the standard frame does not (activation_price_eur_mwh,
# system_activated_volume_mw) AND because activation energy is a SEPARATE stream
# (red-line) — it is NOT folded into the capacity/DA revenue here.
_ACTIVATION_FRAME_COLUMNS = [
    "zone", "product_type", "direction",
    "activation_price_eur_mwh", "system_activated_volume_mw",
]
_ACTIVATION_PRODUCT_CANON = {"afrr": "aFRR", "mfrr": "mFRR"}
_ACTIVATION_DIRECTION_CANON = {"up": "up", "down": "down"}


def _canonical_activation_product(value: object) -> str:
    """Canonicalise an activation product to aFRR/mFRR.

    FCR is REJECTED (it has no separately-paid energy leg); unknown values raise
    — same strictness as the capacity/IDA importers, never silently mis-filed.
    """
    key = _normalize_product_key(value).replace(" ", "")
    canon = _ACTIVATION_PRODUCT_CANON.get(key)
    if canon is None:
        raise DataSourceParseError(
            f"Unknown or unsupported activation product {value!r}; expected "
            "aFRR or mFRR (FCR has no separately-paid energy leg)."
        )
    return canon


def _canonical_activation_direction(value: object) -> str:
    """Canonicalise an activation direction to up/down.

    Energy activation is directional, so there is no ``symmetric`` and a blank
    or unknown value raises (unlike capacity, where a blank defaults symmetric).
    """
    key = _normalize_product_key(value).replace(" ", "")
    canon = _ACTIVATION_DIRECTION_CANON.get(key)
    if canon is None:
        raise DataSourceParseError(
            f"Unknown or unsupported activation direction {value!r}; expected "
            "up or down (energy activation is directional; no symmetric)."
        )
    return canon


def _empty_activation_frame() -> pd.DataFrame:
    """Empty activation frame with a UTC DatetimeIndex and the standard columns."""
    out = pd.DataFrame(columns=_ACTIVATION_FRAME_COLUMNS)
    out.index = pd.DatetimeIndex([], name="timestamp", tz="UTC")
    return out


def parse_activation_import_csv(
    content: str, *, default_zone: str | None = None,
) -> pd.DataFrame:
    """Parse the unified activation-energy import CSV into an activation frame.

    Schema (case-insensitive headers): ``timestamp, zone, product, direction,
    activation_price_eur_mwh, system_activated_volume_mw`` plus an optional
    ``timezone`` column. Timestamps are UTC unless a per-row IANA ``timezone`` is
    given. ``product`` is canonicalised to aFRR/mFRR (FCR is REJECTED — no
    separately-paid energy leg) and ``direction`` to up/down (energy activation
    is directional — no ``symmetric``); an unknown/blank value RAISES, the same
    strictness as the capacity/IDA importers. ``system_activated_volume_mw`` is
    kept SYSTEM-level exactly as imported — the asset/capture share is a model
    assumption applied downstream, NEVER multiplied in here (red-line). Rows with
    an unparseable timestamp, price, or volume are dropped.

    Returns a frame indexed by UTC ``timestamp`` with columns ``zone``,
    ``product_type``, ``direction``, ``activation_price_eur_mwh``,
    ``system_activated_volume_mw``. Raises ``DataSourceParseError`` on missing
    required columns or an unknown product/direction.
    """
    delimiter = _detect_delimiter(content)
    try:
        raw = pd.read_csv(
            io.StringIO(content), sep=delimiter, comment="#",
            dtype=str, keep_default_na=False,
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataSourceParseError(
            f"Could not parse activation import CSV: {exc}"
        ) from exc
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    required = {
        "timestamp", "zone", "product", "direction",
        "activation_price_eur_mwh", "system_activated_volume_mw",
    }
    missing = required - set(raw.columns)
    if missing:
        raise DataSourceParseError(
            "Activation import CSV missing columns: " + ", ".join(sorted(missing))
        )

    price = pd.to_numeric(raw["activation_price_eur_mwh"], errors="coerce")
    volume = pd.to_numeric(raw["system_activated_volume_mw"], errors="coerce")
    tz_col = raw["timezone"] if "timezone" in raw.columns else None
    index = _import_utc_index(raw["timestamp"], tz_col)
    valid = price.notna().to_numpy() & volume.notna().to_numpy() & ~pd.isna(index)
    raw, price, volume, index = raw[valid], price[valid], volume[valid], index[valid]
    if raw.empty:
        return _empty_activation_frame()

    # Canonicalise on the surviving rows so an unknown product/direction RAISES
    # (strict) rather than being silently dropped with the bad-data rows.
    out = pd.DataFrame(
        {
            "zone": [_resolve_import_zone_cell(z, default_zone) for z in raw["zone"]],
            "product_type": [_canonical_activation_product(p) for p in raw["product"]],
            "direction": [_canonical_activation_direction(d) for d in raw["direction"]],
            "activation_price_eur_mwh": price.to_numpy(),
            "system_activated_volume_mw": volume.to_numpy(),
        },
        index=pd.DatetimeIndex(index),
    )
    out.index.name = "timestamp"
    return out


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


def _parse_regelleistung_block_index(dates, time_blocks) -> pd.DatetimeIndex:
    """Convert German reserve date/time-block columns to UTC timestamps."""
    timestamps = []
    for target_date, time_block in zip(pd.Series(dates), pd.Series(time_blocks), strict=True):
        timestamps.append(parse_regelleistung_time_block_start(target_date, time_block))
    return pd.DatetimeIndex(timestamps, name="timestamp")


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
    """Map source-specific product labels into stable dashboard product names.

    If ``direction`` is empty, the base string is scanned for an embedded
    direction token (``Up``/``Down``/``Long``/``Short`` / ``Pos``/``Neg``).
    Without this fallback an auto-fetched row carrying its direction inside
    ``product`` (e.g. ``"FCR-D Up"`` with no separate direction column) loses
    the Up/Down split entirely and collapses with its opposite-direction
    sibling, double-counting revenue.
    """
    tokens = " ".join(
        part for part in [template_hint or "", base or ""] if part
    ).upper()
    word_tokens = {t for t in re.split(r"[\s_\-/]+", tokens) if t}

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
    elif "FCR" in tokens or "POS" in word_tokens:
        # POS-as-FCR-family is the German "POS" / "NEG" reserve-direction
        # convention; it must be a complete word, not a substring of e.g.
        # "Post Qualification" or "Position".
        label = "FCR"
    else:
        label = str(base or template_hint or "Unknown")

    dir_upper = str(direction or "").strip().upper()
    if not dir_upper:
        # Recover direction from the base string when not provided explicitly.
        # Match only as a complete trailing token (so "Post Qualification"
        # doesn't false-positive on "POS"); require exactly one direction
        # token ("Up Down" stays unsuffixed); and require at least one
        # non-direction token alongside it (a bare "POS" carries no product
        # family information so we leave it for the bucket logic).
        base_tokens = [
            t for t in re.split(r"[\s_\-/]+", str(base or "").upper().strip()) if t
        ]
        up_tokens = {"UP", "POS", "LONG"}
        down_tokens = {"DOWN", "NEG", "SHORT"}
        has_up = any(tok in up_tokens for tok in base_tokens)
        has_down = any(tok in down_tokens for tok in base_tokens)
        has_non_direction = any(
            tok not in up_tokens and tok not in down_tokens for tok in base_tokens
        )
        if has_non_direction and has_up and not has_down:
            dir_upper = "UP"
        elif has_non_direction and has_down and not has_up:
            dir_upper = "DOWN"
    if dir_upper in {"UP", "LONG", "POS"} and not label.upper().endswith(" UP"):
        return f"{label} Up"
    if dir_upper in {"DOWN", "SHORT", "NEG"} and not label.upper().endswith(" DOWN"):
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
        if "time_block" in df.columns:
            return _parse_regelleistung_block_index(df["date"], df["time_block"])
        if "hour" in df.columns:
            return _parse_date_hour_index(df["date"], df["hour"])
        return pd.to_datetime(df["date"], utc=True)

    if "settlement_date" in df.columns and "settlement_period" in df.columns:
        return gb_settlement_period_to_utc(
            df["settlement_date"],
            df["settlement_period"],
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
        has_explicit_product = "product" in raw.columns
        labels = _coerce_text_array(
            raw["product"] if has_explicit_product else dataset_name,
            idx,
            default=dataset_name,
        )
        directions = _coerce_text_array(
            raw["direction"] if "direction" in raw.columns else "",
            idx,
            default="",
        )
        canonical = [
            _canonical_product_label(
                label,
                direction,
                None if has_explicit_product else dataset_name,
            )
            for label, direction in zip(labels, directions, strict=True)
        ]
        frames.append(_build_standard_frame(
            idx,
            canonical,
            zone,
            direction=directions,
            capacity=raw["capacity_price_eur_mw"],
        ))

    capacity_product_columns = {
        "fcr_n_price": ("FCR-N", ""),
        "fcr_d_price": ("FCR-D", ""),
        "fcr_d_up_price": ("FCR-D Up", "Up"),
        "fcr_d_down_price": ("FCR-D Down", "Down"),
        "afrr_up_price": ("aFRR Up", "Up"),
        "afrr_down_price": ("aFRR Down", "Down"),
        # ESIOS (REE Spain) ancillary bundle column names — without these
        # entries the wide ESIOS frame is silently ignored downstream.
        "secondary_up_capacity_eur_mw": ("aFRR Up", "Up"),
        "secondary_down_capacity_eur_mw": ("aFRR Down", "Down"),
    }
    for col, (product_label, direction) in capacity_product_columns.items():
        if col in raw.columns:
            frames.append(_build_standard_frame(
                idx,
                product_label,
                zone,
                direction=direction,
                capacity=raw[col],
            ))

    # ESIOS tertiary energy prices (single-sided EUR/MWh) — map to mFRR Up/Down.
    esios_energy_columns = {
        "tertiary_up_energy_eur_mwh":   ("mFRR Up", "Up"),
        "tertiary_down_energy_eur_mwh": ("mFRR Down", "Down"),
    }
    for col, (product_label, direction) in esios_energy_columns.items():
        if col in raw.columns:
            frames.append(_build_standard_frame(
                idx,
                product_label,
                zone,
                direction=direction,
                energy=raw[col],
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
        # Normalize product_type for matching: lowercase, collapse any
        # separator (space/dash/underscore) to a single space. Without this,
        # an upstream label flip from "aFRR Up" to "afrr_up" silently lets
        # both manual and auto data through, double-counting revenue.
        raw_products = manual["product_type"].dropna().astype(str)
        manual_products = set(raw_products.map(_normalize_product_key))
        expanded_manual_products = manual_products | {
            _normalize_product_key(alias)
            for product in raw_products
            for alias in PRODUCT_ALIASES.get(product.strip(), set())
        }
        if not combined.empty and expanded_manual_products:
            combined_keys = combined["product_type"].astype(str).map(_normalize_product_key)
            combined = combined[~combined_keys.isin(expanded_manual_products)]
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

    # Reject CSVs missing the template's required columns rather than
    # silently bucketing the rows into ``product_type == UNKNOWN``. Codex
    # showed that a DE_FCR template with only ``product`` and
    # ``capacity_price_eur_mw`` still produced an annualised revenue —
    # which is a real data-quality footgun for downstream forecasts.
    expected_cols = list(ANCILLARY_TEMPLATES[template_key]["expected_columns"])
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise DataSourceParseError(
            f"Ancillary CSV for template {template_key} is missing required "
            f"column(s): {missing}. Expected columns: {expected_cols}."
        )

    # Parse timestamp
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    if "date" in df.columns:
        if "time_block" in df.columns:
            idx = _parse_regelleistung_block_index(df["date"], df["time_block"])
        elif "hour" in df.columns:
            idx = _parse_date_hour_index(df["date"], df["hour"], template_key=template_key)
        else:
            idx = pd.to_datetime(df["date"], utc=True)
    elif "settlement_date" in df.columns:
        idx = gb_settlement_period_to_utc(
            df["settlement_date"],
            df["settlement_period"],
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
            for product, direction in zip(product_series, direction_series, strict=True)
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


def list_capacity_products(ancillary_df: pd.DataFrame | None) -> list[str]:
    """Return reserve product labels that carry a capacity price.

    Args:
        ancillary_df: Merged ancillary dataset (``build_ancillary_dataset``
            output) or ``None``.

    Returns:
        Sorted, de-duplicated ``product_type`` labels with at least one
        non-null ``capacity_price_eur_mw`` (EUR/MW/h) row. These are the
        products eligible for the joint DA + reserve-capacity co-optimisation.
        Empty when no capacity-priced product is loaded.
    """
    if ancillary_df is None or ancillary_df.empty:
        return []
    if "capacity_price_eur_mw" not in ancillary_df or "product_type" not in ancillary_df:
        return []
    priced = ancillary_df[ancillary_df["capacity_price_eur_mw"].notna()]
    if priced.empty:
        return []
    products = priced["product_type"].fillna("UNKNOWN").astype(str).str.strip()
    return sorted({p for p in products if p})


def capacity_price_for_product(
    ancillary_df: pd.DataFrame | None, product: str | None,
) -> float | None:
    """Duration-weighted mean capacity price (EUR/MW/h) for one product.

    Args:
        ancillary_df: Merged ancillary dataset or ``None``.
        product: ``product_type`` label to price.

    Returns:
        The duration-weighted mean capacity price in EUR/MW/h (the unit
        ``dispatch.solve_joint_capacity_batch`` expects), or ``None`` when the
        product is blank or carries no capacity price.
    """
    # Defensive: the helper is independently callable, so guard a blank product
    # (including whitespace-only) rather than relying on every caller.
    if product is None or not str(product).strip():
        return None
    if ancillary_df is None or ancillary_df.empty:
        return None
    if "capacity_price_eur_mw" not in ancillary_df or "product_type" not in ancillary_df:
        return None
    labels = ancillary_df["product_type"].fillna("UNKNOWN").astype(str).str.strip()
    group = ancillary_df[labels == str(product).strip()]
    cap_prices = group["capacity_price_eur_mw"].dropna()
    if cap_prices.empty:
        return None
    return _capacity_price_mean(cap_prices, product)


def capacity_price_series_for_product(
    ancillary_df: pd.DataFrame | None, product: str | None,
) -> pd.Series | None:
    """Timestamp-indexed reserve capacity-price series (EUR/MW/h) for a product.

    The per-interval counterpart to :func:`capacity_price_for_product`: instead
    of collapsing to a duration-weighted mean it returns the product's full
    block-granular price series, so the 9.2a ceiling and the 9.2b sequential
    batch can price reserve identically per interval (via
    ``simulation.align_reserve_price_to_index``) rather than off a flat scalar.

    Returns ``None`` when the product is blank or carries no capacity price.
    """
    if product is None or not str(product).strip():
        return None
    if ancillary_df is None or ancillary_df.empty:
        return None
    if "capacity_price_eur_mw" not in ancillary_df or "product_type" not in ancillary_df:
        return None
    labels = ancillary_df["product_type"].fillna("UNKNOWN").astype(str).str.strip()
    group = ancillary_df[labels == str(product).strip()]
    cap_prices = group["capacity_price_eur_mw"].dropna()
    if cap_prices.empty:
        return None
    return cap_prices.sort_index()


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
        "capacity_ancillary_eur": 0.0,
        "energy_ancillary_eur": 0.0,
        "product_revenues": {},
        "product_revenue_types": {},
    }

    if ancillary_df.empty:
        return result

    product_revenues: dict[str, float] = {}
    product_revenue_types: dict[str, str] = {}
    grouped = ancillary_df.groupby(
        ancillary_df["product_type"].fillna("UNKNOWN").astype(str).str.strip()
    )
    for product, group in grouped:
        annual_revenue = 0.0
        capacity_revenue = 0.0
        energy_revenue = 0.0

        cap_prices = group["capacity_price_eur_mw"].dropna()
        if not cap_prices.empty:
            avg_cap = _capacity_price_mean(cap_prices, product)
            capacity_revenue = avg_cap * power_mw * HOURS_PER_YEAR * availability
            annual_revenue += capacity_revenue

        energy_prices = group["energy_price_eur_mwh"].dropna()
        if not energy_prices.empty:
            avg_energy = float(energy_prices.mean())
            # Annualise explicit single-sided energy prices with the configured
            # screening assumption for activated hours. Two-sided balancing
            # signals are preserved separately and are not auto-monetised here.
            activation_hours = HOURS_PER_YEAR * ANCILLARY_ENERGY_ACTIVATION_SHARE
            energy_revenue = avg_energy * power_mw * activation_hours
            annual_revenue += energy_revenue

        if annual_revenue <= 0:
            continue

        annual_revenue = round(annual_revenue, 2)
        product_revenues[product] = annual_revenue
        if capacity_revenue > 0 and energy_revenue > 0:
            product_revenue_types[product] = "mixed"
        elif capacity_revenue > 0:
            product_revenue_types[product] = "capacity"
        else:
            product_revenue_types[product] = "energy"
        result["capacity_ancillary_eur"] += capacity_revenue
        result["energy_ancillary_eur"] += energy_revenue
        bucket = _service_bucket(product)
        result[bucket] += annual_revenue

    result["total_ancillary_eur"] = round(
        result["fcr_annual_eur"] + result["afrr_annual_eur"] + result["mfrr_annual_eur"], 2
    )
    result["total_ancillary_per_mw"] = round(result["total_ancillary_eur"] / power_mw, 2)
    result["capacity_ancillary_eur"] = round(result["capacity_ancillary_eur"], 2)
    result["energy_ancillary_eur"] = round(result["energy_ancillary_eur"], 2)
    result["product_revenues"] = dict(sorted(product_revenues.items()))
    result["product_revenue_types"] = dict(sorted(product_revenue_types.items()))
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
    standalone_ancillary = fcr + afrr + mfrr
    gross_additive_total = da_eur + standalone_ancillary
    capacity_ancillary = ancillary_revenue.get("capacity_ancillary_eur", 0.0)
    product_revenue_types = ancillary_revenue.get("product_revenue_types", {})
    has_capacity_ancillary = (
        capacity_ancillary > 0
        or any(kind in {"capacity", "mixed"} for kind in product_revenue_types.values())
    )

    if has_capacity_ancillary and standalone_ancillary > 0:
        # DA arbitrage is the primary dispatch signal; capacity reserves
        # cannot be simultaneously committed at full power.  Use DA as the
        # headline and expose the gross additive figure as a reference only.
        total = da_eur
        headline_total_mode = "conservative_da_primary"
        capacity_stack_warning = (
            "Capacity reserve revenue is not added to the headline total because "
            "it cannot be fully co-dispatched with day-ahead arbitrage. "
            "Gross additive total is shown as a non-co-optimized reference."
        )
    else:
        total = gross_additive_total
        headline_total_mode = "additive_energy_only" if standalone_ancillary > 0 else "da_only"
        capacity_stack_warning = ""

    source_revenues = {"DA Arbitrage": round(da_eur, 2)}
    for product, value in product_revenues.items():
        source_revenues[product] = round(float(value), 2)
    component_total = gross_additive_total

    return {
        "da_arbitrage_eur": round(da_eur, 2),
        "fcr_eur": round(fcr, 2),
        "afrr_eur": round(afrr, 2),
        "mfrr_eur": round(mfrr, 2),
        "standalone_ancillary_eur": round(standalone_ancillary, 2),
        "capacity_ancillary_eur": round(capacity_ancillary, 2),
        "energy_ancillary_eur": round(ancillary_revenue.get("energy_ancillary_eur", 0.0), 2),
        "product_revenues": product_revenues,
        "product_revenue_types": product_revenue_types,
        "source_revenues": source_revenues,
        "total_eur": round(total, 2),
        "total_per_mw": round(total / power_mw, 2) if power_mw > 0 else 0.0,
        "gross_additive_total_eur": round(gross_additive_total, 2),
        "headline_total_mode": headline_total_mode,
        "capacity_stack_warning": capacity_stack_warning,
        "da_pct": round(100.0 * da_eur / component_total, 1) if component_total > 0 else 0.0,
        "ancillary_pct": (
            round(100.0 * standalone_ancillary / component_total, 1)
            if component_total > 0 else 0.0
        ),
    }


def co_optimize_revenue_split(
    da_annual_revenue: float,
    capacity_price_eur_mw_h: float,
    power_mw: float,
    availability: float = ANCILLARY_CAPACITY_AVAILABILITY,
) -> dict:
    """Find optimal time split between DA arbitrage and capacity ancillary.

    Tests commitment fractions from 0% to 100% (in 5% steps) and picks
    the split that maximizes total revenue.  During committed hours the
    BESS earns capacity price; during uncommitted hours it earns DA
    arbitrage pro-rata.

    Args:
        da_annual_revenue: Full-year DA arbitrage revenue at 100% DA (EUR).
        capacity_price_eur_mw_h: Average hourly capacity price (EUR/MW/h).
        power_mw: BESS power rating in MW.
        availability: Availability factor for capacity commitment.

    Returns:
        Dict with optimal_fraction, da_revenue, capacity_revenue,
        total_revenue, and the full sweep DataFrame.
    """
    fractions = [i / 20 for i in range(21)]  # 0.00, 0.05, ..., 1.00
    rows = []
    for frac in fractions:
        cap_rev = frac * HOURS_PER_YEAR * capacity_price_eur_mw_h * power_mw * availability
        da_rev = (1 - frac) * da_annual_revenue
        total = da_rev + cap_rev
        rows.append({
            "commitment_fraction": frac,
            "da_revenue": round(da_rev, 2),
            "capacity_revenue": round(cap_rev, 2),
            "total_revenue": round(total, 2),
        })

    import pandas as pd
    sweep = pd.DataFrame(rows)
    best_idx = int(sweep["total_revenue"].idxmax())
    best = sweep.iloc[best_idx]

    return {
        "optimal_fraction": best["commitment_fraction"],
        "da_revenue": best["da_revenue"],
        "capacity_revenue": best["capacity_revenue"],
        "total_revenue": best["total_revenue"],
        "sweep": sweep,
    }
