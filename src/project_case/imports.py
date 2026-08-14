"""Fail-closed CSV imports used by the Project Case UI (PC-C).

The import layer is deliberately pure: it accepts CSV bytes/text, validates the
complete public shape, constructs the typed PC-A values, and returns a normalised
``DataFrame`` for the UI preview.  It does not depend on Streamlit.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections.abc import Iterable

import pandas as pd

from src.project_case.enums import MAX_PROJECT_LIFE_YEARS
from src.project_case.schema import AugmentationEvent, ProjectCaseValidationError

AUGMENTATION_COLUMNS = (
    "year",
    "cost_eur",
    "capacity_restored_frac",
    "residual_value_eur",
)
EXPLICIT_MULTIPLIER_COLUMNS = ("year", "multiplier")

AUGMENTATION_TEMPLATE_FILENAME = "project_case_augmentation_schedule.csv"
EXPLICIT_MULTIPLIER_TEMPLATE_FILENAME = "project_case_annual_multipliers.csv"

# A one-row example remains valid for every permitted project life.  Users may
# add multiple events in the same year; the lifecycle schema intentionally
# permits that and applies every event in deterministic canonical order.
AUGMENTATION_TEMPLATE_CSV = (
    b"year,cost_eur,capacity_restored_frac,residual_value_eur\n"
    b"1,0,0.10,0\n"
)

_INTEGER_RE = re.compile(r"^-?\d+$")

__all__ = [
    "AUGMENTATION_COLUMNS",
    "AUGMENTATION_TEMPLATE_CSV",
    "AUGMENTATION_TEMPLATE_FILENAME",
    "EXPLICIT_MULTIPLIER_COLUMNS",
    "EXPLICIT_MULTIPLIER_TEMPLATE_FILENAME",
    "explicit_multiplier_template_csv",
    "parse_augmentation_csv",
    "parse_explicit_multiplier_csv",
]


def _decode_csv(data: bytes | bytearray | memoryview | str, *, label: str) -> str:
    """Return UTF-8 CSV text, accepting the byte/text forms used by uploads."""
    if isinstance(data, str):
        text = data
    elif isinstance(data, (bytes, bytearray, memoryview)):
        try:
            text = bytes(data).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ProjectCaseValidationError(f"{label} CSV must be UTF-8 text") from exc
    else:
        raise ProjectCaseValidationError(f"{label} CSV must be bytes or text")

    # ``utf-8-sig`` handles byte input; strip the same BOM from direct text.
    text = text.removeprefix("\ufeff")
    if not text.strip():
        raise ProjectCaseValidationError(f"{label} CSV is empty")
    if "\x00" in text:
        raise ProjectCaseValidationError(f"{label} CSV must not contain NUL bytes")
    return text


def _read_exact_csv(
    data: bytes | bytearray | memoryview | str,
    *,
    columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    """Read a non-empty CSV whose column-name set is exactly ``columns``."""
    text = _decode_csv(data, label=label)

    # pandas deliberately accepts ragged records: short rows become empty cells
    # and some over-wide rows are silently reinterpreted through an inferred
    # index.  Neither behaviour is safe for a fingerprint-bearing Project Case
    # input.  Validate the logical CSV records first (quoted commas/newlines are
    # still handled correctly), then let pandas provide the tabular parsing.
    try:
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise ProjectCaseValidationError(
            f"{label} CSV could not be parsed: {exc}"
        ) from exc
    expected_arity = len(columns)
    header = records[0] if records else []
    header_set = set(header)
    required = set(columns)
    if (
        len(header) != len(header_set)
        or header_set != required
        or len(header) != expected_arity
    ):
        missing = sorted(required - header_set)
        extra = sorted(header_set - required)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ProjectCaseValidationError(
            f"{label} CSV columns must be exactly {list(columns)}"
            + (f" ({', '.join(details)})" if details else "")
        )
    for row_number, record in enumerate(records[1:], start=2):
        if len(record) != expected_arity:
            raise ProjectCaseValidationError(
                f"{label} CSV row {row_number} must contain exactly "
                f"{expected_arity} fields (got {len(record)})"
            )

    try:
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raise ProjectCaseValidationError(f"{label} CSV could not be parsed: {exc}") from exc

    actual = tuple(str(column) for column in frame.columns)
    required = set(columns)
    actual_set = set(actual)
    if len(actual) != len(actual_set):
        raise ProjectCaseValidationError(f"{label} CSV has duplicate column names")
    if actual_set != required or len(actual) != len(columns):
        missing = sorted(required - actual_set)
        extra = sorted(actual_set - required)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ProjectCaseValidationError(
            f"{label} CSV columns must be exactly {list(columns)}"
            + (f" ({', '.join(details)})" if details else "")
        )
    if frame.empty:
        raise ProjectCaseValidationError(f"{label} CSV must contain at least one data row")

    # Work in the canonical column order and trim spreadsheet-style padding.
    normalised = frame.loc[:, list(columns)].copy()
    for column in columns:
        normalised[column] = normalised[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return normalised


def _parse_integer(value: object, *, field: str, row_number: int) -> int:
    raw = str(value).strip()
    if not _INTEGER_RE.fullmatch(raw):
        raise ProjectCaseValidationError(
            f"{field} at CSV row {row_number} must be an integer"
        )
    return int(raw)


def _parse_finite(value: object, *, field: str, row_number: int) -> float:
    raw = str(value).strip()
    if not raw:
        raise ProjectCaseValidationError(
            f"{field} at CSV row {row_number} must be a finite number"
        )
    try:
        result = float(raw)
    except (TypeError, ValueError) as exc:
        raise ProjectCaseValidationError(
            f"{field} at CSV row {row_number} must be a finite number"
        ) from exc
    if not math.isfinite(result):
        raise ProjectCaseValidationError(
            f"{field} at CSV row {row_number} must be finite"
        )
    return result


def parse_augmentation_csv(
    data: bytes | bytearray | memoryview | str,
) -> tuple[tuple[AugmentationEvent, ...], pd.DataFrame]:
    """Parse an augmentation schedule into typed events and a preview frame.

    Identical rows and multiple events in one year are valid.  Event-domain
    rules (non-negative cash values, restoration fraction in ``[0, 1]`` and
    year bounds) are delegated to ``AugmentationEvent`` so this import path and
    programmatic construction cannot drift.
    """
    frame = _read_exact_csv(
        data,
        columns=AUGMENTATION_COLUMNS,
        label="augmentation schedule",
    )
    events: list[AugmentationEvent] = []
    for position, row in enumerate(frame.itertuples(index=False, name=None), start=2):
        year = _parse_integer(row[0], field="year", row_number=position)
        cost = _parse_finite(row[1], field="cost_eur", row_number=position)
        restored = _parse_finite(
            row[2], field="capacity_restored_frac", row_number=position
        )
        residual = _parse_finite(
            row[3], field="residual_value_eur", row_number=position
        )
        try:
            event = AugmentationEvent(year, cost, restored, residual)
        except ProjectCaseValidationError as exc:
            raise ProjectCaseValidationError(
                f"augmentation schedule CSV row {position}: {exc}"
            ) from exc
        events.append(event)

    preview = pd.DataFrame(
        [event.to_payload() for event in events], columns=list(AUGMENTATION_COLUMNS)
    )
    return tuple(events), preview


def _validate_project_life_years(project_life_years: object) -> int:
    if isinstance(project_life_years, bool) or not isinstance(project_life_years, int):
        raise ProjectCaseValidationError("project_life_years must be an integer")
    if not 1 <= project_life_years <= MAX_PROJECT_LIFE_YEARS:
        raise ProjectCaseValidationError(
            f"project_life_years must be in [1, {MAX_PROJECT_LIFE_YEARS}]"
        )
    return project_life_years


def explicit_multiplier_template_csv(project_life_years: int) -> bytes:
    """Return a flat, valid year-1…life explicit-multiplier CSV template."""
    life = _validate_project_life_years(project_life_years)
    rows: Iterable[str] = (f"{year},1.0" for year in range(1, life + 1))
    text = "year,multiplier\n" + "\n".join(rows) + "\n"
    return text.encode("utf-8")


def parse_explicit_multiplier_csv(
    data: bytes | bytearray | memoryview | str,
    project_life_years: int,
) -> tuple[tuple[float, ...], pd.DataFrame]:
    """Parse a complete, unique year-1…life multiplier curve.

    The returned tuple is ordered by project year and satisfies the numerical
    invariants required by ``Projection(ExplicitAnnualMultiplierCurve, ...)``;
    source and as-of provenance remain UI-owned fields.
    """
    life = _validate_project_life_years(project_life_years)
    frame = _read_exact_csv(
        data,
        columns=EXPLICIT_MULTIPLIER_COLUMNS,
        label="explicit annual multiplier",
    )

    by_year: dict[int, float] = {}
    for position, row in enumerate(frame.itertuples(index=False, name=None), start=2):
        year = _parse_integer(row[0], field="year", row_number=position)
        multiplier = _parse_finite(row[1], field="multiplier", row_number=position)
        if multiplier < 0.0:
            raise ProjectCaseValidationError(
                f"multiplier at CSV row {position} must be >= 0"
            )
        if year in by_year:
            raise ProjectCaseValidationError(
                f"explicit annual multiplier CSV has duplicate year {year}"
            )
        by_year[year] = multiplier

    required_years = set(range(1, life + 1))
    actual_years = set(by_year)
    if actual_years != required_years:
        missing = sorted(required_years - actual_years)
        unexpected = sorted(actual_years - required_years)
        details: list[str] = []
        if missing:
            details.append(f"missing years={missing}")
        if unexpected:
            details.append(f"unexpected years={unexpected}")
        raise ProjectCaseValidationError(
            "explicit annual multiplier CSV must contain each project year exactly once"
            + (f" ({', '.join(details)})" if details else "")
        )

    multipliers = tuple(float(by_year[year]) for year in range(1, life + 1))
    if multipliers[0] != 1.0:
        raise ProjectCaseValidationError(
            "explicit annual multiplier year 1 must equal 1.0"
        )

    preview = pd.DataFrame(
        {
            "year": list(range(1, life + 1)),
            "multiplier": list(multipliers),
        },
        columns=list(EXPLICIT_MULTIPLIER_COLUMNS),
    )
    return multipliers, preview
