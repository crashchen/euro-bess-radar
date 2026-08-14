"""Pure PC-C CSV import helpers: exact shapes, typed values, fail-closed errors."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src.project_case import AugmentationEvent, ProjectCaseValidationError
from src.project_case.imports import (
    AUGMENTATION_COLUMNS,
    AUGMENTATION_TEMPLATE_CSV,
    EXPLICIT_MULTIPLIER_COLUMNS,
    explicit_multiplier_template_csv,
    parse_augmentation_csv,
    parse_explicit_multiplier_csv,
)


def test_augmentation_template_has_exact_shape_and_is_valid():
    raw = pd.read_csv(io.BytesIO(AUGMENTATION_TEMPLATE_CSV))
    assert tuple(raw.columns) == AUGMENTATION_COLUMNS

    events, preview = parse_augmentation_csv(AUGMENTATION_TEMPLATE_CSV)
    assert events == (AugmentationEvent(1, 0.0, 0.1, 0.0),)
    assert tuple(preview.columns) == AUGMENTATION_COLUMNS


def test_augmentation_accepts_utf8_bom_text_and_normalises_preview():
    text = (
        "\ufeffresidual_value_eur,capacity_restored_frac,cost_eur,year\n"
        " 50 , 0.25 , 1250.5 , 3 \n"
    )
    events, preview = parse_augmentation_csv(text)

    assert events == (AugmentationEvent(3, 1250.5, 0.25, 50.0),)
    assert preview.to_dict("records") == [
        {
            "year": 3,
            "cost_eur": 1250.5,
            "capacity_restored_frac": 0.25,
            "residual_value_eur": 50.0,
        }
    ]


def test_augmentation_allows_duplicate_rows_and_same_year_events():
    csv = (
        "year,cost_eur,capacity_restored_frac,residual_value_eur\n"
        "4,100,0.1,0\n"
        "4,100,0.1,0\n"
        "4,50,0.2,10\n"
    )
    events, preview = parse_augmentation_csv(csv.encode())

    assert len(events) == 3
    assert list(preview["year"]) == [4, 4, 4]


@pytest.mark.parametrize(
    "csv",
    [
        "year,cost_eur,capacity_restored_frac\n1,2,0.1\n",
        (
            "year,cost_eur,capacity_restored_frac,residual_value_eur,note\n"
            "1,2,0.1,0,x\n"
        ),
        (
            "year,year,capacity_restored_frac,residual_value_eur\n"
            "1,1,0.1,0\n"
        ),
    ],
)
def test_augmentation_rejects_missing_extra_or_duplicate_columns(csv):
    with pytest.raises(ProjectCaseValidationError, match="columns must be exactly"):
        parse_augmentation_csv(csv)


@pytest.mark.parametrize("csv", [b"", b"   \n", b"year,cost_eur,capacity_restored_frac,residual_value_eur\n"])
def test_augmentation_rejects_empty_input_or_no_rows(csv):
    with pytest.raises(ProjectCaseValidationError, match=r"empty|at least one"):
        parse_augmentation_csv(csv)


@pytest.mark.parametrize("bad", ["", "not-a-number", "NaN", "Inf", "-Inf"])
def test_augmentation_rejects_blank_nonnumeric_or_nonfinite_numbers(bad):
    csv = (
        "year,cost_eur,capacity_restored_frac,residual_value_eur\n"
        f"1,{bad},0.1,0\n"
    )
    with pytest.raises(ProjectCaseValidationError, match=r"cost_eur.*finite"):
        parse_augmentation_csv(csv)


def test_augmentation_rejects_fractional_year_and_schema_domain_errors():
    with pytest.raises(ProjectCaseValidationError, match=r"year.*integer"):
        parse_augmentation_csv(
            "year,cost_eur,capacity_restored_frac,residual_value_eur\n1.0,0,0.1,0\n"
        )
    with pytest.raises(ProjectCaseValidationError, match=r"CSV row 2.*cost_eur"):
        parse_augmentation_csv(
            "year,cost_eur,capacity_restored_frac,residual_value_eur\n1,-1,0.1,0\n"
        )
    with pytest.raises(
        ProjectCaseValidationError, match=r"CSV row 2.*capacity_restored_frac"
    ):
        parse_augmentation_csv(
            "year,cost_eur,capacity_restored_frac,residual_value_eur\n1,0,1.1,0\n"
        )


def test_augmentation_rejects_non_utf8_and_non_text_input():
    with pytest.raises(ProjectCaseValidationError, match="UTF-8"):
        parse_augmentation_csv(b"\xff\xfe")
    with pytest.raises(ProjectCaseValidationError, match="bytes or text"):
        parse_augmentation_csv(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "csv",
    [
        (
            "year,cost_eur,capacity_restored_frac,residual_value_eur\n"
            "1,100,0.1\n"
        ),
        (
            "year,cost_eur,capacity_restored_frac,residual_value_eur\n"
            "1,100,0.1,0,unexpected\n"
        ),
    ],
)
def test_augmentation_rejects_ragged_data_rows_instead_of_pandas_coercion(csv):
    with pytest.raises(ProjectCaseValidationError, match=r"row 2.*exactly 4 fields"):
        parse_augmentation_csv(csv)


@pytest.mark.parametrize(
    "parser",
    [
        lambda raw: parse_augmentation_csv(raw),
        lambda raw: parse_explicit_multiplier_csv(raw, 1),
    ],
)
def test_project_case_csv_imports_reject_nul_bytes(parser):
    with pytest.raises(ProjectCaseValidationError, match="must not contain NUL"):
        parser(b"year,multiplier\x00\n1,1\n")


def test_explicit_multiplier_template_covers_exact_life_and_parses():
    template = explicit_multiplier_template_csv(4)
    raw = pd.read_csv(io.BytesIO(template))
    assert tuple(raw.columns) == EXPLICIT_MULTIPLIER_COLUMNS
    assert raw.to_dict("list") == {
        "year": [1, 2, 3, 4],
        "multiplier": [1.0, 1.0, 1.0, 1.0],
    }

    multipliers, preview = parse_explicit_multiplier_csv(template, 4)
    assert multipliers == (1.0, 1.0, 1.0, 1.0)
    assert preview.equals(raw)


def test_explicit_multiplier_reorders_rows_into_project_year_order():
    csv = "multiplier,year\n0.7,3\n1,1\n0.9,2\n"
    multipliers, preview = parse_explicit_multiplier_csv(csv, 3)

    assert multipliers == (1.0, 0.9, 0.7)
    assert preview.to_dict("records") == [
        {"year": 1, "multiplier": 1.0},
        {"year": 2, "multiplier": 0.9},
        {"year": 3, "multiplier": 0.7},
    ]


@pytest.mark.parametrize(
    ("csv", "message"),
    [
        ("year,multiplier\n1,1\n1,0.9\n", "duplicate year 1"),
        ("year,multiplier\n1,1\n3,0.8\n", r"missing years=\[2\]"),
        ("year,multiplier\n1,1\n2,0.9\n3,0.8\n", r"unexpected years=\[3\]"),
        ("year,multiplier\n1,0.9\n2,0.8\n", "year 1 must equal 1.0"),
        ("year,multiplier\n1,1\n2,-0.1\n", "must be >= 0"),
        ("year,multiplier\n1,1\n2,NaN\n", "must be finite"),
        ("year,multiplier\n1,1\n2.0,0.9\n", "must be an integer"),
    ],
)
def test_explicit_multiplier_rejects_invalid_year_or_value_matrix(csv, message):
    with pytest.raises(ProjectCaseValidationError, match=message):
        parse_explicit_multiplier_csv(csv, 2)


@pytest.mark.parametrize(
    "csv",
    [
        "year\n1\n",
        "year,multiplier,note\n1,1,x\n",
    ],
)
def test_explicit_multiplier_rejects_missing_or_extra_columns(csv):
    with pytest.raises(ProjectCaseValidationError, match="columns must be exactly"):
        parse_explicit_multiplier_csv(csv, 1)


@pytest.mark.parametrize(
    "csv",
    [
        "year,multiplier\n1\n",
        "year,multiplier\n1,1,unexpected\n",
    ],
)
def test_explicit_multiplier_rejects_ragged_data_rows(csv):
    with pytest.raises(ProjectCaseValidationError, match=r"row 2.*exactly 2 fields"):
        parse_explicit_multiplier_csv(csv, 1)


@pytest.mark.parametrize("bad_life", [True, 0, 101, 2.0])
def test_multiplier_helpers_reject_invalid_project_life(bad_life):
    with pytest.raises(ProjectCaseValidationError, match="project_life_years"):
        explicit_multiplier_template_csv(bad_life)  # type: ignore[arg-type]
    with pytest.raises(ProjectCaseValidationError, match="project_life_years"):
        parse_explicit_multiplier_csv("year,multiplier\n1,1\n", bad_life)  # type: ignore[arg-type]
