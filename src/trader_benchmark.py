"""External trader revenue-curve parsing and reconciliation.

The benchmark is a user-supplied, annual EUR/MW/year reference. It is not a
price curve, a contracted floor, or a forecast produced by this repository.
The comparison layer keeps asset type, market scope, and revenue basis beside
the numbers so unlike-for-like curves cannot silently acquire the same label.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping

import pandas as pd

from src.analytics import calculate_yearly_revenue_breakdown
from src.config import ALL_ZONES

REQUIRED_COLUMNS = (
    "zone",
    "scenario",
    "year",
    "revenue_eur_per_mw_yr",
    "asset_type",
    "market_scope",
    "revenue_basis",
)
OPTIONAL_COLUMNS = (
    "duration_hours",
    "max_efc_per_day",
    "source",
    "as_of",
)

ALLOWED_ASSET_TYPES = frozenset({"standalone", "co-located", "unknown"})
ALLOWED_MARKET_SCOPES = frozenset(
    {"da-only", "wholesale", "all-in", "unknown"}
)
ALLOWED_REVENUE_BASES = frozenset(
    {"gross", "net-of-wear", "net-of-fees", "unknown"}
)

MODEL_REVENUE_COLUMN = "model_revenue_eur_per_mw_yr"
QUOTE_REVENUE_COLUMN = "benchmark_revenue_eur_per_mw_yr"


def generate_trader_benchmark_template_csv() -> str:
    """Return a commented upload template with a neutral example curve."""
    rows = [
        "# External annual revenue benchmark (user-supplied; not a floor).",
        "# revenue_eur_per_mw_yr = annual revenue normalised by installed MW.",
        "# asset_type = standalone | co-located | unknown.",
        "# market_scope = da-only | wholesale | all-in | unknown.",
        "# revenue_basis = gross | net-of-wear | net-of-fees | unknown.",
        "# duration_hours / max_efc_per_day / source / as_of are optional.",
        (
            "zone,scenario,year,revenue_eur_per_mw_yr,asset_type,"
            "market_scope,revenue_basis,duration_hours,max_efc_per_day,"
            "source,as_of"
        ),
        (
            "IT_SICI,User-supplied example,2029,100000,standalone,"
            "all-in,unknown,6.7,1.2,Replace with source,2026-01-15"
        ),
        (
            "IT_SICI,User-supplied example,2030,90000,standalone,"
            "all-in,unknown,6.7,1.2,Replace with source,2026-01-15"
        ),
    ]
    return "\n".join(rows) + "\n"


def _without_full_line_comments(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _normalise_required_choice(
    frame: pd.DataFrame,
    column: str,
    allowed: frozenset[str],
) -> None:
    values = frame[column].astype("string").str.strip().str.lower()
    bad = sorted(set(values.dropna()) - allowed)
    if values.isna().any() or (values == "").any() or bad:
        raise ValueError(
            f"Benchmark column {column!r} must use one of {sorted(allowed)}; "
            f"invalid value(s): {bad or ['blank']}"
        )
    frame[column] = values


def _normalise_optional_positive(frame: pd.DataFrame, column: str) -> None:
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & numeric.isna()
    finite = numeric.dropna().map(math.isfinite)
    if invalid.any() or not finite.all() or (numeric.dropna() <= 0).any():
        raise ValueError(
            f"Benchmark column {column!r} must contain positive finite values "
            "when supplied."
        )
    frame[column] = numeric.astype(float)


def _validate_scenario_metadata(frame: pd.DataFrame) -> None:
    """Require one audit basis per (zone, scenario), not a drifting label."""
    metadata = [
        "asset_type",
        "market_scope",
        "revenue_basis",
        "duration_hours",
        "max_efc_per_day",
        "source",
        "as_of",
    ]
    for (zone, scenario), group in frame.groupby(["zone", "scenario"], dropna=False):
        for column in metadata:
            values = group[column].dropna().astype(str).str.strip()
            values = values[values != ""]
            if values.nunique() > 1:
                raise ValueError(
                    "Benchmark metadata must be constant within one scenario: "
                    f"{zone}/{scenario} has multiple {column!r} values."
                )


def parse_trader_benchmark_csv(csv_content: str) -> pd.DataFrame:
    """Parse a user-supplied annual revenue benchmark CSV.

    The returned curve is sorted by zone/scenario/year. Revenue remains signed;
    a negative external forecast is valid information. Duplicate years within a
    scenario and drifting basis metadata are rejected because either would make
    a reconciliation chart ambiguous.
    """
    try:
        frame = pd.read_csv(
            io.StringIO(_without_full_line_comments(csv_content))
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "Benchmark CSV contains no header or data rows."
        ) from exc
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Benchmark CSV missing required columns: {missing}. Required: "
            f"{list(REQUIRED_COLUMNS)}."
        )
    if frame.empty:
        raise ValueError("Benchmark CSV contains no data rows.")
    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame["zone"] = frame["zone"].astype("string").str.strip()
    known_zones = set(ALL_ZONES.values())
    bad_zones = sorted(set(frame["zone"].dropna()) - known_zones)
    if frame["zone"].isna().any() or (frame["zone"] == "").any() or bad_zones:
        raise ValueError(
            f"Benchmark CSV contains unknown or blank zone(s): "
            f"{bad_zones or ['blank']}. Use one of: {sorted(known_zones)}."
        )

    frame["scenario"] = frame["scenario"].astype("string").str.strip()
    if frame["scenario"].isna().any() or (frame["scenario"] == "").any():
        raise ValueError("Benchmark column 'scenario' must be non-blank.")

    year_numeric = pd.to_numeric(frame["year"], errors="coerce")
    if (
        year_numeric.isna().any()
        or not year_numeric.map(math.isfinite).all()
        or not (year_numeric == year_numeric.round()).all()
        or not year_numeric.between(2000, 2200).all()
    ):
        raise ValueError(
            "Benchmark column 'year' must contain integer years from 2000 to 2200."
        )
    frame["year"] = year_numeric.astype(int)

    revenue = pd.to_numeric(frame["revenue_eur_per_mw_yr"], errors="coerce")
    if revenue.isna().any() or not revenue.map(math.isfinite).all():
        raise ValueError(
            "Benchmark column 'revenue_eur_per_mw_yr' must be numeric and finite."
        )
    frame["revenue_eur_per_mw_yr"] = revenue.astype(float)

    _normalise_required_choice(frame, "asset_type", ALLOWED_ASSET_TYPES)
    _normalise_required_choice(frame, "market_scope", ALLOWED_MARKET_SCOPES)
    _normalise_required_choice(frame, "revenue_basis", ALLOWED_REVENUE_BASES)
    _normalise_optional_positive(frame, "duration_hours")
    _normalise_optional_positive(frame, "max_efc_per_day")

    frame["source"] = frame["source"].astype("string").str.strip()
    frame.loc[frame["source"] == "", "source"] = pd.NA
    as_of_raw = frame["as_of"]
    as_of = pd.to_datetime(as_of_raw, errors="coerce")
    invalid_as_of = as_of_raw.notna() & as_of.isna()
    if invalid_as_of.any():
        raise ValueError("Benchmark column 'as_of' contains an invalid date.")
    frame["as_of"] = as_of

    duplicates = frame.duplicated(["zone", "scenario", "year"], keep=False)
    if duplicates.any():
        rows = frame.loc[duplicates, ["zone", "scenario", "year"]]
        raise ValueError(
            "Benchmark CSV has duplicate (zone, scenario, year) rows: "
            f"{rows.to_dict(orient='records')}"
        )
    _validate_scenario_metadata(frame)

    columns = [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]
    return frame[columns].sort_values(["zone", "scenario", "year"]).reset_index(
        drop=True
    )


def build_forward_model_yearly(
    daily_by_zone: Mapping[str, pd.DataFrame],
    *,
    power_mw: float,
    duration_hours: float,
    efficiency: float,
    capture_rate: float,
) -> pd.DataFrame:
    """Build the platform's annual forward-DA curve for reconciliation."""
    rows: list[pd.DataFrame] = []
    for zone, daily in daily_by_zone.items():
        if daily is None or daily.empty:
            continue
        yearly = calculate_yearly_revenue_breakdown(
            daily,
            power_mw=power_mw,
            duration_hours=duration_hours,
            roundtrip_efficiency=efficiency,
            capture_rate=capture_rate,
        )
        if yearly.empty:
            continue
        yearly = yearly.rename(
            columns={"revenue_per_mw": MODEL_REVENUE_COLUMN}
        )
        yearly.insert(0, "zone", zone)
        rows.append(
            yearly[
                [
                    "zone",
                    "year",
                    "n_days",
                    "coverage_pct",
                    "is_partial_year",
                    "avg_cycles_per_day",
                    MODEL_REVENUE_COLUMN,
                ]
            ]
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "zone",
                "year",
                "n_days",
                "coverage_pct",
                "is_partial_year",
                "avg_cycles_per_day",
                MODEL_REVENUE_COLUMN,
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(["zone", "year"])


def _endpoint_cagr(values: pd.DataFrame) -> float:
    first = values.iloc[0]
    last = values.iloc[-1]
    years = int(last["year"]) - int(first["year"])
    start = float(first["revenue_eur_per_mw_yr"])
    end = float(last["revenue_eur_per_mw_yr"])
    if years <= 0 or start <= 0 or end <= 0:
        return float("nan")
    return (end / start) ** (1.0 / years) - 1.0


def reconcile_trader_benchmark(
    benchmark: pd.DataFrame,
    model_yearly: pd.DataFrame,
    *,
    zone: str,
    scenario: str,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Join one external scenario to the platform curve on calendar year.

    The ratio is deliberately named ``benchmark_to_model_ratio``. It is an
    observed reconciliation statistic, not an inferred capture rate: scope,
    fees, foresight, market depth, and asset type can all contribute.
    """
    quote = benchmark.loc[
        (benchmark["zone"] == zone) & (benchmark["scenario"] == scenario)
    ].copy()
    if quote.empty:
        raise ValueError(f"No benchmark rows for zone={zone!r}, scenario={scenario!r}.")
    quote = quote.sort_values("year")

    model = model_yearly.loc[model_yearly["zone"] == zone].copy()
    comparison = quote[
        ["year", "revenue_eur_per_mw_yr"]
    ].rename(columns={"revenue_eur_per_mw_yr": QUOTE_REVENUE_COLUMN})
    if not model.empty:
        comparison = comparison.merge(
            model[
                [
                    "year",
                    MODEL_REVENUE_COLUMN,
                    "coverage_pct",
                    "is_partial_year",
                    "avg_cycles_per_day",
                ]
            ],
            on="year",
            how="left",
            validate="one_to_one",
        )
    else:
        comparison[MODEL_REVENUE_COLUMN] = float("nan")
        comparison["coverage_pct"] = float("nan")
        comparison["is_partial_year"] = pd.NA
        comparison["avg_cycles_per_day"] = float("nan")

    comparison["model_minus_benchmark_eur_per_mw_yr"] = (
        comparison[MODEL_REVENUE_COLUMN] - comparison[QUOTE_REVENUE_COLUMN]
    )
    positive_model = comparison[MODEL_REVENUE_COLUMN] > 0
    comparison["benchmark_to_model_ratio"] = float("nan")
    comparison.loc[positive_model, "benchmark_to_model_ratio"] = (
        comparison.loc[positive_model, QUOTE_REVENUE_COLUMN]
        / comparison.loc[positive_model, MODEL_REVENUE_COLUMN]
    )

    overlap = comparison.dropna(subset=[MODEL_REVENUE_COLUMN])
    quote_avg = float(quote["revenue_eur_per_mw_yr"].mean())
    if overlap.empty:
        model_avg = gap_avg = ratio = float("nan")
    else:
        model_avg = float(overlap[MODEL_REVENUE_COLUMN].mean())
        quote_overlap_avg = float(overlap[QUOTE_REVENUE_COLUMN].mean())
        gap_avg = model_avg - quote_overlap_avg
        ratio = quote_overlap_avg / model_avg if model_avg > 0 else float("nan")

    first = quote.iloc[0]
    summary: dict[str, float | int | str] = {
        "zone": zone,
        "scenario": scenario,
        "n_benchmark_years": len(quote),
        "n_overlap_years": len(overlap),
        "benchmark_average_eur_per_mw_yr": quote_avg,
        "model_overlap_average_eur_per_mw_yr": model_avg,
        "model_minus_benchmark_average_eur_per_mw_yr": gap_avg,
        "benchmark_to_model_ratio": ratio,
        "benchmark_endpoint_cagr": _endpoint_cagr(quote),
        "asset_type": str(first["asset_type"]),
        "market_scope": str(first["market_scope"]),
        "revenue_basis": str(first["revenue_basis"]),
    }
    return comparison, summary


def benchmark_comparability_notes(
    benchmark: pd.DataFrame,
    *,
    zone: str,
    scenario: str,
    model_duration_hours: float,
) -> list[str]:
    """Return explicit reasons why a selected quote is not like-for-like."""
    selected = benchmark.loc[
        (benchmark["zone"] == zone) & (benchmark["scenario"] == scenario)
    ]
    if selected.empty:
        return ["No rows exist for the selected zone/scenario."]
    row = selected.iloc[0]
    notes: list[str] = []
    if row["asset_type"] != "standalone":
        notes.append(
            "Asset type is not standalone; co-location value, curtailment, "
            "shared-grid constraints, and PPA opportunity cost are outside "
            "the current platform model."
        )
    if row["market_scope"] != "da-only":
        notes.append(
            f"Benchmark market scope is {row['market_scope']!r}, while the "
            "platform curve here is DA-only."
        )
    if row["revenue_basis"] != "gross":
        notes.append(
            f"Benchmark revenue basis is {row['revenue_basis']!r}, while the "
            "platform curve here is gross dispatch revenue after the sidebar "
            "capture haircut but before wear, fees, tax, and financing."
        )
    duration = row["duration_hours"]
    if pd.notna(duration) and not math.isclose(
        float(duration), float(model_duration_hours), rel_tol=0.0, abs_tol=1e-9
    ):
        notes.append(
            f"Benchmark duration is {float(duration):g}h; the current platform "
            f"case is {float(model_duration_hours):g}h."
        )
    if pd.notna(row["max_efc_per_day"]):
        notes.append(
            "Benchmark declares a cycle cap, while the Forward Scenarios "
            "dispatch path does not impose that cap; compare cycle intensity "
            "before interpreting the revenue gap."
        )
    return notes
