"""Tests for ancillary services module and ancillary fetchers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ancillary import (
    ANCILLARY_TEMPLATES,
    CAPACITY_IMPORT_COLUMNS,
    CAPACITY_IMPORT_DIRECTIONS,
    CAPACITY_IMPORT_PRODUCTS,
    build_ancillary_dataset,
    calculate_ancillary_revenue,
    capacity_price_for_product,
    capacity_price_series_for_product,
    co_optimize_revenue_split,
    generate_capacity_import_template_csv,
    generate_template_csv,
    list_capacity_products,
    merge_revenue_stack,
    normalize_auto_fetch_dataset,
    parse_ancillary_csv,
)
from src.ancillary_fetchers import get_available_fetchers, run_auto_fetch
from src.config import (
    ANCILLARY_CAPACITY_AVAILABILITY,
    ANCILLARY_ENERGY_ACTIVATION_SHARE,
    GBP_EUR_YEARLY,
    HOURS_PER_YEAR,
)

# ── Template CSV generation ──────────────────────────────────────────────────

class TestTemplateGeneration:
    def test_all_templates_generate(self) -> None:
        for key in ANCILLARY_TEMPLATES:
            csv_str = generate_template_csv(key)
            assert len(csv_str) > 0
            lines = [line for line in csv_str.strip().split("\n") if not line.startswith("#")]
            assert len(lines) == 3  # header + 2 example rows

    def test_capacity_import_template_pins_unit_and_schema(self) -> None:
        csv_str = generate_capacity_import_template_csv()
        # The header must pin the two historically-confused semantics.
        assert "UTC" in csv_str
        assert "EUR/MW/h" in csv_str
        assert "block total" in csv_str  # per-hour rate, not a 4h block sum
        data_lines = [
            line for line in csv_str.strip().splitlines() if not line.startswith("#")
        ]
        assert data_lines[0].split(",") == list(CAPACITY_IMPORT_COLUMNS)
        # Every example row: 5 fields, a parseable price, enumerated product/direction.
        for row in data_lines[1:]:
            fields = row.split(",")
            assert len(fields) == len(CAPACITY_IMPORT_COLUMNS)
            float(fields[4])  # capacity_price_eur_mw_h
            assert fields[2] in CAPACITY_IMPORT_PRODUCTS
            assert fields[3] in CAPACITY_IMPORT_DIRECTIONS

    def test_template_has_correct_headers(self) -> None:
        for key, tmpl in ANCILLARY_TEMPLATES.items():
            csv_str = generate_template_csv(key)
            header_line = next(
                line for line in csv_str.strip().split("\n") if not line.startswith("#")
            )
            for col in tmpl["expected_columns"]:
                assert col in header_line

    def test_gb_template_mentions_gbp_units(self) -> None:
        csv_str = generate_template_csv("GB_BALANCING")
        assert "GBP/MWh" in csv_str

    def test_fi_fcr_template_uses_zero_indexed_hours(self) -> None:
        csv_str = generate_template_csv("FI_FCR")
        lines = [line for line in csv_str.strip().split("\n") if not line.startswith("#")]
        assert lines[1].split(",")[1] == "0"
        assert lines[2].split(",")[1] == "1"

    def test_esios_bundle_flows_into_ancillary_stack(self) -> None:
        """ESIOS auto-fetch returns wide columns like secondary_up_capacity_eur_mw;
        normalize_auto_fetch_dataset must recognise them so revenue stack
        downstream sees the data instead of silently dropping it.
        """
        from src.ancillary import build_ancillary_dataset

        idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        esios = pd.DataFrame({
            "timestamp": idx,
            "secondary_up_capacity_eur_mw": [10.0, 12.0, 11.0],
            "secondary_down_capacity_eur_mw": [8.0, 9.0, 8.5],
            "tertiary_up_energy_eur_mwh": [50.0, 55.0, 52.0],
            "tertiary_down_energy_eur_mwh": [20.0, 22.0, 21.0],
        })
        anc = build_ancillary_dataset(
            auto_fetch_results={"Secondary/tertiary reserves": esios},
        )
        assert not anc.empty
        assert set(anc["product_type"].unique()) == {
            "aFRR Up", "aFRR Down", "mFRR Up", "mFRR Down",
        }
        # Capacity columns should be filled for aFRR rows, energy columns for mFRR rows
        afrr_up = anc[anc["product_type"] == "aFRR Up"]
        mfrr_up = anc[anc["product_type"] == "mFRR Up"]
        assert (afrr_up["capacity_price_eur_mw"] == [10.0, 12.0, 11.0]).all()
        assert (mfrr_up["energy_price_eur_mwh"] == [50.0, 55.0, 52.0]).all()

    def test_it_balancing_parses_marginal_prices(self) -> None:
        """IT_BALANCING uses the same marginal_price_up/down columns as
        RO_BALANCING — verify they flow into energy_price_up/down_eur_mwh
        so downstream stack maths see the data on the right side.
        """
        from src.ancillary import parse_ancillary_csv

        csv_str = (
            "date,hour,marginal_price_up,marginal_price_down\n"
            "2025-01-01,1,55.0,30.0\n"
            "2025-01-01,2,60.0,28.0\n"
        )
        df = parse_ancillary_csv(csv_str, "IT_BALANCING")
        assert len(df) == 2
        assert "energy_price_up_eur_mwh" in df.columns
        assert df["energy_price_up_eur_mwh"].iloc[0] == 55.0
        assert df["energy_price_down_eur_mwh"].iloc[0] == 30.0
        assert (df["zone"] == "IT").all()

# ── Parsing ──────────────────────────────────────────────────────────────────

@pytest.fixture
def de_fcr_csv() -> str:
    """Minimal DE_FCR CSV."""
    return (
        "date,product,capacity_price_eur_mw\n"
        "2025-01-01,POS,15.50\n"
        "2025-01-02,POS,16.20\n"
        "2025-01-03,POS,14.80\n"
    )


@pytest.fixture
def gb_balancing_csv() -> str:
    """Minimal GB_BALANCING CSV."""
    return (
        "settlement_date,settlement_period,system_buy_price,system_sell_price\n"
        "2025-01-01,1,55.00,45.00\n"
        "2025-01-01,2,60.00,50.00\n"
    )


class TestParsing:
    def test_de_fcr_parsing(self, de_fcr_csv: str) -> None:
        df = parse_ancillary_csv(de_fcr_csv, "DE_FCR")
        assert len(df) == 3
        assert "capacity_price_eur_mw" in df.columns
        assert df["capacity_price_eur_mw"].iloc[0] == 15.50

    def test_gb_balancing_parsing(self, gb_balancing_csv: str) -> None:
        df = parse_ancillary_csv(gb_balancing_csv, "GB_BALANCING")
        assert len(df) == 2
        assert "energy_price_eur_mwh" in df.columns
        assert "system_buy_price_eur_mwh" in df.columns
        assert "system_sell_price_eur_mwh" in df.columns
        assert df["system_buy_price_eur_mwh"].iloc[0] == pytest.approx(
            55.0 * GBP_EUR_YEARLY[2025]
        )
        assert df["system_sell_price_eur_mwh"].iloc[0] == pytest.approx(
            45.0 * GBP_EUR_YEARLY[2025]
        )
        assert df["energy_price_eur_mwh"].isna().all()

    def test_gb_balancing_uses_london_settlement_timezone(self) -> None:
        csv = (
            "settlement_date,settlement_period,system_buy_price,system_sell_price\n"
            "2025-07-01,1,55.00,45.00\n"
            "2025-07-01,3,60.00,50.00\n"
        )
        df = parse_ancillary_csv(csv, "GB_BALANCING")
        assert list(df.index.astype(str)) == [
            "2025-06-30 23:00:00+00:00",
            "2025-07-01 00:00:00+00:00",
        ]

    def test_de_time_block_parsing_uses_berlin_local_time(self) -> None:
        csv = (
            "date,time_block,product,capacity_price_eur_mw\n"
            "2025-01-01,00:00-04:00,FCR,15.50\n"
            "2025-01-01,04:00,FCR,16.20\n"
        )
        df = parse_ancillary_csv(csv, "DE_FCR")
        assert list(df.index.astype(str)) == [
            "2024-12-31 23:00:00+00:00",
            "2025-01-01 03:00:00+00:00",
        ]

    def test_gb_balancing_parsing_ignores_template_comment_line(self) -> None:
        csv = generate_template_csv("GB_BALANCING")
        df = parse_ancillary_csv(csv, "GB_BALANCING")
        assert len(df) == 2

    def test_normalize_auto_fetch_preserves_system_prices(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="30min", tz="UTC")
        raw = pd.DataFrame({
            "timestamp": idx,
            "system_buy_price_eur": [60.0, 65.0],
            "system_sell_price_eur": [40.0, 45.0],
        })
        df = normalize_auto_fetch_dataset(raw, "System prices")
        assert df["system_buy_price_eur_mwh"].iloc[0] == 60.0
        assert df["system_sell_price_eur_mwh"].iloc[0] == 40.0
        assert df["energy_price_eur_mwh"].isna().all()

    def test_normalize_auto_fetch_preserves_product_dimension(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        raw = pd.DataFrame({
            "timestamp": idx,
            "fcr_n_price": [10.0, 14.0],
            "fcr_d_up_price": [20.0, 22.0],
            "afrr_up_price": [30.0, 34.0],
        })
        df = normalize_auto_fetch_dataset(raw, "FI reserves")
        assert set(df["product_type"]) == {"FCR-N", "FCR-D Up", "aFRR Up"}
        assert len(df) == 6

    def test_column_mapping(self, de_fcr_csv: str) -> None:
        df = parse_ancillary_csv(de_fcr_csv, "DE_FCR")
        expected = {"product_type", "direction", "capacity_price_eur_mw",
                    "energy_price_eur_mwh", "zone"}
        assert expected.issubset(set(df.columns))

    def test_invalid_template_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown template"):
            parse_ancillary_csv("a,b\n1,2", "INVALID")

    def test_missing_required_column_raises(self) -> None:
        """Codex P0 repro: a DE_FCR upload missing ``date`` used to silently
        bucket rows as UNKNOWN and still produce annualised revenue. The
        parser must now reject incomplete uploads instead.
        """
        from src.data_ingestion import DataSourceParseError
        csv = "product,capacity_price_eur_mw\nFCR-N,10.00\n"
        with pytest.raises(DataSourceParseError, match="missing required column"):
            parse_ancillary_csv(csv, "DE_FCR")

    def test_missing_required_column_lists_all(self) -> None:
        """Error message lists EVERY missing column, not just the first."""
        from src.data_ingestion import DataSourceParseError
        csv = "irrelevant\n1\n"
        with pytest.raises(
            DataSourceParseError, match=r"\['date', 'product', 'capacity_price_eur_mw'\]",
        ):
            parse_ancillary_csv(csv, "DE_FCR")

    def test_semicolon_delimiter(self) -> None:
        csv = "date;product;capacity_price_eur_mw\n2025-01-01;POS;10.00\n"
        df = parse_ancillary_csv(csv, "DE_FCR")
        assert len(df) == 1

    def test_fi_fcr_hour_zero_and_twenty_three_round_trip(self) -> None:
        csv = (
            "date,hour,fcr_n_price,fcr_d_price\n"
            "2025-01-01,0,10.00,20.00\n"
            "2025-01-01,23,11.00,21.00\n"
        )
        df = parse_ancillary_csv(csv, "FI_FCR")

        fcr_n_hours = list(df[df["product_type"] == "FCR-N"].index.hour)
        fcr_d_hours = list(df[df["product_type"] == "FCR-D"].index.hour)

        assert fcr_n_hours == [0, 23]
        assert fcr_d_hours == [0, 23]


# ── Revenue calculation ──────────────────────────────────────────────────────

class TestAncillaryRevenue:
    def test_with_capacity_prices(self, de_fcr_csv: str) -> None:
        df = parse_ancillary_csv(de_fcr_csv, "DE_FCR")
        result = calculate_ancillary_revenue(df, power_mw=10.0, duration_hours=1.0)
        assert result["total_ancillary_eur"] == 1289910.0
        assert result["fcr_annual_eur"] == 1289910.0

    def test_energy_prices_use_eur_units(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame({
            "energy_price_eur_mwh": [100.0, 100.0],
            "capacity_price_eur_mw": [float("nan"), float("nan")],
            "product_type": ["GB_BALANCING", "GB_BALANCING"],
            "direction": ["", ""],
            "zone": ["GB", "GB"],
        }, index=idx)

        result = calculate_ancillary_revenue(df, power_mw=1.0, duration_hours=1.0)
        assert result["mfrr_annual_eur"] == 87600.0

    def test_capacity_assumption_reads_from_config_constant(self, de_fcr_csv: str) -> None:
        df = parse_ancillary_csv(de_fcr_csv, "DE_FCR")

        with patch("src.ancillary.ANCILLARY_CAPACITY_AVAILABILITY", 0.50):
            result = calculate_ancillary_revenue(df, power_mw=1.0, duration_hours=1.0)

        expected = round(df["capacity_price_eur_mw"].mean() * HOURS_PER_YEAR * 0.50, 2)
        assert result["fcr_annual_eur"] == expected

    def test_capacity_prices_use_duration_weighted_mean(self) -> None:
        idx = pd.to_datetime(
            ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z", "2025-01-01T07:00:00Z"],
            utc=True,
        )
        df = pd.DataFrame({
            "capacity_price_eur_mw": [10.0, 20.0, 30.0],
            "energy_price_eur_mwh": [float("nan"), float("nan"), float("nan")],
            "product_type": ["FCR", "FCR", "FCR"],
            "direction": ["", "", ""],
            "zone": ["FI", "FI", "FI"],
        }, index=idx)

        with patch("src.ancillary.ANCILLARY_CAPACITY_AVAILABILITY", 1.0):
            result = calculate_ancillary_revenue(df, power_mw=1.0, duration_hours=1.0)

        assert result["fcr_annual_eur"] == round(26.25 * HOURS_PER_YEAR, 2)

    def test_energy_activation_share_reads_from_config_constant(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame({
            "energy_price_eur_mwh": [100.0, 100.0],
            "capacity_price_eur_mw": [float("nan"), float("nan")],
            "product_type": ["GB_BALANCING", "GB_BALANCING"],
            "direction": ["", ""],
            "zone": ["GB", "GB"],
        }, index=idx)

        with patch("src.ancillary.ANCILLARY_ENERGY_ACTIVATION_SHARE", 0.25):
            result = calculate_ancillary_revenue(df, power_mw=1.0, duration_hours=1.0)

        assert result["mfrr_annual_eur"] == 100.0 * HOURS_PER_YEAR * 0.25

    def test_energy_prices_annualise_using_power_not_duration(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame({
            "energy_price_eur_mwh": [120.0, 120.0],
            "capacity_price_eur_mw": [float("nan"), float("nan")],
            "product_type": ["GB_BALANCING", "GB_BALANCING"],
            "direction": ["", ""],
            "zone": ["GB", "GB"],
        }, index=idx)

        result = calculate_ancillary_revenue(df, power_mw=2.0, duration_hours=4.0)
        expected = 120.0 * 2.0 * HOURS_PER_YEAR * ANCILLARY_ENERGY_ACTIVATION_SHARE
        assert result["mfrr_annual_eur"] == expected

    def test_gb_balancing_requires_explicit_energy_price(self, gb_balancing_csv: str) -> None:
        df = parse_ancillary_csv(gb_balancing_csv, "GB_BALANCING")
        result = calculate_ancillary_revenue(df, power_mw=1.0, duration_hours=1.0)
        assert result["mfrr_annual_eur"] == 0.0

    def test_auto_fetch_capacity_data_flows_into_revenue(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        auto_results = {
            "FCR-N/D prices": pd.DataFrame({
                "timestamp": idx,
                "fcr_n_price": [10.0, 14.0],
                "fcr_d_up_price": [20.0, 22.0],
            }),
            "aFRR prices": pd.DataFrame({
                "timestamp": idx,
                "afrr_up_price": [30.0, 34.0],
            }),
        }

        anc_df = build_ancillary_dataset(auto_fetch_results=auto_results)
        result = calculate_ancillary_revenue(anc_df, power_mw=1.0, duration_hours=1.0)
        assert result["fcr_annual_eur"] == 274626.0
        assert result["afrr_annual_eur"] == 266304.0
        assert result["product_revenues"] == {
            "FCR-D Up": 174762.0,
            "FCR-N": 99864.0,
            "aFRR Up": 266304.0,
        }

    def test_manual_upload_overrides_same_product_only(self, de_fcr_csv: str) -> None:
        manual_df = parse_ancillary_csv(de_fcr_csv, "DE_FCR")
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        auto_results = {
            "FCR auctions": pd.DataFrame({
                "timestamp": idx,
                "capacity_price_eur_mw": [99.0, 100.0],
                "product": ["FCR", "FCR"],
            }),
            "aFRR prices": pd.DataFrame({
                "timestamp": idx,
                "afrr_up_price": [30.0, 34.0],
            }),
        }

        anc_df = build_ancillary_dataset(manual_df=manual_df, auto_fetch_results=auto_results)
        assert set(anc_df["product_type"]) == {"FCR", "aFRR Up"}
        manual_fcr = anc_df[anc_df["product_type"] == "FCR"]
        assert manual_fcr["capacity_price_eur_mw"].iloc[0] == 15.50

    def test_family_detection_does_not_false_positive_on_pos_substring(self) -> None:
        """The family-detection branch maps bare 'POS' to FCR (legacy German
        convention) but must not match it as a substring of unrelated words
        like 'Post Qualification' or 'Position'.
        """
        from src.ancillary import _canonical_product_label
        # Substrings must not trigger the FCR family.
        assert _canonical_product_label("Post Qualification", "", "AUC") == "Post Qualification"
        assert _canonical_product_label("Position", "", "AUC") == "Position"
        # But the legitimate German bare 'POS' label still maps to FCR.
        assert _canonical_product_label("POS", "", "DE_FCR") == "FCR"

    def test_direction_extraction_does_not_false_positive_on_substrings(self) -> None:
        """'Post Qualification' contains the substring 'POS' but is not a
        direction; 'Up Down' is bidirectional and must not be classified
        as Up. A bare 'POS' carries no product family info either.
        """
        from src.ancillary import _canonical_product_label
        assert _canonical_product_label("aFRR Post Qualification", "", "AUC") == "aFRR"
        assert _canonical_product_label("aFRR Up Down", "", "AUC") == "aFRR"
        assert _canonical_product_label("POS", "", "DE_FCR") == "FCR"
        # Genuine cases must still extract.
        assert _canonical_product_label("FCR-D Up", "", "AUC") == "FCR-D Up"
        assert _canonical_product_label("aFRR_UP", "", "AUC") == "aFRR Up"

    def test_direction_recovered_from_embedded_product_label(self) -> None:
        """When the auto-fetch source carries direction inside the product
        string (e.g. 'FCR-D Up') without a separate direction column, the
        canonical label must still produce 'FCR-D Up' / 'FCR-D Down' — not
        collapse them both to 'FCR-D' and double-count.
        """
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": idx,
            "product": ["FCR-D Up", "FCR-D Down"],
            "capacity_price_eur_mw": [10.0, 12.0],
        })
        out = normalize_auto_fetch_dataset(df, "AUC")
        assert set(out["product_type"]) == {"FCR-D Up", "FCR-D Down"}

        # Also covers separator/case variants in the embedded direction.
        df2 = pd.DataFrame({
            "timestamp": idx,
            "product": ["aFRR_UP", "aFRR-DOWN"],
            "capacity_price_eur_mw": [5.0, 6.0],
        })
        out2 = normalize_auto_fetch_dataset(df2, "AUC")
        assert set(out2["product_type"]) == {"aFRR Up", "aFRR Down"}

    def test_explicit_product_wins_over_mixed_dataset_title(self) -> None:
        """A combined fetcher title must not relabel explicit FCR rows as aFRR."""
        idx = pd.date_range("2026-06-20", periods=2, freq="4h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": idx,
            "product": ["FCR", "FCR"],
            "direction": ["Symmetric", "Symmetric"],
            "capacity_price_eur_mw": [10.0, 12.0],
        })

        out = normalize_auto_fetch_dataset(df, "FCR/aFRR auctions_FCR")

        assert set(out["product_type"]) == {"FCR"}

    def test_normalize_product_key_collapses_format_variants(self) -> None:
        """Defense-in-depth: if upstream ever sends a label that bypasses
        canonical-label mapping (e.g. a new product not in the keyword list),
        normalize_product_key still maps case/separator variants to one key
        so manual overrides don't double-count silently.
        """
        from src.ancillary import _normalize_product_key
        base = _normalize_product_key("aFRR Up")
        assert base == _normalize_product_key("AFRR UP")
        assert base == _normalize_product_key("afrr_up")
        assert base == _normalize_product_key("aFRR-Up")
        assert base == _normalize_product_key("  aFRR   Up  ")

    def test_manual_override_matches_across_label_format_variants(self) -> None:
        """Manual override must still match auto data even if separators or
        case differ ('aFRR Up' vs 'afrr_up' vs 'AFRR-UP') — otherwise both
        rows pass through and revenue gets double-counted.
        """
        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        manual_df = pd.DataFrame(
            {
                "capacity_price_eur_mw": [99.0, 99.0],
                "energy_price_eur_mwh": [float("nan"), float("nan")],
                "product_type": ["aFRR Up", "aFRR Up"],
                "direction": ["Up", "Up"],
                "zone": ["DE_LU", "DE_LU"],
            },
            index=idx,
        )
        # Auto-fetch labels arrive in alternative casings/separators
        auto_results = {
            "DE auctions": pd.DataFrame({
                "timestamp": idx,
                "product": ["AFRR-UP", "afrr_up"],
                "direction": ["Up", "Up"],
                "capacity_price_eur_mw": [50.0, 50.0],
            }),
        }
        anc_df = build_ancillary_dataset(manual_df=manual_df, auto_fetch_results=auto_results)
        # Auto rows should be removed by override; only manual price (99) remains.
        assert (anc_df["capacity_price_eur_mw"] == 99.0).all()
        assert len(anc_df) == 2

    def test_manual_fi_fcr_d_overrides_auto_fcr_d_aliases(self) -> None:
        idx = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
        manual_df = pd.DataFrame(
            {
                "capacity_price_eur_mw": [50.0],
                "energy_price_eur_mwh": [float("nan")],
                "product_type": ["FCR-D"],
                "direction": [""],
                "zone": ["FI"],
            },
            index=idx,
        )
        auto_results = {
            "Fingrid reserves": pd.DataFrame(
                {
                    "timestamp": idx,
                    "fcr_n_price": [10.0],
                    "fcr_d_up_price": [20.0],
                    "fcr_d_down_price": [30.0],
                }
            )
        }

        anc_df = build_ancillary_dataset(manual_df=manual_df, auto_fetch_results=auto_results)

        assert "FCR-D" in set(anc_df["product_type"])
        assert "FCR-N" in set(anc_df["product_type"])
        assert "FCR-D Up" not in set(anc_df["product_type"])
        assert "FCR-D Down" not in set(anc_df["product_type"])

    def test_empty_df_returns_zeros(self) -> None:
        df = pd.DataFrame(columns=["capacity_price_eur_mw", "energy_price_eur_mwh",
                                    "product_type", "direction", "zone"])
        result = calculate_ancillary_revenue(df)
        assert result["total_ancillary_eur"] == 0.0

    def test_keys(self, de_fcr_csv: str) -> None:
        df = parse_ancillary_csv(de_fcr_csv, "DE_FCR")
        result = calculate_ancillary_revenue(df)
        expected = {"fcr_annual_eur", "afrr_annual_eur", "mfrr_annual_eur",
                    "total_ancillary_eur", "total_ancillary_per_mw",
                    "capacity_ancillary_eur", "energy_ancillary_eur",
                    "product_revenues", "product_revenue_types"}
        assert set(result.keys()) == expected


# ── Revenue stack merging ────────────────────────────────────────────────────

class TestMergeRevenueStack:
    def test_arithmetic(self) -> None:
        da = {"annual_revenue_eur": 50000.0, "annual_revenue_eur_per_mw": 50000.0}
        anc = {
            "fcr_annual_eur": 20000.0,
            "afrr_annual_eur": 10000.0,
            "mfrr_annual_eur": 5000.0,
            "total_ancillary_eur": 35000.0,
            "product_revenues": {"FCR-N": 20000.0, "aFRR Up": 10000.0, "mFRR": 5000.0},
        }
        result = merge_revenue_stack(da, anc, power_mw=1.0)
        assert result["total_eur"] == 85000.0
        assert result["headline_total_mode"] == "additive_energy_only"
        assert result["da_arbitrage_eur"] == 50000.0
        assert abs(result["da_pct"] - 58.8) < 0.1
        assert abs(result["ancillary_pct"] - 41.2) < 0.1
        assert result["source_revenues"]["FCR-N"] == 20000.0

    def test_da_only(self) -> None:
        da = {"annual_revenue_eur": 50000.0, "annual_revenue_eur_per_mw": 50000.0}
        anc = {
            "fcr_annual_eur": 0.0,
            "afrr_annual_eur": 0.0,
            "mfrr_annual_eur": 0.0,
            "total_ancillary_eur": 0.0,
            "product_revenues": {},
        }
        result = merge_revenue_stack(da, anc, power_mw=1.0)
        assert result["total_eur"] == 50000.0
        assert result["gross_additive_total_eur"] == 50000.0
        assert result["headline_total_mode"] == "da_only"
        assert result["da_pct"] == 100.0

    def test_total_per_mw_uses_power_when_da_is_zero(self) -> None:
        da = {"annual_revenue_eur": 0.0, "annual_revenue_eur_per_mw": 0.0}
        anc = {
            "fcr_annual_eur": 12000.0,
            "afrr_annual_eur": 6000.0,
            "mfrr_annual_eur": 2000.0,
            "total_ancillary_eur": 20000.0,
            "product_revenues": {"FCR-N": 12000.0, "aFRR Up": 6000.0, "mFRR": 2000.0},
        }

        result = merge_revenue_stack(da, anc, power_mw=4.0)
        assert result["total_eur"] == 20000.0
        assert result["total_per_mw"] == 5000.0

    def test_total_per_mw_uses_explicit_power_not_da_ratio(self) -> None:
        da = {"annual_revenue_eur": 50000.0, "annual_revenue_eur_per_mw": 12345.0}
        anc = {
            "fcr_annual_eur": 10000.0,
            "afrr_annual_eur": 0.0,
            "mfrr_annual_eur": 0.0,
            "total_ancillary_eur": 10000.0,
            "product_revenues": {"FCR-N": 10000.0},
        }

        result = merge_revenue_stack(da, anc, power_mw=2.0)
        assert result["total_eur"] == 60000.0
        assert result["total_per_mw"] == 30000.0

    def test_capacity_ancillary_is_not_default_additive_headline(self) -> None:
        da = {"annual_revenue_eur": 50000.0, "annual_revenue_eur_per_mw": 50000.0}
        anc = {
            "fcr_annual_eur": 10000.0,
            "afrr_annual_eur": 0.0,
            "mfrr_annual_eur": 0.0,
            "total_ancillary_eur": 10000.0,
            "capacity_ancillary_eur": 10000.0,
            "energy_ancillary_eur": 0.0,
            "product_revenues": {"FCR-N": 10000.0},
            "product_revenue_types": {"FCR-N": "capacity"},
        }

        result = merge_revenue_stack(da, anc, power_mw=1.0)

        assert result["total_eur"] == 50000.0
        assert result["gross_additive_total_eur"] == 60000.0
        assert result["standalone_ancillary_eur"] == 10000.0
        assert result["headline_total_mode"] == "conservative_da_primary"
        assert "not added to the headline total" in result["capacity_stack_warning"]


# ── Auto-fetcher registry ──────────────────────────────────────────────────

class TestGetAvailableFetchers:
    def test_fi_has_fetchers(self) -> None:
        fetchers = get_available_fetchers("FI")
        assert len(fetchers) == 2
        names = [f["name"] for f in fetchers]
        assert "FCR-N/D prices" in names
        assert "aFRR prices" in names

    def test_gb_has_fetchers(self) -> None:
        fetchers = get_available_fetchers("GB")
        assert len(fetchers) == 1
        assert fetchers[0]["source"] == "Elexon"

    def test_de_lu_has_fetchers(self) -> None:
        fetchers = get_available_fetchers("DE_LU")
        assert len(fetchers) == 1
        assert fetchers[0]["source"] == "Regelleistung.net"

    def test_unknown_zone_returns_empty(self) -> None:
        assert get_available_fetchers("XX_UNKNOWN") == []

    def test_entsoe_zones_have_imbalance(self) -> None:
        italy_zones = [
            "IT_NORD", "IT_CNOR", "IT_CSUD", "IT_SUD",
            "IT_CALA", "IT_SICI", "IT_SARD",
        ]
        for zone in ["RO", "SE_3", *italy_zones]:
            fetchers = get_available_fetchers(zone)
            assert len(fetchers) >= 1
            assert fetchers[0]["source"] == "ENTSO-E"


class TestRunAutoFetch:
    @patch("src.data_ingestion.fetch_fingrid_fcr_prices")
    @patch("src.data_ingestion.fetch_fingrid_afrr_prices")
    def test_fi_fetch_success(
        self, mock_afrr: MagicMock, mock_fcr: MagicMock,
    ) -> None:
        """Successful FI fetch returns both datasets."""
        idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        mock_fcr.return_value = pd.DataFrame({
            "timestamp": idx,
            "fcr_n_price": [10.0, 11.0, 12.0],
        })
        mock_afrr.return_value = pd.DataFrame({
            "timestamp": idx,
            "afrr_up_price": [5.0, 6.0, 7.0],
        })

        results = run_auto_fetch(
            "FI",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert len(results) == 2

    @patch("src.data_ingestion.fetch_fingrid_fcr_prices")
    @patch("src.data_ingestion.fetch_fingrid_afrr_prices")
    def test_handles_partial_failure(
        self, mock_afrr: MagicMock, mock_fcr: MagicMock,
    ) -> None:
        """One fetcher failing doesn't prevent others from succeeding."""
        idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        mock_fcr.return_value = pd.DataFrame({
            "timestamp": idx,
            "fcr_n_price": [10.0, 11.0, 12.0],
        })
        import requests as _rq
        mock_afrr.side_effect = _rq.Timeout("API timeout")

        results = run_auto_fetch(
            "FI",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert len(results) == 1  # FCR succeeded, aFRR failed

    def test_unknown_zone_returns_empty(self) -> None:
        results = run_auto_fetch(
            "XX_UNKNOWN",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert results == {}

    @patch("src.data_ingestion.fetch_fingrid_fcr_prices")
    def test_auth_error_propagates_from_run_auto_fetch(
        self, mock_fcr: MagicMock,
    ) -> None:
        """DataSourceAuthError must propagate so the sidebar can prompt the
        user to set the missing API key, instead of being swallowed as
        'no data returned'.
        """
        from src.data_ingestion import DataSourceAuthError as _AuthErr
        mock_fcr.side_effect = _AuthErr("Fingrid auth failed")

        with pytest.raises(_AuthErr):
            run_auto_fetch(
                "FI",
                pd.Timestamp("2025-01-01", tz="UTC"),
                pd.Timestamp("2025-01-02", tz="UTC"),
            )

    @patch("src.data_ingestion.fetch_regelleistung_results")
    def test_de_lu_calls_regelleistung(
        self, mock_regelleistung: MagicMock,
    ) -> None:
        mock_regelleistung.return_value = pd.DataFrame({
            "date": ["2025-01-01"],
            "product": ["FCR"],
            "time_block": ["00:00"],
            "capacity_price_eur_mw": [5.50],
            "direction": ["Symmetric"],
        })
        results = run_auto_fetch(
            "DE_LU",
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert mock_regelleistung.call_count == 2  # FCR + aFRR
        assert len(results) >= 1


# ── Co-optimization ──────────────────────────────────────────────────────────

class TestCoOptimizeRevenueSplit:
    def test_pure_da_when_capacity_zero(self) -> None:
        result = co_optimize_revenue_split(
            da_annual_revenue=100000.0,
            capacity_price_eur_mw_h=0.0,
            power_mw=1.0,
        )
        assert result["optimal_fraction"] == 0.0
        assert result["total_revenue"] == pytest.approx(100000.0)

    def test_some_capacity_when_price_positive(self) -> None:
        result = co_optimize_revenue_split(
            da_annual_revenue=100000.0,
            capacity_price_eur_mw_h=5.0,  # reasonable FCR price
            power_mw=1.0,
        )
        assert result["total_revenue"] >= 100000.0
        assert result["optimal_fraction"] >= 0.0

    def test_full_capacity_when_much_higher(self) -> None:
        result = co_optimize_revenue_split(
            da_annual_revenue=10000.0,
            capacity_price_eur_mw_h=50.0,  # very high capacity price
            power_mw=1.0,
        )
        assert result["optimal_fraction"] == 1.0
        assert result["capacity_revenue"] > result["da_revenue"]

    def test_sweep_has_21_rows(self) -> None:
        result = co_optimize_revenue_split(
            da_annual_revenue=100000.0,
            capacity_price_eur_mw_h=5.0,
            power_mw=1.0,
        )
        assert len(result["sweep"]) == 21

    def test_power_scales_capacity_revenue(self) -> None:
        r1 = co_optimize_revenue_split(100000, 5.0, power_mw=1.0)
        r10 = co_optimize_revenue_split(1000000, 5.0, power_mw=10.0)
        # Same capacity price → same optimal fraction
        assert r1["optimal_fraction"] == r10["optimal_fraction"]

    def test_capacity_revenue_uses_configured_hours_and_availability(self) -> None:
        result = co_optimize_revenue_split(
            da_annual_revenue=0.0,
            capacity_price_eur_mw_h=2.0,
            power_mw=3.0,
        )
        assert result["optimal_fraction"] == 1.0
        assert result["capacity_revenue"] == pytest.approx(
            HOURS_PER_YEAR * ANCILLARY_CAPACITY_AVAILABILITY * 2.0 * 3.0,
        )


class TestCapacityProductHelpers:
    """list_capacity_products / capacity_price_for_product feed the cockpit's
    reserve co-opt strategy row."""

    def _df(self) -> pd.DataFrame:
        idx = pd.to_datetime(
            ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z", "2025-01-01T02:00:00Z"],
            utc=True,
        )
        return pd.DataFrame({
            "capacity_price_eur_mw": [10.0, 20.0, float("nan")],
            "energy_price_eur_mwh": [float("nan"), float("nan"), 5.0],
            "product_type": ["FCR", "FCR", "aFRR Up"],
            "direction": ["", "", "Up"],
            "zone": ["DE_LU", "DE_LU", "DE_LU"],
        }, index=idx)

    def test_lists_only_capacity_priced_products(self) -> None:
        products = list_capacity_products(self._df())
        # aFRR Up has only an energy price -> excluded; FCR has capacity rows.
        assert products == ["FCR"]

    def test_empty_and_none_inputs_return_empty_list(self) -> None:
        assert list_capacity_products(None) == []
        assert list_capacity_products(pd.DataFrame()) == []

    def test_price_is_duration_weighted_mean(self) -> None:
        # Two consecutive 1h FCR rows (10, 20) -> duration-weighted mean 15.
        price = capacity_price_for_product(self._df(), "FCR")
        assert price == pytest.approx(15.0)

    def test_price_none_when_product_has_no_capacity(self) -> None:
        assert capacity_price_for_product(self._df(), "aFRR Up") is None
        assert capacity_price_for_product(self._df(), "missing") is None
        assert capacity_price_for_product(None, "FCR") is None

    def test_price_none_for_blank_or_none_product(self) -> None:
        # Defensive guard: the helper is independently callable.
        df = self._df()
        assert capacity_price_for_product(df, None) is None
        assert capacity_price_for_product(df, "") is None
        assert capacity_price_for_product(df, "   ") is None

    def test_price_series_returns_full_timestamped_series(self) -> None:
        # Per-interval counterpart: the product's raw (block-granular) price
        # series, not the duration-weighted mean. Feeds per-interval reserve
        # pricing in the 9.2a/9.2b batches.
        df = self._df()
        series = capacity_price_series_for_product(df, "FCR")
        assert isinstance(series, pd.Series)
        assert list(series.values) == [10.0, 20.0]  # the 2 non-NaN FCR rows
        assert capacity_price_series_for_product(df, "aFRR Up") is None
        assert capacity_price_series_for_product(df, "") is None
        assert capacity_price_series_for_product(None, "FCR") is None
