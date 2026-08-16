"""PC-CBOR-F64-v1 fingerprint conformance (contract §4.8, red-line #20)."""

from __future__ import annotations

import dataclasses as dc
import json
import struct
from pathlib import Path

import pytest

import src.project_case.fingerprint as fingerprint_module
from src.project_case import ContractQuoteStatus
from src.project_case.fingerprint import (
    encode_envelope,
    encode_value,
    fingerprint_hex,
    sorted_by_encoding,
)
from src.project_case.schema import _issue_strategy_run_result
from tests import pc_case_fixtures as fx

# --- Locked encoder golden vectors (contract §4.8 table) ---------------------
# fmt: off
PROBE_SRR_HEX = "a4677061796c6f6164a468636173685f657572fb3ff8000000000000686f7074696f6e616cf669617661696c61626c65f56b76616c69645f6461746573816a323032362d30312d30316770726f66696c656e50432d43424f522d4636342d76316b6f626a6563745f7479706571537472617465677952756e526573756c746e736368656d615f76657273696f6e6f70726f6a6563742d636173652d7631"
PROBE_SRR_SHA = "7822bc55d4814c1f6a19f28e6a6572707d972fc8db965ce00f77c8654e7dc2c5"
PROBE_PC_HEX = "a4677061796c6f6164a36473656564182a6d646973636f756e745f72617465fb3fb47ae147ae147b7473747261746567795f66696e6765727072696e747840303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303030306770726f66696c656e50432d43424f522d4636342d76316b6f626a6563745f747970656b50726f6a656374436173656e736368656d615f76657273696f6e7170726f6a6563742d636173652d76312e31"
PROBE_PC_SHA = "fca77bf48c520dd54b8c7c4ba5ab5a65f5460afb473b8421b61efd8cb2212b79"
# fmt: on

# --- Full-object golden vectors (PC-A additionally ships these, §4.8) ---------
GOLDEN_DA_ONLY_SRR = "0e90d9d7610135927559f97665eea1295bdf573031ad1ea1d6a57e2f68071d96"
GOLDEN_DA_ID_RESERVE_SRR = "96cd7d61ec05946f4f152f1e8f4e7864836ecac6138c885b6d30b6eef288fa1f"
GOLDEN_PROJECT_CASE = "c504237ddc558956c31d98834352c485e3989d41765c8a7f3cfd9d29bb9351a2"


def test_probe_vector_strategy_run_result():
    payload = {"available": True, "cash_eur": 1.5, "optional": None, "valid_dates": ["2026-01-01"]}
    raw = encode_envelope("StrategyRunResult", payload)
    assert raw.hex() == PROBE_SRR_HEX
    assert fingerprint_hex("StrategyRunResult", payload) == PROBE_SRR_SHA


def test_probe_vector_project_case():
    payload = {"discount_rate": 0.08, "seed": 42, "strategy_fingerprint": "0" * 64}
    raw = encode_envelope("ProjectCase", payload)
    assert raw.hex() == PROBE_PC_HEX
    assert fingerprint_hex("ProjectCase", payload) == PROBE_PC_SHA


def test_checked_in_v11_project_case_vectors_are_exact():
    path = (
        Path(__file__).parents[1] / "docs" / "design" / "project-case-v1.1-fingerprint-vectors.json"
    )
    vectors = json.loads(path.read_text(encoding="utf-8"))
    assert vectors["profile"] == "PC-CBOR-F64-v1"
    assert vectors["object_type"] == "ProjectCase"
    assert vectors["schema_version"] == "project-case-v1.1"
    for vector in vectors["vectors"]:
        raw = encode_envelope("ProjectCase", vector["payload"])
        assert raw.hex() == vector["encoded_hex"]
        assert fingerprint_hex("ProjectCase", vector["payload"]) == vector["sha256"]


def test_full_object_golden_vectors_are_stable():
    assert fx.da_only_srr().fingerprint() == GOLDEN_DA_ONLY_SRR
    assert fx.da_id_reserve_srr().fingerprint() == GOLDEN_DA_ID_RESERVE_SRR
    assert fx.project_case().input_fingerprint() == GOLDEN_PROJECT_CASE


def test_contract_mutations_change_project_case_fingerprint():
    base = fx.project_case(contract=fx.contract_case())
    assert base.contract_case is not None
    terms = base.contract_case.settlement_terms
    source_variant = dc.replace(
        base,
        contract_case=dc.replace(
            base.contract_case,
            settlement_terms=dc.replace(terms, source="independent quote source"),
        ),
    )
    as_of_variant = dc.replace(
        base,
        contract_case=dc.replace(
            base.contract_case,
            settlement_terms=dc.replace(terms, source_as_of_date="2026-08-15"),
        ),
    )
    status_variant = dc.replace(
        base,
        contract_case=dc.replace(
            base.contract_case,
            settlement_terms=dc.replace(
                terms,
                quote_status=ContractQuoteStatus.USER_ASSERTED_INDICATIVE_QUOTE,
                source_document_sha256="cd" * 32,
            ),
        ),
    )
    # Currency is a cross-object invariant, so mutate the contract, valuation,
    # and producer-issued strategy together to keep the variant valid.
    srr = base.market_case.strategy_run_result
    with _issue_strategy_run_result():
        currency_srr = dc.replace(
            srr,
            currency_basis=dc.replace(srr.currency_basis, target_base_year=2027),
        )
    currency_variant = dc.replace(
        base,
        market_case=dc.replace(base.market_case, strategy_run_result=currency_srr),
        valuation_case=dc.replace(base.valuation_case, base_year=2027),
        contract_case=dc.replace(
            base.contract_case,
            settlement_terms=dc.replace(
                terms,
                currency_basis=dc.replace(terms.currency_basis, target_base_year=2027),
            ),
        ),
    )
    variants = (
        fx.project_case(contract=fx.contract_case(start_year=3)),
        fx.project_case(contract=fx.contract_case(rates=(11.0, 20.0))),
        fx.project_case(contract=fx.contract_case(entitlement_factors=(0.6, 1.0))),
        source_variant,
        as_of_variant,
        status_variant,
        currency_variant,
    )
    assert all(case.input_fingerprint() != base.input_fingerprint() for case in variants)


def test_float_wire_type_is_uniform_f64():
    # int-vs-float caller syntax cannot change a real field's bytes (§4.8).
    assert encode_value(10) != encode_value(10.0)  # 10 is an integer field value
    assert encode_value(10.0) == b"\xfb" + struct.pack(">d", 10.0)
    assert encode_value(1.5)[:1] == b"\xfb"  # major 7 / addl 27, always 8-byte


def test_negative_zero_normalises_to_positive_zero():
    assert encode_value(-0.0) == encode_value(0.0)


def test_bool_is_not_int():
    assert encode_value(True) == b"\xf5"
    assert encode_value(False) == b"\xf4"
    assert encode_value(1) == b"\x01"
    assert encode_value(True) != encode_value(1)


def test_null_present_and_smallest_int_head():
    assert encode_value(None) == b"\xf6"
    assert encode_value(0) == b"\x00"
    assert encode_value(23) == b"\x17"
    assert encode_value(24) == b"\x18\x18"  # 1-byte head
    assert encode_value(42) == b"\x18\x2a"


def test_map_keys_sorted_by_encoded_bytes_regardless_of_insertion_order():
    a = encode_value({"b": 1, "a": 2, "ab": 3})
    b = encode_value({"ab": 3, "a": 2, "b": 1})
    assert a == b  # deterministic ordering
    # "a" (0x61 61) < "b" (0x61 62) < "ab" (0x62 6161) by encoded bytes
    assert a.index(b"\x61a") < a.index(b"\x61b") < a.index(b"\x62ab")


def test_sorted_by_encoding_orders_logical_sets():
    assert sorted_by_encoding(["b", "ab", "a"]) == ["a", "b", "ab"]


def test_non_finite_float_rejected():
    with pytest.raises(ValueError):
        encode_value(float("nan"))
    with pytest.raises(ValueError):
        encode_value(float("inf"))


def test_object_type_must_be_known():
    with pytest.raises(ValueError):
        encode_envelope("Nonsense", {})


def test_object_schema_version_registry_is_process_immutable():
    with pytest.raises(TypeError):
        fingerprint_module._OBJECT_SCHEMA_VERSIONS["ProjectCase"] = "forged"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: {**p, "power_mw": 11.0},
        lambda p: {**p, "zone": "FR"},
        lambda p: {**p, "embedded_vom_cost_eur_mwh": 0.6},
    ],
)
def test_payload_mutation_changes_fingerprint(mutate):
    base = fx.da_only_srr().to_payload()
    assert fingerprint_hex("StrategyRunResult", base) != fingerprint_hex(
        "StrategyRunResult", mutate(base)
    )


def test_capture_rate_change_changes_srr_fingerprint():
    base = fx.da_only_srr()
    mutated = base.to_payload()
    mutated["cash_basis"]["capture"]["rate"] = 0.8
    mutated["adapter_provenance"]["capture_rate"] = 0.8
    assert fingerprint_hex("StrategyRunResult", mutated) != base.fingerprint()


def test_project_case_embeds_nested_srr_digest():
    pc = fx.project_case()
    assert pc.to_payload()["market_case"]["strategy_run_fingerprint"] == GOLDEN_DA_ONLY_SRR


def test_determinism_rebuild_same_hash():
    assert fx.project_case().input_fingerprint() == fx.project_case().input_fingerprint()
