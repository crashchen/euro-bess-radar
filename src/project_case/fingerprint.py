"""``PC-CBOR-F64-v1`` — the locked deterministic canonical serialisation.

This is the single fingerprint spec shared by ``RunResult.input_fingerprint`` and
the nested ``StrategyRunResult.fingerprint`` (contract §4.8, red-line #20). It is a
CBOR-*framed* local profile that deliberately does **not** claim RFC 8949 §4.2
*core* deterministic encoding: core deterministic mandates the shortest float that
preserves value, whereas this profile emits one uniform IEEE-754 binary64 wire type
for every real-valued schema field. Both are lossless; they are different profiles
and must not share a name.

Locked normalisation (§4.8):

* every real-valued schema field → finite IEEE-754 binary64, major type 7 / addl 27
  (``-0.0`` normalises to ``+0.0``; int-vs-float caller syntax cannot change bytes);
* integer schema fields → smallest CBOR unsigned-int head (booleans rejected);
* booleans → CBOR simple ``false``/``true``; text → UTF-8 NFC; ``null`` is present
  for every declared-optional key;
* map keys sorted by their encoded bytes; logical sets emitted as arrays sorted by
  each element's canonical encoded bytes; semantic-order arrays keep declared order.

The encoder dispatches purely on the *Python* type of each value, so the schema
layer (``schema.to_payload``) is responsible for producing correctly-typed Python
values: ``float(...)`` for every real field, ``int(...)`` for the six integer
fields, ``bool`` for flags, and pre-sorted lists for logical sets.
"""

from __future__ import annotations

import hashlib
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

PROFILE = "PC-CBOR-F64-v1"
SCHEMA_VERSION = "project-case-v1"

# The only two object types the envelope wraps.
_OBJECT_TYPES = frozenset({"ProjectCase", "StrategyRunResult"})


def _head(major: int, length: int) -> bytes:
    """CBOR head byte(s) for ``major`` with the smallest unsigned additional info."""
    mt = major << 5
    if length < 24:
        return bytes([mt | length])
    if length < 0x100:
        return bytes([mt | 24, length])
    if length < 0x10000:
        return bytes([mt | 25]) + length.to_bytes(2, "big")
    if length < 0x1_0000_0000:
        return bytes([mt | 26]) + length.to_bytes(4, "big")
    if length < 0x1_0000_0000_0000_0000:
        return bytes([mt | 27]) + length.to_bytes(8, "big")
    raise ValueError("integer exceeds uint64 domain")


def encode_value(value: Any) -> bytes:
    """Encode a single Python value under ``PC-CBOR-F64-v1``.

    Dispatch is strictly on Python type. ``bool`` is checked before ``int``
    because ``bool`` subclasses ``int`` in Python and the two are distinct wire
    types here.
    """
    # bool BEFORE int: booleans are their own simple values, never integers.
    if isinstance(value, bool):
        return b"\xf5" if value else b"\xf4"
    if value is None:
        return b"\xf6"
    if isinstance(value, int):
        if value < 0:
            return _head(1, -1 - value)
        return _head(0, value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite float is not encodable under PC-CBOR-F64-v1")
        if value == 0.0:
            value = 0.0  # normalise -0.0 -> +0.0
        return b"\xfb" + struct.pack(">d", value)
    if isinstance(value, str):
        raw = unicodedata.normalize("NFC", value).encode("utf-8")
        return _head(3, len(raw)) + raw
    # Ordered sequence: caller owns element order (sets are pre-sorted upstream).
    if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        out = _head(4, len(value))
        for item in value:
            out += encode_value(item)
        return out
    if isinstance(value, Mapping):
        encoded = [(encode_value(k), encode_value(v)) for k, v in value.items()]
        encoded.sort(key=lambda kv: kv[0])  # map keys sorted by encoded bytes
        out = _head(5, len(encoded))
        for ek, ev in encoded:
            out += ek + ev
        return out
    raise TypeError(f"unencodable type for PC-CBOR-F64-v1: {type(value)!r}")


def sorted_by_encoding(elements: Sequence[Any]) -> list[Any]:
    """Return ``elements`` as a new list sorted by each element's canonical bytes.

    This realises the "logical sets are arrays sorted by each element's canonical
    encoded bytes" rule for set-valued schema fields.
    """
    return sorted(elements, key=encode_value)


def encode_envelope(object_type: str, payload: Mapping[str, Any]) -> bytes:
    """Encode the four-key schema-normalised envelope (§4.8).

    ``{profile, object_type, schema_version, payload}`` — the object's own
    fingerprint field is excluded from ``payload`` by the schema layer.
    """
    if object_type not in _OBJECT_TYPES:
        raise ValueError(f"object_type must be one of {sorted(_OBJECT_TYPES)}")
    return encode_value(
        {
            "profile": PROFILE,
            "object_type": object_type,
            "schema_version": SCHEMA_VERSION,
            "payload": payload,
        }
    )


def fingerprint_hex(object_type: str, payload: Mapping[str, Any]) -> str:
    """SHA-256 (lowercase 64-char hex) of the canonical envelope bytes."""
    return hashlib.sha256(encode_envelope(object_type, payload)).hexdigest()
