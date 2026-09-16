"""Content identity for session-cached batch results.

A cached result may only be shown while every input that produced it is
unchanged. Identity therefore comes from the full content of the frames handed
to a solver (values, index values, index dtype including its timezone, column
names and dtypes), never from a summary such as first/last date and row count:
a same-length correction must invalidate the result too.

These digests are session cache keys. They are not the public Project Case
fingerprint and carry no cross-version stability promise.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import pandas as pd

from src.project_case.fingerprint import encode_value


def frame_content_hash(value: pd.DataFrame | pd.Series | None) -> str | None:
    """Return a SHA-256 digest of a frame's full content, or ``None``.

    Args:
        value: Frame or series to identify. ``None`` means "not supplied" and
            is kept distinct from an empty frame.

    Returns:
        Hex digest covering row values, index values, index dtype/name and the
        column (or series) names and dtypes.
    """
    if value is None:
        return None
    row_hashes = pd.util.hash_pandas_object(value, index=True, categorize=True)
    digest = hashlib.sha256()
    digest.update(row_hashes.to_numpy(dtype="uint64", copy=False).tobytes())
    if isinstance(value, pd.DataFrame):
        layout: dict[str, object] = {
            "kind": "frame",
            "columns": [str(column) for column in value.columns],
            "dtypes": [str(dtype) for dtype in value.dtypes],
        }
    else:
        layout = {
            "kind": "series",
            "name": None if value.name is None else str(value.name),
            "dtype": str(value.dtype),
        }
    layout["rows"] = len(value)
    layout["index_dtype"] = str(value.index.dtype)
    layout["index_names"] = [
        None if name is None else str(name) for name in value.index.names
    ]
    digest.update(encode_value(layout))
    return digest.hexdigest()


def payload_digest(payload: Mapping[str, object]) -> str:
    """Return a SHA-256 digest of a canonical identity payload.

    Args:
        payload: Mapping of JSON-like values (str, bool, int, finite float,
            None, lists/tuples and nested mappings). Key order is irrelevant.

    Returns:
        Hex digest of the canonical encoding.
    """
    return hashlib.sha256(encode_value(payload)).hexdigest()
