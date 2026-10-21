"""
Tafra: a minimalist dataframe

Copyright (c) 2020 Derrick W. Turk and David S. Fulford

Author
------
Derrick W. Turk
David S. Fulford

Notes
-----
Key encoding for group-by and joins: each key column becomes non-negative
int64 codes that preserve numpy ==, and several columns combine into one
dense int64 key. The direct_labels_* functions turn that dense key into
per-row group labels; they rely on the dense range build_composite_key
guarantees.
"""

from __future__ import annotations

__all__ = [
    "EncodedColumn",
    "build_composite_key",
    "dense_range_limit",
    "direct_labels_firstseen",
    "direct_labels_sorted",
    "encode_column",
    "encode_join_keys",
]

import math

import numpy as np

from typing import Any, NamedTuple
from typing import cast

from . import _kernels

# Dtype kinds that represent string-like types for encoding purposes.
_STRING_KINDS: frozenset[str] = frozenset("TUSO")

# Dtype kinds whose values (at most 64 bits) map to int64 with equality kept:
# bool, integers, float by bit pattern, datetime and timedelta by view.
_EXACT_INT64_IMAGE_KINDS: frozenset[str] = frozenset("biufMm")


class EncodedColumn(NamedTuple):
    """
    One key column as non-negative int64 codes in [0, cardinality). Equal
    codes mean equal values under numpy `==`, except that in a float,
    datetime or timedelta column every NaN (and every NaT) is one key, as in
    np.unique. In an object column, NaN values are equal only when they are
    the same Python object (hash and `==` with the identity shortcut).
    """

    codes: np.ndarray[Any, Any]
    cardinality: int


def dense_range_limit(row_count: int) -> int:
    """
    Largest code range used directly; a wider range is re-encoded densely.
    The direct-mapped label arrays are sized by the code range, so this
    caps them at a small multiple of the row count (plus a floor for
    small tables).
    """
    return 4 * row_count + 1024


def hash_encode_object(column: np.ndarray[Any, Any]) -> EncodedColumn:
    """
    First-seen codes from a dict codebook. np.unique sorts via argsort,
    which raises TypeError on object arrays containing mixed Python types
    (e.g., str and int), and on StringDType arrays holding None. Semantics
    match _kernels.encode_strings (hash equality, not ordering).
    """
    codebook: dict[Any, int] = {}
    codes = np.empty(len(column), dtype=np.int64)
    for i in range(len(column)):
        value = column[i]
        code = codebook.get(value)
        if code is None:
            code = len(codebook)
            codebook[value] = code
        codes[i] = code
    return EncodedColumn(codes, len(codebook))


def unique_codes(column: np.ndarray[Any, Any]) -> EncodedColumn:
    """Sorted-order codes from np.unique, as a flat int64 array."""
    unique, codes = np.unique(column, return_inverse=True)
    # numpy 2 can return the inverse in the input's shape
    return EncodedColumn(codes.astype(np.int64, copy=False).reshape(-1), len(unique))


def encode_string_column(column: np.ndarray[Any, Any]) -> EncodedColumn:
    """
    Codes that preserve equality for a string or object column. The C kernel
    and the dict codebook give first-seen codes; a fixed-width (U, S) column
    without the C extension gets np.unique's sorted codes. No caller depends
    on code order.
    """
    if _kernels.HAS_ACCEL:
        # astype(object) gives the C encoder a private copy (see _accel.c)
        codes, unique_count = _kernels.encode_strings(column.astype(object))
        return EncodedColumn(codes, int(unique_count))
    if column.dtype.kind in ("O", "T"):
        return hash_encode_object(column)
    return unique_codes(column)


def encode_int64_dense(keys: np.ndarray[Any, Any]) -> EncodedColumn:
    """Dense codes 0..unique_count-1 for int64 keys."""
    if _kernels.HAS_ACCEL:
        codes, unique_count = _kernels.encode_int64(keys)
        return EncodedColumn(codes, int(unique_count))
    return unique_codes(keys)


def integer_key_bits(column: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    An int64 array whose equality matches numpy `==` on a bool, integer,
    datetime, timedelta or float column of at most 64 bits.
    """
    kind = column.dtype.kind
    if kind == "f":
        values = column.astype(np.float64, copy=True)
        values += 0.0  # -0.0 + 0.0 is +0.0, so the two zeros get one key
        values[np.isnan(values)] = np.nan  # one bit pattern for every NaN
        return values.view(np.int64)
    if kind in ("M", "m") or (kind == "u" and column.dtype.itemsize == 8):
        # NaT is INT64_MIN, so all NaT share one key; a uint64 above the
        # int64 range keeps its bit pattern, which preserves equality
        return column.view(np.int64)
    return column.astype(np.int64, copy=False)


def encode_column(column: np.ndarray[Any, Any]) -> EncodedColumn:
    """Encode one key column; its cardinality is at most dense_range_limit(rows)."""
    if len(column) == 0:
        return EncodedColumn(np.array([], dtype=np.int64), 0)
    kind = column.dtype.kind
    if kind in _STRING_KINDS:
        return encode_string_column(column)
    if kind not in _EXACT_INT64_IMAGE_KINDS or column.dtype.itemsize > 8:
        # complex, longdouble, structured: no exact int64 image
        try:
            return unique_codes(column)
        except TypeError:
            return hash_encode_object(column.astype(object))

    bits = integer_key_bits(column)
    if kind != "f":
        low, high = int(bits.min()), int(bits.max())
        if high - low < dense_range_limit(len(bits)):
            if low == 0:
                return EncodedColumn(bits, high + 1)
            # the result lies in [0, high - low], so int64 wraparound in
            # the subtraction cannot change it
            return EncodedColumn(bits - np.int64(low), high - low + 1)
    # float bit patterns are never a narrow range
    return encode_int64_dense(bits)


def build_composite_key(encoded: list[EncodedColumn], row_count: int) -> np.ndarray[Any, Any]:
    """
    Combine encoded key columns into one non-negative int64 key whose
    range is at most dense_range_limit(row_count).

    Positional encoding: key = c0 * N1*N2*... + c1 * N2*... + c2 * ...
    When the product of cardinalities exceeds the dense limit, columns are
    combined one at a time and the running key is re-encoded densely
    whenever its range passes the limit, so no product can overflow.
    """
    limit = dense_range_limit(row_count)
    if len(encoded) == 1:
        return encoded[0].codes
    if row_count == 0:
        return np.array([], dtype=np.int64)

    cardinalities = [column.cardinality for column in encoded]
    if math.prod(cardinalities) <= limit:
        columns = tuple(column.codes for column in encoded)
        if _kernels.HAS_ACCEL:
            return cast(np.ndarray[Any, Any], _kernels.composite_key(columns, tuple(cardinalities)))
        key = np.zeros(row_count, dtype=np.int64)
        multiplier = 1
        for codes, cardinality in zip(reversed(columns), reversed(cardinalities)):
            key += codes * multiplier
            multiplier *= cardinality
        return key

    key, key_cardinality = encoded[0]
    for codes, cardinality in encoded[1:]:
        # both factors are at most `limit`, so this holds below ~5e8 rows
        if key_cardinality * cardinality > 2**62:
            raise ValueError(
                "Composite key space overflow: too many rows to encode a "
                "multi-column key in int64"
            )
        key = key * np.int64(cardinality) + codes
        key_cardinality *= cardinality
        if key_cardinality > limit:
            key, key_cardinality = encode_int64_dense(key)
    return key


def encode_join_keys(
    combined_columns: list[np.ndarray[Any, Any]], left_count: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Encode join key columns with one shared codebook, so equal values on
    the two sides get equal int64 keys. Each column is the left rows
    followed by the right rows, concatenated, which promotes same-kind
    dtypes to a common one; the caller passes only pairs of one dtype kind,
    or a U column with a T column (both text strings, concatenated to T).
    Returns the left and right keys.
    """
    encoded = [encode_column(column) for column in combined_columns]
    key = build_composite_key(encoded, len(combined_columns[0]))
    return key[:left_count], key[left_count:]


def direct_labels_firstseen(
    data: np.ndarray[Any, Any], n_rows: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int]:
    """
    Assign first-seen-order group labels using direct array mapping.
    O(n + max_key) — no sort. Requires non-negative integer keys; keys from
    build_composite_key have max_key < dense_range_limit(rows).

    Returns (labels, first_seen_indices, n_groups).
    """
    sorted_labels, n_groups = direct_labels_sorted(data)
    # reorder to first-seen
    first_pos = np.empty(n_groups, dtype=np.intp)
    first_pos[sorted_labels[::-1]] = np.arange(n_rows - 1, -1, -1)
    order = np.argsort(first_pos)
    rank = np.empty(n_groups, dtype=np.intp)
    rank[order] = np.arange(n_groups, dtype=np.intp)
    labels = rank[sorted_labels]
    return labels, first_pos[order], n_groups


def direct_labels_sorted(data: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], int]:
    """
    Assign sorted-order group labels using direct array mapping.
    O(n + max_key) — no sort. For Transform (order doesn't matter).

    Returns (labels, n_groups).
    """
    max_key = int(data.max())
    seen = np.zeros(max_key + 1, dtype=bool)
    seen[data] = True
    n_groups = int(seen.sum())
    label_map = np.empty(max_key + 1, dtype=np.intp)
    label_map[seen] = np.arange(n_groups, dtype=np.intp)
    return label_map[data], n_groups
