"""
Tafra: a minimalist dataframe

Copyright (c) 2020 Derrick W. Turk and David S. Fulford

Author
------
Derrick W. Turk
David S. Fulford

Notes
-----
Created on April 25, 2020
"""

from __future__ import annotations

__all__ = [
    "GroupBy",
    "Transform",
    "IterateBy",
    "InnerJoin",
    "LeftJoin",
    "percentile",
    "geomean",
    "harmean",
]

import functools
import math
import operator
import warnings
from itertools import chain
import dataclasses as dc

import numpy as np

from typing import Any, Callable, Mapping, Sequence, Iterable, Iterator
from typing import cast

try:
    from ._accel import (  # type: ignore[import-not-found]
        groupby_sum as _c_sum,
        groupby_sum_int64 as _c_sum_int64,
        groupby_mean as _c_mean,
        groupby_var as _c_var,
        groupby_min as _c_min,
        groupby_max as _c_max,
        groupby_min_int64 as _c_min_int64,
        groupby_max_int64 as _c_max_int64,
        inner_join as _c_inner_join,
        left_join as _c_left_join,
        composite_key as _c_composite_key,
        encode_int64 as _c_encode_int64,
        group_indices as _c_group_indices,
        encode_strings as _c_encode_strings,
    )

    _HAS_ACCEL = True
except ImportError:
    _HAS_ACCEL = False

# Dtype kinds that can hold null values (None, NaN, NaT).
# Used by _non_null_mask to skip scanning for non-nullable types.
# 'T' = StringDType (can hold None with na_object=None)
# 'U' (fixed-width unicode <U) is excluded — it cannot hold None.
_NULLABLE_KINDS: frozenset[str] = frozenset("fMmTO")

# Dtype kinds that represent string-like types for encoding purposes.
_STRING_KINDS: frozenset[str] = frozenset("TUSO")

# A vectorized aggregation: (data, labels, n_groups) -> one value per group.
VectorizedAggregation = Callable[
    [np.ndarray[Any, Any], np.ndarray[Any, Any], int], np.ndarray[Any, Any]
]

# Vectorized aggregation functions that can bypass per-group Python loops.
# Maps function identity to (implementation, dtype kinds it accepts). A data
# dtype outside the accepted kinds takes the per-group path, which calls the
# function itself and so refuses exactly what numpy refuses.
_VECTORIZED_AGGS: dict[int, tuple[VectorizedAggregation, frozenset[str]]] = {}

# Kinds numpy reduces arithmetically (sum, mean, var, prod, median, ...).
_ARITHMETIC_KINDS: frozenset[str] = frozenset("fiub")
# Kinds with a total order that np.minimum / np.maximum accept.
_ORDERED_KINDS: frozenset[str] = frozenset("fiubMm")
# Kinds np.ptp accepts (it subtracts, so bool is refused).
_RANGE_KINDS: frozenset[str] = frozenset("fiuMm")
# Kinds for geomean, harmean and percentile.
_SCALAR_STATISTIC_KINDS: frozenset[str] = frozenset("fiu")
# Builtin `sum` keeps an int32 or float32 element type where np.sum widens
# it; the two agree only for these dtypes.
_BUILTIN_SUM_DTYPES: frozenset[np.dtype[Any]] = frozenset(
    np.dtype(name) for name in ("float64", "int64", "uint64", "bool")
)


def _sorted_segments(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Sort data by group labels, return (sorted_data, offsets, counts)."""
    order = np.argsort(labels, kind="stable")
    sorted_data = data[order]
    counts = np.bincount(labels, minlength=n)
    offsets = np.zeros(n + 1, dtype=np.intp)
    np.cumsum(counts, out=offsets[1:])
    return sorted_data, offsets, counts


def _float_result(result: np.ndarray[Any, Any], data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """numpy keeps a float16 / float32 input at its width; other inputs give float64."""
    if data.dtype.kind == "f":
        return result.astype(data.dtype, copy=False)
    return result


def _integer_accumulator(data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """numpy sums and multiplies unsigned kinds in uint64 and int / bool in int64."""
    return data.astype(np.uint64 if data.dtype.kind == "u" else np.int64, copy=False)


def _float_values(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], skip_nan: bool
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Data as float64 with its labels; NaN rows removed when skip_nan."""
    values = data.astype(np.float64, copy=False)
    if skip_nan and data.dtype.kind == "f":
        valid = ~np.isnan(values)
        return values[valid], labels[valid]
    return values, labels


def _vec_sum(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int, skip_nan: bool = False
) -> np.ndarray[Any, Any]:
    if data.dtype.kind == "f":
        if _HAS_ACCEL:
            result = _c_sum(
                labels.astype(np.int64, copy=False),
                data.astype(np.float64, copy=False),
                n,
                skip_nan,
            )
        else:
            values, value_labels = _float_values(data, labels, skip_nan)
            result = np.bincount(value_labels, weights=values, minlength=n)
        return _float_result(result, data)

    # integers sum exactly, wrapping on overflow like numpy
    accumulator = _integer_accumulator(data)
    if _HAS_ACCEL:
        result = _c_sum_int64(labels.astype(np.int64, copy=False), accumulator.view(np.int64), n)
        return cast(np.ndarray[Any, Any], result.view(accumulator.dtype))
    sorted_data, offsets, _ = _sorted_segments(accumulator, labels, n)
    return np.add.reduceat(sorted_data, offsets[:n])


def _vec_nansum(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    return _vec_sum(data, labels, n, skip_nan=True)


def _vec_mean(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int, skip_nan: bool = False
) -> np.ndarray[Any, Any]:
    if _HAS_ACCEL:
        result = _c_mean(
            labels.astype(np.int64, copy=False), data.astype(np.float64, copy=False), n, skip_nan
        )
    else:
        values, value_labels = _float_values(data, labels, skip_nan)
        sums = np.bincount(value_labels, weights=values, minlength=n)
        counts = np.bincount(value_labels, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = sums / counts
    return _float_result(result, data)


def _vec_nanmean(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    return _vec_mean(data, labels, n, skip_nan=True)


def _vec_var(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int, skip_nan: bool = False
) -> np.ndarray[Any, Any]:
    if _HAS_ACCEL:
        result = _c_var(
            labels.astype(np.int64, copy=False), data.astype(np.float64, copy=False), n, skip_nan
        )
    else:
        # two-pass: subtract each group's mean before squaring, which avoids
        # the cancellation of E[x^2] - E[x]^2 on values with a large offset
        values, value_labels = _float_values(data, labels, skip_nan)
        counts = np.bincount(value_labels, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.bincount(value_labels, weights=values, minlength=n) / counts
            deviations = values - means[value_labels]
            squared_sums = np.bincount(value_labels, weights=deviations * deviations, minlength=n)
            result = squared_sums / counts
    return _float_result(result, data)


def _vec_nanvar(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    return _vec_var(data, labels, n, skip_nan=True)


def _vec_std(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    return np.sqrt(_vec_var(data, labels, n))


def _vec_nanstd(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    return np.sqrt(_vec_var(data, labels, n, skip_nan=True))


def _vec_extreme(
    data: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    n: int,
    want_maximum: bool,
    skip_nan: bool,
) -> np.ndarray[Any, Any]:
    """Grouped min / max. NaN and NaT propagate unless skip_nan, as in numpy."""
    kind = data.dtype.kind
    if _HAS_ACCEL and kind == "f":
        c_function = _c_max if want_maximum else _c_min
        result = c_function(
            labels.astype(np.int64, copy=False), data.astype(np.float64, copy=False), n, skip_nan
        )
        return _float_result(result, data)
    if _HAS_ACCEL and kind in ("i", "u", "b"):
        # exact integer kernel; a uint64 keeps its bits and compares unsigned
        is_unsigned_64 = kind == "u" and data.dtype.itemsize == 8
        values = data.view(np.int64) if is_unsigned_64 else data.astype(np.int64, copy=False)
        c_function = _c_max_int64 if want_maximum else _c_min_int64
        result = c_function(labels.astype(np.int64, copy=False), values, n, is_unsigned_64)
        if is_unsigned_64:
            return cast(np.ndarray[Any, Any], result.view(np.uint64))
        return cast(np.ndarray[Any, Any], result.astype(data.dtype, copy=False))
    # datetime / timedelta, and the pure-numpy path
    ufunc: np.ufunc
    if skip_nan:
        ufunc = np.fmax if want_maximum else np.fmin
    else:
        ufunc = np.maximum if want_maximum else np.minimum
    sorted_data, offsets, _ = _sorted_segments(data, labels, n)
    return ufunc.reduceat(sorted_data, offsets[:n])


def _vec_min(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    return _vec_extreme(data, labels, n, want_maximum=False, skip_nan=False)


def _vec_nanmin(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    return _vec_extreme(data, labels, n, want_maximum=False, skip_nan=True)


def _vec_max(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    return _vec_extreme(data, labels, n, want_maximum=True, skip_nan=False)


def _vec_nanmax(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    return _vec_extreme(data, labels, n, want_maximum=True, skip_nan=True)


def _vec_ptp(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    sorted_data, offsets, _ = _sorted_segments(data, labels, n)
    return np.maximum.reduceat(sorted_data, offsets[:n]) - np.minimum.reduceat(
        sorted_data, offsets[:n]
    )


def _vec_prod(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int, skip_nan: bool = False
) -> np.ndarray[Any, Any]:
    if data.dtype.kind == "f":
        values = np.where(np.isnan(data), 1, data) if skip_nan else data
    else:
        values = _integer_accumulator(data)
    sorted_data, offsets, _ = _sorted_segments(values, labels, n)
    return np.multiply.reduceat(sorted_data, offsets[:n])


def _vec_nanprod(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    return _vec_prod(data, labels, n, skip_nan=True)


def _vec_any(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    sorted_data, offsets, _ = _sorted_segments(data.astype(bool), labels, n)
    return np.logical_or.reduceat(sorted_data, offsets[:n])


def _vec_all(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    sorted_data, offsets, _ = _sorted_segments(data.astype(bool), labels, n)
    return np.logical_and.reduceat(sorted_data, offsets[:n])


def _vec_len(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    return np.bincount(labels, minlength=n)


def _vec_count_nonzero(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    return np.bincount(labels[data != 0], minlength=n)


def _vec_median(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int, skip_nan: bool = False
) -> np.ndarray[Any, Any]:
    sorted_data, offsets, _ = _sorted_segments(data, labels, n)
    result = np.empty(n, dtype=np.float64)
    is_float = data.dtype.kind == "f"
    for group_index in range(n):
        segment = np.sort(sorted_data[offsets[group_index] : offsets[group_index + 1]])
        value_count = len(segment)
        # np.sort places NaN last
        if is_float and value_count > 0 and np.isnan(segment[-1]):
            if not skip_nan:
                result[group_index] = np.nan
                continue
            value_count -= int(np.count_nonzero(np.isnan(segment)))
            if value_count == 0:
                result[group_index] = np.nan
                continue
        middle = value_count // 2
        if value_count % 2 == 1:
            result[group_index] = float(segment[middle])
        else:
            result[group_index] = (float(segment[middle - 1]) + float(segment[middle])) / 2.0
    return _float_result(result, data)


def _vec_nanmedian(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    return _vec_median(data, labels, n, skip_nan=True)


def _vec_builtin_sum(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    # builtin sum keeps the element type, unlike np.sum which widens small ints;
    # only the widths where the two agree are registered for this function
    return _vec_sum(data, labels, n)


def _vec_geomean(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    log_sums = np.bincount(labels, weights=np.log(data.astype(float)), minlength=n)
    counts = np.bincount(labels, minlength=n).astype(float)
    return np.exp(log_sums / counts)


def _vec_harmean(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> np.ndarray[Any, Any]:
    recip_sums = np.bincount(labels, weights=1.0 / data.astype(float), minlength=n)
    counts = np.bincount(labels, minlength=n).astype(float)
    return counts / recip_sums


def _make_vec_percentile(
    q: float,
) -> Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any], int], np.ndarray[Any, Any]]:
    """Create a vectorized percentile function for a given quantile."""

    def _vec_pct(
        data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
    ) -> np.ndarray[Any, Any]:
        sorted_data, offsets, counts = _sorted_segments(data, labels, n)
        result = np.empty(n, dtype=float)
        for g in range(n):
            lo = offsets[g]
            hi = offsets[g + 1]
            result[g] = np.percentile(sorted_data[lo:hi], q)
        return _float_result(result, data)

    return _vec_pct


class _PercentileAgg:
    """Callable aggregation for percentile(q). Registered for vectorized fast path."""

    __slots__ = ("q", "_vec_fn")

    def __init__(self, q: float) -> None:
        self.q = q
        self._vec_fn = _make_vec_percentile(q)

    def __call__(self, data: np.ndarray[Any, Any]) -> Any:
        return np.percentile(data, self.q)

    def __repr__(self) -> str:
        return f"percentile({self.q})"


def percentile(q: float) -> _PercentileAgg:
    """
    Create a percentile aggregation function for use in `group_by`.

    Parameters
    ----------
    q: float
        Percentile in range [0, 100].

    Returns
    -------
    agg: callable
        A callable suitable for `group_by` aggregation that also
        hits the vectorized fast path.

    Example
    -------
    >>> tf.group_by(['g'], {'p90': (percentile(90), 'value')})
    """
    return _PercentileAgg(q)


def _vectorized_aggregation(
    fn: Callable[..., Any], dtype: np.dtype[Any]
) -> VectorizedAggregation | None:
    """
    Return the vectorized fast path for `fn` applied to data of `dtype`, or
    None if there is none. `_VECTORIZED_AGGS` holds only module-level
    functions, which are never freed, so their ids are never reused. A
    `_PercentileAgg` carries its own fast path because its id can be reused
    by another object after it is freed.
    """
    if isinstance(fn, _PercentileAgg):
        return fn._vec_fn if dtype.kind in _SCALAR_STATISTIC_KINDS else None
    entry = _VECTORIZED_AGGS.get(id(fn))
    if entry is None:
        return None
    implementation, accepted_kinds = entry
    if dtype.kind not in accepted_kinds:
        return None
    if fn is sum and dtype not in _BUILTIN_SUM_DTYPES:
        return None
    return implementation


def geomean(data: np.ndarray[Any, Any]) -> Any:
    """Geometric mean aggregation for use in `group_by`."""
    return np.exp(np.mean(np.log(data.astype(float))))


def harmean(data: np.ndarray[Any, Any]) -> Any:
    """Harmonic mean aggregation for use in `group_by`."""
    return len(data) / np.sum(1.0 / data.astype(float))


def _register_vectorized() -> None:
    """Register known numpy reducers for vectorized GroupBy."""
    all_kinds = frozenset("?biufcmMOSUVT")
    registrations: list[tuple[Callable[..., Any], VectorizedAggregation, frozenset[str]]] = [
        (np.sum, _vec_sum, _ARITHMETIC_KINDS),
        (np.nansum, _vec_nansum, _ARITHMETIC_KINDS),
        (np.mean, _vec_mean, _ARITHMETIC_KINDS),
        (np.nanmean, _vec_nanmean, _ARITHMETIC_KINDS),
        (np.std, _vec_std, _ARITHMETIC_KINDS),
        (np.nanstd, _vec_nanstd, _ARITHMETIC_KINDS),
        (np.var, _vec_var, _ARITHMETIC_KINDS),
        (np.nanvar, _vec_nanvar, _ARITHMETIC_KINDS),
        (np.min, _vec_min, _ORDERED_KINDS),
        (np.amin, _vec_min, _ORDERED_KINDS),
        (np.nanmin, _vec_nanmin, _ORDERED_KINDS),
        (np.max, _vec_max, _ORDERED_KINDS),
        (np.amax, _vec_max, _ORDERED_KINDS),
        (np.nanmax, _vec_nanmax, _ORDERED_KINDS),
        (np.ptp, _vec_ptp, _RANGE_KINDS),
        (np.prod, _vec_prod, _ARITHMETIC_KINDS),
        (np.nanprod, _vec_nanprod, _ARITHMETIC_KINDS),
        (np.any, _vec_any, _ARITHMETIC_KINDS),
        (np.all, _vec_all, _ARITHMETIC_KINDS),
        (np.median, _vec_median, _ARITHMETIC_KINDS),
        (np.nanmedian, _vec_nanmedian, _ARITHMETIC_KINDS),
        (np.count_nonzero, _vec_count_nonzero, _ARITHMETIC_KINDS),
        (len, _vec_len, all_kinds),
        (sum, _vec_builtin_sum, _ARITHMETIC_KINDS),
        (geomean, _vec_geomean, _SCALAR_STATISTIC_KINDS),
        (harmean, _vec_harmean, _SCALAR_STATISTIC_KINDS),
    ]
    for function, implementation, accepted_kinds in registrations:
        _VECTORIZED_AGGS[id(function)] = (implementation, accepted_kinds)


_register_vectorized()


JOIN_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

# for the passed argument to an aggregation
InitAggregation = Mapping[
    str, Callable[[np.ndarray[Any, Any]], Any] | tuple[Callable[[np.ndarray[Any, Any]], Any], str]
]


# for the result type of IterateBy
GroupDescription = tuple[
    tuple[Any, ...],  # tuple of unique values from group-by columns
    np.ndarray[Any, Any],  # int array of row indices into original tafra for this group
    "Tafra",  # sub-tafra for the group
]


class Union:
    """
    Union two `Tafra` together. Analogy to SQL UNION or
    `pandas.append`. All column names and dtypes must match.
    """

    @staticmethod
    def _validate(left: "Tafra", right: "Tafra") -> None:
        """
        Validate the `Tafra` before applying.
        """
        # These should be unreachable unless attributes were directly modified
        if len(left._data) != len(left._dtypes):
            raise ValueError("This `Tafra` length of data and dtypes do not match")
        if len(right._data) != len(right._dtypes):
            raise ValueError("right `Tafra` length of data and dtypes do not match")

        # ensure same number of columns
        if len(left._data) != len(right._data) or len(left._dtypes) != len(right._dtypes):
            raise ValueError("This `Tafra` column count does not match right `Tafra` column count.")

        # ensure all columns in this `Tafra` exist in right `Tafra`
        # if len() is same AND all columns in this exist in right,
        # do not need to check right `Tafra` columns in this `Tafra`.
        for (data_column, value), (dtype_column, dtype) in zip(
            left._data.items(), left._dtypes.items()
        ):
            if data_column not in right._data or dtype_column not in right._dtypes:
                raise TypeError(
                    f"This `Tafra` column `{data_column}` does not exist in right `Tafra`."
                )

            # Compare user-declared dtypes (metadata = intent).
            # _format_dtype collapses string variants to 'str'.
            elif dtype != right._dtypes[dtype_column]:
                raise TypeError(
                    f"This `Tafra` column `{data_column}` dtype `{dtype}` "
                    f"does not match right `Tafra` dtype `{right._dtypes[dtype_column]}`."
                )

    def apply(self, left: "Tafra", right: "Tafra") -> "Tafra":
        """
        Apply the `Union_` to the `Tafra`.

        Parameters
        ----------
        left: Tafra
            The left `Tafra` to union.

        right: Tafra
            The right `Tafra` to union.

        Returns
        -------
        tafra: Tafra
            The unioned `Tafra`.
        """
        self._validate(left, right)

        return Tafra(
            {column: np.append(value, right._data[column]) for column, value in left._data.items()},
            left._dtypes.copy(),
        )

    def apply_inplace(self, left: "Tafra", right: "Tafra") -> None:
        """
        In-place version.

        Apply the `Union_` to the `Tafra`.

        Parameters
        ----------
        left: Tafra
            The left `Tafra` to union.

        right: Tafra
            The right `Tafra` to union.

        Returns
        -------
        tafra: Tafra
            The unioned `Tafra`.
        """
        self._validate(left, right)

        for column, value in left._data.items():
            left._data[column] = np.append(value, right._data[column])
        left._update_rows()


@dc.dataclass
class GroupSet:
    """
    A `GroupSet` is the set of columns by which we construct our groups.
    """

    # Key columns are encoded as (codes, cardinality): non-negative int64 codes
    # in [0, cardinality). Equal codes mean equal values under numpy `==`,
    # except that every NaN (and every NaT) is one key, as in np.unique. A
    # code range wider than _dense_range_limit is re-encoded densely so the
    # direct-mapped label arrays stay O(rows) in memory.

    @staticmethod
    def _dense_range_limit(row_count: int) -> int:
        """Largest code range used directly; a wider range is re-encoded densely."""
        return 4 * row_count + 1024

    @staticmethod
    def _hash_encode_object(arr: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], int]:
        # np.unique sorts via argsort, which raises TypeError on object arrays
        # containing mixed Python types (e.g., str and int), and on StringDType
        # arrays holding None. Use a dict codebook — semantics match
        # _c_encode_strings (hash equality, not ordering).
        codebook: dict[Any, int] = {}
        codes = np.empty(len(arr), dtype=np.int64)
        for i in range(len(arr)):
            v = arr[i]
            code = codebook.get(v)
            if code is None:
                code = len(codebook)
                codebook[v] = code
            codes[i] = code
        return codes, len(codebook)

    @staticmethod
    def _encode_strings(column: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], int]:
        """First-seen codes for a string or object column."""
        if _HAS_ACCEL:
            # astype(object) gives the C encoder a private copy (see _accel.c)
            codes, unique_count = _c_encode_strings(column.astype(object))
            return codes, int(unique_count)
        if column.dtype.kind in ("O", "T"):
            return GroupSet._hash_encode_object(column)
        unique, codes = np.unique(column, return_inverse=True)
        return codes.astype(np.int64, copy=False).reshape(-1), len(unique)

    @staticmethod
    def _encode_int64_dense(keys: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], int]:
        """Dense codes 0..unique_count-1 for int64 keys."""
        if _HAS_ACCEL:
            codes, unique_count = _c_encode_int64(np.ascontiguousarray(keys, dtype=np.int64))
            return codes, int(unique_count)
        unique, codes = np.unique(keys, return_inverse=True)
        return codes.astype(np.int64, copy=False).reshape(-1), len(unique)

    @staticmethod
    def _integer_key_bits(column: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
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

    @staticmethod
    def _encode_column(column: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], int]:
        """Encode one key column as (codes, cardinality)."""
        if len(column) == 0:
            return np.array([], dtype=np.int64), 0
        kind = column.dtype.kind
        if kind in _STRING_KINDS:
            return GroupSet._encode_strings(column)
        if kind not in "biufMm" or column.dtype.itemsize > 8:
            # complex, longdouble, structured: no exact int64 image
            try:
                unique, codes = np.unique(column, return_inverse=True)
            except TypeError:
                return GroupSet._hash_encode_object(column.astype(object))
            return codes.astype(np.int64, copy=False).reshape(-1), len(unique)

        bits = GroupSet._integer_key_bits(column)
        if kind != "f":
            low, high = int(bits.min()), int(bits.max())
            if high - low < GroupSet._dense_range_limit(len(bits)):
                if low == 0:
                    return bits, high + 1
                # the result lies in [0, high - low], so int64 wraparound in
                # the subtraction cannot change it
                return bits - np.int64(low), high - low + 1
        # float bit patterns are never a narrow range
        return GroupSet._encode_int64_dense(bits)

    @staticmethod
    def _build_composite_key(
        encoded: list[tuple[np.ndarray[Any, Any], int]], row_count: int
    ) -> np.ndarray[Any, Any]:
        """
        Combine encoded key columns into one non-negative int64 key whose
        range is at most _dense_range_limit(row_count).

        Positional encoding: key = c0 * N1*N2*... + c1 * N2*... + c2 * ...
        When the product of cardinalities exceeds the dense limit, columns are
        combined one at a time and the running key is re-encoded densely
        whenever its range passes the limit, so no product can overflow.
        """
        limit = GroupSet._dense_range_limit(row_count)
        if len(encoded) == 1:
            return encoded[0][0]
        if row_count == 0:
            return np.array([], dtype=np.int64)

        cardinalities = [cardinality for _, cardinality in encoded]
        if math.prod(cardinalities) <= limit:
            columns = tuple(codes for codes, _ in encoded)
            if _HAS_ACCEL:
                return cast(np.ndarray[Any, Any], _c_composite_key(columns, tuple(cardinalities)))
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
                key, key_cardinality = GroupSet._encode_int64_dense(key)
        return key

    @staticmethod
    def _encode_join_keys(
        left_cols: list[np.ndarray[Any, Any]],
        right_cols: list[np.ndarray[Any, Any]],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Encode left and right key columns with one shared codebook, so equal
        values on the two sides get equal int64 keys.
        """
        left_count = len(left_cols[0])
        encoded = []
        for lc, rc in zip(left_cols, right_cols):
            l_kind, r_kind = lc.dtype.kind, rc.dtype.kind
            if (l_kind in _STRING_KINDS) != (r_kind in _STRING_KINDS):
                raise TypeError(
                    f"Cannot encode columns with incompatible dtype kinds: "
                    f"left={lc.dtype} (kind={l_kind}), right={rc.dtype} (kind={r_kind})"
                )
            encoded.append(GroupSet._encode_column(np.concatenate([lc, rc])))
        key = GroupSet._build_composite_key(encoded, left_count + len(right_cols[0]))
        key = np.ascontiguousarray(key, dtype=np.int64)
        return key[:left_count], key[left_count:]

    @staticmethod
    def _decode_unique(
        first_seen_indices: np.ndarray[Any, Any],
        col_arrays: list[np.ndarray[Any, Any]],
    ) -> list[tuple[Any, ...]]:
        """Build unique group tuples by indexing original columns at first-seen rows."""
        if len(col_arrays) == 1:
            return [(v,) for v in col_arrays[0][first_seen_indices]]
        else:
            return [tuple(c[idx] for c in col_arrays) for idx in first_seen_indices]

    @staticmethod
    def _direct_labels_firstseen(
        data: np.ndarray[Any, Any], n_rows: int
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int]:
        """
        Assign first-seen-order group labels using direct array mapping.
        O(n + max_key) — no sort. Requires non-negative integer keys; keys from
        _build_composite_key have max_key < _dense_range_limit(rows).

        Returns (labels, first_seen_indices, n_groups).
        """
        max_key = int(data.max())
        # sorted label map
        seen = np.zeros(max_key + 1, dtype=bool)
        seen[data] = True
        n_groups = int(seen.sum())
        sorted_map = np.empty(max_key + 1, dtype=np.intp)
        sorted_map[seen] = np.arange(n_groups)
        sorted_labels = sorted_map[data]
        # reorder to first-seen
        first_pos = np.empty(n_groups, dtype=np.intp)
        first_pos[sorted_labels[::-1]] = np.arange(n_rows - 1, -1, -1)
        order = np.argsort(first_pos)
        rank = np.empty(n_groups, dtype=np.intp)
        rank[order] = np.arange(n_groups, dtype=np.intp)
        labels = rank[sorted_labels]
        return labels, first_pos[order], n_groups

    @staticmethod
    def _direct_labels_sorted(data: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], int]:
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

    @staticmethod
    def _prepare_keys(
        tafra: "Tafra", columns: Iterable[str]
    ) -> tuple[np.ndarray[Any, Any], list[np.ndarray[Any, Any]]]:
        """Encode columns and build composite integer key."""
        cols = list(columns)
        col_arrays = [tafra._data[c] for c in cols]
        encoded = [GroupSet._encode_column(c) for c in col_arrays]
        data = GroupSet._build_composite_key(encoded, tafra._rows)
        return data, col_arrays

    @staticmethod
    def _build_group_indices(
        tafra: "Tafra", columns: Iterable[str]
    ) -> tuple[list[tuple[Any, ...]], list[np.ndarray[Any, Any]]]:
        """
        Build per-group row index arrays in a single pass.

        Returns (unique, group_indices) where:
        - unique: list of tuples of group-by values, in first-seen order
        - group_indices: list of int arrays (row positions for each group)
        """
        data, col_arrays = GroupSet._prepare_keys(tafra, columns)

        if _HAS_ACCEL and data.dtype == np.int64 and tafra._rows >= 50_000:
            first_seen_idx, group_indices_list, n_groups = _c_group_indices(
                np.ascontiguousarray(data)
            )
            group_indices = list(group_indices_list)
        else:
            labels, first_seen_idx, n_groups = GroupSet._direct_labels_firstseen(data, tafra._rows)
            sorted_row_indices = np.argsort(labels, kind="stable")
            counts = np.bincount(labels, minlength=n_groups)
            splits = np.cumsum(counts[:-1])
            group_indices = list(np.split(sorted_row_indices, splits))

        unique: list[tuple[Any, ...]] = GroupSet._decode_unique(first_seen_idx, col_arrays)

        return unique, group_indices

    @staticmethod
    def _build_group_labels(
        tafra: "Tafra", columns: Iterable[str]
    ) -> tuple[list[tuple[Any, ...]], np.ndarray[Any, Any], int]:
        """
        Build per-row group labels (integers 0..n_groups-1) in first-seen order.

        Returns (unique, labels, n_groups) where labels[i] is the group index
        for row i.
        """
        data, col_arrays = GroupSet._prepare_keys(tafra, columns)

        labels, first_seen_idx, n_groups = GroupSet._direct_labels_firstseen(data, tafra._rows)

        unique: list[tuple[Any, ...]] = GroupSet._decode_unique(first_seen_idx, col_arrays)

        return unique, labels, n_groups

    @staticmethod
    def _validate(tafra: "Tafra", columns: Iterable[str]) -> None:  # pragma: no cover
        """
        Validate the `Tafra` before applying.
        """
        if tafra._rows < 1:
            raise ValueError("No rows exist in `tafra`.")
        tafra._validate_columns(columns)


@dc.dataclass
class AggMethod(GroupSet):
    """
    Basic methods for aggregations over a data table.
    """

    group_by_cols: Iterable[str]
    aggregation: dc.InitVar[InitAggregation]
    _aggregation: Mapping[str, tuple[Callable[[np.ndarray[Any, Any]], Any], str]] = dc.field(
        init=False
    )
    iter_fn: Mapping[str, Callable[[np.ndarray[Any, Any]], Any]]

    def __post_init__(self, aggregation: InitAggregation) -> None:
        self._aggregation = dict()
        for rename, agg in aggregation.items():
            if callable(agg):
                self._aggregation[rename] = (agg, rename)
            elif isinstance(agg, Sequence) and len(agg) == 2 and callable(agg[0]):
                self._aggregation[rename] = agg
            else:
                raise ValueError(f"{rename}: {agg} is not a valid aggregation argument")

        for rename, agg in self.iter_fn.items():
            if not callable(agg):
                raise ValueError(f"{rename}: {agg} is not a valid aggregation argument")

    @staticmethod
    def _promoted_dtype(results: list[np.ndarray[Any, Any]]) -> np.dtype[Any]:
        """
        numpy's promoted dtype of the per-group results (`np.promote_types`
        over their dtypes), or object when they share none, e.g. str and float.
        """
        unique_dtypes = list(dict.fromkeys(result.dtype for result in results))
        try:
            return functools.reduce(np.promote_types, unique_dtypes)
        except TypeError:
            return np.dtype(object)

    @staticmethod
    def _column_from_group_results(values: list[Any]) -> np.ndarray[Any, Any]:
        """
        One column from one result per group. Scalar results take their
        promoted dtype; if any result is not a scalar, the column is object.
        """
        result_types = set(map(type, values))
        if len(result_types) == 1 and issubclass(
            result_types.pop(), (np.generic, bool, int, float, str)
        ):
            # one scalar type: np.array promotes exactly as np.promote_types
            # (datetime units, string widths), and gives object for a Python
            # int beyond int64, as np.asarray does for that value alone
            return np.array(values)
        arrays = [np.asarray(value) for value in values]
        if all(array.ndim == 0 for array in arrays):
            dtype = AggMethod._promoted_dtype(arrays)
        else:
            dtype = np.dtype(object)
        column = np.empty(len(values), dtype=dtype)
        for i, value in enumerate(values):
            column[i] = value
        return column

    def iter_fn_factory(
        self, fn: Callable[[], np.ndarray[Any, Any]]
    ) -> dict[str, np.ndarray[Any, Any]]:
        return {rename: fn() for rename in self.iter_fn.keys()}

    def apply(self, tafra: "Tafra") -> "Tafra":
        raise NotImplementedError


class GroupBy(AggMethod):
    """
    Aggregation by a set of unique values.

    Analogy to SQL `GROUP BY`, not `pandas.DataFrame.groupby()`.

    Parameters
    ----------
    group_by_cols: Iterable[str]
        The column names to group by.

    aggregation: Mapping[str, Callable[[np.ndarray], Any] | \
        Optional. tuple[Callable[[np.ndarray], Any], str]]
        A mapping for columns and aggregation functions. Should be
        given as {'column': fn} or {'new_column': (fn, 'column')}.

    iter_fn: Mapping[str, Callable[[np.ndarray], Any]]
        Optional. A mapping for new columns names to the function to apply to
        the enumeration. Should be given as {'new_column': fn}.
    """

    def apply(self, tafra: "Tafra") -> "Tafra":
        """
        Apply the `GroupBy` to the `Tafra`.

        Parameters
        ----------
        tafra: Tafra
            The tafra to apply the operation to.

        Returns
        -------
        tafra: Tafra
            The aggregated `Tafra`.
        """
        self._validate(
            tafra, (*self.group_by_cols, *(col for (_, col) in self._aggregation.values()))
        )

        # check if all aggregations can be vectorized
        vectorized_functions = {
            rename: _vectorized_aggregation(fn, tafra._data[col].dtype)
            for rename, (fn, col) in self._aggregation.items()
        }
        all_vectorized = not self.iter_fn and all(
            vec_fn is not None for vec_fn in vectorized_functions.values()
        )

        if all_vectorized and self._aggregation:
            # fast path: vectorized aggregation via labels, no per-group loop
            unique, labels, n_groups = self._build_group_labels(tafra, self.group_by_cols)

            result: dict[str, np.ndarray[Any, Any]] = {}
            for i, col in enumerate(self.group_by_cols):
                vals = tafra._data[col]
                # pick one representative value per group
                first_occurrence = np.empty(n_groups, dtype=np.intp)
                first_occurrence[labels] = np.arange(len(labels))
                # overwrite gives last, we want first — reverse
                first_occurrence[labels[::-1]] = np.arange(len(labels) - 1, -1, -1)
                result[col] = vals[first_occurrence]

            for rename, (fn, col) in self._aggregation.items():
                vec_fn = vectorized_functions[rename]
                assert vec_fn is not None
                result[rename] = vec_fn(tafra._data[col], labels, n_groups)

            return Tafra(result)

        # standard path: per-group loop
        unique, group_indices = self._build_group_indices(tafra, self.group_by_cols)
        n_groups = len(unique)

        result = {
            col: np.empty(n_groups, dtype=tafra._data[col].dtype) for col in self.group_by_cols
        }
        group_results: dict[str, list[Any]] = {rename: [] for rename in self._aggregation}
        iter_fn = self.iter_fn_factory(lambda: np.ones(n_groups, dtype=int))

        for i, (u, rows) in enumerate(zip(unique, group_indices)):
            for val, col in zip(u, self.group_by_cols):
                result[col][i] = val

            for rename, (fn, col) in self._aggregation.items():
                group_results[rename].append(fn(tafra._data[col][rows]))

            for rename, fn in self.iter_fn.items():
                iter_fn[rename][i] = fn(np.full(len(rows), i, dtype=int))

        for rename, values in group_results.items():
            result[rename] = self._column_from_group_results(values)
        result.update(iter_fn)
        return Tafra(result)


class Transform(AggMethod):
    """
    Apply a function to each unique set of values and join to the original table.

    Analogy to `pandas.DataFrame.groupby().transform()`,
    i.e. a SQL `GROUP BY` and `LEFT JOIN` back to the original table.

    Parameters
    ----------
    group_by_cols: Iterable[str]
        The column names to group by.

    aggregation: Mapping[str, Callable[[np.ndarray], Any] | \
    tuple[Callable[[np.ndarray], Any], str]]
        Optional. A mapping for columns and aggregation functions. Should be
        given as {'column': fn} or {'new_column': (fn, 'column')}.

    iter_fn: Mapping[str, Callable[[np.ndarray], Any]]
        Optional. A mapping for new columns names to the function to apply to
        the enumeration. Should be given as {'new_column': fn}.
    """

    def apply(self, tafra: "Tafra") -> "Tafra":
        """
        Apply the `Transform` to the `Tafra`.

        Parameters
        ----------
        tafra: Tafra
            The tafra to apply the operation to.

        Returns
        -------
        tafra: Tafra
            The transformed `Tafra`.
        """
        self._validate(
            tafra, (*self.group_by_cols, *(col for (_, col) in self._aggregation.values()))
        )

        # check if all aggregations can be vectorized
        vectorized_functions = {
            rename: _vectorized_aggregation(fn, tafra._data[col].dtype)
            for rename, (fn, col) in self._aggregation.items()
        }
        all_vectorized = not self.iter_fn and all(
            vec_fn is not None for vec_fn in vectorized_functions.values()
        )

        if all_vectorized and self._aggregation:
            # fast path: compute per-group aggregates, broadcast via labels
            # Transform doesn't need first-seen order — use direct mapping
            data, _ = self._prepare_keys(tafra, self.group_by_cols)
            labels, n_groups = self._direct_labels_sorted(data)

            result: dict[str, np.ndarray[Any, Any]] = {}
            for col in self.group_by_cols:
                result[col] = tafra._data[col].copy()

            for rename, (fn, col) in self._aggregation.items():
                vec_fn = vectorized_functions[rename]
                assert vec_fn is not None
                group_values = vec_fn(tafra._data[col], labels, n_groups)
                result[rename] = group_values[labels]

            return Tafra(result)

        # standard path: per-group loop
        unique, group_indices = self._build_group_indices(tafra, self.group_by_cols)

        result = {col: tafra._data[col].copy() for col in self.group_by_cols}
        group_results: dict[str, list[np.ndarray[Any, Any]]] = {
            rename: [] for rename in self._aggregation
        }
        iter_fn = self.iter_fn_factory(lambda: np.ones(tafra._rows, dtype=int))

        for i, (u, rows) in enumerate(zip(unique, group_indices)):
            for rename, (fn, col) in self._aggregation.items():
                group_results[rename].append(np.asarray(fn(tafra._data[col][rows])))

            for rename, fn in self.iter_fn.items():
                iter_fn[rename][rows] = fn(np.full(len(rows), i, dtype=int))

        # a group's result is one value broadcast to its rows, or one value per row
        for rename, arrays in group_results.items():
            column = np.empty(tafra._rows, dtype=self._promoted_dtype(arrays))
            for rows, array in zip(group_indices, arrays):
                column[rows] = array
            result[rename] = column
        result.update(iter_fn)
        return Tafra(result)


@dc.dataclass
class IterateBy(GroupSet):
    """
    A generator that yields a `Tafra` for each set of unique values.

    Analogy to `pandas.DataFrame.groupby()`, i.e. an Sequence of `Tafra` objects.
    Yields tuples of ((unique grouping values, ...), row indices array, subset tafra)

    Parameters
    ----------
    group_by_cols: Iterable[str]
        The column names to group by.
    """

    group_by_cols: Iterable[str]

    def apply(self, tafra: "Tafra") -> Iterator[GroupDescription]:
        """
        Apply the `IterateBy` to the `Tafra`.

        Parameters
        ----------
        tafra: Tafra
            The tafra to apply the operation to.

        Returns
        -------
        tafras: Iterator[GroupDescription]
            An iterator over the grouped `Tafra`.
        """
        self._validate(tafra, self.group_by_cols)
        unique, group_indices = self._build_group_indices(tafra, self.group_by_cols)

        for u, rows in zip(unique, group_indices):
            yield (u, rows, tafra._ndindex(rows))


@dc.dataclass
class Join(GroupSet):
    """
    Base class for SQL-like JOINs.
    """

    on: Iterable[tuple[str, str, str]]
    select: Iterable[str]

    def __post_init__(self) -> None:
        # Materialize iterables to prevent generator exhaustion on reuse.
        self.on = list(self.on)
        self.select = list(self.select)

    def _validate_dtypes(
        self, l_table: "Tafra", r_table: "Tafra"
    ) -> tuple[dict[str, np.dtype[Any]], dict[str, np.dtype[Any]]]:
        """Validate join key dtypes and compute promotions.

        Returns (left_promotions, right_promotions) — separate dicts to
        avoid key collisions when the same column name appears on both
        sides across different predicates. Does NOT mutate the input tables.
        """
        left_promos: dict[str, np.dtype[Any]] = {}
        right_promos: dict[str, np.dtype[Any]] = {}
        for l_column, r_column, _ in self.on:
            l_dtype = l_table._dtypes[l_column]
            r_dtype = r_table._dtypes[r_column]

            if l_dtype == r_dtype:
                continue

            l_base = Tafra._reduce_dtype(l_dtype)
            r_base = Tafra._reduce_dtype(r_dtype)
            if l_base == r_base == "str":
                continue

            l_np = np.dtype(l_dtype)
            r_np = np.dtype(r_dtype)
            if l_np.kind == r_np.kind:
                promoted = np.result_type(l_np, r_np)
                if l_np != promoted:
                    left_promos[l_column] = promoted
                if r_np != promoted:
                    right_promos[r_column] = promoted
            else:
                raise TypeError(
                    f"This `Tafra` column `{l_column}` dtype `{l_dtype}` "
                    f"does not match other `Tafra` dtype `{r_dtype}`."
                )
        return left_promos, right_promos

    @staticmethod
    def _non_null_mask(cols: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any] | None:
        """Build a boolean mask that is True for rows with no null in any key column.

        Returns None if all columns are non-nullable types (int, uint, bool),
        meaning no nulls are possible and no mask is needed.
        """
        # Fast path: non-nullable types cannot hold nulls — skip entirely.
        if not any(c.dtype.kind in _NULLABLE_KINDS for c in cols):
            return None

        n = len(cols[0])
        valid = np.ones(n, dtype=bool)
        for c in cols:
            kind = c.dtype.kind
            if kind == "f":
                valid &= ~np.isnan(c)
            elif kind in ("M", "m"):
                valid &= ~np.isnat(c)
            elif kind == "T":
                # StringDType(na_object=None): None is the null sentinel
                valid &= np.array([x is not None for x in c], dtype=bool)
            elif kind == "O":
                # Object arrays can hold None, NaN, and NaT simultaneously
                def _is_valid(x: Any) -> bool:
                    if x is None:
                        return False
                    if isinstance(x, (float, np.floating)):
                        return not np.isnan(x)
                    if isinstance(x, (np.datetime64, np.timedelta64)):
                        return not np.isnat(x)
                    return True

                valid &= np.array([_is_valid(x) for x in c], dtype=bool)
        return valid

    @staticmethod
    def _validate_ops(ops: Iterable[str]) -> None:
        for op in ops:
            _op = JOIN_OPS.get(op, None)
            if _op is None:
                raise TypeError(f"The operator {op} is not valid.")

    @staticmethod
    def _sort_merge_indices(
        left_key: np.ndarray[Any, Any],
        right_key: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Compute inner-join index pairs using sort-merge.
        Returns (left_indices, right_indices) as intp arrays.
        """
        right_order = np.argsort(right_key, kind="stable")
        right_sorted = right_key[right_order]

        left_lo = np.searchsorted(right_sorted, left_key, side="left")
        left_hi = np.searchsorted(right_sorted, left_key, side="right")
        counts = left_hi - left_lo

        total = int(counts.sum())
        if total == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        # left indices: repeat each left row index by its match count
        li = np.repeat(np.arange(len(left_key), dtype=np.intp), counts)

        # right indices: for each left row, enumerate its match range
        # build offsets into right_sorted for each output row
        offsets = np.repeat(left_lo, counts)
        group_starts = np.cumsum(counts) - counts
        within = np.arange(total, dtype=np.intp) - np.repeat(group_starts, counts)
        ri = right_order[offsets + within]

        return li, ri

    def _resolve_join_cols(
        self, left_t: "Tafra", right_t: "Tafra"
    ) -> tuple[list[str], dict[str, str]]:
        """Compute deduplicated output columns and dtypes.

        Validates that all select columns exist in at least one table.
        """
        # Use a frozenset for O(1) membership testing.
        # select is already materialized to a list in __post_init__.
        select_set = frozenset(self.select)

        # Validate select columns exist
        if select_set:
            valid_cols = set(left_t._data.keys()) | set(right_t._data.keys())
            invalid = select_set - valid_cols
            if invalid:
                raise KeyError(
                    f"Column(s) {tuple(sorted(invalid))} in `select` not found in either table."
                )

        seen_cols: dict[str, None] = {}
        for c in chain(left_t._data.keys(), right_t._data.keys()):
            if not select_set or c in select_set:
                seen_cols[c] = None
        join_cols = list(seen_cols.keys())
        dtypes: dict[str, str] = {
            c: d
            for c, d in chain(right_t._dtypes.items(), left_t._dtypes.items())
            if c in seen_cols
        }
        return join_cols, dtypes

    def apply(self, left_t: "Tafra", right_t: "Tafra") -> "Tafra":
        raise NotImplementedError


class InnerJoin(Join):
    """
    An inner join.

    Analogy to SQL INNER JOIN, or `pandas.merge(..., how='inner')`,

    Parameters
    ----------
    on: Iterable[tuple[str, str, str]]
        The columns and operator to join on. Should be given as
        ('left column', 'right column', 'op') Valid ops are:

        '==' : equal to
        '!=' : not equal to
        '<'  : less than
        '<=' : less than or equal to
        '>'  : greater than
        '>=' : greater than or equal to

    select: Iterable[str] = []
        The columns to return. If not given, all unique columns names
        are returned. If the column exists in both `Tafra`,
        prefers the left over the right.
    """

    def apply(self, left_t: "Tafra", right_t: "Tafra") -> "Tafra":
        """
        Apply the `InnerJoin` to the `Tafra`.

        Parameters
        ----------
        left_t: Tafra
            The left tafra to join.

        right_t: Tafra
            The right tafra to join.

        Returns
        -------
        tafra: Tafra
            The joined `Tafra`.
        """
        # Validate structure — column names and operators (works even on empty tables)
        left_cols, right_cols, ops = list(zip(*self.on))
        left_t._validate_columns(left_cols)
        right_t._validate_columns(right_cols)
        left_promos, right_promos = self._validate_dtypes(left_t, right_t)
        self._validate_ops(ops)

        join_cols, dtypes = self._resolve_join_cols(left_t, right_t)

        # Empty table shortcut — after validation so schema errors aren't hidden
        if left_t._rows < 1 or right_t._rows < 1:
            warnings.warn(
                "Join: one or both tables have zero rows. Returning shortcut result.",
                stacklevel=2,
            )
            return Tafra(
                {
                    c: np.array(
                        [],
                        dtype=left_t._data[c].dtype
                        if c in left_t._data
                        else right_t._data[c].dtype,
                    )
                    for c in join_cols
                },
                dtypes,
            )

        all_equi = all(op_str == "==" for _, _, op_str in self.on)

        if all_equi:
            # Build key arrays — apply promotions on copies, not originals
            left_cols_data = [
                left_t._data[lc].astype(left_promos[lc], copy=False)
                if lc in left_promos
                else left_t._data[lc]
                for lc, _, _ in self.on
            ]
            right_cols_data = [
                right_t._data[rc].astype(right_promos[rc], copy=False)
                if rc in right_promos
                else right_t._data[rc]
                for _, rc, _ in self.on
            ]

            # NULL != NULL: filter out rows with nulls in key columns
            left_valid = self._non_null_mask(left_cols_data)
            right_valid = self._non_null_mask(right_cols_data)
            has_left_nulls = left_valid is not None and not left_valid.all()
            has_right_nulls = right_valid is not None and not right_valid.all()

            if has_left_nulls or has_right_nulls:
                # Synthesize all-True mask for the non-nullable side
                if left_valid is None:
                    left_valid = np.ones(len(left_cols_data[0]), dtype=bool)
                if right_valid is None:
                    right_valid = np.ones(len(right_cols_data[0]), dtype=bool)
                left_cols_filt = [c[left_valid] for c in left_cols_data]
                right_cols_filt = [c[right_valid] for c in right_cols_data]
                left_orig_idx = np.where(left_valid)[0]
                right_orig_idx = np.where(right_valid)[0]
            else:
                left_cols_filt = left_cols_data
                right_cols_filt = right_cols_data

            left_key, right_key = GroupSet._encode_join_keys(left_cols_filt, right_cols_filt)

            if _HAS_ACCEL:
                li, ri = _c_inner_join(left_key, right_key)
            else:
                li, ri = self._sort_merge_indices(left_key, right_key)

            # Map back to original row positions only if nulls were filtered
            if (has_left_nulls or has_right_nulls) and len(li) > 0:
                li = left_orig_idx[li]
                ri = right_orig_idx[ri]

            if len(li) == 0:
                return Tafra(
                    {
                        c: np.array(
                            [],
                            dtype=left_t._data[c].dtype
                            if c in left_t._data
                            else right_t._data[c].dtype,
                        )
                        for c in join_cols
                    },
                    dtypes,
                )

            result: dict[str, np.ndarray[Any, Any]] = {}
            for c in join_cols:
                if c in left_t._data:
                    result[c] = left_t._data[c][li]
                else:
                    result[c] = right_t._data[c][ri]

            return Tafra(result, dtypes)

        else:
            _on = tuple((left_col, right_col, JOIN_OPS[op]) for left_col, right_col, op in self.on)
            right_rows = np.empty(right_t._rows, dtype=bool)
            join: dict[str, list[Any]] = {c: [] for c in join_cols}

            # Precompute null masks for key columns (SQL: NULL never matches)
            left_key_cols = [left_t._data[lc] for lc, _, _ in self.on]
            right_key_cols = [right_t._data[rc] for _, rc, _ in self.on]
            left_null = self._non_null_mask(left_key_cols)
            right_null = self._non_null_mask(right_key_cols)

            for i in range(left_t._rows):
                # Skip rows with null keys — NULL never matches
                if left_null is not None and not left_null[i]:
                    continue

                right_rows[:] = True
                # Exclude right rows with null keys
                if right_null is not None:
                    right_rows &= right_null
                for left_col, right_col, op in _on:
                    right_rows &= op(left_t._data[left_col][i], right_t._data[right_col])

                right_count = int(np.sum(right_rows))
                if right_count <= 0:
                    continue

                for column in join_cols:
                    if column in left_t._data:
                        join[column].extend([left_t._data[column][i]] * right_count)
                    elif column in right_t._data:
                        join[column].extend(right_t._data[column][right_rows])

            return Tafra({c: np.array(v) for c, v in join.items()}, dtypes)


class LeftJoin(Join):
    """
    A left join.

    Analogy to SQL LEFT JOIN, or `pandas.merge(..., how='left')`,

    Parameters
    ----------
    on: Iterable[tuple[str, str, str]]
        The columns and operator to join on. Should be given as
        ('left column', 'right column', 'op') Valid ops are:

        '==' : equal to
        '!=' : not equal to
        '<'  : less than
        '<=' : less than or equal to
        '>'  : greater than
        '>=' : greater than or equal to

    select: Iterable[str] = []
        The columns to return. If not given, all unique columns names
        are returned. If the column exists in both `Tafra`,
        prefers the left over the right.
    """

    @staticmethod
    def _left_join_indices(
        left_key: np.ndarray[Any, Any],
        right_key: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], bool]:
        """
        Compute left-join index pairs using sort-merge.
        Unmatched left rows get right index = -1.
        Returns (left_indices, right_indices, has_null).
        """
        right_order = np.argsort(right_key, kind="stable")
        right_sorted = right_key[right_order]

        left_lo = np.searchsorted(right_sorted, left_key, side="left")
        left_hi = np.searchsorted(right_sorted, left_key, side="right")
        counts = left_hi - left_lo

        # unmatched left rows get count=0 → force to 1 (with sentinel)
        unmatched = counts == 0
        has_null = bool(np.any(unmatched))
        out_counts = np.where(unmatched, 1, counts)

        total = int(out_counts.sum())
        li = np.repeat(np.arange(len(left_key), dtype=np.intp), out_counts)

        # right indices
        ri = np.empty(total, dtype=np.intp)
        pos = 0
        if has_null:
            # mixed matched/unmatched — fill per group
            for i in range(len(left_key)):
                c = int(counts[i])
                if c > 0:
                    ri[pos : pos + c] = right_order[left_lo[i] : left_hi[i]]
                    pos += c
                else:
                    ri[pos] = -1
                    pos += 1
        else:
            # all matched — fully vectorized
            offsets = np.repeat(left_lo, counts)
            group_starts = np.cumsum(counts) - counts
            within = np.arange(total, dtype=np.intp) - np.repeat(group_starts, counts)
            ri = right_order[offsets + within]

        return li, ri, has_null

    def apply(self, left_t: "Tafra", right_t: "Tafra") -> "Tafra":
        """
        Apply the `LeftJoin` to the `Tafra`.

        Parameters
        ----------
        left_t: Tafra
            The left tafra to join.

        right_t: Tafra
            The right tafra to join.

        Returns
        -------
        tafra: Tafra
            The joined `Tafra`.
        """
        # Validate structure — column names and operators (works even on empty tables)
        left_cols, right_cols, ops = list(zip(*self.on))
        left_t._validate_columns(left_cols)
        right_t._validate_columns(right_cols)
        left_promos, right_promos = self._validate_dtypes(left_t, right_t)
        self._validate_ops(ops)

        join_cols, dtypes = self._resolve_join_cols(left_t, right_t)

        # Empty table shortcuts — after validation
        if left_t._rows < 1:
            warnings.warn(
                "Join: one or both tables have zero rows. Returning shortcut result.",
                stacklevel=2,
            )
            return Tafra(
                {
                    c: np.array(
                        [],
                        dtype=left_t._data[c].dtype
                        if c in left_t._data
                        else right_t._data[c].dtype,
                    )
                    for c in join_cols
                },
                dtypes,
            )
        if right_t._rows < 1:
            warnings.warn(
                "Join: one or both tables have zero rows. Returning shortcut result.",
                stacklevel=2,
            )
            n_left = left_t._rows
            shortcut: dict[str, np.ndarray[Any, Any]] = {}
            for c in join_cols:
                if c in left_t._data:
                    shortcut[c] = left_t._data[c].copy()
                else:
                    col_kind = right_t._data[c].dtype.kind
                    if col_kind in ("T", "U"):
                        out = np.empty(
                            n_left,
                            dtype=np.dtypes.StringDType(na_object=None),  # type: ignore[call-arg]
                        )
                        out[:] = None  # type: ignore[assignment]
                        dtypes[c] = "str"
                    elif col_kind == "f":
                        out = np.full(n_left, np.nan, dtype=right_t._data[c].dtype)
                    elif col_kind in ("M", "m"):
                        nat = np.array("NaT", dtype=right_t._data[c].dtype).item()
                        out = np.full(n_left, nat, dtype=right_t._data[c].dtype)
                    else:
                        warnings.warn(
                            f"Left join: column '{c}' "
                            f"(dtype {right_t._data[c].dtype}) "
                            f"has unmatched rows and no native null "
                            f"representation. Dtype has been cast to "
                            f"object. Use .astype(float) if NaN "
                            f"semantics are needed.",
                            stacklevel=3,
                        )
                        out = cast(
                            np.ndarray[Any, Any],
                            np.empty(n_left, dtype=object),
                        )
                        out[:] = None  # type: ignore[assignment]
                        dtypes[c] = "object"
                    shortcut[c] = out
            return Tafra(shortcut, dtypes)
        all_equi = all(op_str == "==" for _, _, op_str in self.on)

        if all_equi:
            # Build key arrays — apply promotions on copies, not originals
            left_cols_data = [
                left_t._data[lc].astype(left_promos[lc], copy=False)
                if lc in left_promos
                else left_t._data[lc]
                for lc, _, _ in self.on
            ]
            right_cols_data = [
                right_t._data[rc].astype(right_promos[rc], copy=False)
                if rc in right_promos
                else right_t._data[rc]
                for _, rc, _ in self.on
            ]

            # NULL != NULL: filter out rows with nulls in key columns
            left_valid = self._non_null_mask(left_cols_data)
            right_valid = self._non_null_mask(right_cols_data)
            has_left_nulls = left_valid is not None and not left_valid.all()
            has_right_nulls = right_valid is not None and not right_valid.all()

            if has_left_nulls or has_right_nulls:
                # Synthesize all-True mask for the non-nullable side
                if left_valid is None:
                    left_valid = np.ones(len(left_cols_data[0]), dtype=bool)
                if right_valid is None:
                    right_valid = np.ones(len(right_cols_data[0]), dtype=bool)
                left_cols_filt = [c[left_valid] for c in left_cols_data]
                right_cols_filt = [c[right_valid] for c in right_cols_data]
                left_orig_idx = np.where(left_valid)[0]
                right_orig_idx = np.where(right_valid)[0]
                left_null_idx = np.where(~left_valid)[0]
            else:
                left_cols_filt = left_cols_data
                right_cols_filt = right_cols_data
                left_null_idx = np.array([], dtype=np.intp)

            left_key, right_key = GroupSet._encode_join_keys(left_cols_filt, right_cols_filt)

            if _HAS_ACCEL:
                li, ri, has_null = _c_left_join(left_key, right_key)
            else:
                li, ri, has_null = self._left_join_indices(left_key, right_key)

            # Map back to original row positions only if nulls were filtered
            if (has_left_nulls or has_right_nulls) and len(li) > 0:
                li = left_orig_idx[li]
                # Safe indexing: avoid right_orig_idx[-1] when ri contains -1
                ri_mapped = np.full_like(ri, -1, dtype=np.intp)
                nonneg = ri >= 0
                if np.any(nonneg):
                    ri_mapped[nonneg] = right_orig_idx[ri[nonneg]]
            else:
                ri_mapped = ri

            # Null-key left rows are always unmatched. li is ascending, so
            # insert each at its left-table position to keep left order.
            if len(left_null_idx) > 0:
                has_null = True
                positions = np.searchsorted(li, left_null_idx)
                li = np.insert(li, positions, left_null_idx)
                ri_mapped = np.insert(ri_mapped, positions, -1)

            ri = ri_mapped

            if has_null:
                for c in join_cols:
                    if c not in left_t._data and dtypes.get(c) != "object":
                        col_kind = right_t._data[c].dtype.kind
                        if col_kind not in ("T", "U", "f", "M", "m"):
                            dtypes[c] = "object"

            result: dict[str, np.ndarray[Any, Any]] = {}
            matched = ri >= 0 if has_null else None
            for c in join_cols:
                if c in left_t._data:
                    result[c] = left_t._data[c][li]
                else:
                    # right column: fill matched rows, null for unmatched
                    if has_null:
                        assert matched is not None
                        col_kind = right_t._data[c].dtype.kind
                        if col_kind in ("T", "U"):
                            # String types: use StringDType(na_object=None)
                            out = np.empty(len(li), dtype=np.dtypes.StringDType(na_object=None))  # type: ignore[call-arg]
                            out[matched] = right_t._data[c][ri[matched]]
                            out[~matched] = None  # type: ignore[assignment]
                            dtypes[c] = "str"
                        elif col_kind == "f":
                            # Float types: use NaN for missing
                            out = np.full(len(li), np.nan, dtype=right_t._data[c].dtype)
                            out[matched] = right_t._data[c][ri[matched]]
                        elif col_kind in ("M", "m"):
                            # datetime64/timedelta64: use NaT for missing
                            nat = np.array("NaT", dtype=right_t._data[c].dtype).item()
                            out = np.empty(len(li), dtype=right_t._data[c].dtype)
                            out[matched] = right_t._data[c][ri[matched]]
                            out[~matched] = nat
                        else:
                            # int, bool, etc.: fall back to object
                            warnings.warn(
                                f"Left join: column '{c}' "
                                f"(dtype {right_t._data[c].dtype}) "
                                f"has unmatched rows and no native null "
                                f"representation. Dtype has been cast to "
                                f"object. Use .astype(float) if NaN "
                                f"semantics are needed.",
                                stacklevel=3,
                            )
                            out = cast(
                                np.ndarray[Any, Any],
                                np.empty(len(li), dtype=object),
                            )
                            out[matched] = right_t._data[c][ri[matched]]
                            out[~matched] = None  # type: ignore[assignment]
                        result[c] = out
                    else:
                        result[c] = right_t._data[c][ri]

            return Tafra(result, dtypes)

        else:
            _on = tuple((left_col, right_col, JOIN_OPS[op]) for left_col, right_col, op in self.on)
            right_rows = np.empty(right_t._rows, dtype=bool)
            join: dict[str, list[Any]] = {c: [] for c in join_cols}
            has_null = False

            # Precompute null masks for key columns (SQL: NULL never matches)
            left_key_cols = [left_t._data[lc] for lc, _, _ in self.on]
            right_key_cols = [right_t._data[rc] for _, rc, _ in self.on]
            left_null = self._non_null_mask(left_key_cols)
            right_null = self._non_null_mask(right_key_cols)

            for i in range(left_t._rows):
                # Null key left rows: always unmatched in left join
                if left_null is not None and not left_null[i]:
                    has_null = True
                    for column in join_cols:
                        if column in left_t._data:
                            join[column].append(left_t._data[column][i])
                        elif column in right_t._data:
                            join[column].append(None)
                            col_kind = right_t._data[column].dtype.kind
                            if (
                                col_kind not in ("T", "U", "f", "M", "m")
                                and dtypes[column] != "object"
                            ):
                                warnings.warn(
                                    f"Left join: column '{column}' "
                                    f"(dtype {right_t._data[column].dtype}) "
                                    f"has unmatched rows and no native null "
                                    f"representation. Dtype has been cast to "
                                    f"object. Use .astype(float) if NaN "
                                    f"semantics are needed.",
                                    stacklevel=3,
                                )
                                dtypes[column] = "object"
                    continue

                right_rows[:] = True
                # Exclude right rows with null keys
                if right_null is not None:
                    right_rows &= right_null
                for left_col, right_col, op in _on:
                    right_rows &= op(left_t._data[left_col][i], right_t._data[right_col])

                right_count = int(np.sum(right_rows))

                for column in join_cols:
                    if column in left_t._data:
                        join[column].extend([left_t._data[column][i]] * max(1, right_count))
                    elif column in right_t._data:
                        if right_count <= 0:
                            has_null = True
                            join[column].append(None)
                            col_kind = right_t._data[column].dtype.kind
                            if (
                                col_kind not in ("T", "U", "f", "M", "m")
                                and dtypes[column] != "object"
                            ):
                                warnings.warn(
                                    f"Left join: column '{column}' "
                                    f"(dtype {right_t._data[column].dtype}) "
                                    f"has unmatched rows and no native null "
                                    f"representation. Dtype has been cast to "
                                    f"object. Use .astype(float) if NaN "
                                    f"semantics are needed.",
                                    stacklevel=3,
                                )
                                dtypes[column] = "object"
                        else:
                            join[column].extend(right_t._data[column][right_rows])

            result_data: dict[str, np.ndarray[Any, Any]] = {}
            for c, v in join.items():
                col_kind = right_t._data[c].dtype.kind if c in right_t._data else ""
                if c not in left_t._data and col_kind in ("T", "U") and has_null:
                    result_data[c] = np.array(
                        v,
                        dtype=np.dtypes.StringDType(na_object=None),  # type: ignore[call-arg]
                    )
                    dtypes[c] = "str"
                elif c not in left_t._data and col_kind == "f" and has_null:
                    result_data[c] = np.array(
                        [np.nan if x is None else x for x in v], dtype=right_t._data[c].dtype
                    )
                elif c not in left_t._data and col_kind in ("M", "m") and has_null:
                    nat = np.array("NaT", dtype=right_t._data[c].dtype).item()
                    result_data[c] = np.array(
                        [nat if x is None else x for x in v], dtype=right_t._data[c].dtype
                    )
                else:
                    result_data[c] = np.array(v)

            return Tafra(result_data, dtypes)


@dc.dataclass
class CrossJoin(Join):
    """
    A cross join (Cartesian product).

    Analogy to SQL CROSS JOIN, or `pandas.merge(left, right, how='cross')`.

    Parameters
    ----------
    select: Iterable[str] = []
        The columns to return. If not given, all unique columns names
        are returned. If the column exists in both `Tafra`,
        prefers the left over the right.
    """

    def apply(self, left_t: "Tafra", right_t: "Tafra") -> "Tafra":
        """
        Apply the `CrossJoin` to the `Tafra`.

        Parameters
        ----------
        left_t: Tafra
            The left tafra to join.

        right_t: Tafra
            The right tafra to join.

        Returns
        -------
        tafra: Tafra
            The joined `Tafra`.
        """
        join_cols, dtypes = self._resolve_join_cols(left_t, right_t)

        # Empty table shortcut — cross join with either side empty → empty result
        if left_t._rows < 1 or right_t._rows < 1:
            warnings.warn(
                "Join: one or both tables have zero rows. Returning shortcut result.",
                stacklevel=2,
            )
            return Tafra(
                {
                    c: np.array(
                        [],
                        dtype=left_t._data[c].dtype
                        if c in left_t._data
                        else right_t._data[c].dtype,
                    )
                    for c in join_cols
                },
                dtypes,
            )

        left_rows = left_t._rows
        right_rows = right_t._rows

        select = list(self.select)
        if len(select) > 0:
            left_cols = [c for c in select if c in left_t._data]
            right_cols = [c for c in select if c in right_t._data]
        else:
            left_cols = list(left_t._data.keys())
            right_cols = list(right_t._data.keys())

        # Handle empty side after filtering
        left_data = (
            dict(left_t[left_cols].key_map(np.repeat, repeats=right_rows)) if left_cols else {}
        )
        right_data = (
            dict(right_t[right_cols].key_map(np.tile, reps=left_rows)) if right_cols else {}
        )

        # Left-first column order, left wins on name conflicts
        result: dict[str, np.ndarray[Any, Any]] = {}
        for c, v in left_data.items():
            result[c] = v
        for c, v in right_data.items():
            if c not in result:
                result[c] = v

        return Tafra(result, dtypes)


# Import here to resolve circular dependency
from .base import Tafra
