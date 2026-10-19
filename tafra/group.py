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

from typing import Any, Callable, Mapping, NamedTuple, Sequence, Iterable, Iterator
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

# Dtype kinds whose values (at most 64 bits) map to int64 with equality kept:
# bool, integers, float by bit pattern, datetime and timedelta by view.
_EXACT_INT64_IMAGE_KINDS: frozenset[str] = frozenset("biufMm")

# Row count from which _build_group_indices groups rows with the C hash
# kernel; below it, the numpy direct-mapping path is used.
_GROUP_INDICES_MIN_ROWS = 50_000

# Right-column dtype kinds a left join fills with their own null for an
# unmatched row: StringDType None ('T', 'U'), NaN ('f'), NaT ('M', 'm').
# Any other kind is cast to object to hold None.
_NATIVE_NULL_FILL_KINDS: frozenset[str] = frozenset("TUfMm")

# A vectorized aggregation: (data, labels, group_count) -> one value per group.
# labels are group indices in [0, group_count), one per row of data.
VectorizedAggregation = Callable[
    [np.ndarray[Any, Any], np.ndarray[Any, Any], int], np.ndarray[Any, Any]
]


@dc.dataclass(frozen=True)
class _VectorizedEntry:
    """
    The fast path registered for one aggregation function. `accepts` says
    whether it applies to data of a dtype; any other dtype takes the
    per-group path, which calls the function itself and so refuses exactly
    what numpy refuses.
    """

    function: Callable[..., Any]  # held so its id() cannot be reused while registered
    implementation: VectorizedAggregation
    accepts: Callable[[np.dtype[Any]], bool]


# Vectorized aggregation functions that can bypass per-group Python loops,
# by the id() of the registered function. Filled by _register_vectorized.
_VECTORIZED_AGGS: dict[int, _VectorizedEntry] = {}

# Kinds numpy reduces arithmetically (sum, mean, var, prod, median, ...).
_ARITHMETIC_KINDS: frozenset[str] = frozenset("fiub")
# Kinds with a total order that np.minimum / np.maximum accept.
_ORDERED_KINDS: frozenset[str] = frozenset("fiubMm")
# Kinds np.ptp accepts (it subtracts, so bool is refused).
_RANGE_KINDS: frozenset[str] = frozenset("fiuMm")
# Kinds for geomean, harmean and percentile.
_SCALAR_STATISTIC_KINDS: frozenset[str] = frozenset("fiu")
# Builtin `sum` adds element by element in the input type: an int32 sum
# stays int32 where np.sum widens it to int64, and a float32 sum rounds at
# every step where np.sum sums pairwise. The two agree only for these dtypes.
_BUILTIN_SUM_DTYPES: frozenset[np.dtype[Any]] = frozenset(
    np.dtype(name) for name in ("float64", "int64", "uint64", "bool")
)


def _sorted_segments(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Sort data by group labels. Returns (sorted_data, offsets): group g is
    sorted_data[offsets[g]:offsets[g + 1]].
    """
    order = np.argsort(labels, kind="stable")
    sorted_data = data[order]
    counts = np.bincount(labels, minlength=group_count)
    offsets = np.zeros(group_count + 1, dtype=np.intp)
    np.cumsum(counts, out=offsets[1:])
    return sorted_data, offsets


def _float_result(result: np.ndarray[Any, Any], data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    numpy keeps a float16 / float32 input at its width, in native byte order;
    other inputs give float64.
    """
    if data.dtype.kind == "f":
        return result.astype(data.dtype.newbyteorder("="), copy=False)
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
    data: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    group_count: int,
    skip_nan: bool = False,
) -> np.ndarray[Any, Any]:
    """np.sum per group (np.nansum when skip_nan): exact for integers, float kept at its width."""
    if data.dtype.kind == "f":
        if _HAS_ACCEL:
            result = _c_sum(
                labels.astype(np.int64, copy=False),
                data.astype(np.float64, copy=False),
                group_count,
                skip_nan,
            )
        else:
            values, value_labels = _float_values(data, labels, skip_nan)
            result = np.bincount(value_labels, weights=values, minlength=group_count)
        return _float_result(result, data)

    # integers sum exactly, wrapping on overflow like numpy
    accumulator = _integer_accumulator(data)
    if _HAS_ACCEL:
        result = _c_sum_int64(
            labels.astype(np.int64, copy=False), accumulator.view(np.int64), group_count
        )
        return cast(np.ndarray[Any, Any], result.view(accumulator.dtype))
    sorted_data, offsets = _sorted_segments(accumulator, labels, group_count)
    return np.add.reduceat(sorted_data, offsets[:group_count])


def _vec_nansum(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.nansum per group."""
    return _vec_sum(data, labels, group_count, skip_nan=True)


def _vec_mean(
    data: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    group_count: int,
    skip_nan: bool = False,
) -> np.ndarray[Any, Any]:
    """np.mean per group (np.nanmean when skip_nan)."""
    if _HAS_ACCEL:
        result = _c_mean(
            labels.astype(np.int64, copy=False),
            data.astype(np.float64, copy=False),
            group_count,
            skip_nan,
        )
    else:
        values, value_labels = _float_values(data, labels, skip_nan)
        sums = np.bincount(value_labels, weights=values, minlength=group_count)
        counts = np.bincount(value_labels, minlength=group_count)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = sums / counts
    return _float_result(result, data)


def _vec_nanmean(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.nanmean per group."""
    return _vec_mean(data, labels, group_count, skip_nan=True)


def _vec_var(
    data: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    group_count: int,
    skip_nan: bool = False,
) -> np.ndarray[Any, Any]:
    """np.var per group, population variance (np.nanvar when skip_nan)."""
    if _HAS_ACCEL:
        result = _c_var(
            labels.astype(np.int64, copy=False),
            data.astype(np.float64, copy=False),
            group_count,
            skip_nan,
        )
    else:
        # two-pass: subtract each group's mean before squaring, which avoids
        # the cancellation of E[x^2] - E[x]^2 on values with a large offset
        values, value_labels = _float_values(data, labels, skip_nan)
        counts = np.bincount(value_labels, minlength=group_count)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.bincount(value_labels, weights=values, minlength=group_count) / counts
            deviations = values - means[value_labels]
            squared_sums = np.bincount(
                value_labels, weights=deviations * deviations, minlength=group_count
            )
            result = squared_sums / counts
    return _float_result(result, data)


def _vec_nanvar(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.nanvar per group."""
    return _vec_var(data, labels, group_count, skip_nan=True)


def _vec_std(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.std per group, the square root of the population variance."""
    return np.sqrt(_vec_var(data, labels, group_count))


def _vec_nanstd(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.nanstd per group."""
    return np.sqrt(_vec_var(data, labels, group_count, skip_nan=True))


def _vec_extreme(
    data: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    group_count: int,
    want_maximum: bool,
    skip_nan: bool,
) -> np.ndarray[Any, Any]:
    """Grouped min / max. NaN and NaT propagate unless skip_nan, as in numpy."""
    kind = data.dtype.kind
    if _HAS_ACCEL and kind == "f":
        c_function = _c_max if want_maximum else _c_min
        result = c_function(
            labels.astype(np.int64, copy=False),
            data.astype(np.float64, copy=False),
            group_count,
            skip_nan,
        )
        return _float_result(result, data)
    if _HAS_ACCEL and kind in ("i", "u", "b"):
        # exact integer kernel; a uint64 keeps its bits and compares unsigned.
        # astype first gives native byte order, so the view reads the value
        is_unsigned_64 = kind == "u" and data.dtype.itemsize == 8
        if is_unsigned_64:
            values = data.astype(np.uint64, copy=False).view(np.int64)
        else:
            values = data.astype(np.int64, copy=False)
        c_function = _c_max_int64 if want_maximum else _c_min_int64
        result = c_function(
            labels.astype(np.int64, copy=False), values, group_count, is_unsigned_64
        )
        if is_unsigned_64:
            return cast(np.ndarray[Any, Any], result.view(np.uint64))
        return cast(np.ndarray[Any, Any], result.astype(data.dtype.newbyteorder("="), copy=False))
    # datetime / timedelta, and the pure-numpy path
    ufunc: np.ufunc
    if skip_nan:
        ufunc = np.fmax if want_maximum else np.fmin
    else:
        ufunc = np.maximum if want_maximum else np.minimum
    sorted_data, offsets = _sorted_segments(data, labels, group_count)
    return ufunc.reduceat(sorted_data, offsets[:group_count])


def _vec_min(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.min per group."""
    return _vec_extreme(data, labels, group_count, want_maximum=False, skip_nan=False)


def _vec_nanmin(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.nanmin per group."""
    return _vec_extreme(data, labels, group_count, want_maximum=False, skip_nan=True)


def _vec_max(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.max per group."""
    return _vec_extreme(data, labels, group_count, want_maximum=True, skip_nan=False)


def _vec_nanmax(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.nanmax per group."""
    return _vec_extreme(data, labels, group_count, want_maximum=True, skip_nan=True)


def _vec_ptp(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.ptp per group: maximum minus minimum."""
    sorted_data, offsets = _sorted_segments(data, labels, group_count)
    return np.maximum.reduceat(sorted_data, offsets[:group_count]) - np.minimum.reduceat(
        sorted_data, offsets[:group_count]
    )


def _vec_prod(
    data: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    group_count: int,
    skip_nan: bool = False,
) -> np.ndarray[Any, Any]:
    """np.prod per group (np.nanprod when skip_nan), integers in numpy's accumulator."""
    if data.dtype.kind == "f":
        values = np.where(np.isnan(data), 1, data) if skip_nan else data
    else:
        values = _integer_accumulator(data)
    sorted_data, offsets = _sorted_segments(values, labels, group_count)
    return np.multiply.reduceat(sorted_data, offsets[:group_count])


def _vec_nanprod(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.nanprod per group."""
    return _vec_prod(data, labels, group_count, skip_nan=True)


def _vec_any(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.any per group."""
    sorted_data, offsets = _sorted_segments(data.astype(bool), labels, group_count)
    return np.logical_or.reduceat(sorted_data, offsets[:group_count])


def _vec_all(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.all per group."""
    sorted_data, offsets = _sorted_segments(data.astype(bool), labels, group_count)
    return np.logical_and.reduceat(sorted_data, offsets[:group_count])


def _vec_len(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """len per group: the row count."""
    return np.bincount(labels, minlength=group_count)


def _vec_count_nonzero(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.count_nonzero per group."""
    return np.bincount(labels[data != 0], minlength=group_count)


def _vec_median(
    data: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    group_count: int,
    skip_nan: bool = False,
) -> np.ndarray[Any, Any]:
    """np.median per group (np.nanmedian when skip_nan); a group holding NaN gives NaN."""
    sorted_data, offsets = _sorted_segments(data, labels, group_count)
    result = np.empty(group_count, dtype=np.float64)
    is_float = data.dtype.kind == "f"
    for group_index in range(group_count):
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
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """np.nanmedian per group."""
    return _vec_median(data, labels, group_count, skip_nan=True)


def _vec_geomean(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """geomean per group: exp of the mean log."""
    log_sums = np.bincount(labels, weights=np.log(data.astype(float)), minlength=group_count)
    counts = np.bincount(labels, minlength=group_count).astype(float)
    return np.exp(log_sums / counts)


def _vec_harmean(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
) -> np.ndarray[Any, Any]:
    """harmean per group: count over the sum of reciprocals."""
    reciprocal_sums = np.bincount(labels, weights=1.0 / data.astype(float), minlength=group_count)
    counts = np.bincount(labels, minlength=group_count).astype(float)
    return counts / reciprocal_sums


class _PercentileAgg:
    """
    Callable aggregation for percentile(q). It carries its own fast path
    (`vectorized`) rather than being registered by id(): an instance can be
    freed, and its id then reused by an unrelated callable.
    """

    __slots__ = ("q",)

    def __init__(self, q: float) -> None:
        self.q = q  # the percentile, in [0, 100]

    def __call__(self, data: np.ndarray[Any, Any]) -> Any:
        return np.percentile(data, self.q)

    def vectorized(
        self, data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], group_count: int
    ) -> np.ndarray[Any, Any]:
        """np.percentile(·, q) per group."""
        sorted_data, offsets = _sorted_segments(data, labels, group_count)
        result = np.empty(group_count, dtype=float)
        for group_index in range(group_count):
            result[group_index] = np.percentile(
                sorted_data[offsets[group_index] : offsets[group_index + 1]], self.q
            )
        return _float_result(result, data)

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
    aggregation_function: Callable[..., Any], dtype: np.dtype[Any]
) -> VectorizedAggregation | None:
    """
    Return the vectorized fast path for `aggregation_function` applied to
    data of `dtype`, or None if there is none.
    """
    if isinstance(aggregation_function, _PercentileAgg):
        if _kind_in(_SCALAR_STATISTIC_KINDS)(dtype):
            return aggregation_function.vectorized
        return None
    entry = _VECTORIZED_AGGS.get(id(aggregation_function))
    if entry is None:
        return None
    return entry.implementation if entry.accepts(dtype) else None


def geomean(data: np.ndarray[Any, Any]) -> Any:
    """Geometric mean aggregation for use in `group_by`."""
    return np.exp(np.mean(np.log(data.astype(float))))


def harmean(data: np.ndarray[Any, Any]) -> Any:
    """Harmonic mean aggregation for use in `group_by`."""
    return len(data) / np.sum(1.0 / data.astype(float))


def _kind_in(kinds: frozenset[str]) -> Callable[[np.dtype[Any]], bool]:
    """
    An `accepts` predicate: the dtype's kind is one of `kinds`, at most 64
    bits. The fast paths compute in float64 or int64, so a wider longdouble
    takes the per-group path and keeps its precision.
    """
    return lambda dtype: dtype.kind in kinds and dtype.itemsize <= 8


def _any_dtype(dtype: np.dtype[Any]) -> bool:
    """An `accepts` predicate for a function defined on every dtype."""
    return True


def _register_vectorized() -> None:
    """Register known numpy reducers for vectorized GroupBy."""
    arithmetic = _kind_in(_ARITHMETIC_KINDS)
    ordered = _kind_in(_ORDERED_KINDS)
    scalar_statistic = _kind_in(_SCALAR_STATISTIC_KINDS)
    registrations: list[
        tuple[Callable[..., Any], VectorizedAggregation, Callable[[np.dtype[Any]], bool]]
    ] = [
        (np.sum, _vec_sum, arithmetic),
        (np.nansum, _vec_nansum, arithmetic),
        (np.mean, _vec_mean, arithmetic),
        (np.nanmean, _vec_nanmean, arithmetic),
        (np.std, _vec_std, arithmetic),
        (np.nanstd, _vec_nanstd, arithmetic),
        (np.var, _vec_var, arithmetic),
        (np.nanvar, _vec_nanvar, arithmetic),
        (np.min, _vec_min, ordered),
        (np.amin, _vec_min, ordered),
        (np.nanmin, _vec_nanmin, ordered),
        (np.max, _vec_max, ordered),
        (np.amax, _vec_max, ordered),
        (np.nanmax, _vec_nanmax, ordered),
        (np.ptp, _vec_ptp, _kind_in(_RANGE_KINDS)),
        (np.prod, _vec_prod, arithmetic),
        (np.nanprod, _vec_nanprod, arithmetic),
        (np.any, _vec_any, arithmetic),
        (np.all, _vec_all, arithmetic),
        (np.median, _vec_median, arithmetic),
        (np.nanmedian, _vec_nanmedian, arithmetic),
        (np.count_nonzero, _vec_count_nonzero, arithmetic),
        (len, _vec_len, _any_dtype),
        # builtin sum keeps the element type, unlike np.sum which widens small
        # ints, so only the dtypes where the two agree take the fast path
        (sum, _vec_sum, lambda dtype: dtype in _BUILTIN_SUM_DTYPES),
        (geomean, _vec_geomean, scalar_statistic),
        (harmean, _vec_harmean, scalar_statistic),
    ]
    for function, implementation, accepts in registrations:
        _VECTORIZED_AGGS[id(function)] = _VectorizedEntry(function, implementation, accepts)


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


@dc.dataclass
class GroupSet:
    """
    A `GroupSet` is the set of columns by which we construct our groups.
    """

    @staticmethod
    def _dense_range_limit(row_count: int) -> int:
        """
        Largest code range used directly; a wider range is re-encoded densely.
        The direct-mapped label arrays are sized by the code range, so this
        caps them at a small multiple of the row count (plus a floor for
        small tables).
        """
        return 4 * row_count + 1024

    @staticmethod
    def _hash_encode_object(column: np.ndarray[Any, Any]) -> EncodedColumn:
        """
        First-seen codes from a dict codebook. np.unique sorts via argsort,
        which raises TypeError on object arrays containing mixed Python types
        (e.g., str and int), and on StringDType arrays holding None. Semantics
        match _c_encode_strings (hash equality, not ordering).
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

    @staticmethod
    def _unique_codes(column: np.ndarray[Any, Any]) -> EncodedColumn:
        """Sorted-order codes from np.unique, as a flat int64 array."""
        unique, codes = np.unique(column, return_inverse=True)
        # numpy 2 can return the inverse in the input's shape
        return EncodedColumn(codes.astype(np.int64, copy=False).reshape(-1), len(unique))

    @staticmethod
    def _encode_strings(column: np.ndarray[Any, Any]) -> EncodedColumn:
        """First-seen codes for a string or object column."""
        if _HAS_ACCEL:
            # astype(object) gives the C encoder a private copy (see _accel.c)
            codes, unique_count = _c_encode_strings(column.astype(object))
            return EncodedColumn(codes, int(unique_count))
        if column.dtype.kind in ("O", "T"):
            return GroupSet._hash_encode_object(column)
        return GroupSet._unique_codes(column)

    @staticmethod
    def _encode_int64_dense(keys: np.ndarray[Any, Any]) -> EncodedColumn:
        """Dense codes 0..unique_count-1 for int64 keys."""
        if _HAS_ACCEL:
            codes, unique_count = _c_encode_int64(keys)
            return EncodedColumn(codes, int(unique_count))
        return GroupSet._unique_codes(keys)

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
    def _encode_column(column: np.ndarray[Any, Any]) -> EncodedColumn:
        """Encode one key column; its cardinality is at most _dense_range_limit(rows)."""
        if len(column) == 0:
            return EncodedColumn(np.array([], dtype=np.int64), 0)
        kind = column.dtype.kind
        if kind in _STRING_KINDS:
            return GroupSet._encode_strings(column)
        if kind not in _EXACT_INT64_IMAGE_KINDS or column.dtype.itemsize > 8:
            # complex, longdouble, structured: no exact int64 image
            try:
                return GroupSet._unique_codes(column)
            except TypeError:
                return GroupSet._hash_encode_object(column.astype(object))

        bits = GroupSet._integer_key_bits(column)
        if kind != "f":
            low, high = int(bits.min()), int(bits.max())
            if high - low < GroupSet._dense_range_limit(len(bits)):
                if low == 0:
                    return EncodedColumn(bits, high + 1)
                # the result lies in [0, high - low], so int64 wraparound in
                # the subtraction cannot change it
                return EncodedColumn(bits - np.int64(low), high - low + 1)
        # float bit patterns are never a narrow range
        return GroupSet._encode_int64_dense(bits)

    @staticmethod
    def _build_composite_key(encoded: list[EncodedColumn], row_count: int) -> np.ndarray[Any, Any]:
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
            return encoded[0].codes
        if row_count == 0:
            return np.array([], dtype=np.int64)

        cardinalities = [column.cardinality for column in encoded]
        if math.prod(cardinalities) <= limit:
            columns = tuple(column.codes for column in encoded)
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
        combined_columns: list[np.ndarray[Any, Any]], left_count: int
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Encode join key columns with one shared codebook, so equal values on
        the two sides get equal int64 keys. Each column is the left rows
        followed by the right rows, concatenated, which promotes same-kind
        dtypes to a common one (Join._validate_dtypes refuses other pairs).
        Returns the left and right keys.
        """
        encoded = [GroupSet._encode_column(column) for column in combined_columns]
        key = GroupSet._build_composite_key(encoded, len(combined_columns[0]))
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
        sorted_labels, n_groups = GroupSet._direct_labels_sorted(data)
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
        column_arrays = [tafra._data[column] for column in columns]
        if not column_arrays:
            raise ValueError("Grouping needs at least one key column.")
        encoded = [GroupSet._encode_column(column) for column in column_arrays]
        data = GroupSet._build_composite_key(encoded, tafra._rows)
        return data, column_arrays

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

        if _HAS_ACCEL and tafra._rows >= _GROUP_INDICES_MIN_ROWS:
            first_seen_idx, group_indices_list, n_groups = _c_group_indices(data)
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
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int]:
        """
        Build per-row group labels (integers 0..n_groups-1) in first-seen order.

        Returns (labels, first_seen_indices, n_groups), as
        _direct_labels_firstseen: labels[i] is the group index for row i and
        first_seen_indices[j] is group j's first row.
        """
        data, _ = GroupSet._prepare_keys(tafra, columns)
        return GroupSet._direct_labels_firstseen(data, tafra._rows)

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
        # Materialize so a generator is not exhausted by validation.
        self.group_by_cols = list(self.group_by_cols)
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
        over their dtypes), or object when they share none, e.g. datetime64
        and float. (A str and a float promote to str.)
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
            result_types.pop(), (np.generic, int, float, str)  # bool is an int
        ):
            # one scalar type: np.array promotes exactly as np.promote_types
            # (datetime units, string widths). Python ints follow np.array:
            # int64 when all fit, uint64 when all are in [2**63, 2**64),
            # float64 for a mix of the two, object beyond 2**64
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

    def _vectorized_plan(self, tafra: "Tafra") -> dict[str, VectorizedAggregation] | None:
        """
        The fast path of every aggregation, by output name, or None when any
        of them must take the per-group loop (no fast path for its function
        and dtype, or an iter_fn is given) or there is no aggregation.
        """
        if self.iter_fn or not self._aggregation:
            return None
        plan: dict[str, VectorizedAggregation] = {}
        for rename, (fn, col) in self._aggregation.items():
            vectorized_function = _vectorized_aggregation(fn, tafra._data[col].dtype)
            if vectorized_function is None:
                return None
            plan[rename] = vectorized_function
        return plan

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

        vectorized_plan = self._vectorized_plan(tafra)
        if vectorized_plan is not None:
            # fast path: vectorized aggregation via labels, no per-group loop
            labels, first_seen_indices, n_groups = self._build_group_labels(
                tafra, self.group_by_cols
            )

            result: dict[str, np.ndarray[Any, Any]] = {
                col: tafra._data[col][first_seen_indices] for col in self.group_by_cols
            }
            for rename, (_, col) in self._aggregation.items():
                result[rename] = vectorized_plan[rename](tafra._data[col], labels, n_groups)

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

        vectorized_plan = self._vectorized_plan(tafra)
        if vectorized_plan is not None:
            # fast path: compute per-group aggregates, broadcast via labels
            # Transform doesn't need first-seen order — use direct mapping
            data, _ = self._prepare_keys(tafra, self.group_by_cols)
            labels, n_groups = self._direct_labels_sorted(data)

            result: dict[str, np.ndarray[Any, Any]] = {}
            for col in self.group_by_cols:
                result[col] = tafra._data[col].copy()

            for rename, (_, col) in self._aggregation.items():
                group_values = vectorized_plan[rename](tafra._data[col], labels, n_groups)
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
                if column.dtype == object:
                    # store numpy scalars, as GroupBy does: casting the array
                    # to object would turn a datetime64 into an int
                    column[rows] = array[()] if array.ndim == 0 else list(array)
                else:
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

    def __post_init__(self) -> None:
        # Materialize so a generator is not exhausted by validation.
        self.group_by_cols = list(self.group_by_cols)

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

    def _validate_dtypes(self, left_table: "Tafra", right_table: "Tafra") -> None:
        """
        Raise TypeError unless each key pair has one dtype kind, or both are
        strings. Same-kind keys of different widths are joinable:
        `_equi_join_keys` concatenates the two sides, which promotes them.
        """
        for left_column, right_column, _ in self.on:
            left_dtype = left_table._dtypes[left_column]
            right_dtype = right_table._dtypes[right_column]
            if left_dtype == right_dtype:
                continue
            if Tafra._reduce_dtype(left_dtype) == Tafra._reduce_dtype(right_dtype) == "str":
                continue
            if np.dtype(left_dtype).kind != np.dtype(right_dtype).kind:
                raise TypeError(
                    f"This `Tafra` column `{left_column}` dtype `{left_dtype}` "
                    f"does not match other `Tafra` dtype `{right_dtype}`."
                )

    @staticmethod
    def _is_non_null_object(value: Any) -> bool:
        """False for the nulls an object array can hold: None, NaN and NaT."""
        if value is None:
            return False
        if isinstance(value, (float, np.floating)):
            return not np.isnan(value)
        if isinstance(value, (np.datetime64, np.timedelta64)):
            return not np.isnat(value)
        return True

    @staticmethod
    def _non_null_mask(columns: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any] | None:
        """Build a boolean mask that is True for rows with no null in any key column.

        Returns None if all columns are non-nullable types (int, uint, bool),
        meaning no nulls are possible and no mask is needed.
        """
        # Fast path: non-nullable types cannot hold nulls — skip entirely.
        if not any(column.dtype.kind in _NULLABLE_KINDS for column in columns):
            return None

        valid = np.ones(len(columns[0]), dtype=bool)
        for column in columns:
            kind = column.dtype.kind
            if kind == "f":
                valid &= ~np.isnan(column)
            elif kind in ("M", "m"):
                valid &= ~np.isnat(column)
            elif kind == "T":
                # StringDType(na_object=None): None is the null sentinel
                valid &= np.array([value is not None for value in column], dtype=bool)
            elif kind == "O":
                # Object arrays can hold None, NaN, and NaT simultaneously
                valid &= np.array(
                    [Join._is_non_null_object(value) for value in column], dtype=bool
                )
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

    def _validate_and_resolve(
        self, left_t: "Tafra", right_t: "Tafra"
    ) -> tuple[list[str], dict[str, str]]:
        """
        Validate key columns, key dtypes and operators (this works on empty
        tables too), then return the output columns and their dtype labels.
        """
        left_columns, right_columns, operators = list(zip(*self.on))
        left_t._validate_columns(left_columns)
        right_t._validate_columns(right_columns)
        self._validate_dtypes(left_t, right_t)
        self._validate_ops(operators)
        return self._resolve_join_cols(left_t, right_t)

    def _is_equi_join(self) -> bool:
        """True when every predicate is `==`, so the hash or sort-merge join applies."""
        return all(operator_name == "==" for _, _, operator_name in self.on)

    def _equi_join_keys(
        self, left_t: "Tafra", right_t: "Tafra"
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        One int64 key per row of each side; equal keys mean equal key values.

        NULL never matches (SQL semantics). A right row with a NULL in any key
        column gets the key -1, which no encoded key equals. A left row with a
        NULL key is then unmatched as well: every right row with the same
        codes holds the same NULL, so it carries -1 now. NULL is judged on
        the promoted values the codes come from: a unit cast can turn a
        datetime into NaT.

        Only the right key may hold -1, and neither key may go to the
        direct-mapping label functions, which need non-negative keys. A join
        that must also keep right NULL rows apart from left NULL rows needs a
        second sentinel.
        """
        combined_columns = [
            np.concatenate([left_t._data[left_column], right_t._data[right_column]])
            for left_column, right_column, _ in self.on
        ]
        left_count = left_t._rows
        left_key, right_key = GroupSet._encode_join_keys(combined_columns, left_count)
        right_valid = self._non_null_mask([column[left_count:] for column in combined_columns])
        if right_valid is not None:
            right_key = np.where(right_valid, right_key, np.int64(-1))
        return left_key, right_key

    @staticmethod
    def _empty_result(
        join_cols: list[str], dtypes: dict[str, str], left_t: "Tafra", right_t: "Tafra"
    ) -> "Tafra":
        """A zero-row result. Each column keeps its source dtype; left wins a shared name."""
        return Tafra(
            {
                column: np.array(
                    [],
                    dtype=left_t._data[column].dtype
                    if column in left_t._data
                    else right_t._data[column].dtype,
                )
                for column in join_cols
            },
            dtypes,
        )

    @staticmethod
    def _warn_zero_rows() -> None:
        """Warn that a join returned its empty-input shortcut."""
        warnings.warn(
            "Join: one or both tables have zero rows. Returning shortcut result.",
            # this function, apply, then the Tafra join method (as before the
            # shortcut moved out of apply)
            stacklevel=3,
        )

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
        join_cols, dtypes = self._validate_and_resolve(left_t, right_t)

        # Empty table shortcut — after validation so schema errors aren't hidden
        if left_t._rows < 1 or right_t._rows < 1:
            self._warn_zero_rows()
            return self._empty_result(join_cols, dtypes, left_t, right_t)

        if self._is_equi_join():
            return self._apply_equi(left_t, right_t, join_cols, dtypes)
        return self._apply_nested_loop(left_t, right_t, join_cols, dtypes)

    def _apply_equi(
        self, left_t: "Tafra", right_t: "Tafra", join_cols: list[str], dtypes: dict[str, str]
    ) -> "Tafra":
        """Equi-join through int64 keys: the C hash join, or a sort-merge."""
        left_key, right_key = self._equi_join_keys(left_t, right_t)
        if _HAS_ACCEL:
            left_indices, right_indices = _c_inner_join(left_key, right_key)
        else:
            left_indices, right_indices = self._sort_merge_indices(left_key, right_key)

        if len(left_indices) == 0:
            return self._empty_result(join_cols, dtypes, left_t, right_t)

        result: dict[str, np.ndarray[Any, Any]] = {}
        for column in join_cols:
            if column in left_t._data:
                result[column] = left_t._data[column][left_indices]
            else:
                result[column] = right_t._data[column][right_indices]
        return Tafra(result, dtypes)

    def _apply_nested_loop(
        self, left_t: "Tafra", right_t: "Tafra", join_cols: list[str], dtypes: dict[str, str]
    ) -> "Tafra":
        """Join with any operator: compare each left row against every right row."""
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

        # right indices: each left row's match range in right_order, in order.
        # An unmatched row's one output slot points at left_lo, which can be
        # one past the end, so it is clipped and then overwritten with -1.
        offsets = np.repeat(left_lo, out_counts)
        group_starts = np.cumsum(out_counts) - out_counts
        within = np.arange(total, dtype=np.intp) - np.repeat(group_starts, out_counts)
        if len(right_order) == 0:  # LeftJoin.apply returns earlier; a direct call
            return li, np.full(total, -1, dtype=np.intp), has_null
        ri = right_order[np.minimum(offsets + within, len(right_order) - 1)]
        if has_null:
            ri[np.repeat(unmatched, out_counts)] = -1

        return li, ri, has_null

    @staticmethod
    def _warn_no_native_null(column: str, dtype: np.dtype[Any], frames_below_apply: int) -> None:
        """
        Warn that right column `column` was cast to object to hold None.
        `frames_below_apply` is the number of calls between `apply` and this
        function, so the warning points at the caller of the Tafra join method.
        """
        warnings.warn(
            f"Left join: column '{column}' "
            f"(dtype {dtype}) "
            f"has unmatched rows and no native null "
            f"representation. Dtype has been cast to "
            f"object. Use .astype(float) if NaN "
            f"semantics are needed.",
            # this function, its callers below apply, apply, the join method, its caller
            stacklevel=frames_below_apply + 4,
        )

    @staticmethod
    def _null_filled_column(
        column: str, source: np.ndarray[Any, Any], right_indices: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], str | None]:
        """
        `source[right_indices]`, with a null where the index is -1 (an
        unmatched left row): None in a StringDType for strings, NaN for floats,
        NaT for datetime / timedelta, and None in an object array otherwise.

        Returns the column and its new `_dtypes` label, or None when the label
        stays. Warns when the column is cast to object.
        """
        matched = right_indices >= 0
        matched_values = source[right_indices[matched]]
        row_count = len(right_indices)
        kind = source.dtype.kind
        if kind in ("T", "U"):
            column_values = np.empty(
                row_count, dtype=np.dtypes.StringDType(na_object=None)  # type: ignore[call-arg]
            )
            column_values[matched] = matched_values
            column_values[~matched] = None  # type: ignore[assignment]
            return column_values, "str"
        if kind == "f":
            column_values = np.full(row_count, np.nan, dtype=source.dtype)
            column_values[matched] = matched_values
            return column_values, None
        if kind in ("M", "m"):
            column_values = np.empty(row_count, dtype=source.dtype)
            column_values[matched] = matched_values
            column_values[~matched] = np.array("NaT", dtype=source.dtype).item()
            return column_values, None
        # bool, int, bytes, complex, object: only object holds None
        LeftJoin._warn_no_native_null(column, source.dtype, frames_below_apply=2)
        column_values = cast(np.ndarray[Any, Any], np.empty(row_count, dtype=object))
        column_values[matched] = matched_values
        column_values[~matched] = None  # type: ignore[assignment]
        return column_values, "object"

    def _gather(
        self,
        left_t: "Tafra",
        right_t: "Tafra",
        join_cols: list[str],
        dtypes: dict[str, str],
        left_indices: np.ndarray[Any, Any],
        right_indices: np.ndarray[Any, Any],
        has_null: bool,
    ) -> "Tafra":
        """
        Result rows from index pairs. A right index of -1 is an unmatched left
        row; when `has_null`, right columns are null-filled there.
        """
        result: dict[str, np.ndarray[Any, Any]] = {}
        for column in join_cols:
            if column in left_t._data:
                result[column] = left_t._data[column][left_indices]
            elif has_null:
                result[column], label = self._null_filled_column(
                    column, right_t._data[column], right_indices
                )
                if label is not None:
                    dtypes[column] = label
            else:
                result[column] = right_t._data[column][right_indices]
        return Tafra(result, dtypes)

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
        join_cols, dtypes = self._validate_and_resolve(left_t, right_t)

        # Empty table shortcuts — after validation
        if left_t._rows < 1:
            self._warn_zero_rows()
            return self._empty_result(join_cols, dtypes, left_t, right_t)
        if right_t._rows < 1:
            self._warn_zero_rows()
            # every left row is unmatched
            unmatched = np.full(left_t._rows, -1, dtype=np.intp)
            return self._gather(
                left_t, right_t, join_cols, dtypes, np.arange(left_t._rows), unmatched, True
            )

        if self._is_equi_join():
            left_key, right_key = self._equi_join_keys(left_t, right_t)
            if _HAS_ACCEL:
                left_indices, right_indices, has_null = _c_left_join(left_key, right_key)
            else:
                left_indices, right_indices, has_null = self._left_join_indices(
                    left_key, right_key
                )
            return self._gather(
                left_t, right_t, join_cols, dtypes, left_indices, right_indices, has_null
            )
        return self._apply_nested_loop(left_t, right_t, join_cols, dtypes)

    @staticmethod
    def _append_unmatched(
        join: dict[str, list[Any]], dtypes: dict[str, str], column: str, dtype: np.dtype[Any]
    ) -> None:
        """
        Append a null to right column `column` of the nested-loop result; the
        first null in a column whose dtype has none relabels it object, with a warning.
        """
        join[column].append(None)
        if dtype.kind not in _NATIVE_NULL_FILL_KINDS and dtypes[column] != "object":
            LeftJoin._warn_no_native_null(column, dtype, frames_below_apply=2)
            dtypes[column] = "object"

    def _apply_nested_loop(
        self, left_t: "Tafra", right_t: "Tafra", join_cols: list[str], dtypes: dict[str, str]
    ) -> "Tafra":
        """Join with any operator: compare each left row against every right row."""
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
                        self._append_unmatched(join, dtypes, column, right_t._data[column].dtype)
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
                        self._append_unmatched(join, dtypes, column, right_t._data[column].dtype)
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
            self._warn_zero_rows()
            return self._empty_result(join_cols, dtypes, left_t, right_t)

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
