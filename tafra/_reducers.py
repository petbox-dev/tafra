"""
Tafra: a minimalist dataframe

Copyright (c) 2020 Derrick W. Turk and David S. Fulford

Author
------
Derrick W. Turk
David S. Fulford

Notes
-----
Vectorized group aggregations: one value per group from (data, labels),
with numpy's value and dtype, and the registry that maps an aggregation
function (np.sum, percentile(q), ...) to its fast path.
"""

from __future__ import annotations

__all__ = [
    "VectorizedAggregation",
    "PercentileAggregation",
    "percentile",
    "geomean",
    "harmean",
    "kind_in",
    "vectorized_aggregation",
]

import dataclasses as dc

import numpy as np

from typing import Any, Callable
from typing import cast

from . import _kernels

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
_VECTORIZED_AGGREGATIONS: dict[int, _VectorizedEntry] = {}

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
        if _kernels.HAS_ACCEL:
            result = _kernels.groupby_sum(
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
    if _kernels.HAS_ACCEL:
        result = _kernels.groupby_sum_int64(
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
    if _kernels.HAS_ACCEL:
        result = _kernels.groupby_mean(
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
    if _kernels.HAS_ACCEL:
        result = _kernels.groupby_var(
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
    if _kernels.HAS_ACCEL and kind == "f":
        c_function = _kernels.groupby_max if want_maximum else _kernels.groupby_min
        result = c_function(
            labels.astype(np.int64, copy=False),
            data.astype(np.float64, copy=False),
            group_count,
            skip_nan,
        )
        return _float_result(result, data)
    if _kernels.HAS_ACCEL and kind in ("i", "u", "b"):
        # exact integer kernel; a uint64 keeps its bits and compares unsigned.
        # astype first gives native byte order, so the view reads the value
        is_unsigned_64 = kind == "u" and data.dtype.itemsize == 8
        if is_unsigned_64:
            values = data.astype(np.uint64, copy=False).view(np.int64)
        else:
            values = data.astype(np.int64, copy=False)
        c_function = _kernels.groupby_max_int64 if want_maximum else _kernels.groupby_min_int64
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


class PercentileAggregation:
    """
    Callable aggregation for percentile(q). It carries its own fast path
    (`vectorized`) rather than being registered by id(): an instance can be
    freed, and its id then reused by an unrelated callable.
    """

    __slots__ = ("q",)

    def __init__(self, q: float) -> None:
        """Hold `q`, the percentile, in [0, 100]."""
        self.q = q

    def __call__(self, data: np.ndarray[Any, Any]) -> Any:
        """The per-group path: np.percentile of one group's data."""
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
        """The call that made this aggregation, e.g. percentile(90)."""
        return f"percentile({self.q})"


def percentile(q: float) -> PercentileAggregation:
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
    return PercentileAggregation(q)


def vectorized_aggregation(
    aggregation_function: Callable[..., Any], dtype: np.dtype[Any]
) -> VectorizedAggregation | None:
    """
    Return the vectorized fast path for `aggregation_function` applied to
    data of `dtype`, or None if there is none.
    """
    if isinstance(aggregation_function, PercentileAggregation):
        if _accepts_scalar_statistic(dtype):
            return aggregation_function.vectorized
        return None
    entry = _VECTORIZED_AGGREGATIONS.get(id(aggregation_function))
    if entry is None:
        return None
    return entry.implementation if entry.accepts(dtype) else None


def geomean(data: np.ndarray[Any, Any]) -> Any:
    """Geometric mean aggregation for use in `group_by`."""
    return np.exp(np.mean(np.log(data.astype(float))))


def harmean(data: np.ndarray[Any, Any]) -> Any:
    """Harmonic mean aggregation for use in `group_by`."""
    return len(data) / np.sum(1.0 / data.astype(float))


def kind_in(kinds: frozenset[str]) -> Callable[[np.dtype[Any]], bool]:
    """
    An `accepts` predicate: the dtype's kind is one of `kinds`, at most 64
    bits. The fast paths compute in float64 or int64, so a wider longdouble
    takes the per-group path and keeps its precision.
    """
    return lambda dtype: dtype.kind in kinds and dtype.itemsize <= 8


# The `accepts` predicate shared by percentile, geomean and harmean.
_accepts_scalar_statistic = kind_in(_SCALAR_STATISTIC_KINDS)


def _any_dtype(dtype: np.dtype[Any]) -> bool:
    """An `accepts` predicate for a function defined on every dtype."""
    return True


def _register_vectorized() -> None:
    """Register known numpy reducers for vectorized GroupBy."""
    arithmetic = kind_in(_ARITHMETIC_KINDS)
    ordered = kind_in(_ORDERED_KINDS)
    registrations = [
        _VectorizedEntry(np.sum, _vec_sum, arithmetic),
        _VectorizedEntry(np.nansum, _vec_nansum, arithmetic),
        _VectorizedEntry(np.mean, _vec_mean, arithmetic),
        _VectorizedEntry(np.nanmean, _vec_nanmean, arithmetic),
        _VectorizedEntry(np.std, _vec_std, arithmetic),
        _VectorizedEntry(np.nanstd, _vec_nanstd, arithmetic),
        _VectorizedEntry(np.var, _vec_var, arithmetic),
        _VectorizedEntry(np.nanvar, _vec_nanvar, arithmetic),
        _VectorizedEntry(np.min, _vec_min, ordered),
        _VectorizedEntry(np.amin, _vec_min, ordered),
        _VectorizedEntry(np.nanmin, _vec_nanmin, ordered),
        _VectorizedEntry(np.max, _vec_max, ordered),
        _VectorizedEntry(np.amax, _vec_max, ordered),
        _VectorizedEntry(np.nanmax, _vec_nanmax, ordered),
        _VectorizedEntry(np.ptp, _vec_ptp, kind_in(_RANGE_KINDS)),
        _VectorizedEntry(np.prod, _vec_prod, arithmetic),
        _VectorizedEntry(np.nanprod, _vec_nanprod, arithmetic),
        _VectorizedEntry(np.any, _vec_any, arithmetic),
        _VectorizedEntry(np.all, _vec_all, arithmetic),
        _VectorizedEntry(np.median, _vec_median, arithmetic),
        _VectorizedEntry(np.nanmedian, _vec_nanmedian, arithmetic),
        _VectorizedEntry(np.count_nonzero, _vec_count_nonzero, arithmetic),
        _VectorizedEntry(len, _vec_len, _any_dtype),
        # builtin sum keeps the element type, unlike np.sum which widens small
        # ints, so only the dtypes where the two agree take the fast path
        _VectorizedEntry(sum, _vec_sum, lambda dtype: dtype in _BUILTIN_SUM_DTYPES),
        _VectorizedEntry(geomean, _vec_geomean, _accepts_scalar_statistic),
        _VectorizedEntry(harmean, _vec_harmean, _accepts_scalar_statistic),
    ]
    for entry in registrations:
        _VECTORIZED_AGGREGATIONS[id(entry.function)] = entry


_register_vectorized()
