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

__all__ = ['GroupBy', 'Transform', 'IterateBy', 'InnerJoin', 'LeftJoin',
           'percentile', 'geomean', 'harmean']

import operator
import warnings
from itertools import chain
import dataclasses as dc

import numpy as np

from typing import (Any, Callable, Mapping, Sequence,
                    Iterable, Iterator)
from typing import cast

try:
    from ._accel import (  # type: ignore[import-not-found]
        groupby_sum as _c_sum,
        groupby_mean as _c_mean,
        groupby_var as _c_var,
        groupby_min as _c_min,
        groupby_max as _c_max,
        groupby_count as _c_count,
        inner_join as _c_inner_join,
        left_join as _c_left_join,
        composite_key as _c_composite_key,
        group_indices as _c_group_indices,
        encode_strings as _c_encode_strings,
    )
    _HAS_ACCEL = True
except ImportError:
    _HAS_ACCEL = False


# Vectorized aggregation functions that can bypass per-group Python loops.
# Maps function identity to a callable(data, labels, n_groups) -> result_array.
_VECTORIZED_AGGS: dict[int, Callable[..., np.ndarray[Any, Any]]] = {}


def _sorted_segments(
    data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Sort data by group labels, return (sorted_data, offsets, counts)."""
    order = np.argsort(labels, kind='stable')
    sorted_data = data[order]
    counts = np.bincount(labels, minlength=n)
    offsets = np.zeros(n + 1, dtype=np.intp)
    np.cumsum(counts, out=offsets[1:])
    return sorted_data, offsets, counts


def _vec_sum(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
             n: int) -> np.ndarray[Any, Any]:
    return np.bincount(labels, weights=data.astype(float), minlength=n)


def _vec_mean(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
              n: int) -> np.ndarray[Any, Any]:
    sums = np.bincount(labels, weights=data.astype(float), minlength=n)
    counts = np.bincount(labels, minlength=n)
    return sums / counts


def _vec_var(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
             n: int) -> np.ndarray[Any, Any]:
    fdata = data.astype(float)
    counts = np.bincount(labels, minlength=n).astype(float)
    sums = np.bincount(labels, weights=fdata, minlength=n)
    sum_sq = np.bincount(labels, weights=fdata * fdata, minlength=n)
    mean = sums / counts
    return sum_sq / counts - mean * mean


def _vec_std(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
             n: int) -> np.ndarray[Any, Any]:
    return np.sqrt(_vec_var(data, labels, n))


def _vec_min(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
             n: int) -> np.ndarray[Any, Any]:
    sorted_data, offsets, _ = _sorted_segments(data, labels, n)
    return np.minimum.reduceat(sorted_data, offsets[:n])


def _vec_max(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
             n: int) -> np.ndarray[Any, Any]:
    sorted_data, offsets, _ = _sorted_segments(data, labels, n)
    return np.maximum.reduceat(sorted_data, offsets[:n])


def _vec_ptp(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
             n: int) -> np.ndarray[Any, Any]:
    sorted_data, offsets, _ = _sorted_segments(data, labels, n)
    return (np.maximum.reduceat(sorted_data, offsets[:n])
            - np.minimum.reduceat(sorted_data, offsets[:n]))


def _vec_prod(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
              n: int) -> np.ndarray[Any, Any]:
    sorted_data, offsets, _ = _sorted_segments(data, labels, n)
    return np.multiply.reduceat(sorted_data.astype(float), offsets[:n])


def _vec_any(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
             n: int) -> np.ndarray[Any, Any]:
    sorted_data, offsets, _ = _sorted_segments(data.astype(bool), labels, n)
    return np.logical_or.reduceat(sorted_data, offsets[:n])


def _vec_all(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
             n: int) -> np.ndarray[Any, Any]:
    sorted_data, offsets, _ = _sorted_segments(data.astype(bool), labels, n)
    return np.logical_and.reduceat(sorted_data, offsets[:n])


def _vec_len(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
             n: int) -> np.ndarray[Any, Any]:
    return np.bincount(labels, minlength=n)


def _vec_count_nonzero(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
                       n: int) -> np.ndarray[Any, Any]:
    return np.bincount(labels, weights=(data != 0).astype(float), minlength=n)


def _vec_median(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
                n: int) -> np.ndarray[Any, Any]:
    sorted_data, offsets, counts = _sorted_segments(data, labels, n)
    result = np.empty(n, dtype=float)
    for g in range(n):
        lo = offsets[g]
        hi = offsets[g + 1]
        seg = np.sort(sorted_data[lo:hi])
        c = int(counts[g])
        mid = c // 2
        if c % 2 == 1:
            result[g] = seg[mid]
        else:
            result[g] = (seg[mid - 1] + seg[mid]) / 2.0
    return result


def _vec_geomean(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
                 n: int) -> np.ndarray[Any, Any]:
    log_sums = np.bincount(labels, weights=np.log(data.astype(float)), minlength=n)
    counts = np.bincount(labels, minlength=n).astype(float)
    return np.exp(log_sums / counts)


def _vec_harmean(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
                 n: int) -> np.ndarray[Any, Any]:
    recip_sums = np.bincount(labels, weights=1.0 / data.astype(float), minlength=n)
    counts = np.bincount(labels, minlength=n).astype(float)
    return counts / recip_sums


def _make_vec_percentile(
    q: float
) -> Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any], int], np.ndarray[Any, Any]]:
    """Create a vectorized percentile function for a given quantile."""
    def _vec_pct(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
                 n: int) -> np.ndarray[Any, Any]:
        sorted_data, offsets, counts = _sorted_segments(data, labels, n)
        result = np.empty(n, dtype=float)
        for g in range(n):
            lo = offsets[g]
            hi = offsets[g + 1]
            result[g] = np.percentile(sorted_data[lo:hi], q)
        return result
    return _vec_pct


class _PercentileAgg:
    """Callable aggregation for percentile(q). Registered for vectorized fast path."""
    __slots__ = ('q', '_vec_fn')

    def __init__(self, q: float) -> None:
        self.q = q
        self._vec_fn = _make_vec_percentile(q)
        _VECTORIZED_AGGS[id(self)] = self._vec_fn

    def __call__(self, data: np.ndarray[Any, Any]) -> Any:
        return np.percentile(data, self.q)

    def __repr__(self) -> str:
        return f'percentile({self.q})'


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


def geomean(data: np.ndarray[Any, Any]) -> Any:
    """Geometric mean aggregation for use in `group_by`."""
    return np.exp(np.mean(np.log(data.astype(float))))


def harmean(data: np.ndarray[Any, Any]) -> Any:
    """Harmonic mean aggregation for use in `group_by`."""
    return len(data) / np.sum(1.0 / data.astype(float))


def _register_vectorized() -> None:
    """Register known numpy reducers for vectorized GroupBy."""
    # Use C accelerated versions when available, fall back to numpy
    if _HAS_ACCEL:
        def _wrap_c(c_fn: Callable[..., Any]) -> Callable[..., np.ndarray[Any, Any]]:
            def _wrapped(data: np.ndarray[Any, Any], labels: np.ndarray[Any, Any],
                         n: int) -> np.ndarray[Any, Any]:
                return c_fn(labels.astype(np.int64, copy=False),
                            data.astype(np.float64, copy=False), n)
            return _wrapped
        _fast_sum = _wrap_c(_c_sum)
        _fast_mean = _wrap_c(_c_mean)
        _fast_var = _wrap_c(_c_var)
        _fast_min = _wrap_c(_c_min)
        _fast_max = _wrap_c(_c_max)
    else:
        _fast_sum = _vec_sum
        _fast_mean = _vec_mean
        _fast_var = _vec_var
        _fast_min = _vec_min
        _fast_max = _vec_max

    _fast_std = (lambda d, l, n: np.sqrt(_fast_var(d, l, n))) if _HAS_ACCEL else _vec_std

    _VECTORIZED_AGGS[id(np.sum)] = _fast_sum
    _VECTORIZED_AGGS[id(np.nansum)] = _fast_sum
    _VECTORIZED_AGGS[id(np.mean)] = _fast_mean
    _VECTORIZED_AGGS[id(np.nanmean)] = _fast_mean
    _VECTORIZED_AGGS[id(np.std)] = _fast_std
    _VECTORIZED_AGGS[id(np.nanstd)] = _fast_std
    _VECTORIZED_AGGS[id(np.var)] = _fast_var
    _VECTORIZED_AGGS[id(np.nanvar)] = _fast_var
    _VECTORIZED_AGGS[id(np.min)] = _fast_min
    _VECTORIZED_AGGS[id(np.nanmin)] = _fast_min
    _VECTORIZED_AGGS[id(np.amin)] = _fast_min
    _VECTORIZED_AGGS[id(np.max)] = _fast_max
    _VECTORIZED_AGGS[id(np.nanmax)] = _fast_max
    _VECTORIZED_AGGS[id(np.amax)] = _fast_max
    _VECTORIZED_AGGS[id(np.ptp)] = _vec_ptp
    _VECTORIZED_AGGS[id(np.prod)] = _vec_prod
    _VECTORIZED_AGGS[id(np.nanprod)] = _vec_prod
    _VECTORIZED_AGGS[id(np.any)] = _vec_any
    _VECTORIZED_AGGS[id(np.all)] = _vec_all
    _VECTORIZED_AGGS[id(np.median)] = _vec_median
    _VECTORIZED_AGGS[id(np.nanmedian)] = _vec_median
    _VECTORIZED_AGGS[id(np.count_nonzero)] = _vec_count_nonzero
    _VECTORIZED_AGGS[id(len)] = _vec_len
    _VECTORIZED_AGGS[id(sum)] = _vec_sum
    _VECTORIZED_AGGS[id(geomean)] = _vec_geomean
    _VECTORIZED_AGGS[id(harmean)] = _vec_harmean


_register_vectorized()


JOIN_OPS: dict[str, Callable[[Any, Any], Any]] = {
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge
}

# for the passed argument to an aggregation
InitAggregation = Mapping[
    str,
    Callable[[np.ndarray[Any, Any]], Any]
    | tuple[Callable[[np.ndarray[Any, Any]], Any], str]
]


# for the result type of IterateBy
GroupDescription = tuple[
    tuple[Any, ...],  # tuple of unique values from group-by columns
    np.ndarray[Any, Any],  # int array of row indices into original tafra for this group
    'Tafra'  # sub-tafra for the group
]


class Union:
    """
    Union two `Tafra` together. Analogy to SQL UNION or
    `pandas.append`. All column names and dtypes must match.
    """
    @staticmethod
    def _validate(left: 'Tafra', right: 'Tafra') -> None:
        """
        Validate the `Tafra` before applying.
        """
        # These should be unreachable unless attributes were directly modified
        if len(left._data) != len(left._dtypes):
            raise ValueError('This `Tafra` length of data and dtypes do not match')
        if len(right._data) != len(right._dtypes):
            raise ValueError('right `Tafra` length of data and dtypes do not match')

        # ensure same number of columns
        if len(left._data) != len(right._data) or len(left._dtypes) != len(right._dtypes):
            raise ValueError(
                'This `Tafra` column count does not match right `Tafra` column count.')

        # ensure all columns in this `Tafra` exist in right `Tafra`
        # if len() is same AND all columns in this exist in right,
        # do not need to check right `Tafra` columns in this `Tafra`.
        for (data_column, value), (dtype_column, dtype) \
                in zip(left._data.items(), left._dtypes.items()):

            if data_column not in right._data or dtype_column not in right._dtypes:
                raise TypeError(
                    f'This `Tafra` column `{data_column}` does not exist in right `Tafra`.')

            # Compare user-declared dtypes (metadata = intent).
            # _format_dtype collapses string variants to 'str'.
            elif dtype != right._dtypes[dtype_column]:
                raise TypeError(
                    f'This `Tafra` column `{data_column}` dtype `{dtype}` '
                    f'does not match right `Tafra` dtype `{right._dtypes[dtype_column]}`.')

    def apply(self, left: 'Tafra', right: 'Tafra') -> 'Tafra':
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
            left._dtypes.copy()
        )

    def apply_inplace(self, left: 'Tafra', right: 'Tafra') -> None:
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

    @staticmethod
    def _encode_columns(
        col_arrays: list[np.ndarray[Any, Any]]
    ) -> tuple[list[np.ndarray[Any, Any]], list[np.ndarray[Any, Any] | None]]:
        """
        Encode columns for structured array compatibility.
        StringDType columns are converted to integer codes via np.unique.

        Returns (encoded_arrays, codebooks) where codebooks[i] is the
        unique values array for encoded columns, or None if no encoding
        was needed.
        """
        encoded = []
        codebooks: list[np.ndarray[Any, Any] | None] = []
        for c in col_arrays:
            if c.dtype.kind in ('T', 'U', 'S', 'O'):
                if _HAS_ACCEL and len(c) >= 50_000:
                    obj_arr = c.astype(object)
                    codes, _ = _c_encode_strings(obj_arr)
                    encoded.append(codes)
                    codebooks.append(None)
                else:
                    uniq, codes = np.unique(c, return_inverse=True)
                    encoded.append(codes)
                    codebooks.append(uniq)
            else:
                encoded.append(c)
                codebooks.append(None)
        return encoded, codebooks

    @staticmethod
    def _encode_columns_paired(
        left_cols: list[np.ndarray[Any, Any]],
        right_cols: list[np.ndarray[Any, Any]],
    ) -> tuple[list[np.ndarray[Any, Any]], list[np.ndarray[Any, Any]]]:
        """
        Encode left and right column pairs with a shared codebook.
        Ensures the same string value gets the same integer code on both sides.
        """
        left_enc = []
        right_enc = []
        for lc, rc in zip(left_cols, right_cols):
            if lc.dtype.kind in ('T', 'U', 'S', 'O'):
                combined = np.concatenate([lc, rc])
                if _HAS_ACCEL and len(combined) >= 50_000:
                    codes, _ = _c_encode_strings(combined.astype(object))
                else:
                    _, codes = np.unique(combined, return_inverse=True)
                left_enc.append(codes[:len(lc)])
                right_enc.append(codes[len(lc):])
            else:
                left_enc.append(lc)
                right_enc.append(rc)
        return left_enc, right_enc

    @staticmethod
    def _build_composite_key(
        encoded: list[np.ndarray[Any, Any]]
    ) -> np.ndarray[Any, Any]:
        """
        Combine multiple integer-coded columns into a single flat key.
        Uses positional encoding: key = c0 * N1*N2*... + c1 * N2*... + c2 * ...
        Falls back to structured array if values would overflow int64.
        """
        if len(encoded) == 1:
            return encoded[0]

        # empty arrays: return empty int64
        if len(encoded[0]) == 0:
            return np.array([], dtype=np.int64)

        # compute cardinality of each column
        cards = [int(c.max()) + 1 for c in encoded]

        # check for overflow: product of all cardinalities must fit int64
        product = 1
        overflow = False
        for card in cards:
            product *= card
            if product > 2**62:
                overflow = True
                break

        if not overflow:
            if _HAS_ACCEL:
                return _c_composite_key(
                    tuple(c.astype(np.int64) for c in encoded),
                    tuple(cards),
                )
            # Python fallback: flat integer key via positional encoding
            key = np.zeros(len(encoded[0]), dtype=np.int64)
            multiplier = 1
            for c, card in zip(reversed(encoded), reversed(cards)):
                key += c.astype(np.int64) * multiplier
                multiplier *= card
            return key
        else:
            raise ValueError(
                'Composite key space overflow: too many unique key '
                'combinations for multi-column join'
            )

    @staticmethod
    def _decode_unique(
        first_seen_indices: np.ndarray[Any, Any],
        col_arrays: list[np.ndarray[Any, Any]],
    ) -> list[tuple[Any, ...]]:
        """Build unique group tuples by indexing original columns at first-seen rows."""
        if len(col_arrays) == 1:
            return [(v,) for v in col_arrays[0][first_seen_indices]]
        else:
            return [
                tuple(c[idx] for c in col_arrays)
                for idx in first_seen_indices
            ]

    @staticmethod
    def _direct_labels_firstseen(
        data: np.ndarray[Any, Any], n_rows: int
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int]:
        """
        Assign first-seen-order group labels using direct array mapping.
        O(n + max_key) — no sort. Requires non-negative integer keys.

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
    def _direct_labels_sorted(
        data: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], int]:
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
        tafra: 'Tafra', columns: Iterable[str]
    ) -> tuple[np.ndarray[Any, Any], list[np.ndarray[Any, Any]]]:
        """Encode columns and build composite integer key."""
        cols = list(columns)
        col_arrays = [tafra._data[c] for c in cols]
        encoded, _ = GroupSet._encode_columns(col_arrays)
        if len(cols) == 1:
            data = encoded[0]
        else:
            data = GroupSet._build_composite_key(encoded)
        return data, col_arrays

    @staticmethod
    def _build_group_indices(
        tafra: 'Tafra', columns: Iterable[str]
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
                np.ascontiguousarray(data))
            group_indices = list(group_indices_list)
        else:
            labels, first_seen_idx, n_groups = GroupSet._direct_labels_firstseen(
                data, tafra._rows)
            sorted_row_indices = np.argsort(labels, kind='stable')
            counts = np.bincount(labels, minlength=n_groups)
            splits = np.cumsum(counts[:-1])
            group_indices = list(np.split(sorted_row_indices, splits))

        unique: list[tuple[Any, ...]] = GroupSet._decode_unique(
            first_seen_idx, col_arrays)

        return unique, group_indices

    @staticmethod
    def _build_group_labels(
        tafra: 'Tafra', columns: Iterable[str]
    ) -> tuple[list[tuple[Any, ...]], np.ndarray[Any, Any], int]:
        """
        Build per-row group labels (integers 0..n_groups-1) in first-seen order.

        Returns (unique, labels, n_groups) where labels[i] is the group index
        for row i.
        """
        data, col_arrays = GroupSet._prepare_keys(tafra, columns)

        labels, first_seen_idx, n_groups = GroupSet._direct_labels_firstseen(
            data, tafra._rows)

        unique: list[tuple[Any, ...]] = GroupSet._decode_unique(
            first_seen_idx, col_arrays)

        return unique, labels, n_groups

    @staticmethod
    def _validate(tafra: 'Tafra', columns: Iterable[str]) -> None:  # pragma: no cover
        """
        Validate the `Tafra` before applying.
        """
        if tafra._rows < 1:
            raise ValueError('No rows exist in `tafra`.')
        tafra._validate_columns(columns)


@dc.dataclass
class AggMethod(GroupSet):
    """
    Basic methods for aggregations over a data table.
    """
    group_by_cols: Iterable[str]
    aggregation: dc.InitVar[InitAggregation]
    _aggregation: Mapping[
        str, tuple[Callable[[np.ndarray[Any, Any]], Any], str]
    ] = dc.field(init=False)
    iter_fn: Mapping[str, Callable[[np.ndarray[Any, Any]], Any]]

    def __post_init__(self, aggregation: InitAggregation) -> None:
        self._aggregation = dict()
        for rename, agg in aggregation.items():
            if callable(agg):
                self._aggregation[rename] = (agg, rename)
            elif (isinstance(agg, Sequence) and len(agg) == 2
                  and callable(agg[0])):
                self._aggregation[rename] = agg
            else:
                raise ValueError(f'{rename}: {agg} is not a valid aggregation argument')

        for rename, agg in self.iter_fn.items():
            if not callable(agg):
                raise ValueError(f'{rename}: {agg} is not a valid aggregation argument')

    def result_factory(
        self, fn: Callable[[str, str], np.ndarray[Any, Any]]
    ) -> dict[str, np.ndarray[Any, Any]]:
        """
        Factory function to generate the dict for the results set.
        A function to take the new column name and source column name
        and return an empty `np.ndarray` should be given.
        """
        return {
            rename: fn(rename, col) for rename, col in (
                *((col, col) for col in self.group_by_cols),
                *((rename, agg[1]) for rename, agg in self._aggregation.items())
            )
        }

    def iter_fn_factory(
        self, fn: Callable[[], np.ndarray[Any, Any]]
    ) -> dict[str, np.ndarray[Any, Any]]:
        return {rename: fn() for rename in self.iter_fn.keys()}

    def apply(self, tafra: 'Tafra') -> 'Tafra':
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

    def apply(self, tafra: 'Tafra') -> 'Tafra':
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
        self._validate(tafra, (
            *self.group_by_cols,
            *(col for (_, col) in self._aggregation.values())
        ))

        # check if all aggregations can be vectorized
        all_vectorized = (
            not self.iter_fn
            and all(id(fn) in _VECTORIZED_AGGS
                    for fn, _ in self._aggregation.values())
        )

        if all_vectorized and self._aggregation:
            # fast path: vectorized aggregation via labels, no per-group loop
            unique, labels, n_groups = self._build_group_labels(
                tafra, self.group_by_cols)

            result: dict[str, np.ndarray[Any, Any]] = {}
            for i, col in enumerate(self.group_by_cols):
                vals = tafra._data[col]
                # pick one representative value per group
                first_occurrence = np.empty(n_groups, dtype=np.intp)
                first_occurrence[labels] = np.arange(len(labels))
                # overwrite gives last, we want first — reverse
                first_occurrence[labels[::-1]] = np.arange(
                    len(labels) - 1, -1, -1)
                result[col] = vals[first_occurrence]

            for rename, (fn, col) in self._aggregation.items():
                vec_fn = _VECTORIZED_AGGS[id(fn)]
                result[rename] = vec_fn(tafra._data[col], labels, n_groups)

            return Tafra(result)

        # standard path: per-group loop
        unique, group_indices = self._build_group_indices(
            tafra, self.group_by_cols)
        n_groups = len(unique)

        result = self.result_factory(
            lambda rename, col: np.empty(n_groups, dtype=tafra._data[col].dtype))
        iter_fn = self.iter_fn_factory(lambda: np.ones(n_groups, dtype=int))

        for i, (u, rows) in enumerate(zip(unique, group_indices)):
            for val, col in zip(u, self.group_by_cols):
                result[col][i] = val

            for rename, (fn, col) in self._aggregation.items():
                result[rename][i] = fn(tafra._data[col][rows])

            for rename, fn in self.iter_fn.items():
                iter_fn[rename][i] = fn(np.full(len(rows), i, dtype=int))

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

    def apply(self, tafra: 'Tafra') -> 'Tafra':
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
        self._validate(tafra, (
            *self.group_by_cols,
            *(col for (_, col) in self._aggregation.values())
        ))

        # check if all aggregations can be vectorized
        all_vectorized = (
            not self.iter_fn
            and all(id(fn) in _VECTORIZED_AGGS
                    for fn, _ in self._aggregation.values())
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
                vec_fn = _VECTORIZED_AGGS[id(fn)]
                group_values = vec_fn(tafra._data[col], labels, n_groups)
                result[rename] = group_values[labels]

            return Tafra(result)

        # standard path: per-group loop
        unique, group_indices = self._build_group_indices(
            tafra, self.group_by_cols)

        result = self.result_factory(
            lambda rename, col: np.empty_like(tafra._data[col]))
        iter_fn = self.iter_fn_factory(lambda: np.ones(tafra._rows, dtype=int))

        for i, (u, rows) in enumerate(zip(unique, group_indices)):
            for col in self.group_by_cols:
                result[col][rows] = tafra._data[col][rows]

            for rename, (fn, col) in self._aggregation.items():
                result[rename][rows] = fn(tafra._data[col][rows])

            for rename, fn in self.iter_fn.items():
                iter_fn[rename][rows] = fn(np.full(len(rows), i, dtype=int))

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

    def apply(self, tafra: 'Tafra') -> Iterator[GroupDescription]:
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

    def _validate_dtypes(self, l_table: 'Tafra', r_table: 'Tafra') -> None:
        for l_column, r_column, _ in self.on:
            l_dtype = l_table._dtypes[l_column]
            r_dtype = r_table._dtypes[r_column]

            if l_dtype == r_dtype:
                continue

            # Check base type via _reduce_dtype (collapses str variants)
            l_base = Tafra._reduce_dtype(l_dtype)
            r_base = Tafra._reduce_dtype(r_dtype)
            if l_base == r_base:
                # Same base type (e.g. both 'str') — allow
                continue

            # Check numpy kind compatibility for numeric promotion
            l_np = np.dtype(l_dtype)
            r_np = np.dtype(r_dtype)
            if l_np.kind == r_np.kind:
                # Same family (e.g. int32 vs int64) — promote via result_type
                promoted = np.result_type(l_np, r_np)
                promoted_label = Tafra._format_dtype(promoted)
                l_table._data[l_column] = l_table._data[l_column].astype(promoted)
                l_table._dtypes[l_column] = promoted_label
                r_table._data[r_column] = r_table._data[r_column].astype(promoted)
                r_table._dtypes[r_column] = promoted_label
            else:
                raise TypeError(
                    f'This `Tafra` column `{l_column}` dtype `{l_dtype}` '
                    f'does not match other `Tafra` dtype `{r_dtype}`.')

    @staticmethod
    def _non_null_mask(
        cols: list[np.ndarray[Any, Any]]
    ) -> np.ndarray[Any, Any]:
        """Build a boolean mask that is True for rows with no null in any key column."""
        n = len(cols[0])
        valid = np.ones(n, dtype=bool)
        for c in cols:
            kind = c.dtype.kind
            if kind == 'f':
                valid &= ~np.isnan(c)
            elif kind in ('M', 'm'):
                valid &= ~np.isnat(c)
            elif kind in ('T', 'U'):
                # StringDType: None is the null sentinel
                valid &= np.array([x is not None for x in c], dtype=bool)
            elif kind == 'O':
                valid &= np.array([x is not None for x in c], dtype=bool)
        return valid

    @staticmethod
    def _validate_ops(ops: Iterable[str]) -> None:
        for op in ops:
            _op = JOIN_OPS.get(op, None)
            if _op is None:
                raise TypeError(f'The operator {op} is not valid.')

    @staticmethod
    def _build_composite_key(
        cols: list[np.ndarray[Any, Any]]
    ) -> np.ndarray[Any, Any]:
        """Build a single sortable key array from multiple columns."""
        if len(cols) == 1:
            return cols[0]
        # structured array for multi-column sort
        dt = np.dtype([(f'f{i}', c.dtype) for i, c in enumerate(cols)])
        key = np.empty(len(cols[0]), dtype=dt)
        for i, c in enumerate(cols):
            key[f'f{i}'] = c
        return key

    @staticmethod
    def _sort_merge_indices(
        left_key: np.ndarray[Any, Any],
        right_key: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Compute inner-join index pairs using sort-merge.
        Returns (left_indices, right_indices) as intp arrays.
        """
        right_order = np.argsort(right_key, kind='stable')
        right_sorted = right_key[right_order]

        left_lo = np.searchsorted(right_sorted, left_key, side='left')
        left_hi = np.searchsorted(right_sorted, left_key, side='right')
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
        self, left_t: 'Tafra', right_t: 'Tafra'
    ) -> tuple[list[str], dict[str, str]]:
        """Compute deduplicated output columns and dtypes."""
        seen_cols: dict[str, None] = {}
        for c in chain(left_t._data.keys(), right_t._data.keys()):
            if not self.select or c in self.select:
                seen_cols[c] = None
        join_cols = list(seen_cols.keys())
        dtypes: dict[str, str] = {
            c: d for c, d in chain(
                right_t._dtypes.items(), left_t._dtypes.items())
            if c in seen_cols
        }
        return join_cols, dtypes

    def apply(self, left_t: 'Tafra', right_t: 'Tafra') -> 'Tafra':
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

    def apply(self, left_t: 'Tafra', right_t: 'Tafra') -> 'Tafra':
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
        # Empty table shortcut — inner join with either side empty → empty result
        if left_t._rows < 1 or right_t._rows < 1:
            warnings.warn(
                'Join: one or both tables have zero rows. '
                'Returning shortcut result.', stacklevel=2)
            join_cols, dtypes = self._resolve_join_cols(left_t, right_t)
            return Tafra(
                {c: np.array(
                    [], dtype=left_t._data[c].dtype
                    if c in left_t._data
                    else right_t._data[c].dtype)
                 for c in join_cols},
                dtypes
            )

        left_cols, right_cols, ops = list(zip(*self.on))
        self._validate(left_t, left_cols)
        self._validate(right_t, right_cols)
        self._validate_dtypes(left_t, right_t)
        self._validate_ops(ops)

        join_cols, dtypes = self._resolve_join_cols(left_t, right_t)
        all_equi = all(op_str == '==' for _, _, op_str in self.on)

        if all_equi:
            # Encode left+right together for consistent codebooks
            left_cols_data = [left_t._data[lc] for lc, _, _ in self.on]
            right_cols_data = [right_t._data[rc] for _, rc, _ in self.on]

            # NULL != NULL: filter out rows with nulls in key columns
            left_valid = self._non_null_mask(left_cols_data)
            right_valid = self._non_null_mask(right_cols_data)
            left_cols_filt = [c[left_valid] for c in left_cols_data]
            right_cols_filt = [c[right_valid] for c in right_cols_data]
            left_orig_idx = np.where(left_valid)[0]
            right_orig_idx = np.where(right_valid)[0]

            l_enc, r_enc = GroupSet._encode_columns_paired(
                left_cols_filt, right_cols_filt)
            left_key = GroupSet._build_composite_key(l_enc)
            right_key = GroupSet._build_composite_key(r_enc)

            if _HAS_ACCEL:
                li, ri = _c_inner_join(
                    np.ascontiguousarray(left_key, dtype=np.int64),
                    np.ascontiguousarray(right_key, dtype=np.int64))
            else:
                li, ri = self._sort_merge_indices(left_key, right_key)

            # Map back to original row positions
            if len(li) > 0:
                li = left_orig_idx[li]
                ri = right_orig_idx[ri]

            if len(li) == 0:
                return Tafra(
                    {c: np.array(
                        [], dtype=left_t._data[c].dtype
                        if c in left_t._data
                        else right_t._data[c].dtype)
                     for c in join_cols},
                    dtypes
                )

            result: dict[str, np.ndarray[Any, Any]] = {}
            for c in join_cols:
                if c in left_t._data:
                    result[c] = left_t._data[c][li]
                else:
                    result[c] = right_t._data[c][ri]

            return Tafra(result, dtypes)

        else:
            _on = tuple(
                (left_col, right_col, JOIN_OPS[op])
                for left_col, right_col, op in self.on
            )
            right_rows = np.empty(right_t._rows, dtype=bool)
            join: dict[str, list[Any]] = {c: [] for c in join_cols}

            for i in range(left_t._rows):
                right_rows[:] = True
                for left_col, right_col, op in _on:
                    right_rows &= op(
                        left_t._data[left_col][i], right_t._data[right_col])

                right_count = int(np.sum(right_rows))
                if right_count <= 0:
                    continue

                for column in join_cols:
                    if column in left_t._data:
                        join[column].extend(
                            [left_t._data[column][i]] * right_count)
                    elif column in right_t._data:
                        join[column].extend(
                            right_t._data[column][right_rows])

            return Tafra(
                {c: np.array(v) for c, v in join.items()},
                dtypes
            )


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
        right_order = np.argsort(right_key, kind='stable')
        right_sorted = right_key[right_order]

        left_lo = np.searchsorted(right_sorted, left_key, side='left')
        left_hi = np.searchsorted(right_sorted, left_key, side='right')
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
                    ri[pos:pos + c] = right_order[left_lo[i]:left_hi[i]]
                    pos += c
                else:
                    ri[pos] = -1
                    pos += 1
        else:
            # all matched — fully vectorized
            offsets = np.repeat(left_lo, counts)
            group_starts = np.cumsum(counts) - counts
            within = np.arange(total, dtype=np.intp) - np.repeat(
                group_starts, counts)
            ri = right_order[offsets + within]

        return li, ri, has_null

    def apply(self, left_t: 'Tafra', right_t: 'Tafra') -> 'Tafra':
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
        # Empty table shortcut
        if left_t._rows < 1:
            warnings.warn(
                'Join: one or both tables have zero rows. '
                'Returning shortcut result.', stacklevel=2)
            join_cols, dtypes = self._resolve_join_cols(left_t, right_t)
            return Tafra(
                {c: np.array(
                    [], dtype=left_t._data[c].dtype
                    if c in left_t._data
                    else right_t._data[c].dtype)
                 for c in join_cols},
                dtypes
            )
        if right_t._rows < 1:
            warnings.warn(
                'Join: one or both tables have zero rows. '
                'Returning shortcut result.', stacklevel=2)
            join_cols, dtypes = self._resolve_join_cols(left_t, right_t)
            n_left = left_t._rows
            shortcut: dict[str, np.ndarray[Any, Any]] = {}
            for c in join_cols:
                if c in left_t._data:
                    shortcut[c] = left_t._data[c].copy()
                else:
                    # Null-fill right columns
                    col_kind = right_t._data[c].dtype.kind
                    if col_kind in ('T', 'U'):
                        out = np.empty(
                            n_left,
                            dtype=np.dtypes.StringDType(na_object=None),  # type: ignore[call-arg]
                        )
                        out[:] = None  # type: ignore[assignment]
                        dtypes[c] = 'str'
                    elif col_kind == 'f':
                        out = np.full(
                            n_left, np.nan, dtype=right_t._data[c].dtype)
                    elif col_kind in ('M', 'm'):
                        nat = np.array(
                            'NaT', dtype=right_t._data[c].dtype).item()
                        out = np.full(
                            n_left, nat, dtype=right_t._data[c].dtype)
                    else:
                        out = cast(
                            np.ndarray[Any, Any],
                            np.empty(n_left, dtype=object),
                        )
                        out[:] = None  # type: ignore[assignment]
                        dtypes[c] = 'object'
                    shortcut[c] = out
            return Tafra(shortcut, dtypes)

        left_cols, right_cols, ops = list(zip(*self.on))
        self._validate(left_t, left_cols)
        self._validate(right_t, right_cols)
        self._validate_dtypes(left_t, right_t)
        self._validate_ops(ops)

        join_cols, dtypes = self._resolve_join_cols(left_t, right_t)
        all_equi = all(op_str == '==' for _, _, op_str in self.on)

        if all_equi:
            left_cols_data = [left_t._data[lc] for lc, _, _ in self.on]
            right_cols_data = [right_t._data[rc] for _, rc, _ in self.on]

            # NULL != NULL: filter out rows with nulls in key columns
            left_valid = self._non_null_mask(left_cols_data)
            right_valid = self._non_null_mask(right_cols_data)
            left_cols_filt = [c[left_valid] for c in left_cols_data]
            right_cols_filt = [c[right_valid] for c in right_cols_data]
            left_orig_idx = np.where(left_valid)[0]
            right_orig_idx = np.where(right_valid)[0]
            left_null_idx = np.where(~left_valid)[0]

            l_enc, r_enc = GroupSet._encode_columns_paired(
                left_cols_filt, right_cols_filt)
            left_key = GroupSet._build_composite_key(l_enc)
            right_key = GroupSet._build_composite_key(r_enc)

            if _HAS_ACCEL:
                li, ri, has_null = _c_left_join(
                    np.ascontiguousarray(left_key, dtype=np.int64),
                    np.ascontiguousarray(right_key, dtype=np.int64))
            else:
                li, ri, has_null = self._left_join_indices(left_key, right_key)

            # Map back to original row positions
            if len(li) > 0:
                li = left_orig_idx[li]
                ri_mapped = np.where(ri >= 0, right_orig_idx[ri], -1)
            else:
                ri_mapped = ri

            # Append null-key left rows (they always get unmatched)
            if len(left_null_idx) > 0:
                has_null = True
                li = np.concatenate([li, left_null_idx])
                ri_mapped = np.concatenate([
                    ri_mapped,
                    np.full(len(left_null_idx), -1, dtype=np.intp)])

            ri = ri_mapped

            if has_null:
                for c in join_cols:
                    if c not in left_t._data and dtypes.get(c) != 'object':
                        col_kind = right_t._data[c].dtype.kind
                        # Kinds with native null: T/U (str), f (float), M/m (datetime)
                        if col_kind not in ('T', 'U', 'f', 'M', 'm'):
                            dtypes[c] = 'object'

            result: dict[str, np.ndarray[Any, Any]] = {}
            matched = ri >= 0
            for c in join_cols:
                if c in left_t._data:
                    result[c] = left_t._data[c][li]
                else:
                    # right column: fill matched rows, null for unmatched
                    if has_null:
                        col_kind = right_t._data[c].dtype.kind
                        if col_kind in ('T', 'U'):
                            # String types: use StringDType(na_object=None)
                            out = np.empty(len(li), dtype=np.dtypes.StringDType(na_object=None))  # type: ignore[call-arg]
                            out[matched] = right_t._data[c][ri[matched]]
                            out[~matched] = None  # type: ignore[assignment]
                            dtypes[c] = 'str'
                        elif col_kind == 'f':
                            # Float types: use NaN for missing
                            out = np.full(len(li), np.nan, dtype=right_t._data[c].dtype)
                            out[matched] = right_t._data[c][ri[matched]]
                        elif col_kind in ('M', 'm'):
                            # datetime64/timedelta64: use NaT for missing
                            nat = np.array('NaT', dtype=right_t._data[c].dtype).item()
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
            _on = tuple(
                (left_col, right_col, JOIN_OPS[op])
                for left_col, right_col, op in self.on
            )
            right_rows = np.empty(right_t._rows, dtype=bool)
            join: dict[str, list[Any]] = {c: [] for c in join_cols}
            has_null = False

            for i in range(left_t._rows):
                right_rows[:] = True
                for left_col, right_col, op in _on:
                    right_rows &= op(
                        left_t._data[left_col][i], right_t._data[right_col])

                right_count = int(np.sum(right_rows))

                for column in join_cols:
                    if column in left_t._data:
                        join[column].extend(
                            [left_t._data[column][i]] * max(1, right_count))
                    elif column in right_t._data:
                        if right_count <= 0:
                            has_null = True
                            join[column].append(None)
                            col_kind = right_t._data[column].dtype.kind
                            if (col_kind not in ('T', 'U', 'f', 'M', 'm')
                                    and dtypes[column] != 'object'):
                                warnings.warn(
                                    f"Left join: column '{column}' "
                                    f"(dtype {right_t._data[column].dtype}) "
                                    f"has unmatched rows and no native null "
                                    f"representation. Dtype has been cast to "
                                    f"object. Use .astype(float) if NaN "
                                    f"semantics are needed.",
                                    stacklevel=3,
                                )
                                dtypes[column] = 'object'
                        else:
                            join[column].extend(
                                right_t._data[column][right_rows])

            result_data: dict[str, np.ndarray[Any, Any]] = {}
            for c, v in join.items():
                col_kind = (right_t._data[c].dtype.kind
                            if c in right_t._data else '')
                if c not in left_t._data and col_kind in ('T', 'U') and has_null:
                    result_data[c] = np.array(
                        v,
                        dtype=np.dtypes.StringDType(na_object=None),  # type: ignore[call-arg]
                    )
                    dtypes[c] = 'str'
                elif c not in left_t._data and col_kind == 'f' and has_null:
                    result_data[c] = np.array(
                        [np.nan if x is None else x for x in v],
                        dtype=right_t._data[c].dtype)
                elif (c not in left_t._data
                      and col_kind in ('M', 'm') and has_null):
                    nat = np.array('NaT', dtype=right_t._data[c].dtype).item()
                    result_data[c] = np.array(
                        [nat if x is None else x for x in v],
                        dtype=right_t._data[c].dtype)
                else:
                    result_data[c] = np.array(v)

            return Tafra(result_data, dtypes)


@dc.dataclass
class CrossJoin(Join):
    """
    A cross join.

    Analogy to SQL CROSS JOIN, or `pandas.merge(..., how='outer')
    using temporary columns of static value to intersect all rows`.

    Parameters
    ----------
    select: Iterable[str] = []
        The columns to return. If not given, all unique columns names
        are returned. If the column exists in both `Tafra`,
        prefers the left over the right.
    """

    def apply(self, left_t: 'Tafra', right_t: 'Tafra') -> 'Tafra':
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
        # Empty table shortcut — cross join with either side empty → empty result
        if left_t._rows < 1 or right_t._rows < 1:
            warnings.warn(
                'Join: one or both tables have zero rows. '
                'Returning shortcut result.', stacklevel=2)
            join_cols, dtypes = self._resolve_join_cols(left_t, right_t)
            return Tafra(
                {c: np.array(
                    [], dtype=left_t._data[c].dtype
                    if c in left_t._data
                    else right_t._data[c].dtype)
                 for c in join_cols},
                dtypes
            )

        self._validate_dtypes(left_t, right_t)

        left_rows = left_t._rows
        right_rows = right_t._rows

        select = list(self.select)
        if len(select) > 0:
            left_cols = [c for c in select if c in left_t._data]
            right_cols = [c for c in select if c in right_t._data]
        else:
            left_cols = list(left_t._data.keys())
            right_cols = list(right_t._data.keys())

        left_data = dict(left_t[left_cols].key_map(np.repeat, repeats=right_rows))
        right_data = dict(right_t[right_cols].key_map(np.tile, reps=left_rows))

        # Left takes precedence on shared column names (matching inner/left join)
        result: dict[str, np.ndarray[Any, Any]] = {}
        for c, v in right_data.items():
            if c not in left_data:
                result[c] = v
        result.update(left_data)

        return Tafra(result)


# Import here to resolve circular dependency
from .base import Tafra
