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
import operator
import warnings
from itertools import chain
import dataclasses as dc

import numpy as np

from typing import Any, Callable, Mapping, Sequence, Iterable, Iterator
from typing import cast

from . import _keys
from . import _kernels
from ._reducers import VectorizedAggregation, vectorized_aggregation
from ._reducers import percentile, geomean, harmean

# Dtype kinds that can hold null values (None, NaN, NaT).
# Used by _non_null_mask to skip scanning for non-nullable types.
# 'T' = StringDType (can hold None with na_object=None)
# 'U' (fixed-width unicode <U) is excluded — it cannot hold None.
_NULLABLE_KINDS: frozenset[str] = frozenset("fMmTO")

# Row count from which _build_group_indices groups rows with the C hash
# kernel; below it, the numpy direct-mapping path is used.
_GROUP_INDICES_MIN_ROWS = 50_000

# Right-column dtype kinds a left join fills with their own null for an
# unmatched row: StringDType None ('T', 'U'), NaN ('f'), NaT ('M', 'm').
# Any other kind is cast to object to hold None.
_NATIVE_NULL_FILL_KINDS: frozenset[str] = frozenset("TUfMm")

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
    def _prepare_keys(
        tafra: "Tafra", columns: Iterable[str]
    ) -> tuple[np.ndarray[Any, Any], list[np.ndarray[Any, Any]]]:
        """Encode columns and build composite integer key."""
        column_arrays = [tafra._data[column] for column in columns]
        if not column_arrays:
            raise ValueError("Grouping needs at least one key column.")
        encoded = [_keys.encode_column(column) for column in column_arrays]
        data = _keys.build_composite_key(encoded, tafra._rows)
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

        if _kernels.HAS_ACCEL and tafra._rows >= _GROUP_INDICES_MIN_ROWS:
            first_seen_idx, group_indices_list, n_groups = _kernels.group_indices(data)
            group_indices = list(group_indices_list)
        else:
            labels, first_seen_idx, n_groups = _keys.direct_labels_firstseen(data, tafra._rows)
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
        _keys.direct_labels_firstseen: labels[i] is the group index for row i and
        first_seen_indices[j] is group j's first row.
        """
        data, _ = GroupSet._prepare_keys(tafra, columns)
        return _keys.direct_labels_firstseen(data, tafra._rows)

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
            vectorized_function = vectorized_aggregation(fn, tafra._data[col].dtype)
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
            labels, n_groups = _keys.direct_labels_sorted(data)

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

    def _promoted_key_columns(
        self, left_t: "Tafra", right_t: "Tafra"
    ) -> list[np.ndarray[Any, Any]]:
        """
        Each key column as the left rows followed by the right rows. The
        concatenation promotes the two sides to one dtype, as a comparison
        does (e.g. datetime64[s] to datetime64[ns]), so NULL is judged on the
        values that are compared: a unit cast can turn a datetime into NaT.
        """
        return [
            np.concatenate([left_t._data[left_column], right_t._data[right_column]])
            for left_column, right_column, _ in self.on
        ]

    def _non_null_masks(
        self, left_t: "Tafra", right_t: "Tafra"
    ) -> tuple[np.ndarray[Any, Any] | None, np.ndarray[Any, Any] | None]:
        """
        Per-row non-NULL masks of the left and right keys, judged on the values
        that are compared. Only a datetime64 or timedelta64 pair of different
        units is promoted first, since only a unit cast can create a NaT; other
        pairs keep their own columns, so no common dtype is needed (e.g. two
        StringDType columns with different na_object values).
        """
        left_columns: list[np.ndarray[Any, Any]] = []
        right_columns: list[np.ndarray[Any, Any]] = []
        for left_column_name, right_column_name, _ in self.on:
            left_column = left_t._data[left_column_name]
            right_column = right_t._data[right_column_name]
            if left_column.dtype.kind in "Mm" and left_column.dtype != right_column.dtype:
                promoted = np.concatenate([left_column, right_column])
                left_column, right_column = promoted[: left_t._rows], promoted[left_t._rows :]
            left_columns.append(left_column)
            right_columns.append(right_column)
        return self._non_null_mask(left_columns), self._non_null_mask(right_columns)

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
        combined_columns = self._promoted_key_columns(left_t, right_t)
        left_count = left_t._rows
        left_key, right_key = _keys.encode_join_keys(combined_columns, left_count)
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
        if _kernels.HAS_ACCEL:
            left_indices, right_indices = _kernels.inner_join(left_key, right_key)
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
        left_null, right_null = self._non_null_masks(left_t, right_t)

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
            if _kernels.HAS_ACCEL:
                left_indices, right_indices, has_null = _kernels.left_join(left_key, right_key)
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
        left_null, right_null = self._non_null_masks(left_t, right_t)

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
