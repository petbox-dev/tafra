import operator
from collections import OrderedDict
from itertools import chain
import dataclasses as dc

import numpy as np

from typing import Any, Callable, Dict, List, Iterable, Tuple, Optional, Union
from typing import cast


JOIN_OPS: Dict[str, Callable[[Any, Any], Any]] = {
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge
}


# for the passed argument to an aggregation
InitAggregation = Dict[
    str,
    Union[
        Callable[[np.ndarray], Any],
        Tuple[Callable[[np.ndarray], Any], str]
    ]
]


# for the result type of IterateBy
GroupDescription = Tuple[
    Tuple[Any, ...],  # tuple of unique values from group-by columns
    np.ndarray,  # int array of row indices into original tafra for this group
    'Tafra'  # sub-tafra for the group
]


@dc.dataclass
class GroupSet():
    """A `GroupSet` is the set of columns by which we construct our groups.
    """

    @staticmethod
    def _unique_groups(tafra: 'Tafra', columns: Iterable[str]) -> List[Any]:
        """Construct a unique set of grouped values.
        Uses `OrderedDict` rather than `set` to maintain order.
        """
        return list(OrderedDict.fromkeys(zip(*(tafra[col] for col in columns))))

    @staticmethod
    def _validate(tafra: 'Tafra', columns: Iterable[str]) -> None:
        rows = tafra.rows
        if rows < 1:
            raise ValueError(f'No rows exist in `tafra`.')

        tafra._validate_columns(columns)


@dc.dataclass
class AggMethod(GroupSet):
    """Basic methods for aggregations over a data table.
    """
    _group_by_cols: Iterable[str]
    aggregation: dc.InitVar[InitAggregation]
    _aggregation: Dict[str, Tuple[Callable[[np.ndarray], Any], str]] = dc.field(init=False)
    _iter_fn: Dict[str, Callable[[np.ndarray], Any]]

    def __post_init__(self, aggregation: InitAggregation):
        self._aggregation = dict()
        for rename, agg in aggregation.items():
            if callable(agg):
                self._aggregation[rename] = cast(
                    Tuple[Callable[[np.ndarray], Any], str],
                    (agg, rename))
            elif (isinstance(agg, Iterable) and len(agg) == 2
                  and callable(cast(Tuple, agg)[0])):
                self._aggregation[rename] = agg
            else:
                raise ValueError(f'{rename}: {agg} is not a valid aggregation argument')

        for rename, agg in self._iter_fn.items():
            if not callable(agg):
                raise ValueError(f'{rename}: {agg} is not a valid aggregation argument')

    def result_factory(self, fn: Callable[[str, str], np.ndarray]) -> Dict[str, np.ndarray]:
        """Factory function to generate the dict for the results set.
        A function to take the new column name and source column name
        and return an empty `np.ndarray` should be given.
        """
        return {
            rename: fn(rename, col) for rename, col in (
                *((col, col) for col in self._group_by_cols),
                *((rename, agg[1]) for rename, agg in self._aggregation.items())
            )
        }

    def iter_fn_factory(self, fn: Callable[[], np.ndarray]) -> Dict[str, np.ndarray]:
        return {rename: fn() for rename in self._iter_fn.keys()}

    def apply(self, tafra: 'Tafra'):
        ...


class GroupBy(AggMethod):
    """Analogy to SQL `GROUP BY`, not `pandas.DataFrame.groupby()`. A `reduce` operation.
    """

    def apply(self, tafra: 'Tafra') -> 'Tafra':
        self._validate(tafra, (
            *self._group_by_cols,
            *(col for (_, col) in self._aggregation.values())
        ))
        unique = self._unique_groups(tafra, self._group_by_cols)
        result = self.result_factory(
            lambda rename, col: np.empty(len(unique), dtype=tafra[col].dtype))
        iter_fn = self.iter_fn_factory(lambda: np.ones(len(unique), dtype=int))
        ones = np.ones(tafra.rows, dtype=int)

        for i, u in enumerate(unique):
            which_rows = np.full(tafra.rows, True)

            for val, col in zip(u, self._group_by_cols):
                which_rows &= tafra[col] == val
                result[col][i] = val

            for rename, (fn, col) in self._aggregation.items():
                result[rename][i] = fn(tafra[col][which_rows])

            for rename, fn in self._iter_fn.items():
                iter_fn[rename][i] = fn(i * ones[which_rows])

        result.update(iter_fn)
        return Tafra(result)


class Transform(AggMethod):
    """Analogy to `pandas.DataFrame.transform()`,
    i.e. a SQL `GROUP BY` and `LEFT JOIN` back to the original table.
    """

    def apply(self, tafra: 'Tafra') -> 'Tafra':
        self._validate(tafra, (
            *self._group_by_cols,
            *(col for (_, col) in self._aggregation.values())
        ))
        unique = self._unique_groups(tafra, self._group_by_cols)
        result = self.result_factory(
            lambda rename, col: np.empty_like(tafra[col]))
        iter_fn = self.iter_fn_factory(lambda: np.ones(tafra.rows, dtype=int))
        ones = np.ones(tafra.rows, dtype=int)

        for i, u in enumerate(unique):
            which_rows = np.full(tafra.rows, True)

            for val, col in zip(u, self._group_by_cols):
                which_rows &= tafra[col] == val
                result[col][which_rows] = tafra[col][which_rows]

            for rename, agg in self._aggregation.items():
                fn, col = agg
                result[rename][which_rows] = fn(tafra[col][which_rows])

            for rename, fn in self._iter_fn.items():
                iter_fn[rename][which_rows] = fn(i * ones[which_rows])

        result.update(iter_fn)
        return Tafra(result)


@dc.dataclass
class IterateBy(GroupSet):
    """Analogy to `pandas.DataFrame.groupby()`, i.e. an Iterable of `Tafra` objects.
    Yields tuples of ((unique grouping values, ...), row indices array, subset tafra)
    """
    _group_by_cols: Iterable[str]

    def apply(self, tafra: 'Tafra') -> Iterable[GroupDescription]:
        self._validate(tafra, self._group_by_cols)
        unique = self._unique_groups(tafra, self._group_by_cols)

        for u in unique:
            which_rows = np.full(tafra.rows, True)

            for val, col in zip(u, self._group_by_cols):
                which_rows &= tafra[col] == val

            yield (u, which_rows, tafra[which_rows])


@dc.dataclass
class Join(GroupSet):
    """Base class for SQL-like JOINs.
    """
    _on: Iterable[Tuple[str, str, str]]
    _select: Iterable[str]

    @staticmethod
    def _validate_dtypes(left_t: 'Tafra', right_t: 'Tafra'):
        for (data_column, left_value), (dtype_column, left_dtype) \
                in zip(left_t._data.items(), left_t._dtypes.items()):
            right_value = right_t._data.get(data_column, None)
            right_dtype = right_t._dtypes.get(dtype_column, None)

            if right_value is None or right_dtype is None:
                continue

            elif left_value.dtype != right_value.dtype:
                raise ValueError(
                    f'This `Tafra` column `{data_column}` dtype `{left_value.dtype}` '
                    f'does not match other `Tafra` dtype `{right_value.dtype}`.')

            elif left_dtype != right_dtype or left_dtype != right_dtype:
                raise ValueError(
                    f'This `Tafra` column `{data_column}` dtype `{left_dtype}` '
                    f'does not match other `Tafra` dtype `{right_dtype}`.')

    @staticmethod
    def _validate_ops(ops: Iterable[str]):
        for op in ops:
            _op = JOIN_OPS.get(op, None)
            if _op is None:
                raise ValueError(f'The operator {op} is not valid.')

    def apply(self, left_t: 'Tafra', right_t: 'Tafra') -> 'Tafra':
        ...


class InnerJoin(Join):
    """Analogy to SQL INNER JOIN, or `pandas.merge(..., how='inner')`,
    """

    def apply(self, left_t: 'Tafra', right_t: 'Tafra') -> 'Tafra':
        left_cols, right_cols, ops = list(zip(*self._on))
        self._validate(left_t, left_cols)
        self._validate(right_t, right_cols)
        self._validate_dtypes(left_t, right_t)
        self._validate_ops(ops)

        _on = tuple((left_col, right_col, JOIN_OPS[op]) for left_col, right_col, op in self._on)

        join: Dict[str, List] = {column: list() for column in chain(
            left_t._data.keys(),
            right_t._data.keys()
        ) if not self._select
            or (self._select and column in self._select)}

        dtypes: Dict[str, str] = {column: dtype for column, dtype in chain(
            left_t._dtypes.items(),
            right_t._dtypes.items()
        ) if column in join.keys()}

        for i in range(left_t.rows):
            right_rows = np.full(right_t.rows, True)

            for left_col, right_col, op in _on:
                right_rows &= op(left_t[left_col][i], right_t[right_col])

            right_count = np.sum(right_rows)

            # this is the only difference from the LeftJoin
            if right_count <= 0:
                continue

            for column in join.keys():
                if column in left_t._data:
                    join[column].extend(max(1, right_count) * [left_t[column][i]])

                elif column in right_t._data:
                    if right_count <= 0:
                        join[column].append(None)
                        if dtypes[column] != 'object': dtypes[column] = 'object'
                    else:
                        join[column].extend(right_t[column][right_rows])

        return Tafra(
            {column: np.array(value)
             for column, value in join.items()},
            dtypes
        )


class LeftJoin(Join):
    """Analogy to SQL LEFT JOIN, or `pandas.merge(..., how='left')`,
    """

    def apply(self, left_t: 'Tafra', right_t: 'Tafra') -> 'Tafra':
        left_cols, right_cols, ops = list(zip(*self._on))
        self._validate(left_t, left_cols)
        self._validate(right_t, right_cols)
        self._validate_dtypes(left_t, right_t)
        self._validate_ops(ops)

        _on = tuple((left_col, right_col, JOIN_OPS[op]) for left_col, right_col, op in self._on)

        join: Dict[str, List] = {column: list() for column in chain(
            left_t._data.keys(),
            right_t._data.keys()
        ) if not self._select
            or (self._select and column in self._select)}

        dtypes: Dict[str, str] = {column: dtype for column, dtype in chain(
            left_t._dtypes.items(),
            right_t._dtypes.items()
        ) if column in join.keys()}

        for i in range(left_t.rows):
            right_rows = np.full(right_t.rows, True)

            for left_col, right_col, op in _on:
                right_rows &= op(left_t[left_col][i], right_t[right_col])

            right_count = np.sum(right_rows)

            for column in join.keys():
                if column in left_t._data:
                    join[column].extend(max(1, right_count) * [left_t[column][i]])

                elif column in right_t._data:
                    if right_count <= 0:
                        join[column].append(None)
                        if dtypes[column] != 'object': dtypes[column] = 'object'
                    else:
                        join[column].extend(right_t[column][right_rows])

        return Tafra(
            {column: np.array(value)
             for column, value in join.items()},
            dtypes
        )


@dc.dataclass
class CrossJoin(Join):
    """Analogy to SQL CROSS JOIN, or `pandas.merge(..., how='outer')
    using temporary columns of static value to intersect all rows`.
    """

    def apply(self, left_t: 'Tafra', right_t: 'Tafra') -> 'Tafra':
        self._validate_dtypes(left_t, right_t)

        join: Dict[str, List] = {column: list() for column in chain(
            left_t._data.keys(),
            right_t._data.keys()
        ) if not self._select
            or (self._select and column in self._select)}

        dtypes: Dict[str, str] = {column: dtype for column, dtype in chain(
            left_t._dtypes.items(),
            right_t._dtypes.items()
        ) if column in join.keys()}

        right_count = right_t.rows

        for i in range(left_t.rows):
            for column in join.keys():
                if column in left_t._data:
                    join[column].extend(max(1, right_count) * [left_t[column][i]])

                elif column in right_t._data:
                    join[column].extend(right_t[column])

        return Tafra(
            {column: np.array(value)
             for column, value in join.items()},
            dtypes
        )


# Import here to resolve circular dependency
from .tafra import Tafra
