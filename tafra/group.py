"""
Tafra: a minimalist dataframe


Author
------
Derrick W. Turk
David S. Fulford

Notes
-----
Created on April 25, 2020
"""
__all__ = ['GroupBy', 'Transform', 'IterateBy', 'InnerJoin', 'LeftJoin']

import operator
from collections import OrderedDict
from itertools import chain
import dataclasses as dc

import numpy as np

from typing import (Any, Callable, Dict, Mapping, List, Tuple, Optional, Union as _Union, Sequence,
                    Iterable, Iterator)
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
InitAggregation = Mapping[
    str,
    _Union[
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


class Union:
    """
    Union two :class:`Tafra` together. Analogy to SQL UNION or
    `pandas.append`. All column names and dtypes must match.
    """
    @staticmethod
    def _validate(left: 'Tafra', right: 'Tafra') -> None:
        """
        Validate the :class:`Tafra` before applying.
        """
        # These should be unreachable unless attributes were directly modified
        if len(left._data) != len(left._dtypes):
            assert 0, 'This `Tafra` length of data and dtypes do not match'
        if len(right._data) != len(right._dtypes):
            assert 0, 'right `Tafra` length of data and dtypes do not match'

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

            elif value.dtype != right._data[data_column].dtype:
                raise TypeError(
                    f'This `Tafra` column `{data_column}` dtype `{value.dtype}` '
                    f'does not match right `Tafra` dtype `{right._data[data_column].dtype}`.')

            # should not happen unless dtypes manually changed, but let's check it
            elif dtype != right._dtypes[dtype_column]:
                raise TypeError(
                    f'This `Tafra` column `{data_column}` dtype `{dtype}` '
                    f'does not match right `Tafra` dtype `{right._dtypes[dtype_column]}`.')

    def apply(self, left: 'Tafra', right: 'Tafra') -> 'Tafra':
        """
        Apply the :class:`Union_` to the :class:`Tafra`.

        Parameters
        ----------
            left: Tafra
                The left :class:`Tafra` to union.

            right: Tafra
                The right :class:`Tafra` to union.

        Returns
        -------
            tafra: Tafra
                The unioned :class`Tafra`.
        """
        self._validate(left, right)

        return Tafra(
            {column: np.append(value, right._data[column]) for column, value in left._data.items()},
            left._dtypes.copy()
        )

    def apply_inplace(self, left: 'Tafra', right: 'Tafra') -> None:
        """
        In-place version.

        Apply the :class:`Union_` to the :class:`Tafra`.

        Parameters
        ----------
            left: Tafra
                The left :class:`Tafra` to union.

            right: Tafra
                The right :class:`Tafra` to union.

        Returns
        -------
            tafra: Tafra
                The unioned :class`Tafra`.
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
    def _unique_groups(tafra: 'Tafra', columns: Iterable[str]) -> List[Any]:
        """
        Construct a unique set of grouped values.
        Uses :class:``OrderedDict`` rather than :class:``set`` to maintain order.
        """
        return list(OrderedDict.fromkeys(zip(*(tafra._data[col] for col in columns))))

    @staticmethod
    def _validate(tafra: 'Tafra', columns: Iterable[str]) -> None:  # pragma: no cover
        """
        Validate the :class:`Tafra` before applying.
        """
        assert tafra._rows >= 1, 'No rows exist in `tafra`.'
        tafra._validate_columns(columns)


@dc.dataclass
class AggMethod(GroupSet):
    """
    Basic methods for aggregations over a data table.
    """
    group_by_cols: Iterable[str]
    aggregation: dc.InitVar[InitAggregation]
    _aggregation: Mapping[str, Tuple[Callable[[np.ndarray], Any], str]] = dc.field(init=False)
    iter_fn: Mapping[str, Callable[[np.ndarray], Any]]

    def __post_init__(self, aggregation: InitAggregation) -> None:
        self._aggregation = dict()
        for rename, agg in aggregation.items():
            if callable(agg):
                self._aggregation[rename] = (agg, rename)
            elif (isinstance(agg, Sequence) and len(agg) == 2
                  and callable(cast(Tuple[Callable[[np.ndarray], Any], str], agg)[0])):
                self._aggregation[rename] = agg
            else:
                raise ValueError(f'{rename}: {agg} is not a valid aggregation argument')

        for rename, agg in self.iter_fn.items():
            if not callable(agg):
                raise ValueError(f'{rename}: {agg} is not a valid aggregation argument')

    def result_factory(self, fn: Callable[[str, str], np.ndarray]) -> Dict[str, np.ndarray]:
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

    def iter_fn_factory(self, fn: Callable[[], np.ndarray]) -> Dict[str, np.ndarray]:
        return {rename: fn() for rename in self.iter_fn.keys()}

    def apply(self, tafra: 'Tafra') -> 'Tafra':
        raise NotImplementedError


class GroupBy(AggMethod):
    """
    Aggregation by a set of unique values.

    Analogy to SQL ``GROUP BY``, not :meth:`pandas.DataFrame.groupby()`.

    Parameters
    ----------
        columns: Iterable[str]
            The column names to group by.

        aggregation: Mapping[str, Union[Callable[[np.ndarray], Any], \
            Optional. Tuple[Callable[[np.ndarray], Any], str]]]
            A mapping for columns and aggregation functions. Should be
            given as {'column': fn} or {'new_column': (fn, 'column')}.

        iter_fn: Mapping[str, Callable[[np.ndarray], Any]]
            Optional. A mapping for new columns names to the function to apply to
            the enumeration. Should be given as {'new_column': fn}.
    """

    def apply(self, tafra: 'Tafra') -> 'Tafra':
        """
        Apply the :class:`GroupBy` to the :class:`Tafra`.

        Parameters
        ----------
            tafra: Tafra
                The tafra to apply the operation to.

        Returns
        -------
            tafra: Tafra
                The aggregated :class:`Tafra`.
        """
        self._validate(tafra, (
            *self.group_by_cols,
            *(col for (_, col) in self._aggregation.values())
        ))
        unique = self._unique_groups(tafra, self.group_by_cols)
        result = self.result_factory(
            lambda rename, col: np.empty(len(unique), dtype=tafra._data[col].dtype))
        iter_fn = self.iter_fn_factory(lambda: np.ones(len(unique), dtype=int))
        ones = np.ones(tafra._rows, dtype=int)

        for i, u in enumerate(unique):
            which_rows = np.full(tafra._rows, True)

            for val, col in zip(u, self.group_by_cols):
                which_rows &= tafra._data[col] == val
                result[col][i] = val

            for rename, (fn, col) in self._aggregation.items():
                result[rename][i] = fn(tafra._data[col][which_rows])

            for rename, fn in self.iter_fn.items():
                iter_fn[rename][i] = fn(i * ones[which_rows])

        result.update(iter_fn)
        return Tafra(result)


class Transform(AggMethod):
    """
    Apply a function to each unique set of values and join to the original table.

    Analogy to :meth:`pandas.DataFrame.groupby().transform()`,
    i.e. a SQL ``GROUP BY`` and ``LEFT JOIN`` back to the original table.

    Parameters
    ----------
        group_by: Iterable[str]
            The column names to group by.

        aggregation: Mapping[str, Union[Callable[[np.ndarray], Any], \
        Tuple[Callable[[np.ndarray], Any], str]]]
            Optional. A mapping for columns and aggregation functions. Should be
            given as {'column': fn} or {'new_column': (fn, 'column')}.

        iter_fn: Mapping[str, Callable[[np.ndarray], Any]]
            Optional. A mapping for new columns names to the function to apply to
            the enumeration. Should be given as {'new_column': fn}.
    """

    def apply(self, tafra: 'Tafra') -> 'Tafra':
        """
        Apply the :class:`Transform` to the :class:`Tafra`.

        Parameters
        ----------
            tafra: Tafra
                The tafra to apply the operation to.

        Returns
        -------
            tafra: Tafra
                The transformed :class:`Tafra`.
        """
        self._validate(tafra, (
            *self.group_by_cols,
            *(col for (_, col) in self._aggregation.values())
        ))
        unique = self._unique_groups(tafra, self.group_by_cols)
        result = self.result_factory(
            lambda rename, col: np.empty_like(tafra._data[col]))
        iter_fn = self.iter_fn_factory(lambda: np.ones(tafra._rows, dtype=int))
        ones = np.ones(tafra._rows, dtype=int)

        for i, u in enumerate(unique):
            which_rows = np.full(tafra._rows, True)

            for val, col in zip(u, self.group_by_cols):
                which_rows &= tafra._data[col] == val
                result[col][which_rows] = tafra._data[col][which_rows]

            for rename, agg in self._aggregation.items():
                fn, col = agg
                result[rename][which_rows] = fn(tafra._data[col][which_rows])

            for rename, fn in self.iter_fn.items():
                iter_fn[rename][which_rows] = fn(i * ones[which_rows])

        result.update(iter_fn)
        return Tafra(result)


@dc.dataclass
class IterateBy(GroupSet):
    """
    A generator that yields a :class:`Tafra` for each set of unique values.

    Analogy to `pandas.DataFrame.groupby()`, i.e. an Sequence of `Tafra` objects.
    Yields tuples of ((unique grouping values, ...), row indices array, subset tafra)

    Parameters
    ----------
        group_by: Iterable[str]
            The column names to group by.
    """
    group_by_cols: Iterable[str]

    def apply(self, tafra: 'Tafra') -> Iterator[GroupDescription]:
        """
        Apply the :class:`IterateBy` to the :class:`Tafra`.

        Parameters
        ----------
            tafra: Tafra
                The tafra to apply the operation to.

        Returns
        -------
            tafras: Iterator[GroupDescription]
                An iterator over the grouped :class:`Tafra`.
        """
        self._validate(tafra, self.group_by_cols)
        unique = self._unique_groups(tafra, self.group_by_cols)

        for u in unique:
            which_rows = np.full(tafra._rows, True)
            for val, col in zip(u, self.group_by_cols):
                which_rows &= tafra._data[col] == val

            if len(u) == 1:
                u = u[0]

            yield (u, which_rows, tafra._ndindex(which_rows))


@dc.dataclass
class Join(GroupSet):
    """
    Base class for SQL-like JOINs.
    """
    on: Iterable[Tuple[str, str, str]]
    select: Iterable[str]

    def _validate_dtypes(self, l_table: 'Tafra', r_table: 'Tafra') -> None:
        for l_column, r_column, _ in self.on:
            l_value = l_table._data[l_column]
            r_value = r_table._data[r_column]
            l_dtype = l_table._dtypes[l_column]
            r_dtype = r_table._dtypes[r_column]

            if l_value.dtype != r_value.dtype:
                raise TypeError(
                    f'This `Tafra` column `{l_column}` dtype `{l_value.dtype}` '
                    f'does not match other `Tafra` dtype `{r_value.dtype}`.')

            # should not happen unless dtypes manually changed, but let's check it
            elif l_dtype != r_dtype:
                raise TypeError(
                    f'This `Tafra` column `{l_column}` dtype `{l_dtype}` '
                    f'does not match other `Tafra` dtype `{r_dtype}`.')

    @staticmethod
    def _validate_ops(ops: Iterable[str]) -> None:
        for op in ops:
            _op = JOIN_OPS.get(op, None)
            if _op is None:
                raise TypeError(f'The operator {op} is not valid.')

    def apply(self, left_t: 'Tafra', right_t: 'Tafra') -> 'Tafra':
        raise NotImplementedError


class InnerJoin(Join):
    """
    An inner join.

    Analogy to SQL INNER JOIN, or `pandas.merge(..., how='inner')`,

    Parameters
    ----------
        right: Tafra
            The right-side :class:`Tafra` to join.

        on: Iterable[Tuple[str, str, str]]
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
            are returned. If the column exists in both :class`Tafra`,
            prefers the left over the right.
    """

    def apply(self, left_t: 'Tafra', right_t: 'Tafra') -> 'Tafra':
        """
        Apply the :class:`InnerJoin` to the :class:`Tafra`.

        Parameters
        ----------
            left_t: Tafra
                The left tafra to join.

            right_t: Tafra
                The right tafra to join.

        Returns
        -------
            tafra: Tafra
                The joined :class:`Tafra`.
        """
        left_cols, right_cols, ops = list(zip(*self.on))
        self._validate(left_t, left_cols)
        self._validate(right_t, right_cols)
        self._validate_dtypes(left_t, right_t)
        self._validate_ops(ops)

        _on = tuple((left_col, right_col, JOIN_OPS[op]) for left_col, right_col, op in self.on)

        join: Dict[str, List[Any]] = {column: list() for column in chain(
            left_t._data.keys(),
            right_t._data.keys()
        ) if not self.select
            or (self.select and column in self.select)}

        # right-to-left so left dtypes overwrite
        dtypes: Dict[str, str] = {column: dtype for column, dtype in chain(
            right_t._dtypes.items(),
            left_t._dtypes.items()
        ) if column in join.keys()}

        for i in range(left_t._rows):
            right_rows = np.full(right_t._rows, True)

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
                    join[column].extend(right_t[column][right_rows])

        return Tafra(
            {column: np.array(value)
             for column, value in join.items()},
            dtypes
        )


class LeftJoin(Join):
    """
    A left join.

    Analogy to SQL LEFT JOIN, or `pandas.merge(..., how='left')`,

    Parameters
    ----------
        right: Tafra
            The right-side :class:`Tafra` to join.

        on: Iterable[Tuple[str, str, str]]
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
            are returned. If the column exists in both :class`Tafra`,
            prefers the left over the right.
    """

    def apply(self, left_t: 'Tafra', right_t: 'Tafra') -> 'Tafra':
        """
        Apply the :class:`LeftJoin` to the :class:`Tafra`.

        Parameters
        ----------
            left_t: Tafra
                The left tafra to join.

            right_t: Tafra
                The right tafra to join.

        Returns
        -------
            tafra: Tafra
                The joined :class:`Tafra`.
        """
        left_cols, right_cols, ops = list(zip(*self.on))
        self._validate(left_t, left_cols)
        self._validate(right_t, right_cols)
        self._validate_dtypes(left_t, right_t)
        self._validate_ops(ops)

        _on = tuple((left_col, right_col, JOIN_OPS[op]) for left_col, right_col, op in self.on)

        join: Dict[str, List[Any]] = {column: list() for column in chain(
            left_t._data.keys(),
            right_t._data.keys()
        ) if not self.select
            or (self.select and column in self.select)}

        dtypes: Dict[str, str] = {column: dtype for column, dtype in chain(
            left_t._dtypes.items(),
            right_t._dtypes.items()
        ) if column in join.keys()}

        for i in range(left_t._rows):
            right_rows = np.full(right_t._rows, True)

            for left_col, right_col, op in _on:
                right_rows &= op(left_t[left_col][i], right_t[right_col])

            right_count = np.sum(right_rows)

            for column in join.keys():
                if column in left_t._data:
                    join[column].extend(max(1, right_count) * [left_t[column][i]])

                elif column in right_t._data:
                    if right_count <= 0:
                        join[column].append(None)
                        if dtypes[column] != 'object':
                            dtypes[column] = 'object'
                    else:
                        join[column].extend(right_t[column][right_rows])

        return Tafra(
            {column: np.array(value)
             for column, value in join.items()},
            dtypes
        )


@dc.dataclass
class CrossJoin(Join):
    """
    A cross join.

    Analogy to SQL CROSS JOIN, or `pandas.merge(..., how='outer')
    using temporary columns of static value to intersect all rows`.

    Parameters
    ----------
        right: Tafra
            The right-side :class:`Tafra` to join.

        select: Iterable[str] = []
            The columns to return. If not given, all unique columns names
            are returned. If the column exists in both :class`Tafra`,
            prefers the left over the right.
    """

    def apply(self, left_t: 'Tafra', right_t: 'Tafra') -> 'Tafra':
        """
        Apply the :class:`CrossJoin` to the :class:`Tafra`.

        Parameters
        ----------
            left_t: Tafra
                The left tafra to join.

            right_t: Tafra
                The right tafra to join.

        Returns
        -------
            tafra: Tafra
                The joined :class:`Tafra`.
        """
        self._validate_dtypes(left_t, right_t)

        left_rows = left_t._rows
        right_rows = right_t._rows

        select = set(self.select)
        if len(select) > 0:
            left_cols = list(select.intersection(list(left_t._data.keys())))
            right_cols = list(select.intersection(list(right_t._data.keys())))

            if len(left_cols) == 0:
                raise IndexError('No columns given to select from left `Tafra`.')
            if len(right_cols) == 0:
                raise IndexError('No columns given to select from right `Tafra`.')

        else:
            left_cols = list(left_t._data.keys())
            right_cols = list(right_t._data.keys())

        left_new = Tafra(left_t[left_cols].col_map(np.tile, reps=right_rows))
        right_new = Tafra(right_t[right_cols].col_map(np.tile, reps=left_rows))

        left_new.update_inplace(right_new)

        return left_new


# Import here to resolve circular dependency
from .base import Tafra
