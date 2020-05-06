"""
Tafra: the innards of a dataframe

Author
------
Derrick W. Turk
David S. Fulford

Notes
-----
Created on April 25, 2020
"""

import sys
import operator
import warnings
from collections import OrderedDict
from itertools import chain
import dataclasses as dc

import numpy as np
from pandas import DataFrame  # just for mypy...

from typing import Any, Callable, Dict, List, Iterable, Tuple, Optional, Union
from typing import cast
from typing_extensions import Protocol


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


JOIN_OPS: Dict[str, Callable[[Any, Any], Any]] = {
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge
}


TAFRA_TYPE: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'int': lambda x: x.astype(int),
    'float': lambda x: x.astype(float),
    'bool': lambda x: x.astype(bool),
    'str': lambda x: x.astype(str),
    'date': lambda x: x.astype('datetime64'),
    'object': lambda x: x.astype(object),
}

RECORD_TYPE: Dict[str, Callable[[Any], Any]] = {
    'int': lambda x: int(x),
    'float': lambda x: float(x),
    'bool': lambda x: bool(x),
    'str': lambda x: str(x),
    'date': lambda x: x.strftime(r'%Y-%m-%d'),
    'object': lambda x: str(x),
}


def _real_has_attribute(obj: object, attr: str) -> bool:
    try:
        obj.__getattribute__(attr)
        return True
    except AttributeError:
        return False


@dc.dataclass
class Tafra:
    """The innards of a dataframe.
    """
    _data: Dict[str, np.ndarray]
    _dtypes: Dict[str, str] = dc.field(default_factory=dict)

    def __post_init__(self):
        rows = None
        for column, values in self._data.items():
            if rows is None:
                rows = len(values)
            else:
                if rows != len(values):
                    raise ValueError('`Tafra` must have consistent row counts.')

        if self._dtypes:
            self.update_dtypes()
        else:
            self._dtypes = {}
        self.coalesce_types()

    def __getitem__(self, item: Union[str, int, slice, np.ndarray]):
        # type is actually Union[np.ndarray, 'Tafra'] but mypy goes insane
        if isinstance(item, str):
            return self._data[item]

        elif isinstance(item, int):
            return self._slice(slice(item, item + 1))

        elif isinstance(item, slice):
            return self._slice(item)

        elif isinstance(item, np.ndarray):
            return self._index(item)

        else:
            raise ValueError(f'Type {type(item)} not supported.')

    def __getattr__(self, attr: str) -> np.ndarray:
        return self._data[attr]

    def __setitem__(self, item: str, value: Union[np.ndarray, Iterable, Any]):
        value = self._validate_value(value)
        # create the dict entry as a np.ndarray if it doesn't exist
        self._data.setdefault(item, np.empty(self.rows, dtype=value.dtype))
        self._data[item] = value
        self._dtypes[item] = self._format_type(value.dtype)

    def __setattr__(self, attr: str, value: Union[np.ndarray, Iterable]):
        if not (_real_has_attribute(self, '_init') and self._init):
            object.__setattr__(self, attr, value)
            return

        value = self._validate_value(value)
        self._data[attr] = value
        self._dtypes[attr] = self._format_type(value.dtype)

    def __len__(self):
        return self.rows

    def _validate_value(self, value: Union[np.ndarray, Iterable, Any]) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            if not isinstance(value, Iterable) or isinstance(value, str):
                value = np.asarray([value])
            else:
                value = np.asarray(value)

        # is it an ndarray now?
        if not isinstance(value, np.ndarray):
            raise ValueError('`Tafra` only supports assigning `ndarray`.')

        if value.ndim > 1:
            sq_value = value.squeeze()
            if sq_value.ndim > 1:
                raise ValueError('`ndarray` or `np.squeeze(ndarray)` must have ndim == 1.')
            elif sq_value.ndim == 1:
                # if value was a single item, squeeze returns zero length item
                warnings.warn('`np.squeeze(ndarray)` applied to set ndim == 1.')
                warnings.resetwarnings()
                value = sq_value
            else:
                assert 0, 'ndim <= 0, unreachable'

        if len(value) != self.rows:
            raise ValueError(
                '`Tafra` must have consistent row counts.\n'
                f'This `Tafra` has {self.rows} rows. Assigned np.ndarray has {len(value)} rows.')

        return value

    def _validate_columns(self, columns: Iterable[str]):
        for column in columns:
            if column not in self._data.keys():
                raise ValueError(f'Column {column} does not exist in `tafra`.')

    def _validate_dtypes(self, dtypes: Dict[str, Any]) -> Dict[str, str]:
        msg = ''
        _dtypes = {}

        self._validate_columns(dtypes.keys())

        for column, _dtype in dtypes.items():
            _dtypes[column] = self._format_type(_dtype)
            if _dtypes[column] not in TAFRA_TYPE:
                msg += f'`{_dtypes[column]}` is not a valid dtype for `{column}.`\n'

        if len(msg) > 0:
            # should be KeyError value Python 3.7.x has a bug with '\n'
            raise ValueError(msg)

        return _dtypes

    @staticmethod
    def _format_type(t: Any) -> str:
        _t = str(t)
        if 'int' in _t: _type = 'int'
        elif 'float' in _t: _type = 'float'
        elif 'bool' in _t: _type = 'bool'
        elif 'str' in _t: _type = 'str'
        elif '<U' in _t: _type = 'str'
        elif 'date' in _t: _type = 'date'
        elif '<M' in _t: _type = 'date'
        elif 'object' in _t: _type = 'object'
        elif 'O' in _t: _type = 'object'
        else: return _t
        return _type

    @staticmethod
    def _apply_type(t: str, array: np.ndarray) -> np.ndarray:
        return TAFRA_TYPE[t](array)

    @classmethod
    def from_dataframe(cls, df: DataFrame, dtypes: Optional[Dict[str, str]] = None) -> 'Tafra':
        if dtypes is None:
            dtypes = {c: t for c, t in zip(df.columns, df.dtypes)}
        dtypes = {c: cls._format_type(t) for c, t in dtypes.items()}

        return cls(
            {c: cls._apply_type(dtypes[c], df[c].values) for c in df.columns},
            {c: dtypes[c] for c in df.columns}
        )

    @property
    def columns(self) -> Tuple[str, ...]:
        """Get the names of the columns.
        Equivalent to `Tafra`.keys().
        """
        return tuple(self._data.keys())

    @property
    def rows(self) -> int:
        """Get the rows of the first item in the data `dict`.
        The `len()` of all values have been previously validated.
        """
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    @property
    def data(self) -> Dict[str, np.ndarray]:
        """Return the data `dict` attribute.
        """
        return self._data

    @property
    def dtypes(self) -> Dict[str, str]:
        """Return the dtypes `dict`.
        """
        return self._dtypes

    def select(self, columns: Iterable[str]) -> 'Tafra':
        """Use column name iterable to slice `tafra` columns analogous to SQL SELECT.
        """
        self._validate_columns(columns)

        return Tafra(
            {column: value
             for column, value in self._data.items() if column in columns},
            {column: value
             for column, value in self._dtypes.items() if column in columns}
        )

    def _slice(self, _slice: slice) -> 'Tafra':
        """Use slice object to slice np.ndarray.
        """
        return Tafra(
            {column: value[_slice]
                for column, value in self._data.items()},
            {column: value
                for column, value in self._dtypes.items()}
        )

    def _index(self, index: np.ndarray) -> 'Tafra':
        """Use numpy indexing to slice np.ndarray.
        """
        if index.ndim != 1:
            raise ValueError(f'Indexing np.ndarray must ndim == 1, got ndim == {index.ndim}')
        return Tafra(
            {column: value[index]
                for column, value in self._data.items()},
            {column: value
                for column, value in self._dtypes.items()}
        )

    def keys(self):
        """Return the keys of the data attribute, i.e. like a `dict.keys()`.
        """
        return self._data.keys()

    def values(self):
        """Return the values of the data attribute, i.e. like a `dict.values()`.
        """
        return self._data.values()

    def items(self):
        """Return the items of the data attribute, i.e. like a `dict.items()`.
        """
        return self._data.items()

    def update(self, other: 'Tafra'):
        """Update the data and dtypes of this `Tafra` with another `Tafra`.
        Length of rows must match, while data of different `dtype` will overwrite.
        """
        rows = self.rows
        for column, values in other._data.items():
            if len(values) != rows:
                raise ValueError(
                    'Other `Tafra` must have consistent row count. '
                    f'This `Tafra` has {rows} rows, other `Tafra` has {len(values)} rows.')
            self._data[column] = values

        self.update_dtypes(other._dtypes)

    def update_dtypes(self, dtypes: Optional[Dict[str, Any]] = None) -> None:
        """Apply new dtypes or update dtype `dict` for missing keys.
        """

        if dtypes is not None:
            self._validate_columns(dtypes.keys())
            dtypes = self._validate_dtypes(dtypes)
            self._dtypes.update(dtypes)

        for column in self._dtypes.keys():
            if self._format_type(self._data[column].dtype) != self._dtypes[column]:
                self._data[column] = self._apply_type(self._dtypes[column], self._data[column])

    def coalesce_types(self) -> None:
        for column in self._data.keys():
            if column not in self._dtypes:
                self._dtypes[column] = self._format_type(self._data[column].dtype)

    def delete(self, column: str):
        """Remove a column from the `Tafra` data and dtypes.
        """
        _ = self._data.pop(column, None)
        _ = self._dtypes.pop(column, None)

    def copy(self, order: str = 'C') -> 'Tafra':
        """Helper function to create a copy of a `Tafra`s data.
        """
        return Tafra(
            {column: value.copy(order=order)
                for column, value in self._data.items()},
            {column: value
                for column, value in self._dtypes.items()}
        )

    def coalesce(self, column: str, fills: Iterable[Any]) -> np.ndarray:
        #TODO: handle dtype?
        value = self._data[column].copy()
        for fill in fills:
            f = np.atleast_1d(fill)
            where_na = np.full(self.rows, False)
            try:
                where_na |= np.isnan(value)
            except:
                pass

            try:
                where_na |= value == None
            except:
                pass

            for w in where_na:
                if len(f) == 1:
                    value[where_na] = f
                else:
                    value[where_na] = f[where_na]

        return value

    def union(self, other: 'Tafra', inplace: bool = False) -> Union['Tafra', None]:
        """Analogy to SQL UNION or `pandas.append`. All column names and dtypes must match.
        """
        # These should be unreachable unless attributes were directly modified
        if len(self._data) != len(self._dtypes):
            assert 0, 'This `Tafra` Length of data and dtypes do not match'
        if len(other._data) != len(other._dtypes):
            assert 0, 'Other `Tafra` Length of data and dtypes do not match'

        # ensure same number of columns
        if len(self._data) != len(other._data) or len(self._dtypes) != len(other._dtypes):
            raise ValueError(
                f'This `Tafra` column count does not match other `Tafra` column count.')

        # ensure all columns in this `Tafra` exist in other `Tafra`
        # if len() is same AND all columns in this exist in other,
        # do not need to check other `Tafra` columns in this `Tafra`.
        for (data_column, value), (dtype_column, dtype) \
                in zip(self._data.items(), self._dtypes.items()):

            if data_column not in other._data or dtype_column not in other._dtypes:
                raise ValueError(
                    f'This `Tafra` column `{data_column}` does not exist in other `Tafra`.')

            elif value.dtype != other._data[data_column].dtype:
                raise ValueError(
                    f'This `Tafra` column `{data_column}` dtype `{value.dtype}` '
                    f'does not match other `Tafra` dtype `{other._data[data_column].dtype}`.')

            elif dtype != other._dtypes[dtype_column]:
                raise ValueError(
                    f'This `Tafra` column `{data_column}` dtype `{dtype}` '
                    f'does not match other `Tafra` dtype `{other._dtypes[dtype_column]}`.')

        if inplace:
            for column, value in self._data.items():
                self._data[column] = np.append(value, other._data[column])
            return None

        # np.append is not done inplace
        return Tafra(
            {column: np.append(value, other._data[column]) for column, value in self._data.items()},
            self._dtypes
        )

    @staticmethod
    def _cast_records(dtype: str, data: np.ndarray, cast_null: bool) -> Any:
        """Cast np.nan to None. Requires changing `dtype` to `object`.
        """
        value: Any = RECORD_TYPE[dtype](data.item())
        if cast_null and dtype == 'float' and np.isnan(data.item()):
            return None
        return value

    def to_records(self, columns: Optional[Iterable[str]] = None,
                   cast_null: bool = True) -> Iterable[Tuple[Any, ...]]:
        """Return a generator of tuples, each tuple being a record (i.e. row)
        and allowing heterogeneous typing.
        Useful for e.g. sending records back to a database.
        """
        if columns is None:
            columns = self.columns
        else:
            self._validate_columns(columns)

        for row in range(self.rows):
            yield tuple(self._cast_records(
                self._dtypes[c], self._data[c][[row]],
                cast_null
            ) for c in columns)
        return

    def to_list(self, columns: Optional[Iterable[str]] = None) -> List[np.ndarray]:
        """Return a list of homogeneously typed columns (as np.ndarrays) in the tafra.
        If a generator is needed, use `Tafra.values()`.
        """
        if columns is None:
            return list(self._data.values())
        return list(self._data[c] for c in columns)

    def group_by(self, group_by: Iterable[str],
                 aggregation: InitAggregation = {},
                 iter_fn: Dict[str, Callable[[np.ndarray], Any]] = dict()) -> 'Tafra':
        """Helper function to implement the `GroupBy` class.
        """
        return GroupBy(group_by, aggregation, iter_fn).apply(self)

    def transform(self, group_by: Iterable[str],
                  aggregation: InitAggregation = {},
                  iter_fn: Dict[str, Callable[[np.ndarray], Any]] = dict()) -> 'Tafra':
        """Helper function to implement the `Transform` class.
        """
        return Transform(group_by, aggregation, iter_fn).apply(self)

    def iterate_by(self, group_by: Iterable[str]) -> Iterable[GroupDescription]:
        """Helper function to implement the `IterateBy` class.
        """
        yield from IterateBy(group_by).apply(self)

    def inner_join(self, right: 'Tafra', on: Iterable[Tuple[str, str, str]],
                   select: Iterable[str] = list()) -> 'Tafra':
        """Helper function to implement the `InnerJoin` class.
        """
        return InnerJoin(on, select).apply(self, right)

    def left_join(self, right: 'Tafra', on: Iterable[Tuple[str, str, str]],
                  select: Iterable[str] = list()) -> 'Tafra':
        """Helper function to implement the `LeftJoin` class.
        """
        return LeftJoin(on, select).apply(self, right)

    def cross_join(self, right: 'Tafra',
                  select: Iterable[str] = list()) -> 'Tafra':
        """Helper function to implement the `CrossJoin` class.
        """
        return CrossJoin([], select).apply(self, right)


@dc.dataclass
class GroupSet():
    """A `GroupSet` is the set of columns by which we construct our groups.
    """

    @staticmethod
    def _unique_groups(tafra: Tafra, columns: Iterable[str]) -> List[Any]:
        """Construct a unique set of grouped values.
        Uses `OrderedDict` rather than `set` to maintain order.
        """
        return list(OrderedDict.fromkeys(zip(*(tafra[col] for col in columns))))

    @staticmethod
    def _validate(tafra: Tafra, columns: Iterable[str]) -> None:
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

    def apply(self, tafra: Tafra):
        ...


class GroupBy(AggMethod):
    """Analogy to SQL `GROUP BY`, not `pandas.DataFrame.groupby()`. A `reduce` operation.
    """

    def apply(self, tafra: Tafra) -> Tafra:
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

    def apply(self, tafra: Tafra) -> Tafra:
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

    def apply(self, tafra: Tafra) -> Iterable[GroupDescription]:
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
    def _validate_dtypes(left_t: Tafra, right_t: Tafra):
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

    def apply(self, left_t: Tafra, right_t: Tafra) -> Tafra:
        ...


class InnerJoin(Join):
    """Analogy to SQL INNER JOIN, or `pandas.merge(..., how='inner')`,
    """

    def apply(self, left_t: Tafra, right_t: Tafra) -> Tafra:
        left_cols, right_cols, ops = list(zip(*self._on))
        self._validate(left_t, left_cols)
        self._validate(right_t, right_cols)
        self._validate_dtypes(left_t, right_t)
        self._validate_ops(ops)

        _on = tuple((left_col, right_col, JOIN_OPS[op]) for left_col, right_col, op in self._on)
        left_unique = self._unique_groups(left_t, left_cols)
        right_unique = self._unique_groups(right_t, right_cols)

        join: Dict[str, List] = {column: list() for column in chain(
            left_t._data.keys(),
            right_t._data.keys()
        ) if (not self._select)
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

    def apply(self, left_t: Tafra, right_t: Tafra) -> Tafra:
        left_cols, right_cols, ops = list(zip(*self._on))
        self._validate(left_t, left_cols)
        self._validate(right_t, right_cols)
        self._validate_dtypes(left_t, right_t)
        self._validate_ops(ops)

        _on = tuple((left_col, right_col, JOIN_OPS[op]) for left_col, right_col, op in self._on)
        left_unique = self._unique_groups(left_t, left_cols)
        right_unique = self._unique_groups(right_t, right_cols)

        join: Dict[str, List] = {column: list() for column in chain(
            left_t._data.keys(),
            right_t._data.keys()
        ) if (not self._select)
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
    """Analogy to SQL CROSS APPLY, or `pandas.merge(..., how='outer')
    using temporary columns of static value to intersect all rows`,
    """

    def apply(self, left_t: Tafra, right_t: Tafra) -> Tafra:
        self._validate_dtypes(left_t, right_t)

        join: Dict[str, List] = {column: list() for column in chain(
            left_t._data.keys(),
            right_t._data.keys()
        ) if (not self._select)
            or (self._select and column in self._select)}

        dtypes: Dict[str, str] = {column: dtype for column, dtype in chain(
            left_t._dtypes.items(),
            right_t._dtypes.items()
        ) if column in join.keys()}

        left_count = left_t.rows
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


Tafra.copy.__doc__ += '\n\nnumpy doc string:\n' + np.ndarray.copy.__doc__  # type: ignore
Tafra.group_by.__doc__ += GroupBy.__doc__  # type: ignore
Tafra.transform.__doc__ += Transform.__doc__  # type: ignore
Tafra.iterate_by.__doc__ += IterateBy.__doc__  # type: ignore
Tafra.inner_join.__doc__ += InnerJoin.__doc__  # type: ignore
Tafra.left_join.__doc__ += LeftJoin.__doc__  # type: ignore
Tafra.cross_join.__doc__ += CrossJoin.__doc__  # type: ignore


if __name__ == '__main__':
    t = Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })

    print('List:\t\t', t.to_list())
    print('Record:\t\t', list(t.to_records()))

    gb = t.group_by(
        ['y', 'z'], {'x': sum}
    )

    print('Group By:\t', gb)

    # transform example

    print('Iterate by y, z:')
    for grp in gb.iterate_by(('y', 'z')):
        print(grp)
