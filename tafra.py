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

import warnings
from collections import OrderedDict
import dataclasses as dc

import numpy as np

from typing import Any, Callable, Dict, List, Tuple, Optional, Iterable, Union, cast, NoReturn

InitAggregation = Dict[
    str,
    Union[
        Callable[[np.ndarray], Any],
        Tuple[Callable[[np.ndarray], Any], str]
    ]
]


TAFRA_TYPE = {
    'int': lambda x: x.astype(int),
    'float': lambda x: x.astype(float),
    'bool': lambda x: x.astype(bool),
    'str': lambda x: x.astype(str),
    'date': lambda x: x.astype('datetime64'),
    'object': lambda x: x.astype(object),
}


def _real_has_attribute(obj: object, attr: str) -> bool:
    try:
        obj.__getattribute__(attr)
        return True
    except AttributeError:
        return False


class DataFrame():
    """A fake class to satisfy typing of a `pandas.DataFrame` without a dependency.
    """
    _data: Dict[str, np.ndarray]
    columns: List[str]
    dtypes: List[str]

    def __getitem__(self, column: str) -> np.ndarray:
        return self._data[column]

    def __setitem__(self, column: str, value: np.ndarray):
        self._data[column] = value


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
            self.update_types()
        else:
            self._dtypes = {}
        self.coalesce_types()

    def update_types(self, dtypes: Optional[Dict[str, Any]] = None) -> None:
        """Apply new dtypes or update dtype `dict` for missing keys.
        """
        if dtypes is not None:
            dtypes = self.__validate_types(dtypes)
            self._dtypes.update(dtypes)

        for column in self._dtypes.keys():
            if self.__format_type(self._data[column].dtype) != self._dtypes[column]:
                self._data[column] = self.__apply_type(self._dtypes[column], self._data[column])

    def coalesce_types(self) -> None:
        for column in self._data.keys():
            if column not in self._dtypes:
                self._dtypes[column] = self.__format_type(self._data[column].dtype)

    def __getitem__(self, item: Union[str, slice, np.ndarray]):
        # type is actually Union[np.ndarray, 'Tafra'] but mypy goes insane
        if isinstance(item, str):
            return self._data[item]

        elif isinstance(item, slice):
            return self._slice(item)

        elif isinstance(item, np.ndarray):
            return self._index(item)

        else:
            raise ValueError(f'Type {type(item)} not supported.')

    def __getattr__(self, attr: str) -> np.ndarray:
        return self._data[attr]

    def __setitem__(self, item: str, value: Union[np.ndarray, Iterable]):
        value = self.__validate(value)
        self._data[item] = value
        self._dtypes[item] = self.__format_type(value.dtype)

    def __setattr__(self, attr: str, value: Union[np.ndarray, Iterable]):
        if not (_real_has_attribute(self, '_init') and self._init):
            object.__setattr__(self, attr, value)
            return

        value = self.__validate(value)
        self._data[attr] = value
        self._dtypes[attr] = self.__format_type(value.dtype)

    def __validate(self, value: Union[np.ndarray, Iterable]) -> np.ndarray:
        if not isinstance(value, np.ndarray):
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

    def __validate_types(self, dtypes: Dict[str, Any]) -> Dict[str, str]:
        msg = ''
        _dtypes = {}
        for column, _dtype in dtypes.items():
            _dtypes[column] = self.__format_type(_dtype)
            if _dtypes[column] not in TAFRA_TYPE:
                msg += f'`{_dtypes[column]}` is not a valid dtype for `{column}.`\n'

        if len(msg) > 0:
            # should be KeyError value Python 3.7.x has a bug with '\n'
            raise ValueError(msg)

        return _dtypes

    @staticmethod
    def __format_type(t: Any) -> str:
        _t = str(t)
        if 'int' in _t: _type = 'int'
        elif 'float' in _t: _type = 'float'
        elif 'bool' in _t: _type = 'bool'
        elif 'str' in _t: _type = 'str'
        elif '<U' in _t: _type = 'str'
        elif 'date' in _t: _type = 'datetime64'
        elif '<M' in _t: _type = 'datetime64'
        elif 'object' in _t: _type = 'object'
        elif 'O' in _t: _type = 'object'
        else: return _t
        return _type

    @staticmethod
    def __apply_type(t: str, array: np.ndarray) -> np.ndarray:
        return TAFRA_TYPE[t](array)

    @classmethod
    def from_dataframe(cls, df: DataFrame, dtypes: Optional[Dict[str, str]] = None) -> 'Tafra':
        if dtypes is None:
            dtypes = {c: t for c, t in zip(df.columns, df.dtypes)}
        dtypes = {c: cls.__format_type(t) for c, t in dtypes.items()}

        return cls(
            {c: cls.__apply_type(dtypes[c], df[c].values) for c in df.columns},
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

        self.update_types(other._dtypes)

    def copy(self, order: str = 'C') -> 'Tafra':
        """Helper function to create a copy of a `Tafra`s data.
        """
        return Tafra(
            {column: value.copy(order=order)
                for column, value in self._data.items()},
            {column: value
                for column, value in self._dtypes.items()}
        )

    @staticmethod
    def cast_null(dtype: str, data: np.ndarray) -> np.ndarray:
        """Cast np.nan to None. Requires changing `dtype` to `object`.
        """
        if dtype != 'float':
            return data

        where_nan = np.isnan(data)
        if not np.any(where_nan):
            return data

        data = data.copy()
        data = data.astype(object)
        data[where_nan] = None

        return data

    def to_records(self, columns: Optional[Iterable[str]] = None,
                   cast_null: bool = True) -> Iterable[Tuple[Any, ...]]:
        """Return a generator of tuples, each tuple being a record (i.e. row)
        and allowing heterogeneous typing.
        Useful for e.g. sending records back to a database.
        """
        _columns: Iterable[str] = self.columns if columns is None else columns

        if cast_null:
            for row in range(self.rows):
                yield tuple(zip((self.cast_null(
                    self._dtypes[c], self._data[c][[row]])[0] for c in _columns)))
        else:
            for row in range(self.rows):
                yield tuple(zip((self._data[c][row] for c in _columns)))

        return

    def to_list(self, columns: Optional[Iterable[str]] = None) -> List[np.ndarray]:
        """Return a list of homogeneously typed columns (as np.ndarrays) in the tafra.
        If a generator is needed, use `Tafra.values()`.
        """
        if columns is None:
            return list(self._data.values())
        return list(self._data[c] for c in columns)

    def group_by(self, group_by: Iterable[str],
                 aggregation: InitAggregation) -> 'Tafra':
        """Helper function to implement the `GroupBy` class.
        """
        return GroupBy(group_by, aggregation).apply(self)

    def transform(self, group_by: Iterable[str],
                  aggregation: InitAggregation) -> 'Tafra':
        """Helper function to implement the `Transform` class.
        """
        return Transform(group_by, aggregation).apply(self)

    def iterate_by(self, group_by: Iterable[str]) -> Iterable['Tafra']:
        """Helper function to implement the `IterateBy` class.
        """
        return IterateBy(group_by, {}).iter(self)


@dc.dataclass
class AggMethod:
    """Basic methods for aggregations over a data table.
    """
    _group_by_cols: Iterable[str]
    # TODO: specify dtype of result?
    aggregation: dc.InitVar[InitAggregation]
    _aggregation: Dict[str, Tuple[Callable[[np.ndarray], Any], str]] = dc.field(
        default_factory=dict, init=False)

    def __post_init__(self, aggregation: InitAggregation):
        for rename, agg in aggregation.items():

            if callable(agg):
                self._aggregation[rename] = cast(
                    Tuple[Callable[[np.ndarray], Any], str],
                    (agg, rename))
            elif (isinstance(agg, Iterable) and len(agg) == 2
                  and callable(cast(Tuple, agg)[0])):
                self._aggregation[rename] = agg
            else:
                raise ValueError(f'{agg} is not a valid aggregation argument')

    def _validate(self, tafra: Tafra) -> None:
        cols = set(tafra.columns)
        for col in self._group_by_cols:
            if col not in cols:
                raise KeyError(f'{col} does not exist in tafra')
        for rename, agg in self._aggregation.items():
            col = self._aggregation[rename][1]
            if col not in cols:
                raise KeyError(f'{col} does not exist in tafra')
        # we don't have to use all the columns!

    def unique_groups(self, tafra: Tafra) -> List[Any]:
        """Construct a unique set of grouped values.
        Uses `OrderedDict` rather than `set` to maintain order.
        """
        return list(OrderedDict.fromkeys(zip(*(tafra[col] for col in self._group_by_cols))))

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

    def apply(self, tafra: Tafra) -> Tafra:
        """Apply the `AggMethod`. Should probably call `unique_groups` to
        obtain the set of grouped values.
        """
        raise NotImplementedError


@dc.dataclass
class GroupBy(AggMethod):
    """Analogy to SQL `GROUP BY`, not `pandas.DataFrame.groupby()`. A `reduce` operation.
    """

    def apply(self, tafra: Tafra) -> Tafra:
        self._validate(tafra)
        unique = self.unique_groups(tafra)
        result = self.result_factory(
            lambda rename, col: np.empty(len(unique), dtype=tafra[col].dtype))

        for i, u in enumerate(unique):
            which_rows = np.full(tafra.rows, True)
            for val, col in zip(u, self._group_by_cols):
                which_rows &= tafra[col] == val
                result[col][i] = val

            for rename, agg in self._aggregation.items():
                fn, col = agg
                result[rename][i] = fn(tafra[col][which_rows])

        return Tafra(result)


@dc.dataclass
class Transform(AggMethod):
    """Analogy to `pandas.DataFrame.transform()`,
    i.e. a SQL `GROUP BY` and `LEFT JOIN` back to the original table.
    """

    def apply(self, tafra: Tafra) -> Tafra:
        self._validate(tafra)
        unique = self.unique_groups(tafra)
        result = self.result_factory(
            lambda rename, col: np.empty_like(tafra[col]))

        for u in unique:
            which_rows = np.full(tafra.rows, True)
            for val, col in zip(u, self._group_by_cols):
                which_rows &= tafra[col] == val
                result[col][which_rows] = tafra[col][which_rows]

            for rename, agg in self._aggregation.items():
                fn, col = agg
                result[rename][which_rows] = fn(tafra[col][which_rows])

        return Tafra(result)

@dc.dataclass
class IterateBy(AggMethod):
    """Analogy to `pandas.DataFrame.groupby()`, i.e. an Iterable of `Tafra` objects.
    """

    def __postinit__(self, *args):
        pass

    def iter(self, tafra: Tafra) -> Iterable[Tafra]:
        self._validate(tafra)
        unique = self.unique_groups(tafra)

        for u in unique:
            which_rows = np.full(tafra.rows, True)
            result = self.result_factory(
                lambda rename, col: np.empty(len(which_rows)))
            for val, col in zip(u, self._group_by_cols):
                which_rows &= tafra[col] == val

            yield tafra[which_rows]


Tafra.copy.__doc__ += '\n\nnumpy doc string:\n' + np.ndarray.copy.__doc__  # type: ignore
Tafra.group_by.__doc__ += GroupBy.__doc__  # type: ignore
Tafra.transform.__doc__ += Transform.__doc__  # type: ignore
Tafra.iterate_by.__doc__ += IterateBy.__doc__  # type: ignore


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
