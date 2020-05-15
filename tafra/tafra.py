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
import warnings
from itertools import chain
import dataclasses as dc

import numpy as np
from .pandas import DataFrame  # just for mypy...

from typing import (Any, Callable, Dict, List, Iterable, Tuple, Optional, Union,
                    KeysView, ValuesView, ItemsView)
from typing import cast
from typing_extensions import Protocol


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

        self._rows = rows

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
        self._data.setdefault(item, np.empty(self._rows, dtype=value.dtype))
        self._data[item] = value
        self._dtypes[item] = self._format_type(value.dtype)

    def __setattr__(self, attr: str, value: Union[np.ndarray, Iterable]):
        if not (_real_has_attribute(self, '_init') and self._init):
            object.__setattr__(self, attr, value)
            return

        value = self._validate_value(value)
        self._data[attr] = value
        self._dtypes[attr] = self._format_type(value.dtype)

    def __len__(self) -> int:
        return self._rows

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f'Tafra(data={self._data}, dtypes={self._dtypes}, rows={self._rows})'

    @staticmethod
    def _html_tr(row) -> str:
        return '<tr>\n{td}\n</tr>'.format(td='\n'.join(f'<td>{td}</td>' for td in row))

    @staticmethod
    def _html_tbody(html_tr_iter) -> str:
        return '<tbody>\n{tr}\n</tbody>'.format(tr='\n'.join(html_tr_iter))

    @staticmethod
    def _html_thead(columns: Iterable[str]) -> str:
        return '<thead>\n<tr>\n{th}\n</tr>\n</thead>' \
            .format(th='\n'.join(f'<th>{c}</th>' for c in columns))

    @staticmethod
    def _html_table(html_thead, html_tbody) -> str:
        return f'<table>\n{html_thead}\n{html_tbody}\n</table>'

    def to_html(self, n: int = 40) -> str:
        html_thead = self._html_thead(chain([''], self._data.keys()))
        html_tr = chain(
            [self._html_tr(chain([''], self._dtypes.values()))],
            (self._html_tr(chain([i], (v[i] for v in self._data.values())))
             for i in range(min(n, self._rows)))
        )
        html_tbody = self._html_tbody(html_tr)
        return self._html_table(html_thead, html_tbody)

    def _repr_html_(self) -> str:
        """Pretty print tables in a Jupyter Notebook"""
        return self.to_html()

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

        if len(value) != self._rows:
            raise ValueError(
                '`Tafra` must have consistent row counts.\n'
                f'This `Tafra` has {self._rows} rows. Assigned np.ndarray has {len(value)} rows.')

        return value

    def _validate_columns(self, columns: Iterable[str]) -> None:
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

    @classmethod
    def as_tafra(cls, data: Union['Tafra', DataFrame]) -> Optional['Tafra']:
        """Returns the unmodified argument if already a `Tafra`, else construct
        a `Tafra` from known types of `pd.DataFrame` or `dict`.
        """
        if isinstance(data, Tafra):
            return data

        elif type(data).__name__ == 'DataFrame':
            return cls.from_dataframe(data)

        elif isinstance(data, dict):
            return cls(data)

        raise TypeError(f'Unknown type `{type(data)}` for conversion to `Tafra`')

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
        return self._rows

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

    def head(self, n: int = 5) -> str:
        return self.to_html(n)

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

    def keys(self) -> KeysView[str]:
        """Return the keys of the data attribute, i.e. like a `dict.keys()`.
        """
        return self._data.keys()

    def values(self) -> ValuesView[np.ndarray]:
        """Return the values of the data attribute, i.e. like a `dict.values()`.
        """
        return self._data.values()

    def items(self) -> ItemsView[str, np.ndarray]:
        """Return the items of the data attribute, i.e. like a `dict.items()`.
        """
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Return the get() function of the data attribute, i.e. like a `dict.get()`.
        """
        return self._data.get(key, default)

    def update(self, other: 'Tafra') -> None:
        """Update the data and dtypes of this `Tafra` with another `Tafra`.
        Length of rows must match, while data of different `dtype` will overwrite.
        """
        rows = self._rows
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

    def rename(self, renames: Dict[str, str], inplace: bool = True) -> Optional['Tafra']:
        """Rename columns in the `Tafra` with a `dict` of {current_name: new_name}.
        """
        self._validate_columns(renames.keys())

        if inplace:
            for cur, new in renames.items():
                self._data[new] = self._data.pop(cur)
            return None

        return Tafra(
            {renames[column]: value.copy()
                for column, value in self._data.items()},
            {renames[column]: value
                for column, value in self._dtypes.items()}
        )

    def delete(self, column: str, inplace: bool = True) -> Optional['Tafra']:
        """Remove a column from the `Tafra` data and dtypes.
        """
        self._validate_columns(column)

        if inplace:
            _ = self._data.pop(column, None)
            _ = self._dtypes.pop(column, None)
            return None

        return Tafra(
            {col: value.copy()
                for col, value in self._data.items() if col not in column},
            {col: value
                for col, value in self._dtypes.items() if col not in column}
        )

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
            where_na = np.full(self._rows, False)
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

    def union(self, other: 'Tafra', inplace: bool = False) -> Optional['Tafra']:
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
    def _cast_records(dtype: str, data: np.ndarray, cast_null: bool) -> Optional[float]:
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

        for row in range(self._rows):
            yield tuple(self._cast_records(
                self._dtypes[c], self._data[c][[row]],
                cast_null
            ) for c in columns)
        return

    def to_list(self, columns: Optional[Iterable[str]] = None,
                inner: bool = False) -> Union[List[np.ndarray], List[List[Any]]]:
        """Return a list of homogeneously typed columns (as np.ndarrays) in the tafra.
        If a generator is needed, use `Tafra.values()`. If `inner == True` each column
        will be cast to a list.
        """
        if columns is None:
            if inner:
                return [list(v) for v in self._data.values()]
            return list(self._data.values())

        else:
            if inner:
                return [list(self._data[c]) for c in columns]
            return list(self._data[c] for c in columns)

    def group_by(self, group_by: Iterable[str],
                 aggregation: 'InitAggregation' = {},
                 iter_fn: Dict[str, Callable[[np.ndarray], Any]] = dict()) -> 'Tafra':
        """Helper function to implement the `GroupBy` class.
        """
        return GroupBy(group_by, aggregation, iter_fn).apply(self)

    def transform(self, group_by: Iterable[str],
                  aggregation: 'InitAggregation' = {},
                  iter_fn: Dict[str, Callable[[np.ndarray], Any]] = dict()) -> 'Tafra':
        """Helper function to implement the `Transform` class.
        """
        return Transform(group_by, aggregation, iter_fn).apply(self)

    def iterate_by(self, group_by: Iterable[str]) -> Iterable['GroupDescription']:
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


# Import here to resolve circular dependency
from .groups import (GroupBy, Transform, IterateBy, InnerJoin, LeftJoin, CrossJoin,
                     InitAggregation, GroupDescription)

Tafra.copy.__doc__ += '\n\nnumpy doc string:\n' + np.ndarray.copy.__doc__  # type: ignore
Tafra.group_by.__doc__ += GroupBy.__doc__  # type: ignore
Tafra.transform.__doc__ += Transform.__doc__  # type: ignore
Tafra.iterate_by.__doc__ += IterateBy.__doc__  # type: ignore
Tafra.inner_join.__doc__ += InnerJoin.__doc__  # type: ignore
Tafra.left_join.__doc__ += LeftJoin.__doc__  # type: ignore
Tafra.cross_join.__doc__ += CrossJoin.__doc__  # type: ignore
