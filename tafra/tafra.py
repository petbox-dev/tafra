from collections import OrderedDict
import dataclasses as dc

import numpy as np

from typing import Any, Callable, Dict, List, Tuple, Optional, Iterable


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
    """
    A fake class to satisfy typing of a `pandas.DataFrame` without a dependency.
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
    """
    The innards of a dataframe.
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
                    raise ValueError('tafra must have consistent row counts')

        if self._dtypes:
            self.update_types()
        else:
            self._dtypes = {c: self.__format_type(v.dtype) for c, v in self._data.items()}

    def update_types(self, dtypes: Optional[Dict[str, str]] = None):
        """
        Apply new dtypes.
        """
        if dtypes is not None:
            self._dtypes.update(dtypes)

        for column in self._dtypes.keys():
            self._dtypes[column] = self.__format_type(self._dtypes[column])
            self._data[column] = self.__apply_type(
                self._dtypes[column], self._data[column])

    def __getitem__(self, column: str) -> np.ndarray:
        return self._data[column]

    def __getattr__(self, column: str) -> np.ndarray:
        return self._data[column]

    def __setitem__(self, column: str, value: np.ndarray):
        self._data[column] = value

    def __setattr__(self, column: str, value: np.ndarray):
        if not (_real_has_attribute(self, '_init') and self._init):
            object.__setattr__(self, column, value)
            return

        if len(value) != self.rows:
            raise ValueError('tafra must have consistent row counts')

        self._data[column] = value

    @staticmethod
    def __format_type(t: Any) -> str:
        _t = str(t)
        if 'int' in _t: _type = 'int'
        elif 'float' in _t: _type = 'float'
        elif 'bool' in _t: _type = 'bool'
        elif 'str' in _t: _type = 'str'
        elif '<U' in _t: _type = 'str'
        elif 'date' in _t: _type = 'datetime64'
        elif 'object' in _t: _type = 'object'
        elif 'O' in _t: _type = 'object'
        return _type

    @staticmethod
    def __apply_type(t: str, array: np.ndarray) -> np.ndarray:
        return TAFRA_TYPE[t](array)

    @classmethod
    def from_dataframe(cls, df: DataFrame, dtypes: Optional[Dict[str, str]] = None) -> 'Tafra':
        if dtypes is None:
            dtypes = {c: t for c, t in zip(df.columns, df.dtypes)}
        return cls({c: cls.__apply_type(
            cls.__format_type(dtypes[c]), df[c].values) for c in df.columns})

    @property
    def columns(self) -> Tuple[str, ...]:
        return tuple(self._data.keys())

    @property
    def rows(self) -> int:
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    @property
    def data(self) -> Dict[str, np.ndarray]:
        return self._data

    @property
    def dtypes(self) -> Tuple[np.dtype, ...]:
        return tuple(value.dtype for value in self._data.values())

    def cast_nulls(self) -> None:
        """
        Cast np.nan to None. Requires changing `dtype` to `object`.
        """
        for column in (k for k, v in self._dtypes.items() if v == 'float'):
            where_nan = np.isnan(self._data[column])
            if np.sum(where_nan) > 0:
                self.update_types({column: 'object'})
                self._data[column][where_nan] = None

    def to_record(self, columns: Optional[Iterable[str]] = None,
                  cast_null: bool = True) -> Tuple[Tuple[Any, ...], ...]:
        """
        Return a tuple of tuples, each inner tuple being a record (i.e. row)
        and allowing heterogeneous typing.
        Useful for e.g. sending records back to a database.
        """
        cols: Iterable[str] = self.columns if columns is None else columns
        if cast_null:
            self.cast_nulls()

        return tuple(zip(*(self._data[c] for c in cols)))

    def to_list(self, columns: Optional[Iterable[str]] = None) -> List[np.ndarray]:
        """
        Return a list of homogeneously typed columns (as np.ndarrays) in the tafra
        """
        if columns is None:
            return list(self._data[c] for c in self.columns)
        return list(self._data[c] for c in columns)

    def group_by(self, group_by: List[str],
                 aggregation: Dict[str, Callable[[np.ndarray], Any]]) -> 'Tafra':
        """
        Helper function to implement the `GroupBy` class.
        """
        return GroupBy(group_by, aggregation).apply(self)

    def transform(self, group_by: List[str],
                  aggregation: Dict[str, Callable[[np.ndarray], Any]]) -> 'Tafra':
        """
        Helper function to implement the `Transform` class.
        """
        return Transform(group_by, aggregation).apply(self)


@dc.dataclass
class AggMethod:
    """
    Basic methods for aggregations over a data table.
    """
    _group_by_cols: List[str]
    # TODO: specify dtype of result?
    _aggregation: Dict[str, Callable[[np.ndarray], Any]]

    def _validate(self, tafra: Tafra):
        cols = set(tafra.columns)
        for col in self._group_by_cols:
            if col not in cols:
                raise KeyError(f'{col} does not exist in tafra')
        for col in self._aggregation.keys():
            if col not in cols:
                raise KeyError(f'{col} does not exist in tafra')
        # we don't have to use all the columns!

    def unique_groups(self, tafra: Tafra) -> List[Any]:
        """
        Construct a unique set of grouped values.
        Uses `OrderedDict` rather than `set` to maintain order.
        """
        return list(OrderedDict.fromkeys(zip(*(tafra[col] for col in self._group_by_cols))))

    def apply(self, tafra: Tafra) -> Tafra:
        """
        Apply the `AggMethod`. Should probably call `unique_groups` to obtain the set of grouped
        values.
        """
        raise NotImplementedError


@dc.dataclass
class GroupBy(AggMethod):
    """
    Analogy to SQL `GROUP BY`, not `pandas.DataFrame.groupby()`. A `reduce` operation.
    """
    def apply(self, tafra: Tafra) -> Tafra:
        self._validate(tafra)
        unique = self.unique_groups(tafra)

        result: Dict[str, np.ndarray] = {
            col: np.empty(len(unique), dtype=tafra[col].dtype) for col in (
                *self._group_by_cols,
                *self._aggregation.keys()
            )
        }

        for i, u in enumerate(unique):
            which_rows = np.full(tafra.rows, True)
            for val, col in zip(u, self._group_by_cols):
                which_rows &= tafra[col] == val
                result[col][i] = val
            for col, fn in self._aggregation.items():
                result[col][i] = fn(tafra[col][which_rows])

        return Tafra(result)


@dc.dataclass
class Transform(AggMethod):
    """
    Analogy to `pandas.DataFrame.transform()`,
    i.e. a SQL `GROUP BY` and `LEFT JOIN` back to the original table.
    """
    def apply(self, tafra: Tafra) -> Tafra:
        self._validate(tafra)
        unique = self.unique_groups(tafra)

        result: Dict[str, np.ndarray] = {
            col: np.empty_like(tafra[col]) for col in (
                *self._group_by_cols,
                *self._aggregation.keys()
            )
        }

        for u in unique:
            which_rows = np.full(tafra.rows, True)
            for val, col in zip(u, self._group_by_cols):
                which_rows &= tafra[col] == val
                result[col][which_rows] = tafra[col][which_rows]
            for col, fn in self._aggregation.items():
                result[col][which_rows] = fn(tafra[col][which_rows])

        return Tafra(result)


Tafra.group_by.__doc__ += GroupBy.__doc__  # type: ignore
Tafra.transform.__doc__ += Transform.__doc__  # type: ignore


if __name__ == '__main__':
    t = Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })

    print('List:\t\t', t.to_list())
    print('Record:\t\t', t.to_record())

    gb = t.group_by(
        ['y', 'z'], {'x': sum}
    )

    print('Group By:\t', gb)
