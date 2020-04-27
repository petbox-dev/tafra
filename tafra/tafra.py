import dataclasses as dc

import numpy as np

from typing import Any, Callable, Dict, List, Tuple, Optional, Iterable


TAFRA_TYPE = {
    'int': lambda x: x.astype(int),
    'float': lambda x: x.astype(float),
    'str': lambda x: x.astype(str),
    'date': lambda x: x.astype('datetime64')
}


def _real_has_attribute(obj, attr):
    try:
        obj.__getattribute__(attr)
        return True
    except AttributeError:
        return False


class DataFrame():
    _data: Dict[str, np.ndarray]
    columns: List[str]

    def __getitem__(self, column: str) -> np.ndarray:
        return self._data[column]

    def __setitem__(self, column: str, value: np.ndarray):
        self._data[column] = value


@dc.dataclass
class Tafra:
    _data: Dict[str, np.ndarray]

    @staticmethod
    def format_type(t, array: np.ndarray):
        return TAFRA_TYPE[t](array)

    @classmethod
    def from_dataframe(cls, df: DataFrame, dtypes: Dict[str, str]) -> 'Tafra':
        return cls({c: Tafra.format_type(dtypes[c], df[c].values) for c in df.columns})

    def __post_init__(self):
        rows = None
        for column, values in self._data.items():
            if rows is None:
                rows = len(values)
            else:
                if rows != len(values):
                    raise ValueError('tafra must have consistent row counts')

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

    @property
    def columns(self) -> Tuple[str, ...]:
        return tuple(self._data.keys())

    @property
    def rows(self) -> int:
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    @property
    def dtypes(self) -> Tuple[np.dtype, ...]:
        return tuple(value.dtype for value in self._data.values())

    def group_by(self, group_by: List[str],
                 aggregation: Dict[str, Callable[[np.ndarray], Any]]) -> 'Tafra':
        return GroupBy(group_by, aggregation).apply(self)

    def transform(self, group_by: List[str],
                  aggregation: Dict[str, Callable[[np.ndarray], Any]]) -> 'Tafra':
        return Transform(group_by, aggregation).apply(self)

    def to_record(self, columns: Optional[Iterable[str]] = None,
                  cast_null: bool = True) -> Tuple[Tuple[Any, ...], ...]:
        """
        Return a tuple of tuples, each inner tuple being a record (i.e. row).
        Useful for e.g. sending records back to a database.
        """
        _cols: Iterable[str]
        _vals: Iterable[Any]

        if columns is None:
            _cols = self.columns
        else:
            _cols = columns

        is_null = lambda x: (isinstance(x, float) or isinstance(x, int)) and np.isnan(x)
        if cast_null:
            _vals = ((
                None if is_null(v) else v
                for v in self._data[c]
            ) for c in _cols)
        else:
            _vals = (self._data[c] for c in _cols)

        return tuple(zip(*_vals))

    def to_list(self, columns: Optional[Iterable[str]] = None) -> List[np.ndarray]:
        """
        Return a list of columns in the tafra
        """
        if columns is None:
            return list(self._data[c] for c in self.columns)
        return list(self._data[c] for c in columns)


@dc.dataclass
class AggMethod:
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

    def apply(self, tafra: Tafra) -> Tafra:
        raise NotImplementedError


@dc.dataclass
class GroupBy(AggMethod):
    def apply(self, tafra: Tafra) -> Tafra:
        self._validate(tafra)

        unique = set(zip(*(tafra[col] for col in self._group_by_cols)))

        result: Dict[str, List[Any]] = {
            col: list() for col in (
                *self._group_by_cols,
                *self._aggregation.keys()
            )
        }

        for u in unique:
            which_rows = np.full(tafra.rows, True)
            for val, col in zip(u, self._group_by_cols):
                result[col].append(val)
                which_rows &= tafra[col] == val
            for col, fn in self._aggregation.items():
                result[col].append(fn(tafra[col][which_rows]))

        tafra_innards: Dict[str, np.ndarray] = dict()
        # preserve dtype on group-by columns
        for col in self._group_by_cols:
            tafra_innards[col] = np.array(result[col], dtype=tafra[col].dtype)
        for col in self._aggregation.keys():
            tafra_innards[col] = np.array(result[col])

        return Tafra(tafra_innards)


@dc.dataclass
class Transform(AggMethod):
    def apply(self, tafra: Tafra) -> Tafra:
        self._validate(tafra)

        unique = set(zip(*(tafra[col] for col in self._group_by_cols)))

        result: Dict[str, np.ndarray] = {
            col: np.full(tafra.rows, None) for col in (
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

        tafra_innards: Dict[str, np.ndarray] = dict()
        # preserve dtype on group-by columns
        for col in self._group_by_cols:
            tafra_innards[col] = np.asarray(
                result[col], dtype=tafra[col].dtype)
        for col in self._aggregation.keys():
            tafra_innards[col] = np.asarray(result[col])

        return Tafra(tafra_innards)


if __name__ == '__main__':
    t = Tafra({
        'x': np.array([1, 2, 3, 4]),
        'y': np.array(['one', 'two', 'one', 'two'], dtype='object'),
    })

    print('List:\t\t', t.to_list())
    print('Record:\t\t', t.to_record())

    gb = t.group_by(
        ['y'], {'x': sum}
    )

    print('Group By:\t', gb)
