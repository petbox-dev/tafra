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
__all__ = ['Tafra']

import sys
import re
import warnings
import pprint as pprint
from itertools import chain
import dataclasses as dc

import numpy as np
from .pandas import DataFrame  # just for mypy...

from typing import (Any, Callable, Dict, Mapping, List, Tuple, Optional, Union, Sequence,
                    Iterable, Iterator, Type, KeysView, ValuesView, ItemsView)
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


@dc.dataclass
class Tafra:
    """
    The innards of a dataframe.

    Parameters
    ----------
        _data: Dict[str, np.ndarray]
            The data used to build the :class:`Tafra`.

        _dtypes: Dict[str, str] = {}
            The dtypes of the ``_data``. If not given, will be inferred from
            the ``_data``.

    """
    _data: Dict[str, np.ndarray]
    _dtypes: Dict[str, str] = dc.field(default_factory=dict)

    def __post_init__(self) -> None:
        rows: Optional[int] = None
        for column, values in self._data.items():
            if rows is None:
                rows = len(values)
            elif rows != len(values):
                raise ValueError('`Tafra` must have consistent row counts.')

        if rows is None:
            raise ValueError('No data provided in constructor statement.')
        self._rows = rows

        self._coalesce_dtypes()
        self.update_dtypes(self._dtypes)

    def __getitem__(self, item: Union[str, int, slice, List[int], List[bool], np.ndarray]) -> Any:
        # return type is actually Union[np.ndarray, 'Tafra'] but mypy goes insane
        if isinstance(item, str):
            return self._data[item]

        elif isinstance(item, int):
            return self._slice(slice(item, item + 1))

        elif isinstance(item, slice):
            return self._slice(item)

        elif isinstance(item, Iterable) or isinstance(item, np.ndarray):
            return self._index(item)

        else:
            raise ValueError(f'Type {type(item)} not supported.')

    def __setitem__(self, item: str, value: Union[np.ndarray, Iterable[Any], Any]) -> None:
        value = self._validate_value(value)
        # create the dict entry as a np.ndarray if it doesn't exist
        self._data.setdefault(item, np.empty(self._rows, dtype=value.dtype))
        self._data[item] = value
        self._dtypes[item] = self._format_type(value.dtype)

    def __len__(self) -> int:
        return self._rows

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f'Tafra(data={self._data}, dtypes={self._dtypes}, rows={self._rows})'

    def _repr_pretty_(self, p: 'IPython.lib.pretty.RepresentationPrinter',  # type: ignore # noqa
                      cycle: bool) -> None:
        """
        A dunder method for IPython to pretty print.

        Parameters
        ----------
            p: IPython.lib.pretty.RepresentationPrinter
                IPython provides this class to handle the object representation.

            cycle: bool
                IPython has detected an infinite loop. Print an alternative
                represenation and return.

        Returns
        -------
            None
                Calls p.text and returns.
        """
        if cycle:
            p.text('Tafra(...)')
        else:
            p.text(self._pretty_format(lambda s: ' ' + pprint.pformat(s, indent=1)[1:].strip()))

    def _repr_html_(self) -> str:
        """
        a dunder moethod for Jupyter Notebook to print HTML.
        """
        return self.to_html()

    def _pretty_format(self, formatter: Callable[[object], str]) -> str:
        """
        Format _data and _dtypes for pretty printing.

        Parameters
        ----------
            formatter: Callabke[[object], str]
                A formatter that operates on the _data and _dtypes :class:`dict`.

        Returns
        -------
            string: str
                The formatted string for printing.
        """
        PATTERN = '(, dtype=[a-z]+)(?=\))'

        return '\n'.join([
            'Tafra(data = {',
            f'{re.sub(PATTERN, "", formatter(self._data))},',
            'dtypes = {',
            f'{re.sub(PATTERN, "", formatter(self._dtypes))},',
            f'rows = {self._rows})'
        ])

    def pformat(self, indent: int = 1, width: int = 80, depth: Optional[int] = None,
                compact: bool = False) -> str:
        """
        Format for pretty printing. Parameters are passed to :class:`pprint.PrettyPrinter`.

        Parameters
        ----------
            indent: int
                Number of spaces to indent for each level of nesting.

            width: int
                Attempted maximum number of columns in the output.

            depth: Optional[int]
                The maximum depth to print out nested structures.

            compact: bool
                If true, several items will be combined in one line.

        Returns
        -------
            formatted string: str
                A formatted string for pretty printing.
        """
        return self._pretty_format(
            lambda s: indent * ' ' + pprint.pformat(
                s, indent, width, depth, compact=compact)[1:].strip())

    def pprint(self, indent: int = 1, width: int = 80, depth: Optional[int] = None,
               compact: bool = False) -> None:
        """
        Pretty print. Parameters are passed to :class:`pprint.PrettyPrinter`.

        Parameters
        ----------
            indent: int
                Number of spaces to indent for each level of nesting.

            width: int
                Attempted maximum number of columns in the output.

            depth: Optional[int]
                The maximum depth to print out nested structures.

            compact: bool
                If true, several items will be combined in one line.

        Returns
        -------
            None: None
        """
        print(self.pformat(indent, width, depth, compact=compact))

    @staticmethod
    def _html_thead(columns: Iterable[Any]) -> str:
        """
        Construct the table head of the HTML representation.

        Parameters
        ----------
            columns: Iterable[Any]
                An iterable of items with defined func:`__repr__` methods.

        Returns
        -------
            HTML: str
                The HTML table head.
        """
        return '<thead>\n<tr>\n{th}\n</tr>\n</thead>' \
            .format(th='\n'.join(f'<th>{c}</th>' for c in columns))

    @staticmethod
    def _html_tr(row: Iterable[Any]) -> str:
        """
        Construct each table row of the HTML representation.

        Parameters
        ----------
            row: Iterable[Any]
                An iterable of items with defined func:`__repr__` methods.

        Returns
        -------
            HTML: str
                The HTML table row.
        """
        return '<tr>\n{td}\n</tr>' \
            .format(td='\n'.join(f'<td>{td}</td>' for td in row))

    @staticmethod
    def _html_tbody(tr: Iterable[str]) -> str:
        """
        Construct the table body of the HTML representation.

        Parameters
        ----------
            tr: Iterable[str]
                An iterable of HTML table rows.

        Returns
        -------
            HTML: str
                The HTML table body.
        """
        return '<tbody>\n{tr}\n</tbody>' \
            .format(tr='\n'.join(tr))

    @staticmethod
    def _html_table(thead: str, tbody: str) -> str:
        """
        Construct the final table of the HTML representation.

        Parameters
        ----------
            thead: str
                An HTML representation of the table head.

            tbody: str
                An HTML representation of the table body.

        Returns
        -------
            HTML: str
                The HTML table.
        """
        return f'<table>\n{thead}\n{tbody}\n</table>'

    def to_html(self, n: int = 20) -> str:
        """
        Construct an HTML table representation of the :class:`Tafra` data.

        Parameters
        ----------
            n: int = 20
                Number of items to print.

        Returns
        -------
            HTML: str
                The HTML table representation.
        """
        thead = self._html_thead(chain([''], self._data.keys()))
        tr = chain(
            [self._html_tr(chain(
                ['dtype'],
                self._dtypes.values()
            ))],
            (self._html_tr(chain(
                [i],
                (v[i] for v in self._data.values())
            ))
                for i in range(min(n, self._rows)))
        )
        tbody = self._html_tbody(tr)
        return self._html_table(thead, tbody)

    def _validate_value(self, value: Union[np.ndarray, Iterable[Any], Any]) -> np.ndarray:
        """
        Validate values as an :class:`np.ndarray` of equal length to
        :attr:`rows` before assignment. Will attempt to create a
        :class:`np.ndarray` if ``value`` is not one already, and will check
        that :attr`np.ndarray.ndim` is ``1``.
        If :attr:`np.ndarray.ndim` ``> 1`` it will attempt :meth:`np.squeeze`
        on ``value``.

        Parameters
        ----------
            value: Union[np.ndarray, Iterable[Any], Any]
                The value to be assigned.

        Returns
        -------
            value: np.ndarray
                The validated value.
        """
        if not isinstance(value, np.ndarray):
            if not isinstance(value, Iterable) or isinstance(value, str):
                value = np.asarray([value])
            else:
                value = np.asarray(list(value))

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
        """
        Validate that the column name(s) exists in :attr:`_data`.

        Parameters
        ----------
            columns: Iterable[str]
                The column names to validate.

        Returns
        -------
            None: None
        """
        for column in columns:
            if column not in self._data.keys():
                raise ValueError(f'Column {column} does not exist in `tafra`.')

    def _validate_dtypes(self, dtypes: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate that the dtypes are defined as valid dtypes in
        :const:`TAFRA_TYPE` and the columns exists in :attr:`_data`.

        Parameters
        ----------
            dtypes: Dict[str, Any]
                The dtypes to validate.

        Returns
        -------
            dtypes: Dict[str, str]
                The validated types.
        """
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
    def _format_type(dtype: Any) -> str:
        """
        Parse a dtype into the internally used string representation.

        Parameters
        ----------
            dtype: Any
                The dtype to parse.

        Returns
        -------
            dtype: str
                The parsed dtype.
        """
        _dtype = str(dtype)

        if 'int' in _dtype:
            return 'int'
        if 'float' in _dtype:
            return 'float'
        if 'bool' in _dtype:
            return 'bool'
        if 'str' in _dtype:
            return 'str'
        if '<U' in _dtype:
            return 'str'
        if 'date' in _dtype:
            return 'date'
        if '<M' in _dtype:
            return 'date'
        if 'object' in _dtype:
            return 'object'
        if 'O' in _dtype:
            return 'object'

        return _dtype

    @staticmethod
    def _apply_type(dtype: str, array: np.ndarray) -> np.ndarray:
        """
        Apply the dtype to the :class:`np.ndarray`.

        Parameters
        ----------
            dtype: str
                The parsed dtype.

            array: np.ndaray
                The array to which the dtype is applied.

        Returns
        -------
            array:
                The array with updated dtype.
        """
        return TAFRA_TYPE[dtype](array)

    @classmethod
    def from_dataframe(cls, df: DataFrame, dtypes: Optional[Dict[str, str]] = None) -> 'Tafra':
        """
        Construct a :class:`Tafra` from a :class:`pd.DataFrame`.

        Parameters
        ----------
            df: pd.DataFrame
                The dataframe used to build the :class:`Tafra`.

            dtypes: Optional[Dict[str, str]] = None
                The dtypes of the columns.

        Returns
        -------
            tafra: Tafra
                The constructed :class:`Tafra`.
        """
        if dtypes is None:
            dtypes = {c: t for c, t in zip(df.columns, df.dtypes)}
        dtypes = {c: cls._format_type(t) for c, t in dtypes.items()}

        return cls(
            {c: cls._apply_type(dtypes[c], df[c].values) for c in df.columns},
            {c: dtypes[c] for c in df.columns}
        )

    @classmethod
    def as_tafra(cls, maybe_tafra: Union['Tafra', DataFrame, Dict[str, Any]]) -> Optional['Tafra']:
        """
        Returns the unmodified `tafra`` if already a `Tafra`, else construct
        a `Tafra` from known types or subtypes of :class:`DataFrame` or `dict`.
        Structural subtypes of :class:`DataFrame` are also valid, as are
        classes that have ``cls.__name__ == 'DataFrame'``.

        Parameters
        ----------
            maybe_tafra: Union['tafra', DataFrame]
                The object to ensure is a :class:`Tafra`.

        Returns
        -------
            tafra: Optional[Tafra]
                The :class:`Tafra`, or None is ``maybe_tafra`` is an unknown
                type.
        """
        if isinstance(maybe_tafra, Tafra):
            return maybe_tafra

        elif isinstance(maybe_tafra, DataFrame):
            return cls.from_dataframe(maybe_tafra)

        elif type(maybe_tafra).__name__ == 'DataFrame':
            return cls.from_dataframe(cast(DataFrame, maybe_tafra))

        elif isinstance(maybe_tafra, dict):
            return cls(maybe_tafra)

        raise TypeError(f'Unknown type `{type(maybe_tafra)}` for conversion to `Tafra`')

    @property
    def columns(self) -> Tuple[str, ...]:
        """
        The names of the columns. Equivalent to `Tafra`.keys().

        Returns
        -------
            columns: Tuple[str, ...]
                The column names.
        """
        return tuple(self._data.keys())

    @property
    def rows(self) -> int:
        """
        The number of rows of the first item in :attr:`data`. The :func:`len()`
        of all items have been previously validated.

        Returns
        -------
            rows: int
                The number of rows of the :class:`Tafra`.
        """
        return self._rows

    @property
    def data(self) -> Dict[str, np.ndarray]:
        """
        The :class:`Tafra` data.

        Returns
        -------
            data: Dict[str, np.ndarray]
                The data.
        """
        return self._data

    @property
    def dtypes(self) -> Dict[str, str]:
        """
        The :class:`Tafra` dtypes.

        Returns
        -------
            dtypes: Dict[str, str]
                The dtypes.
        """
        return self._dtypes

    def head(self, n: int = 5) -> None:
        """
        Display the head of the :class:`Tafra`.

        Parameters
        ----------
            n: int = 5
                The number of rows to display.

        Returns
        -------
            None: None
        """
        if _in_notebook():
            try:
                from IPython.display import display  # type: ignore # noqa
                display(self[:min(self._rows, n)])
                return
            except Exception as e:
                pass

        print(self[:min(self._rows, n)].pformat())

    def select(self, columns: Iterable[str]) -> 'Tafra':
        """
        Use column names to slice the :class:`Tafra` columns analogous to
        SQL SELECT.
        This does not copy the data. Call :meth:`copy` to obtain a copy of
        the sliced data.

        Parameters
        ----------
            columns: Iterable[str]
                The column names to slice from the :class:`Tafra`.

        Returns
        -------
            tafra: Tafra
                the :class:`Tafra` with the sliced columns.
        """
        self._validate_columns(columns)

        return Tafra(
            {column: value
             for column, value in self._data.items() if column in columns},
            {column: value
             for column, value in self._dtypes.items() if column in columns}
        )

    def _slice(self, _slice: slice) -> 'Tafra':
        """
        Use slice object to slice np.ndarray.

        Parameters
        ----------
            _slice: slice
                The ``slice`` object.

        Returns
        -------
            tafra: Tafra
                The sliced :class:`Tafra`.
        """
        return Tafra(
            {column: value[_slice]
                for column, value in self._data.items()},
            {column: value
                for column, value in self._dtypes.items()}
        )

    def _index(self, index: Union[List[int], List[bool], np.ndarray]) -> 'Tafra':
        """
        Use numpy indexing to slice the data :class:`np.ndarray`.

        Parameters
        ----------
            index: Union[Sequence[int], np.ndarray]

        """
        if isinstance(index, List) and not(
                all(isinstance(item, int) for item in index)
                or all(isinstance(item, bool) for item in index)
        ):
            raise ValueError(f'Index list of type `list` does not contain all `int` or `bool`.')

        elif isinstance(index, np.ndarray):
            if not (index.dtype == np.int or index.dtype == np.bool):
                raise ValueError(
                    f'Index array is of dtype={index.dtype}, '
                    'must subtype of `np.int` or `np.bool`.')
            elif index.ndim != 1:
                raise ValueError(f'Indexing np.ndarray must ndim == 1, got ndim == {index.ndim}')

        else:
            raise ValueError(f'Unsupported index type `{type(index).__name__}`.')

        return Tafra(
            {column: value[index]
                for column, value in self._data.items()},
            {column: value
                for column, value in self._dtypes.items()}
        )

    def keys(self) -> KeysView[str]:
        """
        Return the keys of :attr:`data`, i.e. like :meth:`dict.keys()`.

        Returns
        -------
            data keys: KeysView[str]
                The keys of the data property.
        """
        return self._data.keys()

    def values(self) -> ValuesView[np.ndarray]:
        """
        Return the values of :attr:`data`, i.e. like :meth:`dict.values()`.

        Returns
        -------
            data values: ValuesView[np.ndarray]
                The values of the data property.
        """
        return self._data.values()

    def items(self) -> ItemsView[str, np.ndarray]:
        """
        Return the items of :attr:`data`, i.e. like :meth:`dict.items()`.

        Returns
        -------
            items: ItemsView[str, np.ndarray]
                The data items.
        """
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Return from the :meth:`get` function of :attr:`data`, i.e. like
        :meth:`dict.get()`.

        Parameters
        ----------
            key: str
                The key value in the data property.

            default: Any
                The default to return if the key does not exist.

        Returns
        -------
            value: Any
                The value for the key, or the default if the key does not
                exist.
        """
        return self._data.get(key, default)

    def update(self, other: 'Tafra') -> None:
        """
        Update the data and dtypes of this :class:`Tafra` with another
        :class:`Tafra`. Length of rows must match, while data of different
        ``dtype`` will overwrite.

        Parameters
        ----------
            other: Tafra
                The other :class:`Tafra` from which to update.

        Returns
        -------
            None: None
        """
        rows = self._rows
        for column, values in other._data.items():
            if len(values) != rows:
                raise ValueError(
                    'Other `Tafra` must have consistent row count. '
                    f'This `Tafra` has {rows} rows, other `Tafra` has {len(values)} rows.')
            self._data[column] = values

        self.update_dtypes(other._dtypes)

    def _coalesce_dtypes(self) -> None:
        """
        Update :attr:`_dtypes` with missing keys that exist in :attr:`_data`.

        Returns
        -------
            None: None
        """
        for column in self._data.keys():
            if column not in self._dtypes:
                self._dtypes[column] = self._format_type(self._data[column].dtype)

    def update_dtypes(self, dtypes: Dict[str, Any], inplace: bool = True) -> Optional['Tafra']:
        """
        Apply new dtypes.

        Parameters
        ----------
            dtypes: Dict[str, Any]
                The dtypes to update. If ``None``, create from entries in
                :attr:`data`.

            inplace: bool = True
                Perform the operation in place. Otherwise, return a copy.

        Returns
        -------
            tafra: Optional[Tafra]
                The updated :class:`Tafra`.
        """
        if inplace:
            tf = self
        else:
            tf = self.copy()

        tf._validate_columns(dtypes.keys())
        dtypes = tf._validate_dtypes(dtypes)
        tf._dtypes.update(dtypes)

        for column in tf._dtypes.keys():
            if tf._format_type(tf._data[column].dtype) != tf._dtypes[column]:
                tf._data[column] = tf._apply_type(tf._dtypes[column], tf._data[column])

        if inplace:
            return None
        else:
            return tf

    def rename(self, renames: Dict[str, str], inplace: bool = True) -> Optional['Tafra']:
        """
        Rename columns in the :class:`Tafra` from a :class:`dict`.

        Parameters
        ----------
            renames: Dict[str, str]
                The map from current names to new names.

            inplace: bool = True
                Perform the operation in place.

        Returns
        -------
            tafra: Optional[Tafra]
                The :class:`Tafra` with update names.
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
        """
        Remove a column from :attr:`data` and :attr:`dtypes`.

        Parameters
        ----------
            column: str
                The column to remove.

            inplace: bool = True
                Perform the operation in place.

        Returns
        -------
            tafra: Optional[Tafra]
                The :class:`Tafra` with the deleted column.
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
        """
        Create a copy of a :class:`Tafra`.

        Parameters
        ----------
            order: str = 'C' {‘C’, ‘F’, ‘A’, ‘K’}
                Controls the memory layout of the copy. ‘C’ means C-order,
                ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
                ‘C’ otherwise. ‘K’ means match the layout of a as closely as
                possible.

        Returns
        -------
            tafra: Tafra
                A copied :class:`Tafra`.
        """
        return Tafra(
            {column: value.copy(order=order)
                for column, value in self._data.items()},
            {column: value
                for column, value in self._dtypes.items()}
        )

    def coalesce(self, column: str,
                 fills: Iterable[Union[None, str, int, float, bool, np.ndarray]]) -> np.ndarray:
        """
        Fill ``None`` values from ``fills``. Analogous to ``SQL COALESCE`` or
        :meth:`pd.fillna`.

        Parameters
        ----------
            column: str
                The column to coalesce.

            fills: Iterable[Union[str, int, float, bool, np.ndarray]:

        Returns
        -------
            data: np.ndarray
                The coalesced data.
        """
        #TODO: handle dtype?
        iter_fills = iter(fills)
        head = next(iter_fills)

        if column in self._data.keys():
            value = self._data[column].copy()
        else:
            value = np.empty(self._rows, np.asarray(head).dtype)

        for fill in chain([head], iter_fills):
            f = np.atleast_1d(fill)
            where_na = np.full(self._rows, False)
            try:
                where_na |= np.isnan(value)
            except:
                pass

            try:
                where_na |= value == np.array([None])
            except:
                pass

            for w in where_na:
                if len(f) == 1:
                    value[where_na] = f
                else:
                    value[where_na] = f[where_na]

        return value

    def union(self, other: 'Tafra', inplace: bool = False) -> Optional['Tafra']:
        """
        Union two :class:`Tafra` together. Analogy to SQL UNION or
        `pandas.append`. All column names and dtypes must match.

        Parameters
        ----------
            other: Tafra
                The other :class:`Tafra` to union.

            inplace: bool = False
                Perform the operation in place.

        Returns
        -------
            tafra: Tafra
                The unioned :class`Tafra`.
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
    def _cast_record(dtype: str, data: np.ndarray, cast_null: bool) -> Optional[float]:
        """
        Casts needed to generate records for database insert.
        Will cast ``np.nan`` to ``None``. Requires changing ``dtype`` to
        ``object``.

        Parameters
        ----------
            dtype: str
                The dtype of the data value.

            data: np.ndarray
                The data to have its values cast.

            cast_null: bool
                Perform the cast for ``np.nan``

        Returns
        -------
            value: Any
                The cast value.
        """
        value: Any = RECORD_TYPE[dtype](data.item())
        if cast_null and dtype == 'float' and np.isnan(data.item()):
            return None
        return value

    def to_records(self, columns: Optional[Iterable[str]] = None,
                   cast_null: bool = True) -> Iterator[Tuple[Any, ...]]:
        """
        Return a :class:`Iterator` of :class:`Tuple`, each being a record
        (i.e. row) and allowing heterogeneous typing. Useful for e.g. sending
        records back to a database.

        Parameters
        ----------
            columns: Optional[Iterable[str]] = None
                The columns to extract. If ``None``, extract all columns.

            cast_null: bool
                Cast ``np.nan`` to None. Necessary for :mod:``pyodbc``

        Returns
        -------
            records: Iterator[Tuple[Any, ...]]
        """
        if columns is None:
            columns = self.columns
        else:
            self._validate_columns(columns)

        for row in range(self._rows):
            yield tuple(self._cast_record(
                self._dtypes[c], self._data[c][[row]],
                cast_null
            ) for c in columns)
        return

    def to_list(self, columns: Optional[Iterable[str]] = None,
                inner: bool = False) -> Union[List[np.ndarray], List[List[Any]]]:
        """
        Return a list of homogeneously typed columns (as np.ndarrays) in the
        :class:`Tafra`. If a generator is needed, use `Tafra.to_records()`.
        If `inner == True` each column will be cast from :class:`np.ndarray`
        to a :class:`List`.

        Parameters
        ----------
            columns: Optional[Iterable[str]] = None
                The columns to extract. If ``None``, extract all columns.

            inner: bool = False
                Cast all :class:`np.ndarray` to :class`List`.

        Returns
        -------
            list: Union[List[np.ndarray], List[List[Any]]]
        """
        if columns is None:
            if inner:
                return [list(v) for v in self._data.values()]
            return list(self._data.values())

        else:
            if inner:
                return [list(self._data[c]) for c in columns]
            return list(self._data[c] for c in columns)

    def group_by(self, group_by: Iterable[str], aggregation: 'InitAggregation' = {},
                 iter_fn: Mapping[str, Callable[[np.ndarray], Any]] = dict()) -> 'Tafra':
        """
        Helper function to implement :class:`tafra.groups.GroupBy`.
        """
        return GroupBy(group_by, aggregation, iter_fn).apply(self)

    def transform(self, group_by: Iterable[str], aggregation: 'InitAggregation' = {},
                  iter_fn: Dict[str, Callable[[np.ndarray], Any]] = dict()) -> 'Tafra':
        """
        Helper function to implement :class:`tafra.groups.Transform`.
        """
        return Transform(group_by, aggregation, iter_fn).apply(self)

    def iterate_by(self, group_by: Iterable[str]) -> Iterator['GroupDescription']:
        """
        Helper function to implement :class:`tafra.groups.IterateBy`.
        """
        yield from IterateBy(group_by).apply(self)

    def inner_join(self, right: 'Tafra', on: Iterable[Tuple[str, str, str]],
                   select: Iterable[str] = list()) -> 'Tafra':
        """
        Helper function to implement :class:`tafra.groups.InnerJoin`.
        """
        return InnerJoin(on, select).apply(self, right)

    def left_join(self, right: 'Tafra', on: Iterable[Tuple[str, str, str]],
                  select: Iterable[str] = list()) -> 'Tafra':
        """
        Helper function to implement :class:`tafra.groups.LeftJoin`.
        """
        return LeftJoin(on, select).apply(self, right)

    def cross_join(self, right: 'Tafra',
                   select: Iterable[str] = list()) -> 'Tafra':
        """
        Helper function to implement :class:`tafra.groups.CrossJoin`.
        """
        return CrossJoin([], select).apply(self, right)


def _in_notebook() -> bool:
    """
    Checks if running in a Jupyter Notebook.

    Returns
    -------
        in_notebook: bool
    """
    try:
        from IPython import get_ipython  # type: ignore
        if 'IPKernelApp' in get_ipython().config:  # pragma: no cover
            return True
    except Exception as e:
        pass
    return False

# Import here to resolve circular dependency
from .groups import (GroupBy, Transform, IterateBy, InnerJoin, LeftJoin, CrossJoin,
                     InitAggregation, GroupDescription)

Tafra.group_by.__doc__ += GroupBy.__doc__  # type: ignore
Tafra.transform.__doc__ += Transform.__doc__  # type: ignore
Tafra.iterate_by.__doc__ += IterateBy.__doc__  # type: ignore
Tafra.inner_join.__doc__ += InnerJoin.__doc__  # type: ignore
Tafra.left_join.__doc__ += LeftJoin.__doc__  # type: ignore
Tafra.cross_join.__doc__ += CrossJoin.__doc__  # type: ignore
