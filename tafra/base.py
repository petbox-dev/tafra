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
__all__ = ['Tafra']

import sys
import re
import warnings
import pprint as pprint
from datetime import date, datetime
from itertools import chain, islice
import dataclasses as dc

import numpy as np
from .protocol import Series, DataFrame, Cursor  # just for mypy...

from typing import (Any, Callable, Dict, Mapping, List, Tuple, Optional, Union as _Union, Sequence,
                    NamedTuple, Sized, Iterable, Iterator, Type, KeysView, ValuesView, ItemsView)
from typing import cast
from typing_extensions import Protocol
from io import TextIOWrapper

from .csvreader import CSVReader

from .formatter import ObjectFormatter
object_formatter = ObjectFormatter()

# default object formats
object_formatter['Decimal'] = lambda x: x.astype(float)


NAMEDTUPLE_TYPE: Dict[str, Type[Any]] = {
    'int': int,
    'float': float,
    'bool': bool,
    'str': str,
    'date': date,
    'datetime': datetime,
    'object': str,
}

RECORD_TYPE: Dict[str, Callable[[Any], Any]] = {
    'int': int,
    'float': float,
    'bool': bool,
    'str': str,
    'date': lambda x: x.isoformat(),
    'datetime': lambda x: x.isoformat(),
    'object': str,
}


Scalar = _Union[str, int, float, bool]
_Mapping = _Union[
    Mapping[str, Any],
    Mapping[int, Any],
    Mapping[float, Any],
    Mapping[bool, Any],
]
_Element = _Union[Tuple[str, Any], List[Any], _Mapping]
InitVar = _Union[
    Tuple[str, Any],
    _Mapping,
    Sequence[_Element],
    Iterable[_Element],
    Iterator[_Element],
    enumerate
]


@dc.dataclass(repr=False, eq=False)
class Tafra:
    """
    A minimalist dataframe.

    Constructs a :class:`Tafra` from :class:`dict` of data and (optionally)
    dtypes. Types on parameters are the types of the constructed
    :class:`Tafra`, but attempts are made to parse anything that "looks" like
    a dataframe.

    Parameters
    ----------
        _data: Dict[str, np.ndarray]
            The data used to build the :class:`Tafra`.

        _dtypes: Dict[str, str] = {}
            The dtypes of the ``_data``. If not given, will be inferred from
            the ``_data``.

    """
    data: dc.InitVar[InitVar]
    dtypes: dc.InitVar[InitVar] = None

    _data: Dict[str, np.ndarray] = dc.field(init=False)
    _dtypes: Dict[str, str] = dc.field(init=False)

    def __post_init__(self, data: InitVar, dtypes: Optional[InitVar]) -> None:
        # TODO: enable this?
        # if isinstance(self._data, DataFrame):
        #     tf = self.from_dataframe(df=self._data)
        #     self._data = tf._data
        #     self._dtypes = tf._dtypes
        #     self._rows = tf._rows
        #     return

        rows: Optional[int] = None

        # check that the structure is actually a dict
        self._data = self._check_initvar(data)
        if dtypes is None or isinstance(dtypes, property):
            self._dtypes = {}
        else:
            self._dtypes = cast(Dict[str, str], self._check_initvar(dtypes))

        # check that the values are properly formed np.ndarray
        for column, value in self._data.items():
            self._ensure_valid(column, value, check_rows=False)

            if rows is None:
                rows = len(self._data[column])
            elif rows != len(self._data[column]):
                raise ValueError('`Tafra` must have consistent row counts.')

        if rows is None:
            raise ValueError('No data provided in constructor statement.')

        self.update_dtypes_inplace(self._dtypes)
        # must coalesce all dtypes immediately, other functions assume a
        # proper structure of the Tafra
        self._coalesce_dtypes()
        self._update_rows()

    def _check_initvar(self, values: InitVar) -> Dict[str, Any]:
        _values: Dict[Any, Any]

        if isinstance(values, (Mapping, dict)):
            _values = cast(Dict[str, Any], values)

        elif isinstance(values, Sequence):
            _values = self._parse_sequence(values)

        elif isinstance(values, (Iterator, enumerate)):
            _values = self._parse_iterator(cast(Iterator[_Element], values))

        elif isinstance(values, Iterable):
            _values = self._parse_iterable(cast(Iterable[_Element], values))

        else:
            # last ditch attempt
            _values = cast(Dict[Any, Any], values)

        if not isinstance(_values, Dict):
            raise TypeError('Must contain `Dict`, `Mapping`, `Sequence`, Iterable, or Iterator, '
                            f'got `{type(_values)}`')

        # cast all keys to strings if they are not
        # must copy first as mutating the dict changes next(iterator)
        columns = [c for c in _values.keys() if not isinstance(c, str)]
        for column in columns:
            _values[str(column)] = _values.pop(column)

        return _values

    def _parse_sequence(self, values: Sequence[_Element]) -> Dict[Any, Any]:
        head = values[0]
        if isinstance(head, Dict):
            for _dict in values:
                head.update(cast(Dict[Any, Any], _dict))
            _values = head

        elif isinstance(head, Sequence) and len(head) == 2:
            # maybe a Sequence of 2-tuples or 2-lists? Cast and try it.
            _values = dict(
                cast(Iterable[Tuple[Any, Any]], values))

        else:
            raise TypeError('Sequence must contain `Dict`, `Mapping`, or `Sequence`, '
                            f'got `{type(head)}`')

        return _values

    def _parse_iterable(self, values: Iterable[_Element]) -> Dict[Any, Any]:
        iter_values = iter(values)
        head = next(iter_values)
        if isinstance(head, Dict):
            for _dict in iter_values:
                head.update(cast(Dict[Any, Any], _dict))
            _values = head

        elif isinstance(head, Sequence) and len(head) == 2:
            # maybe an Iterable of 2-tuples or 2-lists? Cast and try it.
            _values = dict(chain(
                cast(Iterable[Tuple[Any, Any]], [head]),
                cast(Iterator[Tuple[Any, Any]], values)))

        else:
            raise TypeError('Iterable must contain `Dict`, `Mapping`, or `Sequence`, '
                            f'got `{type(head)}`')

        return _values

    def _parse_iterator(self, values: Iterator[_Element]) -> Dict[Any, Any]:
        head = next(values)

        if isinstance(head, Dict):
            # consume the iterator if its a dict
            for _dict in values:
                head.update(cast(Dict[Any, Any], _dict))
            _values = head

        elif isinstance(head, Sequence) and len(head) == 2:
            # maybe an Iterator of 2-tuples or 2-lists? Cast and try it.
            _values = dict(chain(
                cast(Iterable[Tuple[Any, Any]], [head]),
                cast(Iterator[Tuple[Any, Any]], values)))

        else:
            raise TypeError('Iterator must contain `Dict`, `Mapping`, or `Sequence`, '
                            f'got `{type(head)}`')

        return _values

    def __getitem__(
            self,
            item: _Union[str, int, slice, Sequence[_Union[str, int, bool]], np.ndarray]) -> Any:
        # return type is actually _Union[np.ndarray, 'Tafra'] but mypy goes insane
        if isinstance(item, str):
            return self._data[item]

        elif isinstance(item, (int, slice)):
            return self._slice(item)

        elif isinstance(item, np.ndarray):
            return self._ndindex(item)

        elif isinstance(item, Sequence):
            if isinstance(item[0], str):
                return self.select(cast(Sequence[str], item))
            else:
                return self._index(cast(Sequence[_Union[int, bool]], item))

        else:
            raise TypeError(f'Type {type(item)} not supported.')

    def __setitem__(self, item: str, value: _Union[np.ndarray, Sequence[Any], Any]) -> None:
        self._ensure_valid(item, value)

    def __len__(self) -> int:
        assert self._data is not None, 'Cannot construct a Tafra with no data.'
        return self._rows

    def __iter__(self) -> Iterator['Tafra']:
        for i in range(self._rows):
            yield self[i]

    def iterrows(self) -> Iterator['Tafra']:
        """
        Yield rows as :class:`Tafra`.

        Returns
        -------
            tafras: Iterator[Tafra]
                An iterator of :class:`Tafra`.
        """
        yield from self.__iter__()

    def itertuples(self, name: str = 'Tafra') -> Iterator[Tuple[Any, ...]]:
        """
        Yield rows as :class:`NamedTuple`.

        Parameters
        ----------
            name: str = 'Tafra'
                The name for the :class:`NamedTuple`.

        Returns
        -------
            tuples: Iterator[NamedTuple[Any, ...]]
                An iterator of :class:`NamedTuple`.
        """
        TafraNT = NamedTuple(name, **{  # type: ignore
            column: NAMEDTUPLE_TYPE[self._reduce_dtype(dtype)]
            for column, dtype in self._dtypes.items()})

        for tf in self.__iter__():
            yield TafraNT(*(value.item() for value in tf._data.values()))

    def itercols(self) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Yield columns as :class:`Tuple[str, np.ndarray]` .

        Returns
        -------
            tuples: Iterator[Tuple[str, np.ndarray]]
                An iterator of :class:`Tafra`.
        """
        for column, value in self._data.items():
            yield column, value

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f'Tafra(data={self._data}, dtypes={self._dtypes}, rows={self._rows})'

    def _update_rows(self) -> None:
        iter_values = iter(self._data.values())
        self._rows = len(next(iter_values))
        if not all(len(v) == self._rows for v in iter_values):
            raise TypeError('Uneven length of data.')

    def _slice(self, _slice: _Union[int, slice]) -> 'Tafra':
        """
        Use slice object to slice np.ndarray.

        Parameters
        ----------
            _slice: Union[int, slice]
                The ``slice`` object.

        Returns
        -------
            tafra: Tafra
                The sliced :class:`Tafra`.
        """
        return Tafra(
            {column: value[_slice]
                for column, value in self._data.items()},
            self._dtypes
        )

    def _index(self, index: Sequence[_Union[int, bool]]) -> 'Tafra':
        """
        Use numpy indexing to slice the data :class:`np.ndarray`.

        Parameters
        ----------
            index: Sequence[Union[int, bool]]

        """
        # TODO: just let numpy handle errors?
        # head = index[0]

        # if isinstance(index[0], str):
        #     if all(isinstance(item, str) for item in index):
                # return self.select(cast(Sequence[str], index))
        #     else:
        #         raise IndexError('Index `Sequence` does not contain all `str`.')

        # elif (isinstance(head, bool)
        #         and not(all(isinstance(item, bool) for item in index))):
        #     raise IndexError('Index `Sequence` does not contain all `bool`.')

        # elif (isinstance(head, int)   # type: ignore
        #         and not(all(isinstance(item, int) for item in index))):
        #     raise IndexError('Index `Sequence` does not contain all `int`.')

        return Tafra(
            {column: value[index]
                for column, value in self._data.items()},
            self._dtypes
        )

    def _ndindex(self, index: np.ndarray) -> 'Tafra':
        """
        Use :class:`numpy.ndarray` indexing to slice the data :class:`np.ndarray`.

        Parameters
        ----------
            index: np.ndarray

        """
        # TODO: just let numpy handle errors?
        # if not (index.dtype == np.int or index.dtype == np.bool):
        #     raise IndexError(
        #         f'Index array is of dtype={index.dtype}, '
        #         'must subtype of `np.int` or `np.bool`.')
        if index.ndim != 1:
            raise IndexError(f'Indexing np.ndarray must ndim == 1, got ndim == {index.ndim}')

        return Tafra(
            {column: value[index]
                for column, value in self._data.items()},
            self._dtypes
        )

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
        PATTERN = r'(, dtype=[a-z]+)(?=\))'

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

    def _ensure_valid(self, column: str, value: _Union[np.ndarray, Sequence[Any], Any],
                      check_rows: bool = True) -> None:
        """
        Validate values as an :class:`np.ndarray` of equal length to
        :attr:`rows` before assignment. Will attempt to create a
        :class:`np.ndarray` if ``value`` is not one already, and will check
        that :attr`np.ndarray.ndim` is ``1``. If :attr:`np.ndarray.ndim`
        ``> 1`` it will attempt :meth:`np.squeeze`
        on ``value``.

        Parameters
        ----------
            column: str
                The column to assign to.

            value: Union[np.ndarray, Sequence[Any], Any]
                The value to be assigned.

        Returns
        -------
            None: None
        """
        id_value = id(value)
        rows = self._rows if check_rows else 1

        if isinstance(value, np.ndarray):
            if len(value.shape) == 0:
                value = np.full(rows, value.item())

        elif isinstance(value, str) or not isinstance(value, Sized) or value is None:
            value = np.full(rows, value)

        elif isinstance(value, Iterable):
            value = np.array(value)

        assert isinstance(value, np.ndarray), '`Tafra` only supports assigning `ndarray`.'

        if value.ndim > 1:
            sq_value = value.squeeze()
            if sq_value.ndim > 1:
                raise ValueError('`ndarray` or `np.squeeze(ndarray)` must have ndim == 1.')
            elif sq_value.ndim == 1:
                # if value was a single item, squeeze returns zero length item
                warnings.warn('`np.squeeze(ndarray)` applied to set ndim == 1.')
                warnings.resetwarnings()
                value = sq_value

        assert value.ndim >= 1, '`Tafra` only supports assigning ndim >= 1.'

        if check_rows and len(value) != rows:
            raise ValueError(
                '`Tafra` must have consistent row counts.\n'
                f'This `Tafra` has {rows} rows. Assigned np.ndarray has {len(value)} rows.')

        # special parsing of various object types
        parsed_value = object_formatter.parse_dtype(value)
        if parsed_value is not None:
            value = parsed_value

        # have we modified value?
        if id(value) != id_value:
            self._data[column] = value
            self._dtypes[column] = self._format_dtype(value.dtype)

    def parse_object_dtypes(self) -> 'Tafra':
        """
        Parse the object dtypes using the :class:`ObjectFormatter` instance.
        """
        tafra = self.copy()
        tafra.parse_object_dtypes_inplace()
        return tafra

    def parse_object_dtypes_inplace(self) -> None:
        """
        Inplace version.

        Parse the object dtypes using the :class:`ObjectFormatter` instance.
        """
        for column, value in self._data.items():
            parsed_value = object_formatter.parse_dtype(value)
            if parsed_value is not None:
                self._data[column] = parsed_value
                self._dtypes[column] = self._format_dtype(parsed_value.dtype)

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
        Validate that the dtypes as internally used names and that the columns
        exists in :attr:`_data`.

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
        _dtypes: Dict[str, Any] = {}

        self._validate_columns(dtypes.keys())

        for column, _dtype in dtypes.items():
            _dtypes[column] = self._format_dtype(_dtype)

        return _dtypes

    @staticmethod
    def _format_dtype(dtype: Any) -> str:
        """
        Parse a dtype into the internally used string representation, if
        defined. Otherwise, pass through and let numpy raise error if it is
        not a valid dtype.

        Parameters
        ----------
            dtype: Any
                The dtype to parse.

        Returns
        -------
            dtype: str
                The parsed dtype.
        """
        _dtype = np.dtype(dtype)
        name = _dtype.type.__name__
        if 'str' in name:
            return _dtype.str

        return name.replace('_', '')

    @staticmethod
    def _reduce_dtype(dtype: Any) -> str:
        """
        Parse a dtype to the base type.

        Parameters
        ----------
            dtype: Any
                The dtype to parse.

        Returns
        -------
            dtype: str
                The parsed dtype.
        """
        name = np.dtype(dtype).type.__name__
        m = re.search(r'([a-z]+)', name)
        if m:
            return m.group(1)

        # are there any dtypes without text names?
        return name  # pragma: no cover

    @classmethod
    def from_records(cls, records: Iterable[Iterable[Any]], columns: Iterable[str],
                     dtypes: Optional[Iterable[Any]] = None) -> 'Tafra':
        """
        Construct a :class:`Tafra` from an Iterator of records, e.g. from a
        SQL query. The records should be a nested Iterable, but can also be
        fed a cursor method such as ``cur.fetchmany()`` or ``cur.fetchall()``.

        Parameters
        ----------
            records: ITerable[Iteralble[str]]
                The records to turn into a :class:`Tafra`.

            columns: Iterable[str]
                The column names to use.

            dtypes: Optional[Iterable[Any]] = None
                The dtypes of the columns.

        Returns
        -------
            tafra: Tafra
                The constructed :class:`Tafra`.
        """
        if dtypes is None:
            return Tafra({column: value for column, value in zip(columns, zip(*records))})

        return Tafra(
            {column: value for column, value in zip(columns, zip(*records))},
            {column: value for column, value in zip(columns, dtypes)}
        )

    @classmethod
    def from_series(cls, s: Series, dtype: Optional[str] = None) -> 'Tafra':
        """
        Construct a :class:`Tafra` from a :class:`pd.DataFrame`.

        Parameters
        ----------
            df: pd.DataFrame
                The dataframe used to build the :class:`Tafra`.

            dtypes: Optional[str] = None
                The dtypes of the columns.

        Returns
        -------
            tafra: Tafra
                The constructed :class:`Tafra`.
        """
        if dtype is None:
            dtype = s.dtype
        dtypes = {s.name: cls._format_dtype(dtype)}

        return cls(
            {s.name: s.values.astype(dtypes[s.name])},
            dtypes
        )

    @classmethod
    def from_dataframe(cls, df: DataFrame, dtypes: Optional[Dict[str, Any]] = None) -> 'Tafra':
        """
        Construct a :class:`Tafra` from a :class:`pd.DataFrame`.

        Parameters
        ----------
            df: pd.DataFrame
                The dataframe used to build the :class:`Tafra`.

            dtypes: Optional[Dict[str, Any]] = None
                The dtypes of the columns.

        Returns
        -------
            tafra: Tafra
                The constructed :class:`Tafra`.
        """
        if dtypes is None:
            dtypes = {c: t for c, t in zip(df.columns, df.dtypes)}
        dtypes = {c: cls._format_dtype(t) for c, t in dtypes.items()}

        return cls(
            {c: df[c].values.astype(dtypes[c]) for c in df.columns},
            {c: dtypes[c] for c in df.columns}
        )

    @classmethod
    def read_sql(cls, query: str, cur: Cursor) -> 'Tafra':
        """
        Execute a SQL SELECT statement using a :class:`pyodbc.Cursor` and
        return a Tuple of column names and an Iterator of records.

        Parameters
        ----------
            query: str
                The SQL query.

            cur: pyodbc.Cursor
                The SQL cursor.

        Returns
        -------
            tafra: Tafra
                The constructed :class:`Tafra`.
        """
        cur.execute(query)

        columns, dtypes = zip(*((d[0], d[1]) for d in cur.description))

        head = cur.fetchone()
        if head is None:
            return Tafra({column: () for column in columns})

        return Tafra.from_records(chain([head], cur.fetchall()), columns, dtypes)

    @classmethod
    def read_sql_chunks(cls, query: str, cur: Cursor, chunksize: int = 100) -> Iterator['Tafra']:
        """
        Execute a SQL SELECT statement using a :class:`pyodbc.Cursor` and
        return a Tuple of column names and an Iterator of records.

        Parameters
        ----------
            query: str
                The SQL query.

            cur: pyodbc.Cursor
                The SQL cursor.

        Returns
        -------
            tafra: Tafra
                The constructed :class:`Tafra`.
        """
        cur.execute(query)

        columns, dtypes = zip(*((d[0], d[1]) for d in cur.description))

        head = cur.fetchone()
        if head is None:
            yield Tafra({column: () for column in columns})
            return

        def chunks(iterable: Iterable[Any], chunksize: int = 1000) -> Iterator[Iterable[Any]]:
            for f in iterable:
                yield list(chain([f], islice(iterable, chunksize - 1)))

        for chunk in chunks(chain([head], cur), chunksize):
            yield Tafra.from_records(chunk, columns, dtypes)

    @classmethod
    def read_csv(cls, csv_file: _Union[str, TextIOWrapper], guess_rows: int = 5,
                 dtypes: Optional[Dict[str, str]] = None, **csvkw: Dict[str, Any]
                 ) -> 'Tafra':
        """
        Read a CSV file with a header row, infer the types of each column,
        and return a Tafra containing the file's contents.

        Parameters
        ----------
            csv_file: Union[str, TextIOWrapper]
                The path to the CSV file, or an open file-like object.

            guess_rows: int
                The number of rows to use when guessing column types.

            dtypes: Optional[Dict[str, str]]
                dtypes by column name; by default, all dtypes will be inferred
                from the file contents.

            **csvkw: Dict[str, Any]
                Additional keyword arguments passed to csv.reader.

        Returns
        -------
            tafra: Tafra
                The constructed :class:`Tafra`.
        """
        reader = CSVReader(csv_file, guess_rows, **csvkw)
        # TODO: check the typing of the dtypes argument to Tafra.__init__ -
        #   for now this won't typecheck but is correct.
        return Tafra(reader.read(), dtypes=dtypes) # type: ignore

    @classmethod
    def as_tafra(cls, maybe_tafra: _Union['Tafra', DataFrame, Series, Dict[str, Any], Any]
                 ) -> Optional['Tafra']:
        """
        Returns the unmodified `tafra`` if already a `Tafra`, else construct
        a `Tafra` from known types or subtypes of :class:`DataFrame` or `dict`.
        Structural subtypes of :class:`DataFrame` or :class:`Series` are also
        valid, as are classes that have ``cls.__name__ == 'DataFrame'`` or
        ``cls.__name__ == 'Series'``.

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

        elif isinstance(maybe_tafra, Series):  # pragma: no cover
            return cls.from_series(maybe_tafra)

        elif type(maybe_tafra).__name__ == 'Series':  # pragma: no cover
            return cls.from_series(cast(Series, maybe_tafra))

        elif isinstance(maybe_tafra, DataFrame):  # pragma: no cover
            return cls.from_dataframe(maybe_tafra)

        elif type(maybe_tafra).__name__ == 'DataFrame':  # pragma: no cover
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

    @columns.setter
    def columns(self, value: Any) -> None:
        raise ValueError('Assignment to `columns` is forbidden.')

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
        return self.__len__()

    @rows.setter
    def rows(self, value: Any) -> None:
        raise ValueError('Assignment to `rows` is forbidden.')

    @property  # type: ignore
    def data(self) -> Dict[str, np.ndarray]:
        """
        The :class:`Tafra` data.

        Returns
        -------
            data: Dict[str, np.ndarray]
                The data.
        """
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        raise ValueError('Assignment to `data` is forbidden.')

    @property  # type: ignore
    def dtypes(self) -> Dict[str, str]:
        """
        The :class:`Tafra` dtypes.

        Returns
        -------
            dtypes: Dict[str, str]
                The dtypes.
        """
        return self._dtypes

    @dtypes.setter
    def dtypes(self, value: Any) -> None:
        raise ValueError('Assignment to `dtypes` is forbidden.')

    @property
    def size(self) -> int:
        """
        The :class:`Tafra` size.

        Returns
        -------
            size: int
                The size.
        """
        return self.rows * len(self.columns)

    @size.setter
    def size(self, value: Any) -> None:
        raise ValueError('Assignment to `size` is forbidden.')

    @property
    def ndim(self) -> int:
        """
        The :class:`Tafra` number of dimensions.

        Returns
        -------
            ndim: int
                The number of dimensions.
        """
        return max(2, len(self.columns))

    @ndim.setter
    def ndim(self, value: Any) -> None:
        raise ValueError('Assignment to `ndim` is forbidden.')

    @property
    def shape(self) -> Tuple[int, int]:
        """
        The :class:`Tafra` shape.

        Returns
        -------
            shape: int
                The shape.
        """
        return self.rows, len(self.columns)

    @shape.setter
    def shape(self, value: Any) -> None:
        raise ValueError('Assignment to `shape` is forbidden.')

    def row_map(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Iterator[Any]:
        """
        Map a function over rows. To apply to specific columns, use
        :meth:`select` first. The function must operate on :class:`Tafra`.

        Parameters
        ----------
            fn: Callable[..., Any]
                The function to map.

            *args: Any
                Additional positional arguments to ``fn``.

            **kwargs: Any
                Additional keyword arguments to ``fn``.

        Returns
        -------
            iter_tf: Iterator[Any]
                An iterator to map the function.
        """
        for tf in self.__iter__():
            yield fn(tf, *args, **kwargs)

    def col_map(self, fn: Callable[..., Any], name: bool = True,
                *args: Any, **kwargs: Any) -> Iterator[_Union[Any, Tuple[str, Any]]]:
        """
        Map a function over columns. To apply to specific columns, use
        :meth:`select` first. The function must operate on :class:`Tuple[str, np.ndarray]`.

        Parameters
        ----------
            fn: Callable[..., Any]
                The function to map.

            name: bool
                Return the column name.

            *args: Any
                Additional positional arguments to ``fn``.

            **kwargs: Any
                Additional keyword arguments to ``fn``.

        Returns
        -------
            iter_tf: Iterator[Any]
                An iterator to map the function.
        """
        if name:
            for column, value in self.itercols():
                yield column, fn(value, *args, **kwargs)
            return

        for column, value in self.itercols():
            yield fn(value, *args, **kwargs)

    def head(self, n: int = 5) -> 'Tafra':
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
        return self._slice(slice(n))

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
            {column: value for column, value in self._data.items()
             if column in columns},
            {column: value for column, value in self._dtypes.items()
             if column in columns},
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

    def update(self, other: 'Tafra') -> 'Tafra':
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
        tafra = self.copy()
        tafra.update_inplace(other)
        return tafra

    def update_inplace(self, other: 'Tafra') -> None:
        """
        Inplace version.

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

        for column, value in other._data.items():
            if len(value) != rows:
                raise ValueError(
                    'Other `Tafra` must have consistent row count. '
                    f'This `Tafra` has {rows} rows, other `Tafra` has {len(value)} rows.')
            self._data[column] = value

        self.update_dtypes_inplace(other._dtypes)

    def _coalesce_dtypes(self) -> None:
        """
        Update :attr:`dtypes` with missing keys that exist in :attr:`data`.
        **Must be called if :attr:`data` or :attr:`data` is directly
        modified!**

        Returns
        -------
            None: None
        """
        for column in self._data.keys():
            if column not in self._dtypes:
                self._dtypes[column] = self._format_dtype(self._data[column].dtype)

    def update_dtypes(self, dtypes: Dict[str, Any]) -> 'Tafra':
        """
        Apply new dtypes.

        Parameters
        ----------
            dtypes: Dict[str, Any]
                The dtypes to update. If ``None``, create from entries in
                :attr:`data`.

        Returns
        -------
            tafra: Optional[Tafra]
                The updated :class:`Tafra`.
        """
        tafra = self.copy()
        tafra.update_dtypes_inplace(dtypes)
        return tafra

    def update_dtypes_inplace(self, dtypes: Dict[str, Any]) -> None:
        """
        Inplace version.

        Apply new dtypes.

        Parameters
        ----------
            dtypes: Dict[str, Any]
                The dtypes to update. If ``None``, create from entries in
                :attr:`data`.

        Returns
        -------
            tafra: Optional[Tafra]
                The updated :class:`Tafra`.
        """
        dtypes = self._validate_dtypes(dtypes)
        self._dtypes.update(dtypes)


        for column in dtypes.keys():
            if self._format_dtype(self._data[column].dtype) != self._dtypes[column]:
                self._data[column] = self._data[column].astype(self._dtypes[column])

    def rename(self, renames: Dict[str, str]) -> 'Tafra':
        """
        Rename columns in the :class:`Tafra` from a :class:`dict`.

        Parameters
        ----------
            renames: Dict[str, str]
                The map from current names to new names.

        Returns
        -------
            tafra: Optional[Tafra]
                The :class:`Tafra` with update names.
        """

        tafra = self.copy()
        tafra.rename_inplace(renames)
        return tafra

    def rename_inplace(self, renames: Dict[str, str]) -> None:
        """
        In-place version.

        Rename columns in the :class:`Tafra` from a :class:`dict`.

        Parameters
        ----------
            renames: Dict[str, str]
                The map from current names to new names.

        Returns
        -------
            tafra: Optional[Tafra]
                The :class:`Tafra` with update names.
        """
        self._validate_columns(renames.keys())

        for cur, new in renames.items():
            self._data[new] = self._data.pop(cur)
            self._dtypes[new] = self._dtypes.pop(cur)
        return None

    def delete(self, columns: Iterable[str]) -> 'Tafra':
        """
        Remove a column from :attr:`data` and :attr:`dtypes`.

        Parameters
        ----------
            column: str
                The column to remove.

        Returns
        -------
            tafra: Optional[Tafra]
                The :class:`Tafra` with the deleted column.
        """
        if isinstance(columns, str):
            columns = [columns]

        self._validate_columns(columns)

        return Tafra(
            {column: value.copy() for column, value in self._data.items()
             if column not in columns},
            {column: value for column, value in self._dtypes.items()
             if column not in columns}
        )

    def delete_inplace(self, columns: Iterable[str]) -> None:
        """
        In-place version.

        Remove a column from :attr:`data` and :attr:`dtypes`.

        Parameters
        ----------
            column: str
                The column to remove.

        Returns
        -------
            tafra: Optional[Tafra]
                The :class:`Tafra` with the deleted column.
        """
        if isinstance(columns, str):
            columns = [columns]

        self._validate_columns(columns)

        for column in columns:
            _ = self._data.pop(column, None)
            _ = self._dtypes.pop(column, None)

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
            self._dtypes.copy()
        )

    def coalesce(self, column: str,
                 fills: Iterable[_Union[None, str, int, float, bool, np.ndarray]]) -> np.ndarray:
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
            where_na |= value == np.array([None])
            try:
                where_na |= np.isnan(value)
            except:
                pass
                # pass

            for w in where_na:
                if len(f) == 1:
                    value[where_na] = f
                else:
                    value[where_na] = f[where_na]

        return value

    def coalesce_inplace(self, column: str,
                         fills: Iterable[_Union[None, str, int, float, bool, np.ndarray]]) -> None:
        """
        In-place version.

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
        self._data[column] = self.coalesce(column, fills)
        self.update_dtypes_inplace({column: self._data[column].dtype})

    def _cast_record(self, dtype: str, data: np.ndarray, cast_null: bool) -> Optional[float]:
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
        _dtype = self._reduce_dtype(dtype)
        value: Any = RECORD_TYPE[_dtype](data.item())
        if cast_null and _dtype == 'float' and np.isnan(data.item()):
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
            if isinstance(columns, str):
                columns = [columns]
            self._validate_columns(columns)

        for row in range(self._rows):
            yield tuple(self._cast_record(
                self._dtypes[c], self._data[c][[row]],
                cast_null
            ) for c in columns)
        return

    def to_list(self, columns: Optional[Iterable[str]] = None,
                inner: bool = False) -> _Union[List[np.ndarray], List[List[Any]]]:
        """
        Return a list of homogeneously typed columns (as np.ndarrays). If a
        generator is needed, use `Tafra.to_records()`. If `inner == True`
        each column will be cast from :class:`np.ndarray` to a :class:`List`.

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
            columns = self.columns
        else:
            if isinstance(columns, str):
                columns = [columns]
            self._validate_columns(columns)

        if inner:
            return [list(self._data[c]) for c in columns]
        return [self._data[c] for c in columns]

    def to_tuple(self, columns: Optional[Iterable[str]] = None, name: str = 'Tafra',
                 inner: bool = False) -> _Union[Tuple[np.ndarray], Tuple[Tuple[Any, ...]]]:
        """
        Return a :class:`NamedTuple`. If a generator is needed, use
        :meth:`Tafra.to_records()`. If `inner == True` each column will be
        cast from :class:`np.ndarray` to a :class:`Tuple`.

        Parameters
        ----------
            columns: Optional[Iterable[str]] = None
                The columns to extract. If ``None``, extract all columns.

            inner: bool = False
                Cast all :class:`np.ndarray` to :class`List`.

        Returns
        -------
            list: Union[Tuple[np.ndarray], Tuple[Tuple[Any, ...]]]
        """
        if columns is None:
            columns = self.columns
        else:
            if isinstance(columns, str):
                columns = [columns]
            self._validate_columns(columns)

        # note: mypy does not support dynamically constructed NamedTuple as return type
        TafraNT = NamedTuple(name, **{  # type: ignore
            c: NAMEDTUPLE_TYPE[self._reduce_dtype(self._dtypes[c])]
            for c in columns})

        if inner:
            return TafraNT(*(tuple(self._data[c]) for c in columns))  # type: ignore
        return TafraNT(*(self._data[c] for c in columns))  # type: ignore

    def to_array(self, columns: Optional[Iterable[str]] = None) -> np.ndarray:
        """
        Return an object array.

        Parameters
        ----------
            columns: Optional[Iterable[str]] = None
                The columns to extract. If ``None``, extract all columns.

        Returns
        -------
            array: np.ndarray
        """
        if columns is None:
            columns = self.columns
        else:
            if isinstance(columns, str):
                columns = [columns]
            self._validate_columns(columns)

        return np.array(list(self._data[c] for c in columns)).T

    def to_pandas(self) -> DataFrame:
        """
        Construct a :class:`pandas.DataFrame`.

        Returns
        -------
            dataframe: :class:`pandas.DataFrame`
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError('`pandas` does not appear to be installed.')

        return pd.DataFrame(self._data)

    def union(self, other: 'Tafra') -> 'Tafra':
        """
        Helper function to implement :meth:`tafra.group.Union.apply`.

        Union two :class:`Tafra` together. Analogy to SQL UNION or
        `pandas.append`. All column names and dtypes must match.

        Parameters
        ----------
            other: Tafra
                The other tafra to union.

        Returns
        -------
            tafra: Tafra
                A new tafra with the unioned data.
        """
        return Union().apply(self, other)

    def union_inplace(self, other: 'Tafra') -> None:
        """
        Inplace version.


        Helper function to implement :meth:`tafra.group.Union.apply_inplace`.

        Union two :class:`Tafra` together. Analogy to SQL UNION or
        `pandas.append`. All column names and dtypes must match.

        Parameters
        ----------
            other: Tafra
                The other tafra to union.

        Returns
        -------
            None: None
        """
        Union().apply_inplace(self, other)

    def group_by(self, columns: Iterable[str], aggregation: 'InitAggregation' = {},
                 iter_fn: Mapping[str, Callable[[np.ndarray], Any]] = dict()) -> 'Tafra':
        """
        Helper function to implement :meth:`tafra.group.GroupBy.apply`.

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

        Returns
        -------
            tafra: Tafra
                The aggregated :class:`Tafra`.
        """
        return GroupBy(columns, aggregation, iter_fn).apply(self)

    def transform(self, columns: Iterable[str], aggregation: 'InitAggregation' = {},
                  iter_fn: Dict[str, Callable[[np.ndarray], Any]] = dict()) -> 'Tafra':
        """
        Helper function to implement :meth:`tafra.group.Transform.apply`.

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

        Returns
        -------
            tafra: Tafra
                The transformed :class:`Tafra`.
        """
        return Transform(columns, aggregation, iter_fn).apply(self)

    def iterate_by(self, columns: Iterable[str]) -> Iterator['GroupDescription']:
        """
        Helper function to implement :meth:`tafra.group.IterateBy.apply`.

        A generator that yields a :class:`Tafra` for each set of unique values.

        Analogy to `pandas.DataFrame.groupby()`, i.e. an Sequence of `Tafra` objects.
        Yields tuples of ((unique grouping values, ...), row indices array, subset tafra)

        Parameters
        ----------
            group_by: Iterable[str]
                The column names to group by.

        Returns
        -------
            tafras: Iterator[GroupDescription]
                An iterator over the grouped :class:`Tafra`.
        """
        yield from IterateBy(columns).apply(self)

    def inner_join(self, right: 'Tafra', on: Iterable[Tuple[str, str, str]],
                   select: Iterable[str] = list()) -> 'Tafra':
        """
        Helper function to implement :meth:`tafra.group.InnerJoin.apply`.

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

        Returns
        -------
            tafra: Tafra
                The joined :class:`Tafra`.
        """
        return InnerJoin(on, select).apply(self, right)

    def left_join(self, right: 'Tafra', on: Iterable[Tuple[str, str, str]],
                  select: Iterable[str] = list()) -> 'Tafra':
        """
        Helper function to implement :meth:`tafra.group.LeftJoin.apply`.

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

        Returns
        -------
            tafra: Tafra
                The joined :class:`Tafra`.
        """
        return LeftJoin(on, select).apply(self, right)

    def cross_join(self, right: 'Tafra',
                   select: Iterable[str] = list()) -> 'Tafra':
        """
        Helper function to implement :meth:`tafra.group.CrossJoin.apply`.

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

        Returns
        -------
            tafra: Tafra
                The joined :class:`Tafra`.
        """
        return CrossJoin([], select).apply(self, right)

def _in_notebook() -> bool:  # pragma: no cover
    """
    Checks if running in a Jupyter Notebook.

    Returns
    -------
        in_notebook: bool
    """
    try:
        from IPython import get_ipython  # type: ignore
        if 'IPKernelApp' in get_ipython().config:
            return True
    except Exception as e:
        pass
    return False

# Import here to resolve circular dependency
from .group import (GroupBy, Transform, IterateBy, InnerJoin, LeftJoin, CrossJoin, Union,
                    InitAggregation, GroupDescription)
