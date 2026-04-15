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

__all__ = ["Tafra"]

from pathlib import Path
import re
import warnings
import csv
import pprint as pprint
from datetime import date, datetime
from itertools import chain, islice
from collections import namedtuple
import dataclasses as dc

import numpy as np
from .protocol import Series, DataFrame, Cursor  # just for mypy...

from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
    Sized,
    Iterable,
    Iterator,
    KeysView,
    ValuesView,
    ItemsView,
    IO,
    TYPE_CHECKING,
)
from typing_extensions import Concatenate, Literal, ParamSpec
from typing import cast
from io import TextIOWrapper

from .formatter import ObjectFormatter
from .csvreader import CSVReader


P = ParamSpec("P")


# default object formats
object_formatter = ObjectFormatter()
object_formatter["Decimal"] = lambda x: x.astype(float)


NAMEDTUPLE_TYPE: dict[str, type[Any]] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "date": date,
    "datetime": datetime,
    "object": str,
}

RECORD_TYPE: dict[str, Callable[[Any], Any]] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "date": lambda x: x.isoformat(),
    "datetime": lambda x: x.isoformat(),
    "object": str,
}


Scalar = str | int | float | bool
_Mapping = Mapping[str, Any] | Mapping[int, Any] | Mapping[float, Any] | Mapping[bool, Any]
_Element = tuple[str | int | float | np.ndarray[Any, Any], Any] | list[Any] | _Mapping
InitVar = (
    tuple[str, Any]
    | _Mapping
    | Sequence[_Element]
    | Iterable[_Element]
    | Iterator[_Element]
    | enumerate[Any]
)


@dc.dataclass(repr=False, eq=False)
class Tafra:
    """
    A minimalist dataframe.

    Constructs a `Tafra` from `dict` of data and (optionally)
    dtypes. Types on parameters are the types of the constructed `Tafra`,
    but attempts are made to parse anything that "looks" like the correct data
    structure, including `Iterable`, `Iterator`, `Sequence`,
    and `Mapping` and various combinations.

    Parameters are given as an `InitVar`, defined as::

        InitVar = (
            tuple[str, Any] | _Mapping | Sequence[_Element]
            | Iterable[_Element] | Iterator[_Element] | enumerate
        )
        _Mapping = Mapping[str, Any] | Mapping[int, Any] | Mapping[float, Any] | Mapping[bool, Any]
        _Element = tuple[str | int | float | np.ndarray, Any] | list[Any] | Mapping

    Attributes
    ----------
    data: InitVar
        The data of the Tafra.

    dtypes: InitVar
        The dtypes of the columns.

    validate: bool = True
        Run validation checks of the data. False will improve performance, but `data` and `dtypes`
        will not be validated for conformance to expected data structures.

    check_rows: bool = True
        Run row count checks. False will allow columns of differing lengths, which may break several
        methods.

    """

    data: dc.InitVar[InitVar]
    dtypes: dc.InitVar[InitVar | None] = None
    validate: dc.InitVar[bool] = True
    check_rows: bool = True

    _data: dict[str, np.ndarray[Any, Any]] = dc.field(init=False)
    _dtypes: dict[str, str] = dc.field(init=False)

    if TYPE_CHECKING:

        def __init__(
            self,
            data: InitVar,
            dtypes: InitVar | None = None,
            validate: bool = True,
            check_rows: bool = True,
        ) -> None: ...

    def __post_init__(self, data: InitVar, dtypes: InitVar | None, validate: bool) -> None:
        # TODO: enable this?
        # if isinstance(self._data, DataFrame):
        #     tf = self.from_dataframe(df=self._data)
        #     self._data = tf._data
        #     self._dtypes = tf._dtypes
        #     self._rows = tf._rows
        #     return

        rows: int | None = None

        if validate:
            # check that the structure is actually a dict
            self._data = self._check_initvar(data)
            if dtypes is None or isinstance(dtypes, property):
                self._dtypes = {}
            else:
                self._dtypes = cast(dict[str, str], self._check_initvar(dtypes))

            # check that the values are properly formed np.ndarray
            for column, value in self._data.items():
                self._ensure_valid(column, value, check_rows=False)

                n_rows = len(self._data[column])
                if rows is None:
                    rows = n_rows

                if self.check_rows and rows != n_rows:
                    raise ValueError("`Tafra` must have consistent row counts.")
                elif rows < n_rows:  # pragma: no cover
                    rows = n_rows

            if rows is None:
                raise ValueError("No data provided in constructor statement.")

            self.update_dtypes_inplace(self._dtypes, _from_init=True)
            # must coalesce all dtypes immediately, other functions assume a
            # proper structure of the Tafra
            self._coalesce_dtypes()

        else:
            self._data = cast(dict[str, np.ndarray[Any, Any]], data)
            if dtypes is None or isinstance(dtypes, property):
                self._dtypes = {}
                self._coalesce_dtypes()
            else:
                self._dtypes = cast(dict[str, str], dtypes)

        self._update_rows()

    def _check_initvar(self, values: InitVar) -> dict[str, Any]:
        """
        Pre-process an `InitVar` into a `Dict`.
        """
        _values: dict[Any, Any]

        if isinstance(values, (Mapping, dict)):
            _values = cast(dict[str, Any], values)

        elif isinstance(values, Sequence):
            _values = self._parse_sequence(values)  # type: ignore[arg-type]

        elif isinstance(values, (Iterator, enumerate)):
            _values = self._parse_iterator(cast(Iterator[_Element], values))

        elif isinstance(values, Iterable):
            _values = self._parse_iterable(values)

        else:
            # last ditch attempt
            _values = cast(dict[Any, Any], values)  # type: ignore[unreachable]

        if not isinstance(_values, dict):
            raise TypeError(
                "Must contain `Dict`, `Mapping`, `Sequence`, Iterable, or Iterator, "
                f"got `{type(_values)}`"
            )

        # cast all keys to strings if they are not
        # must copy first as mutating the dict changes next(iterator)
        columns = [c for c in _values.keys() if not isinstance(c, str)]
        for column in columns:
            _values[str(column)] = _values.pop(column)

        return _values

    def _parse_sequence(self, values: Sequence[_Element]) -> dict[Any, Any]:
        """
        Pre-Process a `Sequence` `InitVar` into a `Dict`.
        """
        head = values[0]
        if isinstance(head, dict):
            _values = {}
            for _dict in values:
                _values.update(cast(dict[Any, Any], _dict))

        # maybe a Sequence of 2-tuples or 2-lists? Cast and try it.
        elif isinstance(head, Sequence) and len(head) == 2:
            # is the key an ndarray? turn it into a scalar
            if isinstance(head[0], np.ndarray) and len(np.atleast_1d(head[0])) == 1:
                # mypy doesn't get that we've checked the head of values as an ndarray
                _values = {
                    key.item(): value
                    for key, value in cast(Iterable[tuple[np.ndarray[Any, Any], Any]], values)
                }
            else:
                _values = dict(cast(Iterable[tuple[Any, Any]], values))

        else:
            raise TypeError(
                f"Sequence must contain `Dict`, `Mapping`, or `Sequence`, got `{type(head)}`"
            )

        return _values

    def _parse_iterable(self, values: Iterable[_Element]) -> dict[Any, Any]:
        """
        Pre-Process a `Iterable` `InitVar` into a `Dict`.
        """
        iter_values = iter(values)
        head = next(iter_values)
        if isinstance(head, dict):
            _values = dict(head)
            for _dict in iter_values:
                _values.update(cast(dict[Any, Any], _dict))

        # maybe an Iterable of 2-tuples or 2-lists? Cast and try it.
        elif isinstance(head, Sequence) and len(head) == 2:
            # is the key an ndarray? turn it into a scalar
            if isinstance(head[0], np.ndarray) and len(np.atleast_1d(head[0])) == 1:
                # mypy doesn't get that we've checked the head of values as an ndarray
                _values = {
                    key.item(): value
                    for key, value in chain(
                        cast(Iterable[tuple[np.ndarray[Any, Any], Any]], [head]),
                        cast(Iterator[tuple[np.ndarray[Any, Any], Any]], iter_values),
                    )
                }
            else:
                _values = dict(
                    chain(
                        cast(Iterable[tuple[Any, Any]], [head]),
                        cast(Iterator[tuple[Any, Any]], iter_values),
                    )
                )

        else:
            raise TypeError(
                f"Iterable must contain `Dict`, `Mapping`, or `Sequence`, got `{type(head)}`"
            )

        return _values

    def _parse_iterator(self, values: Iterator[_Element]) -> dict[Any, Any]:
        """
        Pre-Process a `Iterator` `InitVar` into a `Dict`.
        """
        head = next(values)

        if isinstance(head, dict):
            # consume the iterator if its a dict
            _values = dict(head)
            for _dict in values:
                _values.update(cast(dict[Any, Any], _dict))

            # maybe an Iterator of 2-tuples or 2-lists? Cast and try it.
        elif isinstance(head, Sequence) and len(head) == 2:
            # is the key an ndarray? turn it into a scalar
            if isinstance(head[0], np.ndarray) and len(np.atleast_1d(head[0])) == 1:
                # mypy doesn't get that we've checked the head of values as an ndarray
                _values = {
                    key.item(): value
                    for key, value in chain(
                        cast(Iterable[tuple[np.ndarray[Any, Any], Any]], [head]),
                        cast(Iterator[tuple[np.ndarray[Any, Any], Any]], values),
                    )
                }
            else:
                _values = dict(
                    chain(
                        cast(Iterable[tuple[Any, Any]], [head]),
                        cast(Iterator[tuple[Any, Any]], values),
                    )
                )

        else:
            raise TypeError(
                f"Iterator must contain `Dict`, `Mapping`, or `Sequence`, got `{type(head)}`"
            )

        return _values

    def __getitem__(
        self, item: (str | int | slice | Sequence[str | int | bool] | np.ndarray[Any, Any])
    ) -> Any:
        # return type is actually np.ndarray | 'Tafra' but mypy requires user to type check
        # in either case, what we return is a "slice" of the `Tafra`
        if isinstance(item, str):
            return self._data[item]

        elif isinstance(item, int):
            return self._iindex(item)

        elif isinstance(item, slice):
            return self._slice(item)

        elif isinstance(item, np.ndarray):
            return self._ndindex(item)

        elif isinstance(item, Sequence):
            if isinstance(item[0], str):
                return self.select(cast(Sequence[str], item))
            else:
                return self._aindex(cast(Sequence[int | bool], item))

        else:
            raise TypeError(f"Type {type(item)} not supported.")

    def __setitem__(self, item: str, value: np.ndarray[Any, Any] | Sequence[Any] | Any) -> None:
        self._ensure_valid(item, value, set_item=True)

    def __repr__(self) -> str:
        if not hasattr(self, "_rows"):
            return f"Tafra(data={self._data}, dtypes={self._dtypes}, rows=n/a)"
        return f"Tafra(data={self._data}, dtypes={self._dtypes}, rows={self._rows})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        assert self._data is not None, "Internal error: Cannot construct a Tafra with no data."
        return self._rows

    def __iter__(self) -> Iterator["Tafra"]:
        return (self._iindex(i) for i in range(self._rows))

    def __rshift__(self, other: Callable[["Tafra"], "Tafra"]) -> "Tafra":
        return self.pipe(other)

    def iterrows(self) -> Iterator["Tafra"]:
        """
        Yield rows as `Tafra`. Use `itertuples()` for better performance.

        Returns
        -------
        tafras: Iterator[Tafra]
            An iterator of `Tafra`.
        """
        yield from self.__iter__()

    def itertuples(self, name: str | None = "Tafra") -> Iterator[tuple[Any, ...]]:
        """
        Yield rows as `NamedTuple`, or if `name` is `None`, yield
        rows as `tuple`.

        Parameters
        ----------
        name: str | None = 'Tafra'
            The name for the `NamedTuple`. If `None`, construct a
            `Tuple` instead.

        Returns
        -------
        tuples: Iterator[Namedtuple[Any, ...]]
            An iterator of `NamedTuple`.
        """
        if name is None:
            return (tuple(values) for values in zip(*self._data.values()))

        TafraNT = namedtuple(name, self._data.keys())  # type: ignore
        return map(TafraNT._make, zip(*self._data.values()))

    def itercols(self) -> Iterator[tuple[str, np.ndarray[Any, Any]]]:
        """
        Yield columns as `tuple[str, np.ndarray]`, where the `str` is the column
        name.

        Returns
        -------
        tuples: Iterator[tuple[str, np.ndarray]]
            An iterator of `Tafra`.
        """
        return map(tuple, self.data.items())  # type: ignore

    def _update_rows(self) -> None:
        """
        Updates `_rows`. User should call this if they have directly assigned to
        `_data` and need to validate the `Tafra`.
        """
        iter_values = iter(self._data.values())
        self._rows = len(next(iter_values))
        if self.check_rows and not all(len(v) == self._rows for v in iter_values):
            raise TypeError("Uneven length of data.")

    def _slice(self, _slice: slice) -> "Tafra":
        """
        Use a `slice` to slice the `Tafra`.

        Parameters
        ----------
        _slice: slice
            The `slice` object.

        Returns
        -------
        tafra: Tafra
            The sliced `Tafra`.
        """
        return Tafra(
            {column: np.atleast_1d(value[_slice]) for column, value in self._data.items()},
            self._dtypes,
            validate=False,
        )

    def _iindex(self, index: int) -> "Tafra":
        """
        Use a `int` to slice the `Tafra`.

        Parameters
        ----------
        index: int

        Returns
        -------
        tafra: Tafra
            The sliced `Tafra`.
        """
        return Tafra(
            {column: value[[index]] for column, value in self._data.items()},
            self._dtypes,
            validate=False,
        )

    def _aindex(self, index: Sequence[int | bool]) -> "Tafra":
        """
        Use numpy advanced indexing to slice the `Tafra`.

        Parameters
        ----------
        index: Sequence[int | bool]

        Returns
        -------
        tafra: Tafra
            The sliced `Tafra`.
        """
        return Tafra(
            {column: value[index] for column, value in self._data.items()},
            self._dtypes,
            validate=False,
        )

    def _ndindex(self, index: np.ndarray[Any, Any]) -> "Tafra":
        """
        Use `numpy.ndarray` indexing to slice the `Tafra`.

        Parameters
        ----------
        index: np.ndarray

        Returns
        -------
        tafra: Tafra
            The sliced `Tafra`.
        """
        if index.ndim != 1:
            raise IndexError(f"Indexing np.ndarray must ndim == 1, got ndim == {index.ndim}")

        return Tafra(
            {column: value[index] for column, value in self._data.items()},
            self._dtypes,
            validate=False,
        )

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """
        A dunder method for IPython to pretty print.

        Parameters
        ----------
        p: IPython.lib.pretty.RepresentationPrinter
            IPython provides this class to handle the object representation.

        cycle: bool
            IPython has detected an infinite loop. Print an alternative represenation
            and return.

        Returns
        -------
        None
            Calls p.text and returns.
        """
        if cycle:
            p.text("Tafra(...)")
        else:
            p.text(self._pretty_format(lambda s: " " + pprint.pformat(s, indent=1)[1:].strip()))

    def _repr_html_(self) -> str:
        """
        a dunder method for Jupyter Notebook to print HTML.
        """
        return self.to_html()

    def _pretty_format(self, formatter: Callable[[object], str]) -> str:
        """
        Format _data and _dtypes for pretty printing.

        Parameters
        ----------
        formatter: Callable[[object], str]
            A formatter that operates on the _data and _dtypes `dict`.

        Returns
        -------
        string: str
            The formatted string for printing.
        """
        PATTERN = r"(, dtype=[a-z]+)(?=\))"

        return "\n".join(
            [
                "Tafra(data = {",
                f"{re.sub(PATTERN, '', formatter(self._data))},",
                "dtypes = {",
                f"{re.sub(PATTERN, '', formatter(self._dtypes))},",
                f"rows = {self._rows})",
            ]
        )

    def pformat(
        self, indent: int = 1, width: int = 80, depth: int | None = None, compact: bool = False
    ) -> str:
        """
        Format for pretty printing. Parameters are passed to
        `pprint.PrettyPrinter`.

        Parameters
        ----------
        indent: int
            Number of spaces to indent for each level of nesting.

        width: int
            Attempted maximum number of columns in the output.

        depth: int | None
            The maximum depth to print out nested structures.

        compact: bool
            If true, several items will be combined in one line.

        Returns
        -------
        formatted string: str
            A formatted string for pretty printing.
        """
        return self._pretty_format(
            lambda s: (
                indent * " " + pprint.pformat(s, indent, width, depth, compact=compact)[1:].strip()
            )
        )

    def pprint(
        self, indent: int = 1, width: int = 80, depth: int | None = None, compact: bool = False
    ) -> None:
        """
        Pretty print. Parameters are passed to `pprint.PrettyPrinter`.

        Parameters
        ----------
        indent: int
            Number of spaces to indent for each level of nesting.

        width: int
            Attempted maximum number of columns in the output.

        depth: int | None
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
        return "<thead>\n<tr>\n{th}\n</tr>\n</thead>".format(
            th="\n".join(f"<th>{c}</th>" for c in columns)
        )

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
        return "<tr>\n{td}\n</tr>".format(td="\n".join(f"<td>{td}</td>" for td in row))

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
        return "<tbody>\n{tr}\n</tbody>".format(tr="\n".join(tr))

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
        return f"<table>\n{thead}\n{tbody}\n</table>"

    def to_html(self, n: int = 20) -> str:
        """
        Construct an HTML table representation of the `Tafra` data.

        Parameters
        ----------
        n: int = 20
            Number of items to print.

        Returns
        -------
        HTML: str
            The HTML table representation.
        """
        thead = self._html_thead(chain([""], self._data.keys()))
        tr = chain(
            [
                self._html_tr(
                    chain(["dtype"], (self._dtypes[column] for column in self._data.keys()))
                )
            ],
            (
                self._html_tr(chain([i], (v[i] for v in self._data.values())))
                for i in range(min(n, self._rows))
            ),
        )
        tbody = self._html_tbody(tr)
        return self._html_table(thead, tbody)

    def _ensure_valid(
        self,
        column: str,
        value: np.ndarray[Any, Any] | Sequence[Any] | Any,
        check_rows: bool = True,
        set_item: bool = False,
    ) -> None:
        """
        Validate values as an `np.ndarray` of equal length to `rows` before
        assignment. Will attempt to create a `np.ndarray` if `value` is not one
        already, and will check that `np.ndarray.ndim` `== 1`. If
        `np.ndarray.ndim` `> 1` it will attempt `np.squeeze()` on `value`.

        Parameters
        ----------
        column: str
            The column to assign to.

        value: np.ndarray | Sequence[Any] | Any
            The value to be assigned.

        Returns
        -------
        None: None
        """
        _type = type(value).__name__
        id_value = id(value)
        rows = self._rows if check_rows else 1

        if value is None:
            value = np.full(rows, value)

        elif isinstance(value, np.ndarray):
            if value.ndim == 0:
                value = np.full(rows, value.item())
            elif value.ndim == 1 and value.shape[0] == 1 and rows > 1:
                value = np.full(rows, value)

        elif isinstance(value, str):
            _sd = np.dtypes.StringDType(na_object=None)  # type: ignore[call-arg]
            value = np.full(rows, value, dtype=_sd)

        elif isinstance(value, Iterator):
            value = np.asarray(tuple(value))

        elif isinstance(value, Iterable):
            value = np.asarray(value)

        elif not isinstance(value, Sized):  # type: ignore[unreachable]
            value = np.full(rows, value)

        assert isinstance(value, np.ndarray), (
            "Internal error: `Tafra` only supports assigning `ndarray`."
        )

        if value.ndim > 1:
            sq_value = value.squeeze()
            if sq_value.ndim > 1:
                raise ValueError("`ndarray` or `np.squeeze(ndarray)` must have ndim == 1.")
            elif sq_value.ndim == 1:
                # if value was a single item, squeeze returns zero length item
                warnings.warn("`np.squeeze(ndarray)` applied to set ndim == 1.")
                value = sq_value

        assert value.ndim >= 1, "Internal error: `Tafra` only supports assigning ndim == 1."

        if check_rows and len(value) != rows:
            raise ValueError(
                "`Tafra` must have consistent row counts.\n"
                f"This `Tafra` has {rows} rows. Assigned {_type} has {len(value)} rows."
            )

        # special parsing of various object types
        parsed_value = object_formatter.parse_dtype(value)
        if parsed_value is not None:
            value = parsed_value

        # have we modified value?
        if set_item or id(value) != id_value:
            self._data[column] = value
            self._dtypes[column] = self._format_dtype(value.dtype)

    def parse_object_dtypes(self) -> "Tafra":
        """
        Parse the object dtypes using the `ObjectFormatter` instance.
        """
        tafra = self.copy()
        tafra.parse_object_dtypes_inplace()
        return tafra

    def parse_object_dtypes_inplace(self) -> None:
        """
        Inplace version.

        Parse the object dtypes using the `ObjectFormatter` instance.
        """
        for column, value in self._data.items():
            parsed_value = object_formatter.parse_dtype(value, convert_strings=True)
            if parsed_value is not None:
                self._data[column] = parsed_value
                self._dtypes[column] = self._format_dtype(parsed_value.dtype)

    def _validate_columns(self, columns: Iterable[str]) -> None:
        """
        Validate that the column name(s) exists in `_data`.

        Parameters
        ----------
        columns: Iterable[str]
            The column names to validate.

        Returns
        -------
        None: None
        """
        for column in columns:
            if column not in self._data:
                raise ValueError(f"Column {column} does not exist in `tafra`.")

    def _validate_dtypes(self, dtypes: dict[str, Any]) -> dict[str, str]:
        """
        Validate that the dtypes as internally used names and that the columns exists in
        `_data`.

        Parameters
        ----------
        dtypes: dict[str, Any]
            The dtypes to validate.

        Returns
        -------
        dtypes: dict[str, str]
            The validated types.
        """

        self._validate_columns(dtypes.keys())
        return {column: self._format_dtype(dtype) for column, dtype in dtypes.items()}

    @staticmethod
    def _format_dtype(dtype: Any) -> str:
        """
        Parse a dtype into the internally used string representation, if defined.
        Otherwise, pass through and let numpy raise error if it is not a valid dtype.

        Parameters
        ----------
        dtype: Any
            The dtype to parse.

        Returns
        -------
        dtype: str
            The parsed dtype.
        """
        _dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        name = _dtype.type.__name__
        if "str" in name:
            return "str"

        return name.replace("_", "")

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
        m = re.search(r"([a-z]+)", name)
        if m:
            return m.group(1)

        # are there any dtypes without text names?
        return name  # pragma: no cover

    @classmethod
    def from_records(
        cls,
        records: Iterable[Iterable[Any]],
        columns: Iterable[str],
        dtypes: Iterable[Any] | None = None,
        **kwargs: Any,
    ) -> "Tafra":
        """
        Construct a `Tafra` from an Iterator of records, e.g. from a SQL query. The
        records should be a nested Iterable, but can also be fed a cursor method such as
        `cur.fetchmany()` or `cur.fetchall()`.

        Parameters
        ----------
        records: ITerable[Iteralble[str]]
            The records to turn into a `Tafra`.

        columns: Iterable[str]
            The column names to use.

        dtypes: Iterable[Any] | None = None
            The dtypes of the columns.

        Returns
        -------
        tafra: Tafra
            The constructed `Tafra`.
        """
        if dtypes is None:
            return Tafra({column: value for column, value in zip(columns, zip(*records))}, **kwargs)

        return Tafra(
            {column: value for column, value in zip(columns, zip(*records))},
            {column: value for column, value in zip(columns, dtypes)},
            **kwargs,
        )

    @classmethod
    def from_series(cls, s: Series, dtype: str | None = None, **kwargs: Any) -> "Tafra":
        """
        Construct a `Tafra` from a `pandas.Series`. If `dtype` is not
        given, take from `pandas.Series.dtype`.

        Parameters
        ----------
        s: pandas.Series
            The series used to build the `Tafra`.

        dtype: str | None = None
            The dtypes of the column.

        Returns
        -------
        tafra: Tafra
            The constructed `Tafra`.
        """
        if dtype is None:
            dtype = s.dtype
        dtypes = {s.name: cls._format_dtype(dtype)}

        return cls({s.name: s.values.astype(dtypes[s.name])}, dtypes, **kwargs)

    @classmethod
    def from_dataframe(
        cls, df: DataFrame, dtypes: dict[str, Any] | None = None, **kwargs: Any
    ) -> "Tafra":
        """
        Construct a `Tafra` from a `pandas.DataFrame`. If `dtypes` are not
        given, take from `pandas.DataFrame.dtypes`.

        Parameters
        ----------
        df: pandas.DataFrame
            The dataframe used to build the `Tafra`.

        dtypes: dict[str, Any] | None = None
            The dtypes of the columns.

        Returns
        -------
        tafra: Tafra
            The constructed `Tafra`.
        """
        if dtypes is None:
            dtypes = {c: t for c, t in zip(df.columns, df.dtypes)}
        dtypes = {c: cls._format_dtype(t) for c, t in dtypes.items()}

        return cls(
            {c: df[c].values.astype(dtypes[c]) for c in df.columns},
            {c: dtypes[c] for c in df.columns},
            **kwargs,
        )

    @classmethod
    def read_sql(
        cls,
        query: str,
        cur: Cursor,
        params: Sequence[Any] | None = None,
    ) -> "Tafra":
        """
        Execute a SQL SELECT statement using a `pyodbc.Cursor` and return a
        `Tafra` of the result set.

        Parameters
        ----------
        query: str
            The SQL query, optionally containing ``?`` placeholders.

        cur: pyodbc.Cursor
            The `pyodbc` cursor.

        params: Sequence[Any] | None
            Optional sequence of parameter values to bind to ``?`` placeholders
            in the query.  Passed unchanged to ``cur.execute``.

        Returns
        -------
        tafra: Tafra
            The constructed `Tafra`.
        """
        if params is None:
            cur.execute(query)
        else:
            cur.execute(query, params)

        columns, dtypes = zip(*((d[0], d[1]) for d in cur.description))

        head = cur.fetchone()
        if head is None:
            return Tafra({column: () for column in columns})

        return Tafra.from_records(chain([head], cur.fetchall()), columns, dtypes)

    @classmethod
    def read_sql_chunks(
        cls,
        query: str,
        cur: Cursor,
        chunksize: int = 100,
        params: Sequence[Any] | None = None,
    ) -> Iterator["Tafra"]:
        """
        Execute a SQL SELECT statement using a `pyodbc.Cursor` and yield
        chunks of the result set as `Tafra` instances.

        Parameters
        ----------
        query: str
            The SQL query, optionally containing ``?`` placeholders.

        cur: pyodbc.Cursor
            The `pyodbc` cursor.

        chunksize: int
            Number of rows to yield per chunk.

        params: Sequence[Any] | None
            Optional sequence of parameter values to bind to ``?`` placeholders
            in the query.  Passed unchanged to ``cur.execute``.

        Yields
        ------
        tafra: Tafra
            A `Tafra` containing up to `chunksize` rows of the result set.
        """
        if params is None:
            cur.execute(query)
        else:
            cur.execute(query, params)

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
    def read_csv(
        cls,
        csv_file: str | Path | TextIOWrapper | IO[str],
        guess_rows: int = 5,
        missing: str | None = "",
        dtypes: dict[str, Any] | None = None,
        **csvkw: dict[str, Any],
    ) -> "Tafra":
        """
        Read a CSV file with a header row, infer the types of each column,
        and return a Tafra containing the file's contents.

        Parameters
        ----------
        csv_file: str | TextIOWrapper
            The path to the CSV file, or an open file-like object.

        guess_rows: int
            The number of rows to use when guessing column types.

        dtypes: dict[str, str] | None
            dtypes by column name; by default, all dtypes will be inferred
            from the file contents.

        **csvkw: dict[str, Any]
            Additional keyword arguments passed to csv.reader.

        Returns
        -------
        tafra: Tafra
            The constructed `Tafra`.
        """
        reader = CSVReader(cast(str | Path | TextIOWrapper, csv_file), guess_rows, missing, **csvkw)
        return Tafra(reader.read(), dtypes=dtypes)

    @classmethod
    def as_tafra(
        cls, maybe_tafra: Tafra | DataFrame | Series | dict[str, Any] | Any
    ) -> Tafra | None:
        """
        Returns the unmodified `tafra`` if already a `Tafra`, else construct a
        `Tafra` from known types or subtypes of `DataFrame` or `dict`.
        Structural subtypes of `DataFrame` or `Series` are also valid,
        as are classes that have `cls.__name__ == 'DataFrame'` or
        `cls.__name__ == 'Series'`.

        Parameters
        ----------
        maybe_tafra: 'tafra' | DataFrame
            The object to ensure is a `Tafra`.

        Returns
        -------
        tafra: Tafra | None
            The `Tafra`, or None is `maybe_tafra` is an unknown
            type.
        """
        if isinstance(maybe_tafra, Tafra):
            return maybe_tafra

        elif isinstance(maybe_tafra, Series):  # pragma: no cover
            return cls.from_series(maybe_tafra)

        elif type(maybe_tafra).__name__ == "Series":  # pragma: no cover
            return cls.from_series(cast(Series, maybe_tafra))

        elif isinstance(maybe_tafra, DataFrame):  # pragma: no cover
            return cls.from_dataframe(maybe_tafra)

        elif type(maybe_tafra).__name__ == "DataFrame":  # pragma: no cover
            return cls.from_dataframe(cast(DataFrame, maybe_tafra))

        elif isinstance(maybe_tafra, dict):
            return cls(maybe_tafra)

        raise TypeError(f"Unknown type `{type(maybe_tafra)}` for conversion to `Tafra`")

    @property
    def columns(self) -> tuple[str, ...]:
        """
        The names of the columns. Equivalent to `Tafra`.keys().

        Returns
        -------
        columns: tuple[str, ...]
            The column names.
        """
        return tuple(self._data.keys())

    @columns.setter
    def columns(self, value: Any) -> None:
        raise ValueError("Assignment to `columns` is forbidden.")

    @property
    def rows(self) -> int:
        """
        The number of rows of the first item in `data`. The `len()`
        of all items have been previously validated.

        Returns
        -------
        rows: int
            The number of rows of the `Tafra`.
        """
        return self.__len__()

    @rows.setter
    def rows(self, value: Any) -> None:
        raise ValueError("Assignment to `rows` is forbidden.")

    @property  # type: ignore
    def data(self) -> dict[str, np.ndarray[Any, Any]]:
        """
        The `Tafra` data.

        Returns
        -------
        data: dict[str, np.ndarray]
            The data.
        """
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        raise ValueError("Assignment to `data` is forbidden.")

    @property  # type: ignore
    def dtypes(self) -> dict[str, str]:
        """
        The `Tafra` dtypes.

        Returns
        -------
        dtypes: dict[str, str]
            The dtypes.
        """
        return self._dtypes

    @dtypes.setter
    def dtypes(self, value: Any) -> None:
        raise ValueError("Assignment to `dtypes` is forbidden.")

    @property
    def size(self) -> int:
        """
        The `Tafra` size.

        Returns
        -------
        size: int
            The size.
        """
        return self.rows * len(self.columns)

    @size.setter
    def size(self, value: Any) -> None:
        raise ValueError("Assignment to `size` is forbidden.")

    @property
    def ndim(self) -> int:
        """
        The `Tafra` number of dimensions.

        Returns
        -------
        ndim: int
            The number of dimensions.
        """
        return 2

    @ndim.setter
    def ndim(self, value: Any) -> None:
        raise ValueError("Assignment to `ndim` is forbidden.")

    @property
    def shape(self) -> tuple[int, int]:
        """
        The `Tafra` shape.

        Returns
        -------
        shape: int
            The shape.
        """
        return self.rows, len(self.columns)

    @shape.setter
    def shape(self, value: Any) -> None:
        raise ValueError("Assignment to `shape` is forbidden.")

    def row_map(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Iterator[Any]:
        """
        Map a function over rows. To apply to specific columns, use `select()`
        first. The function must operate on `Tafra`.

        Parameters
        ----------
        fn: Callable[..., Any]
            The function to map.

        *args: Any
            Additional positional arguments to `fn`.

        **kwargs: Any
            Additional keyword arguments to `fn`.

        Returns
        -------
        iter_tf: Iterator[Any]
            An iterator to map the function.
        """
        return (fn(tf, *args, **kwargs) for tf in self.__iter__())

    def tuple_map(
        self, fn: Callable[..., Any], *args: Any, name: str | None = "Tafra", **kwargs: Any
    ) -> Iterator[Any]:
        """
        Map a function over rows. This is faster than `row_map()`. To apply to
        specific columns, use `select()` first.

        When `name` is `'Tafra'` (default), the function receives
        `NamedTuple` rows with attribute access (e.g. `r.col`).
        When `name` is `None`, rows are passed as plain `tuple`
        for faster iteration — avoids NamedTuple construction overhead.

        Parameters
        ----------
        fn: Callable[..., Any]
            The function to map.

        *args: Any
            Additional positional arguments to `fn`.

        name: str | None = 'Tafra'
            The name for the `NamedTuple`. If `None`, use plain
            tuples for ~2--4x faster iteration.

        **kwargs: Any
            Additional keyword arguments to `fn`.

        Returns
        -------
        iter_tf: Iterator[Any]
            An iterator to map the function.
        """
        if name is None:
            return (fn(row, *args, **kwargs) for row in zip(*self._data.values()))
        return (fn(tf, *args, **kwargs) for tf in self.itertuples(name))

    def col_map(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Iterator[Any]:
        """
        Map a function over columns. To apply to specific columns, use `select()`
        first. The function must operate on `tuple[str, np.ndarray]`.

        Parameters
        ----------
        fn: Callable[..., Any]
            The function to map.

        *args: Any
            Additional positional arguments to `fn`.

        **kwargs: Any
            Additional keyword arguments to `fn`.

        Returns
        -------
        iter_tf: Iterator[Any]
            An iterator to map the function.
        """

        return (fn(value, *args, **kwargs) for column, value in self.itercols())

    def key_map(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Iterator[tuple[str, Any]]:
        """
        Map a function over columns like `col_map()`, but return `tuple` of the
        key with the function result. To apply to specific columns, use `select()`
        first. The function must operate on `tuple[str, np.ndarray]`.

        Parameters
        ----------
        fn: Callable[..., Any]
            The function to map.

        *args: Any
            Additional positional arguments to `fn`.

        **kwargs: Any
            Additional keyword arguments to `fn`.

        Returns
        -------
        iter_tf: Iterator[Any]
            An iterator to map the function.
        """
        return ((column, fn(value, *args, **kwargs)) for column, value in self.itercols())

    def pipe(
        self, fn: Callable[Concatenate["Tafra", P], "Tafra"], *args: Any, **kwargs: Any
    ) -> "Tafra":
        """
        Apply a function to the `Tafra` and return the resulting `Tafra`. Primarily
        used to build a transformer pipeline.

        Parameters
        ----------
        fn: Callable[[], 'Tafra']
            The function to apply.

        *args: Any
            Additional positional arguments to `fn`.

        **kwargs: Any
            Additional keyword arguments to `fn`.

        Returns
        -------
        tafra: Tafra
            A new `Tafra` result of the function.
        """
        return fn(self, *args, **kwargs)

    def select(self, columns: Iterable[str]) -> "Tafra":
        """
        Use column names to slice the `Tafra` columns analogous to SQL SELECT.
        This does not copy the data. Call `copy()` to obtain a copy of the sliced
        data.

        Parameters
        ----------
        columns: Iterable[str]
            The column names to slice from the `Tafra`.

        Returns
        -------
        tafra: Tafra
            the `Tafra` with the sliced columns.
        """
        if isinstance(columns, str):
            columns = [columns]
        self._validate_columns(columns)

        return Tafra(
            {column: self._data[column] for column in columns},
            {column: self._dtypes[column] for column in columns},
            validate=False,
        )

    def head(self, n: int = 5) -> "Tafra":
        """
        Display the head of the `Tafra`.

        Parameters
        ----------
        n: int = 5
            The number of rows to display.

        Returns
        -------
        None: None
        """
        return self._slice(slice(n))

    def tail(self, n: int = 5) -> "Tafra":
        """
        Return the last `n` rows.

        Parameters
        ----------
        n: int = 5
            The number of rows to return.

        Returns
        -------
        tafra: Tafra
            The last `n` rows.
        """
        return self._slice(slice(-n, None))

    def sort(self, columns: str | Iterable[str], reverse: bool = False) -> "Tafra":
        """
        Return a new `Tafra` sorted by the given columns.

        Parameters
        ----------
        columns: str | Iterable[str]
            Column(s) to sort by. First column is the primary sort key.

        reverse: bool = False
            Sort in descending order.

        Returns
        -------
        tafra: Tafra
            The sorted `Tafra`.
        """
        if isinstance(columns, str):
            columns = [columns]
        result = self._sorted(columns)
        if reverse:
            return result._slice(slice(None, None, -1))
        return result

    def sample(self, n: int, seed: int | None = None) -> "Tafra":
        """
        Return a random sample of `n` rows.

        Parameters
        ----------
        n: int
            Number of rows to sample.

        seed: int | None
            Random seed for reproducibility.

        Returns
        -------
        tafra: Tafra
            The sampled `Tafra`.
        """
        if n > self._rows:
            raise ValueError(f"Cannot sample {n} rows from {self._rows} rows.")
        rng = np.random.default_rng(seed)
        idx = rng.choice(self._rows, size=n, replace=False)
        return self._ndindex(idx)

    def drop_duplicates(self, columns: Iterable[str] | None = None) -> "Tafra":
        """
        Remove duplicate rows, keeping the first occurrence.

        Parameters
        ----------
        columns: Iterable[str] | None
            Columns to check for duplicates. If `None`, use all columns.

        Returns
        -------
        tafra: Tafra
            The deduplicated `Tafra`.
        """
        cols = list(columns) if columns is not None else list(self._data.keys())
        self._validate_columns(cols)

        if len(cols) == 1:
            _, idx = np.unique(self._data[cols[0]], return_index=True)
        else:
            col_arrays = [self._data[c] for c in cols]
            # check for StringDType — can't use structured array
            if any(c.dtype.kind == "T" for c in col_arrays):
                seen: dict[Any, int] = {}
                idx_list: list[int] = []
                for i, k in enumerate(zip(*col_arrays)):
                    if k not in seen:
                        seen[k] = i
                        idx_list.append(i)
                idx = np.array(idx_list, dtype=np.intp)
            else:
                dt = np.dtype([(f"f{i}", c.dtype) for i, c in enumerate(col_arrays)])
                key = np.empty(self._rows, dtype=dt)
                for i, c in enumerate(col_arrays):
                    key[f"f{i}"] = c
                _, idx = np.unique(key, return_index=True)

        idx.sort()
        return self._ndindex(idx)

    def value_counts(self, column: str) -> "Tafra":
        """
        Count occurrences of each unique value in a column.

        Parameters
        ----------
        column: str
            The column to count.

        Returns
        -------
        tafra: Tafra
            A `Tafra` with columns `column` and `'count'`,
            sorted by count descending.
        """
        self._validate_columns([column])
        data = self._data[column]
        unique, counts = np.unique(data, return_counts=True)
        order = np.argsort(-counts)
        return Tafra({column: unique[order], "count": counts[order]}, validate=False)

    def describe(self) -> "Tafra":
        """
        Summary statistics for numeric columns: count, mean, std, min,
        25%, 50%, 75%, max.

        Returns
        -------
        tafra: Tafra
            A `Tafra` with a `'stat'` column and one column per
            numeric column in the original.
        """
        stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        result: dict[str, np.ndarray[Any, Any]] = {
            "stat": np.array(stats, dtype=np.dtypes.StringDType(na_object=None))  # type: ignore[call-arg]
        }
        for col, val in self._data.items():
            if val.dtype.kind in ("i", "u", "f"):
                fval = val.astype(float)
                result[col] = np.array(
                    [
                        float(len(fval)),
                        np.mean(fval),
                        np.std(fval),
                        np.min(fval),
                        np.percentile(fval, 25),
                        np.percentile(fval, 50),
                        np.percentile(fval, 75),
                        np.max(fval),
                    ]
                )
        return Tafra(result, validate=False)

    def shift(self, n: int = 1) -> "Tafra":
        """
        Shift all columns by `n` rows. Positive shifts forward (lag),
        negative shifts backward (lead). Vacated rows are filled with
        `NaN` for numeric columns, `None` for object/string columns.

        .. note::

            Integer columns are cast to `float64` to accommodate `NaN`
            fill values, since numpy integer arrays cannot represent missing
            data. This matches `pandas` behavior.

        Parameters
        ----------
        n: int = 1
            Number of rows to shift. Positive = lag, negative = lead.

        Returns
        -------
        tafra: Tafra
            The shifted `Tafra`.
        """
        if n == 0:
            return self.copy()

        result: dict[str, np.ndarray[Any, Any]] = {}
        for col, val in self._data.items():
            if val.dtype.kind in ("i", "u", "f"):
                shifted = np.empty(self._rows, dtype=float)
                shifted[:] = np.nan
                if n > 0:
                    shifted[n:] = val[: self._rows - n]
                else:
                    shifted[: self._rows + n] = val[-n:]
                result[col] = shifted
            else:
                shifted_obj = np.empty(self._rows, dtype=object)
                shifted_obj[:] = None
                if n > 0:
                    shifted_obj[n:] = val[: self._rows - n]
                else:
                    shifted_obj[: self._rows + n] = val[-n:]
                result[col] = shifted_obj

        return Tafra(result, validate=False)

    def keys(self) -> KeysView[str]:
        """
        Return the keys of `data`, i.e. like `dict.keys()`.

        Returns
        -------
        data keys: KeysView[str]
            The keys of the data property.
        """
        return self._data.keys()

    def values(self) -> ValuesView[np.ndarray[Any, Any]]:
        """
        Return the values of `data`, i.e. like `dict.values()`.

        Returns
        -------
        data values: ValuesView[np.ndarray]
            The values of the data property.
        """
        return self._data.values()

    def items(self) -> ItemsView[str, np.ndarray[Any, Any]]:
        """
        Return the items of `data`, i.e. like `dict.items()`.

        Returns
        -------
        items: ItemsView[str, np.ndarray]
            The data items.
        """
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Return from the `get()` function of `data`, i.e. like
        `dict.get()`.

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

    def update(self, other: "Tafra") -> "Tafra":
        """
        Update the data and dtypes of this `Tafra` with another `Tafra`.
        Length of rows must match, while data of different `dtype` will overwrite.

        Parameters
        ----------
        other: Tafra
            The other `Tafra` from which to update.

        Returns
        -------
        None: None
        """
        tafra = self.copy()
        tafra.update_inplace(other)
        return tafra

    def update_inplace(self, other: "Tafra") -> None:
        """
        Inplace version.

        Update the data and dtypes of this `Tafra` with another `Tafra`.
        Length of rows must match, while data of different `dtype` will overwrite.

        Parameters
        ----------
        other: Tafra
            The other `Tafra` from which to update.

        Returns
        -------
        None: None
        """
        if not isinstance(other, Tafra):
            # should be a Tafra, but if not let's construct one
            other = Tafra(other)  # type: ignore

        rows = self._rows

        for column, value in other._data.items():
            if len(value) != rows:
                raise ValueError(
                    "Other `Tafra` must have consistent row count. "
                    f"This `Tafra` has {rows} rows, other `Tafra` has {len(value)} rows."
                )
            self._data[column] = value

        self.update_dtypes_inplace(other._dtypes)

    def _coalesce_dtypes(self) -> None:
        """
        Update `dtypes` with missing keys that exist in `data`.

        **Must be called if `data` or `data` is directly modified!**

        Returns
        -------
        None: None
        """
        for column in self._data.keys():
            if column not in self._dtypes:
                self._dtypes[column] = self._format_dtype(self._data[column].dtype)

    def update_dtypes(self, dtypes: dict[str, Any]) -> "Tafra":
        """
        Apply new dtypes.

        Parameters
        ----------
        dtypes: dict[str, Any]
            The dtypes to update. If `None`, create from entries in `data`.

        Returns
        -------
        tafra: Tafra | None
            The updated `Tafra`.
        """
        tafra = self.copy()
        tafra.update_dtypes_inplace(dtypes)
        return tafra

    def update_dtypes_inplace(self, dtypes: dict[str, Any], _from_init: bool = False) -> None:
        """
        Inplace version.

        Apply new dtypes.

        Parameters
        ----------
        dtypes: dict[str, Any]
            The dtypes to update. If `None`, create from entries in `data`.

        Returns
        -------
        None
            This method mutates the `Tafra` in place.
        """
        # Preserve raw numpy dtypes for casting, since _format_dtype
        # collapses e.g. <U and StringDType into the same 'str' label.
        raw_dtypes: dict[str, Any] = {}
        for column, dtype in dtypes.items():
            self._validate_columns([column])
            if isinstance(dtype, np.dtype):
                raw_dtypes[column] = dtype
            elif isinstance(dtype, str) and dtype == "str" and not _from_init:
                # 'str' label → StringDType with na_object=None so the
                # column can hold None values. Skip during __post_init__
                # to preserve the original dtype.
                raw_dtypes[column] = np.dtypes.StringDType(na_object=None)  # type: ignore[call-arg]
            else:
                try:
                    raw_dtypes[column] = np.dtype(dtype)
                except TypeError:
                    # StringDType() can't go through np.dtype(); keep as-is
                    raw_dtypes[column] = dtype

        formatted = self._validate_dtypes(dtypes)
        self._dtypes.update(formatted)

        for column, target_dtype in raw_dtypes.items():
            current_dtype = self._data[column].dtype
            # Skip when the target is the ambiguous np.dtype('str') (= <U0)
            # and the source is already a string type. This happens when
            # __post_init__ round-trips through formatted labels ('str').
            # Explicit StringDType() or specific <U widths are NOT skipped.
            if (
                isinstance(target_dtype, np.dtype)
                and target_dtype == np.dtype("str")
                and self._reduce_dtype(current_dtype) == "str"
            ):
                continue
            if current_dtype != target_dtype:
                try:
                    self._data[column] = self._data[column].astype(target_dtype)
                except ValueError:
                    REPL_VALS = [
                        "",
                    ]
                    col_data = self._data[column].astype(object)
                    for repl_val in REPL_VALS:
                        where_repl = np.equal(col_data, repl_val)
                        col_data[where_repl] = None
                    self._data[column] = col_data.astype(target_dtype)

    def rename(self, renames: dict[str, str]) -> "Tafra":
        """
        Rename columns in the `Tafra` from a `dict`.

        Parameters
        ----------
        renames: dict[str, str]
            The map from current names to new names.

        Returns
        -------
        tafra: Tafra | None
            The `Tafra` with update names.
        """

        tafra = self.copy()
        tafra.rename_inplace(renames)
        return tafra

    def rename_inplace(self, renames: dict[str, str]) -> None:
        """
        In-place version.

        Rename columns in the `Tafra` from a `dict`.

        Parameters
        ----------
        renames: dict[str, str]
            The map from current names to new names.

        Returns
        -------
        tafra: Tafra | None
            The `Tafra` with update names.
        """
        self._validate_columns(renames.keys())

        for cur, new in renames.items():
            self._data[new] = self._data.pop(cur)
            self._dtypes[new] = self._dtypes.pop(cur)
        return None

    def delete(self, columns: Iterable[str]) -> "Tafra":
        """
        Remove a column from `data` and `dtypes`.

        Parameters
        ----------
        columns: str
            The column to remove.

        Returns
        -------
        tafra: Tafra | None
            The `Tafra` with the deleted column.
        """
        if isinstance(columns, str):
            columns = [columns]
        self._validate_columns(columns)

        return Tafra(
            {column: value.copy() for column, value in self._data.items() if column not in columns},
            {column: value for column, value in self._dtypes.items() if column not in columns},
            validate=False,
        )

    def delete_inplace(self, columns: Iterable[str]) -> None:
        """
        In-place version.

        Remove a column from `data` and `dtypes`.

        Parameters
        ----------
        columns: str
            The column to remove.

        Returns
        -------
        tafra: Tafra | None
            The `Tafra` with the deleted column.
        """
        if isinstance(columns, str):
            columns = [columns]
        self._validate_columns(columns)

        for column in columns:
            _ = self._data.pop(column, None)
            _ = self._dtypes.pop(column, None)

    def copy(self, order: 'Literal["K", "A", "C", "F"]' = "C") -> "Tafra":
        """
        Create a copy of a `Tafra`.

        Parameters
        ----------
        order: str = 'C' {'C', 'F', 'A', 'K'}
            Controls the memory layout of the copy. 'C' means C-order, 'F' means
            F-order, 'A' means 'F' if a is Fortran contiguous, 'C' otherwise. 'K'
            means match the layout of a as closely as possible.

        Returns
        -------
        tafra: Tafra
            A copied `Tafra`.
        """
        return Tafra(
            {column: value.copy(order=order) for column, value in self._data.items()},
            self._dtypes.copy(),
            validate=False,
        )

    def coalesce(
        self,
        column: str,
        fills: Iterable[Iterable[None | str | int | float | bool | np.ndarray[Any, Any]]],
    ) -> np.ndarray[Any, Any]:
        """
        Fill `None` values from `fills`. Analogous to `SQL COALESCE` or
        `pandas.fillna()`.

        Parameters
        ----------
        column: str
            The column to coalesce.

        fills: Iterable[str | int | float | bool | np.ndarray:

        Returns
        -------
        data: np.ndarray
            The coalesced data.
        """
        # TODO: handle dtype?
        iter_fills = iter(fills)
        head = next(iter_fills)

        if column in self._data.keys():
            value = self._data[column].copy()
        else:
            value = np.empty(self._rows, np.asarray(head).dtype)

        for _fill in chain([head], iter_fills):
            fill = np.atleast_1d(np.asarray(_fill))
            where_na = np.full(self._rows, False)
            where_na |= value == np.array([None])
            try:
                where_na |= np.isnan(value)
            except (TypeError, ValueError):
                pass

            if len(fill) == 1:
                value[where_na] = fill
            else:
                value[where_na] = fill[where_na]

        return value

    def coalesce_inplace(
        self,
        column: str,
        fills: Iterable[Iterable[None | str | int | float | bool | np.ndarray[Any, Any]]],
    ) -> None:
        """
        In-place version.

        Fill `None` values from `fills`. Analogous to `SQL COALESCE` or
        `pandas.fillna()`.

        Parameters
        ----------
        column: str
            The column to coalesce.

        fills: Iterable[str | int | float | bool | np.ndarray:

        Returns
        -------
        data: np.ndarray
            The coalesced data.
        """
        self._data[column] = self.coalesce(column, fills)
        self.update_dtypes_inplace({column: self._data[column].dtype})

    def _cast_record(self, dtype: str, data: np.ndarray[Any, Any], cast_null: bool) -> float | None:
        """
        Casts needed to generate records for database insert.

        Will cast `np.nan` to `None`. Requires changing `dtype` to
        `object`.

        Parameters
        ----------
        dtype: str
            The dtype of the data value.

        data: np.ndarray
            The data to have its values cast.

        cast_null: bool
            Perform the cast for `np.nan`

        Returns
        -------
        value: Any
            The cast value.
        """
        _dtype = self._reduce_dtype(dtype)
        value: Any = RECORD_TYPE[_dtype](data.item())
        if cast_null and _dtype == "float" and np.isnan(data.item()):
            return None
        return value

    def to_records(
        self, columns: Iterable[str] | None = None, cast_null: bool = True
    ) -> Iterator[tuple[Any, ...]]:
        """
        Return a `Iterator` of `Tuple`, each being a record (i.e. row) and
        allowing heterogeneous typing. Useful for e.g. sending records back to a
        database.

        Parameters
        ----------
        columns: Iterable[str] | None = None
            The columns to extract. If `None`, extract all columns.

        cast_null: bool
            Cast `np.nan` to None. Necessary for `pyodbc`

        Returns
        -------
        records: Iterator[tuple[Any, ...]]
        """
        if columns is None:
            columns = self.columns
        else:
            if isinstance(columns, str):
                columns = [columns]
            self._validate_columns(columns)

        return (
            tuple(
                None
                if len(self._data[c]) <= row
                else self._cast_record(self._dtypes[c], self._data[c][[row]], cast_null)
                for c in columns
            )
            for row in range(self._rows)
        )

    def to_list(
        self, columns: Iterable[str] | None = None, inner: bool = False
    ) -> list[np.ndarray[Any, Any]] | list[list[Any]]:
        """
        Return a list of homogeneously typed columns (as `numpy.ndarray`). If a
        generator is needed, use `to_records()`. If `inner == True` each column
        will be cast from `numpy.ndarray` to a `List`.

        Parameters
        ----------
        columns: Iterable[str] | None = None
            The columns to extract. If `None`, extract all columns.

        inner: bool = False
            Cast all `np.ndarray` to `list`.

        Returns
        -------
        list: list[np.ndarray] | list[list[Any]]
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

    def to_tuple(
        self,
        columns: Iterable[str] | None = None,
        name: str | None = "Tafra",
        inner: bool = False,
    ) -> tuple[np.ndarray[Any, Any]] | tuple[tuple[Any, ...]]:
        """
        Return a `NamedTuple` or `Tuple`. If a generator is needed, use
        `to_records()`. If `inner == True` each column will be cast from
        `np.ndarray` to a `Tuple`. If `name` is `None`, returns a
        `Tuple` instead.

        Parameters
        ----------
        columns: Iterable[str] | None = None
            The columns to extract. If `None`, extract all columns.

        name: str | None = 'Tafra'
            The name for the `NamedTuple`. If `None`, construct a
            `Tuple` instead.

        inner: bool = False
            Cast all `np.ndarray` to `list`.

        Returns
        -------
        list: tuple[np.ndarray] | tuple[tuple[Any, ...]]
        """
        if columns is None:
            columns = self.columns
        else:
            if isinstance(columns, str):
                columns = [columns]
            self._validate_columns(columns)

        if name is None:
            if inner:
                return tuple(tuple(self._data[c]) for c in columns)  # type: ignore
            return tuple(self._data[c] for c in columns)  # type: ignore

        TafraNT = namedtuple(name, columns, rename=True)  # type: ignore

        if inner:
            return TafraNT._make((tuple(self._data[c]) for c in columns))  # type: ignore
        return TafraNT._make((self._data[c] for c in columns))  # type: ignore

    def to_array(self, columns: Iterable[str] | None = None) -> np.ndarray[Any, Any]:
        """
        Return an object array.

        Parameters
        ----------
        columns: Iterable[str] | None = None
            The columns to extract. If `None`, extract all columns.

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

        return np.array([self._data[c] for c in columns], dtype=object).T

    def to_pandas(self, columns: Iterable[str] | None = None) -> DataFrame:
        """
        Construct a `pandas.DataFrame`.

        Parameters
        ----------
        columns: Iterable[str]
            The columns to write. IF `None`, write all columns.

        Returns
        -------
        dataframe: `pandas.DataFrame`
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("`pandas` does not appear to be installed.")

        if columns is None:
            columns = self.columns
        else:
            if isinstance(columns, str):
                columns = [columns]
            self._validate_columns(columns)

        return pd.DataFrame(
            {column: pd.Series(value) for column, value in self._data.items() if column in columns}
        )

    def to_csv(
        self, filename: str | Path | TextIOWrapper | IO[str], columns: Iterable[str] | None = None
    ) -> None:
        """
        Write the `Tafra` to a CSV.

        Parameters
        ----------
        filename: str | Path
            The path of the filename to write.

        columns: Iterable[str]
            The columns to write. IF `None`, write all columns.
        """
        if columns is None:
            columns = self.columns
        else:
            if isinstance(columns, str):
                columns = [columns]
            self._validate_columns(columns)

        if isinstance(filename, (str, Path)):
            f = open(filename, "w", newline="")
            should_close = True

        elif isinstance(filename, TextIOWrapper):
            if "w" not in filename.mode:
                raise ValueError(f"file must be opened in write mode, got {filename.mode}")
            f = filename
            should_close = False

            f.reconfigure(newline="")

        else:
            raise TypeError(
                f"`filename` must be `str`, `Path`, or `TextIOWrapper`, got `{type(filename)}`"
            )

        writer = csv.writer(f, delimiter=",", quotechar='"')
        writer.writerow((column for column in self._data.keys() if column in columns))
        writer.writerows(self.to_records(columns))

        if should_close:
            f.close()

    def union(self, other: "Tafra") -> "Tafra":
        """
        Helper function to implement `tafra.group.Union.apply()`.

        Union two `Tafra` together. Analogy to SQL UNION or `pandas.append`. All
        column names and dtypes must match.

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

    def union_inplace(self, other: "Tafra") -> None:
        """
        Inplace version.


        Helper function to implement `tafra.group.Union.apply_inplace()`.

        Union two `Tafra` together. Analogy to SQL UNION or `pandas.append`. All
        column names and dtypes must match.

        Parameters
        ----------
        other: Tafra
            The other tafra to union.

        Returns
        -------
        None: None
        """
        Union().apply_inplace(self, other)

    def group_by(
        self,
        columns: Iterable[str],
        aggregation: InitAggregation | None = None,
        iter_fn: Mapping[str, Callable[[np.ndarray[Any, Any]], Any]] | None = None,
    ) -> "Tafra":
        """
        Helper function to implement `tafra.group.GroupBy.apply()`.

        Aggregation by a set of unique values.

        Analogy to SQL `GROUP BY`, not `pandas.DataFrame.groupby()`.

        Parameters
        ----------
        columns: Iterable[str]
            The column names to group by.

        aggregation: Mapping[str, Callable[[np.ndarray[Any, Any]], Any] | \
        tuple[Callable[[np.ndarray[Any, Any]], Any], str]]
            Optional. A mapping for columns and aggregation functions. Should be
            given as {'column': fn} or {'new_column': (fn, 'column')}.

        iter_fn: Mapping[str, Callable[[np.ndarray[Any, Any]], Any]]
            Optional. A mapping for new columns names to the function to apply to
            the enumeration. Should be given as {'new_column': fn}.

        Returns
        -------
        tafra: Tafra
            The aggregated `Tafra`.
        """
        if aggregation is None:
            aggregation = {}
        if iter_fn is None:
            iter_fn = {}
        return GroupBy(columns, aggregation, iter_fn).apply(self)

    def transform(
        self,
        columns: Iterable[str],
        aggregation: InitAggregation | None = None,
        iter_fn: dict[str, Callable[[np.ndarray[Any, Any]], Any]] | None = None,
    ) -> "Tafra":
        """
        Helper function to implement `tafra.group.Transform.apply()`.

        Apply a function to each unique set of values and join to the original table.
        Analogy to `pandas.DataFrame.groupby().transform()`,
        i.e. a SQL `GROUP BY` and `LEFT JOIN` back to the original table.

        Parameters
        ----------
        columns: Iterable[str]
            The column names to group by.

        aggregation: Mapping[str, Callable[[np.ndarray[Any, Any]], Any] | \
        tuple[Callable[[np.ndarray[Any, Any]], Any], str]]
            Optional. A mapping for columns and aggregation functions. Should be
            given as {'column': fn} or {'new_column': (fn, 'column')}.

        iter_fn: Mapping[str, Callable[[np.ndarray[Any, Any]], Any]]
            Optional. A mapping for new columns names to the function to apply to
            the enumeration. Should be given as {'new_column': fn}.

        Returns
        -------
        tafra: Tafra
            The transformed `Tafra`.
        """
        if aggregation is None:
            aggregation = {}
        if iter_fn is None:
            iter_fn = {}
        return Transform(columns, aggregation, iter_fn).apply(self)

    def iterate_by(self, columns: Iterable[str]) -> Iterator["GroupDescription"]:
        """
        Helper function to implement `tafra.group.IterateBy.apply()`.

        A generator that yields a `Tafra` for each set of unique values. Analogy
        to `pandas.DataFrame.groupby()`, i.e. an `Iterator` of `Tafra`.

        Yields tuples of ((unique grouping values, ...), row indices array, subset
        tafra)

        Parameters
        ----------
        columns: Iterable[str]
            The column names to group by.

        Returns
        -------
        tafras: Iterator[GroupDescription]
            An iterator over the grouped `Tafra`.
        """
        yield from IterateBy(columns).apply(self)

    def inner_join(
        self,
        right: "Tafra",
        on: Iterable[tuple[str, str, str]],
        select: Iterable[str] | None = None,
    ) -> "Tafra":
        """
        Helper function to implement `tafra.group.InnerJoin.apply()`.

        An inner join.

        Analogy to SQL INNER JOIN, or `pandas.merge(..., how='inner')`,

        Parameters
        ----------
        right: Tafra
            The right-side `Tafra` to join.

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
            The columns to return. If not given, all unique columns names are
            returned. If the column exists in both `Tafra`, prefers the left
            over the right.

        Returns
        -------
        tafra: Tafra
            The joined `Tafra`.
        """
        if select is None:
            select = []
        return InnerJoin(on, select).apply(self, right)

    def left_join(
        self,
        right: "Tafra",
        on: Iterable[tuple[str, str, str]],
        select: Iterable[str] | None = None,
    ) -> "Tafra":
        """
        Helper function to implement `tafra.group.LeftJoin.apply()`.

        A left join.

        Analogy to SQL LEFT JOIN, or `pandas.merge(..., how='left')`,

        Parameters
        ----------
        right: Tafra
            The right-side `Tafra` to join.

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
            The columns to return. If not given, all unique columns names are
            returned. If the column exists in both `Tafra`, prefers the left
            over the right.

        Returns
        -------
        tafra: Tafra
            The joined `Tafra`.
        """
        if select is None:
            select = []
        return LeftJoin(on, select).apply(self, right)

    def cross_join(self, right: "Tafra", select: Iterable[str] | None = None) -> "Tafra":
        """
        Helper function to implement `tafra.group.CrossJoin.apply()`.

        A cross join.

        Analogy to SQL CROSS JOIN, or `pandas.merge(left, right, how='cross')`.

        Parameters
        ----------
        right: Tafra
            The right-side `Tafra` to join.

        select: Iterable[str] = []
            The columns to return. If not given, all unique columns names are
            returned. If the column exists in both `Tafra`, prefers the left
            over the right.

        Returns
        -------
        tafra: Tafra
            The joined `Tafra`.
        """
        if select is None:
            select = []
        return CrossJoin([], select).apply(self, right)

    def chunks(self, n: int, sort_by: Iterable[str] | None = None) -> list["Tafra"]:
        """
        Split into `n` roughly equal-sized `Tafra` chunks.

        Parameters
        ----------
        n: int
            Number of chunks.

        sort_by: Iterable[str] | None
            Columns to sort by before splitting.

        Returns
        -------
        chunks: list[Tafra]
            The chunked `Tafra` instances.
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        source = self._sorted(sort_by) if sort_by else self
        splits = np.array_split(np.arange(source._rows), n)
        return [
            Tafra(
                {col: val[idx] for col, val in source._data.items()},
                source._dtypes.copy(),
                validate=False,
            )
            for idx in splits
            if len(idx) > 0
        ]

    def chunk_rows(self, size: int, sort_by: Iterable[str] | None = None) -> list["Tafra"]:
        """
        Split into chunks of at most `size` rows each.

        Parameters
        ----------
        size: int
            Maximum rows per chunk.

        sort_by: Iterable[str] | None
            Columns to sort by before splitting.

        Returns
        -------
        chunks: list[Tafra]
            The chunked `Tafra` instances.
        """
        if size < 1:
            raise ValueError("size must be >= 1")

        n = max(1, (self._rows + size - 1) // size)
        return self.chunks(n, sort_by=sort_by)

    def partition(
        self, columns: Iterable[str], sort_by: Iterable[str] | None = None
    ) -> list[tuple[tuple[Any, ...], "Tafra"]]:
        """
        Split by unique values in `columns`, preserving group integrity.

        Parameters
        ----------
        columns: Iterable[str]
            Columns to partition by.

        sort_by: Iterable[str] | None
            Columns to sort by within each partition.

        Returns
        -------
        partitions: list[tuple[tuple[Any, ...], Tafra]]
            List of (group_key, sub_tafra) pairs.
        """
        from .group import GroupSet

        unique, group_indices = GroupSet._build_group_indices(self, columns)
        result: list[tuple[tuple[Any, ...], "Tafra"]] = []

        for key, rows in zip(unique, group_indices):
            sub = Tafra(
                {col: val[rows] for col, val in self._data.items()},
                self._dtypes.copy(),
                validate=False,
            )
            if sort_by:
                sub = sub._sorted(sort_by)
            result.append((key, sub))

        return result

    def _sorted(self, sort_by: Iterable[str]) -> "Tafra":
        """Return a new Tafra sorted by the given columns."""
        cols = list(sort_by)
        self._validate_columns(cols)

        # lexsort uses last key as primary, so reverse
        keys = [self._data[c] for c in reversed(cols)]
        order = np.lexsort(keys)

        return Tafra(
            {col: val[order] for col, val in self._data.items()},
            self._dtypes.copy(),
            validate=False,
        )

    @classmethod
    def concat(cls, tafras: Iterable["Tafra"]) -> "Tafra":
        """
        Concatenate multiple `Tafra` instances row-wise.

        Parameters
        ----------
        tafras: Iterable[Tafra]
            The tafras to concatenate.

        Returns
        -------
        tafra: Tafra
            The concatenated `Tafra`.
        """
        tafra_list = list(tafras)
        if not tafra_list:
            raise ValueError("No tafras to concatenate.")

        columns = list(tafra_list[0]._data.keys())
        col_set = set(columns)
        for i, t in enumerate(tafra_list[1:], 1):
            if set(t._data.keys()) != col_set:
                raise ValueError(
                    f"Tafra at index {i} has columns {list(t._data.keys())}, expected {columns}."
                )
        return cls(
            {col: np.concatenate([t._data[col] for t in tafra_list]) for col in columns},
            tafra_list[0]._dtypes.copy(),
            validate=False,
        )


def to_field_name(maybe_text: str | int | float) -> str:  # pragma: no cover
    text = str(maybe_text)

    # Remove invalid characters
    mid_text = re.sub("[^0-9a-zA-Z]", "", text)

    # Remove leading characters until we find a letter
    final_text = re.sub("^[^a-zA-Z]+", "", mid_text)

    if final_text == "":
        final_text = "field_" + mid_text

    return final_text


def _in_notebook() -> bool:  # pragma: no cover
    """
    Checks if running in a Jupyter Notebook.

    Returns
    -------
    in_notebook: bool
    """
    try:
        from IPython import get_ipython

        if "IPKernelApp" in get_ipython().config:  # type: ignore[no-untyped-call,unused-ignore]
            return True
    except Exception as e:
        pass
    return False


# Import here to resolve circular dependency
from .group import (
    GroupBy,
    Transform,
    IterateBy,
    InnerJoin,
    LeftJoin,
    CrossJoin,
    Union,
    InitAggregation,
    GroupDescription,
)
