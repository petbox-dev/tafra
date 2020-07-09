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
from pathlib import Path
import csv
import dataclasses as dc

from datetime import date, datetime
import numpy as np

from enum import Enum, auto
from io import TextIOWrapper
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type
from typing import IO, Union, cast

# this doesn't type well in Python
@dc.dataclass(frozen=True)
class ReadableType:
    dtype: Type[Any]
    parse: Callable[[str], Any]

def _parse_bool(val: str) -> bool:
    folded = val.casefold()
    if folded in ('false', 'no', 'f'):
        return False
    if folded in ('true', 'yes', 't'):
        return True
    raise ValueError('not a boolean')

# numpy-stubs is a lie about many of these, hence the type: ignore spam
_TYPE_PRECEDENCE: List[ReadableType] = [
    ReadableType(np.int32, cast(Callable[[str], Any], np.int32)),
    ReadableType(np.int64, cast(Callable[[str], Any], np.int64)),
    # np.float32, # nervous about ever inferring this
    ReadableType(np.float64, cast(Callable[[str], Any], np.float64)),
    ReadableType(bool, _parse_bool),
    # TODO: date,
    # TODO: datetime,
]

_TYPE_OBJECT: ReadableType = ReadableType(object, lambda x: x)

class ReaderState(Enum):
    AWAIT_GUESSABLE = auto()
    EARLY_EOF = auto()
    GUESS = auto()
    READ = auto()
    EOF = auto()
    DONE = auto()

class CSVReader:
    def __init__(self, source: Union[str, Path, TextIOWrapper],
                 guess_rows: int = 5, missing: Optional[str] = '',
                 **csvkw: Dict[str, Any]):
        if isinstance(source, (str, Path)):
            self._stream = open(source, newline='')
            self._should_close = True
        elif isinstance(source, TextIOWrapper):
            source.reconfigure(newline='')
            self._stream = source
            self._should_close = False
        reader = csv.reader(self._stream, dialect='excel', **csvkw)
        self._header = _unique_header(next(reader))
        self._reader = (self._decode_missing(t) for t in reader)
        self._guess_types = {
            col: _TYPE_PRECEDENCE[0] for col in self._header
        }
        self._guess_data: Dict[str, List[Any]] = {
            col: list() for col in self._header
        }
        self._data: Dict[str, List[Any]] = dict()
        self._guess_rows = guess_rows
        self._missing = missing
        self._rows = 0
        self._state = ReaderState.AWAIT_GUESSABLE

    def read(self) -> Dict[str, np.ndarray]:
        while self._state != ReaderState.DONE:
            self._step()
        return self._finalize()

    def _step(self) -> None:
        if self._state == ReaderState.AWAIT_GUESSABLE:
            self.state_await_guessable()
            return

        if self._state == ReaderState.GUESS:
            self.state_guess()
            return

        if self._state == ReaderState.READ:
            self.state_read()
            return

        if self._state == ReaderState.EARLY_EOF:
            self.state_early_eof()
            return

        if self._state == ReaderState.EOF:
            self.state_eof()
            return

        if self._state == ReaderState.DONE:  # pragma: no cover
            return

    def state_await_guessable(self) -> None:
        try:
            row = next(self._reader)
        except StopIteration:
            self._state = ReaderState.EARLY_EOF
            return

        self._rows += 1
        if len(row) != len(self._header):
            raise ValueError(f'length of row #{self._rows}'
                             ' does not match header length')

        for col, val in zip(self._header, row):
            self._guess_data[col].append(val)

        if self._rows == self._guess_rows:
            self._state = ReaderState.GUESS

    def state_guess(self) -> None:
        for col in self._header:
            ty, parsed = _guess_column(_TYPE_PRECEDENCE,
                                       self._guess_data[col])
            self._guess_types[col] = ty
            self._data[col] = parsed
        self._state = ReaderState.READ

    def state_read(self) -> None:
        try:
            row = next(self._reader)
        except StopIteration:
            self._state = ReaderState.EOF
            return

        self._rows += 1
        if len(row) != len(self._header):
            raise ValueError(f'length of row #{self._rows}'
                             ' does not match header length')

        for col, val in zip(self._header, row):
            try:
                self._data[col].append(self._guess_types[col].parse(val)) # type: ignore
            except:
                self._promote(col, val)

    def state_early_eof(self) -> None:
        if self._should_close:
            self._stream.close()

        for col in self._header:
            ty, parsed = _guess_column(_TYPE_PRECEDENCE,
                                       self._guess_data[col])
            self._guess_types[col] = ty
            self._data[col] = parsed

        self._state = ReaderState.DONE

    def state_eof(self) -> None:
        if self._should_close:
            self._stream.close()
        self._state = ReaderState.DONE

    def _promote(self, col: str, val: Optional[str]) -> None:
        ty_ix = _TYPE_PRECEDENCE.index(self._guess_types[col])
        try_next = _TYPE_PRECEDENCE[ty_ix + 1:]
        stringized = self._encode_missing(self._data[col])
        stringized.append(val)
        ty, parsed = _guess_column(try_next, stringized)
        self._guess_types[col] = ty
        self._data[col] = parsed

    def _finalize(self) -> Dict[str, np.ndarray]:
        assert self._state == ReaderState.DONE, 'CSVReader is not in DONE state.'
        return {
            col: np.array(self._data[col], dtype=self._guess_types[col].dtype)
            for col in self._header
        }

    def _decode_missing(self, row: List[str]) -> Sequence[Optional[str]]:
        if self._missing is None:
            return row
        return [v if v != self._missing else None for v in row]

    def _encode_missing(self, row: Sequence[Optional[Any]]) -> List[Optional[str]]:
        return [str(v) if v is not None else self._missing for v in row]

def _unique_header(header: List[str]) -> List[str]:
    uniq: List[str] = list()
    for col in header:
        col_unique = col
        i = 2
        while col_unique in uniq:
            col_unique = f'{col} ({i})'
            i += 1
        uniq.append(col_unique)
    return uniq

# the "real" return type is a dependent pair (t: ReadableType ** List[t.dtype])
def _guess_column(precedence: List[ReadableType], vals: List[Optional[str]]
                  ) -> Tuple[ReadableType, List[Any]]:
    for ty in precedence:
        try:
            # mypy doesn't really get that the thing we're mapping is not a method
            #   on `ty` but a data member
            typed = list(map(ty.parse, vals)) # type: ignore
            return ty, typed
        except:
            next
    return _TYPE_OBJECT, vals
