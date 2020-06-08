import csv
import dataclasses as dc

from datetime import date, datetime
import numpy as np

from enum import Enum, auto
from io import TextIOWrapper
from typing import Any, Callable, Dict, List, Tuple, Type
from typing import Union

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
    ReadableType(np.int32, np.int32), # type: ignore
    ReadableType(np.int64, np.int64), # type: ignore
    # np.float32, # nervous about ever inferring this
    ReadableType(np.float64, np.float64), # type: ignore
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
    def __init__(self, source: Union[str, TextIOWrapper], guess_rows: int = 5,
                 **csvkw: Dict[str, Any]):
        if isinstance(source, str):
            self._stream = open(source, newline='')
            self._should_close = True
        else:
            source.reconfigure(newline='') # type: ignore
            self._stream = source
            self._should_close = False
        self._reader = csv.reader(self._stream, **csvkw)
        self._header = _unique_header(next(self._reader))
        self._guess_types = {
            col: _TYPE_PRECEDENCE[0] for col in self._header
        }
        self._guess_data: Dict[str, List[Any]] = {
            col: list() for col in self._header
        }
        self._data: Dict[str, List[Any]] = dict()
        self._guess_rows = guess_rows
        self._rows = 0
        self._state = ReaderState.AWAIT_GUESSABLE

    def read(self) -> Dict[str, np.ndarray]:
        while self._state != ReaderState.DONE:
            self._step()
        return self._finalize()

    def _step(self) -> None:
        if self._state == ReaderState.AWAIT_GUESSABLE:
            try:
                row = next(self._reader)
            except StopIteration:
                self._state = ReaderState.EARLY_EOF
                return

            self._rows += 1
            if len(row) != len(self._header):
                raise ValueError(f'length of row #{self._rows}' +
                        ' does not match header length')

            for col, val in zip(self._header, row):
                self._guess_data[col].append(val)

            if self._rows == self._guess_rows:
                self._state = ReaderState.GUESS
            return

        if self._state == ReaderState.GUESS:
            for col in self._header:
                ty, parsed = _guess_column(_TYPE_PRECEDENCE,
                        self._guess_data[col])
                self._guess_types[col] = ty
                self._data[col] = parsed
            self._state = ReaderState.READ
            return

        if self._state == ReaderState.READ:
            try:
                row = next(self._reader)
            except StopIteration:
                self._state = ReaderState.EOF
                return

            self._rows += 1
            if len(row) != len(self._header):
                raise ValueError(f'length of row #{self._rows}' +
                        ' does not match header length')

            for col, val in zip(self._header, row):
                try:
                    self._data[col].append(self._guess_types[col].parse(val)) # type: ignore
                except:
                    self._promote(col, val)

            return

        if self._state == ReaderState.EARLY_EOF:
            if self._should_close:
                self._stream.close()

            for col in self._header:
                ty, parsed = _guess_column(_TYPE_PRECEDENCE,
                        self._guess_data[col])
                self._guess_types[col] = ty
                self._data[col] = parsed

            self._state = ReaderState.DONE
            return

        if self._state == ReaderState.EOF:
            if self._should_close:
                self._stream.close()
            self._state = ReaderState.DONE
            return

        if self._state == ReaderState.DONE:
            return

    def _promote(self, col: str, val: str) -> None:
        ty_ix = _TYPE_PRECEDENCE.index(self._guess_types[col])
        try_next = _TYPE_PRECEDENCE[ty_ix + 1:]
        stringized = list(map(str, self._data[col]))
        stringized.append(val)
        ty, parsed = _guess_column(try_next, stringized)
        self._guess_types[col] = ty
        self._data[col] = parsed

    def _finalize(self) -> Dict[str, np.ndarray]:
        if self._state != ReaderState.DONE:
            raise ValueError('CSVReader is not in DONE state.')
        return {
            col: np.array(self._data[col], dtype=self._guess_types[col].dtype)
            for col in self._header
        }

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
def _guess_column(precedence: List[ReadableType], vals: List[str]
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
