import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Type, Iterable, Iterator
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Series(Protocol):
    name: str
    values: np.ndarray
    dtype: str


@runtime_checkable
class DataFrame(Protocol):
    """
    A fake class to satisfy typing of a ``pandas.DataFrame`` without a dependency.
    """
    _data: Dict[str, Series]
    columns: List[str]
    dtypes: List[str]

    def __getitem__(self, column: str) -> Series:
        raise NotImplementedError

    def __setitem__(self, column: str, value: np.ndarray) -> None:
        raise NotImplementedError

@runtime_checkable
class Cursor(Protocol):
    """
    A fake class to satisfy typing of a ``pyodbc.Cursor`` without a dependency.
    """
    description: Tuple[Tuple[str, Type[Any], Optional[int], int, int, int, bool]]

    def __iter__(self) -> Iterator[Tuple[Any, ...]]:
        raise NotImplementedError

    def __next__(self) -> Tuple[Any, ...]:
        raise NotImplementedError

    def execute(self, sql: str) -> None:
        raise NotImplementedError

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        raise NotImplementedError

    def fetchmany(self, size: int) -> List[Tuple[Any, ...]]:
        raise NotImplementedError

    def fetchall(self) -> List[Tuple[Any, ...]]:
        raise NotImplementedError
