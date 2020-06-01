import numpy as np
from typing import Dict, List
from typing_extensions import Protocol, runtime_checkable

@runtime_checkable
class DataFrame(Protocol):
    """A fake class to satisfy typing of a `pandas.DataFrame` without a dependency.
    """
    _data: Dict[str, np.ndarray]
    columns: List[str]
    dtypes: List[str]

    def __getitem__(self, column: str) -> np.ndarray:
        ...

    def __setitem__(self, column: str, value: np.ndarray) -> None:
        ...
