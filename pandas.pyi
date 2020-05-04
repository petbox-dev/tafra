import numpy as np
from typing import Dict, List

class DataFrame():
    """A fake class to satisfy typing of a `pandas.DataFrame` without a dependency.
    """
    _data: Dict[str, np.ndarray]
    columns: List[str]
    dtypes: List[str]

    def __getitem__(self, column: str) -> np.ndarray:
        return self._data[column]

    def __setitem__(self, column: str, value: np.ndarray):
        self._data[column] = value
