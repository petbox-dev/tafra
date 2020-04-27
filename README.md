% "typing" group-by

Let's discuss an interesting data structure we shall call a *tafra*, because
it's just the innards of a dataframe. A *tafra* (plural: *tafrae*) consists of
an ordered list of homogeneous vectors, all of the same length.
Each vector corresponds to a *dimension* or *column* of the tafra, and each is
of the same length. The elements of these vectors correspond to the *rows*
of the tafra. Each column of the tafra has a *type* and a *name*;
the type of a tafra is precisely the ordered list of the types and names of
its columns.
Notice that we do not include the number of rows of a tafra in its type.

We don't have a type system at hand which makes this pleasant, so let's just
keep the types in our heads and impose the checks at runtime.
For now, we'll just expose two operations - first, the ability to construct a
tafra from a dictionary of homogeneous arrays by column name; and second,
the ability to access a column by name.

```python
import dataclasses as dc

import numpy as np

from typing import Dict

@dc.dataclass
class Tafra:
    _data: Dict[str, np.ndarray]

    def __post_init__(self):
        rows = None
        for _, column in self._data.items():
            if rows is None:
                rows = len(column)
            else:
                if rows != len(column):
                    raise ValueError('tafra must have consistent row counts')

    def __getitem__(self, column: str) -> np.ndarray:
        return self._data[column]

    def __getattr__(self, column: str) -> np.ndarray:
        return self._data[column]
```

We can construct tafrae and access their columns like so:
```python
t = Tafra({
    'x': np.array([1, 2, 3, 4]),
    'y': np.array(['one', 'two', 'one', 'two'], dtype='object'),
})

print(t.x ** 37.2)
print(t.y + ' is the loneliest number')
```

We could provide other affordances, such as the ability to add columns to
an existing tafra (with dynamic row-count checking), but let's jump ahead
to a more interesting problem: providing an operation analogous to SQL's
`group by`.

In SQL, `group by` is used to produce a relation where each tuple in the
relation corresponds to the unique combinations of values of one or more
columns in the underlying data.
The columns not used to produce unique combinations (that is, for "grouping")
must somehow be reduced from their original number of values to just one value
per group.
This is achieved by providing an aggregation function, which reduces multiple
values into one - examples include `sum`, `min`, and `average`.

While in SQL we can write arbitrarily complex expressions using aggregation
functions on multiple source columns to produce a single result column, we'll
start with a less ambitious goal: we'll provide the ability to specify the
columns by which to group, as well as an aggregation function for each other
column of interest in the tafra.

Along the way, we'll add a couple of convenience properties to our tafra class.

```python
import dataclasses as dc

import numpy as np

from typing import Any, Callable, Dict, List, Tuple, Optional, Iterable

def _real_has_attribute(obj, attr):
    try:
        obj.__getattribute__(attr)
        return True
    except AttributeError:
        return False


@dc.dataclass
class Tafra:
    _data: Dict[str, np.ndarray]

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

    def __setitem__(self, column: str, value: np.ndarray):
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

    def to_records(self, columns: Optional[Iterable[str]] = None):
        """
        return a list of lists, each list being a record (i.e. row)
        """
        if columns is None:
            return list(zip(*(self._data[c] for c in self.columns)))
        return list(zip(*(self._data[c] for c in columns)))

    def to_list(self, columns: Optional[Iterable[str]] = None):
        """
        Return a list of lists, each list being a column
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


if __name__ == '__main__':
    t = Tafra({
        'x': np.array([1, 2, 3, 4]),
        'y': np.array(['one', 'two', 'one', 'two'], dtype='object'),
    })

    gb = t.group_by(
        ['y'], {'x': sum}
    )

    print(gb)
```

We use a new data class to represent the structure of a `group by` - we can
think of it as a very primitive "abstract syntax tree".
A `group by` is completely defined by the grouping column names (in order) and
the specification of columns to be aggregated and their aggregation functions.
We provide a function for validating a specification against a tafra before
executing it, by checking that the mentioned column names are present.

With these in place, the `group by` logic itself is almost mechanical:
first, identify the unique combinations of the grouping columns.
For each of these, identify which rows in the original tafra match each group,
and for each column / aggregation pair in the spec, accumulate relevant values
and apply the aggregation function.
We accumulate the results into the required structure and produce our resulting
tafra.
