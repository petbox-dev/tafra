# Column Operations

Tafra uses dict-like access for columns and provides methods for selecting,
renaming, updating, and deleting columns.

## Accessing Columns

### By name (`__getitem__` with `str`)

Returns the raw `numpy.ndarray` for the column:

```python
from tafra import Tafra
import numpy as np

t = Tafra({
    'x': np.array([1, 2, 3]),
    'y': np.array([4.0, 5.0, 6.0]),
    'name': np.array(['a', 'b', 'c']),
})

arr = t['x']
print(type(arr))
```

???+ example "Output"

    ```
    <class 'numpy.ndarray'>
    ```

### By integer index

Returns a single-row `Tafra`:

```python
row = t[0]
print(row.rows)
```

???+ example "Output"

    ```
    1
    ```

### By slice

Returns a sliced `Tafra`:

```python
first_two = t[0:2]
print(first_two.rows)
```

???+ example "Output"

    ```
    2
    ```

### By boolean array

Filters rows where the condition is `True`:

```python
mask = t['x'] > 1
filtered = t[mask]
```

???+ example "Output"

    ```
    Tafra with x=[2, 3], y=[5.0, 6.0], name=['b', 'c']
    ```

### By list of column names

Returns a `Tafra` with only the listed columns (like `select`):

```python
subset = t[['x', 'name']]
print(subset.columns)
```

???+ example "Output"

    ```
    ('x', 'name')
    ```

## Setting Columns

Assign an array, list, or scalar to a column name. The value is validated for
length and converted to an `ndarray`:

```python
t['z'] = np.array([7, 8, 9])        # new column
t['x'] = np.array([10, 20, 30])     # overwrite existing column
t['flag'] = True                     # scalar broadcast to all rows
```

## Dict-like Interface

### `keys()`, `values()`, `items()`

```python
print(t.keys())
print(t.values())
print(t.items())
```

???+ example "Output"

    ```
    dict_keys(['x', 'y', 'name', 'z', 'flag'])
    dict_values([array([10, 20, 30]), array([4., 5., 6.]), ...])
    dict_items([('x', array([10, 20, 30])), ...])
    ```

### `get()`

Returns the column array, or a default if the column does not exist:

```python
arr = t.get('x')             # np.ndarray([10, 20, 30])
arr = t.get('missing', None) # None
```

## Selecting Columns

`select()` returns a new `Tafra` with only the specified columns. This does
**not** copy the underlying data -- call `.copy()` if you need independent
arrays.

```python
sub = t.select(['x', 'y'])
print(sub.columns)
```

???+ example "Output"

    ```
    ('x', 'y')
    ```

```python
# With copy
sub_copy = t.select(['x', 'y']).copy()
```

## Updating from Another Tafra

`update()` merges columns from another `Tafra` into this one. Both must have
the same row count. Returns a new `Tafra`.

```python
other = Tafra({'w': np.array([100, 200, 300])})
t2 = t.update(other)
print('w' in t2.keys())
```

???+ example "Output"

    ```
    True
    ```

Use `update_inplace()` for the in-place version:

```python
t.update_inplace(other)
```

## Updating Dtypes

`update_dtypes()` casts columns to new dtypes. Returns a new `Tafra`.

```python
t2 = t.update_dtypes({'x': 'float64'})
print(t2['x'].dtype)
```

???+ example "Output"

    ```
    float64
    ```

Use `update_dtypes_inplace()` for the in-place version:

```python
t.update_dtypes_inplace({'x': 'float64'})
```

## Renaming Columns

`rename()` takes a dict mapping old names to new names. Returns a new `Tafra`.

```python
t2 = t.rename({'x': 'x_val', 'y': 'y_val'})
print(t2.columns)
```

???+ example "Output"

    ```
    ('x_val', 'y_val', 'name', ...)
    ```

Use `rename_inplace()` for the in-place version.

## Deleting Columns

`delete()` removes columns by name. Returns a new `Tafra`.

```python
t2 = t.delete(['z', 'flag'])
print('z' in t2.keys())
```

???+ example "Output"

    ```
    False
    ```

Use `delete_inplace()` for the in-place version:

```python
t.delete_inplace(['z', 'flag'])
```

Both accept a single string or a list of strings.

## Row Iteration

### `iterrows()`

Yields each row as a single-row `Tafra`. Convenient but slow for large data.

```python
for row in t.iterrows():
    print(row['x'], row['y'])
```

### `itertuples()`

Yields rows as `NamedTuple` instances. Faster than `iterrows()`.

```python
for row in t.itertuples():
    print(row.x, row.y)

# As plain tuples (no named fields)
for row in t.itertuples(name=None):
    print(row)
```

???+ example "Output"

    ```
    (10, 4.0, 'a', ...)
    ```

### `itercols()`

Yields `(column_name, ndarray)` tuples:

```python
for name, arr in t.itercols():
    print(f'{name}: {arr.dtype}, len={len(arr)}')
```

## Mapping Functions

### `row_map(fn)` -- map over rows

```python
results = list(t.row_map(lambda row: row['x'] * 2))
```

### `tuple_map(fn)` -- map over named tuples (faster)

```python
results = list(t.tuple_map(lambda row: row.x * 2))
```

### `col_map(fn)` -- map over columns

```python
means = list(t.select(['x', 'y']).col_map(np.mean))
```

### `key_map(fn)` -- map over columns with names

```python
named_means = dict(t.select(['x', 'y']).key_map(np.mean))
print(named_means)
```

???+ example "Output"

    ```
    {'x': 20.0, 'y': 5.0}
    ```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `columns` | `Tuple[str, ...]` | Column names |
| `rows` | `int` | Number of rows |
| `data` | `Dict[str, ndarray]` | Underlying data dict (read-only) |
| `dtypes` | `Dict[str, str]` | Column dtype strings (read-only) |
| `shape` | `Tuple[int, int]` | `(rows, n_columns)` |
| `size` | `int` | `rows * n_columns` |
| `ndim` | `int` | Always `2` |

## Other Operations

| Method | Description |
|--------|-------------|
| `head(n=5)` | First `n` rows |
| `tail(n=5)` | Last `n` rows |
| `sort(columns, reverse=False)` | Sort by one or more columns |
| `sample(n, seed=None)` | Random sample of `n` rows |
| `copy(order='C')` | Deep copy |
| `drop_duplicates(columns=None)` | Remove duplicate rows |
| `value_counts(column)` | Count unique values |
| `describe()` | Summary statistics for numeric columns |
| `shift(n=1)` | Shift rows (lag/lead) |
| `coalesce(column, fills)` | Fill None/NaN from fallback values |
| `pipe(fn)` | Apply a function, return result (also `t >> fn`) |
| `union(other)` | Append rows (like SQL UNION) |
| `to_csv(path)` | Write to CSV |
| `to_pandas()` | Convert to `pandas.DataFrame` |
| `to_records()` | Iterator of row tuples |
| `to_html()` | HTML table string |
