# GroupBy & Partition

Tafra provides SQL-style aggregation, transformation, and partitioning
operations. The key distinction:

- **`group_by`** reduces: one output row per unique group, applying
  aggregation functions.
- **`partition`** splits: returns all original rows grouped into sub-Tafras
  with no aggregation -- designed for parallel dispatch.

## group_by

`group_by()` aggregates data by unique values in one or more columns, producing
one row per group. This is analogous to SQL `GROUP BY`, **not**
`pandas.DataFrame.groupby()`.

### Basic usage

The `aggregation` dict maps output column names to either:

- A function (applied to the column with the same name): `{'col': fn}`
- A tuple of `(function, source_column)`: `{'new_col': (fn, 'source_col')}`

```python
import numpy as np
from tafra import Tafra

t = Tafra({
    'region': np.array(['east', 'east', 'west', 'west', 'west']),
    'product': np.array(['A', 'B', 'A', 'A', 'B']),
    'sales': np.array([100, 200, 150, 175, 250]),
    'cost': np.array([50, 120, 80, 90, 130]),
})

# Group by region, sum sales, and mean cost
result = t.group_by(
    ['region'],
    {
        'sales': np.sum,
        'cost': np.mean,
    },
)
```

???+ example "Output"

    ```
    region   sales   cost
    'east'   300     85.0
    'west'   575     100.0
    ```

### Renaming output columns

Use the `(function, source_column)` tuple form to create new column names:

```python
result = t.group_by(
    ['region'],
    {
        'total_sales': (np.sum, 'sales'),
        'avg_cost': (np.mean, 'cost'),
        'n_rows': (len, 'sales'),
    },
)
```

### Multiple group columns

```python
result = t.group_by(
    ['region', 'product'],
    {
        'total_sales': (np.sum, 'sales'),
    },
)
```

One row per unique (region, product) combination.

## Built-in Aggregation Functions

The following functions are recognized for the **vectorized fast path** (see
below):

| Function | Description |
|----------|-------------|
| `np.sum` / `np.nansum` | Sum |
| `np.mean` / `np.nanmean` | Mean |
| `np.std` / `np.nanstd` | Standard deviation |
| `np.var` / `np.nanvar` | Variance |
| `np.min` / `np.nanmin` / `np.amin` | Minimum |
| `np.max` / `np.nanmax` / `np.amax` | Maximum |
| `np.median` / `np.nanmedian` | Median |
| `np.prod` / `np.nanprod` | Product |
| `np.ptp` | Peak-to-peak (max - min) |
| `np.any` | Logical OR |
| `np.all` | Logical AND |
| `np.count_nonzero` | Count non-zero |
| `len` | Count |
| `sum` (builtin) | Sum |

## Custom Aggregations

### `percentile(q)`

```python
from tafra.group import percentile

result = t.group_by(
    ['region'],
    {
        'p10_sales': (percentile(10), 'sales'),
        'p50_sales': (percentile(50), 'sales'),
        'p90_sales': (percentile(90), 'sales'),
    },
)
```

`percentile(q)` returns a callable that computes the q-th percentile (0-100).
It carries its own vectorized fast path for integer and float columns.

### `geomean` and `harmean`

```python
from tafra.group import geomean, harmean

result = t.group_by(
    ['region'],
    {
        'geo_sales': (geomean, 'sales'),
        'harm_sales': (harmean, 'sales'),
    },
)
```

Both are registered for the vectorized fast path.

### Arbitrary functions

Any callable that takes an `np.ndarray` and returns a scalar works:

```python
result = t.group_by(
    ['region'],
    {
        'range': (lambda x: x.max() - x.min(), 'sales'),
        'cv': (lambda x: x.std() / x.mean(), 'sales'),
    },
)
```

Custom lambda/functions use the standard per-group Python loop (no vectorized
fast path).

The result column takes the dtype that numpy promotes the per-group results
to (`np.promote_types`), not the dtype of the input column: a mean of an
`int64` column is `float64`, and `np.any` of a string column is `bool`.
Results that share no dtype, such as `None` with a float, give an `object`
column. A string mixed with a non-string also gives an `object` column, so
`'low'` with `2.0` gives `['low', 2.0]`; numpy would turn `2.0` into the text
`'2.0'`. In `transform`, a function may return one value
per group (broadcast to the group's rows) or one value per row.

## Vectorized Fast Path

When **all** aggregation functions in a `group_by` call are recognized (see
table above), Tafra uses a vectorized code path based on `np.bincount` and
`ufunc.reduceat` instead of iterating over groups in Python. This is
significantly faster for large datasets.

The fast path is used automatically -- no configuration needed. If any
aggregation function is not recognized, the entire call falls back to the
per-group Python loop. The fast path also applies only to the column dtypes
that numpy reduces with that function: for example, `np.mean` on a string
column takes the per-group loop, which calls `np.mean` and raises the same
error that numpy raises.

The fast path gives the same result as the function applied to each group:

- `np.min`, `np.max`, `np.sum`, `np.mean` and the others propagate NaN;
  `np.nanmin`, `np.nansum`, `np.nanmean` and the others ignore it (an all-NaN
  group gives NaN, or 0 for `np.nansum` and 1 for `np.nanprod`).
- The result dtype is numpy's: `np.sum` of an `int64` column is an exact
  `int64`, `np.min` of a `datetime64` column is a `datetime64`, and a
  `float32` column keeps `float32`.
- Floating-point values agree up to rounding. The fast path adds in float64 in
  row order and numpy adds pairwise in the input width, so the last digits
  can differ (about 1e-14 relative for the sum of a 200,000-row `float64`
  group, and up to about 4e-13 for the variance of values near 1e9), more so
  for `float16` and `float32`, where a sum numpy overflows to `inf` can stay
  finite. An all-NaN group does not emit numpy's `RuntimeWarning`.

If the optional C extension (`_accel.c`) is compiled, `np.sum`, `np.mean`,
`np.var`, `np.std`, `np.min`, `np.max` and their `nan` forms use C
implementations (one pass for most, two passes for variance; exact integer
sums) for even better performance.

## Key Columns

`group_by`, `transform`, `iterate_by` and the equi-joins accept key columns
of any dtype numpy can compare with `==`: integers of any range, `uint64`,
`bool`, floats, `datetime64` and `timedelta64` of any unit, strings
(`StringDType` with `None`, `<U`, `bytes`) and `object`. The two sides of a
join key must share a dtype kind (an `int64` key does not join a `float64`
or `uint64` key, and a `<U` key does not join a `bytes` key), or both be
text strings (`StringDType` and `<U`). Keys follow numpy equality, with
these rules:

- `-0.0` and `0.0` are one key.
- In a float column every NaN is one key in `group_by`, as is every NaT in a
  `datetime64` or `timedelta64` column. In an `object` column, NaN values are
  one key only when they are the same Python object.
- Joins never match a NULL key (SQL `NULL` semantics): NaN in a float column,
  NaT in a `datetime64` or `timedelta64` column, the `na_object` of a
  `StringDType` column (`None` by default; `''` or NaN when the dtype says so),
  and `None`, NaN or NaT in an `object` column. Two `StringDType` keys with
  different `na_object` values join. Other missing-value
  markers are not NULL: a join matches complex NaN to complex NaN, as
  `np.unique` merges them, and `pd.NA` or `pd.NaT` to the same object.
- `datetime64` keys of different units are compared in the finer unit, as
  numpy converts them, and NULL is judged after that conversion, for every
  operator. For a value outside that unit's range (about the years 1678 to
  2262 for `ns`), numpy 2.2 and 2.4 wrap the value without an error: it can
  become NaT (no match) or a valid but different date that matches another
  key. The numpy 2.5 release notes describe unit-conversion casts as
  overflow-checked, so a newer numpy can raise instead. Convert such keys to a
  common, wide-enough unit before the join.

## transform

`transform()` computes group aggregations and **broadcasts the result back** to
every row of the original data. The output has the same number of rows as the
input. Analogous to `pandas.DataFrame.groupby().transform()`, or a SQL
`GROUP BY` with a `LEFT JOIN` back to the original table.

```python
result = t.transform(
    ['region'],
    {
        'region_total': (np.sum, 'sales'),
        'region_avg': (np.mean, 'sales'),
    },
)
print(result.rows)
```

???+ example "Output"

    ```
    5
    ```

    `result` has 5 rows (same as input).
    `region_total` is 300 for east rows, 575 for west rows.
    `region_avg` is 150.0 for east rows, 191.67 for west rows.

Transform also supports the vectorized fast path.

## iterate_by

`iterate_by()` yields one sub-Tafra per group, along with the group key values
and row indices. Analogous to iterating over `pandas.DataFrame.groupby()`.

Each iteration yields a tuple of `(keys, indices, sub_tafra)`:

- `keys` -- tuple of the unique values for the group-by columns
- `indices` -- `np.ndarray` of row indices into the original Tafra
- `sub_tafra` -- the subset `Tafra` for that group

```python
for keys, indices, sub in t.iterate_by(['region']):
    print(f'Region: {keys[0]}, rows: {sub.rows}')
    print(f'  indices: {indices}')
    print(f'  sales: {sub["sales"]}')
```

???+ example "Output"

    ```
    Region: east, rows: 2
      indices: [0 1]
      sales: [100 200]
    Region: west, rows: 3
      indices: [2 3 4]
      sales: [150 175 250]
    ```

### Processing each group

```python
results = []
for keys, indices, sub in t.iterate_by(['region']):
    total = np.sum(sub['sales'])
    results.append(Tafra({
        'region': np.array([keys[0]]),
        'total': np.array([total]),
    }))

combined = Tafra.concat(results)
```

## partition

`partition()` splits a Tafra into sub-Tafras by unique values in the given
columns. Unlike `group_by`, it preserves **all original rows** with no
aggregation. This is designed for dispatching groups to parallel workers.

Returns a list of `(group_key, sub_tafra)` tuples.

```python
parts = t.partition(['region'])

for key, sub in parts:
    print(f'Group {key}: {sub.rows} rows')
    print(f'  sales: {sub["sales"]}')
```

???+ example "Output"

    ```
    Group ('east',): 2 rows
      sales: [100 200]
    Group ('west',): 3 rows
      sales: [150 175 250]
    ```

### Optional sorting within partitions

```python
parts = t.partition(['region'], sort_by=['sales'])
# Each sub-Tafra is sorted by 'sales' within its group
```

### Multiprocessing example

The primary use case for `partition` is parallel processing with
`multiprocessing.Pool.map()`:

```python
from multiprocessing import Pool
from tafra import Tafra

def process_group(args):
    key, sub_tafra = args
    # Perform expensive computation on sub_tafra
    result = sub_tafra.copy()
    result['score'] = sub_tafra['sales'] * 2.0
    return result

t = Tafra({
    'region': np.array(['east', 'east', 'west', 'west', 'west']),
    'sales': np.array([100, 200, 150, 175, 250]),
})

parts = t.partition(['region'])

with Pool(4) as pool:
    results = pool.map(process_group, parts)

combined = Tafra.concat(results)
print(combined.rows)
```

???+ example "Output"

    ```
    5
    ```

## group_by vs partition -- Summary

| | `group_by` | `partition` |
|---|---|---|
| **Output rows** | One per group | All original rows |
| **Aggregation** | Yes (sum, mean, etc.) | None |
| **Return type** | Single `Tafra` | `List[Tuple[key, Tafra]]` |
| **SQL analogy** | `GROUP BY` | No direct analogy |
| **Use case** | Summarize | Parallel dispatch |

## iter_fn Parameter

Both `group_by` and `transform` accept an optional `iter_fn` parameter for
functions that operate on the group enumeration index rather than on column
data:

```python
result = t.group_by(
    ['region'],
    {'total_sales': (np.sum, 'sales')},
    iter_fn={'group_id': lambda x: x[0]},
)
```

The function receives an array filled with the group index (0, 1, 2, ...).
