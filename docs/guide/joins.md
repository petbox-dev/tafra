# Joins

Tafra supports SQL-style joins: inner, left, and cross. The `on` parameter
takes a list of tuples specifying left column, right column, and comparison
operator.

## Join syntax

```python
result = left.inner_join(right, on=[('left_col', 'right_col', '==')])
```

The `on` parameter is a list of `(left_column, right_column, operator)` tuples.
Valid operators:

| Operator | Meaning |
|----------|---------|
| `'=='` | Equal to |
| `'!='` | Not equal to |
| `'<'` | Less than |
| `'<='` | Less than or equal to |
| `'>'` | Greater than |
| `'>='` | Greater than or equal to |

The optional `select` parameter limits which columns appear in the output.
If omitted, all unique column names are returned, preferring the left table
when a column exists in both.

## Inner Join

Returns only rows where the join condition matches in both tables. Analogous
to SQL `INNER JOIN`.

```python
import numpy as np
from tafra import Tafra

orders = Tafra({
    'order_id': np.array([1, 2, 3, 4]),
    'customer_id': np.array([10, 20, 30, 40]),
    'amount': np.array([100.0, 200.0, 150.0, 300.0]),
})

customers = Tafra({
    'customer_id': np.array([10, 20, 50]),
    'name': np.array(['Alice', 'Bob', 'Eve']),
})

result = orders.inner_join(
    customers,
    on=[('customer_id', 'customer_id', '==')],
)
```

???+ example "Output"

    ```
    order_id  customer_id  amount  name
    1         10           100.0   'Alice'
    2         20           200.0   'Bob'
    ```

    `customer_id` 30 and 40 have no match -- excluded.
    `customer_id` 50 has no match in orders -- excluded.

### SQL equivalent

```sql
SELECT *
FROM orders
INNER JOIN customers
  ON orders.customer_id = customers.customer_id
```

### Selecting specific columns

```python
result = orders.inner_join(
    customers,
    on=[('customer_id', 'customer_id', '==')],
    select=['order_id', 'name', 'amount'],
)
print(result.columns)
```

???+ example "Output"

    ```
    ('order_id', 'name', 'amount')
    ```

## Left Join

Returns all rows from the left table, with matching rows from the right.
Unmatched right-side columns are filled with native null values where possible.
Analogous to SQL `LEFT JOIN`. See [Null Handling & Dtype Behavior](#null-handling-dtype-behavior)
for details on how each dtype is handled.

```python
result = orders.left_join(
    customers,
    on=[('customer_id', 'customer_id', '==')],
)
```

???+ example "Output"

    ```
    order_id  customer_id  amount  name
    1         10           100.0   'Alice'
    2         20           200.0   'Bob'
    3         30           150.0   None
    4         40           300.0   None
    ```

### SQL equivalent

```sql
SELECT *
FROM orders
LEFT JOIN customers
  ON orders.customer_id = customers.customer_id
```

## Cross Join

Returns the Cartesian product of both tables -- every row from the left paired
with every row from the right. No `on` parameter is needed.

```python
sizes = Tafra({
    'size': np.array(['S', 'M', 'L']),
})

colors = Tafra({
    'color': np.array(['red', 'blue']),
})

result = sizes.cross_join(colors)
```

???+ example "Output"

    ```
    size  color
    'S'   'red'
    'S'   'blue'
    'M'   'red'
    'M'   'blue'
    'L'   'red'
    'L'   'blue'
    ```

    Result has 3 * 2 = 6 rows.

### SQL equivalent

```sql
SELECT *
FROM sizes
CROSS JOIN colors
```

### Selecting columns

```python
result = sizes.cross_join(colors, select=['size', 'color'])
```

## Multi-column Joins

Join on multiple columns by adding more tuples to `on`:

```python
left = Tafra({
    'year': np.array([2023, 2023, 2024]),
    'region': np.array(['east', 'west', 'east']),
    'value': np.array([10, 20, 30]),
})

right = Tafra({
    'year': np.array([2023, 2024]),
    'region': np.array(['east', 'east']),
    'budget': np.array([100, 200]),
})

result = left.inner_join(
    right,
    on=[
        ('year', 'year', '=='),
        ('region', 'region', '=='),
    ],
)
```

???+ example "Output"

    ```
    year  region  value  budget
    2023  'east'  10     100
    2024  'east'  30     200
    ```

## Non-equality Joins

Operators other than `'=='` enable range joins and inequality conditions:

```python
events = Tafra({
    'event_id': np.array([1, 2]),
    'start': np.array([5, 15]),
})

windows = Tafra({
    'window_id': np.array([1, 2, 3]),
    'threshold': np.array([3, 10, 20]),
})

# Find events where start >= threshold
result = events.inner_join(
    windows,
    on=[('start', 'threshold', '>=')],
)
```

???+ example "Output"

    ```
    event_id  start  window_id  threshold
    1         5      1          3
    2         15     1          3
    2         15     2          10
    ```

!!! note
    Non-equality joins use a row-by-row loop and do not benefit from the
    sort-merge or hash-based fast paths. For large datasets, prefer equality
    joins when possible.

## C-accelerated Hash Join

For equality joins (`'=='` operator), Tafra uses an optimized join algorithm.
If the optional C extension (`_accel.c`) is compiled, joins use an O(n)
hash-based implementation. Otherwise, a pure-Python sort-merge algorithm is
used. Both produce identical results -- the C extension is purely a performance
optimization.

Build the C extension with:

```bash
python setup.py build_ext --inplace
```

## Null Handling & Dtype Behavior

### Dtype validation on join keys

Join keys are validated using the `_dtypes` metadata — the user's declared
intent for column types, not the raw numpy array dtype. This means:

- **String variants match**: `StringDType`, `StringDType(na_object=None)`,
  `<U8`, `<U12` all reduce to `'str'` and can join freely.
- **Same-family numeric types are promoted**: `int32` vs `int64`, or
  `float32` vs `float64`, are promoted to the higher precision type using
  `np.result_type`. The narrower column is cast automatically.
- **Different families are rejected**: `int64` vs `float64`, or `int64` vs
  `str`, raise `TypeError`.

| Left key dtype | Right key dtype | Allowed? | Result dtype |
|---|---|---|---|
| `StringDType` | `<U10` | Yes | both keep original dtypes |
| `<U8` | `<U12` | Yes | both keep original dtypes |
| `int32` | `int64` | Yes (promoted) | `int64` |
| `float32` | `float64` | Yes (promoted) | `float64` |
| `int32` | `int32` | Yes | `int32` |
| `int64` | `float64` | No — `TypeError` | — |
| `int64` | `StringDType` | No — `TypeError` | — |
| `datetime64[ns]` | `datetime64[us]` | Yes | both keep original dtypes |

Type promotion uses `np.result_type` under the hood — the same function numpy
uses to determine output dtypes for binary operations. **Promotion applies only
to key matching** — the original input tables are never modified, and the output
column dtype depends on which table the column came from (left takes precedence
for shared column names).

```python
# int32 key joins with int64 key — promoted for matching, not in output
left = Tafra({'k': np.array([1, 2], dtype=np.int32), 'v': np.array([10, 20])})
right = Tafra({'k': np.array([1, 2], dtype=np.int64), 'info': np.array(['a', 'b'])})
result = left.inner_join(right, on=[('k', 'k', '==')])
# Works — k promoted to int64 internally for matching
# result['k'].dtype is int32 (left table wins)
```

### Null values in join keys

Null values in join keys (`None`, `NaN`, `NaT`) follow SQL semantics:
**NULL never equals NULL**. Rows with null keys are excluded from matching.

| Key dtype | Null value | Behavior |
|---|---|---|
| Float | `NaN` | Excluded from matching |
| String (`StringDType(na_object=None)`) | `None` | Excluded from matching |
| Datetime (`datetime64`) | `NaT` | Excluded from matching |
| Timedelta (`timedelta64`) | `NaT` | Excluded from matching |

For **inner joins**, rows with null keys are dropped from both sides — they
produce no output rows.

For **left joins**, left-side rows with null keys are preserved in the output
with null-filled right columns (since they can't match anything).

```python
left = Tafra({
    'k': np.array([1.0, np.nan, 3.0]),
    'v': np.array([10, 20, 30]),
})
right = Tafra({
    'k': np.array([1.0, np.nan, 3.0]),
    'info': np.array(['a', 'b', 'c']),
})

# Inner join: NaN rows excluded from both sides
result = left.inner_join(right, on=[('k', 'k', '==')])
# Result: k=[1.0, 3.0], v=[10, 30], info=['a', 'c']
# NaN row excluded — NaN != NaN

# Left join: NaN row preserved with null right columns
result = left.left_join(right, on=[('k', 'k', '==')])
# Result: k=[1.0, NaN, 3.0], v=[10, 20, 30], info=['a', None, 'c']
```

!!! tip
    Avoid using float columns as join keys. If you must, filter out NaN
    values first, or cast to integer IDs.

### Left join null-fill by dtype

When a left join has unmatched rows, right-side columns are filled with
native null values where the dtype supports them. Tafra preserves the
original column precision — `float32` stays `float32`, not promoted to
`float64`.

| Column dtype | Null value | Result dtype | Warning? |
|---|---|---|---|
| `StringDType` | `None` | `StringDType(na_object=None)` | No |
| `<U` (fixed unicode) | `None` | `StringDType(na_object=None)` | No |
| `float32` | `NaN` | `float32` (preserved) | No |
| `float64` | `NaN` | `float64` (preserved) | No |
| `datetime64` | `NaT` | original dtype preserved | No |
| `timedelta64` | `NaT` | original dtype preserved | No |
| `int8`/`int16`/`int32`/`int64` | `None` | `object` | **Yes** |
| `bool` | `None` | `object` | **Yes** |
| `bytes` | `None` | `object` | **Yes** |

For integer, boolean, and bytes columns, numpy has no native null
representation. These columns fall back to `object` dtype and a warning is
emitted:

```
UserWarning: Left join: column 'count' (dtype int64) has unmatched rows
and no native null representation. Dtype has been cast to object.
Use .astype(float) if NaN semantics are needed.
```

To recover a numeric dtype after the join:

```python
result = left.left_join(right, on=[('k', 'k', '==')])

# Convert object column with None to float with NaN
result['count'] = np.where(
    result['count'] == None, np.nan,  # noqa: E711
    result['count'],
).astype(float)
```

### Type promotion in join results

**Shared column names**: When both tables have a column with the same name,
the left table's version takes precedence. No type promotion occurs — the
right column is simply excluded.

**Null-fill preserves precision**: `float32` columns stay `float32` after
null-fill (not widened to `float64`). The null value (`NaN`) fits in any
float width.

**String promotion**: `<U` (fixed-width unicode) columns are promoted to
`StringDType(na_object=None)` when null values are needed, since fixed-width
unicode cannot represent `None`.

### Empty table joins

Joining with an empty table (zero rows) is allowed but emits a warning,
since it usually indicates unexpected data:

```
UserWarning: Join: one or both tables have zero rows.
Returning shortcut result.
```

| Scenario | Result |
|---|---|
| Inner join, either table empty | Empty Tafra with correct columns/dtypes |
| Left join, left empty | Empty Tafra with correct columns/dtypes |
| Left join, right empty | Left table with null-filled right columns |
| Cross join, either table empty | Empty Tafra with correct columns/dtypes |

### Column selection

The `select` parameter controls which columns appear in the output.

- If omitted, all unique column names are returned
- Column order follows left-table-then-right-table order (not `select` order)
- `select` acts as a **filter**, not an ordering specification
- Nonexistent column names are silently ignored
- When a column name exists in both tables, the left table's version is used

```python
result = left.inner_join(
    right,
    on=[('k', 'k', '==')],
    select=['k', 'info', 'v'],  # acts as filter, order follows table order
)
print(result.columns)  # ('k', 'v', 'info') — left columns first, then right
```

### Non-equality join performance

Non-equality operators (`!=`, `<`, `<=`, `>`, `>=`) use an O(n × m)
row-by-row loop. They do not benefit from the C extension or sort-merge
optimization. This applies even when mixed with equality conditions —
if any condition uses a non-equality operator, the entire join falls back
to the row-by-row path.

For large datasets, consider restructuring the query to use an equality
join followed by a filter:

```python
# Instead of: left.inner_join(right, on=[('date', 'cutoff', '>=')])
# Do:
result = left.cross_join(right)
mask = result['date'] >= result['cutoff']
result = result._slice(np.where(mask)[0])
```

!!! note "Row ordering"
    Join results are not guaranteed to be in any particular order. The C
    extension (hash-based) and Python fallback (sort-merge) may produce
    rows in different orders. If you need a specific sort order, call
    `result.sort('column')` after the join. This matches SQL semantics —
    `SELECT` without `ORDER BY` has no guaranteed ordering.
