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
Unmatched right-side values are `None`. Analogous to SQL `LEFT JOIN`.

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
