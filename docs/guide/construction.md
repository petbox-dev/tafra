# Construction

Tafra provides several ways to create instances from different data sources.

## From a Dictionary

The most common way to build a `Tafra` is from a `dict` mapping column names to
array-like values. Each value is converted to a 1-D `numpy.ndarray`, and all
columns must have the same length.

```python
import numpy as np
from tafra import Tafra

t = Tafra({
    'id': np.array([1, 2, 3]),
    'name': np.array(['Alice', 'Bob', 'Carol']),
    'score': np.array([91.5, 87.0, 94.2]),
})

print(t.columns)
print(t.rows)
print(t.dtypes)
```

???+ example "Output"

    ```
    ('id', 'name', 'score')
    3
    {'id': 'int64', 'name': 'str', 'score': 'float64'}
    ```

Plain Python lists work too -- they are converted to arrays automatically:

```python
t = Tafra({
    'x': [1, 2, 3],
    'y': [4.0, 5.0, 6.0],
})
```

### Specifying dtypes

Pass a second argument to override inferred dtypes:

```python
t = Tafra(
    {'value': [1, 2, 3]},
    {'value': 'float64'},
)
print(t['value'].dtype)
```

???+ example "Output"

    ```
    float64
    ```

### Scalar broadcast

A scalar or length-1 array is broadcast to match the row count of other columns:

```python
t = Tafra({
    'x': np.array([1, 2, 3]),
    'label': 'constant',       # broadcast to 3 rows
})
print(len(t['label']))
```

???+ example "Output"

    ```
    3
    ```

### Other init forms

The constructor also accepts sequences of 2-tuples, iterators, and
`enumerate` objects:

```python
# From a list of (name, array) tuples
t = Tafra([('a', [1, 2]), ('b', [3, 4])])

# From an iterator
t = Tafra(iter([('a', [1, 2]), ('b', [3, 4])]))

# From enumerate
t = Tafra(enumerate([[10, 20], [30, 40]]))
print(t.columns)
```

???+ example "Output"

    ```
    ('0', '1')
    ```

Integer keys are cast to strings.

### Validation options

```python
# Skip validation for performance (data must already be correct)
t = Tafra({'x': np.array([1, 2, 3])}, validate=False)

# Allow columns of differing lengths (advanced use)
t = Tafra({'x': np.array([1, 2]), 'y': np.array([1, 2, 3])}, check_rows=False)
```

## From a pandas DataFrame

```python
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
t = Tafra.from_dataframe(df)

# Override dtypes
t = Tafra.from_dataframe(df, dtypes={'a': 'float64', 'b': 'float64'})
```

## From a pandas Series

```python
s = pd.Series([10, 20, 30], name='values')
t = Tafra.from_series(s)

# Override dtype
t = Tafra.from_series(s, dtype='float64')
```

## From a Database Cursor

`read_sql` executes a query and builds a `Tafra` directly from the cursor
results. Column names and dtypes are read from `cursor.description`.

```python
import pyodbc

conn = pyodbc.connect('DSN=mydb')
cur = conn.cursor()

t = Tafra.read_sql('SELECT id, name, value FROM my_table', cur)
```

For large result sets, `read_sql_chunks` yields `Tafra` instances in batches:

```python
for chunk in Tafra.read_sql_chunks('SELECT * FROM big_table', cur, chunksize=1000):
    process(chunk)
```

### From records

If you already have an iterable of row tuples and column names:

```python
records = [(1, 'Alice'), (2, 'Bob'), (3, 'Carol')]
columns = ['id', 'name']

t = Tafra.from_records(records, columns)

# With explicit dtypes
t = Tafra.from_records(records, columns, dtypes=['int64', str])
```

## Reading CSV Files

`Tafra.read_csv` reads a CSV with a header row, infers column types from the
first few rows, and returns a `Tafra`.

```python
t = Tafra.read_csv('data.csv')
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `csv_file` | -- | Path string, `Path` object, or open file handle |
| `guess_rows` | `5` | Number of rows used for type inference |
| `missing` | `''` | String value treated as missing/None |
| `dtypes` | `None` | Dict of `{column: dtype}` overrides |
| `**csvkw` | -- | Extra keyword arguments passed to `csv.reader` |

### Type inference

The CSV reader tries types in order of precedence: `int32`, `int64`,
`float64`, `bool`. If none match, the column becomes `StringDType(na_object=None)`.
If a later row fails to parse as the inferred type, the column is
automatically promoted to the next type.

### Override inferred types

```python
# Force 'zip_code' to stay as a string instead of being inferred as int
t = Tafra.read_csv('data.csv', dtypes={'zip_code': 'str'})
```

### Custom delimiter

```python
t = Tafra.read_csv('data.tsv', delimiter='\t')
```

## Auto-conversion with `as_tafra`

`Tafra.as_tafra` converts known types to a `Tafra` or returns the input
unchanged if it is already one:

```python
t = Tafra({'x': [1, 2, 3]})
assert Tafra.as_tafra(t) is t           # no copy

df = pd.DataFrame({'x': [1, 2, 3]})
t2 = Tafra.as_tafra(df)                 # calls from_dataframe

t3 = Tafra.as_tafra({'x': [1, 2, 3]})   # dict constructor
```

## ObjectFormatter

The `ObjectFormatter` handles automatic conversion of `object`-dtype arrays.
By default, `Decimal` values are converted to `float`, and object arrays of
Python strings are converted to numpy `StringDType(na_object=None)`, which
supports `None` values natively.

### Registering custom formatters

```python
from tafra.formatter import ObjectFormatter

fmt = ObjectFormatter()

# Register a formatter for MyType -- key is type(value[0]).__name__
fmt['MyDecimal'] = lambda arr: arr.astype(float)
```

The formatter function must accept an `np.ndarray` and return an `np.ndarray`.
The global formatter used by Tafra is defined in `tafra.base.object_formatter`:

```python
from tafra.base import object_formatter

# Already registered by default:
# object_formatter['Decimal'] = lambda x: x.astype(float)

# Add your own:
object_formatter['Money'] = lambda x: x.astype(float)
```

### String handling

When an object array's first element is a Python `str`, it is automatically
converted to numpy `StringDType(na_object=None)` -- no manual registration
needed. The `na_object=None` enables `None` values in string columns.
