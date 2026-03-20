# Getting Started

## Installation

Install from **PyPI** (includes pre-built C extension for all major platforms):

```bash
pip install tafra
```

Or from **conda-forge**:

```bash
conda install tafra -c conda-forge
```

Both methods install pre-built wheels with the optional C extension already
compiled. No compiler needed.

## Your First Tafra

A `Tafra` is a set of named columns, each a typed numpy array of the same
length -- like a stripped-down dataframe backed directly by numpy.

### Construct from a dict

```python
import numpy as np
from tafra import Tafra

t = Tafra({
    'x': np.array([1, 2, 3, 4]),
    'y': np.array(['one', 'two', 'one', 'two']),
})
```

### Access columns

Column access returns the underlying numpy array directly -- no wrapper objects:

```python
print(t['x'])
print(t['y'])
```

???+ example "Output"

    ```
    [1 2 3 4]
    ['one' 'two' 'one' 'two']
    ```

### Inspect the contents

```python
print(t.pformat())
```

???+ example "Output"

    ```
    Tafra(data = {
     'x': array([1, 2, 3, 4]),
     'y': array(['one', 'two', 'one', 'two'])},
    dtypes = {
     'x': 'int', 'y': 'str'},
    rows = 4)
    ```

### Properties

```python
print(t.rows)
print(t.columns)
print(t.dtypes)
print(t.shape)
```

???+ example "Output"

    ```
    4
    ['x', 'y']
    {'x': 'int', 'y': 'str'}
    (4, 2)
    ```

### Iterate rows

```python
# As named tuples
for row in t.itertuples():
    print(row.x, row.y)

# As single-row Tafra objects
for row in t.iterrows():
    print(row['x'], row['y'])
```

### Select and slice

```python
# Select specific columns (returns a new Tafra, no copy)
t.select(['x'])

# Slice rows with numpy indexing
t._slice(slice(0, 2))   # first two rows
t._index(np.array([0, 3]))  # rows 0 and 3
```

### Read from CSV

```python
t = Tafra.read_csv('data.csv', dtypes={'name': 'str'})
```

### Convert to and from pandas

```python
import pandas as pd

# pandas -> Tafra
df = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
t = Tafra.from_dataframe(df)

# Tafra -> pandas
df = pd.DataFrame(t.data)
```

## Basic Operations

### GroupBy -- aggregate to one row per group

`group_by` reduces rows by applying aggregation functions to each group, like
SQL `GROUP BY`:

```python
t = Tafra({
    'region': np.array(['east', 'east', 'west', 'west']),
    'sales':  np.array([100, 200, 150, 250]),
    'units':  np.array([10, 20, 15, 25]),
})

result = t.group_by(
    ['region'],
    {'sales': np.sum, 'units': np.sum},
)
print(result['region'])
print(result['sales'])
print(result['units'])
```

???+ example "Output"

    ```
    ['east' 'west']
    [300 400]
    [30 40]
    ```

You can also rename columns during aggregation:

```python
result = t.group_by(
    ['region'],
    {'total_sales': (np.sum, 'sales'), 'avg_sales': (np.mean, 'sales')},
)
```

`GroupBy` detects known numpy reducers (`np.sum`, `np.mean`, `np.std`,
`np.min`, `np.max`, `np.median`, `np.prod`, `len`, etc.) and uses vectorized
`np.bincount`/`ufunc.reduceat` instead of per-group Python loops.

### Transform -- aggregate and broadcast back

`transform` groups like `group_by`, but broadcasts the result back to the
original row count (like `pandas.groupby().transform()`):

```python
t = Tafra({
    'region': np.array(['east', 'east', 'west', 'west']),
    'sales':  np.array([100, 200, 150, 250]),
})

result = t.transform(
    ['region'],
    {'sales': np.sum},
)
print(result['sales'])
```

???+ example "Output"

    ```
    [300 300 400 400]
    ```

    `result` has 4 rows, same as the input.

### Inner Join

Joins use SQL-style `(left_col, right_col, operator)` tuples:

```python
left = Tafra({
    'id':   np.array([1, 2, 3]),
    'name': np.array(['Alice', 'Bob', 'Carol']),
})

right = Tafra({
    'id':    np.array([2, 3, 4]),
    'score': np.array([85, 92, 78]),
})

joined = left.inner_join(
    right,
    on=[('id', 'id', '==')],
)
print(joined['id'])
print(joined['name'])
print(joined['score'])
```

???+ example "Output"

    ```
    [2 3]
    ['Bob' 'Carol']
    [85 92]
    ```

`left_join` and `cross_join` follow the same pattern.

### IterateBy -- grouped iteration

`iterate_by` yields each group as a sub-Tafra for custom processing:

```python
for keys, indices, sub_tafra in t.iterate_by(['region']):
    print(f"Region: {keys}, rows: {sub_tafra.rows}")
    # process sub_tafra however you like
```

### Partition -- split for parallel processing

`partition` splits a Tafra into sub-Tafras by group values, designed for
`multiprocessing` dispatch. Unlike `group_by`, it preserves all original rows:

```python
from concurrent.futures import ProcessPoolExecutor

parts = t.partition(['region'], sort_by=['sales'])

with ProcessPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(
        process_fn, [sub for _, sub in parts]
    ))

combined = Tafra.concat(results)
```

## Building from Source

Only needed for development — `pip install tafra` already includes
pre-built wheels with the C extension.

```bash
git clone https://github.com/petbox-dev/tafra.git
cd tafra
pip install -e .
```

### Requirements

- Python >= 3.9
- numpy >= 2.1
- A C compiler (for the `_accel` extension):
    - **Windows**: Visual Studio Build Tools (with Windows SDK) or MinGW-w64
    - **Linux**: `gcc` (usually pre-installed, or `apt install build-essential`)
    - **macOS**: Xcode Command Line Tools (`xcode-select --install`)

### Build the C extension

```bash
python setup.py build_ext --inplace
```

On Windows with MinGW:

```bash
python setup.py build_ext --inplace --compiler=mingw32
```

### Verify the C extension is active

```python
from tafra._accel import groupby_sum
print("C extension active")
```

### Build a distributable wheel

```bash
pip install build
python -m build
```

### Windows build notes

The C extension requires the MSVC compiler to find the Windows SDK headers. If
you get `fatal error C1083: Cannot open include file: 'io.h'`, the Windows SDK
include/lib paths are not set. Two options:

1. **Use a Developer Command Prompt** (recommended): Open "Developer Command
   Prompt for VS" or "Developer PowerShell for VS" from the Start menu. This
   runs `vcvarsall.bat` automatically and sets all required paths.

2. **Use MinGW-w64** instead of MSVC:

    ```bash
    python setup.py build_ext --inplace --compiler=mingw32
    ```

    MinGW-w64 can be installed via conda (`conda install m2w64-gcc -c
    conda-forge`) or from [winlibs.com](https://winlibs.com/).

If building with `python -m build` (which creates an isolated environment), use
`--no-isolation` to inherit your shell's environment variables, or run from a
Developer Command Prompt:

```bash
python -m build --no-isolation
```

## Next Steps

- [**API Reference**](api.md) -- full documentation of all classes and methods
- [**Benchmarks**](benchmarks.md) -- performance comparisons against pandas and polars
- [**Changelog**](changelog.md) -- version history and release notes
- [**GitHub**](https://github.com/petbox-dev/tafra) -- source code, issues, and contributions
