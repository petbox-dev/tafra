# Tafra: a minimalist dataframe

[![PyPI version](https://img.shields.io/pypi/v/tafra.svg)](https://pypi.org/project/tafra/)
[![Python versions](https://img.shields.io/pypi/pyversions/tafra.svg)](https://pypi.org/project/tafra/)
[![Coverage Status](https://coveralls.io/repos/github/petbox-dev/tafra/badge.svg?branch=main)](https://coveralls.io/github/petbox-dev/tafra?branch=main)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-0d9488)](https://petbox-dev.github.io/tafra/)

The `tafra` began life as a thought experiment: how could we reduce the idea
of a da*tafra*me (as expressed in libraries like `pandas` or languages
like R) to its useful essence, while carving away the cruft?
The [original proof of concept](https://usethe.computer/posts/12-typing-groupby.html)
stopped at "group by".

This library expands on the proof of concept to produce a practically
useful `tafra`, which we hope you may find to be a helpful lightweight
substitute for certain uses of `pandas`.

A `tafra` is, more-or-less, a set of named *columns* or *dimensions*.
Each of these is a typed `numpy` array of consistent length, representing
the values for each column by *rows*.

The library provides lightweight syntax for manipulating rows and columns,
support for managing data types, iterators for rows and sub-frames,
`pandas`-like "transform" support and conversion from `pandas` Dataframes,
and SQL-style "group by" and join operations.

| Category | Members |
|---|---|
| Tafra | [Tafra](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra) |
| Aggregations | [Union](https://petbox-dev.github.io/tafra/api/#tafra.group.Union), [GroupBy](https://petbox-dev.github.io/tafra/api/#tafra.group.GroupBy), [Transform](https://petbox-dev.github.io/tafra/api/#tafra.group.Transform), [IterateBy](https://petbox-dev.github.io/tafra/api/#tafra.group.IterateBy), [InnerJoin](https://petbox-dev.github.io/tafra/api/#tafra.group.InnerJoin), [LeftJoin](https://petbox-dev.github.io/tafra/api/#tafra.group.LeftJoin), [CrossJoin](https://petbox-dev.github.io/tafra/api/#tafra.group.CrossJoin) |
| Aggregation Helpers | [union](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.union), [union_inplace](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.union_inplace), [group_by](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.group_by), [transform](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.transform), [iterate_by](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.iterate_by), [inner_join](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.inner_join), [left_join](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.left_join), [cross_join](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.cross_join) |
| Chunking / Partitioning | [chunks](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.chunks), [chunk_rows](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.chunk_rows), [partition](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.partition), [concat](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.concat) |
| Custom Aggregations | [percentile](https://petbox-dev.github.io/tafra/api/#tafra.group.percentile), [geomean](https://petbox-dev.github.io/tafra/api/#tafra.group.geomean), [harmean](https://petbox-dev.github.io/tafra/api/#tafra.group.harmean) |
| Constructors | [as_tafra](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.as_tafra), [from_dataframe](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.from_dataframe), [from_series](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.from_series), [from_records](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.from_records) |
| SQL Readers | [read_sql](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.read_sql), [read_sql_chunks](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.read_sql_chunks) |
| Destructors | [to_records](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.to_records), [to_list](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.to_list), [to_tuple](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.to_tuple), [to_array](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.to_array), [to_pandas](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.to_pandas) |
| Properties | [rows](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.rows), [columns](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.columns), [data](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.data), [dtypes](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.dtypes), [size](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.size), [ndim](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.ndim), [shape](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.shape) |
| Iter Methods | [iterrows](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.iterrows), [itertuples](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.itertuples), [itercols](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.itercols) |
| Functional Methods | [row_map](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.row_map), [tuple_map](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.tuple_map), [col_map](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.col_map), [pipe](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.pipe) |
| Dict-like Methods | [keys](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.keys), [values](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.values), [items](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.items), [get](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.get), [update](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.update), [update_inplace](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.update_inplace), [update_dtypes](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.update_dtypes), [update_dtypes_inplace](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.update_dtypes_inplace) |
| Data Exploration | [head](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.head), [tail](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.tail), [sort](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.sort), [sample](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.sample), [describe](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.describe), [value_counts](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.value_counts), [drop_duplicates](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.drop_duplicates) |
| Time Series | [shift](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.shift) |
| Other Helper Methods | [select](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.select), [copy](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.copy), [rename](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.rename), [rename_inplace](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.rename_inplace), [coalesce](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.coalesce), [coalesce_inplace](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.coalesce_inplace), [_coalesce_dtypes](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra._coalesce_dtypes), [delete](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.delete), [delete_inplace](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.delete_inplace) |
| Printer Methods | [pprint](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.pprint), [pformat](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.pformat), [to_html](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra.to_html) |
| Indexing Methods | [_slice](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra._slice), [_index](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra._index), [_ndindex](https://petbox-dev.github.io/tafra/api/#tafra.base.Tafra._ndindex) |

## Getting Started

```shell
pip install tafra
```

Or from conda-forge:

```shell
conda install tafra -c conda-forge
```

Both provide pre-built wheels with the C extension compiled for your platform.
No compiler needed.

### Building from source

To build from source (including the optional C extension):

```shell
git clone https://github.com/petbox-dev/tafra.git
cd tafra
pip install -e .
```

**Requirements:**

- Python >=3.10
- numpy >=2.1
- A C compiler (optional, for the `_accel` extension):
  - **Windows**: Visual Studio Build Tools (with Windows SDK) or MinGW-w64
  - **Linux**: `gcc` (usually pre-installed, or `apt install build-essential`)
  - **macOS**: Xcode Command Line Tools (`xcode-select --install`)

If no C compiler is available, the package installs without the extension and
falls back to pure Python + numpy at runtime. To verify the C extension is
active:

```python
>>> from tafra._accel import groupby_sum
>>> print("C extension active")
```

To build a distributable wheel:

```shell
pip install build
python -m build
```

#### Windows build notes

The C extension requires the MSVC compiler to find the Windows SDK headers.
If you get `fatal error C1083: Cannot open include file: 'io.h'`, the
Windows SDK include/lib paths are not set. Two options:

1. **Use a Developer Command Prompt** (recommended): Open "Developer Command
   Prompt for VS" or "Developer PowerShell for VS" from the Start menu. This
   runs `vcvarsall.bat` automatically and sets all required paths.

2. **Use MinGW-w64** instead of MSVC:

   ```shell
   python setup.py build_ext --inplace --compiler=mingw32
   ```

   MinGW-w64 can be installed via conda (`conda install m2w64-gcc -c
   conda-forge`) or from [winlibs.com](https://winlibs.com/).

If building with `python -m build` (which creates an isolated environment),
use `--no-isolation` to inherit your shell's environment variables, or run
from a Developer Command Prompt:

```shell
python -m build --no-isolation
```

### A short example

```python
>>> from tafra import Tafra

>>> t = Tafra({
...    'x': np.array([1, 2, 3, 4]),
...    'y': np.array(['one', 'two', 'one', 'two']),
... })

>>> t.pformat()
Tafra(data = {
 'x': array([1, 2, 3, 4]),
 'y': array(['one', 'two', 'one', 'two'])},
dtypes = {
 'x': 'int', 'y': 'str'},
rows = 4)

>>> print('List:', '\n', t.to_list())
List:
 [array([1, 2, 3, 4]), array(['one', 'two', 'one', 'two'], dtype=object)]

>>> print('Records:', '\n', tuple(t.to_records()))
Records:
 ((1, 'one'), (2, 'two'), (3, 'one'), (4, 'two'))

>>> gb = t.group_by(
...     ['y'], {'x': sum}
... )

>>> print('Group By:', '\n', gb.pformat())
Group By:
Tafra(data = {
 'x': array([4, 6]), 'y': array(['one', 'two'])},
dtypes = {
 'x': 'int', 'y': 'str'},
rows = 2)
```

### group_by vs partition

`group_by` **reduces** -- one row per group, applies aggregation functions:

```python
>>> tf.group_by(['wellid'], {'total_oil': (np.sum, 'oil')})
# Returns: one row per wellid, with summed oil
```

`partition` **splits** -- returns all original rows, grouped into sub-Tafras
for independent processing (e.g., multiprocessing):

```python
>>> from concurrent.futures import ProcessPoolExecutor

>>> def forecast_well(tf):
...     """Run a forecast on one well's production data."""
...     # tf contains all rows for a single well, sorted by date
...     return compute_forecast(tf['date'], tf['oil'])

>>> parts = tf.partition(['wellid'], sort_by=['date'])

>>> with ProcessPoolExecutor(max_workers=4) as pool:
...     results = list(pool.map(
...         forecast_well, [sub for _, sub in parts]))

>>> combined = Tafra.concat(results)
```

With 8 workers and ~13 ms of work per group, `partition` achieves ~5x
speedup over serial execution. For light aggregations (sum, mean, std),
`group_by` is 10-100x faster -- use it instead. See
[benchmarks](https://petbox-dev.github.io/tafra/benchmarks/) for
detailed benchmarks.

`chunks` splits by row count (for data-parallel workloads where group
integrity doesn't matter):

```python
>>> for chunk in tf.chunks(n=4, sort_by=['date']):
...     process(chunk)
```

### Flexibility

Have some code that works with `pandas`, or just a way of doing things
that you prefer? `tafra` is flexible:

```python
>>> df = pd.DataFrame(np.c_[
...     np.array([1, 2, 3, 4]),
...     np.array(['one', 'two', 'one', 'two'])
... ], columns=['x', 'y'])

>>> t = Tafra.from_dataframe(df)
```

And going back is just as simple:

```python
>>> df = pd.DataFrame(t.data)
```

## Timings

> **Note:** Benchmarks collected with `tafra` 2.2.0. See
> [benchmarks](https://petbox-dev.github.io/tafra/benchmarks/)
> for full benchmarks against `pandas` 2.3/3.0 and `polars` 1.39.

Lightweight means performant. By minimizing abstraction to access the
underlying `numpy` arrays, `tafra` provides dramatic speedups over
`pandas` and `polars` on construction and access:

```python
# Construction: 100k rows, 5 columns
Tafra():         0.01 ms
pd.DataFrame():  4.22 ms   # 422x slower
pl.DataFrame():  0.03 ms   # 3x slower

# Column access: 100k rows, per call
tf['x']:         0.09 µs
df['x']:        11.47 µs   # 127x slower
plf['x']:        0.57 µs   # 6x slower
```

`tafra` uses vectorized numpy operations (`np.bincount`,
`ufunc.reduceat`) and an optional C extension (single-pass aggregation,
hash-based composite key encoding, hash joins) for GroupBy and joins:

```python
# GroupBy: 10k rows, 50 groups, sum + mean
Tafra+C: 0.15 ms
pandas:  0.71 ms   # 5x slower
polars:  0.54 ms   # 4x slower

# Transform: 1M rows, 1k groups
Tafra+C: 8.44 ms
pandas:  20.90 ms  # 2.5x slower
polars:  9.62 ms   # 1.1x slower

# Numba JIT: 1M rows
Tafra:   7.74 ms
pandas:  7.81 ms   # same (numpy underneath)
polars:  7.87 ms   # +2% (arrow→numpy conversion)
```

### Dtype metadata

Each `Tafra` tracks column dtypes in `_dtypes` — a dict of user-declared
type labels (e.g. `'str'`, `'int64'`, `'float64'`). This metadata is the
source of truth for dtype validation in joins, unions, and dtype updates.
Use `update_dtypes_inplace` to change a column's type:

```python
>>> t.update_dtypes_inplace({'x': 'str'})  # converts to StringDType
>>> t.update_dtypes_inplace({'x': 'float64'})  # converts to float64
```

If you assign directly to `Tafra.data` or `Tafra._data`, you *must* call
`Tafra._coalesce_dtypes()` to resync the metadata.

### Left join null handling

When a left join has unmatched rows, right-side columns are filled with
native null values where possible:

- **String columns** → `StringDType(na_object=None)` with `None`
- **Float columns** → original dtype with `NaN`
- **Datetime/timedelta columns** → original dtype with `NaT`
- **Int/bool/bytes columns** → `object` dtype with `None` (a warning is emitted)
