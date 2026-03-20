# Numerical Performance

## Summary

One of the goals of `tafra` is to provide a fast-as-possible data structure
for numerical computing. To achieve this, all function returns are written
as [generator expressions](https://www.python.org/dev/peps/pep-0289/) wherever
possible.

!!! note

    Benchmarks collected on Windows 11 (Python 3.11). Library versions listed
    at the bottom of this page.

Additionally, because the `data` contains values of ndarrays, the
`map` functions may also take functions that operate on ndarrays. This means
that they are able to take [numba](http://numba.pydata.org/) `@jit`'ed
functions as well.


## Construction & Access

`Tafra` wraps a plain `dict` of `numpy` arrays, so construction and
column access have minimal overhead compared to `pandas`:

```python
from tafra import Tafra
import pandas as pd
import polars as pl
import numpy as np

data = {f'col{i}': np.random.randn(100_000) for i in range(5)}

tf = Tafra(data)          # 0.01 ms
df = pd.DataFrame(data)   # 3.08 ms
plf = pl.DataFrame(data)  # 0.03 ms

# Column access
x = tf['col0']   # 0.09 us per access
x = df['col0']   # 11.5 us
x = plf['col0']  # 0.56 us
```

| Operation | tafra | pandas | polars |
|---|---|---|---|
| Construction (100k rows, 5 cols) | **0.01 ms** | 3.08 ms (308x) | 0.03 ms (3x) |
| Column access (per call) | **0.09 us** | 11.5 us (128x) | 0.56 us (6.2x) |

<div class="chart">
  <div class="chart-title">Construction: 100k rows, 5 columns (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 1%"></div>
      <span class="chart-value">0.01</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 1%"></div>
      <span class="chart-value">0.03</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">3.08</span>
    </div>
  </div>
</div>

`pandas` 3.0 introduced copy-on-write semantics and additional safety checks
in column access, significantly increasing per-access overhead. `polars` is
faster than `pandas` but still 6x slower than `Tafra`'s direct dict lookup.


## Row Mapping

Row-wise mapping applies a Python function to each row. `Tafra` uses
`tuple_map` (NamedTuple access), `pandas` uses `itertuples`, and `polars`
uses `map_elements` on a struct column. All apply a scalar function per row:

```python
import math

def row_fn(a: float, b: float, c: float) -> float:
    return math.sqrt(a * a + b * b) + math.log1p(abs(c))

# tafra (name=None for fast plain-tuple iteration)
result = list(tf.tuple_map(lambda r: (row_fn(r[0], r[1], r[2]),), name=None))

# pandas
result = [row_fn(r.a, r.b, r.c) for r in df.itertuples()]

# polars (map_elements on struct)
result = plf.with_columns(
    pl.struct(['a', 'b', 'c']).map_elements(
        lambda s: row_fn(s['a'], s['b'], s['c']),
        return_dtype=pl.Float64
    ).alias('result')
)
```

| Scale | tafra | pandas | polars |
|---|---|---|---|
| 10k rows | **6.11 ms** | 6.77 ms | 6.30 ms |
| 100k rows | 69.6 ms | 66.9 ms | **48.3 ms** |
| 1M rows | 744 ms | 777 ms | **598 ms** |

<div class="chart">
  <div class="chart-title">Row Map: 10k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 90%"></div>
      <span class="chart-value">6.11</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 93%"></div>
      <span class="chart-value">6.30</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">6.77</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Row Map: 100k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 69%"></div>
      <span class="chart-value">48.3</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 96%"></div>
      <span class="chart-value">66.9</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">69.6</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Row Map: 1M rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 77%"></div>
      <span class="chart-value">598</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 96%"></div>
      <span class="chart-value">744</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">777</span>
    </div>
  </div>
</div>

With `name=None` (plain tuple fast path), `tafra` wins at 10k rows. At
100k+ rows, `polars` `map_elements` pulls ahead — its Rust-backed struct
iteration is faster than Python tuple unpacking at scale. All three
libraries are within 1.3x of each other, a dramatic improvement over the
old `map_rows` API which was 7x slower.


## Vectorized Expressions

When the computation can be expressed as array operations, all three
libraries avoid Python per-row overhead entirely. Each library uses its
native expression API to evaluate `sqrt(a^2 + b^2) + log1p(|c|)`:

```python
# tafra / numpy — direct array ops
result = np.sqrt(tf['a']**2 + tf['b']**2) + np.log1p(np.abs(tf['c']))

# pandas — same numpy ops work through pandas API
result = np.sqrt(df['a']**2 + df['b']**2) + np.log1p(np.abs(df['c']))

# polars — native expression API
result = plf.with_columns(
    ((pl.col('a')**2 + pl.col('b')**2).sqrt()
     + (pl.col('c').abs() + 1).log()).alias('result')
)
```

| Scale | tafra | pandas | polars |
|---|---|---|---|
| 10k rows | **0.18 ms** | 0.35 ms | 0.97 ms |
| 100k rows | 1.72 ms | 2.10 ms | **1.52 ms** |
| 1M rows | 27.1 ms | 23.5 ms | **9.63 ms** |

<div class="chart">
  <div class="chart-title">Vectorized Expression: 10k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 19%"></div>
      <span class="chart-value">0.18</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 36%"></div>
      <span class="chart-value">0.35</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">0.97</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Vectorized Expression: 100k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 72%"></div>
      <span class="chart-value">1.52</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 82%"></div>
      <span class="chart-value">1.72</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">2.10</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Vectorized Expression: 1M rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 36%"></div>
      <span class="chart-value">9.63</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 87%"></div>
      <span class="chart-value">23.5</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">27.1</span>
    </div>
  </div>
</div>

At small scale (10k rows), `tafra`/numpy wins decisively — 5x faster than
polars — because numpy array operations have minimal dispatch overhead.
At 1M rows, polars' Rust SIMD internals pull ahead (2.8x faster than
tafra). Pandas sits in between, benefiting from numpy under the hood
but paying additional wrapper overhead.


## GroupBy & Transform

For aggregation operations, `pandas` uses optimized C/Cython internals that
are difficult to match in pure Python + numpy. `Tafra` 2.1.0 uses
index-based grouping (`np.unique` + `return_inverse`) rather than
per-group boolean masks, which is considerably faster than earlier versions.

### GroupBy

```python
# GroupBy with two aggregations
result = tf.group_by(
    ['group'],
    {'mean': (np.mean, 'value'), 'sum': (np.sum, 'value')}
)
```

| Scale | tafra+C | tafra | pandas | polars |
|---|---|---|---|---|
| 10k rows, 50 groups | **0.15 ms** | 0.16 ms | 0.71 ms | 0.54 ms |
| 10k rows, 500 groups | **0.18 ms** | 0.20 ms | 0.71 ms | 0.58 ms |
| 100k rows, 100 groups | 1.53 ms | 1.78 ms | 2.54 ms | **0.98 ms** |
| 100k rows, 1k groups | 1.16 ms | 1.38 ms | 2.43 ms | **1.00 ms** |
| 1M rows, 100 groups | 18.88 ms | 25.33 ms | 16.15 ms | **2.46 ms** |
| 1M rows, 10k groups | 24.45 ms | 27.30 ms | 28.41 ms | **6.96 ms** |
| 100k rows, 2 col, ~300 grp | 8.72 ms | 8.78 ms | 9.14 ms | **1.73 ms** |
| 1M rows, 2 col, ~300 grp | 97.07 ms | 100.74 ms | 80.12 ms | **11.47 ms** |

<div class="chart">
  <div class="chart-title">GroupBy: 10k rows, 50 groups (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra+C</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 21%"></div>
      <span class="chart-value">0.15</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 23%"></div>
      <span class="chart-value">0.16</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 76%"></div>
      <span class="chart-value">0.54</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">0.71</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">GroupBy: 100k rows, 1k groups (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 41%"></div>
      <span class="chart-value">1.00</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra+C</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 48%"></div>
      <span class="chart-value">1.16</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 57%"></div>
      <span class="chart-value">1.38</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">2.43</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">GroupBy: 1M rows, 10k groups (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 24%"></div>
      <span class="chart-value">6.96</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra+C</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 86%"></div>
      <span class="chart-value">24.45</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 96%"></div>
      <span class="chart-value">27.30</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">28.41</span>
    </div>
  </div>
</div>

At 10k rows, `Tafra+C` is **3--5x faster** than both `pandas` and
`polars`. At 100k rows, `polars` leads while `Tafra+C` is still faster
than `pandas`. At 1M rows, polars' multithreaded Rust internals pull
ahead (4x faster at 10k groups).

### Transform

```python
# Transform broadcasts aggregation results back to original row count
result = tf.transform(['group'], {'m': (np.mean, 'value')})
```

| Scale | tafra+C | tafra | pandas | polars |
|---|---|---|---|---|
| 10k rows, 50 groups | **0.06 ms** | 0.08 ms | 0.50 ms | 0.50 ms |
| 100k rows, 100 groups | **0.80 ms** | 1.01 ms | 2.11 ms | 1.35 ms |
| 1M rows, 1k groups | **8.44 ms** | 11.85 ms | 20.90 ms | 9.62 ms |

<div class="chart">
  <div class="chart-title">Transform: 10k rows, 50 groups (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra+C</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 12%"></div>
      <span class="chart-value">0.06</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 16%"></div>
      <span class="chart-value">0.08</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">0.50</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">0.50</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Transform: 1M rows, 1k groups (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra+C</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 40%"></div>
      <span class="chart-value">8.44</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 46%"></div>
      <span class="chart-value">9.62</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 57%"></div>
      <span class="chart-value">11.85</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">20.90</span>
    </div>
  </div>
</div>

Transform wins across all scales. At 1M rows, Tafra+C (8.4 ms) still beats
polars (9.6 ms) and pandas (20.9 ms).

### Vectorized fast path

String columns are automatically encoded to integer codes for efficient
grouping -- no performance penalty vs numeric-only groups.

`Tafra`'s vectorized fast path uses `np.bincount` and `ufunc.reduceat`
for recognized aggregations: `np.sum`, `np.mean`, `np.std`, `np.var`,
`np.min`, `np.max`, `np.ptp`, `np.prod`, `np.median`, `np.any`,
`np.all`, `np.count_nonzero`, `len`, `sum`, plus all nan-variants.
Custom aggregations `percentile(q)`, `geomean`, and `harmean` also
hit the fast path.

For unrecognized functions, `Tafra` falls back to calling your Python
function directly on numpy arrays for each group -- fully transparent, no
hidden dispatch or "silent" dtype changes.


## Joins

`Tafra` 2.1.0 uses two join algorithms for equality joins:

- **With C extension**: O(n) hash join implemented in C (`_accel.c`) --
  builds a hash table on the right key, probes with the left key, and
  constructs output index arrays in a single pass.
- **Without C extension**: numpy-native sort-merge join -- `argsort` +
  `searchsorted` to find match ranges, then `np.repeat` with offset
  arithmetic to build index arrays.

For non-equality operators (`<`, `<=`, `>`, `>=`, `!=`), both paths
fall back to a nested-loop approach.

```python
# Inner join on equality key
result = left_tf.inner_join(right_tf, [('key', 'key', '==')])

# Left join
result = left_tf.left_join(right_tf, [('key', 'key', '==')])
```

| Benchmark | tafra+C | tafra | pandas | polars |
|---|---|---|---|---|
| Inner join (1k x 1k) | **0.08 ms** | 0.30 ms | 0.83 ms | 0.45 ms |
| Inner join (5k x 5k) | 3.53 ms | 6.90 ms | 7.88 ms | **1.71 ms** |
| Inner join (10k x 10k) | 19.74 ms | 34.35 ms | 42.38 ms | **6.61 ms** |
| Inner join (50k x 50k) | 486.81 ms | 726.87 ms | 757.35 ms | **148.62 ms** |
| Left join (1k x 1k) | **0.09 ms** | 0.34 ms | 0.77 ms | 2.94 ms |
| Left join (5k x 5k) | 3.72 ms | 7.32 ms | 8.13 ms | **1.82 ms** |
| Left join (50k x 50k) | 496.43 ms | 719.56 ms | 801.21 ms | **148.96 ms** |

<div class="chart">
  <div class="chart-title">Inner Join: 1k x 1k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra+C</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 10%"></div>
      <span class="chart-value">0.08</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 36%"></div>
      <span class="chart-value">0.30</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 54%"></div>
      <span class="chart-value">0.45</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">0.83</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Inner Join: 10k x 10k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 16%"></div>
      <span class="chart-value">6.61</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra+C</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 47%"></div>
      <span class="chart-value">19.74</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 81%"></div>
      <span class="chart-value">34.35</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">42.38</span>
    </div>
  </div>
</div>

With the C hash join, `Tafra` is **6--10x faster** than both `pandas` and
`polars` on small-scale joins (1k x 1k). At 10k x 10k, `polars`' Rust
multithreaded join pulls ahead while `Tafra+C` is still 2x faster than
`pandas`. At 50k x 50k, polars is 3.3x faster than `Tafra+C`, which is
still 1.6x faster than `pandas`. `Tafra`'s join also supports arbitrary
comparison operators (`<`, `<=`, `>`, `>=`, `!=`) in the `on`
clause, which neither `pandas` nor `polars` natively offer.


## Partition & Multiprocessing

`group_by` and `partition` both split data by group values, but serve
different purposes:

- `group_by` **reduces** -- applies aggregation functions and returns one row
  per group. Fast for built-in reducers (vectorized, no Python loop).
- `partition` **splits** -- returns all original rows grouped into sub-Tafras.
  Designed for dispatching expensive per-group computation to worker processes.

For **light aggregations** (sum, mean, std), `group_by` is the right tool --
it's 10-100x faster because it avoids serialization overhead entirely:

```python
# Light work: group_by wins decisively
tf.group_by(['group'], {'mean': (np.mean, 'value'), 'std': (np.std, 'value')})
# 2 ms — vectorized, no IPC

# partition + serial map (same aggregations)
# 10 ms — partition + per-group Python calls
```

For **expensive per-group computation** (model fitting, forecasting, simulation),
`partition` + `ProcessPoolExecutor` scales nearly linearly with workers:

```python
from concurrent.futures import ProcessPoolExecutor

def forecast_well(tf):
    """~13 ms of computation per group."""
    # ... expensive model fit + forecast ...
    return result

parts = tf.partition(['wellid'])

# Serial: processes groups one at a time
results = [forecast_well(sub) for _, sub in parts]

# Parallel: distributes across workers
with ProcessPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(forecast_well, [sub for _, sub in parts]))

combined = Tafra.concat(results)
```

Benchmarks with ~13 ms of work per group, 8 workers:

| Scenario | Serial | 8 Workers | Speedup |
|---|---|---|---|
| 50 groups, 10k rows | 681 ms | **138 ms** | **4.9x** |
| 100 groups, 100k rows | 1,443 ms | **318 ms** | **4.5x** |
| 1,000 groups, 100k rows | 13,535 ms | **2,784 ms** | **4.9x** |

The crossover point depends on per-group work cost. Rule of thumb:

- **< 1 ms per group**: use `group_by` (IPC overhead dominates)
- **> 5 ms per group**: use `partition` + workers (parallelism wins)

`Tafra` supports Python's standard multiprocessing serialization natively
(dataclass + numpy arrays), so no special handling is needed.


## Numba Integration

Because `Tafra`'s `data` contains raw `numpy` arrays, `numba`
`@jit`'ed functions work directly with no adapter layer:

```python
from numba import jit
jit_kw = {'fastmath': True}

@jit(**jit_kw)
def hyp(qi: float, Di: float, bi: float, t: np.ndarray) -> np.ndarray:
    Dn = ((1.0 - Di) ** -bi - 1.0) / bi
    return qi / (1.0 + Dn * bi * t) ** (1.0 / bi)

@jit(**jit_kw)
def ndarray_map(qi, Di, bi, t):
    out = np.zeros((qi.shape[0], t.shape[0]))
    for i in range(qi.shape[0]):
        out[i, :] = hyp(qi[i], Di[i], bi[i], t)
    return out

# ~80 us — essentially zero overhead from Tafra
result = ndarray_map(tf['qi'], tf['Di'], tf['bi'], t)
```

The key difference between the three libraries is how much work sits
between your `numba` function and the underlying array:

```python
# tafra — direct ndarray, zero overhead
result = numba_fn(tf['a'], tf['b'], tf['c'])

# pandas — .values extracts ndarray
result = numba_fn(df['a'].values, df['b'].values, df['c'].values)

# polars — .to_numpy() copies from Arrow
result = numba_fn(plf['a'].to_numpy(), plf['b'].to_numpy(), plf['c'].to_numpy())
```

| Scale | tafra | pandas | polars |
|---|---|---|---|
| 10k rows | **0.06 ms** | 0.10 ms | 0.17 ms |
| 100k rows | 0.58 ms | 0.69 ms | **0.57 ms** |
| 1M rows | **7.38 ms** | 8.26 ms | 8.32 ms |

<div class="chart">
  <div class="chart-title">Numba: 10k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 35%"></div>
      <span class="chart-value">0.06</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 59%"></div>
      <span class="chart-value">0.10</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">0.17</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Numba: 100k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 83%"></div>
      <span class="chart-value">0.57</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 84%"></div>
      <span class="chart-value">0.58</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">0.69</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Numba: 1M rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 89%"></div>
      <span class="chart-value">7.38</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 99%"></div>
      <span class="chart-value">8.26</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">8.32</span>
    </div>
  </div>
</div>

`tafra` wins at small scale (10k rows) because `tf['col']` **is** the
ndarray -- zero overhead. `pandas` pays the `.values` accessor cost.
`polars` pays the Arrow-to-numpy conversion cost. At 100k rows it's a
near-tie (all three within 0.12 ms). At 1M rows `tafra` wins again --
the accumulated overhead of `.values` and `.to_numpy()` across three
columns adds up.


## When to Use Tafra

`Tafra` is fastest when your workload is dominated by:

* **Construction and teardown** -- 308x faster than pandas, 3x faster
  than polars
* **Column access** -- 128x faster than pandas, 6x faster than polars
* **Row-wise mapping** -- with `name=None` fast path, fastest at 10k rows;
  polars `map_elements` wins at 100k+ but all three are within 1.3x
* **Vectorized expressions** -- tafra wins at small scale (5x faster
  than polars at 10k rows); polars SIMD wins at 1M rows (2.8x faster)
* **GroupBy at <=10k rows** -- with C extension, 3--5x faster
  than both pandas and polars
* **Transform at all scales** -- Tafra+C wins every benchmark, from 8x
  faster than pandas at 10k rows to 2.5x faster at 1M rows; beats polars
  at 1M rows (8.4 ms vs 9.6 ms)
* **Small-scale joins** -- with C extension, equi-joins at 1k x 1k are
  6--10x faster than both pandas and polars
* **Numba-accelerated computation** -- direct `ndarray` access with zero
  adapter overhead

`polars` is fastest for:

* **GroupBy at >=100k rows** -- Rust multithreaded internals (2--7x faster
  than Tafra depending on scale and group count)
* **Large-scale joins** -- Rust multithreaded hash-join at 50k+ rows
  (3.3x faster than Tafra+C)

`pandas` 3.0 is the slowest of the three on nearly every benchmark due to
copy-on-write overhead. At 1M-row Transform, pandas (20.9 ms) is 2.5x slower
than Tafra+C (8.4 ms).

The general pattern: `Tafra` wins on everything up to ~10k rows and remains
competitive at 100k for single-column operations. `polars` pulls ahead at
100k+ rows where its Rust multithreaded internals dominate. The optional C
extension closes much of the remaining gap -- without it, `Tafra` still beats
pandas everywhere and is competitive with polars at moderate scales.


## Summary Table

All times in milliseconds. Lower is better. **Bold** = fastest.

Tafra+C = with optional C extension. Tafra = pure Python + numpy only.

| Benchmark | Tafra+C | Tafra | pandas | polars |
|---|---|---|---|---|
| Construction (100k rows) | **0.01** | 0.01 | 3.08 | 0.03 |
| Column access (per call, us) | **0.09** | 0.09 | 11.5 | 0.56 |
| Row map (10k rows) | **6.11** | 6.11 | 6.77 | 6.30 |
| Row map (100k rows) | 69.6 | 69.6 | 66.9 | **48.3** |
| Row map (1M rows) | 744 | 744 | 777 | **598** |
| Vectorized expr (10k rows) | **0.18** | 0.18 | 0.35 | 0.97 |
| Vectorized expr (100k rows) | 1.72 | 1.72 | 2.10 | **1.52** |
| Vectorized expr (1M rows) | 27.1 | 27.1 | 23.5 | **9.63** |
| Numba (10k rows) | **0.06** | 0.06 | 0.10 | 0.17 |
| Numba (100k rows) | 0.58 | 0.58 | 0.69 | **0.57** |
| Numba (1M rows) | **7.38** | 7.38 | 8.26 | 8.32 |
| GroupBy (10k, 50 grp, sum+mean) | **0.15** | 0.16 | 0.71 | 0.54 |
| GroupBy (10k, 500 grp) | **0.18** | 0.20 | 0.71 | 0.58 |
| GroupBy (100k, 100 grp) | 1.53 | 1.78 | 2.54 | **0.98** |
| GroupBy (100k, 1k grp) | 1.16 | 1.38 | 2.43 | **1.00** |
| GroupBy (1M, 100 grp) | 18.88 | 25.33 | 16.15 | **2.46** |
| GroupBy (1M, 10k grp) | 24.45 | 27.30 | 28.41 | **6.96** |
| GroupBy (100k, 2 col, ~300 grp) | 8.72 | 8.78 | 9.14 | **1.73** |
| GroupBy (1M, 2 col, ~300 grp) | 97.07 | 100.74 | 80.12 | **11.47** |
| Transform (10k, 50 grp) | **0.06** | 0.08 | 0.50 | 0.50 |
| Transform (100k, 100 grp) | **0.80** | 1.01 | 2.11 | 1.35 |
| Transform (1M, 1k grp) | **8.44** | 11.85 | 20.90 | 9.62 |
| Inner join (1k x 1k) | **0.08** | 0.30 | 0.83 | 0.45 |
| Inner join (5k x 5k) | 3.53 | 6.90 | 7.88 | **1.71** |
| Inner join (10k x 10k) | 19.74 | 34.35 | 42.38 | **6.61** |
| Inner join (50k x 50k) | 486.81 | 726.87 | 757.35 | **148.62** |
| Left join (1k x 1k) | **0.09** | 0.34 | 0.77 | 2.94 |
| Left join (5k x 5k) | 3.72 | 7.32 | 8.13 | **1.82** |
| Left join (50k x 50k) | 496.43 | 719.56 | 801.21 | **148.96** |

---

*Benchmarks collected with tafra 2.1.0, pandas 3.0.1, polars 1.39.0, numpy 2.2.5, numba 0.61.2 on Windows 11 (Python 3.11). C extension active.*
