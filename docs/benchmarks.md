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
df = pd.DataFrame(data)   # 2.83 ms
plf = pl.DataFrame(data)  # 0.03 ms

# Column access
x = tf['col0']   # 0.09 us per access
x = df['col0']   # 10.7 us
x = plf['col0']  # 0.56 us
```

| Operation | tafra | pandas | polars |
|---|---|---|---|
| Construction (100k rows, 5 cols) | **0.01 ms** | 2.83 ms (283x) | 0.03 ms (3x) |
| Column access (per call) | **0.09 us** | 10.7 us (119x) | 0.56 us (6.2x) |

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
      <span class="chart-value">2.83</span>
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
| 10k rows | **5.69 ms** | 6.62 ms | 5.91 ms |
| 100k rows | 62.1 ms | 64.4 ms | **47.0 ms** |
| 1M rows | 643 ms | 649 ms | **477 ms** |

<div class="chart">
  <div class="chart-title">Row Map: 10k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 86%"></div>
      <span class="chart-value">5.69</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 89%"></div>
      <span class="chart-value">5.91</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">6.62</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Row Map: 100k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 73%"></div>
      <span class="chart-value">47.0</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 96%"></div>
      <span class="chart-value">62.1</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">64.4</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Row Map: 1M rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 74%"></div>
      <span class="chart-value">477</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 99%"></div>
      <span class="chart-value">643</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">649</span>
    </div>
  </div>
</div>

With `name=None` (plain tuple fast path), `polars` narrowly wins at 10k rows
with `tafra` within 4%. At 100k+ rows, `polars` `map_elements` pulls
ahead -- its Rust-backed struct iteration is faster than Python tuple
unpacking at scale. `tafra` and `pandas` are within 4% of each other,
while polars is ~30% faster at 1M rows.


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
| 10k rows | **0.07 ms** | 0.32 ms | 0.72 ms |
| 100k rows | **1.57 ms** | 1.84 ms | 1.64 ms |
| 1M rows | 18.8 ms | 19.4 ms | **9.70 ms** |

<div class="chart">
  <div class="chart-title">Vectorized Expression: 10k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 10%"></div>
      <span class="chart-value">0.07</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 44%"></div>
      <span class="chart-value">0.32</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">0.72</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Vectorized Expression: 100k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 85%"></div>
      <span class="chart-value">1.57</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 89%"></div>
      <span class="chart-value">1.64</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">1.84</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Vectorized Expression: 1M rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 50%"></div>
      <span class="chart-value">9.70</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 97%"></div>
      <span class="chart-value">18.8</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">19.4</span>
    </div>
  </div>
</div>

At small scale (10k rows), `tafra`/numpy wins decisively -- 10x faster than
polars -- because numpy array operations have minimal dispatch overhead.
At 100k rows, tafra still leads (1.57 ms vs polars 1.64 ms) with all three
within 17%. At 1M rows, polars' Rust SIMD internals pull ahead (1.9x faster
than tafra). Pandas sits in between, benefiting from numpy under the hood
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
| 10k rows, 50 groups | **0.15 ms** | 0.16 ms | 0.70 ms | 0.50 ms |
| 10k rows, 500 groups | **0.18 ms** | 0.20 ms | 0.73 ms | 0.57 ms |
| 100k rows, 100 groups | **1.11 ms** | 1.28 ms | 2.67 ms | 1.10 ms |
| 100k rows, 1k groups | 1.30 ms | 1.43 ms | 2.45 ms | **1.31 ms** |
| 1M rows, 100 groups | 18.61 ms | 23.61 ms | 15.61 ms | **2.81 ms** |
| 1M rows, 10k groups | 21.15 ms | 25.43 ms | 27.96 ms | **6.20 ms** |
| 100k rows, 2 col, ~300 grp | 4.00 ms | 7.74 ms | 8.76 ms | **1.64 ms** |
| 1M rows, 2 col, ~300 grp | 48.23 ms | 97.27 ms | 79.12 ms | **10.19 ms** |

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
      <div class="chart-bar" style="width: 71%"></div>
      <span class="chart-value">0.50</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">0.70</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">GroupBy: 100k rows, 100 groups (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 41%"></div>
      <span class="chart-value">1.10</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra+C</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 42%"></div>
      <span class="chart-value">1.11</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 48%"></div>
      <span class="chart-value">1.28</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">2.67</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">GroupBy: 1M rows, 10k groups (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 22%"></div>
      <span class="chart-value">6.20</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra+C</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 76%"></div>
      <span class="chart-value">21.15</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 91%"></div>
      <span class="chart-value">25.43</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">27.96</span>
    </div>
  </div>
</div>

At 10k rows, `Tafra+C` is **3--5x faster** than both `pandas` and
`polars`. At 100k rows with 100 groups, `Tafra+C` now matches `polars`
(1.11 ms vs 1.10 ms) -- effectively a dead heat -- while remaining 2.4x
faster than `pandas`. The new C `encode_strings` + `group_indices`
functions cut multi-column GroupBy time in half: 100k rows with 2 string
columns improved from 8.72 ms to 4.00 ms, and 1M rows from 97 ms to 48 ms.
At 1M rows, polars' multithreaded Rust internals pull ahead (3.4x faster
at 10k groups).

### Transform

```python
# Transform broadcasts aggregation results back to original row count
result = tf.transform(['group'], {'m': (np.mean, 'value')})
```

| Scale | tafra+C | tafra | pandas | polars |
|---|---|---|---|---|
| 10k rows, 50 groups | **0.06 ms** | 0.07 ms | 0.49 ms | 0.47 ms |
| 100k rows, 100 groups | **0.82 ms** | 1.01 ms | 2.13 ms | 1.16 ms |
| 1M rows, 1k groups | **8.24 ms** | 12.25 ms | 25.44 ms | 9.76 ms |

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
      <div class="chart-bar" style="width: 14%"></div>
      <span class="chart-value">0.07</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 96%"></div>
      <span class="chart-value">0.47</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">0.49</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Transform: 1M rows, 1k groups (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra+C</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 32%"></div>
      <span class="chart-value">8.24</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 38%"></div>
      <span class="chart-value">9.76</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 48%"></div>
      <span class="chart-value">12.25</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">25.44</span>
    </div>
  </div>
</div>

Transform wins across all scales. At 1M rows, Tafra+C (8.24 ms) still beats
polars (9.76 ms) and pandas (25.44 ms) -- a 3.1x advantage over pandas.

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
| Inner join (1k x 1k) | **0.12 ms** | 0.41 ms | 0.86 ms | 0.45 ms |
| Inner join (5k x 5k) | 3.67 ms | 7.05 ms | 8.17 ms | **1.69 ms** |
| Inner join (10k x 10k) | 15.24 ms | 24.96 ms | 27.88 ms | **3.90 ms** |
| Inner join (50k x 50k) | 389 ms | 537 ms | 654 ms | **128 ms** |
| Left join (1k x 1k) | **0.09 ms** | 0.33 ms | 0.85 ms | 5.81 ms |
| Left join (5k x 5k) | 3.89 ms | 7.69 ms | 7.77 ms | **2.75 ms** |
| Left join (50k x 50k) | 457 ms | 572 ms | 665 ms | **144 ms** |

<div class="chart">
  <div class="chart-title">Inner Join: 1k x 1k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra+C</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 14%"></div>
      <span class="chart-value">0.12</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 48%"></div>
      <span class="chart-value">0.41</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 52%"></div>
      <span class="chart-value">0.45</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">0.86</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Inner Join: 10k x 10k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 14%"></div>
      <span class="chart-value">3.90</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra+C</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 55%"></div>
      <span class="chart-value">15.24</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 89%"></div>
      <span class="chart-value">24.96</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas 3.0</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">27.88</span>
    </div>
  </div>
</div>

With the C hash join, `Tafra` is **4--7x faster** than both `pandas` and
`polars` on small-scale joins (1k x 1k). At 10k x 10k, `polars`' Rust
multithreaded join pulls ahead while `Tafra+C` is still 1.8x faster than
`pandas`. At 50k x 50k, polars is 3.0x faster than `Tafra+C`, which is
still 1.7x faster than `pandas`. `Tafra`'s join also supports arbitrary
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
| 10k rows | **0.06 ms** | 0.09 ms | 0.06 ms |
| 100k rows | 0.56 ms | 0.61 ms | **0.57 ms** |
| 1M rows | **6.88 ms** | 6.95 ms | 6.90 ms |

<div class="chart">
  <div class="chart-title">Numba: 10k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 67%"></div>
      <span class="chart-value">0.06</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 67%"></div>
      <span class="chart-value">0.06</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">0.09</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Numba: 100k rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 92%"></div>
      <span class="chart-value">0.56</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 93%"></div>
      <span class="chart-value">0.57</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">0.61</span>
    </div>
  </div>
</div>

<div class="chart">
  <div class="chart-title">Numba: 1M rows (ms)</div>
  <div class="chart-row">
    <span class="chart-label">tafra</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar fastest" style="width: 99%"></div>
      <span class="chart-value">6.88</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">polars</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 99%"></div>
      <span class="chart-value">6.90</span>
    </div>
  </div>
  <div class="chart-row">
    <span class="chart-label">pandas</span>
    <div class="chart-bar-wrap">
      <div class="chart-bar" style="width: 100%"></div>
      <span class="chart-value">6.95</span>
    </div>
  </div>
</div>

`tafra` ties polars at 10k rows (both 0.06 ms) because `tf['col']` **is**
the ndarray -- zero overhead. `pandas` pays the `.values` accessor cost.
At 100k and 1M rows all three converge -- the actual numba computation
dominates and accessor overhead is negligible. `tafra` still wins at 1M
rows (6.88 ms vs 6.90 ms polars, 6.95 ms pandas) but the differences are
within noise.


## When to Use Tafra

`Tafra` is fastest when your workload is dominated by:

* **Construction and teardown** -- 283x faster than pandas, 3x faster
  than polars
* **Column access** -- 119x faster than pandas, 6x faster than polars
* **Row-wise mapping** -- with `name=None` fast path, within 4% of polars
  at 10k rows; polars `map_elements` wins at 100k+ but tafra and pandas
  are within 4% of each other
* **Vectorized expressions** -- tafra wins at small scale (10x faster
  than polars at 10k rows); tafra wins at 100k; polars SIMD wins at 1M
  rows (1.9x faster)
* **GroupBy at <=10k rows** -- with C extension, 3--5x faster
  than both pandas and polars
* **GroupBy at 100k rows** -- Tafra+C now matches polars at single-column
  100k (1.11 ms vs 1.10 ms); multi-column GroupBy cut in half by new C
  `encode_strings` + `group_indices` (4.00 ms, down from 8.72 ms)
* **Transform at all scales** -- Tafra+C wins every benchmark, from 8x
  faster than pandas at 10k rows to 3.1x faster at 1M rows; beats polars
  at 1M rows (8.24 ms vs 9.76 ms)
* **Small-scale joins** -- with C extension, equi-joins at 1k x 1k are
  4--7x faster than both pandas and polars
* **Numba-accelerated computation** -- direct `ndarray` access with zero
  adapter overhead; ties polars at all scales

`polars` is fastest for:

* **GroupBy at >=1M rows** -- Rust multithreaded internals (3--5x faster
  than Tafra depending on scale and group count)
* **Large-scale joins** -- Rust multithreaded hash-join at 50k+ rows
  (3.0x faster than Tafra+C)

`pandas` 3.0 is the slowest of the three on nearly every benchmark due to
copy-on-write overhead. At 1M-row Transform, pandas (25.4 ms) is 3.1x slower
than Tafra+C (8.24 ms).

The general pattern: `Tafra` wins on everything up to ~10k rows and remains
competitive at 100k for single-column operations -- now matching polars on
100k single-column GroupBy. The new C `composite_key`, `group_indices`, and
`encode_strings` functions halved multi-column GroupBy times. `polars` pulls
ahead at 1M+ rows where its Rust multithreaded internals dominate. The
optional C extension closes much of the remaining gap -- without it, `Tafra`
still beats pandas everywhere and is competitive with polars at moderate
scales.


## Summary Table

All times in milliseconds. Lower is better. **Bold** = fastest.

Tafra+C = with optional C extension. Tafra = pure Python + numpy only.

| Benchmark | Tafra+C | Tafra | pandas | polars |
|---|---|---|---|---|
| Construction (100k rows) | **0.01** | 0.01 | 2.83 | 0.03 |
| Column access (per call, us) | **0.09** | 0.09 | 10.7 | 0.56 |
| Row map (10k rows) | 5.91 | 5.91 | 6.62 | **5.69** |
| Row map (100k rows) | 62.1 | 62.1 | 64.4 | **47.0** |
| Row map (1M rows) | 643 | 643 | 649 | **477** |
| Vectorized expr (10k rows) | **0.07** | 0.07 | 0.32 | 0.72 |
| Vectorized expr (100k rows) | **1.57** | 1.57 | 1.84 | 1.64 |
| Vectorized expr (1M rows) | 18.8 | 18.8 | 19.4 | **9.70** |
| Numba (10k rows) | **0.06** | 0.06 | 0.09 | 0.06 |
| Numba (100k rows) | **0.56** | 0.56 | 0.61 | 0.57 |
| Numba (1M rows) | **6.88** | 6.88 | 6.95 | 6.90 |
| GroupBy (10k, 50 grp, sum+mean) | **0.15** | 0.16 | 0.70 | 0.50 |
| GroupBy (10k, 500 grp) | **0.18** | 0.20 | 0.73 | 0.57 |
| GroupBy (100k, 100 grp) | 1.11 | 1.28 | 2.67 | **1.10** |
| GroupBy (100k, 1k grp) | 1.30 | 1.43 | 2.45 | **1.31** |
| GroupBy (1M, 100 grp) | 18.61 | 23.61 | 15.61 | **2.81** |
| GroupBy (1M, 10k grp) | 21.15 | 25.43 | 27.96 | **6.20** |
| GroupBy (100k, 2 col, ~300 grp) | 4.00 | 7.74 | 8.76 | **1.64** |
| GroupBy (1M, 2 col, ~300 grp) | 48.23 | 97.27 | 79.12 | **10.19** |
| Transform (10k, 50 grp) | **0.06** | 0.07 | 0.49 | 0.47 |
| Transform (100k, 100 grp) | **0.82** | 1.01 | 2.13 | 1.16 |
| Transform (1M, 1k grp) | **8.24** | 12.25 | 25.44 | 9.76 |
| Inner join (1k x 1k) | **0.12** | 0.41 | 0.86 | 0.45 |
| Inner join (5k x 5k) | 3.67 | 7.05 | 8.17 | **1.69** |
| Inner join (10k x 10k) | 15.24 | 24.96 | 27.88 | **3.90** |
| Inner join (50k x 50k) | 389 | 537 | 654 | **128** |
| Left join (1k x 1k) | **0.09** | 0.33 | 0.85 | 5.81 |
| Left join (5k x 5k) | 3.89 | 7.69 | 7.77 | **2.75** |
| Left join (50k x 50k) | 457 | 572 | 665 | **144** |

---

*Benchmarks collected with tafra 2.2.0, pandas 3.0.1, polars 1.39.0, numpy 2.2.5, numba 0.61.2 on Windows 11 (Python 3.11). C extension active.*
