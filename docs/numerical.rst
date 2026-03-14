=====================
Numerical Performance
=====================

Summary
=======

One of the goals of ``tafra`` is to provide a fast-as-possible data structure
for numerical computing. To achieve this, all function returns are written
as `generator expressions <https://www.python.org/dev/peps/pep-0289/>`_ wherever
possible.

.. note::

    Benchmarks on this page were collected with ``tafra`` 2.1.0 on Windows 11,
    tested against ``pandas`` 2.3.3 (numpy 2.2.6, Python 3.10),
    ``pandas`` 3.0.1 (numpy 2.2.5, Python 3.11), and ``polars`` 1.39.0.
    Where versions differ meaningfully, multiple results are shown.

Additionally, because the :attr:`data` contains values of ndarrays, the
``map`` functions may also take functions that operate on ndarrays. This means
that they are able to take `numba <http://numba.pydata.org/>`_ ``@jit``'ed
functions as well.


Construction and Access
=======================

``Tafra`` wraps a plain ``dict`` of ``numpy`` arrays, so construction and
column access have minimal overhead compared to ``pandas``:

.. code-block:: python

    # 100k rows, 5 columns
    >>> %timeit Tafra(data)
    15 µs

    >>> %timeit pd.DataFrame(data)               # pandas 2.3
    4.04 ms                                        # 264x slower

    >>> %timeit pd.DataFrame(data)               # pandas 3.0
    3.55 ms                                        # 309x slower

    >>> %timeit pl.DataFrame(data)               # polars 1.39
    0.04 ms                                        # 2.4x slower

    # Column access, 10k iterations
    >>> %timeit tf['x']
    0.13 µs per access

    >>> %timeit df['x']                          # pandas 2.3
    1.84 µs per access                             # 14x slower

    >>> %timeit df['x']                          # pandas 3.0
    12.1 µs per access                             # 133x slower

    >>> %timeit plf['x']                         # polars 1.39
    0.84 µs per access                             # 6.4x slower

``pandas`` 3.0 introduced copy-on-write semantics and additional safety checks
in column access, significantly increasing per-access overhead. ``polars`` is
faster than ``pandas`` but still 6x slower than ``Tafra``'s direct dict lookup.


Row Mapping
===========

``pandas`` is essentially a standard package for anyone performing data science
with Python, and it provides a wide variety of useful features. However, it's
not particularly aimed at maximizing performance. Let's use an example of a
dataframe of function arguments, and a function that maps scalar arguments into
a vector result. Any function of time serves this purpose, so let's use a
hyperbolic function.

First, let's randomly generate some function arguments and construct both a
``Tafra`` and a ``pandas.DataFrame``:

.. code-block:: python

    >>> from tafra import Tafra
    >>> import pandas as pd
    >>> import numpy as np

    >>> from typing import Tuple, Union, Any

    >>> tf = Tafra({
    ...     'wellid': np.arange(0, 100),
    ...     'qi': np.random.lognormal(np.log(2000.), np.log(3000. / 1000.) / (2 * 1.28), 100),
    ...     'Di': np.random.uniform(.5, .9, 100),
    ...     'bi': np.random.normal(1.0, .2, 100)
    ... })

    >>> df = pd.DataFrame(tf.data)


Next, we define our hyperbolic function and the time array to evaluate:

.. code-block:: python

    >>> import math

    >>> def tan_to_nominal(D: float) -> float:
    ...     return -math.log1p(-D)

    >>> def sec_to_nominal(D: float, b: float) -> float:
    ...     if b <= 1e-4:
    ...         return tan_to_nominal(D)
    ...
    ...     return ((1.0 - D) ** -b - 1.0) / b

    >>> def hyp(qi: float, Di: float, bi: float, t: np.ndarray) -> np.ndarray:
    ...     Dn = sec_to_nominal(Di, bi)
    ...
    ...     if bi <= 1e-4:
    ...         return qi * np.exp(-Dn * t)
    ...
    ...     return qi / (1.0 + Dn * bi * t) ** (1.0 / bi)

    >>> t = 10 ** np.linspace(0, 4, 101)


And let's build a generic ``mapper`` function to map over the named columns:

.. code-block:: python

    >>> def mapper(tf: Union[Tafra, pd.DataFrame]) -> Tuple[int, np.ndarray]:
    ...     return tf['wellid'], hyp(tf['qi'], tf['Di'], tf['bi'], t)


We can call this with the following style and time each approach:

.. code-block:: python

    >>> %timeit Tafra(tf.row_map(mapper))
    1.96 ms

    >>> %timeit pd.DataFrame(dict(df.apply(mapper, axis=1).to_list()))
    2.47 ms                                        # pandas 2.3: 1.3x slower
    2.10 ms                                        # pandas 3.0: 1.2x slower


We see ``Tafra`` is faster. Mapping a function this way is
convenient, but there is some indirection occurring that we can do away with to
obtain direct access to the data of the ``Tafra``, and there is a faster
method for ``pandas`` as well as opposed to :meth:`pandas.DataFrame.apply`.
Instead of constructing a new ``Tafra`` or ``pd.DataFrame`` for each row, we
can instead return a :class:`NamedTuple`, which is faster to construct. Doing so:

.. code-block:: python

    >>> def tuple_mapper(tf: Tuple[Any, ...]) -> Tuple[int, np.ndarray]:
    ...     return tf.wellid, hyp(tf.qi, tf.Di, tf.bi, t)

    >>> %timeit Tafra(tf.tuple_map(tuple_mapper))
    820 µs

    >>> %timeit pd.DataFrame(dict((tuple_mapper(row)) for row in df.itertuples()))
    1.51 ms                                        # pandas 2.3: 1.8x slower
    1.19 ms                                        # pandas 3.0: 1.5x slower


And once again, ``Tafra`` is faster.


GroupBy and Transform
=====================

For aggregation operations, ``pandas`` uses optimized C/Cython internals that
are difficult to match in pure Python + numpy. ``Tafra`` 2.1.0 uses
index-based grouping (``np.unique`` + ``return_inverse``) rather than
per-group boolean masks, which is considerably faster than earlier versions
but still slower than ``pandas`` for large datasets:

.. code-block:: python

    # GroupBy: 10k rows, 50 groups, 2 aggregations (C ext + vectorized fast path)
    >>> %timeit tf.group_by(['group'], {'mean': (np.mean, 'value'), 'sum': (np.sum, 'value')})
    0.15 ms                                        # Tafra+C

    >>> %timeit df.groupby('group')['value'].agg(['mean', 'sum']).reset_index()
    0.73 ms                                        # pandas 2.3: 5x slower

    >>> %timeit plf.group_by('group').agg(...)
    0.60 ms                                        # polars: 4x slower

    # GroupBy: 100k rows, 1000 groups
    >>> %timeit tf.group_by(...)
    1.75 ms                                        # Tafra+C

    >>> %timeit df.groupby(...)
    3.17 ms                                        # pandas 2.3: 1.8x slower

    >>> %timeit plf.group_by(...)
    1.94 ms                                        # polars: ~equal

    # Transform: 10k rows, 50 groups
    >>> %timeit tf.transform(['group'], {'m': (np.mean, 'value')})
    0.06 ms                                        # Tafra+C

    >>> %timeit df.assign(m=df.groupby('group')['value'].transform('mean'))
    0.60 ms                                        # pandas 2.3: 10x slower

    >>> %timeit plf.with_columns(pl.col('value').mean().over('group'))
    1.67 ms                                        # polars: 28x slower

    # Transform: 100k rows, 100 groups
    >>> %timeit tf.transform(...)
    0.80 ms                                        # Tafra+C

    >>> %timeit df ... .transform('mean')
    2.97 ms                                        # pandas 2.3: 3.7x slower

    >>> %timeit plf.with_columns(...)
    3.28 ms                                        # polars: 4.1x slower

With the C extension, ``Tafra`` is **4–28x faster** than both ``pandas`` and
``polars`` for GroupBy and Transform at 10k rows. At 100k rows ``Tafra`` beats
``pandas`` on everything and matches or beats ``polars`` except on multi-column
groupby with many groups. Without the C extension, ``Tafra`` is still faster
than both at 10k and competitive at 100k. The direct array mapping (O(n), no
sort) and single-pass C aggregation eliminate both the ``np.unique`` sort
bottleneck and multi-pass numpy overhead.

String columns are automatically encoded to integer codes for efficient
grouping — no performance penalty vs numeric-only groups.

``Tafra``'s vectorized fast path uses ``np.bincount`` and ``ufunc.reduceat``
for recognized aggregations: ``np.sum``, ``np.mean``, ``np.std``, ``np.var``,
``np.min``, ``np.max``, ``np.ptp``, ``np.prod``, ``np.median``, ``np.any``,
``np.all``, ``np.count_nonzero``, ``len``, ``sum``, plus all nan-variants.
Custom aggregations ``percentile(q)``, ``geomean``, and ``harmean`` also
hit the fast path.

For unrecognized functions, ``Tafra`` falls back to calling your Python
function directly on numpy arrays for each group — fully transparent, no
hidden dispatch or "silent" dtype changes.


Joins
=====

``Tafra`` 2.1.0 uses a numpy-native sort-merge join for equality joins:
``argsort`` + ``searchsorted`` to find match ranges, then ``np.repeat`` with
offset arithmetic to build index arrays — no Python-level per-row iteration.
For non-equality operators (``<``, ``<=``, ``>``, ``>=``, ``!=``), it falls
back to a nested-loop approach.

.. code-block:: python

    # Inner join: 1k x 1k rows, equality on one key
    >>> %timeit left_tf.inner_join(...)           # Tafra+C
    0.08 ms

    >>> %timeit pd.merge(..., how='inner')        # pandas 2.3
    0.93 ms                                        # 12x slower

    >>> %timeit left_pl.join(..., how='inner')    # polars 1.39
    1.53 ms                                        # 19x slower

    # Inner join: 10k x 10k rows
    >>> %timeit left_tf.inner_join(...)           # Tafra+C
    13.8 ms

    >>> %timeit pd.merge(...)                     # pandas 2.3
    34.2 ms                                        # 2.5x slower

    >>> %timeit left_pl.join(...)                 # polars 1.39
    5.50 ms                                        # 2.5x faster

    # Left join: 1k x 1k rows
    >>> %timeit left_tf.left_join(...)            # Tafra+C
    0.08 ms

    >>> %timeit pd.merge(..., how='left')         # pandas 2.3
    0.93 ms                                        # 12x slower

    >>> %timeit left_pl.join(..., how='left')     # polars 1.39
    1.63 ms                                        # 20x slower

With the C hash join, ``Tafra`` is **12–20x faster** than both ``pandas`` and
``polars`` on small-scale joins (1k x 1k). At 10k x 10k, ``polars``' Rust
multithreaded join pulls ahead while ``Tafra`` is still 2.5x faster than
``pandas``. ``Tafra``'s join also supports
arbitrary comparison operators (``<``, ``<=``, ``>``, ``>=``, ``!=``) in the
``on`` clause, which neither ``pandas`` nor ``polars`` natively offer.


Partition and Multiprocessing
=============================

``group_by`` and ``partition`` both split data by group values, but serve
different purposes:

- ``group_by`` **reduces** — applies aggregation functions and returns one row
  per group. Fast for built-in reducers (vectorized, no Python loop).
- ``partition`` **splits** — returns all original rows grouped into sub-Tafras.
  Designed for dispatching expensive per-group computation to worker processes.

For **light aggregations** (sum, mean, std), ``group_by`` is the right tool —
it's 10-100x faster because it avoids serialization overhead entirely:

.. code-block:: python

    # Light work: group_by wins decisively
    >>> tf.group_by(['group'], {'mean': (np.mean, 'value'), 'std': (np.std, 'value')})
    2 ms                                           # vectorized, no IPC

    >>> partition + serial map (same aggregations)
    10 ms                                          # partition + per-group Python calls

For **expensive per-group computation** (model fitting, forecasting, simulation),
``partition`` + ``ProcessPoolExecutor`` scales nearly linearly with workers:

.. code-block:: python

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

Benchmarks with ~13 ms of work per group, 8 workers:

================================  ============  ==============  =========
Scenario                          Serial        8 Workers       Speedup
================================  ============  ==============  =========
50 groups, 10k rows               681 ms        **138 ms**      **4.9x**
100 groups, 100k rows             1,443 ms      **318 ms**      **4.5x**
1,000 groups, 100k rows           13,535 ms     **2,784 ms**    **4.9x**
================================  ============  ==============  =========

The crossover point depends on per-group work cost. Rule of thumb:

- **< 1 ms per group**: use ``group_by`` (IPC overhead dominates)
- **> 5 ms per group**: use ``partition`` + workers (parallelism wins)

``Tafra`` supports Python's standard multiprocessing serialization natively
(dataclass + numpy arrays), so no special handling is needed.


Numba Integration
=================

Because ``Tafra``'s :attr:`data` contains raw ``numpy`` arrays, ``numba``
``@jit``'ed functions work directly with no adapter layer:

.. code-block:: python

    >>> from numba import jit
    >>> jit_kw = {'fastmath': True}

    >>> @jit(**jit_kw)
    ... def tan_to_nominal(D: float) -> float:
    ...     return -math.log1p(-D)

    >>> @jit(**jit_kw)
    ... def sec_to_nominal(D: float, b: float) -> float:
    ...     if b <= 1e-4:
    ...         return tan_to_nominal(D)
    ...
    ...     return ((1.0 - D) ** -b - 1.0) / b

    >>> @jit(**jit_kw)
    ... def hyp(qi: float, Di: float, bi: float, t: np.ndarray) -> np.ndarray:
    ...     Dn = sec_to_nominal(Di, bi)
    ...
    ...     if bi <= 1e-4:
    ...         return qi * np.exp(-Dn * t)
    ...
    ...     return qi / (1.0 + Dn * bi * t) ** (1.0 / bi)

    >>> @jit(**jit_kw)
    ... def ndarray_map(qi, Di, bi, t):
    ...     out = np.zeros((qi.shape[0], t.shape[0]))
    ...     for i in range(qi.shape[0]):
    ...         out[i, :] = hyp(qi[i], Di[i], bi[i], t)
    ...     return out

    >>> %timeit ndarray_map(tf['qi'], tf['Di'], tf['bi'], t)
    # ~80 µs — essentially zero overhead from Tafra


When to use Tafra
=================

``Tafra`` is fastest when your workload is dominated by:

* **Construction and teardown** — 140–320x faster than pandas, competitive
  with polars
* **Column access** — 14–115x faster than pandas, 5–8x faster than polars
* **Row-wise mapping** — 1.8x faster than pandas (polars has no row-wise UDF)
* **GroupBy and Transform at ≤10k rows** — with C extension, 4–28x faster
  than both pandas and polars
* **GroupBy and Transform at 100k rows** — faster than pandas on all
  benchmarks; matches polars on single-column, polars leads on multi-column
* **Small-scale joins** — with C extension, equi-joins at 1k x 1k are
  12–20x faster than both pandas and polars
* **Numba-accelerated computation** — direct ``ndarray`` access with zero
  adapter overhead

``polars`` is fastest for:

* **Large-scale multi-column GroupBy** — Rust multithreaded internals at
  100k+ rows with many groups (2–4x faster than Tafra)
* **Large-scale joins** — Rust multithreaded hash-join at 5k+ rows
  (2–3x faster than Tafra)

``pandas`` is the slowest of the three on nearly every benchmark. Version 3.0
is significantly slower than 2.3 on column access and joins due to copy-on-write
overhead. Its broadest feature set remains its primary advantage.

The general pattern: ``Tafra`` wins on everything up to ~10k rows and competes
at 100k. ``polars`` pulls ahead on large-scale operations where its Rust
multithreaded internals dominate. The optional C extension closes much of the
remaining gap — without it, ``Tafra`` still beats pandas everywhere and is
competitive with polars at moderate scales.


Summary
=======

All times in milliseconds. Lower is better. **Bold** = fastest.

Tafra+C = with optional C extension. Tafra = pure Python + numpy only.

===================================  =========  =========  ===========  ===========  =============
Benchmark                            Tafra+C    Tafra      pandas 2.3   pandas 3.0   polars 1.39
===================================  =========  =========  ===========  ===========  =============
Construction (100k rows)             **0.02**   0.02       2.80         6.46         0.04
Column access (per call, µs)         **0.13**   0.13       1.81         15.7         0.70
Row map (100 rows, tuple_map)        **0.80**   0.80       1.43         1.43         n/a
GroupBy (10k, 50 grp, sum+mean)      **0.15**   0.18       0.73         0.90         0.60
GroupBy (10k, 500 grp)               **0.19**   0.22       0.82         0.84         0.75
GroupBy (100k, 100 grp)              **1.53**   1.77       2.71         4.56         1.74
GroupBy (100k, 1k grp)               **1.75**   1.94       3.17         4.46         1.94
GroupBy (100k, 2 col, ~300 grp)      9.12       8.99       7.19         15.6         **3.23**
Transform (10k, 50 grp)             **0.06**    0.08       0.60         1.50         1.67
Transform (100k, 100 grp)           **0.80**    1.11       2.97         4.55         3.28
Inner join (1k × 1k)                **0.08**    0.30       0.93         2.53         1.53
Inner join (5k × 5k)                 3.43       6.76       9.40         19.0         **4.14**
Inner join (10k × 10k)               13.8       24.0       34.2         38.7         **5.50**
Left join (1k × 1k)                 **0.08**    0.33       0.93         0.89         1.63
Left join (5k × 5k)                  3.47       6.91       9.78         9.54         **1.60**
===================================  =========  =========  ===========  ===========  =============
