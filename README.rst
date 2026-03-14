=============================
Tafra: a minimalist dataframe
=============================

.. image:: https://img.shields.io/pypi/v/tafra.svg
    :target: https://pypi.org/project/tafra/

.. image:: https://img.shields.io/pypi/pyversions/tafra.svg
    :target: https://pypi.org/project/tafra/

.. image:: https://readthedocs.org/projects/tafra/badge/?version=latest
    :target: https://tafra.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


The ``tafra`` began life as a thought experiment: how could we reduce the idea
of a da\ *tafra*\ me (as expressed in libraries like ``pandas`` or languages
like R) to its useful essence, while carving away the cruft?
The `original proof of concept <https://usethe.computer/posts/12-typing-groupby.html>`_
stopped at "group by".

.. `original proof of concept`_

This library expands on the proof of concept to produce a practically
useful ``tafra``, which we hope you may find to be a helpful lightweight
substitute for certain uses of ``pandas``.

A ``tafra`` is, more-or-less, a set of named *columns* or *dimensions*.
Each of these is a typed ``numpy`` array of consistent length, representing
the values for each column by *rows*.

The library provides lightweight syntax for manipulating rows and columns,
support for managing data types, iterators for rows and sub-frames,
`pandas`-like "transform" support and conversion from `pandas` Dataframes,
and SQL-style "group by" and join operations.

+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Tafra                      | `Tafra <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra>`_                                                 |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Aggregations               | `Union <https://tafra.readthedocs.io/en/latest/api.html#tafra.group.Union>`_,                                               |
|                            | `GroupBy <https://tafra.readthedocs.io/en/latest/api.html#tafra.group.GroupBy>`_,                                           |
|                            | `Transform <https://tafra.readthedocs.io/en/latest/api.html#tafra.group.Transform>`_,                                       |
|                            | `IterateBy <https://tafra.readthedocs.io/en/latest/api.html#tafra.group.IterateBy>`_,                                       |
|                            | `InnerJoin <https://tafra.readthedocs.io/en/latest/api.html#tafra.group.InnerJoin>`_,                                       |
|                            | `LeftJoin <https://tafra.readthedocs.io/en/latest/api.html#tafra.group.LeftJoin>`_,                                         |
|                            | `CrossJoin <https://tafra.readthedocs.io/en/latest/api.html#tafra.group.CrossJoin>`_                                        |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Aggregation Helpers        | `union <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.union>`__,                                         |
|                            | `union_inplace <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.union_inplace>`_,                          |
|                            | `group_by <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.group_by>`_,                                    |
|                            | `transform <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.transform>`__,                                 |
|                            | `iterate_by <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.iterate_by>`_,                                |
|                            | `inner_join <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.inner_join>`_,                                |
|                            | `left_join <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.left_join>`_,                                  |
|                            | `cross_join <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.cross_join>`_                                 |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Chunking / Partitioning    | ``chunks``, ``chunk_rows``, ``partition``, ``concat``                                                                       |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Custom Aggregations        | ``percentile``, ``geomean``, ``harmean``                                                                                    |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Constructors               | `as_tafra <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.as_tafra>`_,                                    |
|                            | `from_dataframe <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.from_dataframe>`_,                        |
|                            | `from_series <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.from_series>`_,                              |
|                            | `from_records <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.from_records>`_                             |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| SQL Readers                | `read_sql <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.read_sql>`_,                                    |
|                            | `read_sql_chunks <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.read_sql_chunks>`_                       |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Destructors                | `to_records <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.to_records>`_,                                |
|                            | `to_list <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.to_list>`_,                                      |
|                            | `to_tuple <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.to_tuple>`_,                                    |
|                            | `to_array <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.to_array>`_,                                    |
|                            | `to_pandas <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.to_pandas>`_                                   |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Properties                 | `rows <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.rows>`_,                                            |
|                            | `columns <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.columns>`_,                                      |
|                            | `data <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.data>`_,                                            |
|                            | `dtypes <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.dtypes>`_,                                        |
|                            | `size <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.size>`_,                                            |
|                            | `ndim <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.ndim>`_,                                            |
|                            | `shape <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.shape>`_                                           |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Iter Methods               | `iterrows <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.iterrows>`_,                                    |
|                            | `itertuples <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.itertuples>`_,                                |
|                            | `itercols <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.itercols>`_                                     |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Functional Methods         | `row_map <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.row_map>`_,                                      |
|                            | `tuple_map <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.tuple_map>`_,                                  |
|                            | `col_map <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.col_map>`_,                                      |
|                            | `pipe <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.pipe>`_                                             |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Dict-like Methods          | `keys <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.keys>`_,                                            |
|                            | `values <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.values>`_,                                        |
|                            | `items <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.items>`_,                                          |
|                            | `get <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.get>`_,                                              |
|                            | `update <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.update>`_,                                        |
|                            | `update_inplace <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.update_inplace>`_,                        |
|                            | `update_dtypes <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.update_dtypes>`_,                          |
|                            | `update_dtypes_inplace <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.update_dtypes_inplace>`_           |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Data Exploration           | ``head``, ``tail``, ``sort``, ``sample``, ``describe``, ``value_counts``, ``drop_duplicates``                               |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Time Series                | ``shift``                                                                                                                   |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Other Helper Methods       | `select <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.select>`_,                                        |
|                            | `copy <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.copy>`_,                                            |
|                            | `rename <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.rename>`_,                                        |
|                            | `rename_inplace <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.rename_inplace>`_,                        |
|                            | `coalesce <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.coalesce>`_,                                    |
|                            | `coalesce_inplace <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.coalesce_inplace>`_,                    |
|                            | `_coalesce_dtypes <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra._coalesce_dtypes>`_,                    |
|                            | `delete <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.delete>`_,                                        |
|                            | `delete_inplace <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.delete_inplace>`_                         |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Printer Methods            | `pprint <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.pprint>`_,                                        |
|                            | `pformat <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.pformat>`_,                                      |
|                            | `to_html <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra.to_html>`_                                       |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Indexing Methods           | `_slice <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra._slice>`_,                                        |
|                            | `_index <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra._index>`_,                                        |
|                            | `_ndindex <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra._ndindex>`_                                     |
+----------------------------+-----------------------------------------------------------------------------------------------------------------------------+

Getting Started
===============

Install from conda-forge (includes pre-built C extension — no compiler needed):

.. code-block:: shell

    conda install tafra -c conda-forge

Or install from PyPI with pip:

.. code-block:: shell

    pip install tafra

.. note::

    ``conda install`` provides a pre-built binary with the C extension already
    compiled for your platform. ``pip install`` from PyPI will attempt to
    compile the C extension from source; if no C compiler is available, the
    package installs without it and falls back to pure Python + numpy.


Building from source
--------------------

To build from source (including the optional C extension):

.. code-block:: shell

    git clone https://github.com/petbox-dev/tafra.git
    cd tafra
    pip install -e .

**Requirements:**

- Python >=3.9
- numpy >=2.1
- A C compiler (optional, for the ``_accel`` extension):
  - **Windows**: Visual Studio Build Tools (with Windows SDK) or MinGW-w64
  - **Linux**: ``gcc`` (usually pre-installed, or ``apt install build-essential``)
  - **macOS**: Xcode Command Line Tools (``xcode-select --install``)

If no C compiler is available, the package installs without the extension and
falls back to pure Python + numpy at runtime. To verify the C extension is
active:

.. code-block:: python

    >>> from tafra._accel import groupby_sum
    >>> print("C extension active")

To build a distributable wheel:

.. code-block:: shell

    pip install build
    python -m build

Windows build notes
^^^^^^^^^^^^^^^^^^^

The C extension requires the MSVC compiler to find the Windows SDK headers.
If you get ``fatal error C1083: Cannot open include file: 'io.h'``, the
Windows SDK include/lib paths are not set. Two options:

1. **Use a Developer Command Prompt** (recommended): Open "Developer Command
   Prompt for VS" or "Developer PowerShell for VS" from the Start menu. This
   runs ``vcvarsall.bat`` automatically and sets all required paths.

2. **Use MinGW-w64** instead of MSVC:

   .. code-block:: shell

       python setup.py build_ext --inplace --compiler=mingw32

   MinGW-w64 can be installed via conda (``conda install m2w64-gcc -c
   conda-forge``) or from `winlibs.com <https://winlibs.com/>`_.

If building with ``python -m build`` (which creates an isolated environment),
use ``--no-isolation`` to inherit your shell's environment variables, or run
from a Developer Command Prompt:

.. code-block:: shell

    python -m build --no-isolation


A short example
---------------

.. code-block:: python

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


group_by vs partition
---------------------

``group_by`` **reduces** — one row per group, applies aggregation functions:

.. code-block:: python

    >>> tf.group_by(['wellid'], {'total_oil': (np.sum, 'oil')})
    # Returns: one row per wellid, with summed oil

``partition`` **splits** — returns all original rows, grouped into sub-Tafras
for independent processing (e.g., multiprocessing):

.. code-block:: python

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

With 8 workers and ~13 ms of work per group, ``partition`` achieves ~5x
speedup over serial execution. For light aggregations (sum, mean, std),
``group_by`` is 10-100x faster — use it instead. See
`numerical.rst <https://tafra.readthedocs.io/en/latest/numerical.html>`_ for
detailed benchmarks.

``chunks`` splits by row count (for data-parallel workloads where group
integrity doesn't matter):

.. code-block:: python

    >>> for chunk in tf.chunks(n=4, sort_by=['date']):
    ...     process(chunk)


Flexibility
-----------

Have some code that works with ``pandas``, or just a way of doing things
that you prefer? ``tafra`` is flexible:

.. code-block:: python

    >>> df = pd.DataFrame(np.c_[
    ...     np.array([1, 2, 3, 4]),
    ...     np.array(['one', 'two', 'one', 'two'])
    ... ], columns=['x', 'y'])

    >>> t = Tafra.from_dataframe(df)


And going back is just as simple:

.. code-block:: python

    >>> df = pd.DataFrame(t.data)


Timings
=======

.. note::

    Benchmarks collected with ``tafra`` 2.1.0. See
    `numerical.rst <https://tafra.readthedocs.io/en/latest/numerical.html>`_
    for full benchmarks against ``pandas`` 2.3/3.0 and ``polars`` 1.39.

Lightweight means performant. By minimizing abstraction to access the
underlying ``numpy`` arrays, ``tafra`` provides dramatic speedups over
``pandas`` and ``polars`` on construction and access:

.. code-block:: python

    # Construction: 100k rows, 5 columns
    Tafra():         0.02 ms
    pd.DataFrame():  2.80 ms   # 140x slower
    pl.DataFrame():  0.04 ms   # 2x slower

    # Column access: 100k rows, per access
    tf['x']:         0.13 µs
    df['x']:         1.81 µs   # 14x slower (pandas 2.3)
    plf['x']:        0.70 µs   # 5x slower

``tafra`` uses vectorized numpy operations (``np.bincount``,
``ufunc.reduceat``) and an optional C extension (single-pass aggregation,
hash joins) for GroupBy and joins. With the C extension:

.. code-block:: python

    # GroupBy: 10k rows, 50 groups, sum + mean
    Tafra+C: 0.15 ms
    pandas:  0.73 ms   # 5x slower
    polars:  0.60 ms   # 4x slower

    # Transform: 10k rows, 50 groups
    Tafra+C: 0.06 ms
    pandas:  0.60 ms   # 10x slower
    polars:  1.67 ms   # 28x slower

    # Equi inner join: 1k x 1k
    Tafra+C: 0.08 ms
    pandas:  0.93 ms   # 12x slower
    polars:  1.53 ms   # 19x slower

-   **Import note** If you assign directly to the ``Tafra.data`` or
    ``Tafra._data`` attributes, you *must* call ``Tafra._coalesce_dtypes``
    afterwards in order to ensure the typing is consistent.
