=============================
Tafra: a minimalist dataframe
=============================

.. image:: https://img.shields.io/pypi/v/tafra.svg
    :target: https://pypi.org/project/tafra/

.. image:: https://travis-ci.org/petbox-dev/tafra.svg?branch=master
    :target: https://travis-ci.org/petbox-dev/tafra

.. image:: https://readthedocs.org/projects/tafra/badge/?version=latest
    :target: https://tafra.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/petbox-dev/tafra/badge.svg
    :target: https://coveralls.io/github/petbox-dev/tafra
    :alt: Coverage Status


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

+------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| Tafra                  | `Tafra <https://tafra.readthedocs.io/en/latest/api.html#tafra.base.Tafra>`_,                                                    |
+------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| Aggregations           | `transient_rate(t) <https://tafra.readthedocs.io/en/latest/api.html#petbox.dca.THM.transient_rate>`_,                           |
|                        | `transient_cum(t) <https://tafra.readthedocs.io/en/latest/api.html#petbox.dca.THM.transient_cum>`_,                             |
|                        | `transient_D(t) <https://tafra.readthedocs.io/en/latest/api.html#petbox.dca.THM.transient_D>`_,                                 |
|                        | `transient_beta(t) <https://tafra.readthedocs.io/en/latest/api.html#petbox.dca.THM.transient_beta>`_,                           |
|                        | `transient_b(t) <https://tafra.readthedocs.io/en/latest/api.html#petbox.dca.THM.transient_b>`_                                  |
+------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| Primary Phase          | `add_secondary <https://tafra.readthedocs.io/en/latest/api.html#petbox.dca.PrimaryPhase.add_secondary>`_,                       |
|                        | `add_water <https://tafra.readthedocs.io/en/latest/api.html#petbox.dca.PrimaryPhase.add_water>`_                                |
+------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| Secondary Phase        | `gor(t) <https://tafra.readthedocs.io/en/latest/api.html#petbox.dca.SecondaryPhase.gor>`_,                                      |
|                        | `cgr(t) <https://tafra.readthedocs.io/en/latest/api.html#petbox.dca.SecondaryPhase.cgr>`_                                       |
+------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| Water Phase            | `wor(t) <https://tafra.readthedocs.io/en/latest/api.html#petbox.dca.WaterPhase.wor>`_                                           |
+------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| Utility                | `bourdet <https://tafra.readthedocs.io/en/latest/api.html#petbox.dca.bourdet>`_,                                                |
|                        | `get_time <https://tafra.readthedocs.io/en/latest/api.html#petbox.dca.get_time>`_,                                              |
|                        | `get_time_monthly_vol <https://tafra.readthedocs.io/en/latest/api.html#petbox.dca.get_time_monthly_vol>`_                       |
+------------------------+---------------------------------------------------------------------------------------------------------------------------------+

Getting Started
===============

Install the library with `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block:: shell

    pip install tafra


A short example
---------------

.. code-block:: python

    >>> from tafra import Tafra

    >>> t = Tafra({
    ...    'x': np.array([1, 2, 3, 4]),
    ...    'y': np.array(['one', 'two', 'one', 'two'], dtype='object'),
    ... })

    >>> t.pformat()
    Tafra(data = {
     'x': array([1, 2, 3, 4]),
     'y': array(['one', 'two', 'one', 'two'])},
    dtypes = {
     'x': 'int', 'y': 'object'},
    rows = 4)

    >>> print('List:', '\n', t.to_list())
    List:
     [array([1, 2, 3, 4]), array(['one', 'two', 'one', 'two'], dtype=object)]

    >>> print('Records:', '\n', tuple(t.to_records()))
    Record:
     ((1, 'one'), (2, 'two'), (3, 'one'), (4, 'two'))

    >>> gb = t.group_by(
    ...     ['y'], {'x': sum}
    ... )

    >>> print('Group By:', '\n', gb.pformat())
    Group By:
    Tafra(data = {
     'x': array([4, 6]), 'y': array(['one', 'two'])},
    dtypes = {
     'x': 'int', 'y': 'object'},
    rows = 2)


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

In this case, lightweight also means performant. Beyond any additional
features added to the library, ``tafra`` should provide the necessary
base for organizing data structures for numerical processing. One of the
most important aspects is fast access to the data itself. By minizing
abstraction to access the underlying ``numpy`` arrays, ``tafra`` provides
over an order of magnitude increase in performance.

Construct a ``Tafra`` and a ``DataFrame``:

.. code-block:: python

    >>> t = Tafra({
    ...    'x': np.array([1, 2, 3, 4]),
    ...    'y': np.array(['one', 'two', 'one', 'two'], dtype='object'),
    ... })

    >>> df = pd.DataFrame(t.data)

Read Operations
---------------

Direct access:

.. code-block:: python

    >>> %timemit x = t._data['x']
    55.3 ns ± 5.64 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)


Indirect with some penalty to support ``Tafra`` slicing and ``numpy``'s
advanced indexing:

.. code-block:: python

    >>> %timemit x = t['x']
    219 ns ± 71.6 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)


``pandas`` timing:

.. code-block:: python

    >>> %timemit x = df['x']
    1.55 µs ± 105 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)


As fast as ``pandas`` gets:

.. code-block:: python

    >>> where_col = list(df.columns).index('x')
    >>> %timeit x = df.values[:, where_col]
    48 µs ± 7.77 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


Assignment Operations
---------------------

Direct access:

.. code-block:: python

    >>> x = np.arange(4)

    >>> %timeit tf._data['x'] = x
    65 ns ± 5.55 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)


Indidrect:

.. code-block:: python

    >>> %timeit tf['x'] = x
    7.39 µs ± 950 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)


``pandas`` timing:

.. code-block:: python

    >>> %timeit df['x'] = x
    47.8 µs ± 3.53 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
