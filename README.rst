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


A short example:

.. code-block:: python

    >>> t = Tafra({
    ..:    'x': np.array([1, 2, 3, 4]),
    ..:    'y': np.array(['one', 'two', 'one', 'two'], dtype='object'),
    ..: })

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
    ..:     ['y'], {'x': sum}
    ..: )

    >>> print('Group By:', '\n', gb.pformat())
    Group By:
    Tafra(data = {
     'x': array([4, 6]), 'y': array(['one', 'two'])},
    dtypes = {
     'x': 'int', 'y': 'object'},
    rows = 2)


In this case, lightweight also means performant. Beyond any additional
features added to the library, ``tafra`` should provide the necessary
base for organizing data structures for numerical processing. One of the
most important aspects is fast access to the data itself. By minizing
abstraction to access the underlying ``numpy`` arrays, ``tafra`` provides
over an order of magnitude increase in performance.

.. code-block:: python

    >>> t = Tafra({
    ..:    'x': np.array([1, 2, 3, 4]),
    ..:    'y': np.array(['one', 'two', 'one', 'two'], dtype='object'),
    ..: })

    >>> df = pd.DataFrame(t.data)

    # read operations
    # direct access
    >>> %timemit x = t._data['x']
    55.3 ns ± 5.64 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)

    # indirect with some penalty to support ``numpy``'s advanced indexing
    >>> %timemit x = t['x']
    219 ns ± 71.6 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

    >>> %timemit x = df['x']
    1.55 µs ± 105 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

    # as fast as ``pandas`` gets
    >>> where_col = list(df.columns).index('x')
    >>> %timeit x = df.values[:, where_col]
    48 µs ± 7.77 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


    # assignment operations
    >>> x = np.arange(4)

    >>> %timeit tf._data['x'] = x
    65 ns ± 5.55 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)

    >>> %timeit tf['x'] = x
    7.39 µs ± 950 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    >>> %timeit df['x'] = x
    47.8 µs ± 3.53 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
