=====================
Numerical Performance
=====================

Summary
=======

One of the goals of ``tafra`` is to provide a fast-as-possible data structure
for numerical computing. To achieve this, all function returns are written
as `generator expressions <https://www.python.org/dev/peps/pep-0289/>`_ wherever
possible.

Additionally, because the :attr:``data`` contains values of ndarrays, the
``map`` functions may also take functions that operator on ndarrays. This means
that they are able to take `numba <http://numba.pydata.org/>`_ ``@jit``'ed
functions as well.

``pandas`` is essentially a standard package for anyone performing data science
with Python, and it provides a wide variety of useful features. However, it's
not particularly aimed at maximizing performance. Let's use an example of a
dataframe of function arguments, and a function that maps scalar arguments into
a vector result. Any function of time serves this purpose, so let's use a
hyperbolic function.

First, let's randomnly generate some function arguments and construct both a
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

    >>> tf.head(5)

====== ====== ======= ======= =======
index  wellid qi      Di      bi
====== ====== ======= ======= =======
dtype  int32  float64 float64 float64
0      0      2665.82 0.54095 1.07538
1      1      1245.85 0.81711 0.48448
2      2      1306.56 0.61570 0.54587
3      3      2950.33 0.81956 0.66440
4      4      1963.93 0.56918 0.74165
====== ====== ======= ======= =======


Next, we define our hyperbolic function and the time array to evalute:

.. code-block:: python

    >>> import math

    >>> def tan_to_nominal(D: float) -> float:
    ...     return -math.log1p(-D)

    >>> def sec_to_nominal(D: float, b: float) -> float:
    ...     if b <= 1e-4:
    ...         return tan_to_nominal(Di)
    ...
    ...     return ((1.0 - D) ** -b - 1.0) / b

    >>> def hyp(qi: float, Di: float, bi: float, t: np.ndarray) -> np.ndarray:
    ...     Dn = sec_to_nominal(Di, bi)
    ...
    ...     if bi <= 1e-4:
    ...         return qi * np.exp(-Dn * t)
    ...
    ...    return qi / (1.0 + Dn * bi * t) ** (1.0 / bi)

    >>> t = 10 ** np.linspace(0, 4, 101)


And let's build a generic ``mapper`` function to map over the named columns:

.. code-block:: python

    >>> def mapper(tf: Union[Tafra, pd.DataFrame]) -> Tuple[int, np.ndarray]:
    ...     return tf['wellid'], hyp(tf['qi'], tf['Di'], tf['bi'], t)


We can call this with the following style. The ``pandas`` syntax is a bit
verbose, but :meth:`pandas.DataFrame.from_items()` is deprecated in newer
versions, so this is the recommended way. Let's time each approach:

.. code-block:: python

    >>> %timeit tdcs = Tafra(tf.row_map(mapper))
    3.38 ms ± 129 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


    >>> pdcs = pd.DataFrame(dict(df.apply(mapper, axis=1).to_list())))
    6.86 ms ± 408 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


We see ``Tafra`` is about twice as fast. Mapping a function this way is
convenient, but there is some indirection occuring that we can do away with to
obtain direct access to the data of the ``Tafra``, and there is a faster
method for ``pandas`` as well as opposed to :meth:`pandas.DataFrame.apply`.
Instead of constructing a new ``Tafra`` or ``pd.DataFrame`` for each row, we
can instead return a :class`NamedTuple`, which is faster to construct. Doing so:

.. code-block:: python

    >>> def tuple_mapper(tf: Tuple[Any, ...]) -> Tuple[int, np.ndarray]:
    ...     return tf.wellid, hyp(tf.qi, tf.Di, tf.bi, t)

    >>> %timeit Tafra(tf.tuple_map(tuple_mapper))
    1.68 ms ± 84.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    >>> %timeit pd.DataFrame(dict((tuple_mapper(row)) for row in df.itertuples()))
    3.14 ms ± 121 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


And once again, ``Tafra`` is about twice as fast.

One of the upcoming features of ``pandas`` is the ability to apply ``numba``
``@jit``'ed functions to :meth:`pandas.DataFrame.apply`. The performance
improvement should be significant, especially for long-running functions,
but there will still be overhead in the abstraction before the function is
called. We can demonstrate this by ``@jit``'ing our hyperbolic function and
mapping it over the dataframes, and get an idea of how much improvement is
possible:

.. code-block:: python

    >>> from numba import jit

    >>> @jit
    ...  def tan_to_nominal(D: float) -> float:
    ...     return -np.log1p(-D)

    >>> @jit
    ... def sec_to_nominal(D: float, b: float) -> float:
    ...     if b <= 1e-4:
    ...         return tan_to_nominal(D)
    ...
    ...     return ((1.0 - D) ** -b - 1.0) / b

    >>> @jit
    ... def hyp(qi: float, Di: float, bi: float, t: np.ndarray) -> np.ndarray:
    ...     Dn = sec_to_nominal(Di, bi)
    ...
    ...     if bi <= 1e-4:
    ...         return qi * np.exp(-Dn * t)
    ...
    ...     return qi / (1.0 + Dn * bi * t) ** (1.0 / bi)

    >>> %timeit Tafra(tf.tuple_map(tuple_mapper))
    884 µs ± 41.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    >>> %timeit pd.DataFrame(dict((tuple_mapper(row)) for row in df.itertuples()))
    3.09 ms ± 115 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


Interestingly, we see that ``pandas`` does not get much benefit from this, as
the limit occurs not in the performance of the functions but in the performance
of ``pandas`` itself. We can validate this by skipping the dataframe
construction step:

.. code-block:: python

    >>> %timeit [tf.tuple_map(tuple_mapper)]
    81.9 µs ± 2.91 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    >>> %timeit [(tuple_mapper(row)) for row in df.itertuples()]
    614 µs ± 14.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


Last, we might as the question "If ``pandas`` is incurring some performance
penalty, what is the performance penalty of ``Tafra``?" We'll write a function
that operates on the :class:`numpy.ndarray`s themselves rather than using the
helper :meth:`Tafra.tuple_map`. We can also use ``numpy``'s built in apply
function, :meth:`numpy.apply_along_axis`, but it is considerably slower than
a ``@jit``'ed function.

.. code-block:: python

    >>> @jit(**jit_kw)
    ... def ndarray_map(qi: np.ndarray, Di: np.ndarray, bi: np.ndarray, t: np.ndarray) -> np.ndarray:
    ...     out = np.zeros((qi.shape[0], t.shape[0]))
    ...     for i in range(qi.shape[0]):
    ...         out[i, :] = hyp(qi[i], Di[i], bi[i], t)
    ...
    ...     return out
    81.2 µs ± 9.7 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


And the timing is neglible, meaning ``Tafra``'s :meth:`Tafra.tuple_map` is
essentially as performant as we are able to achieve in Python.
