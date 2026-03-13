===============
Version History
===============

.. automodule:: tafra
   :noindex:

2.0.0
-----

* **Breaking**: Require Python >=3.9 (was >=3.7)
* **Breaking**: Require numpy >=2.1 (was >=1.17)
* Replace Travis CI with GitHub Actions (lint, test matrix 3.9-3.13, docs)
* Modernize ``.readthedocs.yml`` for current RTD build system
* Bump Sphinx >=7.0, sphinx-rtd-theme >=2.0
* Fix deprecated Sphinx ``html_context`` CSS configuration
* Update README badges (remove Travis/Coveralls, add Python versions)


1.1.0
-----

* Fix ``LeftJoin`` dtype merge order (right was overwriting left)
* Fix ``_parse_iterable`` re-iterating consumed iterable
* Fix ``to_csv`` ``UnboundLocalError`` for unsupported file types
* Fix ``CSVReader`` file handle leak on empty files
* Fix ``IterateBy`` yielding inconsistent types (always tuple now)
* Fix ``ObjectFormatter.__setitem__`` catching its own ``ValueError``
* Fix ``ndim`` returning column count instead of 2
* Fix ``_parse_sequence``/``_parse_iterable``/``_parse_iterator`` mutating caller's dicts
* Replace bare ``except`` clauses with specific exception types
* Remove ``warnings.resetwarnings()`` global side effect
* Fix mutable default arguments in function signatures
* Migrate to ``pyproject.toml`` (remove ``setup.py``, ``setup.cfg``, ``.coveragerc``)
* Version now in ``pyproject.toml``, read via ``importlib.metadata``
* Replace ``flake8`` with ``ruff``
* Fix all 68 ``mypy`` errors (strict mode)
* Parameterize all ``np.ndarray`` type annotations
* Add 9 new tests covering all bug fixes (48 tests, 99% coverage)


1.0.10
------

* Add ``pipe`` and overload ``>>`` operator for Tafra objects

1.0.9
-----

* Add test files to build

1.0.8
-----

* Check rows in constructor to ensure equal data length

1.0.7
-----

* Handle missing or NULL values in ``read_csv()``.
* Cast empty elements to None when updating dtypes to avoid failure of ``np.astype()``.
* Update some typing, minor refactoring for performance


1.0.6
-----

* Additional validations in constructor, primary to evaluate Iterables of values
* Split ``col_map`` to ``col_map`` and ``key_map`` as the original function's return signature depending upon an argument.
* Fix some documentation typos


1.0.5
-----

* Add ``tuple_map`` method
* Refactor all iterators and ``..._map`` functions to improve performance
* Unpack ``np.ndarray`` if given as keys to constructor
* Add ``validate=False`` in ``__post_init__`` if inputs are **known** to be valid to improve performance


1.0.4
-----

* Add ``read_csv``, ``to_csv``
* Various refactoring and improvement in data validation
* Add ``typing_extensions`` to dependencies
* Change method of ``dtype`` storage, extract ``str`` representation from ``np.dtype()``


1.0.3
-----

* Add ``read_sql`` and ``read_sql_chunks``
* Add ``to_tuple`` and ``to_pandas``
* Cleanup constructor data validation


1.0.2
-----

* Add object_formatter to expose user formatting for dtype=object
* Improvements to indexing and slicing


1.0.1
-----

* Add iter functions
* Add map functions
* Various constructor improvements


1.0.0
-----

* Initial Release
