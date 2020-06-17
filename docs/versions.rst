===============
Version History
===============

.. automodule:: tafra
   :noindex:


1.0.6
------

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
