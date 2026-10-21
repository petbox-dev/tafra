# Version History

## 2.3.0

* **Feature**: Wheels for Python 3.14 and 3.15, including the free-threaded
  (no-GIL) builds `cp314t` and `cp315t`. The C extension declares that it
  runs without the GIL, so importing it does not re-enable the GIL on a
  free-threaded interpreter. Its hash-table build reads each key once and
  caches its slot, so a thread that writes a key column during a join cannot
  make the build write past its buffer; this also removes a table-sized
  scratch buffer (16.8 MB for a 1M-row right side). CI tests all eight
  interpreters (3.10 to 3.15, plus 3.14t and 3.15t) and fails if the C
  extension did not build, and new tests run `group_by`, `transform` and
  joins on one shared `Tafra` from several threads. A new
  [Thread Safety](guide/thread-safety.md) guide states which concurrent
  operations are safe.
* **Fix**: A callable that Python allocated at the address of a freed
  `percentile(q)` object was routed to that percentile's vectorized fast
  path, so `group_by` and `transform` silently returned percentiles in place
  of the callable's result. `percentile` objects now carry their own fast
  path and are no longer registered by `id()`, which also removes a registry
  entry that leaked for each `percentile` call.
* **Fix**: `group_by`, `transform` and `iterate_by` raised `IndexError` when
  the key columns were given as a generator. A generator now works, and an
  empty list of key columns raises `ValueError` in these and in `partition`.
* **Removed**: `AggMethod.result_factory`, which nothing in tafra called.
* **Fix**: An equi-join on a float key column cast the key to `int64` before
  the C hash join, so `1.2` matched `1.7`, and `1e19`, `inf` and `-inf`
  matched one another. A multi-column key with a float column was wrong on
  both the C and the pure-Python paths. Float keys now match by value, and
  `-0.0` matches `0.0`.
* **Fix**: `group_by`, `transform` and `iterate_by` raised on key columns of
  float, `bool`, `datetime64` with a unit other than `ns`, `uint64` above the
  `int64` range, and `StringDType` holding `None` (below 50,000 rows). Integer
  IDs with a wide range (for example 14-digit well numbers) raised
  `MemoryError`, and two wide integer key columns raised `Composite key space
  overflow`. Every key dtype that numpy compares with `==` now works, and a
  wide key range is re-encoded to dense codes. In `group_by`, all NaN keys
  of a float column form one group and all NaT keys form one group, as in
  `np.unique`.
* **Fix**: A multi-column key with a pre-epoch `datetime64` or a negative
  `timedelta64` produced colliding keys and joined the wrong rows.
* **Fix**: `np.nansum`, `np.nanmean`, `np.nanstd`, `np.nanvar` and
  `np.nanprod` used the NaN-propagating fast path and returned NaN.
  `np.median` and `np.nanmedian` picked the middle of a sorted group with its
  NaN values still in it, so `[1, nan, 4]` gave 4. With the C extension,
  `np.min` / `np.max` skipped NaN, and returned `inf` for an all-NaN group.
  The fast path now gives numpy's NaN result for every registered function.
* **Fix**: With the C extension, `np.min` / `np.max` kept the first of equal
  values, so `-0.0` and `0.0` came out with the wrong sign. They now follow
  numpy: `np.min` / `np.max` keep the last of equal values, and `np.nanmin`
  gives `-0.0` and `np.nanmax` `0.0` in any order.
* **Fix**: With the C extension, `np.sum`, `np.min` and `np.max` converted the
  column to `float64`: an `int64` sum lost precision above 2**53, and `np.min`
  of an `int64`, `bool` or `datetime64` column returned `float64`. Results now
  have numpy's dtype (exact `int64` / `uint64` sums, `datetime64` minimums,
  `float32` stays `float32`). A reducer on a dtype numpy refuses (for
  example `np.mean` of strings) takes the per-group path and raises numpy's
  error.
* **Fix**: The per-group loop (custom functions, and registered functions on
  a dtype the fast path does not take) stored each result in the input
  column's dtype: a mean of an `int64` column was truncated to `int64`,
  `np.any` of a string column gave the strings `'True'`, and a string result
  was cut to the input's `<U` width. The result column now takes numpy's
  promoted dtype of the per-group results (`object` when they share none),
  in `group_by` and `transform`. A string mixed with a non-string also gives
  `object`, so `'low'` with `2.0` keeps `2.0` a float; numpy would turn it
  into the text `'2.0'`.
* **Fix**: An equi left join put left rows with a NULL key (NaN, NaT,
  `None`) at the end of the result. They now keep their left-table position.
* **Fix**: A join with a non-`==` operator judged NULL keys before unit
  promotion, so a `datetime64[s]` key that becomes NaT in `ns` matched under
  `!=`. NULL is now judged on the promoted values for every operator, as the
  `==` join already did.
* **Fix**: `coalesce` into a new numeric column returned uninitialized memory:
  the column was allocated empty and only `None` or NaN slots were filled. A
  new column now starts as the first fill, as SQL `COALESCE`. With no fills,
  `coalesce` raised `StopIteration`; it now returns an existing column
  unchanged and raises `ValueError` for a new column.
* **Fix**: An `==` join on two `StringDType` keys with different `na_object`
  values (for example `None` and `''`) raised `TypeError: Cannot find common
  instance for incompatible dtypes`, and a NULL written as `''` or NaN in such
  a key matched as a value (under `!=`, or NaN to NaN under `==`). Each
  `StringDType` key is now read with `na_object=None`, so its NULLs never
  match, for every operator.
* **Fix**: The pure-Python `np.var` / `np.std` fast path used
  `E[x^2] - E[x]^2`, which returned 2.0 for three values of 1e8 + 0.1. It
  now subtracts the group mean first. The C kernel used Welford's one-pass
  update, with a relative error near 1e-7 on values offset by 1e9 (numpy:
  near 1e-13); it is now two-pass as well.
* **Fix**: The C extension now raises `ValueError` for group labels outside
  `[0, n_groups)`, for labels and data of different lengths, and for
  multi-dimensional input, in place of reading or writing out of bounds.
* **Perf**: The C hash tables used the low bits of `key * GOLDEN_HASH`, so
  keys with trailing zero bits (`datetime64[ns]` dates, IDs with a
  power-of-two step) collided in a few slots. A join on 200,000
  `datetime64[ns]` daily keys took 4.6 to 10 s; it now takes 66 to 103 ms.
  The tables now use the high bits (Fibonacci hashing).
* **Perf**: Exact C kernels for integer sum, min and max. Object and string
  keys use the C encoder at every size, in place of below 50,000 rows only.
  For `object` and `StringDType` keys it is several times faster than the
  dict codebook and `np.unique` at the sizes measured (10 to 49,999 rows);
  for `<U` keys it ranges from somewhat slower to a few times faster than
  `np.unique`, depending on the number of distinct keys.
* **Perf**: The C extension releases the GIL in its integer hashing, join
  and aggregation loops (string and `object` key encoding hashes Python
  objects and keeps it): four threads ran four 2M x 2M joins 2.5x faster than one
  thread on a standard interpreter.

## 2.2.5

* **Fix**: Join and `group_by` on `object`-dtype key columns containing mixed
  Python types (e.g., `str` and `int` across sides, or mixed within one side)
  no longer raise `TypeError: '<' not supported between instances of ...`.
  The pure-Python encoding path now uses a hash-based codebook for object
  arrays, matching the C accelerator's semantics. Previously, `np.unique`'s
  internal `argsort` failed on cross-type comparisons.

## 2.2.4

* **Fix**: Restore `MANIFEST.in` so test CSV fixtures (`test/ex*.csv`) are
  included in the PyPI sdist.  Removing `MANIFEST.in` in 2.2.1 silently
  broke the conda-forge test stage for 2.2.1, 2.2.2, and 2.2.3 because
  `source_files: test` could not find the CSVs.  No code changes.

## 2.2.3

* **Feature**: `read_sql` and `read_sql_chunks` accept an optional
  ``params: Sequence[Any] | None`` argument for parameterized queries using
  ``?`` placeholders.  Passed unchanged to ``cur.execute``.  The `Cursor`
  protocol's ``execute`` signature is updated to match pyodbc's variadic form.
* **Fix**: CSV fixture paths in `test_csv` now resolve relative to
  `Path(__file__).parent` instead of a hardcoded `"test/"` prefix. This fixes
  `FileNotFoundError` in conda-forge builds where pytest's cwd is not the
  repo root.

## 2.2.2

* **Fix**: Multi-column join composite key collision when left and right sides
  have different numbers of unique values. `_build_composite_key` now uses
  shared cardinalities (max over both sides) so positional encoding is
  consistent between left and right keys.
* **Fix**: Constructor no longer auto-converts object arrays of strings to
  `StringDType`. This preserves the caller's dtype and prevents `union()`
  mismatches when one Tafra comes from an external source with object dtype.
  Use `parse_object_dtypes_inplace()` for explicit conversion.
* **Perf**: `_accel.c` v3 — combine bitmask hashing with slot-caching joins.

## 2.2.1

* **Fix**: Join and Union dtype validation now compares `_dtypes` metadata
  (user intent) instead of raw numpy dtypes. Since `_format_dtype` collapses
  `StringDType`, `<U8`, `<U12` etc. all to `'str'`, string columns with
  different underlying representations no longer reject as mismatched.
* **Fix**: `update_dtypes_inplace` preserves raw numpy dtypes for casting and
  maps `'str'` label to `StringDType(na_object=None)` -- explicit
  `update_dtypes_inplace({'x': 'str'})` now converts `<U` columns to
  nullable `StringDType`. Construction preserves the
  original dtype unchanged.
* **Fix**: Left join null-fill preserves column dtypes -- string columns use
  `StringDType(na_object=None)`, float columns use `NaN`, datetime/timedelta
  columns use `NaT`. Only int/bool/bytes fall back to `object` (with a
  warning). Note: left join results with null-filled string columns carry
  `StringDType(na_object=None)`, which differs from plain `StringDType()`.
* Warn when left join casts a column to `object` dtype due to unmatched rows

## 2.2.0

* **Performance**: Add C-accelerated `composite_key` for single-pass
  multi-column key encoding -- eliminates Python loop and temporaries
* **Performance**: Add C-accelerated `group_indices` with O(n) hash-based
  group construction -- replaces `np.unique` + `argsort` + `split` pipeline
* **Performance**: Add C-accelerated `encode_strings` with O(n) hash-based
  string-to-integer encoding -- replaces O(n log n) `np.unique`; 2-2.5x
  faster on string-column groupby
* **Performance**: Multi-column GroupBy 1.9x faster (100k 2-col: 8.72 → 4.00 ms;
  1M 2-col: 97 → 48 ms)
* **Performance**: Add `tuple_map(name=None)` fast path -- skips NamedTuple
  construction, uses `zip(*values)` directly; now beats pandas `itertuples`
* **Modernize**: Replace `Dict`, `List`, `Tuple`, `Optional`, `Union` with
  Python 3.10+ built-in generics and `|` union syntax; add
  `from __future__ import annotations`
* **Docs**: New MkDocs Material website with custom landing page, benchmark
  charts, tutorials, and API reference with structured parameter tables
* Remove RST cross-reference syntax from all docstrings
* Fix docstring parameter name mismatches (`df` → `s`, `column` → `columns`,
  `group_by` → `columns`)
* Normalize numpy-style docstring indentation for griffe compatibility

## 2.1.0

* **Performance**: Rewrite `GroupBy`, `Transform`, `IterateBy` to use
  index-based grouping (`_build_group_indices`) instead of per-group boolean
  masks -- up to 23x faster on multi-column groups
* **Performance**: Add numpy-native sort-merge join for equi-joins in
  `InnerJoin` and `LeftJoin` -- up to 20x faster than pre-2.1, now faster
  than `pandas` (1.4--7.6x) across all tested sizes
* **Performance**: Fix `_validate_columns` to use O(1) dict lookup instead of
  O(n) `.keys()` view
* **Performance**: Skip redundant `np.dtype()` call in `_format_dtype` when
  input is already `np.dtype`
* **Performance**: Use `dtype.kind` check instead of `np.dtype(object)`
  allocation in `ObjectFormatter.parse_dtype`
* **Breaking**: Adopt numpy `StringDType` for string columns -- string data
  previously stored as `dtype=object` now uses `np.dtypes.StringDType()`
  (dtype kind `'T'`); internal dtype name is `'str'` instead of `'object'`
* Auto-convert object arrays of Python strings to `StringDType` during
  construction and column assignment
* `CSVReader` string columns now produce `StringDType` instead of `object`
* Fix `Transform.apply` correctness: group-by column data was being copied
  before the boolean mask was fully constructed
* Fix `update_dtypes_inplace` to handle `StringDType` columns when replacing
  empty strings for numeric conversion
* **Performance**: Add vectorized fast path for `GroupBy` when aggregation
  functions are recognized numpy reducers -- uses `np.bincount` /
  `ufunc.reduceat` instead of per-group Python loops; now faster than
  `pandas` at small-to-medium scales. Recognized functions: `np.sum`,
  `np.mean`, `np.std`, `np.var`, `np.min`, `np.max`, `np.ptp`,
  `np.prod`, `np.median`, `np.any`, `np.all`, `np.count_nonzero`,
  `len`, `sum`, plus all nan-variants
* Add `percentile(q)` aggregation factory for use in `group_by` -- creates
  vectorized fast-path callables for arbitrary percentiles
* Add `geomean` and `harmean` aggregation functions (geometric and harmonic
  means) with vectorized fast path
* **Performance**: Replace `np.unique` (O(n log n) sort) with direct array
  mapping (O(n)) for group label assignment -- GroupBy is now 4--8x faster than
  both pandas and polars at <=10k rows
* **Performance**: String/object columns auto-encoded to integer codes for
  grouping -- eliminates the Python dict fallback for multi-column groups
* Add `chunks(n, sort_by=)`: split into `n` equal-sized `Tafra`
  chunks with optional pre-sort
* Add `chunk_rows(size, sort_by=)`: split by maximum row count
* Add `partition(columns, sort_by=)`: split by group values (like
  `GroupBy` but returns sub-`Tafra` instances for parallel dispatch);
  supports sorting within each partition
* Add `Tafra.concat(tafras)`: concatenate multiple `Tafra` row-wise
* Add `tail(n)`: return last n rows (complement to `head`)
* Add `sort(columns, reverse=)`: public sort API with multi-column and
  descending support
* Add `sample(n, seed=)`: random row sampling with reproducible seed
* Add `drop_duplicates(columns)`: deduplicate rows by column values
* Add `value_counts(column)`: count occurrences of each unique value
* Add `describe()`: summary statistics (count, mean, std, min, quartiles,
  max) for all numeric columns
* Add `shift(n)`: lag/lead rows by n positions, filling with NaN/None
* Optional C extension (`tafra/_accel.c`) for single-pass grouped
  aggregation (Welford variance, sum, mean, min, max, count) and O(n)
  hash-based equi-joins -- falls back to pure Python + numpy if not compiled
* Fix join codebook bug: string-key joins now use shared codebook across
  left/right tables via `_encode_columns_paired`
* `Tafra.concat()` validates that all tafras have matching column sets
* Add 40 new tests (88 total)
* Add `test/bench_tafra.py` performance benchmark suite
* Add `test/bench_vs_pandas_vs_polars.py` -- 5-way comparison (Tafra+C,
  Tafra pure, pandas 2.3, pandas 3.0, polars 1.39)


## 2.0.0

* **Breaking**: Require Python >=3.9 (was >=3.7)
* **Breaking**: Require numpy >=2.1 (was >=1.17)
* Replace Travis CI with GitHub Actions (lint, test matrix 3.9-3.13, docs)
* Modernize `.readthedocs.yml` for current RTD build system
* Bump Sphinx >=7.0, sphinx-rtd-theme >=2.0
* Fix deprecated Sphinx `html_context` CSS configuration
* Update README badges (remove Travis/Coveralls, add Python versions)


## 1.1.0

* Fix `LeftJoin` dtype merge order (right was overwriting left)
* Fix `_parse_iterable` re-iterating consumed iterable
* Fix `to_csv` `UnboundLocalError` for unsupported file types
* Fix `CSVReader` file handle leak on empty files
* Fix `IterateBy` yielding inconsistent types (always tuple now)
* Fix `ObjectFormatter.__setitem__` catching its own `ValueError`
* Fix `ndim` returning column count instead of 2
* Fix `_parse_sequence`/`_parse_iterable`/`_parse_iterator` mutating caller's dicts
* Replace bare `except` clauses with specific exception types
* Remove `warnings.resetwarnings()` global side effect
* Fix mutable default arguments in function signatures
* Migrate to `pyproject.toml` (remove `setup.cfg`, `.coveragerc`; `setup.py`
  retained for C extension builds in 2.1.0)
* Version now in `pyproject.toml`, read via `importlib.metadata`
* Replace `flake8` with `ruff`
* Fix all 68 `mypy` errors (strict mode)
* Parameterize all `np.ndarray` type annotations
* Add 9 new tests covering all bug fixes (48 tests, 99% coverage)


## 1.0.10

* Add `pipe` and overload `>>` operator for Tafra objects

## 1.0.9

* Add test files to build

## 1.0.8

* Check rows in constructor to ensure equal data length

## 1.0.7

* Handle missing or NULL values in `read_csv()`.
* Cast empty elements to None when updating dtypes to avoid failure of `np.astype()`.
* Update some typing, minor refactoring for performance


## 1.0.6

* Additional validations in constructor, primary to evaluate Iterables of values
* Split `col_map` to `col_map` and `key_map` as the original function's return signature depending upon an argument.
* Fix some documentation typos


## 1.0.5

* Add `tuple_map` method
* Refactor all iterators and `..._map` functions to improve performance
* Unpack `np.ndarray` if given as keys to constructor
* Add `validate=False` in `__post_init__` if inputs are **known** to be valid to improve performance


## 1.0.4

* Add `read_csv`, `to_csv`
* Various refactoring and improvement in data validation
* Add `typing_extensions` to dependencies
* Change method of `dtype` storage, extract `str` representation from `np.dtype()`


## 1.0.3

* Add `read_sql` and `read_sql_chunks`
* Add `to_tuple` and `to_pandas`
* Cleanup constructor data validation


## 1.0.2

* Add object_formatter to expose user formatting for dtype=object
* Improvements to indexing and slicing


## 1.0.1

* Add iter functions
* Add map functions
* Various constructor improvements


## 1.0.0

* Initial Release
