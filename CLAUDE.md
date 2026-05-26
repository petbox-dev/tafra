# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Tafra is a minimalist Python dataframe library — a lightweight alternative to pandas backed by numpy arrays. Authors: Derrick W. Turk, David S. Fulford. MIT license.

## Commands

```bash
# Install locally (editable)
pip install -e .

# Lint
ruff check tafra

# Type check (strict mode)
mypy tafra

# Run all tests (coverage enabled via pyproject.toml)
pytest

# Run a single test
pytest test/test_tafra.py::test_constructions

# Build C extension (optional — requires C compiler)
python setup.py build_ext --inplace
# On Windows with MinGW: python setup.py build_ext --inplace --compiler=mingw32

# Build distributable wheel
python -m build

# Run benchmarks
python test/bench_tafra.py
python test/bench_vs_pandas_vs_polars.py

# Build docs (MkDocs Material, deployed to GitHub Pages)
mkdocs build --strict
mkdocs serve  # local preview at http://127.0.0.1:8000/tafra/
```

## Architecture

The library has one core abstraction and a set of aggregation/partitioning operations.

**`Tafra`** (`tafra/base.py`) — The dataframe class. Wraps `dict[str, np.ndarray]` where every column must share the same row count. Tracks dtypes separately in `_dtypes` — this metadata represents the user's declared intent for column types and controls dtype validation in joins, unions, and dtype updates. Implements dict-like access (keys, values, items, get, update). Decorated with `@dataclass`. String columns use numpy `StringDType(na_object=None)` to support `None` values; `_dtypes` stores `'str'` for all string variants (`StringDType`, `<U`).

**Dtype design principle:** `_dtypes` metadata is the user's intent. Join/union validation compares `_dtypes` labels, not raw numpy dtypes. `update_dtypes_inplace` is how users change their intent — it updates the label AND casts the array. The `'str'` label maps to `StringDType(na_object=None)`. If assigning directly to `_data`, you must call `_coalesce_dtypes()` afterwards.

**Aggregation classes** (`tafra/group.py`) — SQL-style operations that operate on `Tafra` instances:
- `Union`, `GroupBy`, `Transform`, `IterateBy`, `InnerJoin`, `LeftJoin`, `CrossJoin`

**Vectorized aggregations** (`tafra/group.py`) — `GroupBy` detects known numpy reducers (`np.sum`, `np.mean`, `np.std`, `np.var`, `np.min`, `np.max`, `np.median`, `np.prod`, `np.ptp`, `np.any`, `np.all`, `len`, `np.count_nonzero`) and uses `np.bincount`/`ufunc.reduceat` instead of per-group Python loops. Custom aggregations: `percentile(q)`, `geomean`, `harmean`.

**C extension** (`tafra/_accel.c`) — Optional compiled acceleration. Single-pass grouped aggregation (Welford variance, sum, mean, min, max, count), O(n) hash-based equi-joins, O(n) composite key encoding (`composite_key`, `group_indices`), and O(n) string-to-integer encoding (`encode_strings`). Falls back to pure Python + numpy if not compiled. Build with `python setup.py build_ext --inplace --compiler=mingw32` (or omit `--compiler` for MSVC).

**Chunking/partitioning** (`tafra/base.py`):
- `chunks(n, sort_by=)` — split into n equal pieces
- `chunk_rows(size, sort_by=)` — split by max row count
- `partition(columns, sort_by=)` — split by group values for parallel dispatch
- `Tafra.concat(tafras)` — concatenate row-wise

**group_by vs partition:**
- `group_by` **reduces**: one row per group, applies aggregation functions (sum, mean, etc.)
- `partition` **splits**: returns all original rows grouped into sub-Tafras, no aggregation — designed for `multiprocessing.Pool.map()` dispatch

**Supporting modules:**
- `protocol.py` — Typing protocols for duck-typing compatibility (Series, DataFrame, Cursor)
- `formatter.py` — `ObjectFormatter` for custom dtype parsing (e.g., Decimal → float); string conversion from object to `StringDType(na_object=None)` is opt-in via `parse_object_dtypes_inplace()`
- `csvreader.py` — CSV reader with type inference; string columns produce `StringDType(na_object=None)` for nullable string support

## Testing

- pytest with hypothesis for property-based testing
- `build_tafra()` helper creates a standard 6-row test fixture
- `check_tafra()` validates structural integrity of a Tafra instance
- Mock `Series`, `DataFrame`, `Cursor` classes in tests match the protocol definitions
- `test/bench_tafra.py` — internal performance benchmarks
- `test/bench_vs_pandas_vs_polars.py` — comparison vs pandas and polars

## Configuration

- `pyproject.toml` — ruff (max-line-length=100), mypy (strict), pytest addopts, coverage

## Version bump

Update the version in all three places (they must stay in sync — conda-forge pulls from `meta.yaml`, PyPI from `pyproject.toml`):

1. `pyproject.toml` — `version = "X.Y.Z"`
2. `recipe/meta.yaml` — `{% set version = "X.Y.Z" %}`
3. `docs/changelog.md` — add a new `## X.Y.Z` section at the top with bulleted `**Fix**:` / `**Feature**:` entries
