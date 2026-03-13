# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Tafra is a minimalist Python dataframe library — a lightweight alternative to pandas backed by numpy arrays. Authors: Derrick W. Turk, David S. Fulford. MIT license.

## Commands

```bash
# Install locally (editable)
pip install -e .

# Lint
flake8 tafra

# Type check (strict mode)
mypy tafra

# Run all tests (coverage enabled via setup.cfg)
pytest

# Run a single test
pytest test/test_tafra.py::test_constructions

# Build docs
sphinx-build -W -b html docs docs/_build/html
```

## Architecture

The library has one core abstraction and a set of aggregation operations.

**`Tafra`** (`tafra/base.py`) — The dataframe class. Wraps `Dict[str, np.ndarray]` where every column must share the same row count. Tracks dtypes separately in `_dtypes`. Implements dict-like access (keys, values, items, get, update). Decorated with `@dataclass`.

**Aggregation classes** (`tafra/group.py`) — SQL-style operations that operate on `Tafra` instances:
- `Union`, `GroupBy`, `Transform`, `IterateBy`, `InnerJoin`, `LeftJoin`, `CrossJoin`

**Supporting modules:**
- `protocol.py` — Typing protocols for duck-typing compatibility (Series, DataFrame, Cursor)
- `formatter.py` — `ObjectFormatter` for custom dtype parsing (e.g., Decimal → float)
- `csvreader.py` — CSV reader with type inference

## Testing

- pytest with hypothesis for property-based testing
- `build_tafra()` helper creates a standard 6-row test fixture
- `check_tafra()` validates structural integrity of a Tafra instance
- Mock `Series`, `DataFrame`, `Cursor` classes in tests match the protocol definitions

## Configuration

- `setup.cfg` — flake8 (max-line-length=100, extensive ignore list), mypy (strict), pytest addopts
- `.coveragerc` — coverage exclusion rules
- `.travis.yml` — CI runs flake8, mypy, pytest, sphinx-build on Python 3.7/3.8
