# The C extension is detected at several sites, and a failed load loses its cause

Severity: low
Opened: 2026-10-21

## What happens now

`tafra/_kernels.py` sets `HAS_ACCEL` from one try-import of all 14 kernels and
discards the `ImportError`. CI, publish and the recipe assert `HAS_ACCEL` with
the message `'C extension missing'` only, so "not built" and "stale build
missing one kernel" look the same.

Other sites detect the extension on their own:

- `test/test_tafra.py` (C extension direct tests): own `_HAS_ACCEL` from an
  import of 3 kernels.
- `README.md`, `docs/getting-started.md`, `recipe/meta.yaml`: "verify" with
  `from tafra._accel import groupby_sum`, which passes on a build that lacks a
  later kernel while the library runs on pure numpy.
- `test/bench_vs_pandas_vs_polars.py` and `test/profile_join.py`: identical
  `_disable_accel` / `_restore_accel`, with no `try/finally`.

## Evidence

Found by the /code-fix BY-SYMBOL and DETECTION auditors, 2026-10-21. All sites
are pre-existing; the move to `_kernels` changed none of them.

## What would fix it

Keep the `ImportError` as `_kernels.IMPORT_ERROR` and show it in the CI
assertion; point the docs snippets and the test module at `_kernels.HAS_ACCEL`;
add one context manager that disables acceleration, for the scripts.

## Files

- `tafra/_kernels.py`, `test/test_tafra.py`, `README.md`,
  `docs/getting-started.md`, `recipe/meta.yaml`,
  `test/bench_vs_pandas_vs_polars.py`, `test/profile_join.py`
