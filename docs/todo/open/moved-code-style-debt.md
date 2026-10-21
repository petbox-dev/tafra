# Code moved into _keys and _reducers keeps shorthand names and mypy --strict errors

Severity: low
Opened: 2026-10-21

## What happens now

The move of key encoding and reducers out of `tafra/group.py` kept the bodies
byte-identical, so their style debt moved with them:

- Shorthand names against the project naming rule: `n_rows`, `n_groups`,
  `first_pos` in `tafra/_keys.py` (`direct_labels_*`); `col_arrays`,
  `first_seen_idx`, `li`, `ri`, `c`, `d` in `tafra/group.py`. `_reducers`
  calls the same quantity `group_count`.
- `mypy --strict tafra` reports 7 errors in 2 files, the same count as at
  4201d46: four `no-any-return` in `tafra/_reducers.py` (kernel results are
  untyped), two `no-any-return` and one IPython `attr-defined` in
  `tafra/base.py`. The project's own `mypy tafra` passes.

## Evidence

Counts run 2026-10-21 by the /code-fix reducer reviewer (4201d46 archive and
working tree) and confirmed on the working tree after the `_kernels` re-export
fix.

## What would fix it

Rename to `row_count`, `group_count`, `first_seen_positions` and so on across
`_keys` and `group.py`; add `tafra/_accel.pyi` so the kernels are typed, which
removes the `no-any-return` errors and the `cast(...)` calls.

## Files

- `tafra/_keys.py`, `tafra/_reducers.py`, `tafra/group.py`, `tafra/base.py`
