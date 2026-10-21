# The join NULL-key rule lives in group.py, apart from the key-equality rule in _keys.py

Severity: low
Opened: 2026-10-21

## What happens now

`tafra/_keys.py` decides which key values are equal (NaN folding, -0.0, NaT).
Which key values are NULL is decided in `tafra/group.py`: `_NULLABLE_KINDS`,
`Join._non_null_mask`, `Join._is_non_null_object`, and the right-key `-1`
sentinel in `Join._equi_join_keys`. `_keys.encode_join_keys` encodes only; the
caller concatenates the sides first and patches NULL afterwards.

## Evidence

- Found by the /code-fix structural and duplication auditors, 2026-10-21.
- Adding one NULL marker (see `join-null-markers-not-covered.md`) touches both
  modules.
- `Tafra.coalesce` (tafra/base.py) has a third definition of "missing": for an
  object column `[nan, None, 1]` the join treats rows 0 and 1 as NULL, while
  `coalesce` fills only row 1 (run by the BY-RULE auditor, 2026-10-21).

## What would fix it

Move `_NULLABLE_KINDS`, `_non_null_mask` and `_is_non_null_object` into
`_keys`, and let `encode_join_keys(left_columns, right_columns)` own the
concatenation and the `-1` sentinel. Decide whether `coalesce` uses the same
rule, or document why not. Behaviour-preserving apart from `coalesce`.

## Files

- `tafra/group.py`: `Join._non_null_mask`, `Join._is_non_null_object`,
  `Join._equi_join_keys`, `Join._non_null_masks`
- `tafra/_keys.py`: `encode_join_keys`
- `tafra/base.py`: `Tafra.coalesce`
