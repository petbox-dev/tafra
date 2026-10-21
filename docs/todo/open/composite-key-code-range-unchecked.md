# C composite_key does not check that codes are inside their cardinality

Severity: low
Opened: 2026-10-19

## What happens now

`_accel.composite_key(columns, cardinalities)` computes
`key = c0 * N1 * ... + c1 * ...` without checking `0 <= c < N`. A code out of
range gives a key outside `[0, prod(N))` that can equal another row's key, so
two different groups would merge silently. Every other kernel that takes labels
raises `ValueError` for an out-of-range label.

## Evidence

Run 2026-10-19:

```
composite_key((array([0, 2]), array([1, 0])), (2, 2)) -> [1, 4]
```

4 is outside `[0, 4)`; with other codes it can collide (for example `(0, 2)`
and `(1, 0)` with cardinalities `(2, 2)` both give 2, per the /code-fix PLUMBING
auditor). The comment above `accel_composite_key` states the precondition
("codes in [0, cardinality)"); the comment in its body says "unsigned
arithmetic: a code outside its cardinality wraps instead of being undefined
behavior".

## Blast radius

None through tafra today: `tafra._keys.encode_column` always produces codes in
`[0, cardinality)`. Only a direct caller of `_accel.composite_key` can hit it.

## What would fix it

Validate each code in the C loop (as `LABEL_OUT_OF_RANGE` does for labels) and
raise `ValueError`; add a test next to
`test_groupby_kernels_reject_out_of_range_labels`. Measure the cost first: the
check adds one compare per code in a hot loop.

## Files

- `tafra/_accel.c`: `accel_composite_key`
- `test/test_accel_semantics.py`
