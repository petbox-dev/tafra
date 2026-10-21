# Non-equi joins corrupt StringDType None payloads and raise on a None key

Severity: medium
Opened: 2026-10-19

## What happens now

`InnerJoin._apply_nested_loop` and `LeftJoin._apply_nested_loop` (any join with
an operator other than `==`) build each output column from a Python list with
`np.array(values)`. Two defects follow:

1. A `StringDType(na_object=None)` payload column loses its nulls: `None`
   becomes the string `'None'` and the column becomes `<U`.
2. A `None` in a right key column (object or StringDType) makes the comparison
   raise, because `op(left_value, right_column)` runs over the whole right
   column before the NULL mask is applied.

The equi-join path is correct for both: it gathers by index and masks NULL keys.

## Evidence

Run 2026-10-19 on the working tree (C extension and pure path give the same):

```
left  = {k: [1, 2, 3], s: StringDType ['a', None, 'c']}
right = {k: [2, 3], r: [10, 20]}
left.inner_join(right, [('k', 'k', '<=')])['s']
  -> dtype <U4, ['a', 'a', 'None', 'None', 'c']

object keys ['a','b'] vs ['a', None], op '>'
  -> TypeError '<' not supported between instances of 'NoneType' and 'str'
StringDType keys ['a','b'] vs ['a', None], op '<'
  -> ValueError 'greater' not supported for null values that are not nan-like or strings.
```

The same results at BASE 390157d (checked by the /code-fix WRITE auditor), so
this is not a regression from 2.3.0.

## What was ruled out

- Not a dtype-label problem: `_dtypes` still says `'str'`; the array itself is
  `<U4` holding the text `'None'`.
- `!=` does not raise on a None key; only the ordering operators do.

## What would fix it

- Build nested-loop output by collecting `(left_index, right_index)` pairs and
  gathering, as `_gather` does for the equi path, so payload dtypes survive.
- Apply `right_null` before the comparison: compare only `right_column[right_null]`.

## Files

- `tafra/group.py`: `InnerJoin._apply_nested_loop`, `LeftJoin._apply_nested_loop`
