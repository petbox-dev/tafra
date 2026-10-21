# drop_duplicates and value_counts disagree with group_by on key equality

Severity: medium
Opened: 2026-10-19

## What happens now

`group_by`, `transform`, `iterate_by`, `partition` and the equi-joins encode
keys with `tafra._keys.encode_column`: one NaN key per float column, `None`
and mixed object types allowed. `Tafra.drop_duplicates` and
`Tafra.value_counts` do not use it. They call `np.unique` (on a structured
array for several columns), which does not merge NaN in a structured array and
cannot sort mixed objects or StringDType holding `None`.

## Evidence

Run 2026-10-19:

```
t = {k: [nan, nan, 1.0], j: [1, 1, 1]}
t.drop_duplicates(['k', 'j'])        -> 3 rows   (two identical rows kept)
t.group_by(['k', 'j'], ...)          -> 2 rows
object k = ['a', None, 1]: drop_duplicates(['k'])
  -> TypeError '<' not supported between instances of 'str' and 'NoneType'
StringDType s = ['a', None, 'a']: value_counts('s'), drop_duplicates(['s'])
  -> ValueError Cannot compare null that is not a nan-like value
```

`drop_duplicates` is unchanged since main (2.2.5, 390157d), so the code is not
new. The disagreement became visible in 2.3.0 for the keys `group_by` started
to accept then: a float NaN in a multi-column key and StringDType `None` raised
in `group_by` at 390157d. Object keys such as `['a', None, 1]` were already
accepted there.

## What would fix it

Build both on `GroupSet._build_group_indices` (first-seen row per group), as
`partition` already does: `drop_duplicates` keeps the first-seen rows,
`value_counts` takes the group sizes. Delete the structured-array and dict
branches. Row order does not change: `drop_duplicates` already sorts the
first-occurrence indices (`idx.sort()`), so it returns first-seen rows
(2026-10-21: keys `[3, 1, 3, 2, 1]` -> `[3, 1, 2]`).

The NaN disagreement needs two or more key columns: one float column goes
through `np.unique`, which merges NaN on numpy 2.

## Files

- `tafra/base.py`: `Tafra.drop_duplicates`, `Tafra.value_counts`
- `tafra/group.py`: `GroupSet._build_group_indices`
