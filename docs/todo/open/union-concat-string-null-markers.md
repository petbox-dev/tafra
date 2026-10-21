# union and concat raise on StringDType columns with different na_object values

Severity: low
Opened: 2026-10-21

## What happens now

Joins read each StringDType key with `na_object=None` (`Join._none_null_strings`),
so two keys whose `na_object` values differ join. `Tafra.union` and
`Tafra.concat` append columns with `np.append` / `np.concatenate` and no such
cast, so the same pair of columns raises.

## Evidence

Run 2026-10-21:

```
a = {s: StringDType(na_object=None) ['a', None]}
b = {s: StringDType(na_object='')   ['b', '']}
a.union(b)            -> TypeError Cannot find common instance for incompatible dtypes
Tafra.concat([a, b])  -> TypeError Cannot find common instance for incompatible dtypes
```

Pre-existing: the join fix (fix/review-followups) did not touch these paths;
found by the correctness auditor of that branch.

## What is still unverified

- Whether the result should keep one side's `na_object` or always take
  `na_object=None` (tafra's string convention).

## What would fix it

Cast each StringDType column whose `na_object` is not None to
`StringDType(na_object=None)` before appending, as `Join._none_null_strings`
does, and pin it with the example above for `union`, `union_inplace` and
`concat`.

## Files

- `tafra/base.py`: `Tafra.union`, `Tafra.union_inplace`, `Tafra.concat`
- `tafra/group.py`: `Union`, `Join._none_null_strings`
