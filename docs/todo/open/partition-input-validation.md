# partition skips validation and consumes a sort_by generator

Severity: low
Opened: 2026-10-19

## What happens now

`Tafra.partition` calls `GroupSet._build_group_indices` directly. It does not
call `GroupSet._validate`, which `group_by`, `transform` and `iterate_by` use,
so bad input gives raw numpy errors. Its `sort_by` argument is typed
`Iterable[str]` but is passed to `_sorted` once per group; a generator is
empty from the second group on.

## Evidence

Run 2026-10-19:

```
empty table, partition(['g'])        -> ValueError zero-size array to reduction operation maximum which has no identity
partition(['zz']) (no such column)   -> KeyError 'zz'          (iterate_by: ValueError "Column zz does not exist")
partition(['g'], sort_by=(c for c in ['x']))
                                      -> TypeError need sequence of keys with len > 0 in lexsort
```

All three are identical at BASE 390157d.

## What would fix it

At the top of `partition`: `columns = list(columns)`,
`sort_by = list(sort_by) if sort_by is not None else None`, then
`GroupSet._validate(self, columns)`. Decide first what an empty table should
return (an empty list is the natural answer; `_validate` raises).

## Files

- `tafra/base.py`: `Tafra.partition`
- `tafra/group.py`: `GroupSet._validate`
