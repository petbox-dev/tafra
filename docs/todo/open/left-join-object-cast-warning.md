# Equi left join warns "cast to object" for a right column that is already object

Severity: low
Opened: 2026-10-19

## What happens now

When an equi left join has unmatched rows, `LeftJoin._null_filled_column`
warns "has unmatched rows and no native null representation. Dtype has been
cast to object" for every right column whose kind is not in
`_NATIVE_NULL_FILL_KINDS` -- including kind `O`, which is already object and
already holds `None`. The nested-loop path (`_append_unmatched`) checks
`dtypes[column] != "object"` and does not warn. The two sibling paths disagree.

## Evidence

Run 2026-10-19:

```
left = {k: [1, 2]}, right = {k: [1], o: object [{'x': 1}]}
left_join(on '==')  -> [{'x': 1}, None], warning "Left join: column 'o' (dtype object) ..."
left_join(on '<=')  -> [{'x': 1}, None], no warning
```

Identical at BASE 390157d and at HEAD 99c4e47 (pre-existing). The right-empty
shortcut also warns, because it goes through the same `_null_filled_column`.

## What would fix it

In `_null_filled_column`, warn only when `kind != "O"`. Pin it with a
`warnings.catch_warnings(record=True)` test on both paths.

## Files

- `tafra/group.py`: `LeftJoin._null_filled_column`, `LeftJoin._append_unmatched`
