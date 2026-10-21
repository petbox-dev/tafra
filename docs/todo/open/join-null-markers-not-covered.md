# pd.NA, pd.NaT and complex NaN are not NULL in joins

Severity: low
Opened: 2026-10-19

## What happens now

`Join._non_null_mask` treats as NULL: NaN in a float column, NaT in a
datetime/timedelta column, `None` in a StringDType column, and `None`, float
NaN or NaT in an object column. Other missing markers are joined as ordinary
values. The behaviour is documented in `docs/guide/groupby-partition.md`
("Key Columns"), so this is a decision to make, not an undocumented defect.

## Evidence

Run 2026-10-19 (C and pure paths agree):

```
object keys [pd.NA, 'a'] inner_join [pd.NA]          -> 1 row (pd.NA matched pd.NA)
complex keys [complex(nan, 0), 1] inner_join [complex(nan, 0)] -> 1 row
```

Also reported by the /code-fix auditors: `pd.NaT` in an object column matches
itself. Same at BASE 390157d.

A `StringDType` whose `na_object` is NaN or `''` no longer belongs here: since
2026-10-21 each StringDType key is cast to `na_object=None` before the NULL
check, so those NULLs never match (`test_string_keys_with_different_null_markers`).

## What would fix it

Extend `_NULLABLE_KINDS` with `'c'` and add a complex branch
(`np.isnan(column.real) | np.isnan(column.imag)`); extend
`Join._is_non_null_object` with `pd.NA` / `pd.NaT` checks that do not import
pandas (for example `value is getattr(sys.modules.get('pandas'), 'NA', None)`).
Then update the guide.

## What is still unverified

- Whether any user relies on `pd.NA` keys matching.

## Files

- `tafra/group.py`: `_NULLABLE_KINDS`, `Join._non_null_mask`, `Join._is_non_null_object`
- `docs/guide/groupby-partition.md`: "Key Columns"
