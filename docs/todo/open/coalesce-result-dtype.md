# coalesce does not define the result dtype when fills differ in dtype

Severity: low
Opened: 2026-10-19

## What happens now

`Tafra.coalesce` (tafra/base.py) carried the comment `# TODO: handle dtype?`
(migrated here). It copies the existing column, or, for a new column, starts
from the first fill (before 2026-10-21 a new numeric column was returned
uninitialized; fixed), then assigns each later fill into the `None` and NaN
slots. No promotion is applied when a fill's dtype differs from the column's.

## Evidence

Run 2026-10-21:

```
int64 column [1, 2] coalesced with [[2.5]]      -> dtype int64 (no slot filled)
object column [None, 1] coalesced with [[1.5]]  -> [1.5, 1] (object)
```

## What is still unverified

- What the original author meant by "handle dtype".

## What would unblock it

Run `coalesce` on an object column with `None` filled by `1.5`, and on a new
column whose fills are `[1, 2.5]`; record the dtypes here; then decide whether
the result should follow `np.result_type` of the column and all fills.

## Files

- `tafra/base.py`: `Tafra.coalesce`
