# The join behaviour on a numpy whose datetime unit casts raise is untested

Severity: low
Opened: 2026-10-21

## What happens now

Joins promote mixed-unit datetime keys with `np.concatenate`. Two tests in
`test/test_accel_semantics.py` branch on whether that concatenation raises
`OverflowError`, and the guide says a newer numpy "can raise instead". The
raise branches have never run.

## Evidence

Run 2026-10-21, `np.concatenate([datetime64[ns] [1], datetime64[s] [2**54]])`:

```
numpy 2.2.6 (Python 3.10)  -> [..., NaT]   (wraps, no error)
numpy 2.4.3 (Python 3.14)  -> [..., NaT]   (wraps, no error)
```

The numpy 2.5.0 release notes (gh-31378) say datetime arithmetic now raises
`OverflowError` and that this "matches the overflow checking already performed
by unit-conversion casts". They do not name the exception type for casts.
https://numpy.org/devdocs/release/2.5.0-notes.html

## What is still unverified

- Whether numpy 2.5 raises on this cast, and with which exception type.

## What would unblock it

Install numpy 2.5 in a scratch environment, run
`test_null_key_made_by_unit_promotion_never_matches` and
`test_nested_loop_judges_null_on_promoted_key`, and correct the guide sentence
and the test comments to what was observed.

## Files

- `test/test_accel_semantics.py`, `docs/guide/groupby-partition.md`
