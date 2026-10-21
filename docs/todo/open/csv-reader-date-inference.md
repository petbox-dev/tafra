# CSV reader does not infer date or datetime columns

Severity: low
Opened: 2026-10-19

## What happens now

`tafra/csvreader.py` infers column types from `_TYPE_PRECEDENCE`: int32,
int64, float64, bool. Anything else is read as `StringDType(na_object=None)`.
The list ended with `# TODO: date,` and `# TODO: datetime,` (migrated here), so
date columns arrive as strings and the caller must convert them.

## What is still unverified

- Which date formats the CSVs read with tafra actually use; ISO 8601 is the
  only format `np.datetime64` parses directly.

## What would fix it

Add a `ReadableType` for `datetime64[D]` (and one for a time-of-day precision)
that parses ISO 8601 with `np.datetime64` and refuses anything else, placed
after bool. Test with a column that mixes valid dates and empty cells.

## Files

- `tafra/csvreader.py`: `_TYPE_PRECEDENCE`, `ReadableType`
