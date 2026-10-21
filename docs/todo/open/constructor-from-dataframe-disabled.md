# Tafra(DataFrame) construction is disabled in a commented-out block

Severity: low
Opened: 2026-10-19

## What happens now

`Tafra.__post_init__` (tafra/base.py) starts with a commented-out block that
would build a `Tafra` from a pandas `DataFrame` passed to the constructor, under
the comment `# TODO: enable this?` (migrated here). `Tafra.from_dataframe`
exists and is the supported route.

## Evidence

- The block and the comment date from commit ca5ad6b (2020-06-01, "additional
  constructor validation"); there is no recorded reason for disabling it.
- Run 2026-10-21: `Tafra(pandas.DataFrame({'a': [1, 2]}))` raises
  `TypeError: Iterable must contain Dict, Mapping, or Sequence, got <class 'str'>`.

## What is still unverified

- Whether the block was disabled for a reason (for example the
  `DataFrame` protocol check being too loose).

## What would unblock it

Either enable the block with a test or delete it.

## Files

- `tafra/base.py`: `Tafra.__post_init__`, `Tafra.from_dataframe`
