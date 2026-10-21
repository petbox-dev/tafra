# Open work

Work that is not done lives here: this index, plus one document per item in
`open/` holding that item's evidence. Closed items move to `closed/` and keep
their document.

Rules:

- Never leave a `# TODO:` in the code. Fix it now, or open a document here.
- Name the document for the problem, not the fix.
- A document starts with this header, nothing added:

  ```
  # <the problem in one line>

  Severity: high | medium | low
  Opened: YYYY-MM-DD
  ```

  Then the sections that earn their place: what happens now, evidence
  (observed values, verbatim errors), what was ruled out, blast radius, what is
  still unverified, what would fix or unblock it, files. Target 60 lines.
- Adding an item means adding its row to the open table below. `Opened` matches
  the document. Sort by severity, then oldest first.
- To close: move the document to `closed/`, add
  `Closed: YYYY-MM-DD -- settled by: <one line>` and
  `If this was wrong, you will see: <symptom>` at its top, and move its row to
  `## Done` (newest first). Close when no person has an action left.

| Item | Opened | Severity | Summary |
|---|---|---|---|
| [drop-duplicates-key-semantics](open/drop-duplicates-key-semantics.md) | 2026-10-19 | medium | `drop_duplicates` keeps duplicate NaN rows (two or more key columns); it and `value_counts` raise on mixed or `None` keys that `group_by` accepts |
| [non-equi-join-null-handling](open/non-equi-join-null-handling.md) | 2026-10-19 | medium | Non-equi joins turn a StringDType `None` into `'None'` and raise on a `None` key |
| [coalesce-result-dtype](open/coalesce-result-dtype.md) | 2026-10-19 | low | `coalesce` applies no dtype promotion when fills differ from the column |
| [composite-key-code-range-unchecked](open/composite-key-code-range-unchecked.md) | 2026-10-19 | low | C `composite_key` accepts out-of-range codes, which can merge groups (direct callers only) |
| [constructor-from-dataframe-disabled](open/constructor-from-dataframe-disabled.md) | 2026-10-19 | low | `Tafra(DataFrame)` support sits in a commented-out block with no recorded reason |
| [csv-reader-date-inference](open/csv-reader-date-inference.md) | 2026-10-19 | low | The CSV reader reads date columns as strings |
| [join-null-markers-not-covered](open/join-null-markers-not-covered.md) | 2026-10-19 | low | `pd.NA`, `pd.NaT` and complex NaN keys match in joins (documented) |
| [left-join-object-cast-warning](open/left-join-object-cast-warning.md) | 2026-10-19 | low | Equi left join warns "cast to object" for a column already object; the nested path does not |
| [partition-input-validation](open/partition-input-validation.md) | 2026-10-19 | low | `partition` skips `_validate` and exhausts a `sort_by` generator |
| [extension-detection-sites](open/extension-detection-sites.md) | 2026-10-21 | low | The C extension is detected at several sites, and a failed load loses its cause |
| [join-null-rule-outside-keys-module](open/join-null-rule-outside-keys-module.md) | 2026-10-21 | low | The join NULL-key rule lives in `group.py`, apart from the key-equality rule in `_keys.py`; `coalesce` has a third rule |
| [moved-code-style-debt](open/moved-code-style-debt.md) | 2026-10-21 | low | Moved key/reducer code keeps shorthand names and 7 `mypy --strict` errors |
| [numpy-datetime-cast-overflow-unverified](open/numpy-datetime-cast-overflow-unverified.md) | 2026-10-21 | low | The join behaviour on a numpy whose datetime unit casts raise is untested |
| [union-concat-string-null-markers](open/union-concat-string-null-markers.md) | 2026-10-21 | low | `union` and `concat` raise on StringDType columns whose `na_object` values differ; joins handle them |

## Done

| Item | Closed | Settled by |
|---|---|---|
