# Thread Safety

Tafra supports the free-threaded (no-GIL) builds of Python 3.14 and 3.15
(`python3.14t`, `python3.15t`). The C extension declares that it runs
without the GIL, so importing tafra does not re-enable the GIL.

```python
import sys
import tafra

sys._is_gil_enabled()  # False on a free-threaded interpreter
```

## What is safe

Many threads can **read** one shared `Tafra` at the same time. A read is any
operation that returns a new `Tafra` or a new value and does not change the
input:

- `group_by`, `transform`, `iterate_by`
- `inner_join`, `left_join`, `cross_join`, `union`
- `partition`, `chunks`, `chunk_rows`, `select`, `head`, `sort`
- column access (`t['col']`), `to_records`, `to_list`, `to_pandas`

```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tafra import Tafra

wells = Tafra({
    'well': np.array(['A', 'A', 'B', 'B']),
    'oil': np.array([100.0, 90.0, 250.0, 240.0]),
})

def total_oil(_: int) -> Tafra:
    return wells.group_by(['well'], {'oil': (np.sum, 'oil')})

with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(total_oil, range(8)))
```

The test suite runs `group_by`, `transform` and the joins from several
threads on one shared `Tafra`, and compares each result to the
single-thread result.

## What is not safe

Do not **change** a `Tafra` while another thread uses it. These operations
change a `Tafra`:

- `t['col'] = values` (`__setitem__`)
- `update_inplace`, `update_dtypes_inplace`, `rename_inplace`,
  `delete_inplace`, `coalesce_inplace`, `parse_object_dtypes_inplace`
- a write into a column array: `t['col']` returns the stored `ndarray`, not a
  copy, so `t['col'][0] = 1.0` changes the shared `Tafra`

A concurrent change can give wrong or mixed results, or an exception. Tafra
does not lock a `Tafra`. This is the same rule that numpy applies to a
shared `ndarray`. The C extension reads each key value once, so a concurrent
change to a key column cannot corrupt memory; the result is still wrong.

To change data that other threads read, give each thread its own copy
(`t.copy()`), or protect the `Tafra` with a `threading.Lock`.

## Threads or processes

On a free-threaded interpreter, threads run Python code in parallel, and
they share memory with no serialization cost. On a standard (GIL)
interpreter, only one thread runs Python code at a time; use
[`partition`](groupby-partition.md) with a process pool for CPU-bound work
per group. The C extension releases the GIL inside its integer hashing, join
and aggregation loops, so on a standard interpreter those loops in different
threads also run in parallel. Encoding a string or `object` key column calls
Python's hash on each value, so that step holds the GIL.
