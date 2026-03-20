# Chunking

Tafra provides methods to split data into pieces for batch processing or
parallel dispatch, and to reassemble the results.

## chunks(n)

Split into `n` roughly equal-sized pieces:

```python
import numpy as np
from tafra import Tafra

t = Tafra({
    'id': np.arange(10),
    'value': np.random.rand(10),
})

pieces = t.chunks(3)
print(len(pieces))
print(pieces[0].rows)
print(pieces[1].rows)
print(pieces[2].rows)
```

???+ example "Output"

    ```
    3
    4
    3
    3
    ```

If the row count does not divide evenly, earlier chunks get one extra row.

## chunk_rows(size)

Split into chunks of at most `size` rows each:

```python
pieces = t.chunk_rows(4)
print(len(pieces))
print(pieces[0].rows)
print(pieces[1].rows)
print(pieces[2].rows)
```

???+ example "Output"

    ```
    3
    4
    3
    3
    ```

## sort_by Parameter

Both `chunks` and `chunk_rows` accept an optional `sort_by` parameter to sort
the data before splitting. This ensures related rows end up in the same chunk:

```python
t = Tafra({
    'group': np.array(['B', 'A', 'B', 'A', 'B', 'A']),
    'value': np.array([1, 2, 3, 4, 5, 6]),
})

# Without sort_by: rows split in original order
pieces = t.chunks(2)

# With sort_by: sorted by 'group' first, then split
pieces = t.chunks(2, sort_by=['group'])
```

???+ example "Output"

    ```
    pieces[0]: group=['A','A','A'], value=[2,4,6]
    pieces[1]: group=['B','B','B'], value=[1,3,5]
    ```

## Tafra.concat()

Reassemble chunks (or any list of Tafras with matching columns) into a single
Tafra:

```python
combined = Tafra.concat(pieces)
print(combined.rows)
```

???+ example "Output"

    ```
    6
    ```

All Tafras must have the same column names. Dtypes are taken from the first
Tafra in the list.

## Parallel Processing Patterns

### With multiprocessing.Pool

```python
from multiprocessing import Pool

def process_chunk(chunk):
    # Perform expensive computation
    result = chunk.copy()
    result['result'] = chunk['value'] ** 2
    return result

t = Tafra({
    'id': np.arange(1000),
    'value': np.random.rand(1000),
})

chunks = t.chunks(4)

with Pool(4) as pool:
    results = pool.map(process_chunk, chunks)

combined = Tafra.concat(results)
print(combined.rows)
```

???+ example "Output"

    ```
    1000
    ```

### partition vs chunks

Use `partition` when you need to split by **group values** (each group stays
intact). Use `chunks` or `chunk_rows` when you need **evenly sized** pieces
regardless of group boundaries.

```python
# partition: split by group identity
parts = t.partition(['region'])
# Each sub-Tafra contains all rows for one region

# chunks: split into N equal pieces
pieces = t.chunks(4)
# Each piece has ~rows/4 rows, groups may be split across pieces
```

### Batch database inserts

```python
t = Tafra({
    'id': np.arange(10000),
    'value': np.random.rand(10000),
})

for chunk in t.chunk_rows(1000):
    records = list(chunk.to_records())
    cur.executemany('INSERT INTO table (id, value) VALUES (?, ?)', records)
```
