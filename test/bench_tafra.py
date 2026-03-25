"""
Benchmarks for tafra performance.

Run with:
    python test/bench_tafra.py
    pytest test/bench_tafra.py -v  (if pytest-benchmark is installed)
"""
from __future__ import annotations

import time
import numpy as np
from contextlib import contextmanager

from tafra import Tafra, GroupBy, Transform, IterateBy, InnerJoin, LeftJoin


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed * 1000:.2f} ms")


def make_tafra(n_rows: int, n_groups: int) -> Tafra:
    rng = np.random.default_rng(42)
    return Tafra({
        'group_a': rng.integers(0, n_groups, size=n_rows).astype('int64'),
        'group_b': rng.choice(['x', 'y', 'z'], size=n_rows),
        'value1': rng.standard_normal(n_rows),
        'value2': rng.standard_normal(n_rows),
        'value3': rng.integers(0, 1000, size=n_rows).astype('float64'),
    })


def make_join_tafras(left_rows: int, right_rows: int) -> tuple:
    rng = np.random.default_rng(42)
    left = Tafra({
        'key': rng.integers(0, 100, size=left_rows).astype('int64'),
        'left_val': rng.standard_normal(left_rows),
    })
    right = Tafra({
        'key': rng.integers(0, 100, size=right_rows).astype('int64'),
        'right_val': rng.standard_normal(right_rows),
    })
    return left, right


def bench_construction(n_rows: int, n_iter: int = 100) -> None:
    rng = np.random.default_rng(42)
    data = {
        'a': rng.standard_normal(n_rows),
        'b': rng.standard_normal(n_rows),
        'c': rng.integers(0, 100, size=n_rows).astype('int64'),
    }

    with timer(f"Tafra() validated, {n_rows} rows x {n_iter} iters"):
        for _ in range(n_iter):
            Tafra(data.copy())

    with timer(f"Tafra() unvalidated, {n_rows} rows x {n_iter} iters"):
        for _ in range(n_iter):
            Tafra({k: v for k, v in data.items()}, validate=False)


def bench_getitem(n_rows: int, n_iter: int = 10000) -> None:
    t = make_tafra(n_rows, 50)

    with timer(f"__getitem__ str, {n_iter} iters"):
        for _ in range(n_iter):
            t['value1']

    with timer(f"__getitem__ list[str], {n_iter} iters"):
        for _ in range(n_iter):
            t[['value1', 'value2']]

    mask = t['group_a'] < 25
    with timer(f"__getitem__ bool mask, {n_iter} iters"):
        for _ in range(n_iter):
            t[mask]


def bench_setitem(n_rows: int, n_iter: int = 1000) -> None:
    t = make_tafra(n_rows, 50)
    new_values = np.ones(n_rows)

    with timer(f"__setitem__ existing col, {n_iter} iters"):
        for _ in range(n_iter):
            t['value1'] = new_values

    with timer(f"__setitem__ new col, {n_iter} iters"):
        for i in range(n_iter):
            t[f'new_{i}'] = new_values


def bench_groupby(n_rows: int, n_groups: int) -> None:
    t = make_tafra(n_rows, n_groups)

    gb = GroupBy(
        group_by_cols=['group_a'],
        aggregation={'mean_v1': (np.mean, 'value1'), 'sum_v2': (np.sum, 'value2')},
        iter_fn={},
    )
    with timer(f"GroupBy 1 col, {n_groups} groups, {n_rows} rows"):
        gb.apply(t)

    gb2 = GroupBy(
        group_by_cols=['group_a', 'group_b'],
        aggregation={'mean_v1': (np.mean, 'value1')},
        iter_fn={},
    )
    unique_combos = len(set(zip(t['group_a'], t['group_b'])))
    with timer(f"GroupBy 2 cols, ~{unique_combos} groups, {n_rows} rows"):
        gb2.apply(t)


def bench_transform(n_rows: int, n_groups: int) -> None:
    t = make_tafra(n_rows, n_groups)

    tr = Transform(
        group_by_cols=['group_a'],
        aggregation={'mean_v1': (np.mean, 'value1'), 'sum_v2': (np.sum, 'value2')},
        iter_fn={},
    )
    with timer(f"Transform 1 col, {n_groups} groups, {n_rows} rows"):
        tr.apply(t)


def bench_iterateby(n_rows: int, n_groups: int) -> None:
    t = make_tafra(n_rows, n_groups)

    ib = IterateBy(group_by_cols=['group_a'])
    with timer(f"IterateBy 1 col, {n_groups} groups, {n_rows} rows"):
        list(ib.apply(t))


def bench_innerjoin(left_rows: int, right_rows: int) -> None:
    left, right = make_join_tafras(left_rows, right_rows)

    ij = InnerJoin(
        on=[('key', 'key', '==')],
        select=[],
    )
    with timer(f"InnerJoin {left_rows}x{right_rows}"):
        ij.apply(left, right)


def bench_leftjoin(left_rows: int, right_rows: int) -> None:
    left, right = make_join_tafras(left_rows, right_rows)

    lj = LeftJoin(
        on=[('key', 'key', '==')],
        select=[],
    )
    with timer(f"LeftJoin {left_rows}x{right_rows}"):
        lj.apply(left, right)


def bench_build_group_indices(n_rows: int, n_groups: int) -> None:
    t = make_tafra(n_rows, n_groups)

    with timer(f"_build_group_indices 1 col, {n_rows} rows"):
        GroupBy._build_group_indices(t, ['group_a'])

    with timer(f"_build_group_indices 2 cols, {n_rows} rows"):
        GroupBy._build_group_indices(t, ['group_a', 'group_b'])


def bench_validate_columns(n_rows: int, n_iter: int = 10000) -> None:
    t = make_tafra(n_rows, 50)
    cols = list(t._data.keys())

    with timer(f"_validate_columns {len(cols)} cols x {n_iter} iters"):
        for _ in range(n_iter):
            t._validate_columns(cols)


def bench_ensure_valid(n_rows: int, n_iter: int = 1000) -> None:
    t = make_tafra(n_rows, 50)
    arr = np.ones(n_rows)

    with timer(f"_ensure_valid ndarray, {n_iter} iters"):
        for _ in range(n_iter):
            t._ensure_valid('value1', arr)

    with timer(f"_ensure_valid scalar, {n_iter} iters"):
        for _ in range(n_iter):
            t._ensure_valid('value1', 1.0)


def bench_copy(n_rows: int, n_iter: int = 100) -> None:
    t = make_tafra(n_rows, 50)

    with timer(f"copy(), {n_rows} rows x 5 cols x {n_iter} iters"):
        for _ in range(n_iter):
            t.copy()


def bench_select(n_rows: int, n_iter: int = 1000) -> None:
    t = make_tafra(n_rows, 50)

    with timer(f"select() 2 cols, {n_iter} iters"):
        for _ in range(n_iter):
            t.select(['value1', 'value2'])


def bench_format_dtype(n_iter: int = 10000) -> None:
    dt = np.dtype('float64')

    with timer(f"_format_dtype, {n_iter} iters"):
        for _ in range(n_iter):
            Tafra._format_dtype(dt)


def bench_parse_dtype(n_iter: int = 10000) -> None:
    from tafra import object_formatter
    arr = np.ones(100)

    with timer(f"parse_dtype (non-object), {n_iter} iters"):
        for _ in range(n_iter):
            object_formatter.parse_dtype(arr)


def _heavy_worker(sub: 'Tafra') -> 'Tafra':
    """Simulate expensive per-group work (~20-50ms): model fit + forecast."""
    v = sub['value1']
    t = np.linspace(0, 1, 2000)
    result = np.zeros_like(t)
    for _ in range(100):
        result += np.sin(np.mean(v) * t) * np.exp(-np.std(v) * t)
        result += np.cos(np.var(v) * t) * np.log1p(np.abs(t * np.mean(v)))
    return Tafra({
        'group_a': sub['group_a'][:1],
        'forecast': np.array([np.sum(result)]),
    }, validate=False)


def _light_worker(sub: 'Tafra') -> 'Tafra':
    """Light per-group work: simple aggregations."""
    return Tafra({
        'group_a': sub['group_a'][:1],
        'mean_v1': np.array([np.mean(sub['value1'])]),
        'std_v1': np.array([np.std(sub['value1'])]),
    }, validate=False)


def _warmup(x: int) -> int:
    return x


def bench_partition_vs_groupby(n_rows: int, n_groups: int,
                               pool: 'ProcessPoolExecutor',
                               n_workers: int) -> None:
    t = make_tafra(n_rows, n_groups)

    print(f"  [{n_groups} groups, {n_rows} rows, {n_workers} workers]")

    with timer("  group_by (light agg)"):
        t.group_by(['group_a'], {
            'mean_v1': (np.mean, 'value1'),
            'std_v1': (np.std, 'value1'),
        })

    with timer("  partition+serial (light)"):
        parts = t.partition(['group_a'])
        results = [_light_worker(sub) for _, sub in parts]
        Tafra.concat(results)

    with timer("  partition+serial (heavy)"):
        parts = t.partition(['group_a'])
        results = [_heavy_worker(sub) for _, sub in parts]
        Tafra.concat(results)

    with timer(f"  partition+{n_workers} workers (heavy)"):
        parts = t.partition(['group_a'])
        results = list(pool.map(_heavy_worker, [sub for _, sub in parts]))
        Tafra.concat(results)


if __name__ == '__main__':
    print("=" * 60)
    print("Tafra Performance Benchmarks")
    print("=" * 60)

    print("\n--- Construction ---")
    bench_construction(1000)
    bench_construction(100_000)

    print("\n--- Access ---")
    bench_getitem(10_000)
    bench_setitem(10_000)
    bench_select(10_000)

    print("\n--- Internal hot paths ---")
    bench_validate_columns(10_000)
    bench_ensure_valid(10_000)
    bench_format_dtype()
    bench_parse_dtype()
    bench_copy(10_000)

    print("\n--- Unique groups ---")
    bench_build_group_indices(10_000, 100)
    bench_build_group_indices(100_000, 1000)

    print("\n--- GroupBy ---")
    bench_groupby(10_000, 50)
    bench_groupby(10_000, 500)
    bench_groupby(100_000, 100)

    print("\n--- Transform ---")
    bench_transform(10_000, 50)
    bench_transform(10_000, 500)
    bench_transform(100_000, 100)

    print("\n--- IterateBy ---")
    bench_iterateby(10_000, 50)
    bench_iterateby(100_000, 100)

    print("\n--- Partition vs GroupBy ---")
    from concurrent.futures import ProcessPoolExecutor
    import os
    n_workers = min(os.cpu_count() or 4, 8)
    pool = ProcessPoolExecutor(max_workers=n_workers)
    list(pool.map(_warmup, range(n_workers)))  # warm up workers
    bench_partition_vs_groupby(10_000, 50, pool, n_workers)
    bench_partition_vs_groupby(100_000, 100, pool, n_workers)
    bench_partition_vs_groupby(100_000, 1000, pool, n_workers)
    pool.shutdown(wait=True)

    print("\n--- Joins ---")
    bench_innerjoin(500, 500)
    bench_innerjoin(1000, 1000)
    bench_leftjoin(500, 500)
    bench_leftjoin(1000, 1000)

    print("\n--- Joins (larger) ---")
    bench_innerjoin(5000, 5000)
    bench_innerjoin(10000, 10000)
    bench_leftjoin(5000, 5000)
    bench_leftjoin(10000, 10000)

    print("\n" + "=" * 60)
    print("Done.")
