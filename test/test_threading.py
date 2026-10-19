"""
Concurrent read-only use of one shared Tafra from many threads.

On a free-threaded build (3.14t, 3.15t) this runs the C extension with no GIL.
The row count is above group._GROUP_INDICES_MIN_ROWS, so the per-group path
also runs the C group_indices kernel; the other C kernels run at any size.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
from tafra import Tafra
from tafra import group

ROW_COUNT = group._GROUP_INDICES_MIN_ROWS + 10_000
THREAD_COUNT = 8
ITERATIONS_PER_THREAD = 4


def build_shared_tafras() -> tuple[Tafra, Tafra]:
    """A left table with string, int and float columns, and a right lookup table."""
    generator = np.random.default_rng(seed=20261019)
    left = Tafra({
        'label': np.array([f'key_{value}' for value in generator.integers(0, 500, ROW_COUNT)]),
        'group': generator.integers(0, 40, ROW_COUNT),
        'value': generator.normal(size=ROW_COUNT),
        # 14-digit IDs: a wide range, so the C int64 encoder runs
        'well_id': 42_000_000_000_000 + generator.integers(0, 300, ROW_COUNT) * 1_000_003,
        'count': generator.integers(-1000, 1000, ROW_COUNT),
    })
    right = Tafra({
        'label': np.array([f'key_{value}' for value in range(0, 500, 2)]),
        'weight': np.arange(250, dtype=np.float64),
    })
    return left, right


def run_concurrently(operation: Callable[[], Tafra]) -> None:
    """Run `operation` from THREAD_COUNT threads at once; every result must equal one run alone."""
    expected = operation()
    start_together = threading.Barrier(THREAD_COUNT)

    def repeat_operation() -> list[Tafra]:
        start_together.wait(timeout=60)
        return [operation() for _ in range(ITERATIONS_PER_THREAD)]

    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        futures = [executor.submit(repeat_operation) for _ in range(THREAD_COUNT)]
        results = [result for future in futures for result in future.result()]

    for result in results:
        assert result.keys() == expected.keys()
        for column in expected.keys():
            np.testing.assert_array_equal(result[column], expected[column])


def test_concurrent_group_by() -> None:
    left, _ = build_shared_tafras()
    run_concurrently(lambda: left.group_by(
        ['label', 'group'],
        {'total': (np.sum, 'value'), 'mean': (np.mean, 'value'), 'variance': (np.var, 'value'),
         'low': (np.min, 'value'), 'high': (np.max, 'value'), 'count': (len, 'value')},
    ))


def test_concurrent_group_by_custom_function() -> None:
    left, _ = build_shared_tafras()
    run_concurrently(lambda: left.group_by(
        ['label'], {'spread': (lambda values: float(values.max() - values.min()), 'value')},
    ))


def test_concurrent_group_by_integer_key() -> None:
    left, _ = build_shared_tafras()
    run_concurrently(lambda: left.group_by(
        ['well_id'],
        {'total': (np.sum, 'count'), 'low': (np.min, 'count'), 'high': (np.max, 'count')},
    ))


def test_concurrent_transform_custom_function() -> None:
    left, _ = build_shared_tafras()
    run_concurrently(lambda: left.transform(
        ['well_id'], {'spread': (lambda values: float(values.max() - values.min()), 'value')},
    ))


def test_concurrent_transform() -> None:
    left, _ = build_shared_tafras()
    run_concurrently(lambda: left.transform(
        ['label', 'group'], {'mean': (np.mean, 'value'), 'variance': (np.var, 'value')},
    ))


def test_concurrent_inner_join() -> None:
    left, right = build_shared_tafras()
    run_concurrently(lambda: left.inner_join(right, [('label', 'label', '==')]))


def test_concurrent_left_join() -> None:
    left, right = build_shared_tafras()
    run_concurrently(lambda: left.left_join(right, [('label', 'label', '==')]))
