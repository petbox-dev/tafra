"""
Concurrent use of one shared Tafra from many threads.

On a free-threaded build (3.14t, 3.15t) this runs the C extension with no GIL.
The row count is above group._GROUP_INDICES_MIN_ROWS, so the per-group path
also runs the C group_indices kernel; the other C kernels run at any size.

The read-only tests require every thread to get the single-threaded result.
The writer tests let a thread rewrite key arrays during the work: the result
may be wrong or raise, but the process must not crash and every row index a
kernel returns must be inside its array.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import pytest
from tafra import Tafra
from tafra import _kernels
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


# twice the threshold, so the Tafra-level group_by also runs the C group_indices kernel
WRITER_KEY_COUNT = 2 * group._GROUP_INDICES_MIN_ROWS
# A join kernel that re-reads a key instead of its cached slot crashed in
# 5 of 5 runs at 5 rounds on a GIL build (measured 2026-10-19); 10 gives margin.
WRITER_ROUNDS = 10


class KeyWriter:
    """A thread that rewrites `keys` in place with random values until stopped."""

    def __init__(self, keys: np.ndarray, key_range: int) -> None:
        self.keys = keys
        self.key_range = key_range
        self.stop_requested = threading.Event()
        self.rewrite_count = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        generator = np.random.default_rng(seed=7)
        while not self.stop_requested.is_set():
            self.keys[:] = generator.integers(0, self.key_range, len(self.keys))
            self.rewrite_count += 1

    def __enter__(self) -> 'KeyWriter':
        self.thread.start()
        return self

    def __exit__(self, *exception_info: object) -> None:
        self.stop_requested.set()
        self.thread.join(timeout=60)


@pytest.mark.skipif(not _kernels.HAS_ACCEL, reason="C extension not available")
def test_c_kernels_survive_concurrent_key_writes() -> None:
    """Each C kernel reads a key once, so a writer cannot make it index out of range."""
    generator = np.random.default_rng(seed=3)
    key_range = WRITER_KEY_COUNT // 2  # about two rows per key: many matches
    left = generator.integers(0, key_range, WRITER_KEY_COUNT)
    right = generator.integers(0, key_range, WRITER_KEY_COUNT)
    with KeyWriter(right, key_range) as right_writer, KeyWriter(left, key_range) as left_writer:
        rewrites_at_start = right_writer.rewrite_count + left_writer.rewrite_count
        for _ in range(WRITER_ROUNDS):
            left_indices, right_indices = _kernels.inner_join(left, right)
            # about two rows per key: an empty result would make the range checks vacuous
            assert 0 < len(left_indices) == len(right_indices)
            assert ((left_indices >= 0) & (left_indices < len(left))).all()
            assert ((right_indices >= 0) & (right_indices < len(right))).all()

            left_indices, right_indices, _ = _kernels.left_join(left, right)
            assert ((left_indices >= 0) & (left_indices < len(left))).all()
            assert ((right_indices >= -1) & (right_indices < len(right))).all()

            first_seen, groups, group_count = _kernels.group_indices(right)
            assert len(groups) == group_count == len(first_seen)
            assert sum(len(rows) for rows in groups) == len(right)
            codes, unique_count = _kernels.encode_int64(right)
            assert ((codes >= 0) & (codes < unique_count)).all()
    # the writers kept rewriting while the kernels ran, not only before them
    assert right_writer.rewrite_count + left_writer.rewrite_count > rewrites_at_start


def test_tafra_operations_survive_concurrent_key_writes() -> None:
    """
    A writer on a key column can make results wrong or raise, but never crash:
    this asserts only that the process survives.
    The joins copy the key columns (np.concatenate) before a kernel runs, so the
    writer reaches the C kernels only through group_by; the kernel-level test
    covers the join kernels.
    """
    generator = np.random.default_rng(seed=5)
    key_range = WRITER_KEY_COUNT // 2
    table = Tafra({
        'k': generator.integers(0, key_range, WRITER_KEY_COUNT),
        'v': generator.normal(size=WRITER_KEY_COUNT),
    })
    lookup = Tafra({'k': np.arange(key_range), 'w': np.arange(key_range, dtype=np.float64)})
    with KeyWriter(table['k'], key_range) as writer:
        for _ in range(WRITER_ROUNDS // 4):
            for operation in (
                lambda: table.group_by(['k'], {'v': np.sum}),
                lambda: table.group_by(['k'], {'v': (lambda values: values.max(), 'v')}),
                lambda: table.inner_join(lookup, [('k', 'k', '==')]),
                lambda: table.left_join(lookup, [('k', 'k', '==')]),
            ):
                try:
                    operation()
                except (IndexError, ValueError):
                    pass  # a key changed mid-operation; the guide allows an exception
    assert writer.rewrite_count > 0
