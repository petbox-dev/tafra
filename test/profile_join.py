"""
Profile the C-accelerated vs pure-Python join paths.

Usage:
  # Quick timing comparison (full Tafra API, matching bench_vs_pandas_vs_polars):
  python test/profile_join.py

  # Isolated index-only timing:
  python test/profile_join.py --index-only

  # Size sweep:
  python test/profile_join.py --sweep

  # Run one phase for perf record (source-level profiling):
  perf record -g --call-graph dwarf python test/profile_join.py --phase c-inner
  perf report
  perf annotate --source accel_inner_join
"""
from __future__ import annotations

import argparse
import time
import numpy as np

from tafra import Tafra
import tafra.group as _grp

from tafra._accel import inner_join as _c_inner_join, left_join as _c_left_join

N = 50_000
N_KEYS = 100
N_REPS = 7


def make_tafras():
    rng = np.random.default_rng(42)
    left = Tafra({
        'key': rng.integers(0, N_KEYS, size=N).astype('int64'),
        'left_val': rng.standard_normal(N),
    })
    right = Tafra({
        'key': rng.integers(0, N_KEYS, size=N).astype('int64'),
        'right_val': rng.standard_normal(N),
    })
    return left, right


def make_keys():
    rng = np.random.default_rng(42)
    left_key = rng.integers(0, N_KEYS, size=N).astype('int64')
    right_key = rng.integers(0, N_KEYS, size=N).astype('int64')
    return left_key, right_key


def _disable_accel():
    saved = _grp._HAS_ACCEL
    _grp._HAS_ACCEL = False
    _grp._VECTORIZED_AGGS.clear()
    _grp._register_vectorized()
    return saved


def _restore_accel(saved):
    _grp._HAS_ACCEL = saved
    _grp._VECTORIZED_AGGS.clear()
    _grp._register_vectorized()


def median_of(fn, n=N_REPS):
    times = []
    fn()  # warm up
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[n // 2]


def bench_phase(phase):
    """Run one phase in a tight loop for perf record."""
    left_tf, right_tf = make_tafras()
    left_key, right_key = make_keys()

    if phase == 'c-inner':
        _c_inner_join(left_key, right_key)
        for _ in range(N_REPS):
            _c_inner_join(left_key, right_key)

    elif phase == 'py-inner':
        _grp.Join._sort_merge_indices(left_key, right_key)
        for _ in range(N_REPS):
            _grp.Join._sort_merge_indices(left_key, right_key)

    elif phase == 'c-left':
        _c_left_join(left_key, right_key)
        for _ in range(N_REPS):
            _c_left_join(left_key, right_key)

    elif phase == 'py-left':
        _grp.Join._left_join_indices(left_key, right_key)
        for _ in range(N_REPS):
            _grp.Join._left_join_indices(left_key, right_key)

    elif phase == 'full-c-inner':
        left_tf.inner_join(right_tf, [('key', 'key', '==')])
        for _ in range(N_REPS):
            left_tf.inner_join(right_tf, [('key', 'key', '==')])

    elif phase == 'full-py-inner':
        saved = _disable_accel()
        left_tf.inner_join(right_tf, [('key', 'key', '==')])
        for _ in range(N_REPS):
            left_tf.inner_join(right_tf, [('key', 'key', '==')])
        _restore_accel(saved)

    elif phase == 'full-c-left':
        left_tf.left_join(right_tf, [('key', 'key', '==')])
        for _ in range(N_REPS):
            left_tf.left_join(right_tf, [('key', 'key', '==')])

    elif phase == 'full-py-left':
        saved = _disable_accel()
        left_tf.left_join(right_tf, [('key', 'key', '==')])
        for _ in range(N_REPS):
            left_tf.left_join(right_tf, [('key', 'key', '==')])
        _restore_accel(saved)


def timing_full():
    """Time the full Tafra join API (matching bench_vs_pandas_vs_polars)."""
    left_tf, right_tf = make_tafras()

    print(f"Full Tafra join API — {N:,} x {N:,}, {N_KEYS} keys\n")
    print(f"{'Join type':<20s} {'C accel (ms)':>12s} {'Python (ms)':>12s} {'Ratio':>8s}")
    print("-" * 56)

    for label, join_method in [
        ("inner join", "inner_join"),
        ("left join", "left_join"),
    ]:
        c_ms = median_of(lambda: getattr(left_tf, join_method)(
            right_tf, [('key', 'key', '==')]))

        saved = _disable_accel()
        py_ms = median_of(lambda: getattr(left_tf, join_method)(
            right_tf, [('key', 'key', '==')]))
        _restore_accel(saved)

        ratio = c_ms / py_ms
        print(f"{label:<20s} {c_ms:>12.2f} {py_ms:>12.2f} {ratio:>7.2f}x")


def timing_index_only():
    """Time just the index-building step (C vs sort-merge)."""
    left_key, right_key = make_keys()

    c_li, _ = _c_inner_join(left_key, right_key)
    print(f"Index-only — {N:,} x {N:,}, {N_KEYS} keys, output {len(c_li):,} rows\n")
    print(f"{'Join type':<20s} {'C accel (ms)':>12s} {'Python (ms)':>12s} {'Ratio':>8s}")
    print("-" * 56)

    for label, c_fn, py_fn in [
        ("inner join",
         lambda: _c_inner_join(left_key, right_key),
         lambda: _grp.Join._sort_merge_indices(left_key, right_key)),
        ("left join",
         lambda: _c_left_join(left_key, right_key),
         lambda: _grp.Join._left_join_indices(left_key, right_key)),
    ]:
        c_ms = median_of(c_fn)
        py_ms = median_of(py_fn)
        ratio = c_ms / py_ms
        print(f"{label:<20s} {c_ms:>12.2f} {py_ms:>12.2f} {ratio:>7.2f}x")


def sweep():
    """Run inner join at multiple sizes to find crossover."""
    print(f"\n{'N':<10s} {'Keys':>6s} {'Output':>14s}"
          f" {'C (ms)':>10s} {'Py (ms)':>10s} {'Ratio':>8s}")
    print("-" * 62)

    rng = np.random.default_rng(42)
    for n, n_keys in [
        (1_000, 100), (5_000, 100), (10_000, 100),
        (20_000, 100), (50_000, 100), (100_000, 100),
        (50_000, 1000), (50_000, 10000), (50_000, 49000),
    ]:
        left_key = rng.integers(0, n_keys, size=n).astype('int64')
        right_key = rng.integers(0, n_keys, size=n).astype('int64')

        c_li, _ = _c_inner_join(left_key, right_key)
        out_rows = len(c_li)

        c_ms = median_of(lambda: _c_inner_join(left_key, right_key))
        py_ms = median_of(lambda: _grp.Join._sort_merge_indices(left_key, right_key))
        ratio = c_ms / py_ms
        winner = "<" if ratio < 1 else ">"
        print(f"{n:<10d} {n_keys:>6d} {out_rows:>14,}"
              f" {c_ms:>10.2f} {py_ms:>10.2f} {ratio:>7.2f}x {winner}")


ALL_PHASES = [
    'c-inner', 'py-inner', 'c-left', 'py-left',
    'full-c-inner', 'full-py-inner', 'full-c-left', 'full-py-left',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default=None, choices=ALL_PHASES,
                        help='Run one phase in tight loop (for perf record)')
    parser.add_argument('--index-only', action='store_true',
                        help='Time just the index step (C fn vs sort-merge)')
    parser.add_argument('--sweep', action='store_true',
                        help='Size sweep (index-only)')
    args = parser.parse_args()

    if args.phase:
        bench_phase(args.phase)
    elif args.index_only:
        timing_index_only()
    elif args.sweep:
        sweep()
    else:
        timing_full()
