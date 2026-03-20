"""
Tafra vs pandas vs polars performance comparison.

Run with: python test/bench_vs_pandas_vs_polars.py
"""
import time
import math
import sys
import numpy as np
import pandas as pd

from tafra import Tafra

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    from tafra._accel import groupby_sum as _c_test
    HAS_ACCEL = True
except ImportError:
    HAS_ACCEL = False

import tafra.group as _grp


def median_of(fn, n=7):
    """Return median timing in ms."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[n // 2]


def _disable_accel():
    """Temporarily disable C acceleration."""
    saved = _grp._HAS_ACCEL
    _grp._HAS_ACCEL = False
    # re-register vectorized aggs with Python fallbacks
    _grp._VECTORIZED_AGGS.clear()
    _grp._register_vectorized()
    return saved


def _restore_accel(saved):
    """Restore C acceleration state."""
    _grp._HAS_ACCEL = saved
    _grp._VECTORIZED_AGGS.clear()
    _grp._register_vectorized()


def _tafra_time(fn):
    """Time fn with accel on and off, return (accel_ms, pure_ms)."""
    if HAS_ACCEL:
        t_accel = median_of(fn)
        saved = _disable_accel()
        t_pure = median_of(fn)
        _restore_accel(saved)
        return t_accel, t_pure
    else:
        t_pure = median_of(fn)
        return None, t_pure


def fmt(val, best=False):
    if val is None:
        return "n/a".rjust(9)
    s = f"{val:.2f}".rjust(9)
    if best:
        s = f"*{val:.2f}".rjust(9)
    return s


def print_row(label, t_accel, t_pure, t_pandas, t_polars):
    """Print one benchmark row, marking the best."""
    vals = {'accel': t_accel, 'pure': t_pure, 'pandas': t_pandas, 'polars': t_polars}
    real_vals = {k: v for k, v in vals.items() if v is not None}
    best_key = min(real_vals, key=real_vals.get) if real_vals else None

    print(f"  {label:<35s}"
          f" {fmt(t_accel, best_key == 'accel')}"
          f" {fmt(t_pure, best_key == 'pure')}"
          f" {fmt(t_pandas, best_key == 'pandas')}"
          f" {fmt(t_polars, best_key == 'polars')}")


def print_header():
    accel_label = "Tafra+C" if HAS_ACCEL else ""
    print(f"  {'Benchmark':<35s}"
          f" {accel_label:>9s}"
          f" {'Tafra':>9s}"
          f" {'pandas':>9s}"
          f" {'polars':>9s}")
    print(f"  {'-'*35} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")


# ============================================================
# Benchmarks
# ============================================================

def bench_construction():
    print("\n--- Construction (100k rows, 5 cols) ---")
    print_header()
    rng = np.random.default_rng(42)
    data = {
        'a': rng.standard_normal(100_000),
        'b': rng.standard_normal(100_000),
        'c': rng.integers(0, 100, size=100_000).astype('int64'),
        'd': rng.standard_normal(100_000),
        'e': rng.standard_normal(100_000),
    }

    # construction doesn't use accel, same for both
    t_tafra = median_of(lambda: Tafra(data.copy()), n=21)
    t_pandas = median_of(lambda: pd.DataFrame(data.copy()), n=21)
    t_polars = median_of(lambda: pl.DataFrame(data.copy()), n=21) if HAS_POLARS else None
    print_row("Tafra(data)", t_tafra, t_tafra, t_pandas, t_polars)


def bench_column_access():
    print("\n--- Column Access (100k rows, 10k iters) ---")
    print_header()
    rng = np.random.default_rng(42)
    n = 100_000
    tf = Tafra({'x': rng.standard_normal(n), 'y': rng.standard_normal(n)})
    df = pd.DataFrame(tf.data)
    n_iter = 10_000

    t_tafra = median_of(lambda: [tf['x'] for _ in range(n_iter)])
    t_pandas = median_of(lambda: [df['x'] for _ in range(n_iter)])
    t_polars = None
    if HAS_POLARS:
        plf = pl.DataFrame(tf.data)
        t_polars = median_of(lambda: [plf['x'] for _ in range(n_iter)])
    print_row("tf['x'] x 10k", t_tafra, t_tafra, t_pandas, t_polars)


def bench_row_mapping():
    print("\n--- Row Mapping (UDF per row: tuple_map / itertuples / map_elements) ---")
    print_header()
    rng = np.random.default_rng(42)

    def row_fn(a: float, b: float, c: float) -> float:
        return math.sqrt(a * a + b * b) + math.log1p(abs(c))

    for n_rows, label in [
        (10_000, "10k rows"),
        (100_000, "100k rows"),
        (1_000_000, "1M rows"),
    ]:
        tf = Tafra({
            'a': rng.standard_normal(n_rows),
            'b': rng.standard_normal(n_rows),
            'c': rng.standard_normal(n_rows),
        })
        df = pd.DataFrame(tf.data)

        def tuple_mapper(r: tuple) -> tuple:
            return (row_fn(r[0], r[1], r[2]),)

        n_rep = max(1, min(7, 50_000 // n_rows))
        t_tafra = median_of(lambda: list(tf.tuple_map(tuple_mapper, name=None)), n=n_rep)
        t_pandas = median_of(
            lambda: [row_fn(r.a, r.b, r.c) for r in df.itertuples()], n=n_rep)
        t_polars = None
        if HAS_POLARS:
            plf = pl.DataFrame(tf.data)
            t_polars = median_of(lambda: plf.with_columns(
                pl.struct(['a', 'b', 'c']).map_elements(
                    lambda s: row_fn(s['a'], s['b'], s['c']),
                    return_dtype=pl.Float64
                ).alias('result')
            ), n=n_rep)
        print_row(label, t_tafra, t_tafra, t_pandas, t_polars)


def bench_vectorized_expr():
    print("\n--- Vectorized Expression (numpy / pandas / polars native) ---")
    print_header()
    rng = np.random.default_rng(42)

    # sqrt(a^2 + b^2) + log1p(|c|)
    for n_rows, label in [
        (10_000, "10k rows"),
        (100_000, "100k rows"),
        (1_000_000, "1M rows"),
    ]:
        tf = Tafra({
            'a': rng.standard_normal(n_rows),
            'b': rng.standard_normal(n_rows),
            'c': rng.standard_normal(n_rows),
        })
        df = pd.DataFrame(tf.data)

        # tafra/numpy: direct array ops
        t_tafra = median_of(lambda: np.sqrt(tf['a']**2 + tf['b']**2) + np.log1p(np.abs(tf['c'])))
        # pandas: same numpy ops on .values, or pandas API
        t_pandas = median_of(lambda: np.sqrt(df['a']**2 + df['b']**2) + np.log1p(np.abs(df['c'])))
        t_polars = None
        if HAS_POLARS:
            plf = pl.DataFrame(tf.data)
            t_polars = median_of(lambda: plf.with_columns(
                ((pl.col('a')**2 + pl.col('b')**2).sqrt()
                 + (pl.col('c').abs() + 1).log()).alias('result')
            ))
        print_row(label, t_tafra, t_tafra, t_pandas, t_polars)


def bench_groupby():
    print("\n--- GroupBy (sum + mean) ---")
    print_header()
    rng = np.random.default_rng(42)

    for n_rows, n_groups, label in [
        (10_000, 50, "10k rows, 50 grp"),
        (10_000, 500, "10k rows, 500 grp"),
        (100_000, 100, "100k rows, 100 grp"),
        (100_000, 1000, "100k rows, 1k grp"),
        (1_000_000, 100, "1M rows, 100 grp"),
        (1_000_000, 10000, "1M rows, 10k grp"),
    ]:
        tf = Tafra({
            'group': rng.integers(0, n_groups, size=n_rows).astype('int64'),
            'value': rng.standard_normal(n_rows),
        })
        df = pd.DataFrame(tf.data)

        t_accel, t_pure = _tafra_time(lambda: tf.group_by(
            ['group'], {'mean_val': (np.mean, 'value'), 'sum_val': (np.sum, 'value')}))
        t_pandas = median_of(
            lambda: df.groupby('group')['value'].agg(['mean', 'sum']).reset_index())
        t_polars = None
        if HAS_POLARS:
            plf = pl.DataFrame(tf.data)
            t_polars = median_of(lambda: plf.group_by('group').agg(
                pl.col('value').mean().alias('mean_val'),
                pl.col('value').sum().alias('sum_val'),
            ))
        print_row(label, t_accel, t_pure, t_pandas, t_polars)

    # multi-column
    for n_rows, label in [(100_000, "100k, 2 col, ~300 grp"), (1_000_000, "1M, 2 col, ~300 grp")]:
        tf = Tafra({
            'g1': rng.integers(0, 100, size=n_rows).astype('int64'),
            'g2': rng.choice(['x', 'y', 'z'], size=n_rows),
            'value': rng.standard_normal(n_rows),
        })
        df = pd.DataFrame(tf.data)
        t_accel, t_pure = _tafra_time(lambda: tf.group_by(
            ['g1', 'g2'], {'mean_val': (np.mean, 'value')}))
        t_pandas = median_of(
            lambda: df.groupby(['g1', 'g2'])['value'].agg('mean').reset_index())
        t_polars = None
        if HAS_POLARS:
            plf = pl.DataFrame(tf.data)
            t_polars = median_of(lambda: plf.group_by('g1', 'g2').agg(
                pl.col('value').mean().alias('mean_val')))
        print_row(label, t_accel, t_pure, t_pandas, t_polars)


def bench_transform():
    print("\n--- Transform (mean) ---")
    print_header()
    rng = np.random.default_rng(42)

    for n_rows, n_groups, label in [
        (10_000, 50, "10k rows, 50 grp"),
        (100_000, 100, "100k rows, 100 grp"),
        (1_000_000, 1000, "1M rows, 1k grp"),
    ]:
        tf = Tafra({
            'group': rng.integers(0, n_groups, size=n_rows).astype('int64'),
            'value': rng.standard_normal(n_rows),
        })
        df = pd.DataFrame(tf.data)

        t_accel, t_pure = _tafra_time(lambda: tf.transform(
            ['group'], {'mean_val': (np.mean, 'value')}))
        t_pandas = median_of(
            lambda: df.assign(mean_val=df.groupby('group')['value'].transform('mean')))
        t_polars = None
        if HAS_POLARS:
            plf = pl.DataFrame(tf.data)
            t_polars = median_of(lambda: plf.with_columns(
                pl.col('value').mean().over('group').alias('mean_val')))
        print_row(label, t_accel, t_pure, t_pandas, t_polars)


def bench_joins():
    print("\n--- Inner Join (equi) ---")
    print_header()
    rng = np.random.default_rng(42)

    for n, label in [(1000, "1k x 1k"), (5000, "5k x 5k"), (10000, "10k x 10k"),
                     (50000, "50k x 50k")]:
        left_tf = Tafra({
            'key': rng.integers(0, 100, size=n).astype('int64'),
            'left_val': rng.standard_normal(n),
        })
        right_tf = Tafra({
            'key': rng.integers(0, 100, size=n).astype('int64'),
            'right_val': rng.standard_normal(n),
        })
        left_df = pd.DataFrame(left_tf.data)
        right_df = pd.DataFrame(right_tf.data)

        t_accel, t_pure = _tafra_time(lambda: left_tf.inner_join(
            right_tf, [('key', 'key', '==')]))
        t_pandas = median_of(lambda: pd.merge(
            left_df, right_df, on='key', how='inner'))
        t_polars = None
        if HAS_POLARS:
            left_pl = pl.DataFrame(left_tf.data)
            right_pl = pl.DataFrame(right_tf.data)
            t_polars = median_of(lambda: left_pl.join(right_pl, on='key', how='inner'))
        print_row(label, t_accel, t_pure, t_pandas, t_polars)

    print("\n--- Left Join (equi) ---")
    print_header()
    for n, label in [(1000, "1k x 1k"), (5000, "5k x 5k"), (50000, "50k x 50k")]:
        left_tf = Tafra({
            'key': rng.integers(0, 100, size=n).astype('int64'),
            'left_val': rng.standard_normal(n),
        })
        right_tf = Tafra({
            'key': rng.integers(0, 100, size=n).astype('int64'),
            'right_val': rng.standard_normal(n),
        })
        left_df = pd.DataFrame(left_tf.data)
        right_df = pd.DataFrame(right_tf.data)

        t_accel, t_pure = _tafra_time(lambda: left_tf.left_join(
            right_tf, [('key', 'key', '==')]))
        t_pandas = median_of(lambda: pd.merge(
            left_df, right_df, on='key', how='left'))
        t_polars = None
        if HAS_POLARS:
            left_pl = pl.DataFrame(left_tf.data)
            right_pl = pl.DataFrame(right_tf.data)
            t_polars = median_of(lambda: left_pl.join(right_pl, on='key', how='left'))
        print_row(label, t_accel, t_pure, t_pandas, t_polars)


def bench_numba():
    try:
        from numba import jit
    except ImportError:
        print("\n--- Numba (skipped, not installed) ---")
        return

    print("\n--- Numba JIT (array function) ---")
    print_header()
    rng = np.random.default_rng(42)

    @jit(nopython=True, fastmath=True)
    def numba_fn(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        out = np.empty(a.shape[0])
        for i in range(a.shape[0]):
            out[i] = math.sqrt(a[i] * a[i] + b[i] * b[i]) + math.log1p(abs(c[i]))
        return out

    # Warm up JIT
    _warmup = np.ones(10)
    numba_fn(_warmup, _warmup, _warmup)

    for n_rows, label in [
        (10_000, "10k rows"),
        (100_000, "100k rows"),
        (1_000_000, "1M rows"),
    ]:
        tf = Tafra({
            'a': rng.standard_normal(n_rows),
            'b': rng.standard_normal(n_rows),
            'c': rng.standard_normal(n_rows),
        })
        df = pd.DataFrame(tf.data)

        # tafra: direct ndarray access
        t_tafra = median_of(lambda: numba_fn(tf['a'], tf['b'], tf['c']))

        # pandas: .values to get ndarray
        t_pandas = median_of(lambda: numba_fn(df['a'].values, df['b'].values, df['c'].values))

        t_polars = None
        if HAS_POLARS:
            plf = pl.DataFrame(tf.data)
            t_polars = median_of(lambda: numba_fn(
                plf['a'].to_numpy(), plf['b'].to_numpy(), plf['c'].to_numpy()))

        print_row(label, t_tafra, t_tafra, t_pandas, t_polars)


if __name__ == '__main__':
    print("=" * 80)
    libs = f"pandas {pd.__version__}, numpy {np.__version__}"
    if HAS_POLARS:
        libs += f", polars {pl.__version__}"
    accel_status = "C extension ACTIVE" if HAS_ACCEL else "C extension NOT available"
    print(f"Tafra Performance Comparison ({accel_status})")
    print(libs)
    print(f"All times in ms. * = fastest in row.")
    print("=" * 80)

    bench_construction()
    bench_column_access()
    bench_row_mapping()
    bench_vectorized_expr()
    bench_numba()
    bench_groupby()
    bench_transform()
    bench_joins()

    print("\n" + "=" * 80)
    print("Done.")
