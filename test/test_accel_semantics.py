"""
Semantics of the accelerated key-encoding, join and group-by paths.

Every test runs twice: once with the C extension and once with the pure
numpy fallback. Expected values are literals or come from numpy itself
applied group by group, never from tafra.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Iterator

import numpy as np
import pytest  # type: ignore

import tafra.group as group
from tafra import Tafra
from tafra.group import geomean, harmean, percentile

STRING_DTYPE = np.dtypes.StringDType(na_object=None)


@pytest.fixture(params=["accelerated", "pure"])
def accel_mode(request: Any, monkeypatch: Any) -> Iterator[str]:
    """Run the test with the C extension and with the numpy fallback."""
    use_accel = request.param == "accelerated"
    if use_accel and not group._HAS_ACCEL:
        pytest.skip("C extension not available")
    saved_registry = dict(group._VECTORIZED_AGGS)
    monkeypatch.setattr(group, "_HAS_ACCEL", use_accel)
    group._VECTORIZED_AGGS.clear()
    group._register_vectorized()
    yield request.param
    group._VECTORIZED_AGGS.clear()
    group._VECTORIZED_AGGS.update(saved_registry)


def _rows(table: Tafra) -> list[tuple[Any, ...]]:
    """Rows of a Tafra as tuples, in table order. NaN becomes the string 'nan'."""

    def normalize(value: Any) -> Any:
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return "nan"
        if isinstance(value, np.generic):
            return value.item()
        return value

    columns = list(table.columns)
    return [
        tuple(normalize(table[column][i]) for column in columns) for i in range(table.rows)
    ]


def _inner_join(left: Tafra, right: Tafra, on: list[tuple[str, str, str]]) -> Tafra:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return left.inner_join(right, on)


def _left_join(left: Tafra, right: Tafra, on: list[tuple[str, str, str]]) -> Tafra:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return left.left_join(right, on)


# ----------------------------------------------------------------------------
# Join keys
# ----------------------------------------------------------------------------


class TestJoinKeys:
    def test_float_key_fractions_do_not_match(self, accel_mode: str) -> None:
        left = Tafra({"k": np.array([1.2, 2.9, 1.5]), "a": np.array([10, 20, 30])})
        right = Tafra({"k": np.array([1.7, 2.1, 1.5]), "b": np.array([100, 200, 300])})
        assert _rows(_inner_join(left, right, [("k", "k", "==")])) == [(1.5, 30, 300)]

    def test_left_join_float_key_fractions_are_unmatched(self, accel_mode: str) -> None:
        left = Tafra({"k": np.array([1.2, 1.5]), "a": np.array([10, 20])})
        right = Tafra({"k": np.array([1.7, 1.5]), "b": np.array([100.0, 200.0])})
        result = _left_join(left, right, [("k", "k", "==")])
        assert sorted(_rows(result), key=repr) == sorted(
            [(1.2, 10, "nan"), (1.5, 20, 200.0)], key=repr
        )

    def test_float_keys_beyond_int64_range(self, accel_mode: str) -> None:
        left = Tafra({"k": np.array([1e19, np.inf, 3.0]), "a": np.array([1, 2, 3])})
        right = Tafra({"k": np.array([-1e19, -np.inf, 3.0]), "b": np.array([4, 5, 6])})
        assert _rows(_inner_join(left, right, [("k", "k", "==")])) == [(3.0, 3, 6)]

    def test_negative_zero_matches_zero(self, accel_mode: str) -> None:
        left = Tafra({"k": np.array([-0.0]), "a": np.array([1])})
        right = Tafra({"k": np.array([0.0]), "b": np.array([2])})
        assert len(_inner_join(left, right, [("k", "k", "==")])) == 1

    def test_two_column_float_key(self, accel_mode: str) -> None:
        left = Tafra({"x": np.array([1.5]), "y": np.array([0.7]), "a": np.array([1])})
        right = Tafra(
            {"x": np.array([1.0, 1.5]), "y": np.array([0.0, 0.7]), "b": np.array([2, 3])}
        )
        on = [("x", "x", "=="), ("y", "y", "==")]
        assert _rows(_inner_join(left, right, on)) == [(1.5, 0.7, 1, 3)]

    def test_pre_epoch_datetime_in_two_column_key(self, accel_mode: str) -> None:
        left = Tafra(
            {
                "a": np.array([1]),
                "d": np.array([-1_000_000_000], dtype="datetime64[ns]"),
                "v": np.array([1]),
            }
        )
        right = Tafra(
            {
                "a": np.array([0, 0]),
                "d": np.array([0, 999_999_999], dtype="datetime64[ns]"),
                "w": np.array([2, 3]),
            }
        )
        assert len(_inner_join(left, right, [("a", "a", "=="), ("d", "d", "==")])) == 0

    def test_day_unit_datetime_in_two_column_key(self, accel_mode: str) -> None:
        days = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        left = Tafra({"a": np.array([1, 1]), "d": days, "v": np.array([10, 20])})
        right = Tafra({"a": np.array([1]), "d": days[1:], "w": np.array([30])})
        on = [("a", "a", "=="), ("d", "d", "==")]
        result = _inner_join(left, right, on)
        assert result["v"].tolist() == [20]
        assert result["w"].tolist() == [30]

    def test_uint64_above_int64_max_in_two_column_key(self, accel_mode: str) -> None:
        big = 2**63 + 1
        left = Tafra(
            {
                "u": np.array([big, 5], dtype=np.uint64),
                "j": np.array([0, 0]),
                "a": np.array([1, 2]),
            }
        )
        right = Tafra(
            {
                "u": np.array([big, 7], dtype=np.uint64),
                "j": np.array([0, 0]),
                "b": np.array([3, 4]),
            }
        )
        result = _inner_join(left, right, [("u", "u", "=="), ("j", "j", "==")])
        assert result["a"].tolist() == [1]
        assert result["b"].tolist() == [3]

    def test_three_wide_columns_join(self, accel_mode: str) -> None:
        """Cardinality product 3000**3 would need a 2.7e10-slot direct map."""
        values = np.arange(3000, dtype=np.int64) * 7
        left = Tafra({"a": values, "b": values, "c": values, "v": np.arange(3000)})
        right = Tafra({"a": values[::-1], "b": values[::-1], "c": values[::-1]})
        on = [("a", "a", "=="), ("b", "b", "=="), ("c", "c", "==")]
        result = _inner_join(left, right, on)
        assert sorted(result["v"].tolist()) == list(range(3000))


# ----------------------------------------------------------------------------
# Group-by keys
# ----------------------------------------------------------------------------

REPEATS = (1, 12_000)


def _tile(array: np.ndarray[Any, Any], repeats: int) -> np.ndarray[Any, Any]:
    return np.tile(array, repeats)


class TestGroupByKeys:
    @pytest.mark.parametrize("repeats", REPEATS)
    def test_float_key(self, accel_mode: str, repeats: int) -> None:
        keys = _tile(np.array([1.5, 2.5, 1.5, np.nan, np.nan, -0.0, 0.0]), repeats)
        values = _tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), repeats)
        table = Tafra({"k": keys, "x": values})
        for aggregation in (np.sum, lambda array: array.sum()):
            result = table.group_by(["k"], {"x": aggregation})
            assert _rows(result) == [
                (1.5, 4.0 * repeats),
                (2.5, 2.0 * repeats),
                ("nan", 9.0 * repeats),
                (-0.0, 13.0 * repeats),
            ]

    def test_nan_bit_patterns_form_one_group(self, accel_mode: str) -> None:
        """np.unique treats every NaN as one value; so does group_by."""
        negative_nan = np.copysign(np.nan, -1.0)
        table = Tafra({"k": np.array([np.nan, negative_nan, 1.0]), "x": np.array([1.0, 2.0, 4.0])})
        result = table.group_by(["k"], {"x": np.sum})
        assert _rows(result) == [("nan", 3.0), (1.0, 4.0)]

    @pytest.mark.parametrize("repeats", REPEATS)
    def test_bool_key(self, accel_mode: str, repeats: int) -> None:
        table = Tafra(
            {
                "b": _tile(np.array([True, False, True]), repeats),
                "x": _tile(np.array([1.0, 2.0, 3.0]), repeats),
            }
        )
        for aggregation in (np.sum, lambda array: array.sum()):
            result = table.group_by(["b"], {"x": aggregation})
            assert _rows(result) == [(True, 4.0 * repeats), (False, 2.0 * repeats)]

    @pytest.mark.parametrize("repeats", REPEATS)
    def test_day_datetime_key(self, accel_mode: str, repeats: int) -> None:
        days = np.array(["2020-01-02", "2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        table = Tafra({"d": _tile(days, repeats), "x": _tile(np.array([1.0, 2.0, 3.0]), repeats)})
        result = table.group_by(["d"], {"x": np.sum})
        assert result["d"].tolist() == days[:2].tolist()
        assert result["x"].tolist() == [4.0 * repeats, 2.0 * repeats]

    def test_nat_forms_one_group(self, accel_mode: str) -> None:
        days = np.array(["NaT", "2020-01-01", "NaT"], dtype="datetime64[D]")
        table = Tafra({"d": days, "x": np.array([1.0, 2.0, 4.0])})
        result = table.group_by(["d"], {"x": np.sum})
        assert np.isnat(result["d"][0]) and result["x"].tolist() == [5.0, 2.0]

    @pytest.mark.parametrize("repeats", REPEATS)
    def test_large_integer_ids(self, accel_mode: str, repeats: int) -> None:
        ids = np.array([42_000_000_000_001, 42_000_000_000_000, 42_000_000_000_001])
        table = Tafra({"api": _tile(ids, repeats), "x": _tile(np.array([1.0, 2.0, 3.0]), repeats)})
        result = table.group_by(["api"], {"x": np.sum})
        assert _rows(result) == [
            (42_000_000_000_001, 4.0 * repeats),
            (42_000_000_000_000, 2.0 * repeats),
        ]

    def test_two_wide_integer_columns(self, accel_mode: str) -> None:
        table = Tafra(
            {
                "a": np.array([10**10, 0, 10**10]),
                "b": np.array([3 * 10**10, 10**10, 3 * 10**10]),
                "x": np.array([1.0, 2.0, 3.0]),
            }
        )
        result = table.group_by(["a", "b"], {"x": np.sum})
        assert _rows(result) == [(10**10, 3 * 10**10, 4.0), (0, 10**10, 2.0)]

    def test_three_wide_columns_group_by(self, accel_mode: str) -> None:
        values = np.arange(3000, dtype=np.int64) * 7
        table = Tafra({"a": values, "b": values, "c": values, "x": np.ones(3000)})
        result = table.group_by(["a", "b", "c"], {"x": np.sum})
        assert len(result) == 3000
        assert result["a"].tolist() == values.tolist()

    def test_uint64_above_int64_max(self, accel_mode: str) -> None:
        keys = np.array([2**63 + 1, 1, 2**63 + 1], dtype=np.uint64)
        table = Tafra({"u": keys, "x": np.array([1.0, 2.0, 3.0])})
        result = table.group_by(["u"], {"x": np.sum})
        assert _rows(result) == [(2**63 + 1, 4.0), (1, 2.0)]

    @pytest.mark.parametrize("repeats", REPEATS)
    def test_string_key_with_none(self, accel_mode: str, repeats: int) -> None:
        keys = _tile(np.array(["a", None, "b", "a"], dtype=STRING_DTYPE), repeats)
        table = Tafra({"s": keys, "x": _tile(np.array([1.0, 2.0, 3.0, 4.0]), repeats)})
        result = table.group_by(["s"], {"x": np.sum})
        assert _rows(result) == [("a", 5.0 * repeats), (None, 2.0 * repeats), ("b", 3.0 * repeats)]


# ----------------------------------------------------------------------------
# Reducers: compare with numpy applied group by group
# ----------------------------------------------------------------------------

# Three groups: a mixed group, an all-NaN group for floats, a plain group.
GROUP_LABELS = np.array([0, 0, 0, 1, 1, 2, 2, 2])

DATA_BY_DTYPE: dict[str, np.ndarray[Any, Any]] = {
    "float64": np.array([1.0, np.nan, 4.0, np.nan, np.nan, 2.0, 3.0, 2.5]),
    "float32": np.array([1.0, 6.0, 4.0, 3.5, 9.0, 2.0, 3.0, 2.5], dtype=np.float32),
    "int64": np.array([2**53, 1, -7, 5, 3, 9, 7, 2**62], dtype=np.int64),
    "int32": np.array([5, 3, -7, 5, 3, 9, 7, 1], dtype=np.int32),
    "uint64": np.array([2**64 - 1, 3, 1, 5, 3, 9, 7, 2**63 + 5], dtype=np.uint64),
    "bool": np.array([True, False, True, False, False, True, True, True]),
    "datetime64[D]": np.array(
        [
            "2020-01-03",
            "NaT",
            "2020-01-01",
            "2020-02-01",
            "2020-03-01",
            "2021-01-01",
            "2021-01-02",
            "2021-01-03",
        ],
        dtype="datetime64[D]",
    ),
}

REDUCERS: dict[str, Callable[..., Any]] = {
    "np.sum": np.sum,
    "np.nansum": np.nansum,
    "np.mean": np.mean,
    "np.nanmean": np.nanmean,
    "np.std": np.std,
    "np.nanstd": np.nanstd,
    "np.var": np.var,
    "np.nanvar": np.nanvar,
    "np.min": np.min,
    "np.nanmin": np.nanmin,
    "np.max": np.max,
    "np.nanmax": np.nanmax,
    "np.ptp": np.ptp,
    "np.prod": np.prod,
    "np.nanprod": np.nanprod,
    "np.any": np.any,
    "np.all": np.all,
    "np.median": np.median,
    "np.nanmedian": np.nanmedian,
    "np.count_nonzero": np.count_nonzero,
    "len": len,
    "sum": sum,
    "geomean": geomean,
    "harmean": harmean,
    "percentile(30)": percentile(30),
}


def _numpy_per_group(
    function: Callable[..., Any], data: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any] | None:
    """numpy's answer for each group, or None if numpy refuses any group."""
    results = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for label in range(int(GROUP_LABELS.max()) + 1):
            try:
                results.append(function(data[GROUP_LABELS == label]))
            except Exception:  # noqa: BLE001
                return None
    return np.array(results)


@pytest.mark.parametrize("dtype_name", list(DATA_BY_DTYPE))
@pytest.mark.parametrize("reducer_name", list(REDUCERS))
def test_vectorized_reducer_matches_numpy(
    accel_mode: str, reducer_name: str, dtype_name: str
) -> None:
    function = REDUCERS[reducer_name]
    data = DATA_BY_DTYPE[dtype_name]
    expected = _numpy_per_group(function, data)

    vectorized = group._vectorized_aggregation(function, data.dtype)
    if vectorized is None:
        pytest.skip("no vectorized path for this dtype; the per-group path calls numpy")
    assert expected is not None, "a vectorized path exists where numpy refuses the input"

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        result = vectorized(data, GROUP_LABELS.astype(np.intp), int(GROUP_LABELS.max()) + 1)

    assert result.dtype == expected.dtype, (result.dtype, expected.dtype)
    if expected.dtype.kind == "f":
        np.testing.assert_allclose(result, expected, rtol=1e-6, equal_nan=True)
    else:
        np.testing.assert_array_equal(result, expected)


def test_nan_reducers_through_group_by(accel_mode: str) -> None:
    table = Tafra(
        {
            "g": np.array([0, 0, 1, 1, 2, 2]),
            "x": np.array([1.0, np.nan, np.nan, np.nan, 2.0, 3.0]),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = table.group_by(
            ["g"],
            {
                "minimum": (np.min, "x"),
                "nan_minimum": (np.nanmin, "x"),
                "nan_sum": (np.nansum, "x"),
                "nan_mean": (np.nanmean, "x"),
            },
        )
    np.testing.assert_array_equal(result["minimum"], [np.nan, np.nan, 2.0])
    np.testing.assert_array_equal(result["nan_minimum"], [1.0, np.nan, 2.0])
    np.testing.assert_array_equal(result["nan_sum"], [1.0, 0.0, 5.0])
    np.testing.assert_array_equal(result["nan_mean"], [1.0, np.nan, 2.5])


def test_integer_sum_is_exact_through_group_by(accel_mode: str) -> None:
    table = Tafra({"g": np.array([0, 0]), "x": np.array([2**53, 1], dtype=np.int64)})
    result = table.group_by(["g"], {"x": np.sum})
    assert result["x"].dtype == np.int64
    assert result["x"].tolist() == [2**53 + 1]


def test_variance_of_large_offset_values(accel_mode: str) -> None:
    table = Tafra(
        {
            "g": np.array([0, 0, 0, 0, 1, 1, 1]),
            "x": np.array([1e9, 1e9 + 1, 1e9 + 2, 1e9 + 3, 1e8 + 0.1, 1e8 + 0.1, 1e8 + 0.1]),
        }
    )
    result = table.group_by(["g"], {"x": np.var})
    np.testing.assert_allclose(result["x"], [1.25, 0.0], atol=1e-6)


def test_transform_min_keeps_integer_dtype(accel_mode: str) -> None:
    table = Tafra({"g": np.array([0, 0, 1]), "x": np.array([5, 3, 9], dtype=np.int64)})
    result = table.transform(["g"], {"x": np.min})
    assert result["x"].dtype == np.int64
    assert result["x"].tolist() == [3, 3, 9]


# ----------------------------------------------------------------------------
# C entry points: hashing and input validation
# ----------------------------------------------------------------------------

accel_only = pytest.mark.skipif(not group._HAS_ACCEL, reason="C extension not available")


@accel_only
@pytest.mark.parametrize("shift", [20, 32, 44])
def test_hash_join_keys_with_trailing_zero_bits(shift: int) -> None:
    """Keys that share low zero bits must not pile into one hash slot."""
    from tafra import _accel  # type: ignore[attr-defined]

    keys = np.arange(100_000, dtype=np.int64) << shift
    start = time.perf_counter()
    left_indices, right_indices = _accel.inner_join(keys, keys[::-1].copy())
    _accel.group_indices(keys)
    elapsed = time.perf_counter() - start
    assert len(left_indices) == 100_000
    assert elapsed < 1.0, f"hash clustering: {elapsed:.2f} s for 100k keys << {shift}"


@accel_only
def test_groupby_kernels_reject_mismatched_lengths() -> None:
    from tafra import _accel  # type: ignore[attr-defined]

    with pytest.raises(ValueError):
        _accel.groupby_sum(np.zeros(4, dtype=np.int64), np.ones(2), 1, False)


@accel_only
def test_groupby_kernels_reject_out_of_range_labels() -> None:
    from tafra import _accel  # type: ignore[attr-defined]

    with pytest.raises(ValueError):
        _accel.groupby_sum(np.array([0, 5], dtype=np.int64), np.ones(2), 2, False)
    with pytest.raises(ValueError):
        _accel.groupby_min(np.array([0, -1], dtype=np.int64), np.ones(2), 2, False)


@accel_only
def test_encode_int64_first_seen_codes() -> None:
    from tafra import _accel  # type: ignore[attr-defined]

    codes, unique_count = _accel.encode_int64(
        np.array([7 << 40, -3, 7 << 40, 0, -3], dtype=np.int64)
    )
    assert codes.tolist() == [0, 1, 0, 2, 1]
    assert unique_count == 3


@accel_only
def test_group_indices_empty_input() -> None:
    from tafra import _accel  # type: ignore[attr-defined]

    first_seen, groups, group_count = _accel.group_indices(np.array([], dtype=np.int64))
    assert first_seen.tolist() == [] and groups == [] and group_count == 0


# ----------------------------------------------------------------------------
# Per-group path: result dtype follows numpy's type promotion of the results
# ----------------------------------------------------------------------------


class TestPerGroupResultDtype:
    def test_group_by_mean_of_int_column_is_float(self, accel_mode: str) -> None:
        table = Tafra({"g": np.array([0, 0, 1]), "i": np.array([1, 2, 5])})
        result = table.group_by(["g"], {"m": (lambda values: values.mean(), "i")})
        assert result["m"].dtype == np.float64
        assert result["m"].tolist() == [1.5, 5.0]

    def test_group_by_any_of_string_column_is_bool(self, accel_mode: str) -> None:
        table = Tafra({"g": np.array([0, 0, 1]), "s": np.array(["a", "", ""], dtype=STRING_DTYPE)})
        result = table.group_by(["g"], {"a": (np.any, "s")})
        assert result["a"].dtype == np.bool_
        assert result["a"].tolist() == [True, False]

    def test_group_by_count_of_float_column_is_int(self, accel_mode: str) -> None:
        table = Tafra({"g": np.array([0, 0, 1]), "x": np.array([1.5, 2.5, 3.5])})
        result = table.group_by(["g"], {"n": (lambda values: int(len(values)), "x")})
        assert result["n"].dtype == np.int64
        assert result["n"].tolist() == [2, 1]

    def test_group_by_string_result_is_not_truncated(self, accel_mode: str) -> None:
        table = Tafra({"g": np.array([0, 0, 1]), "s": np.array(["a", "b", "c"])})
        result = table.group_by(["g"], {"j": (lambda values: "".join(values), "s")})
        assert result["j"].tolist() == ["ab", "c"]

    def test_group_by_none_and_float_results_give_object(self, accel_mode: str) -> None:
        table = Tafra({"g": np.array([0, 1]), "x": np.array([1.0, 2.0])})
        result = table.group_by(
            ["g"], {"y": (lambda values: None if values[0] < 1.5 else float(values[0]), "x")}
        )
        assert result["y"].dtype == object
        assert result["y"].tolist() == [None, 2.0]

    def test_group_by_string_and_float_results_follow_numpy(self, accel_mode: str) -> None:
        """numpy promotes str with float to str: np.array(['low', 2.0]) is <U32."""
        table = Tafra({"g": np.array([0, 1]), "x": np.array([1.0, 2.0])})
        result = table.group_by(
            ["g"], {"y": (lambda values: "low" if values[0] < 1.5 else float(values[0]), "x")}
        )
        assert result["y"].dtype == np.array(["low", 2.0]).dtype
        assert result["y"].tolist() == ["low", "2.0"]

    def test_transform_mean_of_int_column_is_float(self, accel_mode: str) -> None:
        table = Tafra({"g": np.array([0, 0, 1]), "i": np.array([1, 2, 5])})
        result = table.transform(["g"], {"m": (lambda values: values.mean(), "i")})
        assert result["m"].dtype == np.float64
        assert result["m"].tolist() == [1.5, 1.5, 5.0]

    def test_transform_per_row_result_of_int_column_is_float(self, accel_mode: str) -> None:
        table = Tafra({"g": np.array([0, 1, 0]), "i": np.array([1, 5, 2])})
        result = table.transform(["g"], {"d": (lambda values: values - values.mean(), "i")})
        assert result["d"].dtype == np.float64
        assert result["d"].tolist() == [-0.5, 0.0, 0.5]

    def test_transform_none_result_gives_object(self, accel_mode: str) -> None:
        table = Tafra({"g": np.array([0, 1, 0]), "x": np.array([1.0, 2.0, 3.0])})
        result = table.transform(
            ["g"], {"y": (lambda values: None if len(values) == 1 else values.sum(), "x")}
        )
        assert result["y"].dtype == object
        assert result["y"].tolist() == [4.0, None, 4.0]


# ----------------------------------------------------------------------------
# Left join row order
# ----------------------------------------------------------------------------


def test_left_join_keeps_null_key_rows_in_left_order(accel_mode: str) -> None:
    left = Tafra({"k": np.array([np.nan, 1.0, np.nan, 2.0]), "a": np.array([0, 1, 2, 3])})
    right = Tafra({"k": np.array([2.0, 1.0, 2.0]), "b": np.array([20.0, 10.0, 30.0])})
    result = _left_join(left, right, [("k", "k", "==")])
    assert result["a"].tolist() == [0, 1, 2, 3, 3]
    np.testing.assert_array_equal(result["b"], [np.nan, 10.0, np.nan, 20.0, 30.0])
