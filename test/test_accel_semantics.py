"""
Semantics of the accelerated key-encoding, join and group-by paths.

Every test that takes the `accel_mode` fixture runs twice: once with the C
extension and once with the pure numpy fallback; `accel_only` tests call the
C entry points directly. Expected values are literals or come from numpy
itself applied group by group, never from tafra (geomean, harmean and
percentile are checked against their own per-group definitions).
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Iterator

import numpy as np
import pytest  # type: ignore

import tafra.group as group
from tafra import _kernels, _reducers
from tafra import Tafra
from tafra.group import geomean, harmean, percentile

# the nullable string dtype tafra uses for str columns
STRING_DTYPE = np.dtypes.StringDType(na_object=None)


@pytest.fixture(params=["accelerated", "pure"])
def accel_mode(request: Any, monkeypatch: Any) -> Iterator[str]:
    """Run the test with the C extension and with the numpy fallback."""
    use_accel = request.param == "accelerated"
    if use_accel and not _kernels.HAS_ACCEL:
        pytest.skip("C extension not available")
    # every reducer, encoder and join reads HAS_ACCEL when it is called
    monkeypatch.setattr(_kernels, "HAS_ACCEL", use_accel)
    yield request.param


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
    """Inner join with the empty-table warning silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return left.inner_join(right, on)


def _left_join(left: Tafra, right: Tafra, on: list[tuple[str, str, str]]) -> Tafra:
    """Left join with the null-cast and empty-table warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return left.left_join(right, on)


# ----------------------------------------------------------------------------
# Join keys
# ----------------------------------------------------------------------------


class TestJoinKeys:
    """Equi-join keys match exactly when numpy `==` holds, and NULL never matches."""

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
        """Three wide key columns, combined through the composite-key re-encode."""
        values = np.arange(3000, dtype=np.int64) * 7
        left = Tafra({"a": values, "b": values, "c": values, "v": np.arange(3000)})
        right = Tafra({"a": values[::-1], "b": values[::-1], "c": values[::-1]})
        on = [("a", "a", "=="), ("b", "b", "=="), ("c", "c", "==")]
        result = _inner_join(left, right, on)
        assert sorted(result["v"].tolist()) == list(range(3000))


# One key column per NULL kind: rows 0 and 2 hold a value both sides share,
# row 1 holds the kind's NULL on both sides, which must never match.
SHARED_NAN = float("nan")
SHARED_NAT = np.datetime64("NaT", "ns")
NULL_KEYS: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = {
    "string_none": (
        np.array(["a", None, "b"], dtype=STRING_DTYPE),
        np.array(["a", None, "b"], dtype=STRING_DTYPE),
    ),
    "object_none": (
        np.array(["a", None, "b"], dtype=object),
        np.array(["a", None, "b"], dtype=object),
    ),
    "object_shared_nan": (
        np.array([1.0, SHARED_NAN, 2.0], dtype=object),
        np.array([1.0, SHARED_NAN, 2.0], dtype=object),
    ),
    "object_nat": (
        np.array([1, SHARED_NAT, 2], dtype=object),
        np.array([1, SHARED_NAT, 2], dtype=object),
    ),
    "datetime_nat": (
        np.array(["2020-01-01", "NaT", "2020-01-02"], dtype="datetime64[D]"),
        np.array(["2020-01-01", "NaT", "2020-01-02"], dtype="datetime64[D]"),
    ),
    "timedelta_nat": (
        np.array([1, "NaT", 2], dtype="timedelta64[s]"),
        np.array([1, "NaT", 2], dtype="timedelta64[s]"),
    ),
}


@pytest.mark.parametrize("null_kind", list(NULL_KEYS))
def test_null_keys_never_match(accel_mode: str, null_kind: str) -> None:
    left_keys, right_keys = NULL_KEYS[null_kind]
    left = Tafra({"k": left_keys, "j": np.array([1, 1, 1]), "lv": np.array([0, 1, 2])})
    right = Tafra({"k": right_keys, "j": np.array([1, 1, 1]), "rv": np.array([10, 11, 12])})
    for on in ([("k", "k", "==")], [("k", "k", "=="), ("j", "j", "==")]):
        inner = _inner_join(left, right, on)
        assert sorted(zip(inner["lv"].tolist(), inner["rv"].tolist())) == [(0, 10), (2, 12)]
        joined = _left_join(left, right, on)
        assert joined["lv"].tolist() == [0, 1, 2]
        assert joined["rv"].tolist() == [10, None, 12]


def test_null_key_made_by_unit_promotion_never_matches(accel_mode: str) -> None:
    """2**54 s is NaT once promoted to ns, so it must not match a left NaT key."""
    left = Tafra(
        {"k": np.array(["NaT", "2020-01-01"], dtype="datetime64[ns]"), "a": np.array([1, 2])}
    )
    right = Tafra({"k": np.array([2**54], dtype="datetime64[s]"), "b": np.array([10.0])})
    on = [("k", "k", "==")]
    try:
        np.concatenate([left["k"], right["k"]])
    except OverflowError:
        # a numpy that checks the unit cast raises (2.2 and 2.4 wrap 2**54 s to NaT;
        # the 2.5 release notes describe the cast as overflow-checked), and so does the join
        with pytest.raises(OverflowError):
            _inner_join(left, right, on)
        with pytest.raises(OverflowError):
            _left_join(left, right, on)
        return
    assert len(_inner_join(left, right, on)) == 0
    assert np.isnan(_left_join(left, right, on)["b"]).all()


def test_nested_loop_judges_null_on_promoted_key(accel_mode: str) -> None:
    """A right key that is NaT only once promoted to ns is NULL on the `!=` path too."""
    left = Tafra({"k": np.array(["2020-01-01"], dtype="datetime64[ns]"), "a": np.array([1])})
    right = Tafra({"k": np.array([2**54], dtype="datetime64[s]"), "b": np.array([10.0])})
    on = [("k", "k", "!=")]
    try:
        np.concatenate([left["k"], right["k"]])
    except OverflowError:
        # a numpy that checks the unit cast raises before any NULL is judged
        with pytest.raises(OverflowError):
            _inner_join(left, right, on)
        with pytest.raises(OverflowError):
            _left_join(left, right, on)
        return
    assert len(_inner_join(left, right, on)) == 0
    assert np.isnan(_left_join(left, right, on)["b"]).all()
    # the same key on the left side is NULL too: the left row stays unmatched
    swapped = Tafra({"k": right["k"], "a": np.array([1])})
    other = Tafra({"k": left["k"], "b": np.array([10.0])})
    assert len(_inner_join(swapped, other, on)) == 0
    assert np.isnan(_left_join(swapped, other, on)["b"]).all()


def test_nested_loop_keys_need_no_common_dtype(accel_mode: str) -> None:
    """Two StringDType keys with different na_object values compare without a common dtype."""
    left = Tafra({"k": np.array(["a", "b"], dtype=STRING_DTYPE), "a": np.array([1, 2])})
    right = Tafra({"k": np.array(["a"], dtype=STRING_DTYPE), "b": np.array([10.0])})
    right._data["k"] = np.array(["a"], dtype=np.dtypes.StringDType(na_object=""))
    assert _inner_join(left, right, [("k", "k", "!=")])["a"].tolist() == [2]


# ----------------------------------------------------------------------------
# Group-by keys
# ----------------------------------------------------------------------------

# Tile each fixture once, and enough times to pass group._GROUP_INDICES_MIN_ROWS
# with the shortest (3-row) fixture, so the C group_indices path runs too.
REPEATS = (1, group._GROUP_INDICES_MIN_ROWS // 3 + 1)


class TestGroupByKeys:
    """group_by accepts every key dtype numpy compares with `==`."""

    @pytest.mark.parametrize("repeats", REPEATS)
    def test_float_key(self, accel_mode: str, repeats: int) -> None:
        keys = np.tile(np.array([1.5, 2.5, 1.5, np.nan, np.nan, -0.0, 0.0]), repeats)
        values = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), repeats)
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
                "b": np.tile(np.array([True, False, True]), repeats),
                "x": np.tile(np.array([1.0, 2.0, 3.0]), repeats),
            }
        )
        for aggregation in (np.sum, lambda array: array.sum()):
            result = table.group_by(["b"], {"x": aggregation})
            assert _rows(result) == [(True, 4.0 * repeats), (False, 2.0 * repeats)]

    @pytest.mark.parametrize("repeats", REPEATS)
    def test_day_datetime_key(self, accel_mode: str, repeats: int) -> None:
        days = np.array(["2020-01-02", "2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        table = Tafra(
            {"d": np.tile(days, repeats), "x": np.tile(np.array([1.0, 2.0, 3.0]), repeats)}
        )
        result = table.group_by(["d"], {"x": np.sum, "n": (lambda values: len(values), "x")})
        assert result["d"].tolist() == days[:2].tolist()
        assert result["x"].tolist() == [4.0 * repeats, 2.0 * repeats]
        assert result["n"].tolist() == [2 * repeats, repeats]

    def test_nat_forms_one_group(self, accel_mode: str) -> None:
        days = np.array(["NaT", "2020-01-01", "NaT"], dtype="datetime64[D]")
        table = Tafra({"d": days, "x": np.array([1.0, 2.0, 4.0])})
        result = table.group_by(["d"], {"x": np.sum})
        assert np.isnat(result["d"][0]) and result["x"].tolist() == [5.0, 2.0]

    @pytest.mark.parametrize("repeats", REPEATS)
    def test_large_integer_ids(self, accel_mode: str, repeats: int) -> None:
        ids = np.array([42_000_000_000_001, 42_000_000_000_000, 42_000_000_000_001])
        table = Tafra(
            {"api": np.tile(ids, repeats), "x": np.tile(np.array([1.0, 2.0, 3.0]), repeats)}
        )
        result = table.group_by(["api"], {"x": (lambda values: float(values.sum()), "x")})
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
        keys = np.tile(np.array(["a", None, "b", "a"], dtype=STRING_DTYPE), repeats)
        table = Tafra({"s": keys, "x": np.tile(np.array([1.0, 2.0, 3.0, 4.0]), repeats)})
        result = table.group_by(["s"], {"x": (lambda values: float(values.sum()), "x")})
        assert _rows(result) == [("a", 5.0 * repeats), (None, 2.0 * repeats), ("b", 3.0 * repeats)]


# ----------------------------------------------------------------------------
# Key column arguments
# ----------------------------------------------------------------------------


def _key_table() -> Tafra:
    """Three rows, two groups by k."""
    return Tafra({"k": np.array([1, 2, 1]), "v": np.array([1.0, 2.0, 3.0])})


def test_group_by_accepts_a_generator_of_key_columns(accel_mode: str) -> None:
    result = _key_table().group_by((column for column in ["k"]), {"v": np.sum})
    assert _rows(result) == [(1, 4.0), (2, 2.0)]


def test_transform_and_iterate_by_accept_a_generator(accel_mode: str) -> None:
    table = _key_table()
    transformed = table.transform((column for column in ["k"]), {"v": np.sum})
    assert transformed["v"].tolist() == [4.0, 2.0, 4.0]
    groups = list(table.iterate_by(column for column in ["k"]))
    assert [values for values, _, _ in groups] == [(1,), (2,)]


@pytest.mark.parametrize("operation", ["group_by", "transform", "iterate_by"])
def test_no_key_columns_raises(accel_mode: str, operation: str) -> None:
    table = _key_table()
    with pytest.raises(ValueError, match="key column"):
        if operation == "iterate_by":
            list(table.iterate_by([]))
        else:
            getattr(table, operation)([], {"v": np.sum})


# ----------------------------------------------------------------------------
# Reducers: compare with numpy applied group by group
# ----------------------------------------------------------------------------

# Three groups: a mixed group, an all-NaN group for floats, a plain group.
# DATA_BY_DTYPE lists each group's rows together; INTERLEAVE then mixes the
# groups (keeping each group's own row order), so a reducer that does not
# sort rows by label gives wrong groups.
GROUP_LABELS = np.array([0, 0, 0, 1, 1, 2, 2, 2])
INTERLEAVE = np.array([3, 0, 5, 1, 6, 2, 7, 4])

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

GROUP_LABELS = GROUP_LABELS[INTERLEAVE]
DATA_BY_DTYPE = {name: data[INTERLEAVE] for name, data in DATA_BY_DTYPE.items()}

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
    "np.amin": np.amin,
    "np.nanmin": np.nanmin,
    "np.max": np.max,
    "np.amax": np.amax,
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
            except (TypeError, ValueError):  # numpy refuses this dtype
                return None
    return np.array(results)


# Pairs numpy accepts that deliberately take the per-group path: builtin sum
# rounds differently from np.sum on narrow types; the rest have no kernel for
# datetime or bool.
PER_GROUP_BY_DESIGN: frozenset[tuple[str, str]] = frozenset(
    {
        ("np.any", "datetime64[D]"),
        ("np.all", "datetime64[D]"),
        ("np.count_nonzero", "datetime64[D]"),
        ("sum", "float32"),
        ("sum", "int32"),
        ("geomean", "bool"),
        ("geomean", "datetime64[D]"),
        ("harmean", "bool"),
        ("harmean", "datetime64[D]"),
        ("percentile(30)", "datetime64[D]"),
    }
)


@pytest.mark.parametrize("dtype_name", list(DATA_BY_DTYPE))
@pytest.mark.parametrize("reducer_name", list(REDUCERS))
def test_vectorized_reducer_matches_numpy(
    accel_mode: str, reducer_name: str, dtype_name: str
) -> None:
    function = REDUCERS[reducer_name]
    data = DATA_BY_DTYPE[dtype_name]
    expected = _numpy_per_group(function, data)

    vectorized = _reducers.vectorized_aggregation(function, data.dtype)
    if vectorized is None:
        # no fast path: only where numpy refuses, or for a listed pair
        assert expected is None or (reducer_name, dtype_name) in PER_GROUP_BY_DESIGN
        pytest.skip("no vectorized path for this dtype; the per-group path calls numpy")
    assert (reducer_name, dtype_name) not in PER_GROUP_BY_DESIGN
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


@pytest.mark.parametrize("dtype_name", [">u8", ">i8", ">u4", ">f8"])
@pytest.mark.parametrize("reducer", [np.min, np.max, np.sum])
def test_non_native_byte_order_reduces_like_numpy(
    accel_mode: str, dtype_name: str, reducer: Callable[..., Any]
) -> None:
    """A big-endian column reduces to numpy's values, not to its byte-swapped bits."""
    values = np.array([5, 3, 9, 2]).astype(dtype_name)
    table = Tafra({"g": np.array([0, 0, 1, 1]), "v": values})
    result = table.group_by(["g"], {"v": (reducer, "v")})
    assert result["v"].tolist() == [reducer(values[:2]), reducer(values[2:])]
    assert result["v"].dtype.isnative


def test_variance_matches_numpy_on_offset_values(accel_mode: str) -> None:
    """Grouped np.var is as accurate as numpy's two-pass result on offset data."""
    generator = np.random.default_rng(seed=1)
    labels = np.repeat(np.arange(300), 60)
    values = generator.normal(size=labels.size) + 1e9
    table = Tafra({"g": labels, "v": values})
    result = table.group_by(["g"], {"v": (np.var, "v")})
    expected = [np.var(values[labels == label]) for label in range(300)]
    np.testing.assert_allclose(result["v"], expected, rtol=1e-11)


def test_fast_path_refuses_dtypes_wider_than_64_bits() -> None:
    """The kernels compute in 64 bits, so a wider dtype takes the per-group path."""
    complex_kind = _reducers.kind_in(frozenset("c"))
    assert complex_kind(np.dtype(np.complex64))  # 8 bytes
    assert not complex_kind(np.dtype(np.complex128))  # 16 bytes


@pytest.mark.skipif(
    np.dtype(np.longdouble).itemsize <= 8, reason="longdouble is float64 on this platform"
)
def test_longdouble_sum_keeps_its_precision(accel_mode: str) -> None:
    values = np.array([1.0, 1e-18, 1e-18], dtype=np.longdouble)
    table = Tafra({"g": np.array([0, 0, 0]), "v": values})
    result = table.group_by(["g"], {"v": (np.sum, "v")})
    assert result["v"][0] == np.sum(values)


def test_transform_min_keeps_integer_dtype(accel_mode: str) -> None:
    table = Tafra({"g": np.array([0, 0, 1]), "x": np.array([5, 3, 9], dtype=np.int64)})
    result = table.transform(["g"], {"x": np.min})
    assert result["x"].dtype == np.int64
    assert result["x"].tolist() == [3, 3, 9]


# ----------------------------------------------------------------------------
# C entry points: hashing and input validation
# ----------------------------------------------------------------------------

accel_only = pytest.mark.skipif(not _kernels.HAS_ACCEL, reason="C extension not available")


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

    with pytest.raises(ValueError, match="same length"):
        _accel.groupby_sum(np.zeros(4, dtype=np.int64), np.ones(2), 1, False)


@accel_only
@pytest.mark.parametrize(
    "kernel_name",
    [
        "groupby_sum",
        "groupby_sum_int64",
        "groupby_mean",
        "groupby_var",
        "groupby_min",
        "groupby_max",
        "groupby_min_int64",
        "groupby_max_int64",
    ],
)
@pytest.mark.parametrize("bad_label", [2, 5, -1])
def test_groupby_kernels_reject_out_of_range_labels(kernel_name: str, bad_label: int) -> None:
    from tafra import _accel  # type: ignore[attr-defined]

    kernel = getattr(_accel, kernel_name)
    data = np.ones(2, dtype=np.int64 if kernel_name.endswith("_int64") else np.float64)
    with pytest.raises(ValueError, match="group label"):
        kernel(np.array([0, bad_label], dtype=np.int64), data, 2, False)


@accel_only
@pytest.mark.parametrize("entry_point", ["encode_strings", "encode_int64", "group_indices"])
def test_key_kernels_reject_multidimensional_input(entry_point: str) -> None:
    from tafra import _accel  # type: ignore[attr-defined]

    keys = np.zeros((2, 2), dtype=object if entry_point == "encode_strings" else np.int64)
    with pytest.raises(ValueError, match="one-dimensional"):
        getattr(_accel, entry_point)(keys)


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

    def test_transform_object_result_keeps_numpy_scalars(self, accel_mode: str) -> None:
        """An object column holds each group's result as group_by does, not cast to int."""
        stamp = np.datetime64("2020-01-01T00:00:00.000000001", "ns")
        table = Tafra({"g": np.array([0, 1, 0]), "x": np.array([1.0, 2.0, 3.0])})
        aggregation = {"y": (lambda values: stamp if values[0] == 1.0 else 1.0, "x")}
        transformed = table.transform(["g"], aggregation)
        grouped = table.group_by(["g"], aggregation)
        assert transformed["y"].dtype == object
        assert transformed["y"][0] == stamp and transformed["y"][2] == stamp
        assert grouped["y"][0] == transformed["y"][0]
        per_row = table.transform(
            ["g"], {"y": (lambda values: values.astype("M8[s]") if len(values) > 1 else 1.0, "x")}
        )
        assert per_row["y"][0] == np.datetime64(1, "s")


# ----------------------------------------------------------------------------
# Left join row order
# ----------------------------------------------------------------------------


def test_left_join_keeps_null_key_rows_in_left_order(accel_mode: str) -> None:
    left = Tafra({"k": np.array([np.nan, 1.0, np.nan, 2.0]), "a": np.array([0, 1, 2, 3])})
    right = Tafra({"k": np.array([2.0, 1.0, 2.0]), "b": np.array([20.0, 10.0, 30.0])})
    result = _left_join(left, right, [("k", "k", "==")])
    assert result["a"].tolist() == [0, 1, 2, 3, 3]
    np.testing.assert_array_equal(result["b"], [np.nan, 10.0, np.nan, 20.0, 30.0])
