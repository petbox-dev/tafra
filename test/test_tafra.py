from __future__ import annotations

from pathlib import Path
import platform
import warnings
from decimal import Decimal
from datetime import date, datetime

import numpy as np
from tafra import Tafra, object_formatter
import pandas as pd  # type: ignore
from itertools import islice

from typing import Any, Iterator

import pytest  # type: ignore
from unittest.mock import MagicMock


class TestClass:
    ...


class Series:
    name: str = 'x'
    values: np.ndarray = np.arange(5)
    dtype: str = 'int'


class DataFrame:
    _data: dict[str, Series] = {'x': Series(), 'y': Series()}
    columns: list[str] = ['x', 'y']
    dtypes: list[str] = ['int', 'int']

    def __getitem__(self, column: str) -> Series:
        return self._data[column]

    def __setitem__(self, column: str, value: np.ndarray) -> None:
        self._data[column].values = value


class Cursor:
    description = (
        ('Fruit', str, None, 1, 1, 1, True),
        ('Amount', int, None, 1, 1, 1, True),
        ('Price', float, None, 1, 1, 1, True)
    )
    _iter = [
        ('Apples', 5, .95),
        ('Pears', 2, .80)
    ]
    idx = 0

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        return self

    def __next__(self) -> tuple[Any, ...]:
        try:
            item = self._iter[self.idx]
        except IndexError:
            raise StopIteration()
        self.idx += 1
        return item

    def execute(self, sql: str) -> None:
        ...

    def fetchone(self) -> tuple[Any, ...] | None:
        try:
            return next(self)
        except StopIteration:
            return None

    def fetchmany(self, size: int) -> list[tuple[Any, ...]]:
        return list(islice(self, size))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return [rec for rec in self]


def build_tafra() -> Tafra:
    return Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })


def check_tafra(t: Tafra, check_rows: bool = True) -> bool:
    assert len(t._data) == len(t._dtypes)
    for c in t.columns:
        assert isinstance(t[c], np.ndarray)
        assert isinstance(t.data[c], np.ndarray)
        assert isinstance(t._data[c], np.ndarray)
        assert isinstance(t.dtypes[c], str)
        assert isinstance(t._dtypes[c], str)
        if check_rows:
            assert t._rows == len(t._data[c])
        pd.Series(t._data[c])

    columns = [c for c in t.columns][:-1]

    _ = t.to_records()
    _ = t.to_records(columns=columns)
    _ = t.to_list()
    _ = t.to_list(columns=columns)
    _ = t.to_list(inner=True)
    _ = t.to_list(columns=columns, inner=True)
    _ = t.to_tuple()
    _ = t.to_tuple(columns=columns)
    _ = t.to_tuple(name=None)
    _ = t.to_tuple(name='tf')
    _ = t.to_tuple(columns=columns, name=None)
    _ = t.to_tuple(columns=columns, name='tf')
    _ = t.to_tuple(inner=True)
    _ = t.to_tuple(inner=True, name=None)
    _ = t.to_tuple(inner=True, name='tf')
    _ = t.to_tuple(columns=columns, inner=True)
    _ = t.to_tuple(columns=columns, inner=True, name=None)
    _ = t.to_tuple(columns=columns, inner=True, name='tf')
    _ = t.to_array()
    _ = t.to_array(columns=columns)
    df = t.to_pandas()
    df = t.to_pandas(columns=columns)
    assert isinstance(df, pd.DataFrame)
    write_path = Path('test/test_to_csv.csv')
    t.to_csv(write_path)
    # t.to_csv(write_path, columns=columns)

    return True

def test_constructions() -> None:
    t = build_tafra()
    check_tafra(t)

    t = Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'z': np.array([0, 0, 0, 1, 1, 1])
    }, validate=False)
    check_tafra(t)

    t = Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'z': np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    }, validate=False, check_rows=False)
    check_tafra(t, check_rows=False)

    with pytest.raises(TypeError) as e:
        t = Tafra()  # type: ignore # noqa

    with pytest.raises(ValueError) as e:
        t = Tafra({})

    t = Tafra({'x': None})
    with warnings.catch_warnings(record=True) as w:
        check_tafra(t)

    t = Tafra({'x': Decimal('1.23456')})
    check_tafra(t)

    t = Tafra({'x': np.array(1)})
    check_tafra(t)

    t = Tafra({'x': np.array([1])})
    check_tafra(t)

    t = Tafra({'x': [True, False]})
    check_tafra(t)

    t = Tafra({'x': 'test'})
    check_tafra(t)

    t = Tafra((('x', np.arange(6)),))
    check_tafra(t)

    t = Tafra([('x', np.arange(6))])
    check_tafra(t)

    t = Tafra([['x', np.arange(6)]])
    check_tafra(t)

    t = Tafra([(np.array('x'), np.arange(6))])
    check_tafra(t)

    t = Tafra([(np.array(['x']), np.arange(6))])
    check_tafra(t)

    t = Tafra([('x', np.arange(6)), ('y', np.linspace(0, 1, 6))])
    check_tafra(t)

    t = Tafra([['x', np.arange(6)], ('y', np.linspace(0, 1, 6))])
    check_tafra(t)

    t = Tafra([('x', np.arange(6)), ['y', np.linspace(0, 1, 6)]])
    check_tafra(t)

    t = Tafra([['x', np.arange(6)], ['y', np.linspace(0, 1, 6)]])
    check_tafra(t)

    t = Tafra([{'x': np.arange(6)}, {'y': np.linspace(0, 1, 6)}])
    check_tafra(t)

    t = Tafra(iter([{'x': np.arange(6)}, {'y': np.linspace(0, 1, 6)}]))
    check_tafra(t)

    def iterator() -> Iterator[dict[str, np.ndarray]]:
        yield {'x': np.array([1, 2, 3, 4, 5, 6])}
        yield {'y': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        )}
        yield {'z': np.array([0, 0, 0, 1, 1, 1])}

    t = Tafra(iterator())
    check_tafra(t)

    class DictIterable:
        def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
            yield {'x': np.array([1, 2, 3, 4, 5, 6])}
            yield {'y': np.array(
                ['one', 'two', 'one', 'two', 'one', 'two'],
                dtype=np.dtypes.StringDType(),
            )}
            yield {'z': np.array([0, 0, 0, 1, 1, 1])}

    t = Tafra(DictIterable())
    check_tafra(t)

    t = Tafra(iter(DictIterable()))
    check_tafra(t)

    class SequenceIterable:
        def __iter__(self) -> Iterator[Any]:
            yield ('x', np.array([1, 2, 3, 4, 5, 6]))
            yield ['y', np.array(
                ['one', 'two', 'one', 'two', 'one', 'two'],
                dtype=np.dtypes.StringDType(),
            )]
            yield ('z', np.array([0, 0, 0, 1, 1, 1]))

    t = Tafra(SequenceIterable())
    check_tafra(t)

    class SequenceIterable2:
        def __iter__(self) -> Iterator[Any]:
            yield (np.array(['x']), np.array([1, 2, 3, 4, 5, 6]))
            yield [np.array(['y']), np.array(['one', 'two', 'one', 'two', 'one', 'two'],
                                             dtype=np.dtypes.StringDType())]
            yield (np.array(['z']), np.array([0, 0, 0, 1, 1, 1]))

    t = Tafra(SequenceIterable2())
    check_tafra(t)

    t = Tafra(iter(SequenceIterable2()))
    check_tafra(t)

    t = Tafra(enumerate(np.arange(6)))
    check_tafra(t)

    t = build_tafra()
    df = pd.DataFrame(t.data)
    _ = Tafra.from_series(df['x'])
    check_tafra(_)

    _ = Tafra.from_dataframe(df)
    check_tafra(_)

    _ = Tafra.as_tafra(df)
    check_tafra(_)

    _ = Tafra.as_tafra(df['x'])
    check_tafra(_)

    _ = Tafra.as_tafra(t)
    check_tafra(_)

    _ = Tafra.as_tafra({'x': np.array(1)})
    check_tafra(_)

    _ = Tafra.from_series(Series())
    check_tafra(_)

    _ = Tafra.as_tafra(Series())
    check_tafra(_)

    _ = Tafra.from_dataframe(DataFrame())  # type: ignore
    check_tafra(_)

    _ = Tafra.as_tafra(DataFrame())
    check_tafra(_)

    with pytest.raises(TypeError) as e:
        t = Tafra([{1, 2}])  # type: ignore

    class BadIterable:
        def __iter__(self) -> Iterator[Any]:
            yield {1, 2}
            yield {3.1412159, .5772156}

    with pytest.raises(TypeError) as e:
        t = Tafra(BadIterable())

    with pytest.raises(TypeError) as e:
        t = Tafra(iter(BadIterable()))

    with pytest.raises(TypeError) as e:
        _ = Tafra(np.arange(6))

    with pytest.raises(TypeError) as e:
        _ = Tafra.as_tafra(np.arange(6))

    with pytest.raises(ValueError) as e:
        t = Tafra({'x': np.array([1, 2]), 'y': np.array([3., 4., 5.])})

def test_read_sql() -> None:

    cur = Cursor()
    columns, dtypes = zip(*((d[0], d[1]) for d in cur.description))
    records = cur.fetchall()
    t = Tafra.from_records(records, columns)
    check_tafra(t)

    t = Tafra.from_records(records, columns, dtypes)
    check_tafra(t)

    cur = Cursor()
    t = Tafra.read_sql('SELECT * FROM [Table]', cur)  # type: ignore
    check_tafra(t)

    cur = Cursor()
    cur._iter = []
    t = Tafra.read_sql('SELECT * FROM [Table]', cur)  # type: ignore
    check_tafra(t)

    cur = Cursor()
    for t in Tafra.read_sql_chunks('SELECT * FROM [Table]', cur):  # type: ignore
        check_tafra(t)

    cur = Cursor()
    cur._iter = []
    for t in Tafra.read_sql_chunks('SELECT * FROM [Table]', cur):  # type: ignore
        check_tafra(t)


def test_destructors() -> None:
    def gen_values() -> Iterator[dict[str, np.ndarray]]:
        yield {'x': np.arange(6)}
        yield {'y': np.arange(6)}

    t = Tafra(gen_values())
    check_tafra(t)

    t = build_tafra()
    t = t.update_dtypes({'x': 'float'})
    t.data['x'][2] = np.nan
    check_tafra(t)

    _ = tuple(t.to_records())
    _ = tuple(t.to_records(columns='x'))
    _ = tuple(t.to_records(columns=['x']))
    _ = tuple(t.to_records(columns=['x', 'y']))
    _ = tuple(t.to_records(cast_null=False))
    _ = tuple(t.to_records(columns='x', cast_null=False))
    _ = tuple(t.to_records(columns=['x'], cast_null=False))
    _ = tuple(t.to_records(columns=['x', 'y'], cast_null=False))

    _ = t.to_list()
    _ = t.to_list(columns='x')
    _ = t.to_list(columns=['x'])
    _ = t.to_list(columns=['x', 'y'])

    _ = t.to_list(inner=True)
    _ = t.to_list(columns='x', inner=True)
    _ = t.to_list(columns=['x'], inner=True)
    _ = t.to_list(columns=['x', 'y'], inner=True)

    _ = t.to_tuple()
    _ = t.to_tuple(columns='x')
    _ = t.to_tuple(columns=['x'])
    _ = t.to_tuple(columns=['x', 'y'])

    _ = t.to_tuple(inner=True)
    _ = t.to_tuple(columns='x', inner=True)
    _ = t.to_tuple(columns=['x'], inner=True)
    _ = t.to_tuple(columns=['x', 'y'], inner=True)

    _ = t.to_array()
    _ = t.to_array(columns='x')
    _ = t.to_array(columns=['x'])
    _ = t.to_array(columns=['x', 'y'])

    _ = t.to_pandas()
    _ = t.to_pandas(columns='x')
    _ = t.to_pandas(columns=['x'])
    _ = t.to_pandas(columns=['x', 'y'])

    filepath = Path('test/test_to_csv.csv')
    t.to_csv(filepath)
    t.to_csv(filepath, columns='x')
    t.to_csv(filepath, columns=['x'])
    t.to_csv(filepath, columns=['x', 'y'])


def test_properties() -> None:
    t = build_tafra()
    _ = t.columns
    _ = t.rows
    _ = t.data
    _ = t.dtypes
    _ = t.size
    _ = t.ndim
    _ = t.shape

    with pytest.raises(ValueError) as e:
        t.columns = ['x', 'a']  # type: ignore

    with pytest.raises(ValueError) as e:
        t.rows = 3

    with pytest.raises(ValueError) as e:
        t.data = {'x': np.arange(6)}

    with pytest.raises(ValueError) as e:
        t.dtypes = {'x': 'int'}

    with pytest.raises(ValueError) as e:
        t.size = 3

    with pytest.raises(ValueError) as e:
        t.ndim = 3

    with pytest.raises(ValueError) as e:
        t.shape = (10, 2)

def test_views() -> None:
    t = build_tafra()
    _ = t.keys()
    _ = t.values()
    _ = t.items()
    _ = t.get('x')

def test_assignment() -> None:
    t = build_tafra()
    t['x'] = np.arange(6)
    t['x'] = 3
    t['x'] = 6
    t['x'] = 'test'
    t['x'] = list(range(6))
    t['x'] = np.array(6)
    t['x'] = np.array([6])
    t['x'] = iter([1, 2, 3, 4, 5, 6])
    t['x'] = range(6)
    check_tafra(t)

    with pytest.raises(ValueError) as e:
        t['x'] = np.arange(3)

def test_dtype_update() -> None:
    t = build_tafra()
    assert t._data['x'].dtype != np.dtype(object)
    t.update_dtypes_inplace({'x': 'O'})
    assert t._data['x'].dtype == np.dtype(object)
    check_tafra(t)

    t = build_tafra()
    assert t._data['x'].dtype != np.dtype(object)
    _ = t.update_dtypes({'x': 'O'})
    assert _._data['x'].dtype == np.dtype(object)
    check_tafra(_)


def test_select() -> None:
    t = build_tafra()
    _ = t.select('x')
    _ = t.select(['x'])
    _ = t.select(['x', 'y'])

    with pytest.raises(ValueError) as e:
        _ = t.select('a')

def test_formatter() -> None:
    _ = str(object_formatter)

    t = Tafra({'x': Decimal(1.2345)})
    assert t._dtypes['x'] == 'float64'
    assert t['x'].dtype == np.dtype(float)

    object_formatter['Decimal'] = lambda x: x.astype(int)
    t = Tafra({'x': Decimal(1.2345)})
    assert t._dtypes['x'] == np.dtype(int).name
    assert t['x'].dtype == np.dtype(int)

    _ = str(object_formatter)

    for fmt in object_formatter:
        pass

    _ = object_formatter.copy()

    del object_formatter['Decimal']

    with pytest.raises(ValueError) as e:
        object_formatter['Decimal'] = lambda x: 'int'  # type: ignore

    _ = str(object_formatter)

def test_prints() -> None:
    t = build_tafra()
    _ = t.pformat()
    t.pprint()
    t.head(5)

    mock = MagicMock()
    mock.text = print
    t._repr_pretty_(mock, True)
    t._repr_pretty_(mock, False)

    _ = t._repr_html_()

def test_dunder() -> None:
    t = build_tafra()
    l = len(t)
    s = str(t)

def test_update() -> None:
    t = build_tafra()
    t2 = build_tafra()
    _ = t2.update(t2)
    check_tafra(_)

    t.update_inplace(t2)
    check_tafra(t)

    _ = t.update(t2._data)  # type: ignore
    check_tafra(_)

def test_coalesce_dtypes() -> None:
    t = build_tafra()
    t._data['a'] = np.arange(6)
    assert 'a' not in t._dtypes

    t._coalesce_dtypes()
    assert 'a' in t._dtypes
    check_tafra(t)

def test_update_dtypes() -> None:
    t = build_tafra()
    t.update_dtypes_inplace({'x': float})
    check_tafra(t)
    assert t['x'].dtype == 'float'
    assert isinstance(t['x'][0], np.float64)

    t = build_tafra()
    _ = t.update_dtypes({'x': float})
    check_tafra(_)
    assert _['x'].dtype == 'float'
    assert isinstance(_['x'][0], np.float64)

def test_rename() -> None:
    t = build_tafra()
    t.rename_inplace({'x': 'a'})
    assert 'a' in t.data
    assert 'a' in t.dtypes
    assert 'x' not in t.data
    assert 'x' not in t.dtypes
    check_tafra(t)

    t = build_tafra()
    _ = t.rename({'x': 'a'})
    assert 'a' in _.data
    assert 'a' in _.dtypes
    assert 'x' not in _.data
    assert 'x' not in _.dtypes
    check_tafra(_)

def test_delete() -> None:
    t = build_tafra()
    t.delete_inplace('x')
    assert 'x' not in t.data
    assert 'x' not in t.dtypes
    check_tafra(t)

    t = build_tafra()
    t.delete_inplace(['x'])
    assert 'x' not in t.data
    assert 'x' not in t.dtypes
    check_tafra(t)

    t = build_tafra()
    t.delete_inplace(['x', 'y'])
    assert 'x' not in t.data
    assert 'y' not in t.dtypes
    assert 'x' not in t.data
    assert 'y' not in t.dtypes
    check_tafra(t)

    t = build_tafra()
    _ = t.delete('x')
    assert 'x' not in _.data
    assert 'x' not in _.dtypes
    check_tafra(t)
    check_tafra(_)

    t = build_tafra()
    _ = t.delete(['x'])
    assert 'x' not in _.data
    assert 'x' not in _.dtypes
    check_tafra(t)
    check_tafra(_)

    t = build_tafra()
    _ = t.delete(['x', 'y'])
    assert 'x' not in _.data
    assert 'y' not in _.dtypes
    assert 'x' not in _.data
    assert 'y' not in _.dtypes
    check_tafra(t)
    check_tafra(_)

def test_iter_methods() -> None:
    t = build_tafra()
    for _ in t:
        pass

    for _ in t.iterrows():
        pass

    for _ in t.itercols():
        pass

    for _ in t.itertuples():
        pass

    for _ in t.itertuples(name='test'):
        pass

    for _ in t.itertuples(name=None):
        pass

def test_groupby() -> None:
    t = build_tafra()
    gb = t.group_by(
        ['y', 'z'], {'x': sum}, {'count': len}
    )
    check_tafra(gb)

def test_groupby_iter_fn() -> None:
    t = build_tafra()
    gb = t.group_by(
        ['y', 'z'], {
            'x': sum,
            'new_x': (sum, 'x')
        }, {'count': len}
    )
    check_tafra(gb)

def test_transform() -> None:
    t = build_tafra()
    tr = t.transform(
        ['y', 'z'], {'x': sum}, {'id': max}
    )
    check_tafra(tr)

def test_iterate_by_attr() -> None:
    t = build_tafra()
    t.id = np.empty(t.rows, dtype=int)  # type: ignore
    t['id'] = np.empty(t.rows, dtype=int)
    for i, (u, ix, grouped) in enumerate(t.iterate_by(['y', 'z'])):
        t['x'][ix] = sum(grouped['x'])
        t.id[ix] = len(grouped['x'])  # type: ignore
        t['id'][ix] = max(grouped['x'])
    check_tafra(t)

def test_iterate_by() -> None:
    t = build_tafra()
    for u, ix, grouped in t.iterate_by(['y']):
        assert isinstance(grouped, Tafra)

def group_by_in_iterate_by() -> None:
    t = build_tafra()
    for u, ix, grouped in t.iterate_by(['y']):
        assert isinstance(grouped.group_by(['z'], {'x': sum}), Tafra)

def test_update_transform() -> None:
    t = build_tafra()
    t.update(t.transform(['y'], {}, {'id': max}))

    for u, ix, it in t.iterate_by(['y']):
        t['x'][ix] = it['x'] - np.mean(it['x'])
    check_tafra(t)

def test_transform_assignment() -> None:
    t = build_tafra()
    for u, ix, it in t.iterate_by(['y']):
        it['x'][0] = 9
    check_tafra(t)
    check_tafra(it)

def test_invalid_agg() -> None:
    t = build_tafra()
    with pytest.raises(ValueError) as e:
        gb = t.group_by(
            ['y', 'z'], {sum: 'x'}  # type: ignore
        )

    with pytest.raises(ValueError) as e:
        gb = t.group_by(
            ['y', 'z'], {}, {len: 'count'}  # type: ignore
        )

def test_map() -> None:
    t = build_tafra()

    def repeat(tf: Tafra, repeats: int) -> Tafra:
        return [tf for _ in range(repeats)]

    _ = list(t.row_map(repeat, 6))
    _ = list(t.tuple_map(repeat, 6))
    _ = list(t.col_map(repeat, repeats=6))
    _ = Tafra(t.key_map(np.repeat, repeats=6))

def test_pipe() -> None:
    def fn1(t: Tafra) -> Tafra:
        return t[t['y'] == 'one']
    def fn2(t: Tafra) -> Tafra:
        return t[t['z'] == 0]

    t = build_tafra()
    check_tafra(t.pipe(fn1))
    check_tafra(t >> fn1)
    check_tafra(t.pipe(fn1).pipe(fn2))
    check_tafra(t >> fn1 >> fn2)

    def fn3(t: Tafra, i: int) -> Tafra:
        return t[t['x'] == i]

    check_tafra(t.pipe(fn3, 1))
    check_tafra(t.pipe(fn3, i=1))
    check_tafra(t >> (lambda t: fn3(t, i=1)))

def test_union() -> None:
    t = build_tafra()
    t2 = build_tafra()

    _ = t2.union(t)
    check_tafra(_)
    assert len(_) == len(t) + len(t2)

    t2.union_inplace(t)
    check_tafra(t2)
    assert len(t2) == 2 * len(t)

    t = build_tafra()
    t2 = build_tafra()
    t._dtypes['a'] = 'int'
    with pytest.raises(Exception) as e:
        t.union_inplace(t2)

    t = build_tafra()
    t2._dtypes['a'] = 'int'
    with pytest.raises(Exception) as e:
        t.union_inplace(t2)

    t = build_tafra()
    t2 = build_tafra()
    t['a'] = np.arange(6)
    with pytest.raises(ValueError) as e:
        t.union_inplace(t2)

    t = build_tafra()
    t2 = build_tafra()
    t2['a'] = np.arange(6)
    with pytest.raises(ValueError) as e:
        t.union_inplace(t2)

    t = build_tafra()
    t2 = build_tafra()
    t.rename_inplace({'x': 'a'})
    with pytest.raises(TypeError) as e:
        t.union_inplace(t2)

    t = build_tafra()
    t2 = build_tafra()
    t2.rename_inplace({'x': 'a'})
    with pytest.raises(TypeError) as e:
        t.union_inplace(t2)

    t = build_tafra()
    t2 = build_tafra()
    t.update_dtypes_inplace({'x': float})
    with pytest.raises(TypeError) as e:
        t.union_inplace(t2)

    t = build_tafra()
    t2 = build_tafra()
    t2._dtypes['x'] = 'float'
    with pytest.raises(TypeError) as e:
        t.union_inplace(t2)

def test_slice() -> None:
    t = build_tafra()
    _ = t[:3]
    _['x'][0] = 0
    check_tafra(_)

    t = build_tafra()
    _ = t[slice(0, 3)]
    _['x'][0] = 7
    check_tafra(_)
    check_tafra(t)

    t = build_tafra()
    _ = t[:3].copy()
    _['x'][0] = 9
    check_tafra(_)
    check_tafra(t)

    t = build_tafra()
    _ = t[t['x'] <= 4]
    _['x'][1] = 15
    check_tafra(_)
    check_tafra(t)

    t = build_tafra()
    _ = t[2]
    _ = t[[1, 3]]
    _ = t[np.array([2, 4])]
    _ = t[[True, False, True, True, False, True]]
    _ = t[np.array([True, False, True, True, False, True])]
    _ = t[['x', 'y']]
    _ = t[('x', 'y')]
    _ = t[[True, 2]]
    check_tafra(_)
    check_tafra(t)

    with pytest.raises(IndexError) as e:
        _ = t[np.array([[1, 2]])]

    with pytest.raises(IndexError) as e:
        _ = t[[True, False]]

    with pytest.raises(IndexError) as e:
        _ = t[np.array([True, False])]

    with pytest.raises(IndexError) as e:
        _ = t[(1, 2)]  # noqa

    with pytest.raises(IndexError) as e:
        _ = t[(1, 2.)]  # type: ignore # noqa

    with pytest.raises(ValueError) as e:
        _ = t[['x', 2]]

    with pytest.raises(TypeError) as e:
        _ = t[{'x': [1, 2]}]  # type: ignore

    with pytest.raises(TypeError) as e:
        _ = t[TestClass()]  # type: ignore # noqa

    with pytest.raises(IndexError) as e:
        _ = t[[1, 2.]]  # type: ignore

    with pytest.raises(IndexError) as e:
        _ = t[np.array([1, 2.])]


def test_invalid_dtypes() -> None:
    t = build_tafra()
    with pytest.raises(Exception) as e:
        t.update_dtypes({'x': 'flot', 'y': 'st'})

def test_invalid_assignment() -> None:
    t = build_tafra()
    _ = build_tafra()
    _._data['x'] = np.arange(5)

    with pytest.raises(Exception) as e:
        _._update_rows()

    with pytest.raises(Exception) as e:
        _ = t.update(_)

    with pytest.raises(Exception) as e:
        t.update_inplace(_)

    with warnings.catch_warnings(record=True) as w:
        t['x'] = np.arange(6)[:, None]
        assert str(w[0].message) == '`np.squeeze(ndarray)` applied to set ndim == 1.'

    with warnings.catch_warnings(record=True) as w:
        t['x'] = np.atleast_2d(np.arange(6))
        assert str(w[0].message) == '`np.squeeze(ndarray)` applied to set ndim == 1.'

    with warnings.catch_warnings(record=True) as w:
        t['x'] = np.atleast_2d(np.arange(6)).T
        assert str(w[0].message) == '`np.squeeze(ndarray)` applied to set ndim == 1.'

    with warnings.catch_warnings(record=True) as w:
        t['x'] = np.atleast_2d(np.arange(6))
        assert str(w[0].message) == '`np.squeeze(ndarray)` applied to set ndim == 1.'

    with pytest.raises(Exception) as e:
        t['x'] = np.repeat(np.arange(6)[:, None], repeats=2, axis=1)

def test_datetime() -> None:
    t = build_tafra()
    t['d'] = np.array([np.datetime64(_, 'D') for _ in range(6)])
    t.update_dtypes({'d': '<M8[D]'})
    check_tafra(t)

def test_object_parse() -> None:
    t = build_tafra()
    t['d'] = np.array([datetime.fromisoformat(f'2020-0{_+1}-01') for _ in range(6)])
    assert t._dtypes['d'] == 'object'
    check_tafra(t)

    object_formatter['datetime'] = lambda x: x.astype('datetime64[D]')
    t2 = t.parse_object_dtypes()
    assert t2['d'].dtype == np.dtype('datetime64[D]')
    check_tafra(t2)

    t.parse_object_dtypes_inplace()
    assert t['d'].dtype == np.dtype('datetime64[D]')
    check_tafra(t)

def test_coalesce() -> None:
    t = Tafra({'x': np.array([1, 2, None, 4, None])})
    t['x'] = t.coalesce('x', [[1, 2, 3, None, 5], [None, None, None, None, 'five']])
    t['y'] = t.coalesce('y', [[1, 2, 3, None, 5], [None, None, None, None, 'five']])
    assert np.all(t['x'] != np.array(None))
    assert t['y'][3] == np.array(None)
    check_tafra(t)

    t = Tafra({'x': np.array([1, 2, None, 4, None])})
    t.coalesce_inplace('x', [[1, 2, 3, None, 5], [None, None, None, None, 'five']])
    t.coalesce_inplace('y', [[1, 2, 3, None, 5], [None, None, None, None, 'five']])
    assert np.all(t['x'] != np.array(None))
    assert t['y'][3] == np.array(None)
    check_tafra(t)

    t = Tafra({'x': np.array([None])})
    t.coalesce('x', [[1], [None]])
    check_tafra(t)

def test_left_join_equi() -> None:
    l = Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })

    r = Tafra({
        'a': np.array([1, 2, 3, 4, 5, 6]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.left_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([2, 2, 2, 3, 3, 3])
    })
    t = l.left_join(r, [('x', 'a', '=='), ('z', 'c', '==')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        '_a': np.array([1, 1, 2, 2, 3, 3]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.left_join(r, [('x', 'a', '=='), ('x', '_a', '==')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 2, 2, 3, 3]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.left_join(r, [('x', 'a', '<')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

def test_inner_join() -> None:
    l = Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })

    r = Tafra({
        'a': np.array([1, 2, 3, 4, 5, 6]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.inner_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 2, 2, 3, 3]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.inner_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.inner_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })

    t = l.inner_join(r, [('x', 'a', '<=')], ['x', 'y', 'a', 'b'])
    check_tafra(t)


def test_cross_join() -> None:
    l = Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })

    r = Tafra({
        'a': np.array([1, 2, 3, 4, 5, 6]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.cross_join(r)
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 2, 2, 3, 3]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.cross_join(r)
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.cross_join(r)
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })

    t = l.cross_join(r, select=['x', 'z', 'a', 'c'])
    check_tafra(t)

    with pytest.raises(IndexError) as e:
        t = l.cross_join(r, select=['x', 'z'])

    with pytest.raises(IndexError) as e:
        t = l.cross_join(r, select=['a', 'c'])

def test_left_join_invalid() -> None:
    l = Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })

    r = Tafra({
        'a': np.array([1, 2, 3, 4, 5, 6]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })

    with pytest.raises(TypeError) as e:
        t = l.left_join(r, [('x', 'a', '===')], ['x', 'y', 'a', 'b'])

    r = Tafra({
        'a': np.array([1, 2, 3, 4, 5, 6], dtype='float'),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })

    with pytest.raises(TypeError) as e:
        t = l.left_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])

    # Corrupted _dtypes should cause join validation to fail —
    # we compare metadata (user intent), not raw array dtypes.
    r = Tafra({
        'a': np.array([1, 2, 3, 4, 5, 6]),
        'b': np.array(
            ['one', 'two', 'one', 'two', 'one', 'two'],
            dtype=np.dtypes.StringDType(),
        ),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })

    l._dtypes['x'] = 'float'
    # Should fail: metadata says 'float' but right is 'int64'
    with pytest.raises(TypeError):
        t = l.left_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])


def test_mixed_string_dtypes() -> None:
    """StringDType vs <U should interop in joins, unions, and concat."""
    left = Tafra({
        'key': np.array(['a', 'b', 'c'], dtype=np.dtypes.StringDType()),
        'val': np.array([1, 2, 3]),
    })
    right = Tafra({
        'key': np.array(['b', 'c', 'd'], dtype='<U1'),
        'info': np.array([10, 20, 30]),
    })

    # Inner join: StringDType vs <U
    t = left.inner_join(right, on=[('key', 'key', '==')], select=['key', 'val', 'info'])
    assert len(t) == 2
    assert set(t['key']) == {'b', 'c'}

    # Left join: StringDType vs <U (with nulls) — int column falls back to object
    t = left.left_join(right, on=[('key', 'key', '==')], select=['key', 'val', 'info'])
    assert len(t) == 3
    a_idx = np.where(t['key'] == 'a')[0][0]
    assert t['info'][a_idx] is None  # int col → object with None
    assert t['info'].dtype == object

    # Left join: <U vs StringDType (reversed) — int col also object
    t = right.left_join(left, on=[('key', 'key', '==')], select=['key', 'info', 'val'])
    assert len(t) == 3
    d_idx = np.where(t['key'] == 'd')[0][0]
    assert t['val'][d_idx] is None

    # Left join with string right column: should preserve StringDType with None
    right_str = Tafra({
        'key': np.array(['b', 'c'], dtype='<U1'),
        'name': np.array(['Bob', 'Carol'], dtype=np.dtypes.StringDType()),
    })
    t = left.left_join(right_str, on=[('key', 'key', '==')], select=['key', 'val', 'name'])
    assert len(t) == 3
    a_idx = np.where(t['key'] == 'a')[0][0]
    assert t['name'][a_idx] is None
    assert t['name'].dtype == np.dtypes.StringDType(na_object=None)

    # Left join with float right column: should use NaN
    right_flt = Tafra({
        'key': np.array(['b', 'c'], dtype='<U1'),
        'score': np.array([1.5, 2.5]),
    })
    t = left.left_join(right_flt, on=[('key', 'key', '==')], select=['key', 'val', 'score'])
    assert len(t) == 3
    a_idx = np.where(t['key'] == 'a')[0][0]
    assert np.isnan(t['score'][a_idx])
    assert t['score'].dtype == np.float64

    # Different <U widths
    wide = Tafra({
        'key': np.array(['abc', 'def'], dtype='<U10'),
        'val': np.array([1, 2]),
    })
    narrow = Tafra({
        'key': np.array(['abc', 'xyz'], dtype='<U3'),
        'info': np.array([10, 20]),
    })
    t = wide.left_join(narrow, on=[('key', 'key', '==')], select=['key', 'val', 'info'])
    assert len(t) == 2

    # Union: mixed string dtypes
    left2 = Tafra({
        'x': np.array([1, 2]),
        'name': np.array(['a', 'b'], dtype=np.dtypes.StringDType()),
    })
    right2 = Tafra({
        'x': np.array([3, 4]),
        'name': np.array(['c', 'd'], dtype='<U1'),
    })
    t = left2.union(right2)
    assert len(t) == 4

    # Concat: mixed string dtypes
    t = Tafra.concat([left2, right2])
    assert len(t) == 4
    assert t['name'].dtype == np.dtypes.StringDType()  # upcasts to StringDType

    # Left join with datetime column: should use NaT for unmatched
    left_dt = Tafra({
        'key': np.array([1, 2, 3]),
        'val': np.array([10, 20, 30]),
    })
    right_dt = Tafra({
        'key': np.array([2, 3]),
        'ts': np.array(['2024-01-01', '2024-06-15'],
                        dtype='datetime64[D]'),
    })
    t = left_dt.left_join(
        right_dt, on=[('key', 'key', '==')],
        select=['key', 'val', 'ts'],
    )
    assert len(t) == 3
    assert t['ts'].dtype.kind == 'M'  # preserved datetime64
    idx_1 = np.where(t['key'] == 1)[0][0]
    assert np.isnat(t['ts'][idx_1])  # unmatched → NaT

    # Left join with timedelta column: should use NaT for unmatched
    right_td = Tafra({
        'key': np.array([2, 3]),
        'dur': np.array([10, 20], dtype='timedelta64[D]'),
    })
    t = left_dt.left_join(
        right_td, on=[('key', 'key', '==')],
        select=['key', 'val', 'dur'],
    )
    assert len(t) == 3
    assert t['dur'].dtype.kind == 'm'  # preserved timedelta64
    idx_1 = np.where(t['key'] == 1)[0][0]
    assert np.isnat(t['dur'][idx_1])  # unmatched → NaT


def test_update_dtypes_string_conversion() -> None:
    """update_dtypes_inplace should not no-op between <U and StringDType."""
    # <U -> StringDType: currently skipped (both reduce to 'str'),
    # but should at least not crash and should update _dtypes label
    t = Tafra({'x': np.array(['hello', 'world'], dtype='<U10')})
    assert t['x'].dtype == np.dtype('<U10')
    t.update_dtypes_inplace({'x': np.dtypes.StringDType()})
    assert t._dtypes['x'] == 'str'

    # int -> float conversion should work
    t2 = Tafra({'x': np.array([1, 2, 3])})
    t2.update_dtypes_inplace({'x': 'float64'})
    assert t2['x'].dtype == np.float64

    # float -> int conversion should work
    t3 = Tafra({'x': np.array([1.0, 2.0, 3.0])})
    t3.update_dtypes_inplace({'x': 'int64'})
    assert t3['x'].dtype == np.int64


def test_left_join_object_fallback_warning() -> None:
    """Left join should warn when int/bool columns fall back to object."""
    import warnings

    left = Tafra({'k': np.array([1, 2, 3]), 'v': np.array([10, 20, 30])})
    right = Tafra({'k': np.array([2, 3]), 'count': np.array([100, 200])})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        t = left.left_join(right, on=[('k', 'k', '==')])

    # Should have warned about 'count' column
    object_warnings = [x for x in w if 'cast to object' in str(x.message)]
    assert len(object_warnings) == 1
    assert "'count'" in str(object_warnings[0].message)
    assert t['count'].dtype == object

    # String and float columns should NOT warn
    right2 = Tafra({
        'k': np.array([2, 3]),
        'name': np.array(['b', 'c'], dtype=np.dtypes.StringDType()),
        'score': np.array([1.5, 2.5]),
    })

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        t2 = left.left_join(right2, on=[('k', 'k', '==')])

    object_warnings = [x for x in w if 'cast to object' in str(x.message)]
    assert len(object_warnings) == 0
    assert t2['name'].dtype == np.dtypes.StringDType(na_object=None)
    assert t2['score'].dtype == np.float64


class TestStringDtypeInterop:
    """Comprehensive tests for StringDType / <U interop (v2.2.1 bugfixes)."""

    # -- Bug 1: Join rejects StringDType vs <U --

    def test_inner_join_stringdtype_vs_fixed_u(self) -> None:
        left = Tafra({
            'k': np.array(['a', 'b'], dtype=np.dtypes.StringDType()),
            'v': np.array([1, 2]),
        })
        right = Tafra({
            'k': np.array(['b', 'c'], dtype='<U1'),
            'w': np.array([10, 20]),
        })
        t = left.inner_join(right, on=[('k', 'k', '==')])
        assert len(t) == 1
        assert t['k'][0] == 'b'

    def test_left_join_stringdtype_vs_fixed_u(self) -> None:
        left = Tafra({
            'k': np.array(['a', 'b'], dtype=np.dtypes.StringDType()),
            'v': np.array([1, 2]),
        })
        right = Tafra({
            'k': np.array(['b'], dtype='<U1'),
            'w': np.array([10]),
        })
        t = left.left_join(right, on=[('k', 'k', '==')])
        assert len(t) == 2

    def test_inner_join_fixed_u_vs_stringdtype(self) -> None:
        """Reversed direction."""
        left = Tafra({
            'k': np.array(['a', 'b'], dtype='<U1'),
            'v': np.array([1, 2]),
        })
        right = Tafra({
            'k': np.array(['b', 'c'], dtype=np.dtypes.StringDType()),
            'w': np.array([10, 20]),
        })
        t = left.inner_join(right, on=[('k', 'k', '==')])
        assert len(t) == 1

    # -- Bug 2: Join rejects <U8 vs <U12 --

    def test_join_different_u_widths(self) -> None:
        left = Tafra({
            'k': np.array(['DELAWARE', 'MIDLAND'], dtype='<U8'),
            'v': np.array([1, 2]),
        })
        right = Tafra({
            'k': np.array(['DELAWARE BASIN', 'MIDLAND'], dtype='<U14'),
            'w': np.array([10, 20]),
        })
        t = left.left_join(right, on=[('k', 'k', '==')])
        assert len(t) == 2
        # 'DELAWARE' != 'DELAWARE BASIN', so only MIDLAND matches
        mid_idx = np.where(t['k'] == 'MIDLAND')[0][0]
        assert t['w'][mid_idx] == 20

    # -- Bug 3: Union rejects mixed string dtypes --

    def test_union_stringdtype_vs_fixed_u(self) -> None:
        left = Tafra({
            'x': np.array([1, 2]),
            'name': np.array(['a', 'b'], dtype=np.dtypes.StringDType()),
        })
        right = Tafra({
            'x': np.array([3, 4]),
            'name': np.array(['c', 'd'], dtype='<U1'),
        })
        t = left.union(right)
        assert len(t) == 4

    # -- Bug 4: update_dtypes_inplace silent no-op --

    def test_update_dtypes_str_label_converts_to_stringdtype(self) -> None:
        t = Tafra({'x': np.array(['hello', 'world'], dtype='<U10')})
        assert t['x'].dtype == np.dtype('<U10')
        t.update_dtypes_inplace({'x': 'str'})
        assert isinstance(t['x'].dtype, np.dtypes.StringDType)

    def test_update_dtypes_str_label_supports_none(self) -> None:
        t = Tafra({'x': np.array(['hello', 'world'], dtype='<U10')})
        t.update_dtypes_inplace({'x': 'str'})
        t['x'][0] = None  # type: ignore[assignment]
        assert t['x'][0] is None

    def test_update_dtypes_explicit_stringdtype(self) -> None:
        t = Tafra({'x': np.array(['hello', 'world'], dtype='<U10')})
        t.update_dtypes_inplace({'x': np.dtypes.StringDType()})
        assert t._dtypes['x'] == 'str'

    def test_construction_preserves_original_dtype(self) -> None:
        """__post_init__ should NOT convert <U to StringDType."""
        t = Tafra({'x': np.array(['hello', 'world'], dtype='<U10')})
        assert t['x'].dtype == np.dtype('<U10')

        t2 = Tafra({
            'x': np.array(['hello'], dtype=np.dtypes.StringDType()),
        })
        assert isinstance(t2['x'].dtype, np.dtypes.StringDType)

    # -- Bug 5: Left join null-fill dtype preservation --

    def test_left_join_null_string_preserves_dtype(self) -> None:
        left = Tafra({'k': np.array([1, 2, 3]), 'v': np.array([10, 20, 30])})
        right = Tafra({
            'k': np.array([2, 3]),
            'name': np.array(['Bob', 'Carol'],
                             dtype=np.dtypes.StringDType()),
        })
        t = left.left_join(right, on=[('k', 'k', '==')])
        assert isinstance(t['name'].dtype, np.dtypes.StringDType)
        idx = np.where(t['k'] == 1)[0][0]
        assert t['name'][idx] is None

    def test_left_join_null_fixed_u_becomes_stringdtype(self) -> None:
        """<U columns also get StringDType(na_object=None) for null fill."""
        left = Tafra({'k': np.array([1, 2, 3]), 'v': np.array([10, 20, 30])})
        right = Tafra({
            'k': np.array([2, 3]),
            'name': np.array(['Bob', 'Carol'], dtype='<U5'),
        })
        t = left.left_join(right, on=[('k', 'k', '==')])
        assert isinstance(t['name'].dtype, np.dtypes.StringDType)
        idx = np.where(t['k'] == 1)[0][0]
        assert t['name'][idx] is None

    def test_left_join_null_float_uses_nan(self) -> None:
        left = Tafra({'k': np.array([1, 2, 3]), 'v': np.array([10, 20, 30])})
        right = Tafra({
            'k': np.array([2, 3]),
            'score': np.array([1.5, 2.5]),
        })
        t = left.left_join(right, on=[('k', 'k', '==')])
        assert t['score'].dtype == np.float64
        idx = np.where(t['k'] == 1)[0][0]
        assert np.isnan(t['score'][idx])

    def test_left_join_null_float32_preserves_width(self) -> None:
        left = Tafra({'k': np.array([1, 2]), 'v': np.array([10, 20])})
        right = Tafra({
            'k': np.array([2]),
            'score': np.array([1.5], dtype=np.float32),
        })
        t = left.left_join(right, on=[('k', 'k', '==')])
        assert t['score'].dtype == np.float32
        assert np.isnan(t['score'][0])

    def test_left_join_null_datetime_uses_nat(self) -> None:
        left = Tafra({'k': np.array([1, 2, 3]), 'v': np.array([10, 20, 30])})
        right = Tafra({
            'k': np.array([2, 3]),
            'ts': np.array(['2024-01-01', '2024-06-15'],
                           dtype='datetime64[D]'),
        })
        t = left.left_join(right, on=[('k', 'k', '==')])
        assert t['ts'].dtype.kind == 'M'
        idx = np.where(t['k'] == 1)[0][0]
        assert np.isnat(t['ts'][idx])

    def test_left_join_null_timedelta_uses_nat(self) -> None:
        left = Tafra({'k': np.array([1, 2, 3]), 'v': np.array([10, 20, 30])})
        right = Tafra({
            'k': np.array([2, 3]),
            'dur': np.array([10, 20], dtype='timedelta64[D]'),
        })
        t = left.left_join(right, on=[('k', 'k', '==')])
        assert t['dur'].dtype.kind == 'm'
        idx = np.where(t['k'] == 1)[0][0]
        assert np.isnat(t['dur'][idx])

    def test_left_join_null_int_falls_back_to_object(self) -> None:
        left = Tafra({'k': np.array([1, 2, 3]), 'v': np.array([10, 20, 30])})
        right = Tafra({
            'k': np.array([2, 3]),
            'count': np.array([100, 200]),
        })
        t = left.left_join(right, on=[('k', 'k', '==')])
        assert t['count'].dtype == object
        idx = np.where(t['k'] == 1)[0][0]
        assert t['count'][idx] is None

    def test_left_join_null_bool_falls_back_to_object(self) -> None:
        left = Tafra({'k': np.array([1, 2, 3]), 'v': np.array([10, 20, 30])})
        right = Tafra({
            'k': np.array([2, 3]),
            'flag': np.array([True, False]),
        })
        t = left.left_join(right, on=[('k', 'k', '==')])
        assert t['flag'].dtype == object
        idx = np.where(t['k'] == 1)[0][0]
        assert t['flag'][idx] is None

    def test_left_join_null_bytes_falls_back_to_object(self) -> None:
        """Byte strings (kind='S') should NOT go through string path."""
        left = Tafra({'k': np.array([1, 2, 3]), 'v': np.array([10, 20, 30])})
        right = Tafra({
            'k': np.array([2, 3]),
            'raw': np.array([b'abc', b'def'], dtype='S3'),
        })
        t = left.left_join(right, on=[('k', 'k', '==')])
        assert t['raw'].dtype == object
        idx = np.where(t['k'] == 1)[0][0]
        assert t['raw'][idx] is None

    # -- Bug 6: Object fallback warning --

    def test_left_join_warns_on_object_fallback(self) -> None:
        import warnings
        left = Tafra({'k': np.array([1, 2, 3]), 'v': np.array([10, 20, 30])})
        right = Tafra({
            'k': np.array([2, 3]),
            'count': np.array([100, 200]),
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            left.left_join(right, on=[('k', 'k', '==')])
        msgs = [x for x in w if 'cast to object' in str(x.message)]
        assert len(msgs) == 1
        assert "'count'" in str(msgs[0].message)

    def test_left_join_no_warning_for_native_nulls(self) -> None:
        import warnings
        left = Tafra({'k': np.array([1, 2, 3]), 'v': np.array([10, 20, 30])})
        right = Tafra({
            'k': np.array([2, 3]),
            'name': np.array(['b', 'c'],
                             dtype=np.dtypes.StringDType()),
            'score': np.array([1.5, 2.5]),
            'ts': np.array(['2024-01-01', '2024-06-15'],
                           dtype='datetime64[D]'),
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            left.left_join(right, on=[('k', 'k', '==')])
        msgs = [x for x in w if 'cast to object' in str(x.message)]
        assert len(msgs) == 0

    # -- Bug 7: Metadata validation in joins --

    def test_join_rejects_corrupted_metadata(self) -> None:
        """Corrupted _dtypes should cause join to fail."""
        left = Tafra({
            'k': np.array([1, 2, 3]),
            'v': np.array([10, 20, 30]),
        })
        right = Tafra({
            'k': np.array([2, 3]),
            'w': np.array([100, 200]),
        })
        left._dtypes['k'] = 'float64'
        with pytest.raises(TypeError):
            left.left_join(right, on=[('k', 'k', '==')])

    def test_join_accepts_matching_metadata(self) -> None:
        """Same _dtypes label = same user intent, even if array widths differ."""
        left = Tafra({
            'k': np.array(['abc', 'def'], dtype='<U3'),
            'v': np.array([1, 2]),
        })
        right = Tafra({
            'k': np.array(['abc', 'xyz'],
                          dtype=np.dtypes.StringDType()),
            'w': np.array([10, 20]),
        })
        # Both have _dtypes['k'] == 'str'
        assert left._dtypes['k'] == right._dtypes['k'] == 'str'
        t = left.inner_join(right, on=[('k', 'k', '==')])
        assert len(t) == 1

    # -- Join key dtype combinations --

    def test_join_on_int64_keys(self) -> None:
        left = Tafra({
            'k': np.array([1, 2, 3], dtype=np.int64),
            'v': np.array([10, 20, 30]),
        })
        right = Tafra({
            'k': np.array([2, 3, 4], dtype=np.int64),
            'w': np.array([100, 200, 300]),
        })
        t = left.inner_join(right, on=[('k', 'k', '==')])
        assert len(t) == 2
        assert set(t['k']) == {2, 3}

    def test_join_on_float64_keys(self) -> None:
        left = Tafra({
            'k': np.array([1.0, 2.0, 3.0]),
            'v': np.array([10, 20, 30]),
        })
        right = Tafra({
            'k': np.array([2.0, 3.0, 4.0]),
            'w': np.array([100, 200, 300]),
        })
        t = left.inner_join(right, on=[('k', 'k', '==')])
        assert len(t) == 2

    def test_join_on_stringdtype_keys(self) -> None:
        sd = np.dtypes.StringDType()
        left = Tafra({
            'k': np.array(['a', 'b', 'c'], dtype=sd),
            'v': np.array([1, 2, 3]),
        })
        right = Tafra({
            'k': np.array(['b', 'c', 'd'], dtype=sd),
            'w': np.array([10, 20, 30]),
        })
        t = left.inner_join(right, on=[('k', 'k', '==')])
        assert len(t) == 2
        assert set(t['k']) == {'b', 'c'}

    def test_join_on_fixed_u_keys(self) -> None:
        left = Tafra({
            'k': np.array(['a', 'b', 'c'], dtype='<U1'),
            'v': np.array([1, 2, 3]),
        })
        right = Tafra({
            'k': np.array(['b', 'c', 'd'], dtype='<U1'),
            'w': np.array([10, 20, 30]),
        })
        t = left.inner_join(right, on=[('k', 'k', '==')])
        assert len(t) == 2

    def test_join_rejects_int_vs_float_keys(self) -> None:
        left = Tafra({
            'k': np.array([1, 2, 3], dtype=np.int64),
            'v': np.array([10, 20, 30]),
        })
        right = Tafra({
            'k': np.array([2.0, 3.0, 4.0]),
            'w': np.array([100, 200, 300]),
        })
        with pytest.raises(TypeError):
            left.inner_join(right, on=[('k', 'k', '==')])

    def test_join_rejects_int_vs_string_keys(self) -> None:
        left = Tafra({
            'k': np.array([1, 2, 3]),
            'v': np.array([10, 20, 30]),
        })
        right = Tafra({
            'k': np.array(['a', 'b', 'c'],
                          dtype=np.dtypes.StringDType()),
            'w': np.array([10, 20, 30]),
        })
        with pytest.raises(TypeError):
            left.inner_join(right, on=[('k', 'k', '==')])

    # -- Concat mixed string dtypes --

    def test_concat_stringdtype_and_fixed_u(self) -> None:
        t1 = Tafra({'x': np.array(['a', 'b'], dtype=np.dtypes.StringDType())})
        t2 = Tafra({'x': np.array(['c', 'd'], dtype='<U1')})
        t = Tafra.concat([t1, t2])
        assert len(t) == 4
        # numpy upcasts to StringDType
        assert isinstance(t['x'].dtype, np.dtypes.StringDType)

    # -- Left join full match (no nulls) preserves dtypes --

    def test_left_join_full_match_preserves_dtypes(self) -> None:
        left = Tafra({'k': np.array([1, 2]), 'v': np.array([10, 20])})
        right = Tafra({
            'k': np.array([1, 2]),
            'name': np.array(['A', 'B'],
                             dtype=np.dtypes.StringDType()),
            'score': np.array([1.5, 2.5]),
            'count': np.array([100, 200]),
        })
        t = left.left_join(right, on=[('k', 'k', '==')])
        # No nulls → original dtypes preserved, no StringDType(na_object)
        assert isinstance(t['name'].dtype, np.dtypes.StringDType)
        assert t['score'].dtype == np.float64
        assert t['count'].dtype == np.int64


def test_csv() -> None:
    write_path = 'test/test_to_csv.csv'

    def write_reread(t: Tafra) -> None:
        t.to_csv(write_path)
        t2 = Tafra.read_csv(write_path, dtypes=t.dtypes)

        for c1, c2 in zip(t.columns, t2.columns):
            assert np.array_equal(t.data[c1], t2.data[c2])
            assert np.array_equal(t.dtypes[c1], t2.dtypes[c2])

    # straightforward CSV - inference heuristic works
    path = Path('test/ex1.csv')
    t = Tafra.read_csv(path)
    assert t.dtypes['a'] == 'int32'
    assert t.dtypes['b'] == 'bool'
    assert t.dtypes['c'] == 'float64'
    assert t.rows == 6
    assert len(t.columns) == 3
    check_tafra(t)
    write_reread(t)

    # test again with TextIOWrapper
    with open('test/ex1.csv', 'r') as f:
        t = Tafra.read_csv(f)
    assert t.dtypes['a'] == 'int32'
    assert t.dtypes['b'] == 'bool'
    assert t.dtypes['c'] == 'float64'
    assert t.rows == 6
    assert len(t.columns) == 3
    check_tafra(t)
    write_reread(t)

    with open(write_path, 'w') as f:
        t.to_csv(f)
    with pytest.raises(ValueError) as e:
        with open(write_path) as f:
            t.to_csv(f)

    # short CSV - ends during inference period
    t = Tafra.read_csv('test/ex2.csv')
    assert t.dtypes['a'] == 'int32'
    assert t.dtypes['b'] == 'bool'
    assert t.dtypes['c'] == 'float64'
    assert t.rows == 2
    assert len(t.columns) == 3
    check_tafra(t)
    write_reread(t)

    # harder CSV - promote to object during inference period,
    #   duplicate column name
    t = Tafra.read_csv('test/ex3.csv')
    assert t.dtypes['a'] == 'int32'
    assert t.dtypes['b'] == 'str'
    assert t.dtypes['b (2)'] == 'float64'
    assert t.rows == 6
    assert len(t.columns) == 3
    check_tafra(t)
    write_reread(t)

    # as above, but with a promotion required after inference period
    #   (heuristic fails)
    t = Tafra.read_csv('test/ex4.csv')
    assert t.dtypes['a'] == 'int32'
    assert t.dtypes['b'] == 'str'
    assert t.dtypes['b (2)'] == 'float64'
    assert t.rows == 6
    assert len(t.columns) == 3
    check_tafra(t)
    write_reread(t)

    # bad CSV - missing column on row #4
    with pytest.raises(ValueError) as e:
        t = Tafra.read_csv('test/ex5.csv')

    # bad CSV - missing column on row #4 - after guess rows
    with pytest.raises(ValueError) as e:
        t = Tafra.read_csv('test/ex5.csv', guess_rows=2)

    # missing column - but numpy will automatically convert missing (None) to nan
    t = Tafra.read_csv('test/ex6.csv')
    assert t.dtypes['dp'] == 'float64'
    assert t.dtypes['dp_prime'] == 'float64'
    assert t.dtypes['dp_prime_te'] == 'float64'
    assert t.dtypes['t'] == 'float64'
    assert t.dtypes['te'] == 'float64'
    check_tafra(t)

    # missing column - do not automatically cast
    t = Tafra.read_csv('test/ex6.csv', missing=None)
    assert t.dtypes['dp'] == 'float64'
    assert t.dtypes['dp_prime'] == 'str'
    assert t.dtypes['dp_prime_te'] == 'str'
    assert t.dtypes['t'] == 'float64'
    assert t.dtypes['te'] == 'float64'
    check_tafra(t)

    t.update_dtypes_inplace({'dp_prime': float, 'dp_prime_te': 'float64'})
    assert t.dtypes['dp_prime'] == 'float64'
    assert t.dtypes['dp_prime_te'] == 'float64'
    check_tafra(t)

    # force dtypes on missing columns
    t = Tafra.read_csv(
        'test/ex6.csv', missing=None,
        dtypes={'dp_prime': np.float64, 'dp_prime_te': np.float32},
    )
    assert t.dtypes['dp'] == 'float64'
    assert t.dtypes['dp_prime'] == 'float64'
    assert t.dtypes['dp_prime_te'] == 'float32'
    assert t.dtypes['t'] == 'float64'
    assert t.dtypes['te'] == 'float64'
    check_tafra(t)

    # override a column type
    t = Tafra.read_csv('test/ex4.csv', dtypes={'a': 'float32'})
    assert t.dtypes['a'] == 'float32'
    assert t.dtypes['b'] == 'str'
    assert t.dtypes['b (2)'] == 'float64'
    assert t.rows == 6
    assert len(t.columns) == 3
    check_tafra(t)
    write_reread(t)


def test_left_join_dtype_preserves_left() -> None:
    """Left dtypes should win over right dtypes for shared column names."""
    l = Tafra({
        'key': np.array([1, 2, 3]),
        'val': np.array([1.0, 2.0, 3.0]),
    })
    r = Tafra({
        'key': np.array([1, 2, 3]),
        'other': np.array([10, 20, 30]),
    })
    t = l.left_join(r, [('key', 'key', '==')])
    assert t._dtypes['key'] == l._dtypes['key']

    t2 = l.inner_join(r, [('key', 'key', '==')])
    assert t2._dtypes['key'] == l._dtypes['key']

    # both joins should produce the same dtype for shared columns
    assert t._dtypes['key'] == t2._dtypes['key']


def test_to_csv_unsupported_type() -> None:
    """to_csv should raise TypeError for unsupported file argument types."""
    t = build_tafra()
    with pytest.raises(TypeError):
        t.to_csv(123)  # type: ignore


def test_parse_iterable_true_iterable() -> None:
    """Constructing from a non-rewindable iterable should not skip/duplicate elements."""
    def gen_pairs() -> Iterator[tuple[str, np.ndarray]]:
        yield ('a', np.array([1, 2, 3]))
        yield ('b', np.array([4, 5, 6]))
        yield ('c', np.array([7, 8, 9]))

    class PairIterable:
        def __iter__(self) -> Iterator[tuple[str, np.ndarray]]:
            return gen_pairs()

    t = Tafra(PairIterable())
    assert list(t.columns) == ['a', 'b', 'c']
    assert np.array_equal(t['a'], np.array([1, 2, 3]))
    assert np.array_equal(t['b'], np.array([4, 5, 6]))
    assert np.array_equal(t['c'], np.array([7, 8, 9]))


def test_iterate_by_single_column_yields_tuple() -> None:
    """IterateBy should always yield a tuple for group values, even with one column."""
    t = build_tafra()
    for u, ix, grouped in t.iterate_by(['y']):
        assert isinstance(u, tuple), f'Expected tuple, got {type(u)}'


def test_csv_reader_empty_file() -> None:
    """CSVReader should raise ValueError on an empty file."""
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    try:
        with pytest.raises(ValueError, match='empty'):
            Tafra.read_csv(path)
    finally:
        os.unlink(path)


def test_csv_reader_unsupported_source_type() -> None:
    """CSVReader should raise TypeError for unsupported source types."""
    from tafra.csvreader import CSVReader
    with pytest.raises(TypeError):
        CSVReader(123)  # type: ignore


def test_ndim_always_two() -> None:
    """ndim should always be 2 regardless of column count."""
    t1 = Tafra({'a': np.array([1])})
    assert t1.ndim == 2

    t2 = build_tafra()  # 3 columns
    assert t2.ndim == 2

    t3 = Tafra({f'c{i}': np.arange(3) for i in range(10)})
    assert t3.ndim == 2


def test_object_formatter_validation() -> None:
    """ObjectFormatter should reject functions that don't return ndarray."""
    # function that returns wrong type
    with pytest.raises(ValueError):
        object_formatter['Bad'] = lambda x: 'not an array'  # type: ignore

    # function that raises an exception
    with pytest.raises(ValueError):
        object_formatter['Bad'] = lambda x: x.no_such_method()  # type: ignore


def test_parse_sequence_no_mutation() -> None:
    """Constructing from a list of dicts should not mutate the input dicts."""
    d1 = {'x': np.array([1, 2, 3])}
    d2 = {'y': np.array([4, 5, 6])}
    d1_keys_before = set(d1.keys())
    d2_keys_before = set(d2.keys())

    _ = Tafra([d1, d2])

    assert set(d1.keys()) == d1_keys_before
    assert set(d2.keys()) == d2_keys_before


def test_build_group_indices_single_col() -> None:
    """_build_group_indices returns correct groups and row indices for 1 column."""
    from tafra.group import GroupSet
    t = Tafra({
        'g': np.array([2, 1, 2, 1, 3]),
        'v': np.array([10., 20., 30., 40., 50.]),
    })
    unique, indices = GroupSet._build_group_indices(t, ['g'])
    # should preserve first-seen order: 2, 1, 3
    assert [u[0] for u in unique] == [2, 1, 3]
    np.testing.assert_array_equal(indices[0], np.array([0, 2]))  # g==2
    np.testing.assert_array_equal(indices[1], np.array([1, 3]))  # g==1
    np.testing.assert_array_equal(indices[2], np.array([4]))     # g==3


def test_build_group_indices_multi_col() -> None:
    """_build_group_indices returns correct groups and row indices for 2 columns."""
    from tafra.group import GroupSet
    t = Tafra({
        'a': np.array([1, 1, 2, 2, 1]),
        'b': np.array(['x', 'y', 'x', 'y', 'x']),
        'v': np.arange(5.0),
    })
    unique, indices = GroupSet._build_group_indices(t, ['a', 'b'])
    # first-seen order: (1,'x'), (1,'y'), (2,'x'), (2,'y')
    assert unique[0] == (1, 'x')
    assert unique[1] == (1, 'y')
    assert unique[2] == (2, 'x')
    assert unique[3] == (2, 'y')
    np.testing.assert_array_equal(indices[0], np.array([0, 4]))
    np.testing.assert_array_equal(indices[1], np.array([1]))
    np.testing.assert_array_equal(indices[2], np.array([2]))
    np.testing.assert_array_equal(indices[3], np.array([3]))


def test_build_group_indices_single_group() -> None:
    """All rows in one group."""
    from tafra.group import GroupSet
    t = Tafra({
        'g': np.array([5, 5, 5]),
        'v': np.arange(3.0),
    })
    unique, indices = GroupSet._build_group_indices(t, ['g'])
    assert len(unique) == 1
    assert unique[0] == (5,)
    np.testing.assert_array_equal(indices[0], np.array([0, 1, 2]))


def test_groupby_values_match_original() -> None:
    """GroupBy must produce identical results before and after optimization."""
    t = Tafra({
        'g1': np.array([1, 1, 2, 2, 3, 3]),
        'g2': np.array(['a', 'b', 'a', 'b', 'a', 'b']),
        'val': np.array([10., 20., 30., 40., 50., 60.]),
    })
    gb = t.group_by(['g1', 'g2'], {'s': (np.sum, 'val'), 'm': (np.mean, 'val')}, {'count': len})
    assert len(gb) == 6
    mask = (gb['g1'] == 1) & (gb['g2'] == 'a')
    assert gb['s'][mask][0] == 10.0
    assert gb['m'][mask][0] == 10.0


def test_transform_values_match_original() -> None:
    """Transform must produce same-length result with correct per-group aggregations."""
    t = Tafra({
        'g': np.array([1, 1, 2, 2, 2]),
        'val': np.array([10., 20., 30., 40., 50.]),
    })
    tr = t.transform(['g'], {'mean_val': (np.mean, 'val')})
    assert len(tr) == 5
    np.testing.assert_array_almost_equal(tr['mean_val'][:2], [15.0, 15.0])
    np.testing.assert_array_almost_equal(tr['mean_val'][2:], [40.0, 40.0, 40.0])


def test_inner_join_no_match() -> None:
    """InnerJoin with no matching rows returns empty Tafra."""
    l = Tafra({'key': np.array([1, 2, 3]), 'lv': np.array([10., 20., 30.])})
    r = Tafra({'key': np.array([4, 5, 6]), 'rv': np.array([40., 50., 60.])})
    t = l.inner_join(r, [('key', 'key', '==')])
    assert len(t) == 0


def test_inner_join_many_to_many() -> None:
    """InnerJoin handles many-to-many correctly."""
    l = Tafra({'key': np.array([1, 1]), 'lv': np.array([10., 20.])})
    r = Tafra({'key': np.array([1, 1]), 'rv': np.array([30., 40.])})
    t = l.inner_join(r, [('key', 'key', '==')])
    assert len(t) == 4


def test_inner_join_string_keys() -> None:
    """InnerJoin on string columns with non-overlapping values must use shared codebook."""
    sd = np.dtypes.StringDType()
    l = Tafra({
        'name': np.array(['alice', 'bob', 'carol'], dtype=sd),
        'lv': np.array([1., 2., 3.]),
    })
    r = Tafra({
        'name': np.array(['bob', 'dave', 'carol'], dtype=sd),
        'rv': np.array([20., 40., 30.]),
    })
    t = l.inner_join(r, [('name', 'name', '==')])
    assert len(t) == 2
    # bob and carol match
    names = set(t['name'])
    assert 'bob' in names
    assert 'carol' in names


def test_left_join_string_keys() -> None:
    """LeftJoin on string columns with unmatched values."""
    sd = np.dtypes.StringDType()
    l = Tafra({
        'name': np.array(['alice', 'bob'], dtype=sd),
        'lv': np.array([1., 2.]),
    })
    r = Tafra({
        'name': np.array(['bob', 'carol'], dtype=sd),
        'rv': np.array([20., 30.]),
    })
    t = l.left_join(r, [('name', 'name', '==')])
    assert len(t) == 2
    # alice has no match -> rv is NaN (float columns use NaN, not None)
    alice_idx = np.where(t['name'] == 'alice')[0][0]
    assert np.isnan(t['rv'][alice_idx])
    assert t['rv'].dtype == np.float64  # preserved, not object


def test_left_join_no_match_preserves_left() -> None:
    """LeftJoin with no matches still returns all left rows."""
    l = Tafra({'key': np.array([1, 2, 3]), 'lv': np.array([10., 20., 30.])})
    r = Tafra({'key': np.array([4, 5, 6]), 'rv': np.array([40., 50., 60.])})
    t = l.left_join(r, [('key', 'key', '==')])
    assert len(t) == 3
    assert all(np.isnan(t['rv'][i]) for i in range(3))


def test_string_column_uses_stringdtype() -> None:
    """String columns should use numpy's StringDType, not object or fixed-width U."""
    sd = np.dtypes.StringDType()
    t = Tafra({'s': np.array(['hello', 'world'], dtype=sd)})
    assert t['s'].dtype == sd

    # Assigning a Python str scalar should produce StringDType
    t2 = Tafra({'x': np.array([1, 2])})
    t2['label'] = 'constant'
    assert t2['label'].dtype.kind == 'T'


def test_object_string_array_converted_to_stringdtype() -> None:
    """Object arrays of strings should be auto-converted to StringDType."""
    obj_arr = np.array(['a', 'b', 'c'], dtype=object)
    t = Tafra({'s': obj_arr})
    assert t['s'].dtype.kind == 'T'


def test_csv_reader_string_columns_use_stringdtype() -> None:
    """CSVReader should produce StringDType for string columns."""
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.write(fd, b'name,value\nalice,1\nbob,2\n')
    os.close(fd)
    try:
        t = Tafra.read_csv(path)
        assert t['name'].dtype.kind == 'T'
    finally:
        os.unlink(path)


def test_format_dtype_stringdtype() -> None:
    """_format_dtype should return 'str' for StringDType."""
    assert Tafra._format_dtype(np.dtypes.StringDType()) == 'str'
    assert Tafra._format_dtype(np.dtype('<U10')) == 'str'


def test_chunks_basic() -> None:
    """chunks() splits into n roughly equal pieces."""
    t = Tafra({'x': np.arange(10), 'y': np.arange(10.0)})
    parts = t.chunks(3)
    assert len(parts) == 3
    total = sum(len(p) for p in parts)
    assert total == 10
    combined = Tafra.concat(parts)
    np.testing.assert_array_equal(combined['x'], t['x'])


def test_chunks_sorted() -> None:
    """chunks() with sort_by sorts before splitting."""
    t = Tafra({'x': np.array([3, 1, 2, 5, 4]), 'v': np.arange(5.0)})
    parts = t.chunks(2, sort_by=['x'])
    combined = Tafra.concat(parts)
    np.testing.assert_array_equal(combined['x'], np.array([1, 2, 3, 4, 5]))


def test_chunk_rows() -> None:
    """chunk_rows() splits by max row count."""
    t = Tafra({'x': np.arange(10)})
    parts = t.chunk_rows(3)
    assert len(parts) == 4
    assert all(len(p) <= 3 for p in parts)
    assert sum(len(p) for p in parts) == 10


def test_partition_basic() -> None:
    """partition() splits by group values."""
    t = Tafra({
        'g': np.array([1, 2, 1, 2, 3]),
        'v': np.array([10., 20., 30., 40., 50.]),
    })
    parts = t.partition(['g'])
    assert len(parts) == 3
    keys = [k for k, _ in parts]
    assert keys == [(1,), (2,), (3,)]
    # group 1 rows
    np.testing.assert_array_equal(parts[0][1]['v'], np.array([10., 30.]))
    # group 2 rows
    np.testing.assert_array_equal(parts[1][1]['v'], np.array([20., 40.]))


def test_partition_sorted() -> None:
    """partition() with sort_by sorts within each group."""
    t = Tafra({
        'g': np.array([1, 1, 1]),
        'v': np.array([30., 10., 20.]),
    })
    parts = t.partition(['g'], sort_by=['v'])
    np.testing.assert_array_equal(parts[0][1]['v'], np.array([10., 20., 30.]))


def test_concat_roundtrip() -> None:
    """concat(chunks()) reproduces the original data."""
    t = build_tafra()
    combined = Tafra.concat(t.chunks(3))
    for col in t.columns:
        np.testing.assert_array_equal(combined[col], t[col])


def test_concat_empty_raises() -> None:
    """concat() with empty list raises ValueError."""
    with pytest.raises(ValueError):
        Tafra.concat([])


def test_tail() -> None:
    t = Tafra({'x': np.arange(10)})
    assert len(t.tail(3)) == 3
    np.testing.assert_array_equal(t.tail(3)['x'], np.array([7, 8, 9]))


def test_sort() -> None:
    t = Tafra({'x': np.array([3, 1, 2]), 'y': np.array([30., 10., 20.])})
    s = t.sort('x')
    np.testing.assert_array_equal(s['x'], np.array([1, 2, 3]))
    np.testing.assert_array_equal(s['y'], np.array([10., 20., 30.]))

    r = t.sort('x', reverse=True)
    np.testing.assert_array_equal(r['x'], np.array([3, 2, 1]))


def test_sort_multi_col() -> None:
    t = Tafra({
        'a': np.array([1, 1, 2, 2]),
        'b': np.array([20., 10., 40., 30.]),
    })
    s = t.sort(['a', 'b'])
    np.testing.assert_array_equal(s['b'], np.array([10., 20., 30., 40.]))


def test_sample() -> None:
    t = Tafra({'x': np.arange(100)})
    s = t.sample(10, seed=42)
    assert len(s) == 10
    # reproducible
    s2 = t.sample(10, seed=42)
    np.testing.assert_array_equal(s['x'], s2['x'])

    with pytest.raises(ValueError):
        t.sample(200)


def test_drop_duplicates() -> None:
    t = Tafra({
        'x': np.array([1, 2, 1, 2, 3]),
        'y': np.array([10., 20., 10., 20., 30.]),
    })
    d = t.drop_duplicates(['x'])
    assert len(d) == 3
    np.testing.assert_array_equal(d['x'], np.array([1, 2, 3]))

    # all columns
    d2 = t.drop_duplicates()
    assert len(d2) == 3


def test_drop_duplicates_string() -> None:
    """drop_duplicates works with StringDType columns."""
    t = Tafra({
        'g': np.array(['a', 'b', 'a', 'b'], dtype=np.dtypes.StringDType()),
        'v': np.array([1, 2, 3, 4]),
    })
    d = t.drop_duplicates(['g'])
    assert len(d) == 2


def test_value_counts() -> None:
    t = Tafra({
        'x': np.array([1, 2, 1, 1, 2, 3]),
    })
    vc = t.value_counts('x')
    assert len(vc) == 3
    # sorted by count descending
    assert vc['count'][0] == 3  # x=1 appears 3 times
    assert vc['x'][0] == 1


def test_describe() -> None:
    t = Tafra({
        'x': np.array([1., 2., 3., 4., 5.]),
        'name': np.array(['a', 'b', 'c', 'd', 'e'], dtype=np.dtypes.StringDType()),
    })
    d = t.describe()
    assert 'stat' in d.columns
    assert 'x' in d.columns
    assert 'name' not in d.columns  # non-numeric excluded
    assert len(d) == 8
    # check mean
    mean_row = np.where(d['stat'] == 'mean')[0][0]
    assert d['x'][mean_row] == 3.0


def test_shift_forward() -> None:
    t = Tafra({
        'x': np.array([1., 2., 3., 4., 5.]),
    })
    s = t.shift(1)
    assert np.isnan(s['x'][0])
    np.testing.assert_array_equal(s['x'][1:], np.array([1., 2., 3., 4.]))


def test_shift_backward() -> None:
    t = Tafra({
        'x': np.array([1., 2., 3., 4., 5.]),
    })
    s = t.shift(-1)
    assert np.isnan(s['x'][-1])
    np.testing.assert_array_equal(s['x'][:-1], np.array([2., 3., 4., 5.]))


def test_shift_zero() -> None:
    t = Tafra({'x': np.array([1., 2., 3.])})
    s = t.shift(0)
    np.testing.assert_array_equal(s['x'], t['x'])


def test_shift_string_column() -> None:
    """shift fills string columns with None."""
    t = Tafra({
        'name': np.array(['a', 'b', 'c'], dtype=np.dtypes.StringDType()),
    })
    s = t.shift(1)
    assert s['name'][0] is None


def test_vectorized_std_var() -> None:
    """Vectorized std/var match per-group numpy results."""
    t = Tafra({
        'g': np.array([1, 1, 1, 2, 2]),
        'v': np.array([10., 20., 30., 40., 50.]),
    })
    gb = t.group_by(['g'], {'s': (np.std, 'v'), 'va': (np.var, 'v')})
    np.testing.assert_almost_equal(gb['s'][0], np.std([10., 20., 30.]))
    np.testing.assert_almost_equal(gb['va'][0], np.var([10., 20., 30.]))
    np.testing.assert_almost_equal(gb['s'][1], np.std([40., 50.]))


def test_vectorized_prod() -> None:
    t = Tafra({
        'g': np.array([1, 1, 2, 2]),
        'v': np.array([2., 3., 4., 5.]),
    })
    gb = t.group_by(['g'], {'p': (np.prod, 'v')})
    assert gb['p'][0] == 6.0
    assert gb['p'][1] == 20.0


def test_vectorized_any_all() -> None:
    t = Tafra({
        'g': np.array([1, 1, 2, 2]),
        'v': np.array([True, False, True, True]),
    })
    gb = t.group_by(['g'], {'a': (np.any, 'v'), 'b': (np.all, 'v')})
    assert gb['a'][0]
    assert not gb['b'][0]
    assert gb['a'][1]
    assert gb['b'][1]


def test_vectorized_median() -> None:
    t = Tafra({
        'g': np.array([1, 1, 1, 2, 2]),
        'v': np.array([30., 10., 20., 50., 40.]),
    })
    gb = t.group_by(['g'], {'m': (np.median, 'v')})
    assert gb['m'][0] == 20.0
    assert gb['m'][1] == 45.0


def test_vectorized_ptp() -> None:
    t = Tafra({
        'g': np.array([1, 1, 2, 2]),
        'v': np.array([10., 30., 40., 50.]),
    })
    gb = t.group_by(['g'], {'r': (np.ptp, 'v')})
    assert gb['r'][0] == 20.0
    assert gb['r'][1] == 10.0


def test_percentile_agg() -> None:
    from tafra import percentile
    t = Tafra({
        'g': np.array([1, 1, 1, 1, 1]),
        'v': np.array([10., 20., 30., 40., 50.]),
    })
    gb = t.group_by(['g'], {
        'p50': (percentile(50), 'v'),
        'p90': (percentile(90), 'v'),
    })
    np.testing.assert_almost_equal(gb['p50'][0], np.percentile([10., 20., 30., 40., 50.], 50))
    np.testing.assert_almost_equal(gb['p90'][0], np.percentile([10., 20., 30., 40., 50.], 90))


def test_geomean_harmean() -> None:
    from tafra import geomean, harmean
    t = Tafra({
        'g': np.array([1, 1, 1]),
        'v': np.array([2., 4., 8.]),
    })
    gb = t.group_by(['g'], {'geo': (geomean, 'v'), 'har': (harmean, 'v')})
    np.testing.assert_almost_equal(gb['geo'][0], (2. * 4. * 8.) ** (1./3.))
    np.testing.assert_almost_equal(gb['har'][0], 3. / (1./2. + 1./4. + 1./8.))


# ================================================================
# C extension direct tests
# ================================================================

try:
    from tafra._accel import composite_key, group_indices, encode_strings
    _HAS_ACCEL = True
except ImportError:
    _HAS_ACCEL = False


@pytest.mark.skipif(not _HAS_ACCEL, reason='C extension not available')
class TestCompositeKey:
    def test_single_column(self) -> None:
        r = composite_key((np.array([0, 1, 2], dtype=np.int64),), (3,))
        np.testing.assert_array_equal(r, [0, 1, 2])

    def test_two_columns(self) -> None:
        a = np.array([0, 1, 2], dtype=np.int64)
        b = np.array([0, 1, 0], dtype=np.int64)
        r = composite_key((a, b), (3, 2))
        np.testing.assert_array_equal(r, [0, 3, 4])

    def test_three_columns(self) -> None:
        a = np.array([0, 1, 0], dtype=np.int64)
        b = np.array([0, 0, 1], dtype=np.int64)
        c = np.array([0, 1, 1], dtype=np.int64)
        r = composite_key((a, b, c), (2, 2, 2))
        np.testing.assert_array_equal(r, [0, 5, 3])

    def test_empty(self) -> None:
        r = composite_key((np.array([], dtype=np.int64),), (1,))
        assert len(r) == 0

    def test_single_element(self) -> None:
        r = composite_key(
            (np.array([2], dtype=np.int64), np.array([1], dtype=np.int64)),
            (3, 2))
        np.testing.assert_array_equal(r, [5])


@pytest.mark.skipif(not _HAS_ACCEL, reason='C extension not available')
class TestGroupIndices:
    def test_empty(self) -> None:
        fs, gl, ng = group_indices(np.array([], dtype=np.int64))
        assert ng == 0
        assert len(fs) == 0
        assert len(gl) == 0

    def test_single_element(self) -> None:
        fs, gl, ng = group_indices(np.array([42], dtype=np.int64))
        assert ng == 1
        np.testing.assert_array_equal(gl[0], [0])

    def test_all_same(self) -> None:
        fs, gl, ng = group_indices(np.array([5, 5, 5, 5], dtype=np.int64))
        assert ng == 1
        np.testing.assert_array_equal(gl[0], [0, 1, 2, 3])

    def test_all_unique(self) -> None:
        fs, gl, ng = group_indices(np.array([10, 20, 30], dtype=np.int64))
        assert ng == 3
        for i in range(3):
            np.testing.assert_array_equal(gl[i], [i])

    def test_first_seen_order(self) -> None:
        fs, gl, ng = group_indices(np.array([3, 1, 3, 2, 1], dtype=np.int64))
        assert ng == 3
        # First-seen: 3 at idx 0, 1 at idx 1, 2 at idx 3
        np.testing.assert_array_equal(fs, [0, 1, 3])
        np.testing.assert_array_equal(gl[0], [0, 2])  # group 3
        np.testing.assert_array_equal(gl[1], [1, 4])  # group 1
        np.testing.assert_array_equal(gl[2], [3])      # group 2

    def test_large_group_count(self) -> None:
        """Triggers realloc of first_seen/counts arrays."""
        keys = np.arange(1000, dtype=np.int64)
        fs, gl, ng = group_indices(keys)
        assert ng == 1000
        for i in range(1000):
            np.testing.assert_array_equal(gl[i], [i])


@pytest.mark.skipif(not _HAS_ACCEL, reason='C extension not available')
class TestEncodeStrings:
    def test_empty(self) -> None:
        codes, nu = encode_strings(np.array([], dtype=object))
        assert nu == 0
        assert len(codes) == 0

    def test_single(self) -> None:
        codes, nu = encode_strings(np.array(['a'], dtype=object))
        assert nu == 1
        np.testing.assert_array_equal(codes, [0])

    def test_all_same(self) -> None:
        codes, nu = encode_strings(np.array(['x', 'x', 'x'], dtype=object))
        assert nu == 1
        np.testing.assert_array_equal(codes, [0, 0, 0])

    def test_first_seen_order(self) -> None:
        codes, nu = encode_strings(np.array(['b', 'a', 'c', 'a', 'b'], dtype=object))
        assert nu == 3
        # b=0, a=1, c=2 (first-seen order)
        np.testing.assert_array_equal(codes, [0, 1, 2, 1, 0])

    def test_high_cardinality(self) -> None:
        arr = np.array([f'str_{i}' for i in range(10000)], dtype=object)
        codes, nu = encode_strings(arr)
        assert nu == 10000
        assert len(set(codes)) == 10000

    def test_integers_as_objects(self) -> None:
        """encode_strings works with any hashable objects, not just strings."""
        arr = np.array([1, 2, 1, 3, 2], dtype=object)
        codes, nu = encode_strings(arr)
        assert nu == 3
        np.testing.assert_array_equal(codes, [0, 1, 0, 2, 1])
