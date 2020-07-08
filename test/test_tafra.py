from pathlib import Path
import platform
import warnings
from decimal import Decimal
from datetime import date, datetime

import numpy as np
from tafra import Tafra, object_formatter
import pandas as pd  # type: ignore
from itertools import islice

from typing import Dict, List, Any, Iterator, Iterable, Sequence, Tuple, Optional, Type

import pytest  # type: ignore
from unittest.mock import MagicMock


class TestClass:
    ...


class Series:
    name: str = 'x'
    values: np.ndarray = np.arange(5)
    dtype: str = 'int'


class DataFrame:
    _data: Dict[str, Series] = {'x': Series(), 'y': Series()}
    columns: List[str] = ['x', 'y']
    dtypes: List[str] = ['int', 'int']

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

    def __iter__(self) -> Iterator[Tuple[Any, ...]]:
        return self

    def __next__(self) -> Tuple[Any, ...]:
        try:
            item = self._iter[self.idx]
        except IndexError:
            raise StopIteration()
        self.idx += 1
        return item

    def execute(self, sql: str) -> None:
        ...

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        try:
            return next(self)
        except:
            return None

    def fetchmany(self, size: int) -> List[Tuple[Any, ...]]:
        return list(islice(self, size))

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return [rec for rec in self]


def build_tafra() -> Tafra:
    return Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })


def check_tafra(t: Tafra) -> bool:
    assert len(t._data) == len(t._dtypes)
    for c in t.columns:
        assert isinstance(t[c], np.ndarray)
        assert isinstance(t.data[c], np.ndarray)
        assert isinstance(t._data[c], np.ndarray)
        assert isinstance(t.dtypes[c], str)
        assert isinstance(t._dtypes[c], str)
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
    write_path = Path('test/test_to_csv.csv')
    t.to_csv(write_path)
    # t.to_csv(write_path, columns=columns)
    assert isinstance(df, pd.DataFrame)

    return True

def test_constructions() -> None:
    t = build_tafra()
    check_tafra(t)

    t = Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'z': np.array([0, 0, 0, 1, 1, 1])
    }, validate=False)
    check_tafra(t)

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

    def iterator() -> Iterator[Dict[str, np.ndarray]]:
        yield {'x': np.array([1, 2, 3, 4, 5, 6])}
        yield {'y': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object')}
        yield {'z': np.array([0, 0, 0, 1, 1, 1])}

    t = Tafra(iterator())
    check_tafra(t)

    class DictIterable:
        def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
            yield {'x': np.array([1, 2, 3, 4, 5, 6])}
            yield {'y': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object')}
            yield {'z': np.array([0, 0, 0, 1, 1, 1])}

    t = Tafra(DictIterable())
    check_tafra(t)

    t = Tafra(iter(DictIterable()))
    check_tafra(t)

    class SequenceIterable:
        def __iter__(self) -> Iterator[Any]:
            yield ('x', np.array([1, 2, 3, 4, 5, 6]))
            yield ['y', np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object')]
            yield ('z', np.array([0, 0, 0, 1, 1, 1]))

    t = Tafra(SequenceIterable())
    check_tafra(t)

    class SequenceIterable2:
        def __iter__(self) -> Iterator[Any]:
            yield (np.array(['x']), np.array([1, 2, 3, 4, 5, 6]))
            yield [np.array(['y']), np.array(['one', 'two', 'one', 'two', 'one', 'two'],
                                             dtype='object')]
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
    def gen_values() -> Iterator[Dict[str, np.ndarray]]:
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
    if platform.system() == 'Windows':
        assert t._dtypes['x'] == 'int32'
    elif platform.system() == 'Linux':
        assert t._dtypes['x'] == 'int64'
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
    _ = t2.union(t)
    check_tafra(_)

    t2.union_inplace(t)
    check_tafra(t2)
    assert len(t2) == 2 * len(t)

    t2 = build_tafra()
    _ = t2.union(t)
    check_tafra(_)
    assert len(_) == len(t) + len(t2)

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
    assert isinstance(t['x'][0], np.float)

    t = build_tafra()
    _ = t.update_dtypes({'x': float})
    check_tafra(_)
    assert _['x'].dtype == 'float'
    assert isinstance(_['x'][0], np.float)

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
    _ = list(t.row_map(np.repeat, 6))
    _ = list(t.tuple_map(np.repeat, 6))
    _ = list(t.col_map(np.repeat, repeats=6))
    _ = Tafra(t.key_map(np.repeat, repeats=6))

def test_union() -> None:
    t = build_tafra()
    t2 = build_tafra()
    t.union_inplace(t2)
    check_tafra(t)

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
        'y': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })

    r = Tafra({
        'a': np.array([1, 2, 3, 4, 5, 6]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.left_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([2, 2, 2, 3, 3, 3])
    })
    t = l.left_join(r, [('x', 'a', '=='), ('z', 'c', '==')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        '_a': np.array([1, 1, 2, 2, 3, 3]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.left_join(r, [('x', 'a', '=='), ('x', '_a', '==')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 2, 2, 3, 3]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.left_join(r, [('x', 'a', '<')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

def test_inner_join() -> None:
    l = Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })

    r = Tafra({
        'a': np.array([1, 2, 3, 4, 5, 6]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.inner_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 2, 2, 3, 3]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.inner_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.inner_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })

    t = l.inner_join(r, [('x', 'a', '<=')], ['x', 'y', 'a', 'b'])
    check_tafra(t)


def test_cross_join() -> None:
    l = Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })

    r = Tafra({
        'a': np.array([1, 2, 3, 4, 5, 6]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.cross_join(r)
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 2, 2, 3, 3]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.cross_join(r)
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.cross_join(r)
    check_tafra(t)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
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
        'y': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })

    r = Tafra({
        'a': np.array([1, 2, 3, 4, 5, 6]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })

    with pytest.raises(TypeError) as e:
        t = l.left_join(r, [('x', 'a', '===')], ['x', 'y', 'a', 'b'])

    r = Tafra({
        'a': np.array([1, 2, 3, 4, 5, 6], dtype='float'),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })

    with pytest.raises(TypeError) as e:
        t = l.left_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])

    r = Tafra({
        'a': np.array([1, 2, 3, 4, 5, 6]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })

    l._dtypes['x'] = 'float'
    with pytest.raises(TypeError) as e:
        t = l.left_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])

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
    assert t.dtypes['b'] == 'object'
    assert t.dtypes['b (2)'] == 'float64'
    assert t.rows == 6
    assert len(t.columns) == 3
    check_tafra(t)
    write_reread(t)

    # as above, but with a promotion required after inference period
    #   (heuristic fails)
    t = Tafra.read_csv('test/ex4.csv')
    assert t.dtypes['a'] == 'int32'
    assert t.dtypes['b'] == 'object'
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

    # missing column, override dtypes
    t = Tafra.read_csv('test/ex6.csv')
    assert t.dtypes['dp'] == 'float64'
    assert t.dtypes['dp_prime'] == 'object'
    assert t.dtypes['dp_prime_te'] == 'object'
    assert t.dtypes['t'] == 'float64'
    assert t.dtypes['te'] == 'float64'
    check_tafra(t)

    t.update_dtypes_inplace({'dp_prime': float, 'dp_prime_te': 'float64'})
    assert t.dtypes['dp_prime'] == 'float64'
    assert t.dtypes['dp_prime_te'] == 'float64'
    check_tafra(t)

    t = Tafra.read_csv('test/ex6.csv', dtypes={'dp_prime': np.float, 'dp_prime_te': np.float32})
    assert t.dtypes['dp'] == 'float64'
    assert t.dtypes['dp_prime'] == 'float64'
    assert t.dtypes['dp_prime_te'] == 'float32'
    assert t.dtypes['t'] == 'float64'
    assert t.dtypes['te'] == 'float64'
    check_tafra(t)

    # override a column type
    t = Tafra.read_csv('test/ex4.csv', dtypes={'a': 'float32'})
    assert t.dtypes['a'] == 'float32'
    assert t.dtypes['b'] == 'object'
    assert t.dtypes['b (2)'] == 'float64'
    assert t.rows == 6
    assert len(t.columns) == 3
    check_tafra(t)
    write_reread(t)
