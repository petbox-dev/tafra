import warnings

import numpy as np
from tafra import Tafra
import pandas as pd  # type: ignore

from typing import Dict, List, Any

import pytest  # type: ignore
from unittest.mock import MagicMock

print = MagicMock()

def build_tafra() -> Tafra:
    return Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })

def test_constructions() -> None:
    with pytest.raises(TypeError) as e:
        t = Tafra()  # type: ignore # noqa

    with pytest.raises(ValueError) as e:
        t = Tafra({})

    with pytest.raises(ValueError) as e:
        t = Tafra({'x': None})  # type: ignore

    t = Tafra({'x': np.array(1)})
    t = Tafra({'x': np.array([1])})
    t = Tafra({'x': [True, False]})  # type: ignore
    t = Tafra({'x': 'test'})  # type: ignore
    t.update_dtypes({'x': 'O'})

    t = build_tafra()
    t.update_dtypes({'x': 'float'})
    t.data['x'][2] = np.nan
    _ = tuple(t.to_records())
    _ = tuple(t.to_records(columns='x'))
    _ = tuple(t.to_records(columns=['x']))
    _ = tuple(t.to_records(columns=['x', 'y']))
    _ = tuple(t.to_records(cast_null=False))
    _ = tuple(t.to_records(columns='x', cast_null=False))
    _ = tuple(t.to_records(columns=['x'], cast_null=False))
    _ = tuple(t.to_records(columns=['x', 'y'], cast_null=False))
    _ = t.to_list()
    _ = t.to_list(inner=True)
    _ = t.to_list(columns='x')
    _ = t.to_list(columns='x', inner=True)
    _ = t.to_list(columns=['x'])
    _ = t.to_list(columns=['x'], inner=True)
    _ = t.to_list(columns=['x', 'y'])
    _ = t.to_list(columns=['x', 'y'], inner=True)

    t = build_tafra()
    df = pd.DataFrame(t.data)
    _ = Tafra.from_series(df['x'])
    _ = Tafra.from_dataframe(df)
    # _ = Tafra.as_tafra(df['x'])
    _ = Tafra.as_tafra(df)
    _ = Tafra.as_tafra(t)
    _ = Tafra.as_tafra({'x': np.array(1)})

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

    _ = Tafra.as_tafra(Series())
    _ = Tafra.as_tafra(DataFrame())

    with pytest.raises(TypeError) as e:
        _ = Tafra(np.arange(6))

    with pytest.raises(TypeError) as e:
        _ = Tafra.as_tafra(np.arange(6))

def test_properties() -> None:
    t = build_tafra()
    _ = t.columns
    _ = t.rows
    _ = t.data
    _ = t.dtypes

    with pytest.raises(ValueError) as e:
        t.columns = ['x', 'a']  # type: ignore

    with pytest.raises(ValueError) as e:
        t.rows = 3

    with pytest.raises(ValueError) as e:
        t.data = {'x': np.arange(6)}

    with pytest.raises(ValueError) as e:
        t.dtypes = {'x': 'int'}

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

    with pytest.raises(ValueError) as e:
        t['x'] = np.arange(3)

def test_select() -> None:
    t = build_tafra()
    _ = t.select('x')
    _ = t.select(['x'])
    _ = t.select(['x', 'y'])

    with pytest.raises(ValueError) as e:
        _ = t.select('a')

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
    t2.union(t, inplace=True)
    assert len(t2) == 2 * len(t)

    t2 = build_tafra()
    _ = t2.union(t, inplace=False)
    assert len(_) == len(t) + len(t2)

def test_update_dtypes() -> None:
    t = build_tafra()
    t.update_dtypes({'x': float})
    assert t['x'].dtype == 'float'
    assert isinstance(t['x'][0], np.float)

    t = build_tafra()
    _ = t.update_dtypes({'x': float}, inplace=False)
    assert _['x'].dtype == 'float'
    assert isinstance(_['x'][0], np.float)

def test_rename() -> None:
    t = build_tafra()
    t.rename({'x': 'a'})
    assert 'a' in t.data
    assert 'x' not in t.data

    t = build_tafra()
    _ = t.rename({'x': 'a'}, inplace=False)
    assert 'a' in _.data
    assert 'x' not in _.data

def test_delete() -> None:
    t = build_tafra()
    t.delete('x')
    assert 'x' not in t.data

    t = build_tafra()
    t.delete(['x'])
    assert 'x' not in t.data

    t = build_tafra()
    t.delete(['x', 'y'])
    assert 'x' not in t.data
    assert 'y' not in t.data

    t = build_tafra()
    _ = t.delete('x', inplace=False)
    assert 'x' not in _.data

    t = build_tafra()
    _ = t.delete(['x'], inplace=False)
    assert 'x' not in _.data

    t = build_tafra()
    _ = t.delete(['x', 'y'], inplace=False)
    assert 'x' not in _.data
    assert 'y' not in _.data

def test_groupby() -> None:
    t = build_tafra()
    gb = t.group_by(
        ['y', 'z'], {'x': sum}, {'count': len}
    )

def test_groupby_iter_fn() -> None:
    t = build_tafra()
    gb = t.group_by(
        ['y', 'z'], {
            'x': sum,
            'new_x': (sum, 'x')
        }, {'count': len}
    )

def test_transform() -> None:
    t = build_tafra()
    tr = t.transform(
        ['y', 'z'], {'x': sum}, {'id': max}
    )

def test_iterate_by_attr() -> None:
    t = build_tafra()
    t.id = np.empty(t.rows, dtype=int)  # type: ignore
    t['id'] = np.empty(t.rows, dtype=int)
    for i, (u, ix, grouped) in enumerate(t.iterate_by(['y', 'z'])):
        t['x'][ix] = sum(grouped['x'])
        t.id[ix] = len(grouped['x'])  # type: ignore
        t['id'][ix] = max(grouped['x'])

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

def test_transform_assignment() -> None:
    t = build_tafra()
    for u, ix, it in t.iterate_by(['y']):
        it['x'][0] = 9

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

def test_union() -> None:
    t = build_tafra()
    t2 = build_tafra()
    t.union(t2)

    t = build_tafra()
    t2 = build_tafra()
    t._dtypes['a'] = 'int'
    with pytest.raises(Exception) as e:
        t.union(t2)

    t = build_tafra()
    t2._dtypes['a'] = 'int'
    with pytest.raises(Exception) as e:
        t.union(t2)

    t = build_tafra()
    t2 = build_tafra()
    t['a'] = np.arange(6)
    with pytest.raises(ValueError) as e:
        t.union(t2)

    t = build_tafra()
    t2 = build_tafra()
    t2['a'] = np.arange(6)
    with pytest.raises(ValueError) as e:
        t.union(t2)

    t = build_tafra()
    t2 = build_tafra()
    t.rename({'x': 'a'})
    with pytest.raises(TypeError) as e:
        t.union(t2)

    t = build_tafra()
    t2 = build_tafra()
    t2.rename({'x': 'a'})
    with pytest.raises(TypeError) as e:
        t.union(t2)

    t = build_tafra()
    t2 = build_tafra()
    t.update_dtypes({'x': float}, inplace=True)
    with pytest.raises(TypeError) as e:
        t.union(t2)

    t = build_tafra()
    t2 = build_tafra()
    t2._dtypes['x'] = 'float'
    with pytest.raises(TypeError) as e:
        t.union(t2)

def test_slice() -> None:
    t = build_tafra()

    _ = t[:3]
    _['x'][0] = 0

    _ = t[slice(0, 3)]
    _['x'][0] = 7

    _ = t[:3].copy()
    _['x'][0] = 9
    t['x']

    _ = t[t['x'] <= 4]
    _['x'][1] = 15

    _ = t[2]
    _ = t[[1, 3]]
    _ = t[np.array([2, 4])]
    _ = t[[True, False, True, True, False, True]]
    _ = t[np.array([True, False, True, True, False, True])]

    with pytest.raises(IndexError) as e:
        _ = t[[True, False]]

    with pytest.raises(IndexError) as e:
        _ = t[np.array([True, False])]

    with pytest.raises(TypeError) as e:
        _ = t[(1, 2)]  # type: ignore # noqa

    with pytest.raises(TypeError) as e:
        _ = t[{'x': [1, 2]}]  # type: ignore # noqa

    class TestClass:
        ...

    with pytest.raises(TypeError) as e:
        _ = t[TestClass()]  # type: ignore # noqa

    with pytest.raises(IndexError) as e:
        _ = t[[1, 2.]]  # type: ignore

    with pytest.raises(IndexError) as e:
        _ = t[np.array([1, 2.])]

    with pytest.raises(IndexError) as e:
        _ = t[np.array([[1, 2]])]

def test_invalid_dtypes() -> None:
    t = build_tafra()
    with pytest.raises(Exception) as e:
        t.update_dtypes({'x': 'flot', 'y': 'st'})

def test_invalid_assignment() -> None:
    t = build_tafra()
    o = build_tafra()
    o._data['x'] = np.arange(5)

    with pytest.raises(Exception) as e:
        o.__post_init__()

    with pytest.raises(Exception) as e:
        t.update(o)

    with warnings.catch_warnings(record=True) as w:
        t['x'] = list(range(6))
        t['x'] = np.arange(6)[:, None]
        t['x'] = np.atleast_2d(np.arange(6))
        t['x'] = np.atleast_2d(np.arange(6)).T
        t['x'] = np.atleast_2d(np.arange(6))

        with pytest.raises(Exception) as e:
            t['x'] = np.repeat(np.arange(6)[:, None], repeats=2, axis=1)

def test_datetime() -> None:
    t = build_tafra()
    t['d'] = np.array([np.datetime64(_, 'D') for _ in range(6)])
    t.update_dtypes({'d': '<M8[D]'})

    _ = tuple(t.to_records())

    _ = t.to_list()

def test_coalesce() -> None:
    t = Tafra({'x': np.array([1, 2, None, 4, None])})
    t['x'] = t.coalesce('x', [[1, 2, 3, None, 5], [None, None, None, None, 'five']])  # type: ignore
    t['y'] = t.coalesce('y', [[1, 2, 3, None, 5], [None, None, None, None, 'five']])  # type: ignore
    assert np.all(t['x'] != np.array(None))
    assert t['y'][3] == np.array(None)

    t = Tafra({'x': np.array([1, 2, None, 4, None])})
    t.coalesce('x', [[1, 2, 3, None, 5], [None, None, None, None, 'five']], inplace=True)  # type: ignore
    t.coalesce('y', [[1, 2, 3, None, 5], [None, None, None, None, 'five']], inplace=True)  # type: ignore
    assert np.all(t['x'] != np.array(None))
    assert t['y'][3] == np.array(None)

    t = Tafra({'x': np.array([None])})
    t.coalesce('x', [[1], [None]])  # type: ignore

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

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([2, 2, 2, 3, 3, 3])
    })
    t = l.left_join(r, [('x', 'a', '=='), ('z', 'c', '==')], ['x', 'y', 'a', 'b'])

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        '_a': np.array([1, 1, 2, 2, 3, 3]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.left_join(r, [('x', 'a', '=='), ('x', '_a', '==')], ['x', 'y', 'a', 'b'])

    r = Tafra({
        'a': np.array([1, 1, 2, 2, 3, 3]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.left_join(r, [('x', 'a', '<')], ['x', 'y', 'a', 'b'])

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

    r = Tafra({
        'a': np.array([1, 1, 2, 2, 3, 3]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.inner_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.inner_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })

    t = l.inner_join(r, [('x', 'a', '<=')], ['x', 'y', 'a', 'b'])


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

    r = Tafra({
        'a': np.array([1, 1, 2, 2, 3, 3]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.cross_join(r)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.cross_join(r)

    r = Tafra({
        'a': np.array([1, 1, 1, 2, 2, 2]),
        'b': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'c': np.array([0, 0, 0, 1, 1, 1])
    })

    t = l.cross_join(r)
