import warnings

import numpy as np
from tafra import Tafra
import pandas as pd  # type: ignore

import pytest  # type: ignore
from unittest.mock import MagicMock

print = MagicMock()

def build_tafra() -> Tafra:
    return Tafra({
        'x': np.array([1, 2, 3, 4, 5, 6]),
        'y': np.array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='object'),
        'z': np.array([0, 0, 0, 1, 1, 1])
    })


def test_to_from_dataframe() -> None:
    t = build_tafra()
    df = pd.DataFrame(t.data)
    Tafra.as_tafra(df)

def test_prints() -> None:
    t = build_tafra()
    t.pformat()
    t.pprint()
    t.head(5)

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
    t.id = np.empty(t.rows, dtype=int)
    t['id'] = np.empty(t.rows, dtype=int)
    for i, (u, ix, grouped) in enumerate(t.iterate_by(['y', 'z'])):
        t['x'][ix] = sum(grouped['x'])
        t.id[ix] = len(grouped['x'])
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
        'c': np.array([0, 0, 0, 1, 1, 1])
    })
    t = l.left_join(r, [('x', 'a', '==')], ['x', 'y', 'a', 'b'])

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

# def test_mismatch() -> None:
#     t = build_tafra()
#     with pytest.raises(Exception) as e:
#         t.union(_t)

def test_update_dtypes() -> None:
    t = build_tafra()
    t2 = build_tafra()
    t3 = build_tafra()
    t3.update_dtypes({'x': float})

    t.union(t2)

    with pytest.raises(Exception) as e:
        t.union(t3)

def test_update() -> None:
    t = build_tafra()
    t2 = build_tafra()
    t2.union(t, inplace=True)


def test_slice() -> None:
    t = build_tafra()
    x = t[:3]
    x['x'][0] = 0

    x = t[slice(0, 3)]
    x['x'][0] = 7

    z = t[:3].copy()
    z['x'][0] = 9
    t['x']

    a = t[t['x'] <= 4]
    a['x'][1] = 15

def test_invalid_dtypes() -> None:
    t = build_tafra()
    with pytest.raises(Exception) as e:
        t.update_dtypes({'x': 'flot', 'y': 'st'})

def test_update_2() -> None:
    t = build_tafra()
    t.update_dtypes({'x': int})
    o = t.copy()

    o.update_dtypes({'x': float})
    t.update(o)

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

    _ = tuple(t.to_records())
    _ = t.to_list()

def test_coalesce() -> None:
    t = Tafra({'x': np.array([1, 2, None, 4, None])})
    t['x'] = t.coalesce('x', [[1, 2, 3, None, 5], [None, None, None, None, 'five']])
