# Iris Dataset

This tutorial walks through a complete analysis of the classic Iris flower
dataset using Tafra. We construct the data from inline arrays, explore columns,
compute basic statistics, and use `group_by` to aggregate by species.

## Load the data

We use a 15-row subset of the Iris dataset (5 per species) to keep output
readable.

```python
import numpy as np
from tafra import Tafra

iris = Tafra({
    'sepal_length': np.array([5.1, 4.9, 4.7, 5.0, 5.4,
                               7.0, 6.4, 6.9, 5.5, 6.5,
                               6.3, 5.8, 7.1, 6.3, 6.5]),
    'sepal_width':  np.array([3.5, 3.0, 3.2, 3.6, 3.9,
                               3.2, 3.2, 3.1, 2.3, 2.8,
                               3.3, 2.7, 3.0, 2.9, 3.0]),
    'petal_length': np.array([1.4, 1.4, 1.3, 1.4, 1.7,
                               4.7, 4.5, 4.9, 4.0, 4.6,
                               6.0, 5.1, 5.9, 5.6, 5.8]),
    'petal_width':  np.array([0.2, 0.2, 0.2, 0.2, 0.4,
                               1.4, 1.5, 1.5, 1.3, 1.5,
                               2.5, 1.9, 2.1, 1.8, 1.8]),
    'species':      np.array(['setosa', 'setosa', 'setosa', 'setosa', 'setosa',
                               'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                               'virginica', 'virginica', 'virginica', 'virginica', 'virginica']),
})

print(f'Rows: {iris.rows}')
print(f'Columns: {list(iris.columns)}')
```

Output:

```
Rows: 15
Columns: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
```

## Column access

Accessing a column returns the underlying `np.ndarray` directly -- no wrapper
objects, no copies:

```python
lengths = iris['sepal_length']
print(type(lengths))  # <class 'numpy.ndarray'>
print(lengths[:5])    # [5.1 4.9 4.7 5.  5.4]
```

You can use any numpy operation on the result:

```python
print(f'Mean sepal length: {np.mean(iris["sepal_length"]):.2f}')
print(f'Std sepal width:   {np.std(iris["sepal_width"]):.2f}')
print(f'Min petal length:  {np.min(iris["petal_length"]):.2f}')
print(f'Max petal width:   {np.max(iris["petal_width"]):.2f}')
```

Output:

```
Mean sepal length: 5.97
Std sepal width:   0.36
Min petal length:  1.30
Max petal width:   2.50
```

## Group by species -- basic aggregation

`group_by` produces one row per unique value in the group columns.
The aggregation dict maps output column names to functions. When the output
name matches an existing column, the function is applied to that column:

```python
summary = iris.group_by(
    ['species'],
    {
        'sepal_length': np.mean,
        'sepal_width': np.mean,
        'petal_length': np.mean,
        'petal_width': np.mean,
    },
)

for col in summary.columns:
    vals = summary[col]
    if vals.dtype.kind == 'f':
        print(f'{col:14s}  {vals[0]:6.2f}  {vals[1]:6.2f}  {vals[2]:6.2f}')
    else:
        print(f'{col:14s}  {str(vals[0]):>6s}  {str(vals[1]):>6s}  {str(vals[2]):>6s}')
```

Output:

```
species        setosa  versicolor  virginica
sepal_length     5.02    6.46    6.40
sepal_width      3.44    2.92    2.98
petal_length     1.44    4.54    5.68
petal_width      0.24    1.44    2.02
```

## Renaming output columns

Use the `(function, source_column)` tuple form to give aggregated columns
new names:

```python
stats = iris.group_by(
    ['species'],
    {
        'mean_sepal_len': (np.mean, 'sepal_length'),
        'std_sepal_len':  (np.std, 'sepal_length'),
        'mean_petal_len': (np.mean, 'petal_length'),
        'std_petal_len':  (np.std, 'petal_length'),
        'count':          (len, 'sepal_length'),
    },
)

print(f'Columns: {list(stats.columns)}')
print(f'Rows:    {stats.rows}')
print()
print(f'{"species":>12s}  {"mean_sl":>8s}  {"std_sl":>8s}  {"mean_pl":>8s}  {"std_pl":>8s}  {"n":>3s}')
for i in range(stats.rows):
    print(f'{str(stats["species"][i]):>12s}'
          f'  {stats["mean_sepal_len"][i]:8.2f}'
          f'  {stats["std_sepal_len"][i]:8.3f}'
          f'  {stats["mean_petal_len"][i]:8.2f}'
          f'  {stats["std_petal_len"][i]:8.3f}'
          f'  {stats["count"][i]:3.0f}')
```

Output:

```
Columns: ['species', 'mean_sepal_len', 'std_sepal_len', 'mean_petal_len', 'std_petal_len', 'count']
Rows:    3

     species  mean_sl    std_sl  mean_pl    std_pl    n
      setosa      5.02     0.228      1.44     0.141    5
  versicolor      6.46     0.506      4.54     0.296    5
   virginica      6.40     0.418      5.68     0.327    5
```

## Iterating by species

`iterate_by` yields `(keys, indices, sub_tafra)` tuples -- useful when you
need full control over per-group processing:

```python
for keys, indices, sub in iris.iterate_by(['species']):
    species = keys[0]
    sl = sub['sepal_length']
    print(f'{species}: n={sub.rows}, sepal_length range=[{sl.min():.1f}, {sl.max():.1f}]')
```

Output:

```
setosa: n=5, sepal_length range=[4.7, 5.4]
versicolor: n=5, sepal_length range=[5.5, 7.0]
virginica: n=5, sepal_length range=[5.8, 7.1]
```

## Multiple group columns

You can group by more than one column. Here we add a size category and
group by both species and size:

```python
iris['size'] = np.where(iris['sepal_length'] > 6.0, 'large', 'small')

multi = iris.group_by(
    ['species', 'size'],
    {
        'mean_petal_len': (np.mean, 'petal_length'),
        'count': (len, 'petal_length'),
    },
)

for i in range(multi.rows):
    print(f'{str(multi["species"][i]):>12s}  {str(multi["size"][i]):>5s}'
          f'  mean_petal={multi["mean_petal_len"][i]:.2f}'
          f'  n={multi["count"][i]:.0f}')
```

Output:

```
      setosa  small  mean_petal=1.44  n=5
  versicolor  small  mean_petal=4.00  n=1
  versicolor  large  mean_petal=4.68  n=4
   virginica  large  mean_petal=5.68  n=5
```

## Summary

This tutorial covered:

- Constructing a `Tafra` from a dict of numpy arrays
- Accessing columns (returns `ndarray` directly)
- Computing statistics with standard numpy functions
- `group_by` with same-name and renamed output columns
- `iterate_by` for per-group processing
- Grouping by multiple columns
