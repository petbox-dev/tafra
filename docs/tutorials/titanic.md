# Titanic Analysis

This tutorial demonstrates survival analysis on a Titanic-style dataset.
We cover missing-value handling with `ObjectFormatter`, grouped aggregation,
`transform` to broadcast group statistics back to every row, and `partition`
for parallel dispatch.

## Build the dataset

We construct an inline dataset with 20 passengers. Some `age` values are
missing (represented as `None` in the source, which numpy stores as `nan`).

```python
import numpy as np
from tafra import Tafra

titanic = Tafra({
    'pclass':   np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]),
    'sex':      np.array(['male', 'female', 'female', 'male', 'male', 'female',
                           'male', 'female', 'male', 'female', 'male', 'male', 'female',
                           'male', 'male', 'female', 'male', 'male', 'female', 'male']),
    'age':      np.array([38.0, 26.0, 35.0, 54.0, np.nan, 58.0,
                           34.0, 28.0, np.nan, 14.0, 36.0, 45.0, 30.0,
                           22.0, np.nan, np.nan, 19.0, 32.0, 16.0, 25.0]),
    'survived': np.array([0, 1, 1, 0, 0, 1,
                           0, 1, 0, 1, 0, 0, 1,
                           0, 0, 1, 0, 0, 1, 0]),
    'fare':     np.array([71.3, 71.3, 53.1, 51.9, 30.5, 26.6,
                           13.0, 13.0, 13.0, 13.0, 26.0, 26.0, 13.0,
                           7.9, 8.1, 7.7, 7.9, 7.9, 7.7, 8.1]),
})

print(f'Rows: {titanic.rows}')
print(f'Missing ages: {np.sum(np.isnan(titanic["age"]))}')
```

Output:

```
Rows: 20
Missing ages: 4
```

## Handle missing values

Replace `nan` ages with the class-specific median age. We use `transform`
to compute the per-class median and broadcast it to every row, then fill
missing values:

```python
class_medians = titanic.transform(
    ['pclass'],
    {
        'median_age': (np.nanmedian, 'age'),
    },
)

# class_medians has the same 20 rows as titanic, with median_age filled per class
missing = np.isnan(titanic['age'])
titanic['age'][missing] = class_medians['median_age'][missing]

print(f'Missing ages after fill: {np.sum(np.isnan(titanic["age"]))}')
print(f'Filled ages: {titanic["age"][np.array([4, 8, 14, 15])]}')
```

Output:

```
Missing ages after fill: 0
Filled ages: [38.  34.  22.  22. ]
```

The missing values were filled with their class median: 38.0 for class 1,
34.0 for class 2, and 22.0 for class 3.

## Survival rate by class

Use `group_by` with `np.mean` on the `survived` column (0/1 encoding) to
compute survival rates:

```python
by_class = titanic.group_by(
    ['pclass'],
    {
        'survival_rate': (np.mean, 'survived'),
        'mean_age':      (np.mean, 'age'),
        'mean_fare':     (np.mean, 'fare'),
        'count':         (len, 'survived'),
    },
)

for i in range(by_class.rows):
    print(f'Class {by_class["pclass"][i]}:'
          f'  survival={by_class["survival_rate"][i]:.2%}'
          f'  age={by_class["mean_age"][i]:.1f}'
          f'  fare={by_class["mean_fare"][i]:.1f}'
          f'  n={by_class["count"][i]:.0f}')
```

Output:

```
Class 1:  survival=50.00%  age=40.2  fare=50.8  n=6
Class 2:  survival=42.86%  age=31.6  fare=16.7  n=7
Class 3:  survival=28.57%  age=22.7  fare=7.9  n=7
```

## Survival rate by class and sex

Group by two columns to get a cross-tabulation:

```python
cross = titanic.group_by(
    ['pclass', 'sex'],
    {
        'survival_rate': (np.mean, 'survived'),
        'count':         (len, 'survived'),
    },
)

for i in range(cross.rows):
    print(f'Class {cross["pclass"][i]} / {str(cross["sex"][i]):>6s}:'
          f'  survival={cross["survival_rate"][i]:.0%}'
          f'  n={cross["count"][i]:.0f}')
```

Output:

```
Class 1 /   male:  survival=0%  n=3
Class 1 / female:  survival=100%  n=3
Class 2 /   male:  survival=0%  n=4
Class 2 / female:  survival=100%  n=3
Class 3 /   male:  survival=0%  n=5
Class 3 / female:  survival=100%  n=2
```

## Transform: add group statistics to rows

`transform` computes group-level aggregates and broadcasts them back to every
row -- the output has the same row count as the input. This is useful for
computing relative metrics (e.g. "how does this passenger's fare compare to
the class average?"):

```python
enriched = titanic.transform(
    ['pclass'],
    {
        'class_mean_fare': (np.mean, 'fare'),
        'class_survival':  (np.mean, 'survived'),
    },
)

# enriched has 20 rows, same as titanic
print(f'Rows: {enriched.rows}')

# Compute fare ratio for first 5 passengers
fare_ratio = titanic['fare'][:5] / enriched['class_mean_fare'][:5]
print(f'Fare ratios (first 5): {np.round(fare_ratio, 2)}')
```

Output:

```
Rows: 20
Fare ratios (first 5): [1.4  1.4  1.05 1.02 0.6 ]
```

## Partition by class for parallel analysis

`partition` splits the data into sub-Tafras by group, preserving all rows.
This is designed for dispatching to `multiprocessing.Pool.map()`:

```python
parts = titanic.partition(['pclass'])

for key, sub in parts:
    surv = np.mean(sub['survived'])
    print(f'Class {key[0]}: {sub.rows} rows, survival={surv:.2%}')
```

Output:

```
Class 1: 6 rows, survival=50.00%
Class 2: 7 rows, survival=42.86%
Class 3: 7 rows, survival=28.57%
```

In a real pipeline you would pass these partitions to worker processes:

```python
from multiprocessing import Pool

def analyze_class(args):
    key, sub = args
    return Tafra({
        'pclass':        np.array([key[0]]),
        'survival_rate': np.array([np.mean(sub['survived'])]),
        'mean_fare':     np.array([np.mean(sub['fare'])]),
    })

# Example (serial execution here for demonstration):
results = [analyze_class(part) for part in parts]
combined = Tafra.concat(results)

for i in range(combined.rows):
    print(f'Class {combined["pclass"][i]}: '
          f'survival={combined["survival_rate"][i]:.2%}, '
          f'fare={combined["mean_fare"][i]:.1f}')
```

Output:

```
Class 1: survival=50.00%, fare=50.8
Class 2: survival=42.86%, fare=16.7
Class 3: survival=28.57%, fare=7.9
```

## iterate_by for custom per-group logic

When you need full control over what happens per group, use `iterate_by`:

```python
for keys, indices, sub in titanic.iterate_by(['pclass']):
    survivors = sub['age'][sub['survived'] == 1]
    non_surv  = sub['age'][sub['survived'] == 0]
    print(f'Class {keys[0]}:')
    print(f'  Survivor mean age:     {np.mean(survivors):.1f} (n={len(survivors)})')
    print(f'  Non-survivor mean age: {np.mean(non_surv):.1f} (n={len(non_surv)})')
```

Output:

```
Class 1:
  Survivor mean age:     39.7 (n=3)
  Non-survivor mean age: 40.7 (n=3)
Class 2:
  Survivor mean age:     24.0 (n=3)
  Non-survivor mean age: 37.3 (n=4)
Class 3:
  Survivor mean age:     19.0 (n=2)
  Non-survivor mean age: 24.9 (n=5)
```

## Summary

This tutorial covered:

- Building a dataset with missing values (`nan`)
- Using `transform` to impute missing values with per-group medians
- `group_by` for survival rate computation
- Cross-tabulation by grouping on multiple columns
- `transform` to broadcast group statistics back to every row
- `partition` for parallel dispatch
- `iterate_by` for custom per-group logic
