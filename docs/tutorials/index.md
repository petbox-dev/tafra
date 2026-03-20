# Tutorials

These tutorials walk through complete analyses using Tafra, from loading
data to producing results. Each uses inline numpy arrays so there are no
external data dependencies.

| Tutorial | Description |
|----------|-------------|
| [Iris Dataset](iris.md) | Group-by aggregation on the classic Iris flower measurements. Covers construction, column access, and `group_by` with multiple statistics. |
| [Titanic Analysis](titanic.md) | Survival analysis with missing-value handling via `ObjectFormatter`, `transform` to broadcast group stats, `partition` for parallel dispatch, and cross-tabulation. |
| [Time Series](timeseries.md) | Datetime grouping patterns -- aggregate by year/month, partition for parallel forecasting, chunk for batch processing, and reassemble with `Tafra.concat()`. |

## Prerequisites

All tutorials assume tafra and numpy are installed:

```bash
pip install tafra
```

No other packages are required.
