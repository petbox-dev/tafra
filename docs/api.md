# API Reference

All methods grouped by category. Click any method to jump to its full documentation below.

| Category | Methods |
|---|---|
| **Aggregations** | [Union](api.md#tafra.group.Union), [GroupBy](api.md#tafra.group.GroupBy), [Transform](api.md#tafra.group.Transform), [IterateBy](api.md#tafra.group.IterateBy), [InnerJoin](api.md#tafra.group.InnerJoin), [LeftJoin](api.md#tafra.group.LeftJoin), [CrossJoin](api.md#tafra.group.CrossJoin) |
| **Aggregation Helpers** | [union](api.md#tafra.base.Tafra.union), [union_inplace](api.md#tafra.base.Tafra.union_inplace), [group_by](api.md#tafra.base.Tafra.group_by), [transform](api.md#tafra.base.Tafra.transform), [iterate_by](api.md#tafra.base.Tafra.iterate_by), [inner_join](api.md#tafra.base.Tafra.inner_join), [left_join](api.md#tafra.base.Tafra.left_join), [cross_join](api.md#tafra.base.Tafra.cross_join) |
| **Chunking / Partitioning** | [chunks](api.md#tafra.base.Tafra.chunks), [chunk_rows](api.md#tafra.base.Tafra.chunk_rows), [partition](api.md#tafra.base.Tafra.partition), [concat](api.md#tafra.base.Tafra.concat) |
| **Custom Aggregations** | [percentile](api.md#tafra.group.percentile), [geomean](api.md#tafra.group.geomean), [harmean](api.md#tafra.group.harmean) |
| **Constructors** | [as_tafra](api.md#tafra.base.Tafra.as_tafra), [from_dataframe](api.md#tafra.base.Tafra.from_dataframe), [from_series](api.md#tafra.base.Tafra.from_series), [from_records](api.md#tafra.base.Tafra.from_records) |
| **SQL Readers** | [read_sql](api.md#tafra.base.Tafra.read_sql), [read_sql_chunks](api.md#tafra.base.Tafra.read_sql_chunks) |
| **Destructors** | [to_records](api.md#tafra.base.Tafra.to_records), [to_list](api.md#tafra.base.Tafra.to_list), [to_tuple](api.md#tafra.base.Tafra.to_tuple), [to_array](api.md#tafra.base.Tafra.to_array), [to_pandas](api.md#tafra.base.Tafra.to_pandas) |
| **Properties** | [rows](api.md#tafra.base.Tafra.rows), [columns](api.md#tafra.base.Tafra.columns), [data](api.md#tafra.base.Tafra.data), [dtypes](api.md#tafra.base.Tafra.dtypes), [size](api.md#tafra.base.Tafra.size), [ndim](api.md#tafra.base.Tafra.ndim), [shape](api.md#tafra.base.Tafra.shape) |
| **Iter Methods** | [iterrows](api.md#tafra.base.Tafra.iterrows), [itertuples](api.md#tafra.base.Tafra.itertuples), [itercols](api.md#tafra.base.Tafra.itercols) |
| **Functional Methods** | [row_map](api.md#tafra.base.Tafra.row_map), [tuple_map](api.md#tafra.base.Tafra.tuple_map), [col_map](api.md#tafra.base.Tafra.col_map), [pipe](api.md#tafra.base.Tafra.pipe) |
| **Dict-like Methods** | [keys](api.md#tafra.base.Tafra.keys), [values](api.md#tafra.base.Tafra.values), [items](api.md#tafra.base.Tafra.items), [get](api.md#tafra.base.Tafra.get), [update](api.md#tafra.base.Tafra.update), [update_inplace](api.md#tafra.base.Tafra.update_inplace), [update_dtypes](api.md#tafra.base.Tafra.update_dtypes), [update_dtypes_inplace](api.md#tafra.base.Tafra.update_dtypes_inplace) |
| **Data Exploration** | [head](api.md#tafra.base.Tafra.head), [tail](api.md#tafra.base.Tafra.tail), [sort](api.md#tafra.base.Tafra.sort), [sample](api.md#tafra.base.Tafra.sample), [describe](api.md#tafra.base.Tafra.describe), [value_counts](api.md#tafra.base.Tafra.value_counts), [drop_duplicates](api.md#tafra.base.Tafra.drop_duplicates) |
| **Time Series** | [shift](api.md#tafra.base.Tafra.shift) |
| **Other Helpers** | [select](api.md#tafra.base.Tafra.select), [copy](api.md#tafra.base.Tafra.copy), [rename](api.md#tafra.base.Tafra.rename), [rename_inplace](api.md#tafra.base.Tafra.rename_inplace), [coalesce](api.md#tafra.base.Tafra.coalesce), [coalesce_inplace](api.md#tafra.base.Tafra.coalesce_inplace), [delete](api.md#tafra.base.Tafra.delete), [delete_inplace](api.md#tafra.base.Tafra.delete_inplace) |
| **Printer Methods** | [pprint](api.md#tafra.base.Tafra.pprint), [pformat](api.md#tafra.base.Tafra.pformat), [to_html](api.md#tafra.base.Tafra.to_html) |

## Tafra

::: tafra.base.Tafra
    options:
      show_source: true
      members_order: source

## Aggregations

::: tafra.group.Union

::: tafra.group.GroupBy

::: tafra.group.Transform

::: tafra.group.IterateBy

::: tafra.group.InnerJoin

::: tafra.group.LeftJoin

::: tafra.group.CrossJoin

## ObjectFormatter

::: tafra.formatter.ObjectFormatter
