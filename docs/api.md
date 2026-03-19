# API Reference

All methods grouped by category. Click any method to jump to its full documentation below.

| Category | Methods |
|---|---|
| **Aggregations** | [Union](#tafra.group.Union), [GroupBy](#tafra.group.GroupBy), [Transform](#tafra.group.Transform), [IterateBy](#tafra.group.IterateBy), [InnerJoin](#tafra.group.InnerJoin), [LeftJoin](#tafra.group.LeftJoin), [CrossJoin](#tafra.group.CrossJoin) |
| **Aggregation Helpers** | [union](#tafra.base.Tafra.union), [union_inplace](#tafra.base.Tafra.union_inplace), [group_by](#tafra.base.Tafra.group_by), [transform](#tafra.base.Tafra.transform), [iterate_by](#tafra.base.Tafra.iterate_by), [inner_join](#tafra.base.Tafra.inner_join), [left_join](#tafra.base.Tafra.left_join), [cross_join](#tafra.base.Tafra.cross_join) |
| **Chunking / Partitioning** | [chunks](#tafra.base.Tafra.chunks), [chunk_rows](#tafra.base.Tafra.chunk_rows), [partition](#tafra.base.Tafra.partition), [concat](#tafra.base.Tafra.concat) |
| **Custom Aggregations** | [percentile](#tafra.group.percentile), [geomean](#tafra.group.geomean), [harmean](#tafra.group.harmean) |
| **Constructors** | [as_tafra](#tafra.base.Tafra.as_tafra), [from_dataframe](#tafra.base.Tafra.from_dataframe), [from_series](#tafra.base.Tafra.from_series), [from_records](#tafra.base.Tafra.from_records) |
| **SQL Readers** | [read_sql](#tafra.base.Tafra.read_sql), [read_sql_chunks](#tafra.base.Tafra.read_sql_chunks) |
| **Destructors** | [to_records](#tafra.base.Tafra.to_records), [to_list](#tafra.base.Tafra.to_list), [to_tuple](#tafra.base.Tafra.to_tuple), [to_array](#tafra.base.Tafra.to_array), [to_pandas](#tafra.base.Tafra.to_pandas) |
| **Properties** | [rows](#tafra.base.Tafra.rows), [columns](#tafra.base.Tafra.columns), [data](#tafra.base.Tafra.data), [dtypes](#tafra.base.Tafra.dtypes), [size](#tafra.base.Tafra.size), [ndim](#tafra.base.Tafra.ndim), [shape](#tafra.base.Tafra.shape) |
| **Iter Methods** | [iterrows](#tafra.base.Tafra.iterrows), [itertuples](#tafra.base.Tafra.itertuples), [itercols](#tafra.base.Tafra.itercols) |
| **Functional Methods** | [row_map](#tafra.base.Tafra.row_map), [tuple_map](#tafra.base.Tafra.tuple_map), [col_map](#tafra.base.Tafra.col_map), [pipe](#tafra.base.Tafra.pipe) |
| **Dict-like Methods** | [keys](#tafra.base.Tafra.keys), [values](#tafra.base.Tafra.values), [items](#tafra.base.Tafra.items), [get](#tafra.base.Tafra.get), [update](#tafra.base.Tafra.update), [update_inplace](#tafra.base.Tafra.update_inplace), [update_dtypes](#tafra.base.Tafra.update_dtypes), [update_dtypes_inplace](#tafra.base.Tafra.update_dtypes_inplace) |
| **Data Exploration** | [head](#tafra.base.Tafra.head), [tail](#tafra.base.Tafra.tail), [sort](#tafra.base.Tafra.sort), [sample](#tafra.base.Tafra.sample), [describe](#tafra.base.Tafra.describe), [value_counts](#tafra.base.Tafra.value_counts), [drop_duplicates](#tafra.base.Tafra.drop_duplicates) |
| **Time Series** | [shift](#tafra.base.Tafra.shift) |
| **Other Helpers** | [select](#tafra.base.Tafra.select), [copy](#tafra.base.Tafra.copy), [rename](#tafra.base.Tafra.rename), [rename_inplace](#tafra.base.Tafra.rename_inplace), [coalesce](#tafra.base.Tafra.coalesce), [coalesce_inplace](#tafra.base.Tafra.coalesce_inplace), [delete](#tafra.base.Tafra.delete), [delete_inplace](#tafra.base.Tafra.delete_inplace) |
| **Printer Methods** | [pprint](#tafra.base.Tafra.pprint), [pformat](#tafra.base.Tafra.pformat), [to_html](#tafra.base.Tafra.to_html) |

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

## Custom Aggregations

::: tafra.group.percentile

::: tafra.group.geomean

::: tafra.group.harmean

## ObjectFormatter

::: tafra.formatter.ObjectFormatter
