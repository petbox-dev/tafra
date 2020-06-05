=============
API Reference
=============

Summary
=======

Tafra
-----

.. currentmodule:: tafra.base

.. autosummary::

    Tafra


Aggregations
------------

.. currentmodule:: tafra.group

.. autosummary::

    Union
    GroupBy
    Transform
    IterateBy
    InnerJoin
    LeftJoin
    CrossJoin


Methods
-------

.. currentmodule:: tafra.base.Tafra

.. autosummary::

    from_records
    from_dataframe
    from_series
    read_sql
    read_sql_chunks
    as_tafra
    to_records
    to_list
    to_tuple
    to_array
    to_pandas
    rows
    columns
    data
    dtypes
    size
    ndim
    shape
    head
    keys
    values
    items
    get
    iterrows
    itertuples
    itercols
    row_map
    col_map
    select
    copy
    update
    update_inplace
    update_dtypes
    update_dtypes_inplace
    rename
    rename_inplace
    coalesce
    coalesce_inplace
    _coalesce_dtypes
    delete
    delete_inplace
    pprint
    pformat
    to_html


Helper Methods
--------------

.. currentmodule:: tafra.base.Tafra

.. autosummary::

    union
    union_inplace
    group_by
    transform
    iterate_by
    inner_join
    left_join
    cross_join


Object Formatter
----------------

.. currentmodule:: tafra.formatter

.. autoclass:: ObjectFormatter

    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: __delitem__


Detailed Reference
==================

Tafra
-----

.. currentmodule:: tafra.base

Methods
~~~~~~~

.. autoclass:: Tafra

    .. automethod:: from_dataframe
    .. automethod:: from_series
    .. automethod:: from_records
    .. automethod:: read_sql
    .. automethod:: read_sql_chunks
    .. automethod:: as_tafra
    .. automethod:: to_records
    .. automethod:: to_list
    .. automethod:: to_tuple
    .. automethod:: to_array
    .. automethod:: to_pandas
    .. autoattribute:: rows
    .. autoattribute:: columns
    .. autoattribute:: data
    .. autoattribute:: dtypes
    .. autoattribute:: size
    .. autoattribute:: ndim
    .. autoattribute:: shape
    .. automethod:: head
    .. automethod:: keys
    .. automethod:: values
    .. automethod:: items
    .. automethod:: get
    .. automethod:: iterrows
    .. automethod:: itertuples
    .. automethod:: itercols
    .. automethod:: row_map
    .. automethod:: col_map
    .. automethod:: select
    .. automethod:: copy
    .. automethod:: update
    .. automethod:: update_inplace
    .. automethod:: update_dtypes
    .. automethod:: update_dtypes_inplace
    .. automethod:: rename
    .. automethod:: rename_inplace
    .. automethod:: coalesce
    .. automethod:: coalesce_inplace
    .. automethod:: _coalesce_dtypes
    .. automethod:: delete
    .. automethod:: delete_inplace
    .. automethod:: union
    .. automethod:: union_inplace
    .. automethod:: pprint
    .. automethod:: pformat
    .. automethod:: to_html

Helper Methods
~~~~~~~~~~~~~~

.. class:: tafra.base.Tafra
    :noindex:

    .. automethod:: group_by
    .. automethod:: transform
    .. automethod:: iterate_by
    .. automethod:: inner_join
    .. automethod:: left_join
    .. automethod:: cross_join

Aggregations
------------

.. currentmodule:: tafra.group

.. autoclass:: Union

    .. automethod:: apply
    .. automethod:: apply_inplace

.. autoclass:: GroupBy

    .. automethod:: apply

.. autoclass:: Transform

    .. automethod:: apply

.. autoclass:: IterateBy

    .. automethod:: apply

.. autoclass:: InnerJoin

    .. automethod:: apply

.. autoclass:: LeftJoin

    .. automethod:: apply

.. autoclass:: CrossJoin

    .. automethod:: apply
