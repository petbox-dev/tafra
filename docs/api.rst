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

.. currentmodule:: tafra.groups

.. autosummary::

    GroupBy
    Transform
    IterateBy
    InnerJoin
    LeftJoin


Methods
-------

.. currentmodule:: tafra.base.Tafra

.. autosummary::

    from_dataframe
    from_series
    as_tafra
    to_records
    to_list
    columns
    rows
    data
    dtypes
    head
    keys
    values
    items
    get
    select
    update
    update_dtypes
    rename
    coalesce
    copy
    delete
    union
    pprint
    pformat
    to_html


Helper Methods
--------------

.. currentmodule:: tafra.base.Tafra

.. autosummary::

    group_by
    transform
    iterate_by
    inner_join
    left_join
    cross_join


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
    .. automethod:: as_tafra
    .. automethod:: to_records
    .. automethod:: to_list
    .. autoattribute:: columns
    .. autoattribute:: rows
    .. autoattribute:: data
    .. autoattribute:: dtypes
    .. automethod:: head
    .. automethod:: keys
    .. automethod:: values
    .. automethod:: items
    .. automethod:: get
    .. automethod:: select
    .. automethod:: update
    .. automethod:: update_dtypes
    .. automethod:: rename
    .. automethod:: coalesce
    .. automethod:: copy
    .. automethod:: delete
    .. automethod:: union
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

.. currentmodule:: tafra.groups

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
