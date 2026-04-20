"""
Tafra: a minimalist dataframe

Copyright (c) 2020 Derrick W. Turk and David S. Fulford

Author
------
Derrick W. Turk
David S. Fulford

Notes
-----
Created on April 25, 2020
"""

from importlib.metadata import version as _version

__version__ = _version("tafra")

from .base import Tafra, object_formatter
from .group import Union, GroupBy, Transform, IterateBy, InnerJoin, LeftJoin, CrossJoin
from .group import percentile, geomean, harmean

read_sql = Tafra.read_sql
read_sql_chunks = Tafra.read_sql_chunks
read_csv = Tafra.read_csv
as_tafra = Tafra.as_tafra

__all__ = [
    "Tafra",
    "object_formatter",
    "Union",
    "GroupBy",
    "Transform",
    "IterateBy",
    "InnerJoin",
    "LeftJoin",
    "CrossJoin",
    "percentile",
    "geomean",
    "harmean",
    "read_sql",
    "read_sql_chunks",
    "read_csv",
    "as_tafra",
]
