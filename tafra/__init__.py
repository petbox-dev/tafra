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
__version__ = '1.0.8'

from .base import Tafra, object_formatter
from .group import GroupBy, Transform, IterateBy, InnerJoin, LeftJoin

read_sql = Tafra.read_sql
read_sql_chunks = Tafra.read_sql_chunks
read_csv = Tafra.read_csv
as_tafra = Tafra.as_tafra
