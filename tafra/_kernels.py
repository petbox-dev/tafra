"""
Tafra: a minimalist dataframe

Copyright (c) 2020 Derrick W. Turk and David S. Fulford

Author
------
Derrick W. Turk
David S. Fulford

Notes
-----
The optional C extension (_accel.c): its kernels, and whether they loaded.
Every caller reads HAS_ACCEL through this module when it is called, so
setting tafra._kernels.HAS_ACCEL to False runs the pure numpy paths.
Use a kernel name only when HAS_ACCEL is True: a build missing one kernel
leaves the names before it bound, so __all__ lists only HAS_ACCEL.
"""

__all__ = ["HAS_ACCEL"]

try:
    # `name as name` re-exports each kernel for mypy (no_implicit_reexport)
    from ._accel import (  # type: ignore[import-not-found]
        groupby_sum as groupby_sum,
        groupby_sum_int64 as groupby_sum_int64,
        groupby_mean as groupby_mean,
        groupby_var as groupby_var,
        groupby_min as groupby_min,
        groupby_max as groupby_max,
        groupby_min_int64 as groupby_min_int64,
        groupby_max_int64 as groupby_max_int64,
        inner_join as inner_join,
        left_join as left_join,
        composite_key as composite_key,
        encode_int64 as encode_int64,
        group_indices as group_indices,
        encode_strings as encode_strings,
    )

    HAS_ACCEL = True
except ImportError:
    HAS_ACCEL = False
