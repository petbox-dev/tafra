"""Build configuration for optional C extension.

The C extension is optional — if compilation fails (missing compiler,
missing headers, etc.), the package installs without it and falls back
to pure Python + numpy at runtime.
"""
import os
import sys
import numpy as np
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """build_ext that treats C extension failure as non-fatal."""

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"\n*** WARNING: Failed to build C extension: {e}")
            print("*** tafra will use pure Python + numpy (slower but functional).\n")


ext_modules = [
    Extension(
        "tafra._accel",
        sources=["tafra/_accel.c"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalBuildExt},
)
