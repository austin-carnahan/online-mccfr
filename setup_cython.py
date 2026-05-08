"""Build script for Cython fast_ops module.

Usage:
    python setup_cython.py build_ext --inplace

This compiles src/fast_ops.pyx → src/fast_ops.so (or .cpython-*.so)
which can then be imported as `from src.fast_ops import ...`
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "src.fast_ops",
        sources=["src/fast_ops.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    ),
)
