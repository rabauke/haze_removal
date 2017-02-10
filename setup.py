from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension('dehaze', ['dehaze.pyx'],
              extra_compile_args=['-O3'],
              include_dirs = [],
              libraries = [],
              library_dirs = []),
]

setup(
    name = 'dehaze',
    ext_modules = cythonize(extensions),
)
