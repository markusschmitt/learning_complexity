from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

samplers_extension = Extension(
    name="samplers",
    sources=["samplers.pyx"],
    libraries=["samplers"],
    library_dirs=["cpp"],
    include_dirs=["cpp",numpy.get_include()],
    extra_compile_args=["-std=c++11", "-O3", "-fPIC"],
    language="c++"
)
setup(
    name="samplers",
    ext_modules=cythonize([samplers_extension])
)

# following https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/
