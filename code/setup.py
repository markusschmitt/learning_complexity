from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

wolff_extension = Extension(
    name="wolff_sampler",
    sources=["wolff_sampler.pyx"],
    libraries=["wolffsampler"],
    library_dirs=["cpp"],
    include_dirs=["cpp",numpy.get_include()],
    language="c++"
)
setup(
    name="wolff_sampler",
    ext_modules=cythonize([wolff_extension])
)

# following https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/
