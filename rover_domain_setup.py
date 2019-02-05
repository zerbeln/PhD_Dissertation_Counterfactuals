
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Compiler import Options

Options.annotate = True

setup(
    name = 'Test app',
    ext_modules=[
    Extension('rover_domain',
        sources=['rover_domain.pyx'],
                extra_compile_args=['-std=c++11'],
                language='c++')
    ],
    cmdclass = {'build_ext': build_ext}
)