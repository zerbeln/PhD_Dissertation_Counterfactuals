from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('Cython_Code/ccea.pyx', compiler_directives={'language_level' : "3"}))
setup(ext_modules=cythonize('Cython_Code/neural_network.pyx', compiler_directives={'language_level' : "3"}))
setup(ext_modules=cythonize('Cython_Code/homogeneous_rewards.pyx', compiler_directives={'language_level' : "3"}))
setup(ext_modules=cythonize('Cython_Code/supervisor.pyx', compiler_directives={'language_level' : "3"}))
setup(ext_modules=cythonize('AADI_RoverDomain/rover_domain_cython.pyx', compiler_directives={'language_level' : "3"}))
setup(ext_modules=cythonize('AADI_RoverDomain/rover.pyx', compiler_directives={'language_level' : "3"}))
