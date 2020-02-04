from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('Cython_Code/ccea.pyx', compiler_directives={'language_level' : "3"}))
setup(ext_modules=cythonize('Cython_Code/neural_network.pyx', compiler_directives={'language_level' : "3"}))
setup(ext_modules=cythonize('Cython_Code/standard_rewards.pyx', compiler_directives={'language_level' : "3"}))
setup(ext_modules=cythonize('Cython_Code/suggestion_rewards.pyx', compiler_directives={'language_level' : "3"}))
setup(ext_modules=cythonize('Cython_Code/supervisor.pyx', compiler_directives={'language_level' : "3"}))
setup(ext_modules=cythonize('AADI_RoverDomain/rover_domain.pyx', compiler_directives={'language_level' : "3"}))
setup(ext_modules=cythonize('AADI_RoverDomain/rover.pyx', compiler_directives={'language_level' : "3"}))
