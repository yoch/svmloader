from setuptools import setup, Extension
from Cython.Build import build_ext # , cythonize
import sys
import numpy as np


if '--no-cython' in sys.argv:
    USE_CYTHON = False
    sys.argv.remove('--no-cython')
else:
    USE_CYTHON = True

ext = '.pyx' if USE_CYTHON else '.c'


setup(
    name = 'svmloader',
    version = '0.5',
    description = 'a very fast parser for sparse matrix at libsvm format',
    author = 'J. Melka',
    url = 'https://github.com/yoch/svmloader',
    ext_modules = [
        Extension('svmloader',
                    sources=['svmloader'+ext],
                    include_dirs=[np.get_include()])
    ],
    package_data = {'': ['*.pyx']},
    license = 'GPL3',
    cmdclass = {'build_ext': build_ext},
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering'
    ],
    keywords='libsvm,sparse matrix,csr',
    install_requires=['numpy', 'scipy']
)
