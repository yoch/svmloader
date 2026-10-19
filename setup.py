from setuptools import setup, Extension
import io
import os
import sys


DESCRIPTION = 'a very fast parser for sparse matrix at libsvm format'

# README.md ships in the sdist (see MANIFEST.in); fall back to the short
# description if it is missing.
try:
    with io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md'), encoding='utf-8') as f:
        LONG_DESCRIPTION = f.read()
except IOError:
    LONG_DESCRIPTION = DESCRIPTION


if '--no-cython' in sys.argv:
    USE_CYTHON = False
    sys.argv.remove('--no-cython')
else:
    USE_CYTHON = True

ext = '.pyx' if USE_CYTHON else '.c'

if USE_CYTHON:
    try:
        from Cython.Build import build_ext
    except ImportError:
        raise SystemExit(
            'building svmloader from svmloader.pyx requires Cython; '
            'install Cython, or build from a release sdist that ships the '
            'generated svmloader.c using --no-cython'
        )
    cmdclass = {'build_ext': build_ext}
else:
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'svmloader.c')):
        raise SystemExit(
            '--no-cython requires the generated svmloader.c, which is not '
            'tracked in git. Generate it with "cython svmloader.pyx", or use '
            'a release sdist (built with "setup.py --no-cython sdist").'
        )
    cmdclass = {}

import numpy as np


setup(
    name = 'svmloader',
    version = '0.5.1',
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    long_description_content_type = 'text/markdown',
    author = 'J. Melka',
    url = 'https://github.com/yoch/svmloader',
    project_urls = {
        'Documentation': 'https://svmloader.readthedocs.io',
        'Source': 'https://github.com/yoch/svmloader',
        'Issues': 'https://github.com/yoch/svmloader/issues',
    },
    ext_modules = [
        Extension('svmloader',
                    sources=['svmloader'+ext],
                    include_dirs=[np.get_include()])
    ],
    package_data = {'': ['*.pyx']},
    license = 'GPL3',
    python_requires = '>=3.9',
    cmdclass = cmdclass,
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    keywords='libsvm,sparse matrix,csr,svmlight,load_svmlight_file,sparse,scipy,cython',
    install_requires=['numpy', 'scipy']
)
