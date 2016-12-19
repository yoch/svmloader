from setuptools import setup, Extension
from Cython.Build import build_ext # , cythonize


setup(
  name = 'svmloader',
  version = '0.4',
  description = 'a simplist but very fast parser for sparse matrix at libsvm format',
  author = 'J. Melka',
  url = 'https://github.com/yoch/svmloader',
  ext_modules = [Extension('svmloader', sources=['svmloader.pyx'])],
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
  setup_requires=['Cython'],
  install_requires=['numpy', 'scipy']
)
