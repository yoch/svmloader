from setuptools import setup, Extension
from Cython.Build import build_ext # , cythonize


ext = [
    Extension('svmloader',
        sources=['svmloader.pyx'],
        extra_compile_args=['-std=c++11'],
        language='c++'
    )
]


setup(
  name = 'svmloader',
  version = '0.1',
  description = 'a simplist but very fast parser for sparse matrix at libsvm format',
  author = 'J. Melka',
  url = 'https://github.com/yoch/svmloader',
  ext_modules = ext, # cythonize(som_ext) don't work
  license = 'GPL3',
  cmdclass = {'build_ext': build_ext},
  classifiers = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Operating System :: OS Independent',
    'Programming Language :: C++'
  ],
  install_requires=['numpy', 'scipy']
)
