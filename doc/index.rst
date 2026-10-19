.. svmloader documentation master file, created by
   sphinx-quickstart on Wed Dec 14 22:52:54 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :description: svmloader: fast loader for libsvm / svmlight sparse data files in Python (Cython), returning SciPy CSR matrices; an alternative to sklearn.datasets.load_svmlight_file.
   :keywords: libsvm, svmlight, load_svmlight_file, sparse matrix, csr, scipy, cython, python

svmloader documentation
=======================

.. toctree::
   :maxdepth: 2


**svmloader** is a fast loader for libsvm / svmlight sparse data files in Python (Cython),
an alternative to ``sklearn.datasets.load_svmlight_file``.

* Install: ``pip install svmloader`` (`PyPI <https://pypi.org/project/svmloader/>`_)
* Source code and issue tracker: `GitHub <https://github.com/yoch/svmloader>`_

It is not fully equivalent to `sklearn.datasets.load_svmlight_file`, 
in particular `query_id` are not supported and `dtype` is restricted.

The types of data and labels are distinguished.
The labels types supported are `int` and `float` (default `int`), 
and data can be parsed as `numpy.float64` or `numpy.float32` type (`float64` by default).

Compressed data in .gz or .bz2 format is supported as well.


API
---

.. automodule:: svmloader

    .. autofunction:: load_svmfile(filename, dtype='d', ltype='i', nfeatures=None, zero_based=True, multilabels=False)
    .. autofunction:: load_svmfiles(filenames, dtype='d', ltype='i', zero_based=True, multilabels=False)
    .. autofunction:: save_svmfile(filename, mat, labels=None, zero_based=False)
