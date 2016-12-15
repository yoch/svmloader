.. svmloader documentation master file, created by
   sphinx-quickstart on Wed Dec 14 22:52:54 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to svmloader's documentation!
=====================================


**svmloader** is a simplist but very fast python module (written in cython) to load 
sparse data written at libsvm format.

It is not functionnaly equivalent to :class:`sklearn.datasets.load_svmlight_file`, 
and handle only the simplest cases.

The `labels` type supported are `int` and `float` (default `int`), and data can be parsed as `numpy.float64` or `numpy.float32` type (`float64` by default).  
Multiple labels currently are not supported.


.. toctree::
   :maxdepth: 2

.. automodule:: svmloader

	.. autofunction:: load_svmfile(filename, dtype='d', ltype='l', nfeatures=None, zero_based=True)
	.. autofunction:: load_svmfiles(filenames, dtype='d', ltype='l', zero_based=True)

