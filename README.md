svmloader
=========

**svmloader** is a simplist but very fast python module (written in
cython) to load sparse data written at libsvm format.

It is not functionnaly equivalent to
"sklearn.datasets.load_svmlight_file", and handle only the simplest
cases. *labels* are supposed to be of integer type, and data is parsed
as *numpy.float64* type.


Install
-------

Simply use `pip install svmloader`.

Alternatively, you can clone the repository and run `python setup.py install`.


Dependencies :
- Cython
- numpy
- scipy


Documentation
-------------

See [here](http://svmloader.readthedocs.io/en/latest/).
