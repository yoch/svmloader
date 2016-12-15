svmloader
=========

**svmloader** is a simplist but very fast python module (written in
cython) to load sparse data written at libsvm format.

It is not functionnaly equivalent to
"sklearn.datasets.load_svmlight_file", and handle only the simplest
cases. *labels* are supposed to be of integer type, and data is parsed
as *numpy.float64* type.

Compressed data in .gz or .bz2 format is supported as well.

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


Benchmarks
----------

Benchmarks on data taken from [libsvm datasets](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).


| dataset *(shape, nonzeros)*                       | sklearn  | svmloader |
| :---                                              |     ---: |    ---: |
| **mnist.scale** (60000x780, 8994156)              | 78.1s    |  1.5s   |
| **rcv1_test.multiclass** (518571x47236, 33486015) | 1004.3s  |  7.9s   |
