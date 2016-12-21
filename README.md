svmloader
=========

**svmloader** is a very fast python module (written in cython) 
intended to load sparse data written at libsvm format.

It is not fully equivalent to `sklearn.datasets.load_svmlight_file`, 
in particular `query_id` are not supported and `dtype` is restricted.

The types of data and labels are distinguished.
The labels types supported are `int` and `float` (default `int`), 
and data can be parsed as `numpy.float64` or `numpy.float32` type (`float64` by default).

Compressed data in .gz or .bz2 format is supported as well.

Install
-------

Simply use `pip install svmloader`.

Alternatively, you can clone the repository and run `python setup.py install`.


Dependencies :
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
