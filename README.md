svmloader
=========

[![PyPI](https://img.shields.io/pypi/v/svmloader.svg)](https://pypi.org/project/svmloader/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/yoch/svmloader/blob/master/LICENSE)

Fast loader for libsvm / svmlight sparse data files in Python (Cython),
an alternative to `sklearn.datasets.load_svmlight_file`.
**svmloader** loads them as `scipy.sparse.csr_matrix` matrices and `numpy` label arrays.

It is not fully equivalent to `sklearn.datasets.load_svmlight_file`, 
in particular `query_id` are not supported and `dtype` is restricted.

The types of data and labels are distinguished.
The labels types supported are `int` and `float` (default `int`), 
and data can be parsed as `numpy.float64` or `numpy.float32` type (`float64` by default).

Compressed data in .gz or .bz2 format is supported as well
(see [Benchmarks](#benchmarks) for the cost of bz2 decompression).

Install
-------

Simply use `pip install svmloader` (Python 3.9+).

Alternatively, you can clone the repository and run `pip install .`.


Dependencies :
- numpy
- scipy


Usage
-----

```python
from svmloader import load_svmfile, load_svmfiles

# X is a scipy.sparse.csr_matrix, y a numpy array of labels.
# Column indices are zero-based by default: pass zero_based=False
# for one-based files (as in the libsvm datasets).
X, y = load_svmfile('train.libsvm', zero_based=False)

# float32 data, float labels, bz2 file decompressed on the fly
X, y = load_svmfile('train.libsvm.bz2', dtype='f', ltype='d', zero_based=False)

# several files at once, with the same number of columns
X_train, y_train, X_test, y_test = load_svmfiles(['train.libsvm', 'test.libsvm.gz'], zero_based=False)
```


Documentation
-------------

See [here](https://svmloader.readthedocs.io/en/latest/).


Benchmarks
----------

Benchmarks on data taken from [libsvm datasets](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/),
with Python 3.12, numpy 2.5, scipy 1.18 and scikit-learn 1.9.1 on an Intel
Core i7-3770. Times are the best of 3 runs (2 for rcv1) in a fresh process,
with the files in the page cache.

| dataset *(shape, nonzeros)*                       | sklearn 1.9 | svmloader |
| :---                                              |        ---: |      ---: |
| **mnist.scale** (60000x780, 8994156)              | 5.0s        | 0.88s     |
| **rcv1_test.multiclass** (518571x47236, 33486015) | 21.1s       | 7.6s      |

`svmloader` is 5-6x faster than scikit-learn on mnist and about 3x faster on
rcv1. For reference, libsvm's `svm_read_problem` takes 5.0s on mnist.

Compressed files are decompressed on the fly. bzip2 decompression is
CPU-bound and much slower than parsing: on the same machine, loading
`mnist.bz2` takes 3-8s (versus 0.9s uncompressed) and the 815 MB rcv1 file
takes 60-100s from `.bz2` (versus 7.6s uncompressed). The range reflects
CPU throttling under sustained decompression; `.gz` files decompress much
faster.
