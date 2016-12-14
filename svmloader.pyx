cimport cython
from libc.stdlib cimport strtoul, strtod
from libc.stdint cimport uint32_t
import array
from cpython cimport array
import numpy as np
import scipy.sparse as sp


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
#@cython.initializedcheck(False)
cdef _load_svmfile(fp, bint zero_based):
    cdef char * s
    cdef char * end
    cdef uint32_t idx
    cdef double value
    cdef bytes label
    cdef bytes rest
    cdef Py_ssize_t sz
    cdef Py_ssize_t nrows

    cdef array.array data = array.array('d')
    cdef array.array indices = array.array('L')
    cdef array.array indptr = array.array('L', [0])
    cdef array.array labels = array.array('l')

    sz = 0
    nrows = 0
    for line in fp:
        if line[0] == b'#':
            continue

        # get the label
        label, rest = line.split(maxsplit=1)
        nrows += 1

        array.resize_smart(labels, nrows)
        labels.data.as_ints[nrows-1] = int(label)
        s = rest

        while s[0] != '\n' and s[0] != 0:
            # get index
            idx = strtoul(s, &end, 10)
            s = end
            # ensure we have correct separator
            while s[0]==' ':
                s += 1
            assert s[0] == ':'
            s += 1
            # get value
            value = strtod(s, &end)
            s = end
            while s[0]==' ':
                s += 1

            if not zero_based:
                idx -= 1

            array.resize_smart(indices, sz+1)
            array.resize_smart(data, sz+1)
            indices.data.as_uints[sz] = idx
            data.data.as_doubles[sz] = value
            sz += 1

        array.resize_smart(indptr, nrows+1)
        indptr.data.as_uints[nrows] = len(data)

    y = np.asarray(labels)

    return (data, indices, indptr, y)



import os.path

def _openfile(filename):
    _, ext = os.path.splitext(filename)
    if ext == ".gz":
        import gzip
        fp = gzip.open(filename, "rb")
    elif ext == ".bz2":
        from bz2 import BZ2File
        fp = BZ2File(filename, "rb")
    else:
        fp = open(filename, "rb")
    return fp


def load_svmfile(filename, nfeatures=None, zero_based=True):
    fp = _openfile(filename)
    data, indices, indptr, y = _load_svmfile(fp, zero_based)
    fp.close()

    if nfeatures is None:
        X = sp.csr_matrix((data, indices, indptr))
    else:
        X = sp.csr_matrix((data, indices, indptr), (len(indptr)-1, nfeatures))

    return X, y

def load_svmfiles(filenames, zero_based=True):
    Xlst = []
    ylst = []
    for filename in filenames:
        fp = _openfile(filename)
        data, indices, indptr, y = _load_svmfile(fp, zero_based)
        Xlst.append((data, indices, indptr))
        ylst.append(y)
        fp.close()
    # get nfeatures as the maximum indice
    nfeatures = max(max(indices) for _, indices, _ in Xlst) + 1
    lst = []
    for (data, indices, indptr), y in zip(Xlst, ylst):
        X = sp.csr_matrix((data, indices, indptr), (len(indptr)-1, nfeatures))
        lst.append(X)
        lst.append(y)
    return lst
