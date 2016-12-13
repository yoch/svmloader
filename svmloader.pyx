cimport cython
import numpy as np
from libc.stdlib cimport strtoul, strtod
from libc.stdint cimport uint32_t
from libcpp cimport bool as bool_t
from libcpp.vector cimport vector
from cpython cimport array
from cython cimport view
import scipy.sparse as sp


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _load_svmfile(fp, bool_t zero_based):
    cdef vector[double] data
    cdef vector[uint32_t] indices
    cdef vector[uint32_t] indptr
    cdef vector[int] labels
    cdef char * s
    cdef char * end
    cdef uint32_t idx
    cdef float value
    cdef bytes label, rest

    indptr.push_back(0)

    for line in fp:
        if line[0] == b'#':
            continue

        # get the label
        label, rest = line.split(maxsplit=1)
        labels.push_back(int(label))
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
            val = strtod(s, &end)
            s = end
            while s[0]==' ':
                s += 1

            if not zero_based:
                idx -= 1
            indices.push_back(idx)
            data.push_back(val)

        indptr.push_back(data.size())

    cdef view.array data_view = <double[:data.size()]> data.data()
    cdef view.array indices_view = <uint32_t[:indices.size()]> indices.data()
    cdef view.array indptr_view = <uint32_t[:indptr.size()]> indptr.data()
    cdef view.array labels_view = <int[:labels.size()]> labels.data()

    X = sp.csr_matrix((data_view.copy(), indices_view.copy(), indptr_view.copy()))
    y = np.asarray(labels_view.copy())

    return (X, y)



import os.path

def load_svmfile(filename, zero_based=True):
    _, ext = os.path.splitext(filename)
    if ext == ".gz":
        import gzip
        fp = gzip.open(filename, "rb")
    elif ext == ".bz2":
        from bz2 import BZ2File
        fp = BZ2File(filename, "rb")
    else:
        fp = open(filename, "rb")

    return _load_svmfile(fp, zero_based)
