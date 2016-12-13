cimport cython
import numpy as np
from libc.stdlib cimport strtoul, strtod
from libc.stdint cimport uint32_t
from cpython cimport array
import scipy.sparse as sp


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.initializedcheck(False)
cdef _load_svmfile(fp, bint zero_based):
    cdef char * s
    cdef char * end
    cdef uint32_t idx, sz
    cdef float value
    cdef bytes label, rest

    cdef array.array[float] data = array('f')
    cdef array.array[uint32_t] indices = array('L')
    cdef array.array[uint32_t] indptr = array('L', [0])
    cdef array.array[int] labels = array('l')

    sz = 0
    for line in fp:
        if line[0] == b'#':
            continue

        # get the label
        label, rest = line.split(maxsplit=1)
        labels.resize_smart(len(labels)+1)
        labels[len(labels)-1] = int(label)
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

            indices.resize_smart(sz+1)
            data.resize_smart(sz+1)
            indices[sz] = idx
            data[sz] = idx
            sz += 1

        indptr.resize_smart(len(indptr)+1)
        indptr[len(indptr)-1] = len(data)

    X = sp.csr_matrix((data, indices, indptr))
    y = np.asarray(labels)

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
