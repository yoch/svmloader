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
cdef _load_svmfile(fp, dtype, ltype, bint zero_based):
    cdef char * s
    cdef char * end
    cdef uint32_t idx
    cdef double value
    cdef bytes label
    cdef bytes rest

    cdef char dt = 'f' if dtype == 'f' else 'd'  # defaut is double
    cdef char lt = 'd' if ltype == 'd' else 'i'  # default is int
    cdef array.array data = array.array(dtype)
    cdef array.array indices = array.array('I')
    cdef array.array indptr = array.array('I', [0])
    cdef array.array labels = array.array(ltype)

    cdef Py_ssize_t sz = 0
    cdef Py_ssize_t nrows = 0

    for line in fp:
        if line[0] == '#':
            continue

        # get the label
        label, rest = line.split(None, 1)
        nrows += 1

        array.resize_smart(labels, nrows)
        if lt == 'i':
            labels.data.as_ints[nrows-1] = int(label)
        else:
            labels.data.as_doubles[nrows-1] = float(label)
        s = rest

        while s[0] != '#' and s[0] != '\n' and s[0] != 0:
            # get index
            idx = strtoul(s, &end, 10)
            if s==end:
                raise ValueError('invalid index')
            s = end

            if not zero_based:
                if idx==0:
                    raise ValueError('invalid index 0 with one-based indexes')
                idx -= 1

            # ensure we have correct separator
            while s[0]==' ':
                s += 1
            if s[0] != ':':
                raise ValueError('invalid separator')
            s += 1
            # get value
            value = strtod(s, &end)
            if s==end:
                raise ValueError('invalid value')
            s = end
            while s[0]==' ':
                s += 1

            array.resize_smart(indices, sz+1)
            array.resize_smart(data, sz+1)
            indices.data.as_uints[sz] = idx
            if dt == 'd':
                data.data.as_doubles[sz] = value
            else:
                data.data.as_floats[sz] = value
            sz += 1

        array.resize_smart(indptr, nrows+1)
        indptr.data.as_uints[nrows] = sz

    return (np.frombuffer(data, dtype=dtype), 
            np.frombuffer(indices, dtype='I'),
            np.frombuffer(indptr, dtype='I'), 
            np.frombuffer(labels, dtype=ltype))



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


def load_svmfile(filename, dtype='d', ltype='i', nfeatures=None, zero_based=True):
    """\
    Load a sparse matrix from filename at svmlib format.

    Files in .gz or .bz2 format will be uncompressed on the fly.

    :param filename: the file name
    :type filename: str
    :param dtype: type of data, must be either 'd' (double) or 'f' (float)
    :type dtype: str
    :param ltype: type of labels, must be either 'i' (int) or 'd' (double)
    :type ltype: str
    :param nfeatures: the number of columns (infered from file if is None)
    :type nfeatures: int
    :param zero_based: indicates if columns indexes are zero-based or one-based
    :type zero_based: bool
    :returns: (labels, sparse_matrix) tuple
    :rtype: (:class:`numpy.ndarray`, :class:`scipy.sparse.csr_matrix`)
    """
    assert(dtype=='f' or dtype=='d'), 'dtype must be "d" or "f"'
    assert(ltype=='i' or ltype=='d'), 'ltype must be "i" or "d"'

    fp = _openfile(filename)
    data, indices, indptr, y = _load_svmfile(fp, dtype, ltype, zero_based)
    fp.close()

    if nfeatures is None:
        X = sp.csr_matrix((data, indices, indptr))
    else:
        X = sp.csr_matrix((data, indices, indptr), (len(indptr)-1, nfeatures))

    return X, y

def load_svmfiles(filenames, dtype='d', ltype='i', zero_based=True):
    """\
    Load a sparse matrix list from list of filenames at svmlib format.

    Files in .gz or .bz2 format will be uncompressed on the fly.

    The number of features will be infered from the maximum indice found
    on all files.

    :param filenames: the list of files names
    :type filenames: list
    :param dtype: type of data, must be either 'd' (double) or 'f' (float)
    :type dtype: str
    :param ltype: type of labels, must be either 'i' (int) or 'd' (double)
    :type ltype: str
    :param zero_based: indicates if columns indexes are zero-based or one-based
    :type zero_based: bool
    :returns: a list [labels_0, matrix_0, .., labels_n, matrix_n]
    """
    assert(dtype=='f' or dtype=='d'), 'dtype must be "d" or "f"'
    assert(ltype=='i' or ltype=='d'), 'ltype must be "i" or "d"'

    Xlst = []
    ylst = []
    for filename in filenames:
        fp = _openfile(filename)
        data, indices, indptr, y = _load_svmfile(fp, dtype, ltype, zero_based)
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
