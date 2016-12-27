cimport cython
from libc.stdlib cimport strtol, strtod
from libc.limits cimport INT_MAX
import array
from cpython cimport array
import numpy as np
import scipy.sparse as sp


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
#@cython.initializedcheck(False)
cdef _load_svmfile(fp, dtype, ltype, bint zero_based, bint multilabels):
    cdef char * s
    cdef char * end
    cdef Py_ssize_t idx
    cdef Py_ssize_t last_idx
    cdef double value

    cdef char dt = 'f' if dtype == 'f' else 'd'  # defaut is double
    cdef char lt = 'd' if ltype == 'd' else 'i'  # default is int
    cdef array.array data = array.array(dtype)
    cdef array.array indices = array.array('i')
    cdef array.array indptr = array.array('i', [0])

    if not multilabels:
        labels = array.array(ltype)
    else:
        labels = []
        cls = float if lt == 'd' else int

    cdef Py_ssize_t sz = 0
    cdef Py_ssize_t nrows = 0

    for line in fp:
        s = line

        if s[0] == '#':
            continue

        nrows += 1

        # get the label
        if not multilabels:
            array.resize_smart(labels, nrows)
            if lt == 'i':
                labels[nrows-1] = strtol(s, &end, 10)
            else:
                labels[nrows-1] = strtod(s, &end)
            if s==end:
                raise ValueError('invalid label')
            s = end
        else:
            labl, line = line.split(None, 1)
            s = line
            label = sorted([cls(val) for val in labl.split(b',')])
            labels.append(tuple(label))

        # process the line
        last_idx = -1
        while s[0] != '#' and s[0] != '\n' and s[0] != 0:
            # get index
            idx = strtol(s, &end, 10)
            if s == end:
                raise ValueError('invalid index')
            if idx < 0 or idx > INT_MAX:
                raise ValueError('invalid index (out of range)')
            s = end

            if not zero_based:
                if idx == 0:
                    raise ValueError('invalid index 0 with one-based indexes')
                idx -= 1

            if idx <= last_idx:
                raise ValueError('indices should be sorted and uniques')
            last_idx = idx

            # ensure we have correct separator
            while s[0] == ' ':
                s += 1
            if s[0] != ':':
                raise ValueError('invalid separator')
            s += 1

            # get value
            value = strtod(s, &end)
            if s == end:
                raise ValueError('invalid value')
            s = end
            while s[0] == ' ':
                s += 1

            sz += 1
            array.resize_smart(indices, sz)
            array.resize_smart(data, sz)
            indices.data.as_ints[sz-1] = idx
            if dt == 'd':
                data.data.as_doubles[sz-1] = value
            else:
                data.data.as_floats[sz-1] = value

        array.resize_smart(indptr, nrows+1)
        indptr.data.as_ints[nrows] = sz

    if not multilabels:
        labels = np.frombuffer(labels, dtype=ltype)

    return (np.frombuffer(data, dtype=dtype),
            np.frombuffer(indices, dtype='i'),
            np.frombuffer(indptr, dtype='i'),
            labels)



def _openfile(filename):
    import os.path
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


def load_svmfile(filename, dtype='d', ltype='i', nfeatures=None, zero_based=True, multilabels=False):
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
    :param multilabels: indicates if file uses multiple labels per row
    :type multilabels: bool
    :returns: (labels, sparse_matrix) tuple
    :rtype: (:class:`numpy.ndarray`, :class:`scipy.sparse.csr_matrix`)
    """
    assert(dtype=='f' or dtype=='d'), 'dtype must be "d" or "f"'
    assert(ltype=='i' or ltype=='d'), 'ltype must be "i" or "d"'

    fp = _openfile(filename)
    data, indices, indptr, y = _load_svmfile(fp, dtype, ltype, zero_based, multilabels)
    fp.close()

    if nfeatures is None:
        X = sp.csr_matrix((data, indices, indptr), copy=False)
    else:
        X = sp.csr_matrix((data, indices, indptr), shape=(len(indptr)-1, nfeatures), copy=False)

    return X, y

def load_svmfiles(filenames, dtype='d', ltype='i', zero_based=True, multilabels=False):
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
    :param multilabels: indicates if file uses multiple labels per row
    :type multilabels: bool
    :returns: a list [labels_0, matrix_0, .., labels_n, matrix_n]
    """
    assert(dtype=='f' or dtype=='d'), 'dtype must be "d" or "f"'
    assert(ltype=='i' or ltype=='d'), 'ltype must be "i" or "d"'

    Xlst = []
    ylst = []
    for filename in filenames:
        fp = _openfile(filename)
        data, indices, indptr, y = _load_svmfile(fp, dtype, ltype, zero_based, multilabels)
        Xlst.append((data, indices, indptr))
        ylst.append(y)
        fp.close()
    # get nfeatures as the maximum indice
    nfeatures = max(max(indices) for _, indices, _ in Xlst) + 1
    lst = []
    for (data, indices, indptr), y in zip(Xlst, ylst):
        X = sp.csr_matrix((data, indices, indptr), shape=(len(indptr)-1, nfeatures), copy=False)
        lst.append(X)
        lst.append(y)
    return lst
